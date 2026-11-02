"""
Book example, not repository code.

Chapter 40: the production gap, in numbers for the released 400M model.

    python book/code-examples/ch40_gap_arithmetic.py

Nothing in this chapter is implemented in the repository. This script computes
what each missing piece would be working with, from `model.py` unchanged:

1. the KV cache per token, checked against a real cache built by `tiny()`;
2. what `torch.cat` in the cache costs over one long generation;
3. weights at fp32, bf16, int8 and int4, and how many full contexts fit beside
   them on a 16 GB card;
4. how much memory a contiguous cache reserves against a paged one (illustrative
   request lengths, seeded);
5. speculative decoding's expected tokens per target-model call (illustrative);
6. the attention score matrix that a fused kernel never writes out;
7. training memory with AdamW, and with its state partitioned over 8 GPUs.

Each size is given twice where it differs: as the repository actually serves
the model (fp32 weights under bf16 autocast, so fp32 keys and bf16 values in
the cache, plus the bf16 copy of the experts the grouped path stacks), and for a
hypothetical deployment that stores weights and cache in bf16.

Results: ./part_viii_ix_gap.json, or book/measurements/ with --save.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anulm import model as M                                     # noqa: E402  repository code, unchanged
from trace_forward import released                    # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_viii_ix_gap.json"
GiB, MiB = 2 ** 30, 2 ** 20


def tiny() -> M.AnuLMConfig:
    return M.AnuLMConfig(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8,
                         n_kv_head=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
                         intermediate_size=64, n_group=1, topk_group=1)


def kv_per_token(cfg, bytes_per: int) -> int:
    return cfg.n_layer * 2 * cfg.n_kv_head * cfg.head_dim * bytes_per


def check_cache_formula() -> None:
    cfg = tiny()
    m = M.AnuLM(cfg).eval()
    cache = m.new_cache()
    with torch.no_grad():
        m(torch.randint(0, 64, (1, 20)), cache=cache, start=0)
    held = sum(t.numel() * t.element_size() for layer in cache for t in layer.values())
    print(f"1. tiny(): cache after 20 tokens holds {held:,} bytes = 20 x {held // 20} "
          f"(formula L*2*Hkv*Dh*4 = {kv_per_token(cfg, 4)})")
    assert held == 20 * kv_per_token(cfg, 4)


def main() -> None:
    r = {}
    check_cache_formula()
    cfg = released()
    kv = {k: kv_per_token(cfg, b) for k, b in (("bf16", 2), ("fp32", 4))}
    kv["autocast (fp32 keys, bf16 values)"] = (kv["bf16"] + kv["fp32"]) // 2
    ctx = cfg.block_size
    print(f"   released: {kv['bf16']:,} B/token in bf16, {kv['autocast (fp32 keys, bf16 values)']:,} under "
          f"autocast, {kv['fp32']:,} in fp32; a full {ctx:,}-token context = {ctx * kv['bf16'] / MiB:.0f} MiB (bf16), "
          f"{ctx * kv['autocast (fp32 keys, bf16 values)'] / MiB:.0f} MiB as served")
    r["kv_bytes_per_token"] = kv

    # 2. torch.cat re-copies the whole cache at every decoding step
    served = kv["autocast (fp32 keys, bf16 values)"]
    copied = sum(t * served for t in range(1, ctx + 1))
    copied_bf16 = sum(t * kv["bf16"] for t in range(1, ctx + 1))
    print(f"\n2. growing the cache with torch.cat over {ctx:,} steps copies {copied / 1e9:.2f} GB as served "
          f"({copied_bf16 / 1e9:.2f} GB in bf16), against {ctx * served / 1e6:.1f} MB "
          f"({ctx * served / MiB:.0f} MiB) if each step wrote only its own token")
    r["cat_copy_bytes_full_context"] = {"served": copied, "bf16": copied_bf16}

    # 3. weights at four precisions, and contexts that fit beside them
    with torch.device("meta"):
        m = M.AnuLM(cfg)
    total, active = m.num_params()
    mats = [p for n, p in m.named_parameters() if p.dim() == 2]
    rows = sum(p.shape[0] for p in mats)
    n_mat = sum(p.numel() for p in mats)
    other = total - n_mat
    card = 16 * GiB
    w = {"fp32": 4 * total, "bf16": 2 * total,
         "int8 (per-row fp16 scales)": n_mat + 2 * rows + 4 * other,
         "int4 (fp16 scale per 64 weights)": n_mat // 2 + 2 * (n_mat // 64) + 4 * other}
    print(f"\n3. {total:,} parameters; {n_mat:,} in {len(mats)} matrices with {rows:,} rows; {other:,} in vectors")
    fits = {}
    for k, b in w.items():
        fits[k] = int((card - b) // (ctx * kv["bf16"]))
        print(f"   {k:34s} {b / 1e9:6.3f} GB   full bf16 contexts beside it on 16 GiB: {fits[k]:,}")
    experts = sum(p.numel() for n, p in m.named_parameters() if ".experts." in n)
    stack = 2 * experts                                   # MoE._stacked_weights: a bf16 copy for grouped_mm
    as_served = int((card - w["fp32"]) // (ctx * served))
    with_stack = int((card - w["fp32"] - stack) // (ctx * served))
    print(f"   as served: fp32 weights, {served:,} B/token of cache: {as_served} full contexts on 16 GiB; "
          f"{with_stack} with the grouped path's bf16 expert stack ({stack / 1e9:.2f} GB)")
    r.update(params=total, active=active, weight_bytes=w, contexts_on_16GiB=fits,
             served={"contexts": as_served, "contexts_with_stack": with_stack, "expert_stack_bytes": stack})

    # 4. contiguous vs paged reservations (illustrative request lengths)
    rng = np.random.default_rng(0)
    lens = rng.integers(16, ctx + 1, size=1000)
    block = 16
    contiguous = len(lens) * ctx
    paged = int(sum(math.ceil(n / block) * block for n in lens))
    used = int(lens.sum())
    print(f"\n4. 1,000 requests, lengths uniform in [16, {ctx}] (seed 0, illustrative): tokens used {used:,}")
    print(f"   reserved: contiguous {contiguous:,} ({100 * (1 - used / contiguous):.1f}% idle); "
          f"pages of {block}: {paged:,} ({100 * (1 - used / paged):.1f}% idle)")
    r["paging_illustration"] = {"used": used, "contiguous": contiguous, "paged16": paged}

    # 5. speculative decoding: tokens per target call (illustrative acceptance rates)
    print("\n5. speculative decoding, gamma drafted tokens, each accepted with probability alpha (illustrative):")
    spec = {}
    for a in (0.6, 0.8, 0.9):
        e = (1 - a ** 5) / (1 - a)                       # gamma = 4: expected accepted + 1 corrected
        spec[str(a)] = round(e, 3)
        print(f"   alpha {a}: {e:.2f} tokens per target call (gamma 4)")
    r["speculative_tokens_per_call_gamma4"] = spec

    # 6. the score matrix of one layer
    scores = cfg.n_head * ctx * ctx * 4
    print(f"\n6. one layer's scores for one {ctx:,}-token sequence: H x S x S x 4 B = {scores / MiB:.0f} MiB in fp32; "
          f"x {cfg.n_layer} layers = {cfg.n_layer * scores / GiB:.2f} GiB if all were kept")
    r["score_matrix_bytes_per_layer"] = scores

    # 7. training state with AdamW, and partitioned over N GPUs
    n = 8
    plain = 16 * total                                    # fp32 weights, grads, two moments
    zero = {"optimizer states": 4 + 4 + 8 / n, "+ gradients": 4 + 12 / n, "+ parameters": 16 / n}
    print(f"\n7. AdamW in fp32: 4 (weights) + 4 (grads) + 8 (two moments) = 16 B/param = {plain / 1e9:.2f} GB "
          f"= {plain / GiB:.2f} GiB")
    for k, b in zero.items():
        print(f"   partitioning {k:18s} over {n} GPUs: {b:5.2f} B/param = {b * total / 1e9:.2f} GB per GPU")
    print(f"   one data-parallel step exchanges gradients the size of the model: {4 * total / 1e9:.2f} GB in fp32")
    r.update(adamw_fp32_bytes=plain, zero_bytes_per_param_8gpu=zero, grad_bytes_fp32=4 * total)
    ex1 = 64 * 512 * kv["bf16"]
    ex3 = (1 - 0.8 ** 3) / (1 - 0.8)
    print(f"\nexercises: 64 x 512 tokens of bf16 cache = {ex1:,} B = {ex1 / MiB:.0f} MiB; "
          f"alpha 0.8, gamma 2: {ex3:.2f} tokens per call")
    OUT.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
