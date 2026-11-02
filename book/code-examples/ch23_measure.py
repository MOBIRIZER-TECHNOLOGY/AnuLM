"""
Book example, not repository code.

Chapter 23: two measurements on the GPU, both through model.py and train.py
unchanged.

1. Router precision. `ckpt_base_v2.pt` runs over held-out windows of
   `data/v2/val.u16` under bf16 autocast, as in training. A forward hook on
   every Router takes the input the fp32 gate saw and calls the same
   `Router._gate` again, this time under bf16 autocast: the behaviour before the
   fix that `Router.forward` documents. It counts the tokens whose chosen expert
   set differs. Statistics only; no held-out text is decoded or saved.
2. Memory and time per training step for the released configuration (random
   weights: memory does not depend on their values) at the released batch,
   4 x 1,024, with fused AdamW: plain, gradient checkpointing, chunked loss, both.

    python book/code-examples/ch23_measure.py [--save]   # needs a CUDA GPU with about 12 GB free
    python book/code-examples/ch23_measure.py --report   # derived numbers from the saved JSON

With `--save` the results go to book/measurements/part_v_a_memory.json; without
it, to ./part_v_a_memory.json in the current directory. `--report` reads the
saved file, prints the derived numbers the chapter quotes, and checks whether
SDPA's flash kernel accepts an explicit mask.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged
from anulm.training import train as T  # noqa: E402  the repository's train.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
NAME = "part_v_a_memory.json"
SAVED = ROOT / "book" / "measurements" / NAME
OUT = SAVED if "--save" in sys.argv else Path(NAME)
DEV = "cuda"
B, S, WINDOWS = 4, 1024, 32
BF16 = dict(device_type="cuda", dtype=torch.bfloat16)


def router_flips():
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = M.AnuLM(cfg)
    m.load_state_dict(ck["model"])
    m = m.to(DEV).eval()
    del ck
    stats = {}

    def hook(router, args, kwargs, out):
        x_flat, seq_len = args[0], kwargs.get("seq_len")
        with torch.autocast(**BF16):                               # the gate as it ran before the fix
            idx16, _, _ = router._gate(x_flat.float(), seq_len)
        a, b = out[0].sort(-1).values, idx16.sort(-1).values
        d = stats.setdefault(router.layer_no, [0, 0])
        d[0] += int((a != b).any(-1).sum())                       # tokens whose expert set changed
        d[1] += len(a)

    for i, layer in enumerate(m.layers):
        if isinstance(layer.mlp, M.MoE):
            layer.mlp.router.layer_no = i
            layer.mlp.router.register_forward_hook(hook, with_kwargs=True)
    _, va, _ = T.load_memmap(str(ROOT / "data" / "v2" / "meta.json"))
    with torch.no_grad(), torch.autocast(**BF16):
        for ix in T.strided_windows(va, B, S, WINDOWS):
            x, y = T.make_batch(va, ix, S, DEV)
            m(x, y)
    flipped = sum(v[0] for v in stats.values())
    total = sum(v[1] for v in stats.values())
    per_layer = {k: v[0] / v[1] for k, v in sorted(stats.items())}
    print(f"router in bf16 vs fp32: {flipped:,} of {total:,} token-layer choices differ "
          f"({100 * flipped / total:.2f}%), over {WINDOWS} held-out windows of {S}")
    lo, hi = min(per_layer, key=per_layer.get), max(per_layer, key=per_layer.get)
    print(f"  per layer: lowest layer {lo} {100 * per_layer[lo]:.2f}%, highest layer {hi} {100 * per_layer[hi]:.2f}%")
    del m
    torch.cuda.empty_cache()
    return {"flipped": flipped, "total": total, "per_layer": per_layer, "windows": WINDOWS, "seq": S}


def step_cost(grad_ckpt: bool, ce_chunk: int, va):
    torch.manual_seed(0)
    cfg = M.AnuLMConfig.nano_350m(vocab_size=32768, block_size=S, sliding_window=256, max_window_layers=10,
                                  num_experts=24, num_experts_per_tok=4, num_shared_experts=0,
                                  moe_intermediate_size=192, moe_impl="grouped",
                                  seq_balance_alpha=1e-4, router_z_alpha=1e-3)
    m = M.AnuLM(cfg).to(DEV).train()
    if grad_ckpt:
        m.enable_gradient_checkpointing()
    if ce_chunk:
        m.enable_chunked_loss(ce_chunk)
    args = SimpleNamespace(lr=6e-4, weight_decay=0.1, device=DEV, optimizer="adamw")
    opt = T.make_optimizer(m, args)
    batches = T.strided_windows(va, B, S, B * 8)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    static = torch.cuda.memory_allocated()
    times = []
    for i, ix in enumerate(batches):
        x, y = T.make_batch(va, ix, S, DEV)
        torch.cuda.synchronize()
        t = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(**BF16):
            _, loss = m(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        m.update_expert_biases()
        torch.cuda.synchronize()
        if i == 0:                                   # what warn_if_spilling sees: the peak after step 0
            first = (torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        if i >= 3:                                   # the first steps allocate optimizer state and warm kernels
            times.append(time.perf_counter() - t)
    peak = torch.cuda.max_memory_allocated()
    res = {"peak_gb": peak / 2**30, "reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
           "first_step_peak_gb": first[0] / 2**30, "first_step_reserved_gb": first[1] / 2**30,
           "card_gb": torch.cuda.get_device_properties(DEV).total_memory / 2**30,
           "weights_gb": static / 2**30, "ms_per_step": 1e3 * float(np.median(times)),
           "ms_all": [round(1e3 * t, 1) for t in times]}
    del m, opt
    torch.cuda.empty_cache()
    return res


CONFIGS = {"plain": (False, 0), "grad_ckpt": (True, 0), "ce_chunk 2048": (False, 2048), "both": (True, 2048)}


def report():
    r = json.loads(SAVED.read_text(encoding="utf-8"))
    st, gib, n = r["steps"], 2 ** 30, 397_659_648
    p = st["plain"]
    resident = 12 * n / gib
    print(f"weights + AdamW moments, resident between steps: {resident:.2f} GiB; gradients {4 * n / gib:.2f} GiB")
    print(f"plain: peak after step 0 {p['first_step_peak_gb']:.2f} GiB (no moments yet), over 8 steps "
          f"{p['peak_gb']:.2f}: +{p['peak_gb'] - p['first_step_peak_gb']:.2f} GiB, against moments of {8 * n / gib:.2f}")
    for k, v in st.items():
        print(f"  {k:14s} peak {v['peak_gb']:5.2f} = resident {resident:.2f} + gradients, activations, logits, "
              f"workspace {v['peak_gb'] - resident:.2f} GiB")
    print(f"chunked loss saves {p['peak_gb'] - st['ce_chunk 2048']['peak_gb']:.2f} GiB of peak; "
          f"checkpointing {p['peak_gb'] - st['grad_ckpt']['peak_gb']:.2f} GiB")
    print(f"plain step reserves {p['reserved_gb']:.2f} of {p['card_gb']:.2f} GiB: {p['card_gb'] - p['reserved_gb']:.2f} GiB "
          f"headroom; warn_if_spilling's line {0.92 * p['card_gb']:.2f}; after step 0 {p['first_step_reserved_gb']:.2f}")
    import warnings
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.randn(1, 4, 64, 64, device=DEV, dtype=torch.bfloat16)
    mask = torch.ones(64, 64, dtype=torch.bool, device=DEV).tril()
    print(f"SDPA kernels on this machine (torch {torch.__version__}, bf16, head width 64):")
    for be in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
        row = []
        for name, kw in (("is_causal", dict(is_causal=True)), ("explicit mask", dict(attn_mask=mask))):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    with sdpa_kernel(be):
                        torch.nn.functional.scaled_dot_product_attention(q, q, q, **kw)
                    row.append(f"{name}: ok")
                except RuntimeError:
                    why = any("not compiled with flash" in str(x.message) for x in w)
                    row.append(f"{name}: refused" + (" (this build was not compiled with flash attention)" if why else ""))
        print(f"  {be.name:20s} " + "; ".join(row))


def main():
    if "--report" in sys.argv:
        return report()
    if len(sys.argv) > 1 and sys.argv[1] in CONFIGS:   # one configuration, in a fresh process
        _, va, _ = T.load_memmap(str(ROOT / "data" / "v2" / "meta.json"))
        print("RESULT " + json.dumps(step_cost(*CONFIGS[sys.argv[1]], va)))
        return
    import subprocess
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {free / 2**30:.1f} of {total / 2**30:.1f} GB free")
    assert free > 12 * 2**30, "needs about 12 GB free; run when the card is idle"
    res = {"router": router_flips(), "steps": {}}
    for name in CONFIGS:                             # a fresh process each: no allocator state carried over
        out = subprocess.run([sys.executable, __file__, name], capture_output=True, text=True, check=True).stdout
        r = json.loads(out.split("RESULT ", 1)[1])
        res["steps"][name] = r
        print(f"{name:14s} peak {r['peak_gb']:5.2f} GB allocated, {r['reserved_gb']:5.2f} reserved; after step 0 "
              f"{r['first_step_peak_gb']:5.2f} / {r['first_step_reserved_gb']:5.2f} | {r['ms_per_step']:6.1f} ms/step")
    print(f"card {r['card_gb']:.2f} GB; warn_if_spilling's threshold 0.92 x card = {0.92 * r['card_gb']:.2f} GB; "
          f"weights {r['weights_gb']:.2f} GB; {B} x {S} tokens per step")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
