"""
Book example, not repository code.

Chapter 41, checkpoints 1 and 4: what you can check about a training run
before it starts, and how your model compares with the released one. It
trains nothing.

    python book/code-examples/ch41_checkpoints.py first-loss            # CPU, about 6 s
    python book/code-examples/ch41_checkpoints.py released              # needs ckpt_base_v2.pt and data/v2/val.u16
    python book/code-examples/ch41_checkpoints.py released --ckpt release/AnuLM-Base-v2-400M
    python book/code-examples/ch41_checkpoints.py released --save       # the book's own run

first-loss
    Builds the model `python train.py --preset 30b --device cpu` builds, from
    the same seed, data and flags. It prints the loss of the first training
    batch the sampler draws (what train.py logs as step 0, before any update)
    and the validation loss before any step, from train.py's own `evaluate`.
    An untrained model should score close to ln V, the loss of a uniform guess
    over V = 259 byte ids.

released
    Loads the released checkpoint (any path `load_checkpoint` accepts), lists
    every configuration field that differs from `trace_forward.released()` and
    from the defaults, and measures how many tokens top-p would keep (the
    nucleus size) at each position of 8 windows of 512 tokens from the held-out
    split `data/v2/val.u16`, in fp32: the MoE runs on the `sparse` path, because
    the checkpoint's own `grouped` path computes the experts in bf16. GPU if
    at least 4 GB is free, else CPU. Only statistics are printed and saved,
    never text. Results go to ./ch41_released.json, or with --save to
    book/measurements/part_x_released.json.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged
from anulm.training import train as T  # noqa: E402  the repository's train.py, unchanged
from ch41_top_p import top_p  # noqa: E402
from trace_forward import released  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SAVED = ROOT / "book" / "measurements" / "part_x_released.json"


def first_loss() -> None:
    torch.manual_seed(1337)                                  # train.py's default --seed
    data = T.load_bytes(None)                                # data/input.txt, as train.py with no --data
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus {len(data) / 1e6:.2f}M bytes  ->  train {len(train_data) / 1e6:.2f}M / "
          f"val {len(val_data) / 1e6:.2f}M")
    sampler = T.WindowSampler(len(train_data), 256, 16, 1337)
    print(f"sampler: {len(sampler.order)} windows/epoch; 1000 steps of 16 = "
          f"{1000 * 16 / len(sampler.order):.2f} epochs")
    cfg = M.AnuLMConfig.nano_30b(block_size=256, yarn=False, yarn_original_context=256, moe_impl="sparse",
                                 vocab_size=259, tokenizer_path=None, seq_balance_alpha=0.0,
                                 router_z_alpha=0.0, sliding_window=None, max_window_layers=0)
    model = M.AnuLM(cfg)
    total, active = model.num_params()
    print(f"params: {total:,} total, {active:,} active per token ({100 * active / total:.1f}%)")
    x, y = T.make_batch(train_data, sampler.next(), 256, "cpu")       # the batch train.py's step 0 uses
    with torch.no_grad():
        _, loss0 = model.train()(x, y)
    print(f"step 0 training loss (first batch, before the update): {loss0.item():.4f}")
    args = argparse.Namespace(batch_size=16, block_size=256, device="cpu", eval_iters=20, eval_windows=0,
                              autocast=contextlib.nullcontext())
    val, sem = T.evaluate(model, val_data, args)
    print(f"val loss before any step: {val:.4f} +/- {sem:.4f}   ln V = ln 259 = {math.log(259):.4f}"
          f"   (difference {val - math.log(259):.4f})")
    assert abs(val - math.log(259)) < 0.35, "the same tolerance as test_model.py's initial_loss_is_ln_vocab"
    t1, a1 = M.AnuLM(M.AnuLMConfig.nano_30b(n_kv_head=1)).num_params()    # Exercise 1: --cfg n_kv_head=1
    print(f"exercise 1, n_kv_head=1: {t1:,} total, {a1:,} active ({total - t1:,} fewer)")


def config_differences(cfg) -> dict:
    mine, default = dataclasses.asdict(released()), dataclasses.asdict(M.AnuLMConfig())
    ck = dataclasses.asdict(cfg)
    return {"differs_from_released()": {k: [mine[k], ck[k]] for k in ck if ck[k] != mine[k]},
            "differs_from_defaults": {k: [default[k], ck[k]] for k in ck if ck[k] != default[k]}}


def pick_device() -> str:
    """The GPU only when at least 4 GB is free; a busy or broken GPU falls back to the CPU."""
    try:
        if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2**30 >= 4:
            return "cuda"
    except Exception as e:                                   # e.g. an AcceleratorError on a busy card
        print(f"(GPU unavailable: {type(e).__name__}; using the CPU)")
    return "cpu"


def released_vs_yours(ckpt: str, out: Path, windows=8, seq=512, seed=0) -> None:
    ck = M.load_checkpoint(ckpt, "cpu")
    cfg = ck["cfg"]
    diff = config_differences(cfg)
    print(f"fields where {ckpt} differs from released():")
    for k, (a, b) in diff["differs_from_released()"].items():
        print(f"  {k:<22} released() {a!r:<14} checkpoint {b!r}")
    m = M.AnuLM(dataclasses.replace(cfg, moe_impl="sparse"))      # fp32 experts: grouped would use bf16
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()}, strict=True)
    total, active = m.num_params()
    print(f"params: {total:,} total, {active:,} active; stored step {ck.get('step', 0):,}, "
          f"val {ck.get('val_loss', float('nan')):.4f} +/- {ck.get('val_sem', float('nan')):.4f} "
          f"({ck.get('eval_spec')})")
    t30, a30 = M.AnuLM(M.AnuLMConfig.nano_30b()).num_params()
    print(f"against nano_30b ({t30:,} / {a30:,}): {total / t30:.1f}x the parameters, {active / a30:.1f}x the active")
    dev = pick_device()
    m = m.to(dev).eval()
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    starts = np.random.default_rng(seed).integers(0, len(val) - seq - 1, size=windows)
    sizes = {0.5: [], 0.9: []}
    top50 = []
    with torch.no_grad():
        for s in starts:
            row = torch.from_numpy(val[s: s + seq + 1].astype(np.int64))[None].to(dev)
            x, y = row[:, :-1], row[:, 1:]
            logits, _ = m(x, y)                                        # (1, seq, V): targets ask for every row
            probs = F.softmax(logits[0].float(), dim=-1)               # (seq, V)
            keep = x[0] != 32767                                       # skip the position after a document end
            for p in sizes:
                sizes[p] += (top_p(probs, p) > 0).sum(-1)[keep].tolist()
            top50 += probs.topk(50, dim=-1).values.sum(-1)[keep].tolist()
    precision = "fp32 weights and activations, moe_impl sparse, no autocast"
    res = {"checkpoint": str(ckpt), "data": "data/v2/val.u16", "windows": windows, "seq": seq,
           "seed": seed, "device": dev, "precision": precision, "positions": len(top50), "config": diff,
           "num_params": [total, active], "nucleus": {}}
    print(f"data/v2/val.u16: {windows} windows of {seq} tokens, seed {seed}; {precision}; {dev}")
    for p, v in sizes.items():
        a = np.array(v)
        q = {k: int(np.percentile(a, k)) for k in (10, 25, 50, 75, 90)}
        edges = [2 ** k for k in range(16)]                  # bins (2^(k-1), 2^k]: 1, 2, 3-4, 5-8, ...
        hist = [int(((a > e // 2) & (a <= e)).sum()) if e > 1 else int((a == 1).sum()) for e in edges]
        assert sum(hist) == len(a)
        res["nucleus"][str(p)] = {"percentiles": q, "max": int(a.max()), "min": int(a.min()),
                                  "share_size_1": float((a == 1).mean()), "share_over_50": float((a > 50).mean()),
                                  "hist_upper_edges": edges, "hist": hist}
        print(f"top-p {p}: nucleus size percentiles {q}, min {a.min()}, max {a.max()}; "
              f"size 1 at {(a == 1).mean():.1%} of positions, over 50 at {(a > 50).mean():.1%}")
    t = np.array(top50)
    res["top50_mass"] = {k: float(np.percentile(t, k)) for k in (10, 50, 90)}
    print(f"mass inside sample.py's default top-k 50: 10th pct {np.percentile(t, 10):.3f}, "
          f"median {np.percentile(t, 50):.3f}, 90th pct {np.percentile(t, 90):.3f}  ({len(t):,} positions)")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="first-loss", choices=["first-loss", "released"])
    ap.add_argument("--ckpt", default=str(ROOT / "ckpt_base_v2.pt"))
    ap.add_argument("--save", action="store_true", help="write the book's measurement file")
    ap.add_argument("--out", default="ch41_released.json")
    a = ap.parse_args()
    if a.what == "first-loss":
        first_loss()
    else:
        released_vs_yours(a.ckpt, SAVED if a.save else Path(a.out))
