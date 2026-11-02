"""
Book example, not repository code.

Chapter 24 measurement: how precise is the released model's val loss?

    python book/code-examples/ch24_measure_val_noise.py          # needs the GPU, about a minute

It loads ckpt_base_v2.pt and its own held-out split, data/v2/val.u16 (documents
hashed out of training by experiments/build_v2.py), and measures:

1. the checkpoint's recorded number, re-read through train.evaluate() with the
   checkpoint's own eval_spec (400 strided windows of 1,024 tokens, batch 4);
2. ten window sets of train.py's default evaluation (eval_windows, 20 batches
   of 4), one per seed, and three larger random sets of 400 windows;
3. the loss and bits per byte of each source in the split, with strided
   windows inside each source's stretch (build_v2.py writes the sources one
   after another; their token counts and bytes-per-token ratios are in
   data/v2/meta.json), and the stratified estimate they give together.

Only statistics are kept. No held-out text is decoded, printed or saved.
With --save the results go to book/measurements/part_v_b_val_noise.json;
without it, to the current directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.training import train as T  # noqa: E402  the repository's train.py, unchanged
from anulm.model import AnuLM, load_checkpoint  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Results go to book/measurements/ only with --save, so a reader's rerun never overwrites them.
SAVE = "--save" in sys.argv[1:]
OUT = (ROOT / "book" / "measurements" if SAVE else Path.cwd()) / "part_v_b_val_noise.json"
DEV = "cuda"
BLOCK = 1024


@torch.no_grad()
def batch_losses(m, data, batches, ac):
    """The loss AnuLM.forward returns for each batch, exactly as train.evaluate() reads it."""
    m.eval()
    out = []
    for ix in batches:
        x, y = T.make_batch(data, ix, BLOCK, DEV)
        with ac:
            _, loss = m(x, y)
        out.append(loss.item())
    return torch.tensor(out, dtype=torch.float64)


def mean_and_sem(b):
    """train.evaluate()'s formula for equal-sized batches: standard error across batches."""
    mean = b.mean().item()
    var = ((b - mean) ** 2).mean().item()
    return mean, (var / len(b)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="a quick timing run; writes nothing")
    ap.add_argument("--save", action="store_true", help="write to book/measurements/")
    a = ap.parse_args()
    t0 = time.time()
    ck = load_checkpoint(str(ROOT / "ckpt_base_v2.pt"), "cpu")
    cfg = replace(ck["cfg"], moe_impl="grouped")
    m = AnuLM(cfg).to(DEV).eval()
    m.load_state_dict(ck["model"])
    _, va, meta = T.load_memmap(str(ROOT / "data" / "v2" / "meta.json"))
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    print(f"checkpoint step {ck['step'] + 1:,}; recorded val {ck['val_loss']:.4f} +/- {ck['val_sem']:.4f}; "
          f"eval_spec {ck['eval_spec']}")
    print(f"val split {len(va):,} tokens")
    R = {"checkpoint": "ckpt_base_v2.pt", "step": ck["step"] + 1, "recorded_val": ck["val_loss"],
         "recorded_sem": ck["val_sem"], "eval_spec": ck["eval_spec"], "val_tokens": len(va), "block": BLOCK,
         "dtype": "bf16 autocast, moe_impl grouped"}

    # 1. the recorded number, through the repository's evaluate() and the checkpoint's eval_spec
    args = argparse.Namespace(batch_size=4, block_size=BLOCK, device=DEV, autocast=ac,
                              eval_windows=400, eval_iters=20)
    mean, sem = T.evaluate(m, va, args)
    print(f"\n1. evaluate(), 400 strided windows, batch 4: {mean:.4f} +/- {sem:.4f}")
    R["strided400"] = {"mean": mean, "sem": sem}
    # The same 400 windows batched 1 and 8 at a time: the mean must not move, the "+/-" does,
    # because evaluate() takes its spread across batches and strided batches hold neighbours.
    R["strided400_by_batch"] = {"4": {"mean": mean, "sem": sem}}
    for bs in (1, 8):
        args.batch_size = bs
        mu, se = T.evaluate(m, va, args)
        R["strided400_by_batch"][str(bs)] = {"mean": mu, "sem": se}
        print(f"   the same 400 windows, batch {bs}: {mu:.4f} +/- {se:.4f}")
    args.batch_size = 4

    # 2. train.py's default evaluation (eval_windows, 20 batches) for ten seeds; evaluate() itself
    #    for seed 0, which is the only seed it uses, then the same loop for the others.
    args.eval_windows = 0
    m0, s0 = T.evaluate(m, va, args)
    w0 = batch_losses(m, va, T.eval_windows(va, 4, BLOCK, 20, seed=0), ac)
    assert abs(mean_and_sem(w0)[0] - m0) < 1e-6, "the loop must reproduce evaluate() on seed 0"
    n_small, n_big = (3, 1) if a.small else (10, 3)
    small = [{"seed": 0, "mean": m0, "sem": s0}]
    for seed in range(1, n_small):
        mu, se = mean_and_sem(batch_losses(m, va, T.eval_windows(va, 4, BLOCK, 20, seed=seed), ac))
        small.append({"seed": seed, "mean": mu, "sem": se})
    for s in small:
        print(f"2. random, 20 batches x 4 = 80 windows, seed {s['seed']}: {s['mean']:.4f} +/- {s['sem']:.4f}")
    big = []
    for seed in range(100, 100 + n_big):
        mu, se = mean_and_sem(batch_losses(m, va, T.eval_windows(va, 4, BLOCK, 100, seed=seed), ac))
        big.append({"seed": seed, "mean": mu, "sem": se})
        print(f"   random, 100 batches x 4 = 400 windows, seed {seed}: {mu:.4f} +/- {se:.4f}")
    sm = torch.tensor([s["mean"] for s in small])
    R["random80"] = {"sets": small, "mean_of_means": sm.mean().item(), "sd_of_means": sm.std().item(),
                     "spread": (sm.max() - sm.min()).item(), "min": sm.min().item(), "max": sm.max().item(),
                     "mean_sem": float(sum(s["sem"] for s in small) / len(small))}
    R["random400"] = {"sets": big}
    print(f"   80-window sets: mean of means {sm.mean():.4f}, sd {sm.std():.4f}, "
          f"range {sm.min():.4f} to {sm.max():.4f}, typical +/- {R['random80']['mean_sem']:.4f}")

    # 3. per source, with strided windows inside each source's stretch of the split
    edges, lo = {}, 0
    for name, s in meta["sources"].items():           # build_v2.py writes the sources one after another
        edges[name] = (lo, lo + s["val_tokens"])
        lo += s["val_tokens"]
    assert lo == len(va), "the sources' val tokens must add up to the split"
    per_n = {"english": 400, "hindi_web": 400, "hindi_wiki": 100, "python": 200}
    if a.small:
        per_n = {k: 40 for k in per_n}
    per, strat, strat_var = {}, 0.0, 0.0
    print("\n3. per source")
    for name, (a0, a1) in edges.items():
        part = va[a0:a1]
        b = batch_losses(m, part, T.strided_windows(part, 4, BLOCK, per_n[name]), ac)
        mu, se = mean_and_sem(b)
        share = (a1 - a0) / len(va)
        bpt = meta["sources"][name]["bytes_per_token"]
        per[name] = {"tokens": a1 - a0, "share": share, "windows": per_n[name], "loss": mu, "sem": se,
                     "bytes_per_token": bpt, "bits_per_byte": mu / math.log(2) / bpt}
        strat += share * mu
        strat_var += (share * se) ** 2
        print(f"   {name:11s} {100 * share:5.1f}% of tokens  {per_n[name]:4d} windows  loss {mu:.3f} +/- {se:.3f}"
              f"  bytes/token {bpt:.3f}  bits/byte {per[name]['bits_per_byte']:.3f}")
    R["per_source"] = per
    R["stratified"] = {"mean": strat, "sem": strat_var ** 0.5}
    print(f"   stratified estimate (token shares x source means): {strat:.4f} +/- {strat_var ** 0.5:.4f}")
    R["bytes_per_token_meta"] = meta["bytes_per_token"]
    R["bits_per_byte_train_py"] = mean / math.log(2) / meta["bytes_per_token"]
    print(f"   train.py's bits/byte: {mean:.4f} / ln 2 / {meta['bytes_per_token']} = {R['bits_per_byte_train_py']:.4f}")

    # where the default random windows and 80 strided windows fall (Figure 24.1)
    R["positions"] = {"random80_seed0": [int(i) for b in T.eval_windows(va, 4, BLOCK, 20, seed=0) for i in b],
                      "strided80": [int(i) for b in T.strided_windows(va, 4, BLOCK, 80) for i in b],
                      "edges": {k: list(v) for k, v in edges.items()}}
    R["seconds"] = time.time() - t0
    print(f"\n{R['seconds']:.0f}s")
    if not a.small:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(R, indent=1), encoding="utf-8")
        print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
