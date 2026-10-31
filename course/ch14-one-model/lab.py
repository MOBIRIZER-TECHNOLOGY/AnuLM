"""
Chapter 14 lab: one model, three tasks, and a score that can hide a regression.

    python lab.py                      # your course/build/post.py
    python lab.py --impl solution      # the reference solution

Needs ../ch08-pretrain/base.pt. CPU, about five minutes.

Chapter 8's model already writes Shakespeare: that is the skill to keep, like
the chat model AnuLM-Omni started from. Two new tasks join it: Chapter 9's
sums and Chapter 10's Hindi. Shares of the batches: Shakespeare 15%, sums
35%, Hindi 50%, batches of 8. Three runs see the same 2,400 batches:

  one batch a step     2,400 optimizer steps of one batch (the first Omni run)
  accumulate 4         600 steps, each summing 4 batches from any tasks
  one batch, lr / 4    2,400 steps at a quarter of the learning rate

Every 200 batches: each task's held-out loss, and the run's single score
(omni_train.py's: the mean of each task's loss divided by its loss at the
start). Writes results.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lab1 = load(COURSE / "ch01-smallest-model" / "lab.py")
lab9 = load(COURSE / "ch09-sft" / "lab.py")
BATCHES, BATCH, BLOCK, LR = 2400, 8, 128, 3e-4
SHARES = {"shakespeare": 0.15, "sums": 0.35, "hindi": 0.50}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    a = p.parse_args()
    sol = a.impl == "solution"
    Bm = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S9 = load(COURSE / "build" / ("solutions/ch09.py" if sol else "post.py"))
    S = load(COURSE / "build" / ("solutions/ch14.py" if sol else "post.py"))
    base = torch.load(COURSE / "ch08-pretrain" / "base.pt", weights_only=False)

    sh_train, sh_val = lab1.load_bytes(str(COURSE.parent / "data" / "input.txt"))
    hi_train, hi_val = lab1.load_bytes(str(COURSE.parent / "data" / "bench_hindi.txt"))
    enc = lambda s: list(s.encode("utf-8"))
    rnd = random.Random(0)
    train_items, test_items = lab9.make_items(6000, rnd), lab9.make_items(200, rnd)
    sx, sy = S9.pack([S9.pair_ids(enc, S9.BYTE_EOS, q, ans) for q, ans in train_items], BLOCK, S9.BYTE_EOS)
    tx, ty = S9.pack([S9.pair_ids(enc, S9.BYTE_EOS, q, ans) for q, ans in test_items], BLOCK, S9.BYTE_EOS)
    names = list(SHARES)

    def draw_from(name, g):
        if name == "shakespeare":
            return lab1.batch(sh_train, BLOCK, BATCH, g)
        if name == "hindi":
            return lab1.batch(hi_train, BLOCK, BATCH, g)
        i = torch.randint(len(sx), (BATCH,), generator=g)
        return sx[i], sy[i]

    @torch.no_grad()
    def held(m):
        m.eval()
        g = torch.Generator().manual_seed(0)
        out = {"shakespeare": sum(m(*lab1.batch(sh_val, BLOCK, 16, g))[1].item() for _ in range(10)) / 10,
               "sums": m(tx, ty)[1].item(),
               "hindi": sum(m(*lab1.batch(hi_val, BLOCK, 16, g))[1].item() for _ in range(10)) / 10}
        m.train()
        return out

    out = {"batches": BATCHES, "batch": BATCH, "lr": LR, "shares": SHARES}
    for name, accum, lr in (("one batch a step", 1, LR), ("accumulate 4", 4, LR), ("one batch, lr / 4", 1, LR / 4)):
        torch.manual_seed(0)
        m = Bm.MoELM(**base["sizes"])
        m.load_state_dict(base["model"])
        m.train()
        opt = Bm.make_optimizer(m, lr)
        rng = np.random.default_rng(0)
        g = torch.Generator().manual_seed(0)
        seen = {k: 0 for k in names}

        def draw():
            try:
                k = names[S.pick_task(list(SHARES.values()), rng)]
            except NotImplementedError as e:
                sys.exit(f"course/build/post.py is not finished yet ({e}). Run course/checks/ch14.py, or use --impl solution.")
            seen[k] += 1
            return draw_from(k, g)

        first = held(m)
        curve, t0 = [(0, first, 1.0)], time.time()
        for step in range(1, BATCHES // accum + 1):
            for grp in opt.param_groups:
                grp["lr"] = lr * min(1.0, step * accum / 100)
            S.omni_step(m, opt, draw, accum)
            if (step * accum) % 200 == 0:
                h = held(m)
                curve.append((step * accum, h, S.score(h, first)))
                print(f"   {name:18s} batch {step * accum:5d}  score {curve[-1][2]:.3f}  "
                      + "  ".join(f"{k} {v:.3f}" for k, v in h.items()) + f"  ({time.time() - t0:.0f}s)")
        out[name] = {"accum": accum, "lr": lr, "curve": curve, "seen": seen}
        last = curve[-1][1]
        print(f"{name}: score {curve[-1][2]:.3f}; Shakespeare {first['shakespeare']:.3f} -> {last['shakespeare']:.3f}, "
              f"sums {first['sums']:.3f} -> {last['sums']:.3f}, Hindi {first['hindi']:.3f} -> {last['hindi']:.3f}; "
              f"batches seen {seen}")
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
