"""
Chapter 16 lab: keep pre-training a finished model, and teach it a longer window.

    python lab.py                      # your course/build/further.py
    python lab.py --impl solution      # the reference solution

Needs ../ch08-pretrain/base.pt, which was trained on windows of 128 bytes and
finished its schedule at the minimum learning rate. CPU, about eight minutes.

Part 1, resuming. Continue it on the same text at the peak learning rate
(1e-3), two ways, 300 steps each: jumping straight there, and ramping up over
100 steps (rewarm). Validation loss every 25 steps.

Part 2, a longer window. Read windows of 512 bytes, four times the trained
length. Loss in each quarter of the window for:
  plain RoPE        the base model, positions it never saw
  YaRN, no training the base model with extend_context (factor 4)
  trained, plain    400 steps on 512-byte windows, plain RoPE
  trained, YaRN     400 steps on 512-byte windows, with YaRN
Writes results.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

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
TRAINED, LONG, PEAK = 128, 512, 1e-3


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    a = p.parse_args()
    sol = a.impl == "solution"
    Bm = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S = load(COURSE / "build" / ("solutions/ch16.py" if sol else "further.py"))
    base = torch.load(COURSE / "ch08-pretrain" / "base.pt", weights_only=False)
    train_data, val = lab1.load_bytes(str(COURSE.parent / "data" / "input.txt"))
    theta = base["sizes"]["rope_theta"]

    def fresh(block=None):
        m = Bm.MoELM(**dict(base["sizes"], block_size=block or base["sizes"]["block_size"]))
        m.load_state_dict(base["model"])
        return m.train()

    def val_loss(m, ctx):
        return lab1.evaluate(m, val, ctx, torch.Generator().manual_seed(0))

    out = {"trained": TRAINED, "long": LONG, "peak": PEAK, "resume": {}}
    start = val_loss(fresh(), TRAINED)
    print(f"the base model: validation loss {start:.4f} on {TRAINED}-byte windows")
    for name, ramp in (("jump", 0), ("rewarm 100", 100)):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        m = fresh()
        opt = Bm.make_optimizer(m, PEAK)
        curve, worst = [(0, start)], start
        for step in range(1, 301):
            try:
                lr = S.rewarm(PEAK, 1000 + step, 1000, ramp)        # as if resumed at step 1,000
            except NotImplementedError as e:
                sys.exit(f"course/build/further.py is not finished yet ({e}). Run course/checks/ch16.py, "
                         "or use --impl solution.")
            Bm.train_step(m, opt, [lab1.batch(train_data, TRAINED, 16, g)], lr)
            if step % 25 == 0:
                curve.append((step, val_loss(m, TRAINED)))
                worst = max(worst, curve[-1][1])
        out["resume"][name] = {"curve": curve, "worst": worst, "final": curve[-1][1]}
        print(f"resume, {name:10s}: worst {worst:.4f}, after 300 steps {curve[-1][1]:.4f}")

    def by_position(m):
        m.eval()
        r = S.loss_by_position(m, val, LONG, 40, 4)
        m.train()
        return r

    pos = {}
    pos["plain RoPE"] = by_position(fresh(LONG))
    m = fresh(LONG)
    S.extend_context(m, LONG, LONG / TRAINED, TRAINED, theta)
    pos["YaRN, no training"] = by_position(m)
    for name, yarn in (("trained, plain", False), ("trained, YaRN", True)):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(1)
        m = fresh(LONG)
        if yarn:
            S.extend_context(m, LONG, LONG / TRAINED, TRAINED, theta)
        opt = Bm.make_optimizer(m, 3e-4)
        t0 = time.time()
        for step in range(1, 401):
            Bm.train_step(m, opt, [lab1.batch(train_data, LONG, 8, g)], S.rewarm(3e-4, 1000 + step, 1000, 50))
        pos[name] = by_position(m)
        print(f"   {name} trained in {time.time() - t0:.0f}s")
    out["by_position"] = pos
    print(f"\nloss by quarter of a {LONG}-byte window (the model was trained on the first {TRAINED}):")
    for name, v in pos.items():
        print(f"  {name:18s} " + "  ".join(f"{x:.4f}" for x in v) + f"   last - first {v[-1] - v[0]:+.4f}")
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
