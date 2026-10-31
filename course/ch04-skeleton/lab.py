"""
Chapter 4 lab: train the skeleton you built -- embeddings, RMSNorm, SwiGLU,
residual blocks, no attention yet -- and see exactly how far it gets.

    python lab.py                      # your course/build/anulm.py
    python lab.py --impl solution      # the reference solution
    python lab.py --data my_text.txt   # your own text

CPU, about two minutes. Writes curve.csv and results.json for the figures.

The prediction to test: with no attention, each position sees only its own
byte, so however deep the model, it can only learn P(next byte | this byte).
That is the bigram table of Chapter 1, learned by gradient descent instead
of counted. It should stop where the table stopped.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
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


lab1 = load(COURSE / "ch01-smallest-model" / "lab.py")     # data, batches, the bigram table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    p.add_argument("--data", default=str(COURSE.parent / "data" / "input.txt"))
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()
    M = load(COURSE / "build" / ("solutions/ch04.py" if a.impl == "solution" else "anulm.py"))

    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    train, val = lab1.load_bytes(a.data)
    _, bigram_loss = lab1.bigram(train, val)

    try:
        model = M.SkeletonLM(vocab_size=259, hidden_size=128, n_layer=a.layers, intermediate_size=256)
    except NotImplementedError as e:
        sys.exit(f"course/build/anulm.py is not finished yet ({e}).\n"
                 f"Fill it in until `python course/checks/ch04.py` passes, or run with --impl solution.")
    n_params = sum(t.numel() for t in model.parameters())
    ctx = 128

    @torch.no_grad()
    def evaluate():
        model.eval()
        eg = torch.Generator().manual_seed(0)
        loss = sum(model(*lab1.batch(val, ctx, 32, eg))[1].item() for _ in range(20)) / 20
        model.train()
        return loss

    print(f"skeleton ({'reference' if a.impl == 'solution' else 'yours'}): {n_params:,} parameters, "
          f"{a.layers} blocks, no attention")
    print(f"start: validation loss {evaluate():.3f} (ln 259 = {math.log(259):.3f})")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    rows, t0 = [(0, evaluate())], time.time()
    for step in range(1, a.steps + 1):
        for group in opt.param_groups:
            group["lr"] = 1e-3 * min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        _, loss = model(*lab1.batch(train, ctx, 32, g))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            rows.append((step, evaluate()))
            print(f"  step {step:5d}  validation loss {rows[-1][1]:.3f}  ({time.time() - t0:.0f}s)")
    final = rows[-1][1]
    print(f"\nskeleton {final:.3f}   bigram table {bigram_loss:.3f}   "
          f"(difference {final - bigram_loss:+.3f} nats/byte)")

    ch1 = COURSE / "ch01-smallest-model" / "results.json"
    tiny = json.loads(ch1.read_text(encoding="utf-8"))["tiny_val_loss"] if ch1.exists() else None
    with open(HERE / "curve.csv", "w", newline="") as f:
        csv.writer(f).writerows([("step", "val_loss")] + [(s, round(v, 4)) for s, v in rows])
    (HERE / "results.json").write_text(json.dumps({
        "data": Path(a.data).name, "params": n_params, "layers": a.layers, "steps": a.steps,
        "seconds": round(time.time() - t0), "start_loss": round(rows[0][1], 4), "final_loss": round(final, 4),
        "bigram_loss": round(bigram_loss, 4), "ch1_transformer_loss": tiny,
        "ln_vocab": round(math.log(259), 4)}, indent=1), encoding="utf-8")
    print("saved curve.csv, results.json")


if __name__ == "__main__":
    main()
