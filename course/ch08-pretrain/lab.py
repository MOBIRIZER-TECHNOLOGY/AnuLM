"""
Chapter 8 lab: pre-train your model with your own training machinery, and see
the four things a long run depends on.

    python lab.py                      # your course/build/anulm.py
    python lab.py --impl solution      # the reference solution
    python lab.py --data my_text.txt   # your own text

CPU, about five minutes. Chapter 7's balanced expert model, on Shakespeare:

  1. a warmup-stable-decay run of 1,500 steps, with a resume point saved at step 800;
  2. a crash and a resume: a fresh process loads that point and continues; its
     loss must match the uninterrupted run's exactly;
  3. decaying a copy: from the same point, a second model decays its learning rate
     over 200 steps, while the main run is still at its peak;
  4. the error bar: the finished model measured on six different sets of windows.
Writes results.json for the figures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
import tempfile
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
SIZES = dict(vocab_size=259, block_size=256, n_layer=4, hidden_size=128, n_head=4, head_dim=32, n_kv_head=2,
             rope_theta=10000.0, intermediate_size=256, num_experts=8, num_experts_per_tok=3,
             num_shared_experts=1, moe_intermediate_size=64)
STEPS, WARMUP, LR, MIN_LR, DECAY_FRAC = 1500, 100, 1e-3, 1e-4, 0.2
SAVE_AT, COPY_STEPS, CTX = 799, 200, 128


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    p.add_argument("--data", default=str(COURSE.parent / "data" / "input.txt"))
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()
    M = load(COURSE / "build" / ("solutions/ch08.py" if a.impl == "solution" else "anulm.py"))
    train_data, val_data = lab1.load_bytes(a.data)

    def fresh():
        torch.manual_seed(a.seed)
        m = M.MoELM(**SIZES).train()
        return m, M.make_optimizer(m, LR)

    @torch.no_grad()
    def evaluate(m, seed=0):
        m.eval()
        g = torch.Generator().manual_seed(seed)
        losses = [m(*lab1.batch(val_data, CTX, 32, g))[1].item() for _ in range(20)]
        m.train()
        return sum(losses) / len(losses)

    def steps(m, opt, g, first, last, lr_fn, log, label):
        for step in range(first, last + 1):
            x, y = lab1.batch(train_data, CTX, 32, g)
            M.train_step(m, opt, [(x[:16], y[:16]), (x[16:], y[16:])], lr_fn(step))   # 2 micro-batches
            if (step + 1) % 100 == 0:
                log.append((step + 1, round(evaluate(m), 4)))
                print(f"   {label:8s} step {step + 1:5d}  lr {lr_fn(step):.1e}  validation {log[-1][1]:.4f}")

    try:
        m, opt = fresh()
    except NotImplementedError as e:
        sys.exit(f"course/build/anulm.py is not finished yet ({e}).\n"
                 f"Fill it in until `python course/checks/ch08.py` passes, or run with --impl solution.")
    wsd = lambda s: M.lr_at(s, STEPS, LR, MIN_LR, WARMUP, "wsd", DECAY_FRAC)
    out = {"steps": STEPS, "decay_start": int(STEPS * (1 - DECAY_FRAC)), "save_at": SAVE_AT + 1,
           "copy_steps": COPY_STEPS, "lr": LR, "min_lr": MIN_LR,
           "schedule": [(s, wsd(s)) for s in range(0, STEPS, 10)]}

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "run.last"
        # 1. the main run, with a resume point at step 800
        print("1. warmup-stable-decay, 1,500 steps")
        g = torch.Generator().manual_seed(a.seed)
        main_log, t0 = [(0, round(evaluate(m), 4))], time.time()
        steps(m, opt, g, 0, SAVE_AT, wsd, main_log, "main")
        M.save_checkpoint(ckpt, m, opt, step=SAVE_AT, data_rng=g.get_state())
        steps(m, opt, g, SAVE_AT + 1, STEPS - 1, wsd, main_log, "main")
        out["main"] = main_log
        out["seconds"] = round(time.time() - t0)

        # 2. a crash and a resume: a new model, a different random start, then the checkpoint
        print("2. resume from step 800 in a fresh model, 100 steps")
        torch.manual_seed(999)
        m2 = M.MoELM(**SIZES).train()
        opt2 = M.make_optimizer(m2, LR)
        ck = M.load_checkpoint(ckpt, m2, opt2)
        g2 = torch.Generator()
        g2.set_state(ck["data_rng"])
        resumed = []
        steps(m2, opt2, g2, ck["step"] + 1, ck["step"] + 100, wsd, resumed, "resumed")
        main_at = dict(main_log)[ck["step"] + 101]
        out["resume"] = {"step": ck["step"] + 101, "resumed": resumed[-1][1], "uninterrupted": main_at}
        print(f"   at step {ck['step'] + 101}: resumed {resumed[-1][1]:.4f}, uninterrupted {main_at:.4f}"
              f" -> {'identical' if resumed[-1][1] == main_at else 'DIFFERENT'}")

        # 3. decay a copy over 200 steps from the same point
        print("3. decay a copy from step 800 over 200 steps")
        torch.manual_seed(999)
        m3 = M.MoELM(**SIZES).train()
        opt3 = M.make_optimizer(m3, LR)
        ck = M.load_checkpoint(ckpt, m3, opt3)
        g3 = torch.Generator()
        g3.set_state(ck["data_rng"])
        first = ck["step"] + 1
        copy_lr = lambda s: LR + (MIN_LR - LR) * (s - first + 1) / COPY_STEPS
        copy_log = [(first, dict(main_log)[first])]
        steps(m3, opt3, g3, first, first + COPY_STEPS - 1, copy_lr, copy_log, "copy")
        out["copy"] = copy_log

    # 4. the error bar
    sets = [evaluate(m, seed) for seed in range(6)]
    out["window_sets"] = [round(v, 4) for v in sets]
    print(f"4. the final model on six window sets: {' '.join(f'{v:.4f}' for v in sets)}  "
          f"(spread {max(sets) - min(sets):.4f}, sd {statistics.stdev(sets):.4f})")
    end = first + COPY_STEPS
    print(f"\nat step {end}: main run (still at peak lr) {dict(main_log)[end]:.4f}, decayed copy {copy_log[-1][1]:.4f};"
          f" main run after its own decay (step {STEPS}) {main_log[-1][1]:.4f}")
    torch.manual_seed(7)
    sample = m.eval().generate(torch.tensor([list(b"ROMEO:\n")]), 200, temperature=0.8, top_k=40,
                               repetition_penalty=1.1)
    out["sample"] = bytes(sample[0].tolist()).decode("utf-8", errors="replace")
    print(out["sample"])
    (HERE / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    # The finished model is Part III's base: Chapters 9-12 fine-tune it.
    torch.save({"sizes": SIZES, "model": m.state_dict(), "data": Path(a.data).name}, HERE / "base.pt")
    print("saved results.json and base.pt (the base model for Part III)")


if __name__ == "__main__":
    main()
