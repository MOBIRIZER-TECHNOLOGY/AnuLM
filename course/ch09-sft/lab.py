"""
Chapter 9 lab: fine-tune Chapter 8's Shakespeare model to answer questions.

    python lab.py                      # your course/build/post.py and anulm.py
    python lab.py --impl solution      # the reference solutions

CPU, about twelve minutes. The task is new to the model: two-digit sums asked
in English ("Question: what is 23 plus 45?\nAnswer: 68"). Four runs, 1,500
steps each, same data and settings:

  answer-only   from the Chapter 8 base, loss on the answer only (finetune.py's way)
  whole text    from the same base, loss on every token, the prompt included
  no base       answer-only, from random weights: what did pre-training give?
  with replay   answer-only from the base, every step also a batch of Shakespeare

Every 250 steps: exact-match accuracy on 200 held-out sums. At the end: how
much each base-started model forgot of Shakespeare. Writes results.json, and
sft.pt (the answer-only model, which Chapter 11 starts from).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
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
STEPS, LR, BATCH, BLOCK = 1500, 3e-4, 16, 128


def make_items(n, rnd):
    out = []
    for _ in range(n):
        a, b = rnd.randint(10, 99), rnd.randint(10, 99)
        out.append((f"what is {a} plus {b}?", str(a + b)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    p.add_argument("--arms", nargs="*", help="run only these arms; results.json keeps the others")
    a = p.parse_args()
    sol = a.impl == "solution"
    B = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S = load(COURSE / "build" / ("solutions/ch09.py" if sol else "post.py"))
    base = torch.load(COURSE / "ch08-pretrain" / "base.pt", weights_only=False)
    enc = lambda s: list(s.encode("utf-8"))
    rnd = random.Random(0)
    train_items, test_items = make_items(6000, rnd), make_items(200, rnd)
    shakespeare_train, shakespeare_val = lab1.load_bytes(str(COURSE.parent / "data" / "input.txt"))
    try:
        pairs = [S.pair_ids(enc, S.BYTE_EOS, q, ans) for q, ans in train_items]
    except NotImplementedError as e:
        sys.exit(f"course/build/post.py is not finished yet ({e}). Run course/checks/ch09.py, or use --impl solution.")

    @torch.no_grad()
    def accuracy(m):
        m.eval()
        right = 0
        for q, ans in test_items:
            prompt = torch.tensor([enc(S.PROMPT_EN.format(q=q))])
            out = m.generate(prompt, 5, temperature=1e-5)                    # greedy
            text = bytes([t for t in out[0, prompt.shape[1]:].tolist() if t < 256]).decode("utf-8", "replace")
            right += text.strip().split()[:1] == [ans]
        m.train()
        return right / len(test_items)

    @torch.no_grad()
    def shakespeare(m):
        m.eval()
        g = torch.Generator().manual_seed(0)
        loss = sum(m(*lab1.batch(shakespeare_val, 128, 32, g))[1].item() for _ in range(20)) / 20
        m.train()
        return loss

    path = HERE / "results.json"
    old = json.loads(path.read_text(encoding="utf-8")) if a.arms and path.exists() else {}
    out = {**old, "steps": STEPS, "test": len(test_items), "example": S.PROMPT_EN.format(q=test_items[0][0])
           + " " + test_items[0][1]}
    for name, from_base, answer_only, replay in (("answer-only", True, True, False),
                                                 ("whole text", True, False, False),
                                                 ("no base", False, True, False),
                                                 ("with replay", True, True, True)):
        if a.arms and name not in a.arms:
            continue
        torch.manual_seed(0)
        m = B.MoELM(**base["sizes"])
        if from_base:
            m.load_state_dict(base["model"])
        m.train()
        opt = B.make_optimizer(m, LR)
        ps = pairs if answer_only else [([], pp + aa) for pp, aa in pairs]   # whole text: no prompt mask
        x, y = S.pack(ps, BLOCK, S.BYTE_EOS)
        g = torch.Generator().manual_seed(0)
        before = shakespeare(m) if from_base else None
        curve, t0 = [(0, accuracy(m))], time.time()
        for step in range(1, STEPS + 1):
            i = torch.randint(len(x), (BATCH,), generator=g)
            batches = [(x[i], y[i])]
            if replay:                       # half of each step's gradient: the text it already knew
                batches.append(lab1.batch(shakespeare_train, BLOCK, BATCH, g))
            B.train_step(m, opt, batches, LR * min(1.0, step / 50))
            if step % 250 == 0:
                curve.append((step, accuracy(m)))
                print(f"   {name:11s} step {step:5d}  accuracy {curve[-1][1]:.1%}  ({time.time() - t0:.0f}s)")
        after = shakespeare(m) if from_base else None
        if name == "answer-only":            # Chapter 11 starts from this model
            torch.save({"sizes": base["sizes"], "model": m.state_dict()}, HERE / "sft.pt")
        out[name] = {"curve": curve, "final": curve[-1][1], "shakespeare_before": before, "shakespeare_after": after}
        print(f"{name}: {curve[-1][1]:.1%} of {len(test_items)} sums right"
              + (f"; Shakespeare loss {before:.3f} -> {after:.3f}" if from_base else ""))
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
