"""
Chapter 18 lab: two-turn conversations, and what grading which turns does.

    python lab.py                      # your course/build/further.py
    python lab.py --impl solution      # the reference solution

Needs ../ch09-sft/sft.pt: Chapter 9's model, which already answers single sums. CPU,
about nine minutes.

Every conversation has two turns, and the second only makes sense after the
first:

    User: what is 23 plus 45?
    Assistant: 68
    User: and plus 17?
    Assistant: 85

Three runs from Chapter 9's model, 1,500 steps each on 6,000 conversations:
  every reply     both replies graded (conversation_ids)
  last reply      only the second reply graded
  user too        everything graded, the user's words included
Then 200 new conversations, held out: is the first reply right? Is the second,
when the model reads its OWN first reply, as it would in a real chat?
Part 2 counts what make_chat.py's split leaked in the repository's chat data.
Writes results.json.
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
ROOT = COURSE.parent
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


STEPS, BATCH, BLOCK, LR, EOS = 1500, 16, 128, 3e-4, 257


def talks(n, rnd):
    out = []
    for _ in range(n):
        a, b, c = rnd.randint(10, 99), rnd.randint(10, 99), rnd.randint(10, 99)
        out.append([(f"what is {a} plus {b}?", str(a + b)), (f"and plus {c}?", str(a + b + c))])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    a = p.parse_args()
    sol = a.impl == "solution"
    Bm = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S = load(COURSE / "build" / ("solutions/ch18.py" if sol else "further.py"))
    base = torch.load(COURSE / "ch09-sft" / "sft.pt", weights_only=False)
    enc = lambda s: list(s.encode("utf-8"))
    rnd = random.Random(0)
    train, test = talks(6000, rnd), talks(200, rnd)
    try:
        convs = [S.conversation_ids(enc, EOS, t) for t in train]
    except NotImplementedError as e:
        sys.exit(f"course/build/further.py is not finished yet ({e}). Run course/checks/ch18.py, or use --impl solution.")
    example = bytes(t for t in convs[0][0] if t < 256).decode()

    def regrade(ids, labels, how):
        if how == "every reply":
            return ids, labels
        if how == "user too":
            return ids, list(ids)
        last = max(i for i, t in enumerate(labels[:-1]) if t == EOS)                 # the first reply's EOS
        return ids, [-1] * (last + 1) + labels[last + 1:]

    def reply(m, ids):
        out = m.generate(torch.tensor([ids]), 6, temperature=1e-5, eos_id=EOS)[0, len(ids):].tolist()
        got = out[: out.index(EOS) + 1] if EOS in out else out
        return got, bytes(t for t in got if t < 256).decode("utf-8", "replace").strip()

    @torch.no_grad()
    def score(m):
        m.eval()
        first = second = second_given = 0
        for (q1, a1), (q2, a2) in test:
            ids = enc(S.CHAT.format(q=q1))
            got, text = reply(m, ids)
            first += text == a1
            convo = ids + got + ([] if got[-1:] == [EOS] else [EOS]) + enc(S.CHAT.format(q=q2))
            second += reply(m, convo)[1] == a2
            truth = ids + enc(" " + a1) + [EOS] + enc(S.CHAT.format(q=q2))
            second_given += reply(m, truth)[1] == a2
        m.train()
        n = len(test)
        return {"first": first / n, "second": second / n, "second_given_right_first": second_given / n}

    out = {"steps": STEPS, "example": example, "arms": {}}
    for how in ("every reply", "last reply", "user too"):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        m = Bm.MoELM(**base["sizes"])
        m.load_state_dict(base["model"])
        m.train()
        opt = Bm.make_optimizer(m, LR)
        x, y = S.pack_conversations([regrade(i, l, how) for i, l in convs], BLOCK, EOS)
        t0 = time.time()
        for step in range(1, STEPS + 1):
            i = torch.randint(len(x), (BATCH,), generator=g)
            Bm.train_step(m, opt, [(x[i], y[i])], LR * min(1.0, step / 50))
        r = score(m)
        out["arms"][how] = r
        print(f"{how:12s} first reply {r['first']:.1%}   second reply {r['second']:.1%}   "
              f"(second after a correct first: {r['second_given_right_first']:.1%})  ({time.time() - t0:.0f}s)")

    # Part 2: the repository's chat split, held-out questions that training also has
    norm = lambda q: " ".join(q.lower().split())
    rows = lambda f: [json.loads(l) for l in open(ROOT / "data" / f, encoding="utf-8")]
    tr, he = rows("chat_train.jsonl"), rows("chat_heldout.jsonl")
    tq = {norm(r["question"]) for r in tr}
    pairs = {(norm(r["question"]), norm(r["answer"])) for r in tr}
    leak = sum(norm(r["question"]) in tq for r in he)
    exact = sum((norm(r["question"]), norm(r["answer"])) in pairs for r in he)
    out["leak"] = {"train": len(tr), "held": len(he), "question_in_train": leak, "pair_in_train": exact}
    print(f"\nmake_chat.py's split: {leak} of {len(he)} held-out questions are also in training "
          f"({exact} with the same answer)")
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
