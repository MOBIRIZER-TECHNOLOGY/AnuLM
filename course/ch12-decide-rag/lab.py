"""
Chapter 12 lab: look facts up with BM25, and make calibrated decisions with Decide.

    python lab.py                      # your course/build/post.py
    python lab.py --impl solution      # the reference solution

Needs ../ch08-pretrain/base.pt. CPU, about ten minutes.

Part 1, lookup. 240 Hindi passages with 1,190 questions (XQuAD) and 300
English passages with 1,933 questions (SQuAD), both CC BY-SA 4.0, in data/.
For every question: is the passage it was written from the first result (R@1),
or in the first five (R@5)? Four retrievers: BM25; BM25 with Python's \\w+ as
the word pattern; BM25 without length normalisation (b = 0); and plain word
overlap, with no idf.

Part 2, decisions. BANKING77 (CC BY 4.0, data/banking77 in the repository):
customer messages and 77 intents. Each example offers four options, the right
one and three others at random, and Chapter 8's model with a Decide head
scores all four in one pass. Messages are cut to 80 characters so that every
example fits the model's 256 bytes; the few that still don't are left out.
(decide.py offers all 77 at once; its tokenizer makes that fit.) Then: accuracy, calibration before and after
fitting a temperature, and how much it can answer at a given confidence.
Writes results.json.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

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


K_OPTIONS, BLOCK, STEPS, BATCH, LR, CUT = 4, 256, 3000, 16, 1e-3, 80
BOS = 256


def read_set(name):
    lines = (HERE / "data" / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0])["docs"], [json.loads(l) for l in lines[1:]]


def recall(index, docs, questions):
    r1 = r5 = 0
    for q in questions:
        top = index.search(q["q"], 5)
        r1 += top[:1] == [q["doc"]]
        r5 += q["doc"] in top
    return r1 / len(questions), r5 / len(questions)


def lookup(S):
    out = {}
    W = re.compile(r"\w+")
    for name in ("xquad_hi", "squad_en"):
        docs, qs = read_set(name)
        try:
            runs = {"BM25": S.BM25(docs),
                    "BM25, \\w+ words": S.BM25(docs, tokenize=lambda t: W.findall(t.lower())),
                    "BM25, b = 0": S.BM25(docs, b=0.0)}
        except NotImplementedError as e:
            sys.exit(f"course/build/post.py is not finished yet ({e}). Run course/checks/ch12.py, or use --impl solution.")
        overlap = S.BM25(docs)
        overlap.scores = lambda q, ix=overlap: torch.tensor(
            [float(len(set(S.words(q)) & set(c))) for c in ix.tf], dtype=torch.float64)
        runs["word overlap, no idf"] = overlap
        out[name] = {"passages": len(docs), "questions": len(qs)}
        for label, index in runs.items():
            r1, r5 = recall(index, docs, qs)
            out[name][label] = {"r1": r1, "r5": r5}
            print(f"{name}  {label:22s} R@1 {r1:6.1%}   R@5 {r5:6.1%}")
        best = runs["BM25"]
        q = qs[0]
        out[name]["example"] = {"question": q["q"], "answer": q["answer"],
                                "top": docs[best.search(q["q"], 1)[0]][:300]}
    return out


def decisions(Bm, S, STEPS=STEPS, LR=LR):
    rows = lambda f: [(r["text"], r["category"]) for r in csv.DictReader(open(COURSE.parent / "data" / "banking77" / f,
                                                                              encoding="utf-8"))]
    train, test = rows("train.csv"), rows("test.csv")
    labels = sorted({c for _, c in train})
    rnd = random.Random(0)
    rnd.shuffle(train)
    val, train = train[:1000], train[1000:]
    enc = lambda s: list(s.encode("utf-8"))

    def example(text, gold, rng):
        opts = [gold] + rng.sample([l for l in labels if l != gold], K_OPTIONS - 1)
        rng.shuffle(opts)
        ids, pos = S.encode_options(enc, BOS, text[:CUT], opts)
        return ids, pos, opts.index(gold)

    def batch(exs):
        L = max(len(e[0]) for e in exs)
        x = torch.full((len(exs), L), 258, dtype=torch.long)              # PAD; never read before an option
        for b, e in enumerate(exs):
            x[b, :len(e[0])] = torch.tensor(e[0])
        return x, torch.tensor([e[1] for e in exs]), torch.tensor([e[2] for e in exs])

    fixed = random.Random(1)
    fits = lambda exs: [e for e in exs if len(e[0]) <= BLOCK]
    val_ex = fits([example(t, c, fixed) for t, c in val])
    test_ex = fits([example(t, c, fixed) for t, c in test])
    print(f"\nBANKING77: {len(labels)} intents; {len(train):,} training messages, {len(val_ex):,} for fitting the "
          f"temperature, {len(test_ex):,} of {len(test):,} test messages fit; {K_OPTIONS} options each")
    print("for example:", bytes(t for t in test_ex[0][0] if t < 256).decode("utf-8", "replace"))

    base = torch.load(COURSE / "ch08-pretrain" / "base.pt", weights_only=False)
    torch.manual_seed(0)
    m = Bm.MoELM(**base["sizes"])
    m.load_state_dict(base["model"])
    d = S.Decider(m, base["sizes"]["hidden_size"])
    opt = Bm.make_optimizer(d, LR)

    @torch.no_grad()
    def logits_of(exs):
        d.eval()
        out = []
        for i in range(0, len(exs), 64):
            x, pos, _ = batch(exs[i:i + 64])
            out.append(d(x, pos))
        d.train()
        return torch.cat(out), torch.tensor([e[2] for e in exs])

    g = random.Random(2)
    curve, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        x, pos, y = batch(fits([example(t, c, g) for t, c in g.sample(train, BATCH)]))
        lr = LR * min(1.0, step / 50) * (0.1 + 0.9 * (1 - step / STEPS))
        for grp in opt.param_groups:
            grp["lr"] = lr
        loss = F.cross_entropy(d(x, pos), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
        opt.step()
        m.update_expert_biases()
        if step % 100 == 0:
            lv, yv = logits_of(val_ex)
            acc = (lv.argmax(1) == yv).float().mean().item()
            curve.append((step, acc))
            print(f"   step {step:4d}  loss {loss.item():.3f}  validation accuracy {acc:.1%}  ({time.time() - t0:.0f}s)")
    lv, yv = logits_of(val_ex)
    lt, yt = logits_of(test_ex)
    T = S.fit_temperature(lv, yv)
    before, after = S.calibration(F.softmax(lt, -1), yt), S.calibration(F.softmax(lt / T, -1), yt)

    def reliability(p, y, bins=10):
        conf, pred = p.max(1)
        ok = (pred == y).float()
        out = []
        for b in range(bins):
            sel = (conf > b / bins) & (conf <= (b + 1) / bins)
            if sel.any():
                out.append((b, int(sel.sum()), conf[sel].mean().item(), ok[sel].mean().item()))
        return out

    res = {"options": K_OPTIONS, "test_all": len(test), "intents": len(labels), "train": len(train), "val": len(val_ex),
           "test": len(test_ex), "steps": STEPS, "curve": curve, "temperature": T, "before": before,
           "after": after, "reliability_before": reliability(F.softmax(lt, -1), yt),
           "reliability_after": reliability(F.softmax(lt / T, -1), yt), "chance": 1 / K_OPTIONS}
    print(f"\ntest accuracy {before['accuracy']:.1%} (chance {1 / K_OPTIONS:.1%}); fitted temperature {T}")
    for name, c in (("before", before), ("after", after)):
        cov = c["answer_if_conf>=0.9"]
        print(f"  {name:6s} ECE {c['ece']:.3f}  Brier {c['brier']:.3f}  mean confidence {c['mean_confidence']:.1%}  "
              f"answers {cov['coverage']:.1%} at >= 0.9, {cov['accuracy']:.1%} of those right")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--skip-lookup", action="store_true")
    p.add_argument("--out", default="results.json")
    a = p.parse_args()
    sol = a.impl == "solution"
    Bm = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S = load(COURSE / "build" / ("solutions/ch12.py" if sol else "post.py"))
    out = {"lookup": None if a.skip_lookup else lookup(S), "decide": decisions(Bm, S, a.steps, a.lr)}
    (HERE / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
