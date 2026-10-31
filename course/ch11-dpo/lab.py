"""
Chapter 11 lab: preference-tune Chapter 9's sums model on its own mistakes.

    python lab.py                      # your course/build/post.py
    python lab.py --impl solution      # the reference solution

Needs ../ch09-sft/sft.pt (run Chapter 9's lab first). CPU, about three minutes.

1. The Chapter 9 model answers every training question four times, sampling
   at temperature 1. Arithmetic can be checked, so each answer is right or
   wrong without anyone rating it. A question with at least one right and one
   wrong answer gives a preference pair: (right, wrong).
2. Four runs from the same model on the same pairs, same steps:
     dpo             DPO against a frozen copy of the starting model
     dpo, lr / 10    the same at a tenth of the learning rate
     dpo + sft       DPO plus the chosen answers' ordinary fine-tuning loss
     sft on chosen   ordinary fine-tuning on the right answers only
   Does the wrong half of each pair teach anything, and at what price?
3. Every 50 steps: greedy accuracy, sampled accuracy (4 samples per question at
   temperature 1), and the log-probability of held-out chosen and rejected
   answers. Writes results.json.
"""

from __future__ import annotations

import argparse
import copy
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


lab9 = load(COURSE / "ch09-sft" / "lab.py")
STEPS, BATCH, LR, BETA, SAMPLES = 400, 16, 5e-5, 0.1, 4
EOS = 257


def enc(s):
    return list(s.encode("utf-8"))


@torch.no_grad()
def sample_answers(m, questions, template, temperature, n):
    """n answers per question, as id lists ending at EOS (or after 5 ids)."""
    m.eval()
    out = []
    prompts = torch.tensor([enc(template.format(q=q)) for q in questions])    # all the same length
    for _ in range(n):
        gen = m.generate(prompts, 5, temperature=temperature, eos_id=EOS)[:, prompts.shape[1]:]
        out.append([row[: row.index(EOS) + 1] if EOS in row else row for row in gen.tolist()])
    m.train()
    return [[out[k][i] for k in range(n)] for i in range(len(questions))]


def text(ids):
    return bytes(t for t in ids if t < 256).decode("utf-8", "replace").strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    a = p.parse_args()
    sol = a.impl == "solution"
    Bm = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S9 = load(COURSE / "build" / ("solutions/ch09.py" if sol else "post.py"))
    S = load(COURSE / "build" / ("solutions/ch11.py" if sol else "post.py"))
    ck = torch.load(COURSE / "ch09-sft" / "sft.pt", weights_only=False)

    def start():
        m = Bm.MoELM(**ck["sizes"])
        m.load_state_dict(ck["model"])
        return m.train()

    rnd = random.Random(0)
    train_items, test_items = lab9.make_items(6000, rnd), lab9.make_items(200, rnd)   # Chapter 9's split
    T = S9.PROMPT_EN

    # 1. preference pairs from the model's own samples, labelled by checking the sum
    torch.manual_seed(0)
    t0 = time.time()
    samples = []
    for i in range(0, len(train_items), 500):
        samples += sample_answers(start(), [q for q, _ in train_items[i:i + 500]], T, 1.0, SAMPLES)
    pairs, n_right = [], 0
    for (q, ans), outs in zip(train_items, samples):
        good = [o for o in outs if text(o) == ans and o[-1:] == [EOS]]
        bad = [o for o in outs if text(o) != ans]
        n_right += len(good)
        if good and bad:
            pairs.append((enc(T.format(q=q)), good[0], bad[0]))
    rnd.shuffle(pairs)
    held, pairs = pairs[:100], pairs[100:]
    print(f"{len(train_items) * SAMPLES:,} sampled answers in {time.time() - t0:.0f}s, {n_right / (len(train_items) * SAMPLES):.1%} right; "
          f"{len(pairs) + len(held):,} questions gave a (right, wrong) pair; 100 held out")
    print("for example:", repr(text(held[0][0]) + " " + text(held[0][1])), "preferred to", repr(text(held[0][2])))

    @torch.no_grad()
    def measure(m):
        m.eval()
        greedy = sum(text(o[0]) == ans for (q, ans), o in
                     zip(test_items, sample_answers(m, [q for q, _ in test_items], T, 1e-5, 1))) / len(test_items)
        torch.manual_seed(1)
        outs = sample_answers(m, [q for q, _ in test_items], T, 1.0, SAMPLES)
        sampled = sum(text(o) == ans for (q, ans), os in zip(test_items, outs) for o in os) / (len(test_items) * SAMPLES)
        lp_c = S.answer_logprobs(m, [h[0] for h in held], [h[1] for h in held]).mean().item()
        lp_r = S.answer_logprobs(m, [h[0] for h in held], [h[2] for h in held]).mean().item()
        m.train()
        return {"greedy": greedy, "sampled": sampled, "logp_chosen": lp_c, "logp_rejected": lp_r}

    try:
        first = measure(start())
    except NotImplementedError as e:
        sys.exit(f"course/build/post.py is not finished yet ({e}). Run course/checks/ch11.py, or use --impl solution.")
    print(f"start: greedy {first['greedy']:.1%}, sampled {first['sampled']:.1%}, "
          f"held-out log p chosen {first['logp_chosen']:.2f}, rejected {first['logp_rejected']:.2f}")
    out = {"samples": len(train_items) * SAMPLES, "sampled_right": n_right / (len(train_items) * SAMPLES),
           "pairs": len(pairs), "held": len(held), "steps": STEPS, "lr": LR, "beta": BETA,
           "example": [text(held[0][0]), text(held[0][1]), text(held[0][2])], "start": first}

    # 2. two ways to use the same pairs
    for name, lr, sft_weight in (("dpo", LR, 0.0), ("dpo, lr / 10", LR / 10, 0.0), ("dpo + sft", LR, 1.0),
                                 ("sft on chosen", LR, None)):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        policy = start()
        reference = copy.deepcopy(policy).eval()
        for t in reference.parameters():
            t.requires_grad_(False)
        opt = Bm.make_optimizer(policy, lr, weight_decay=0.0)
        if sft_weight is None:
            x, y = S9.pack([(pp, c) for pp, c, _ in pairs], 128, EOS)
        curve, t0 = [(0, first)], time.time()
        for step in range(1, STEPS + 1):
            i = torch.randint(len(pairs), (BATCH,), generator=g).tolist()
            if sft_weight is not None:
                for grp in opt.param_groups:
                    grp["lr"] = lr * min(1.0, step / 20)
                S.dpo_step(policy, reference, opt, [pairs[k][0] for k in i], [pairs[k][1] for k in i],
                           [pairs[k][2] for k in i], BETA, sft_weight=sft_weight)
            else:
                j = torch.randint(len(x), (BATCH,), generator=g)
                Bm.train_step(policy, opt, [(x[j], y[j])], lr * min(1.0, step / 20))
            if step % 50 == 0:
                curve.append((step, measure(policy)))
                c = curve[-1][1]
                print(f"   {name:13s} step {step:4d}  greedy {c['greedy']:.1%}  sampled {c['sampled']:.1%}  "
                      f"log p chosen {c['logp_chosen']:.2f}  rejected {c['logp_rejected']:.2f}  ({time.time() - t0:.0f}s)")
        out[name] = {"lr": lr, "sft_weight": sft_weight, "curve": curve, "final": curve[-1][1]}
        f = curve[-1][1]
        print(f"{name}: greedy {first['greedy']:.1%} -> {f['greedy']:.1%}, sampled {first['sampled']:.1%} -> {f['sampled']:.1%}")
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
