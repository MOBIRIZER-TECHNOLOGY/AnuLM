"""
Chapter 17 lab: teach a small model with a teacher, two kinds of teacher.

    python lab.py                      # your course/build/further.py
    python lab.py --impl solution      # the reference solution

Needs ../ch08-pretrain/base.pt (the model teacher). CPU, about seven minutes.

A student a ninth of Chapter 8's size trains from scratch on Shakespeare,
1,000 steps, four ways:
  alone              plain next-byte loss
  bigram teacher     plus the KL to the corpus's own bigram statistics, weight
                     2 fading to 0 by half-way (bigram_teacher.py's best)
  bigram as a prior  the same statistics ADDED to the logits, for good
  model teacher      90% of the loss from Chapter 8's model's distribution
                     (temperature 1), 10% from the bytes themselves
twice: on all of the training text, and on its first 30,000 bytes only.
Validation loss every 100 steps. Writes results.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


lab1 = load(COURSE / "ch01-smallest-model" / "lab.py")
STEPS, BATCH, CTX, LR, SMALL = 1000, 16, 128, 2e-3, 30_000
STUDENT = dict(vocab_size=259, block_size=256, n_layer=2, hidden_size=64, n_head=4, head_dim=16, n_kv_head=2,
               rope_theta=10000.0, intermediate_size=128, num_experts=4, num_experts_per_tok=2,
               num_shared_experts=1, moe_intermediate_size=32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    a = p.parse_args()
    sol = a.impl == "solution"
    Bm = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S = load(COURSE / "build" / ("solutions/ch17.py" if sol else "further.py"))
    base = torch.load(COURSE / "ch08-pretrain" / "base.pt", weights_only=False)
    train_data, val = lab1.load_bytes(str(COURSE.parent / "data" / "input.txt"))
    teacher = Bm.MoELM(**base["sizes"])
    teacher.load_state_dict(base["model"])
    teacher.eval()
    try:
        bigram = S.build_bigram_teacher(train_data, 259, k=32)
    except NotImplementedError as e:
        sys.exit(f"course/build/further.py is not finished yet ({e}). Run course/checks/ch17.py, or use --impl solution.")
    log_uni, idx, vals = bigram
    prior = lambda x: log_uni.expand(*x.shape, -1).scatter_add(-1, idx[x], vals[x])   # the bigram's logits

    @torch.no_grad()
    def val_loss(m, add_prior):
        m.eval()
        g = torch.Generator().manual_seed(0)
        total = 0.0
        for _ in range(20):
            x, y = lab1.batch(val, CTX, 32, g)
            logits = m(x)[0] + (F.log_softmax(prior(x), -1) if add_prior else 0)
            total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        m.train()
        return total / 20

    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        tl = sum(F.cross_entropy(teacher(x)[0].reshape(-1, 259), y.reshape(-1)).item()
                 for x, y in (lab1.batch(val, CTX, 32, g) for _ in range(20))) / 20
    n_student = sum(p.numel() for p in Bm.MoELM(**STUDENT).parameters())
    n_teacher = sum(p.numel() for p in teacher.parameters())
    print(f"teacher: {n_teacher:,} parameters, validation loss {tl:.4f}; student: {n_student:,} parameters")
    out = {"steps": STEPS, "teacher_params": n_teacher, "student_params": n_student, "teacher_loss": tl,
           "small_bytes": SMALL, "arms": {}, "small": {}}
    runs = [(n, train_data, "arms") for n in ("alone", "bigram teacher", "bigram as a prior", "model teacher")]
    runs += [(n, train_data[:SMALL], "small") for n in ("alone", "bigram teacher", "model teacher")]
    for name, data, where in runs:
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        m = Bm.MoELM(**STUDENT).train()
        opt = Bm.make_optimizer(m, LR)
        add_prior = name == "bigram as a prior"
        curve, t0 = [], time.time()
        for step in range(1, STEPS + 1):
            for grp in opt.param_groups:
                grp["lr"] = LR * min(1.0, step / 100) * (0.1 + 0.9 * (1 - step / STEPS))
            x, y = lab1.batch(data, CTX, BATCH, g)
            logits, _ = m(x)
            if add_prior:
                logits = logits + F.log_softmax(prior(x), -1)
            if name == "model teacher":
                with torch.no_grad():
                    t_logits = teacher(x)[0]
                loss = S.kd_loss(logits, t_logits, y, 1.0, 0.9)
            else:
                loss = F.cross_entropy(logits.reshape(-1, 259), y.reshape(-1))
                if name == "bigram teacher":
                    w = S.teacher_weight(step, STEPS, 2.0, 0.5)
                    if w > 0:
                        loss = loss + w * S.teacher_kl(logits, x, bigram)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            m.update_expert_biases()
            if step % 100 == 0:
                curve.append((step, val_loss(m, add_prior)))
        out[where][name] = {"curve": curve, "final": curve[-1][1]}
        print(f"{where:5s} {name:18s} validation loss " + " ".join(f"{v:.3f}" for _, v in curve) + f"  ({time.time() - t0:.0f}s)")
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
