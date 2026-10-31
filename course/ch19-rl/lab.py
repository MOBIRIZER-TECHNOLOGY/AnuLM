"""
Chapter 19 lab: reinforcement learning on Chapter 9's sums model, with a checker as the reward.

    python lab.py                      # your course/build/further.py and post.py
    python lab.py --impl solution      # the reference solutions

Needs ../ch09-sft/sft.pt. CPU, about six minutes.

Each step: 8 training sums, 8 answers sampled for each at temperature 1, every
answer checked by a line of Python (reward 1 or 0). Two ways to learn from the
same samples, 200 steps each:
  GRPO                 each answer pushed up or down by how it did against its
                       own group, with a KL pull back to the starting model
  rejection sampling   fine-tune on the samples that were right (Chapter 9's
                       loss), ignore the rest
Every 50 steps: greedy and sampled accuracy on 200 held-out sums, and the share
of groups that had anything to teach (some answers right, some wrong).
Writes results.json.
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
STEPS, PROMPTS, GROUP, LR, BETA, EOS = 200, 8, 8, 2e-5, 0.04, 257


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    a = p.parse_args()
    sol = a.impl == "solution"
    Bm = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S11 = load(COURSE / "build" / ("solutions/ch11.py" if sol else "post.py"))
    S = load(COURSE / "build" / ("solutions/ch19.py" if sol else "further.py"))
    ck = torch.load(COURSE / "ch09-sft" / "sft.pt", weights_only=False)
    enc = lambda s: list(s.encode("utf-8"))
    rnd = random.Random(0)
    train, test = lab9.make_items(6000, rnd), lab9.make_items(200, rnd)                 # Chapter 9's split
    template = "Question: {q}\nAnswer:"

    def start():
        m = Bm.MoELM(**ck["sizes"])
        m.load_state_dict(ck["model"])
        return m.train()

    @torch.no_grad()
    def sample(m, questions, n, temperature):
        """n answers per question: lists of ids (ending at EOS when the model stopped)."""
        m.eval()
        prompts = torch.tensor([enc(template.format(q=q)) for q in questions]).repeat_interleave(n, 0)
        gen = m.generate(prompts, 5, temperature=temperature, eos_id=EOS)[:, prompts.shape[1]:].tolist()
        m.train()
        return [g[: g.index(EOS) + 1] if EOS in g else g for g in gen]

    text = lambda ids: bytes(t for t in ids if t < 256).decode("utf-8", "replace")

    def measure(m):
        qs, ans = [q for q, _ in test], [x for _, x in test]
        greedy = sum(text(o).strip() == x for o, x in zip(sample(m, qs, 1, 1e-5), ans)) / len(test)
        torch.manual_seed(1)
        outs = sample(m, qs, 4, 1.0)
        sampled = sum(text(o).strip() == ans[k // 4] for k, o in enumerate(outs)) / len(outs)
        return greedy, sampled

    first = measure(start())
    print(f"Chapter 9's model: greedy {first[0]:.1%}, sampled {first[1]:.1%} of {len(test)} held-out sums")
    out = {"steps": STEPS, "prompts": PROMPTS, "group": GROUP, "lr": LR, "beta": BETA, "start": first, "arms": {}}
    for name in ("GRPO", "rejection sampling"):
        torch.manual_seed(0)
        rng = random.Random(1)
        policy = start()
        reference = copy.deepcopy(policy).eval()
        opt = Bm.make_optimizer(policy, LR, weight_decay=0.0)
        curve, signal, t0 = [(0, first[0], first[1], None)], [], time.time()
        for step in range(1, STEPS + 1):
            batch = rng.sample(train, PROMPTS)
            outs = sample(policy, [q for q, _ in batch], GROUP, 1.0)
            prompts = [enc(template.format(q=q)) for q, _ in batch for _ in range(GROUP)]
            try:
                r = torch.tensor([S.reward(text(o), batch[k // GROUP][1]) for k, o in enumerate(outs)])
            except NotImplementedError as e:
                sys.exit(f"course/build/further.py is not finished yet ({e}). Run course/checks/ch19.py, "
                         "or use --impl solution.")
            groups = r.view(PROMPTS, GROUP)
            signal.append(((groups.std(1) > 0).float().mean()).item())
            lengths = torch.tensor([max(len(o), 1) for o in outs], dtype=torch.float)
            if name == "GRPO":
                adv = S.group_advantages(groups).reshape(-1)
                with torch.no_grad():
                    ref = S11.answer_logprobs(reference, prompts, outs)
                logp = S11.answer_logprobs(policy, prompts, outs)
                loss = S.grpo_loss(logp, ref, lengths, adv, BETA)
            else:
                keep = (r > 0).nonzero().flatten().tolist()
                if not keep:
                    continue
                logp = S11.answer_logprobs(policy, [prompts[k] for k in keep], [outs[k] for k in keep])
                loss = -(logp / lengths[keep]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            policy.update_expert_biases()
            if step % 50 == 0:
                g, s = measure(policy)
                curve.append((step, g, s, sum(signal[-50:]) / 50))
                print(f"   {name:18s} step {step:4d}  greedy {g:.1%}  sampled {s:.1%}  "
                      f"groups with something to teach {curve[-1][3]:.0%}  ({time.time() - t0:.0f}s)")
        out["arms"][name] = {"curve": curve}
        print(f"{name}: greedy {first[0]:.1%} -> {curve[-1][1]:.1%}, sampled {first[1]:.1%} -> {curve[-1][2]:.1%}")
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
