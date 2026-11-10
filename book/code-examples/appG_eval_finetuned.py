"""
Book example, not repository code.

Appendix G: the statistics of evaluating a fine-tuned model, where the measure
is passes out of n rather than a loss. Everything here runs on a CPU in
seconds; the worked numbers are made up for teaching and say so.

    python book/code-examples/appG_eval_finetuned.py

1. Wilson intervals for pass rates: G.2's 40/50 and 400/500, G.9's 45/60 and
   39/60, and Chapter 33's measured 5/20 and 2/20.
2. The gap between two models on the same 60 prompts: a paired bootstrap, and
   eval_context.py's paired() on the same per-prompt differences.
3. A judge with a taste for length: 600 simulated pairs, the raw win rate
   against the length-controlled one, fitted by a four-line Newton solver.
4. A split by hash of the item id: the sealed share over 1,000 ids, the same
   on every run and every machine.
5. Export fidelity as KL and top-token agreement: by hand on a three-way
   distribution, and tiny() against a copy whose weights were rounded through
   bfloat16, the way Chapter 39's export stores them.
"""

from __future__ import annotations

import copy
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.eval.eval_context import paired  # noqa: E402  the repository's paired error, unchanged
from anulm.model import AnuLM  # noqa: E402
from tests.test_model import tiny  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for k passes out of n: well behaved near 0% and 100%,
    unlike p +/- z*sqrt(p(1-p)/n)."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def paired_bootstrap(diffs, reps: int = 10_000, seed: int = 0):
    """Mean of per-item differences and its 95% interval, by resampling items."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, float)
    means = d[rng.integers(0, len(d), size=(reps, len(d)))].mean(1)
    return d.mean(), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def fit_logistic(x, y, iters: int = 25):
    """win ~ sigmoid(a + b x) by Newton's method. Returns (a, b)."""
    X = np.stack([np.ones_like(x), x], 1)
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (y - p)
        hess = (X * (p * (1 - p))[:, None]).T @ X
        w = w + np.linalg.solve(hess, grad)
    return float(w[0]), float(w[1])


def split_of(item_id: str, sealed_share: int = 50) -> str:
    """'sealed' or 'dev', from a hash of the id: the same answer on every machine and run."""
    h = int(hashlib.sha256(item_id.encode()).hexdigest(), 16)
    return "sealed" if h % 100 < sealed_share else "dev"


def kl_rows(p_logits: torch.Tensor, q_logits: torch.Tensor):
    """KL(p || q) at every position (nats) and whether the top token agrees."""
    lp, lq = F.log_softmax(p_logits.float(), -1), F.log_softmax(q_logits.float(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    return kl, lp.argmax(-1) == lq.argmax(-1)


def main() -> None:
    sig = lambda t: 1 / (1 + math.exp(-t))

    # 1. intervals for pass rates
    print("1. Wilson 95% intervals")
    for label, k, n in [("G.2", 40, 50), ("G.2", 400, 500), ("G.9 fine-tune", 45, 60), ("G.9 base", 39, 60),
                        ("Ch 33 dense reader", 5, 20), ("Ch 33 hybrid reader", 2, 20)]:
        lo, hi = wilson(k, n)
        print(f"   {label:<20} {k:>3}/{n:<3} = {100 * k / n:5.1f}%   [{100 * lo:5.1f}%, {100 * hi:5.1f}%]")

    # 2. two models on the same prompts
    both_pass, both_fail, cand_only, base_only = 40, 8, 9, 3
    diffs = [1] * cand_only + [-1] * base_only + [0] * (both_pass + both_fail)
    n = len(diffs)
    cand = (both_pass + cand_only) / n
    base = (both_pass + base_only) / n
    mean, lo, hi = paired_bootstrap(diffs)
    a = [1.0] * both_pass + [0.0] * both_fail + [0.0] * cand_only + [1.0] * base_only     # baseline, per prompt
    b = [1.0] * both_pass + [0.0] * both_fail + [1.0] * cand_only + [0.0] * base_only     # candidate, same order
    pm, pse = paired(a, b)
    print(f"\n2. {n} prompts: candidate {100 * cand:.1f}% {tuple(round(100 * v, 1) for v in wilson(both_pass + cand_only, n))}, "
          f"baseline {100 * base:.1f}% {tuple(round(100 * v, 1) for v in wilson(both_pass + base_only, n))}")
    print(f"   paired gap {100 * mean:+.1f} points, bootstrap 95% [{100 * lo:+.1f}, {100 * hi:+.1f}] "
          f"(10,000 resamples, seed 0)")
    print(f"   eval_context.paired(): mean {pm:+.4f}, standard error {pse:.4f}; "
          f"+/- 1.96 se = [{100 * (pm - 1.96 * pse):+.1f}, {100 * (pm + 1.96 * pse):+.1f}] points")

    # 3. a judge that likes long answers
    rng = np.random.default_rng(0)
    n_pairs, a_true, b_true = 600, math.log(0.7 / 0.3), 1.5
    x = rng.normal(-0.8, 0.6, n_pairs)                       # ln(candidate length / baseline length): usually shorter
    y = (rng.random(n_pairs) < 1 / (1 + np.exp(-(a_true + b_true * x)))).astype(float)
    a_hat, b_hat = fit_logistic(x, y)
    boots = []
    for _ in range(300):
        i = rng.integers(0, n_pairs, n_pairs)
        boots.append(sig(fit_logistic(x[i], y[i])[0]))
    print(f"\n3. {n_pairs} simulated pairs, true equal-length win rate {100 * sig(a_true):.0f}%, "
          f"true length taste b = {b_true}; candidate shorter on {100 * (x < 0).mean():.0f}% of pairs")
    print(f"   raw win rate {100 * y.mean():.1f}%; length-controlled sigmoid(a) = {100 * sig(a_hat):.1f}% "
          f"[{100 * np.percentile(boots, 2.5):.1f}%, {100 * np.percentile(boots, 97.5):.1f}%]; fitted b = {b_hat:.2f}")

    # 4. a split that never moves
    ids = [f"item-{i:04d}" for i in range(1000)]
    sealed = sum(split_of(i) == "sealed" for i in ids)
    again = sum(split_of(i) == "sealed" for i in ids)
    print(f"\n4. split_of over {len(ids):,} ids: sealed {sealed} ({100 * sealed / len(ids):.1f}%), "
          f"identical on a second pass: {sealed == again}; item-0000 is {split_of('item-0000')!r}")

    # 5. export fidelity
    p = torch.tensor([0.70, 0.20, 0.10])
    q = torch.tensor([0.66, 0.23, 0.11])
    print(f"\n5. KL by hand: sum p ln(p/q) = {(p * (p / q).log()).sum().item():.4f} nats; "
          f"top token unchanged: {bool(p.argmax() == q.argmax())}")
    torch.manual_seed(0)
    src = AnuLM(tiny()).eval()
    rounded = copy.deepcopy(src)
    with torch.no_grad():
        for prm in rounded.parameters():
            prm.copy_(prm.to(torch.bfloat16).to(torch.float32))     # stored in bf16, computed in fp32 (Ch 39)
    ids = torch.randint(0, 64, (8, 33))
    x, tgt = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():
        ls, _ = src(x, tgt)                                        # targets: logits at every position
        lr, _ = rounded(x, tgt)
    kl, agree = kl_rows(ls, lr)
    print(f"   tiny() against its bfloat16-rounded copy over {kl.numel()} positions: mean KL {kl.mean():.2e} nats, "
          f"99th percentile {kl.quantile(0.99):.2e}, top token agrees at {100 * agree.float().mean():.1f}%")


if __name__ == "__main__":
    main()
