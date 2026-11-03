"""
Book example, not repository code.

Chapter 5: softmax, temperature and cross-entropy, by hand, checked against
PyTorch and against model.py:

- softmax as exp-and-normalise, and why the maximum is subtracted first;
- what bf16 does to a 32,768-way log-softmax, which is why model.py casts to fp32;
- temperature as division, and temperature 0 as greedy decoding in generate;
- top-k as -inf before the softmax;
- cross-entropy as logsumexp(z) - z[target], and ignore_index=-1;
- its gradient, (softmax(z) - onehot(target)) / n.

    python book/code-examples/ch05_softmax.py     # a few seconds on a CPU
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.model import AnuLM                  # noqa: E402  unchanged
from tests.test_model import tiny              # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def softmax_naive(z: torch.Tensor) -> torch.Tensor:
    e = z.exp()
    return e / e.sum(-1, keepdim=True)


def softmax_stable(z: torch.Tensor) -> torch.Tensor:
    e = (z - z.max(-1, keepdim=True).values).exp()      # largest entry becomes e^0 = 1
    return e / e.sum(-1, keepdim=True)


def by_hand() -> None:
    z = torch.tensor([2.0, 1.0, 0.1])                   # illustrative logits
    p = softmax_stable(z)
    print(f"z {[round(v, 4) for v in z.tolist()]} -> p {[round(v, 4) for v in p.tolist()]}  sum {p.sum().item():.4f}")
    assert torch.allclose(p, F.softmax(z, -1))
    big = z + 1000                                      # same differences, shifted
    print(f"z + 1000: naive {softmax_naive(big).tolist()}  stable {[round(v, 4) for v in softmax_stable(big).tolist()]}")
    for T in (0.5, 1.0, 2.0):
        print(f"T = {T}: p {[round(v, 4) for v in F.softmax(z / T, -1).tolist()]}")


def bf16_cost(seed: int = 0) -> None:
    """Where a bf16 cross-entropy goes wrong: the logits, the log-softmax, or the mean."""
    torch.manual_seed(seed)
    z = torch.randn(4096, 32768) * 3.0                  # a batch of logit rows, V = 32,768
    t = torch.randint(0, 32768, (4096,))
    exact = F.cross_entropy(z.double(), t).item()
    rounded = F.cross_entropy(z.bfloat16().double(), t).item() - exact          # bf16 logits, fp64 maths
    lp16 = F.log_softmax(z.bfloat16(), -1)                                        # bf16 log-softmax
    lsm = -lp16.double().gather(1, t[:, None]).mean().item() - exact              # ... averaged in fp64
    whole = F.cross_entropy(z.bfloat16(), t).item() - exact                       # bf16 in, bf16 mean
    with torch.autocast("cpu", dtype=torch.bfloat16):
        auto = F.cross_entropy(z.bfloat16(), t)                                   # autocast upcasts it
    print(f"cross-entropy over 4,096 rows, fp64: {exact:.6f}; error from bf16 logits {rounded:+.1e}, "
          f"bf16 log-softmax {lsm:+.1e}, F.cross_entropy on bf16 {whole:+.4f}; "
          f"under autocast: {auto.dtype}, {auto.item() - exact:+.1e}")
    err = (lp16.double() - F.log_softmax(z.double(), -1)).abs()
    i = int(err.argmax())
    print(f"largest bf16 log-prob error {err.max().item():.2f}, at log-prob "
          f"{F.log_softmax(z.double(), -1).flatten()[i].item():.1f}")


def temperature_zero_is_greedy() -> None:
    torch.manual_seed(0)
    m = AnuLM(tiny()).eval()
    prompt = torch.randint(0, 64, (1, 4))
    torch.manual_seed(1)
    cold = m.generate(prompt.clone(), 10, temperature=0.0)            # T <= 0: argmax, no draw
    greedy = m.generate(prompt.clone(), 10, top_k=1)
    print(f"temperature 0 {cold[0, 4:].tolist()}\ntop_k 1       {greedy[0, 4:].tolist()}")
    assert torch.equal(cold, greedy)


def top_k_cut(z: torch.Tensor, k: int) -> torch.Tensor:
    z = z.clone()
    v, _ = torch.topk(z, k)
    z[z < v[..., [-1]]] = -float("inf")                 # generate's lines 1166-1167
    return F.softmax(z, -1)


def cross_entropy_by_hand(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    keep = t != -1
    lse = torch.logsumexp(z[keep], -1)                  # log of the softmax denominator, stable
    return (lse - z[keep].gather(1, t[keep][:, None])[:, 0]).mean()


def gradient_is_p_minus_y(seed: int = 0) -> None:
    torch.manual_seed(seed)
    z = torch.randn(6, 10, requires_grad=True)
    t = torch.tensor([3, 1, -1, 7, 0, -1])             # two rows not graded
    loss = F.cross_entropy(z, t, ignore_index=-1)
    loss.backward()
    keep = t != -1
    p = F.softmax(z.detach(), -1)
    y = F.one_hot(t.clamp(min=0), 10).float() * keep[:, None]
    mine = (p - y) * keep[:, None] / keep.sum()         # ungraded rows get no gradient
    print(f"cross-entropy {loss.item():.6f}  by hand {cross_entropy_by_hand(z.detach(), t).item():.6f}")
    print(f"gradient: largest |autograd - (p - y)/n| = {(z.grad - mine).abs().max().item():.1e}; "
          f"rows 2 and 5 all zero: {bool((z.grad[~keep] == 0).all())}")
    assert torch.allclose(z.grad, mine, atol=1e-7)


if __name__ == "__main__":
    by_hand()
    p = top_k_cut(torch.tensor([[2.0, 1.0, 0.1, -1.0]]), 2)
    print(f"top-2 of [2.0, 1.0, 0.1, -1.0]: p {[round(v, 4) for v in p[0].tolist()]}")
    bf16_cost()
    temperature_zero_is_greedy()
    gradient_is_p_minus_y()
