"""
Book example, not repository code.

Chapter 11: scaled dot-product attention written out in plain PyTorch, then
checked against PyTorch's fused function and against model.py's own `_sdpa`.

    python book/code-examples/ch11_attention_from_scratch.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def attention(q, k, v, mask=None, scale=None):
    """q: (..., S, d); k, v: (..., T, d); mask: (S, T) bool, True = may attend."""
    d = q.shape[-1]
    scale = 1 / math.sqrt(d) if scale is None else scale
    scores = (q @ k.transpose(-2, -1)) * scale           # (..., S, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)                     # each row sums to 1
    return weights @ v, weights                          # (..., S, d), (..., S, T)


def causal(S: int, T: int | None = None) -> torch.Tensor:
    """(S, T) bool: query i may see key j when j <= i + (T - S)."""
    T = S if T is None else T
    return torch.ones(S, T, dtype=torch.bool).tril(T - S)


def main() -> None:
    # 1. The toy of 11.3
    X = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])
    W_q = torch.tensor([[1., 0.], [0., 2.]])
    W_k = torch.eye(2)
    W_v = torch.tensor([[0., 1.], [1., 0.]])
    out, w = attention(X @ W_q, X @ W_k, X @ W_v, causal(3))
    print("toy weights, row bit:", [round(x, 3) for x in w[2].tolist()])
    print("toy output,  row bit:", [round(x, 3) for x in out[2].tolist()])

    # 2. The released model's shape: 16 heads of width 64, 12 tokens, 2 sequences
    torch.manual_seed(0)
    B, H, S, d = 2, 16, 12, 64
    q, k, v = (torch.randn(B, H, S, d) for _ in range(3))
    mine, w = attention(q, k, v, causal(S))
    fused = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    repo = M._sdpa(q, k, v, None, S, d ** -0.5, 0.0)
    print(f"shapes: q {tuple(q.shape)} -> weights {tuple(w.shape)} -> out {tuple(mine.shape)}")
    print(f"rows sum to 1: largest error {(w.sum(-1) - 1).abs().max():.1e}")
    print(f"future weights: largest {w.masked_select(~causal(S)).abs().max():.1e}")
    print(f"mine vs PyTorch's fused function: largest difference {(mine - fused).abs().max():.1e}")
    print(f"mine vs model._sdpa:              largest difference {(mine - repo).abs().max():.1e}")

    # 3. A cached step: 3 new queries after 9 cached keys (T > S), the case of Chapters 12 and 27
    q3 = q[..., -3:, :]
    m = causal(3, S)
    mine3, _ = attention(q3, k, v, m)
    fused3 = F.scaled_dot_product_attention(q3, k, v, attn_mask=m)
    print(f"3 queries, 12 keys, mask {tuple(m.shape)}: largest difference {(mine3 - fused3).abs().max():.1e}")
    print("first new query may see keys", int(m[0].sum()), "of", S)


if __name__ == "__main__":
    main()
