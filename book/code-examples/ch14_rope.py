"""
Book example, not repository code.

Chapter 14: word order, by rotation.

    python book/code-examples/ch14_rope.py

14.1  Without positions, a causal attention layer cannot tell the order of
      the tokens before a query: shuffle them and its output does not move.
14.3  The toy `The dog bit` rotated by hand at D_h = 2, where RoPE is one
      2-D rotation per position; the scores depend only on the gap.
14.4  The released model's 32 frequencies at θ = 1e6, against θ = 1e4.
14.5  The cached tables: what triggers a rebuild, and what `start` slices.
14.6  Split halves against consecutive pairs: the same rotation, up to a
      fixed permutation of the dimensions.

`model.py` is imported unchanged.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged
from trace_forward import released, tiny  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
cpu = torch.device("cpu")


def order_blind() -> None:
    print("14.1  Shuffle the tokens before the last one (tiny() attention layer, causal)")
    torch.manual_seed(0)
    cfg = tiny()
    attn = M.AnuLM(cfg).layers[2].attn.eval()
    x = torch.randn(1, 6, cfg.hidden_size)                     # six token vectors
    shuffled = x[:, [3, 0, 4, 1, 2, 5]]                         # same tokens, last one kept last
    cos, sin = M.Rotary(cfg, cfg.head_dim)(6, cpu, torch.float32)
    no_pos = (torch.ones_like(cos), torch.zeros_like(sin))      # cos 1, sin 0: no rotation
    with torch.no_grad():
        for name, (c, s) in (("no positions", no_pos), ("RoPE", (cos, sin))):
            d = (attn(x, c, s)[0, -1] - attn(shuffled, c, s)[0, -1]).abs().max()
            print(f"  {name:12s}: last output moves by {d:.1e}")


def rotate(v, angle):
    """Turn a 2-vector by `angle` radians: the matrix [[cos, −sin], [sin, cos]]."""
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def toy() -> None:
    print("\n14.3  The toy at D_h = 2: one frequency, θ_0 = 1 radian per position (illustrative)")
    Q = torch.tensor([[1., 0.], [0., 4.], [1., 2.]])           # Chapter 11's queries
    K = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])           # and keys
    rot = M.Rotary(M.AnuLMConfig(), dim=2)
    print(f"  inv_freq {rot.inv_freq.tolist()}")
    for start in (0, 5):
        cos, sin = rot(3, cpu, torch.float32, start=start)     # positions start .. start + 2
        q, k = M.apply_rope(Q[None, None], K[None, None], cos, sin)
        hand = torch.stack([rotate(Q[i], start + i) for i in range(3)])
        assert torch.allclose(q[0, 0], hand, atol=1e-6)
        s = q[0, 0] @ k[0, 0].T
        print(f"  positions {start}–{start + 2}: bit's query {[round(v, 3) for v in q[0, 0, 2].tolist()]}, "
              f"scores of bit {[round(v, 3) for v in s[2].tolist()]}")
    print(f"  |q_bit| before {Q[2].norm():.3f}, after {q[0, 0, 2].norm():.3f}")


def frequencies() -> None:
    print("\n14.4  The 32 frequencies of a 64-wide head")
    for theta in (1e4, 1e6):
        cfg = M.AnuLMConfig(rope_theta=theta)
        f = M.Rotary._build_inv_freq(cfg, 64)                   # radians per position, (32,)
        wave = 2 * math.pi / f                                  # positions per full turn
        print(f"  θ = {theta:.0e}: fastest {f[0]:.3f} rad/token ({wave[0]:.2f} tokens a turn), slowest "
              f"{f[-1]:.2e} ({wave[-1]:,.0f} tokens); pairs that turn fully within 1,024 tokens: "
              f"{int((wave <= 1024).sum())} of 32")


def caching() -> None:
    print("\n14.5  The cached tables")
    rot = M.Rotary(tiny(), dim=8)
    for n, dtype, start in ((8, torch.float32, 0), (4, torch.float32, 2), (8, torch.bfloat16, 0),
                            (20, torch.bfloat16, 0)):
        cos, _ = rot(n, cpu, dtype, start=start)
        print(f"  rot({n:2d}, {str(dtype)[6:]:8s}, start={start}): returns {tuple(cos.shape)} {str(cos.dtype)[6:]}; "
              f"table now {rot._cached_len} rows, {str(rot.cos_cached.dtype)[6:]}")


def conventions() -> None:
    print("\n14.6  Split halves (model.py) against consecutive pairs (the RoFormer paper)")
    torch.manual_seed(0)
    D, S = 64, 10
    cos, sin = M.Rotary(released(), D)(S, cpu, torch.float32)
    f = M.Rotary._build_inv_freq(released(), D)
    q, k = torch.randn(1, 1, S, D), torch.randn(1, 1, S, D)

    def consecutive(x):                                         # turn pair (2i, 2i+1) by pos·f[i]
        a = torch.arange(S)[:, None] * f[None, :]               # (S, D/2)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2], out[..., 1::2] = x1 * a.cos() - x2 * a.sin(), x2 * a.cos() + x1 * a.sin()
        return out

    perm = torch.cat([torch.arange(0, D, 2), torch.arange(1, D, 2)])   # 0,2,..,62, 1,3,..,63
    qh, kh = M.apply_rope(q[..., perm], k[..., perm], cos, sin)
    print(f"  apply_rope(x[perm]) vs consecutive(x)[perm]: {(qh - consecutive(q)[..., perm]).abs().max():.1e}")
    s1, s2 = qh @ kh.transpose(-2, -1), consecutive(q) @ consecutive(k).transpose(-2, -1)
    print(f"  scores, the two conventions: {(s1 - s2).abs().max():.1e}")


def exercises() -> None:
    print("\n14.9  Answers")
    f = M.Rotary._build_inv_freq(released(), 64)
    print(f"  1. pair 1 of the released head: {f[1]:.4f} rad/token, a turn every {2 * math.pi / f[1]:.1f} tokens")
    rot = M.Rotary(released(), 64)
    cos, _ = rot(1024, cpu, torch.float32)
    print(f"  2. released tables at 1,024 positions: {2 * cos.numel() * 4:,} bytes in fp32")


if __name__ == "__main__":
    order_blind()
    toy()
    frequencies()
    caching()
    conventions()
    exercises()
