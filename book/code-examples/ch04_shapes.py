"""
Book example, not repository code.

Chapter 4: the shape bookkeeping of GQAttention.forward done by hand on tiny(),
and checked against the module's own output at each step:

- nn.Linear as x @ W.T, and which rows of W make which head;
- view, split and transpose, and the strides they leave behind;
- why view needs contiguous() after a transpose;
- broadcasting in apply_rope and RMSNorm;
- the batched matrix multiply q @ k^T, and repeat_interleave as a broadcast;
- what a matrix multiply costs.

    python book/code-examples/ch04_shapes.py     # under a second on a CPU

It imports model.py and test_model.py unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "book" / "code-examples"))
from anulm import model as M                       # noqa: E402  unchanged
from tests.test_model import tiny             # noqa: E402
from trace_forward import released      # noqa: E402  Chapter 20's released configuration

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def linear_is_x_wT(attn: M.GQAttention, x: torch.Tensor) -> torch.Tensor:
    W = attn.query_key_value.weight                     # (out, in) = ((H + 2Hkv) * Dh, D)
    mine = x @ W.T                                      # (B, S, D) @ (D, out) -> (B, S, out)
    theirs = attn.query_key_value(x)
    print(f"W {tuple(W.shape)}   x {tuple(x.shape)}   x @ W.T {tuple(mine.shape)}")
    assert torch.allclose(mine, theirs, atol=1e-6)
    return theirs


def heads_by_hand(attn: M.GQAttention, x: torch.Tensor, qkv: torch.Tensor):
    B, S, _ = x.shape
    H, Hkv, Dh = attn.n_head, attn.n_kv_head, attn.head_dim
    v4 = qkv.view(B, S, H + 2 * Hkv, Dh)                # cut the last axis into 8 slots of Dh
    q, k, v = v4.split([H, Hkv, Hkv], dim=-2)           # 4 query, 2 key, 2 value slots
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    W = attn.query_key_value.weight
    for h in range(H):                                  # query head h is rows h*Dh .. (h+1)*Dh of W
        assert torch.allclose(q[:, h], x @ W[h * Dh:(h + 1) * Dh].T, atol=1e-6)
    for j in range(Hkv):                                # key head j comes after all H query heads
        assert torch.allclose(k[:, j], x @ W[(H + j) * Dh:(H + j + 1) * Dh].T, atol=1e-6)
    print(f"view {tuple(v4.shape)}  q {tuple(q.shape)}  k {tuple(k.shape)}  v {tuple(v.shape)}")
    print(f"q strides {q.stride()}  contiguous: {q.is_contiguous()}")
    return q, k, v


def view_needs_contiguous(y: torch.Tensor) -> None:
    B, H, S, Dh = y.shape
    t = y.transpose(1, 2)                               # (B, S, H, Dh), same memory, new strides
    try:
        t.view(B, S, -1)
        print("view after transpose: worked")
    except RuntimeError as e:
        print(f"view after transpose: RuntimeError ({str(e).split('.')[0]})")
    flat = t.contiguous().view(B, S, -1)                # copy into the new order, then view
    print(f"contiguous().view -> {tuple(flat.shape)}; head 1 sits in columns {Dh}..{2 * Dh - 1}")
    assert torch.equal(flat[..., Dh:2 * Dh], y[:, 1])


def broadcasting(q: torch.Tensor, cos: torch.Tensor) -> None:
    c = cos[None, None, :, :]                            # (S, Dh) -> (1, 1, S, Dh)
    out = q * c                                          # (B, H, S, Dh) * (1, 1, S, Dh)
    print(f"cos {tuple(cos.shape)} -> {tuple(c.shape)}; q * cos {tuple(out.shape)}")
    assert torch.equal(out[1, 3], q[1, 3] * cos)        # every (batch, head) pair sees the same table
    x = torch.randn(2, 5, 32)
    ms = x.pow(2).mean(-1, keepdim=True)                # (B, S, 1): one number per token
    print(f"RMSNorm: mean(-1, keepdim=True) {tuple(ms.shape)}; x * rsqrt(...) {tuple((x * ms.rsqrt()).shape)}")
    assert torch.allclose(M.RMSNorm(32)(x), x * torch.rsqrt(ms + 1e-6), atol=1e-6)


def scores_and_groups(attn: M.GQAttention, q: torch.Tensor, k: torch.Tensor) -> None:
    scores = q @ k.repeat_interleave(attn.n_rep, dim=1).transpose(-2, -1)   # (B, H, S, Dh) @ (B, H, Dh, T)
    print(f"q @ k^T: {tuple(q.shape)} @ {tuple(k.repeat_interleave(attn.n_rep, dim=1).transpose(-2, -1).shape)}"
          f" -> {tuple(scores.shape)}")
    kk = k.repeat_interleave(attn.n_rep, dim=1)
    print("query head -> key/value head:", [h // attn.n_rep for h in range(attn.n_head)])
    for h in range(attn.n_head):
        assert torch.equal(kk[:, h], k[:, h // attn.n_rep])


def flops() -> None:
    cfg = released()
    out = (cfg.n_head + 2 * cfg.n_kv_head) * cfg.head_dim
    print(f"released query_key_value: (1, {cfg.hidden_size}) @ ({cfg.hidden_size}, {out}) per token: "
          f"{cfg.hidden_size * out:,} multiply-adds = {2 * cfg.hidden_size * out:,} FLOPs")


if __name__ == "__main__":
    torch.manual_seed(0)
    model = M.AnuLM(tiny()).eval()
    attn = model.layers[0].attn
    x = torch.randn(2, 5, 32)                            # (B, S, D)
    with torch.no_grad():
        qkv = linear_is_x_wT(attn, x)
        q, k, v = heads_by_hand(attn, x, qkv)
        kk, vv = k.repeat_interleave(attn.n_rep, dim=1), v.repeat_interleave(attn.n_rep, dim=1)
        y = M._sdpa(q, kk, vv, None, 5, attn.scale, 0.0)          # attention's output, (B, H, S, Dh)
        print(f"_sdpa output {tuple(y.shape)} strides {y.stride()} contiguous: {y.is_contiguous()}")
        y2 = M._sdpa(q.contiguous(), kk, vv, None, 5, attn.scale, 0.0)          # the same, from a contiguous q
        print(f"_sdpa output from a contiguous q: strides {y2.stride()} contiguous: {y2.is_contiguous()}")
        view_needs_contiguous(y.contiguous())          # a copy laid out as (B, H, S, Dh)
        cos, sin = model.rotary(5, x.device, x.dtype)
        broadcasting(q, cos)
        scores_and_groups(attn, q, k)
    flops()
