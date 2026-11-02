"""
Book example, not repository code.

Chapter 13: many heads, shared keys.

    python book/code-examples/ch13_gqa.py

13.2  The fused projection: which rows of `query_key_value.weight` make each
      query, key and value head, checked against `GQAttention.forward`'s split.
13.3–13.4  QK-norm per head with one shared gain, and what the cache holds,
      in fp32 and under bf16 autocast.
13.4  Which key/value head each query head reads: `repeat_interleave` against
      a tiled copy and against SDPA's own `enable_gqa`.
13.5  MHA, GQA and MQA as settings of `n_kv_head`: projection width, attention
      parameters per layer and KV cache per token, for `tiny()` and the
      released model.

`model.py` is imported unchanged; every check runs its code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged
from trace_forward import released, tiny  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def fused_layout() -> None:
    print("13.2  One fused projection, cut into heads (tiny(): H=4, Hkv=2, Dh=8, D=32)")
    torch.manual_seed(0)
    cfg = tiny()
    attn = M.AnuLM(cfg).layers[0].attn
    H, Hkv, Dh = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    W = attn.query_key_value.weight                          # ((H + 2 Hkv) Dh, D) = (64, 32)
    x = torch.randn(2, 5, cfg.hidden_size)                   # (B, S, D)
    qkv = attn.query_key_value(x).view(2, 5, H + 2 * Hkv, Dh)
    q, k, v = qkv.split([H, Hkv, Hkv], dim=-2)               # as GQAttention.forward does
    rows = lambda i: W[i * Dh:(i + 1) * Dh]                  # the Dh rows of "head slot" i
    for h in range(H):
        assert torch.allclose(q[:, :, h], x @ rows(h).T, atol=1e-6)
    for j in range(Hkv):
        assert torch.allclose(k[:, :, j], x @ rows(H + j).T, atol=1e-6)
        assert torch.allclose(v[:, :, j], x @ rows(H + Hkv + j).T, atol=1e-6)
    print(f"  weight {tuple(W.shape)}; rows 0–{H * Dh - 1} queries, {H * Dh}–{(H + Hkv) * Dh - 1} keys, "
          f"{(H + Hkv) * Dh}–{(H + 2 * Hkv) * Dh - 1} values; q {tuple(q.shape)}, k {tuple(k.shape)}")


def grouping() -> None:
    print("\n13.4  Which key/value head each query head reads (H=4, Hkv=2)")
    k = torch.arange(2.).view(1, 2, 1, 1)                   # head j holds the number j
    print("  repeat_interleave:", k.repeat_interleave(2, dim=1).flatten().tolist())
    print("  repeat (tiled):   ", k.repeat(1, 2, 1, 1).flatten().tolist())
    torch.manual_seed(0)
    q = torch.randn(1, 4, 5, 8)
    k, v = torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8)
    native = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    inter = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), is_causal=True)
    tiled = F.scaled_dot_product_attention(q, k.repeat(1, 2, 1, 1), v.repeat(1, 2, 1, 1), is_causal=True)
    print(f"  enable_gqa vs repeat_interleave: {(native - inter).abs().max():.1e}; "
          f"vs tiled: {(native - tiled).abs().max():.2f}")


def kv_arithmetic() -> None:
    print("\n13.5  MHA, GQA and MQA as settings of n_kv_head")
    for name, base in (("tiny()", tiny()), ("released", released())):
        H, Dh, D, L = base.n_head, base.head_dim, base.hidden_size, base.n_layer
        for hkv in sorted({H, base.n_kv_head, 1}, reverse=True):
            attn = M.GQAttention(M.AnuLMConfig(**{**base.__dict__, "n_kv_head": hkv}),
                                 M.Rotary(base, Dh))
            n = sum(p.numel() for p in attn.parameters())
            per_tok = L * 2 * hkv * Dh * 2                 # k and v, every layer, bf16
            kind = "MHA" if hkv == H else "MQA" if hkv == 1 else "GQA"
            print(f"  {name:8s} {kind} Hkv={hkv:2d}: qkv width {attn.query_key_value.out_features:5,}; "
                  f"attention params per layer {n:10,}; KV cache {per_tok:7,} bytes/token in bf16")


def qk_norm_and_cache() -> None:
    print("\n13.3–13.4  QK-norm per head, and what the cache holds (tiny())")
    torch.manual_seed(0)
    model = M.AnuLM(tiny()).eval()
    attn = model.layers[0].attn
    q = torch.randn(2, 4, 5, 8)                              # (B, H, S, Dh)
    by_head = torch.stack([attn.query_layernorm(q[:, h]) for h in range(4)], 1)
    print(f"  gain shape {tuple(attn.query_layernorm.weight.shape)}; whole tensor vs one head at a time: "
          f"{(attn.query_layernorm(q) - by_head).abs().max():.1e}")
    cache = model.new_cache()
    idx = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        model(idx, cache=cache)
        x = model.layers[0].input_layernorm(model.embed_tokens(idx))
        k = attn.query_key_value(x).view(1, 9, 8, 8)[:, :, 4:6].transpose(1, 2)   # the Hkv key heads
        cos, sin = model.rotary(9, x.device, x.dtype)
        k = attn.key_layernorm(k)
        _, k = M.apply_rope(k, k, cos, sin)
    print(f"  cache[0]['k'] {tuple(cache[0]['k'].shape)} (B, Hkv, T, Dh); equals the keys after QK-norm "
          f"and RoPE: {torch.allclose(cache[0]['k'], k, atol=1e-6)}")
    cache = model.new_cache()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        model(idx, cache=cache)
    print(f"  under bf16 autocast the cache holds k in {str(cache[0]['k'].dtype)[6:]}, "
          f"v in {str(cache[0]['v'].dtype)[6:]}")


def exercises() -> None:
    print("\n13.8  Answers")
    cfg = released()
    print(f"  1. released qkv weight: ({(cfg.n_head + 2 * cfg.n_kv_head) * cfg.head_dim}, {cfg.hidden_size})")
    print(f"  2. query head 9 reads key/value head {9 // (cfg.n_head // cfg.n_kv_head)}")
    T = 1024
    copy = 2 * 1 * cfg.n_head * T * cfg.head_dim * 2          # k and v, (B=1, H, T, Dh), bf16
    served = 1 * cfg.n_head * T * cfg.head_dim * (4 + 2)      # fp32 k and bf16 v, as under autocast
    print(f"  3. repeat_interleave copies at T={T}, one layer: {copy:,} bytes "
          f"({copy // 2 ** 20} MiB) all in bf16, against {copy // 4:,} for the Hkv heads; "
          f"{served:,} bytes with fp32 keys and bf16 values, as under autocast")


if __name__ == "__main__":
    fused_layout()
    qk_norm_and_cache()
    grouping()
    kv_arithmetic()
    exercises()
