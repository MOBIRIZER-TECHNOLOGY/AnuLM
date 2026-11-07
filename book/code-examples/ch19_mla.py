"""Book example, not repository code.

Chapter 19: multi-head latent attention, as `model.MLAttention` computes it.

    python book/code-examples/ch19_mla.py

1. One forward pass of `MLAttention` at the `tiny_mla` sizes, recomputed step
   by step with every shape printed, and checked against the module.
2. What the cache holds after a prompt and one more token, and a check that
   the cached step equals the uncached forward.
3. Why the rotary key is separate: the latent part of a score can be computed
   from the cached latent directly ("absorbing" kv_b_proj into the query),
   while a rotated key cannot. AnuLM does not do this; it re-expands the
   latent every step.
4. Cache arithmetic: numbers per token per layer for MLA and GQA, at the
   repository's sizes.
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
from trace_forward import released  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def shape(t):
    return "(" + ", ".join(str(d) for d in t.shape) + ")"


def build():
    torch.manual_seed(0)
    cfg = M.AnuLMConfig.tiny_mla()
    rot = M.Rotary(cfg, cfg.qk_rope_head_dim)          # AnuLM.__init__ sizes the tables to the rope slice
    attn = M.MLAttention(cfg, rot).eval()
    return cfg, rot, attn


def by_steps():
    cfg, rot, a = build()
    B, S = 2, 5
    x = torch.randn(B, S, cfg.hidden_size)
    cos, sin = rot(S, x.device, x.dtype)
    with torch.no_grad():
        q = a.q_proj(x).view(B, S, a.n_head, a.q_head_dim).transpose(1, 2)
        q = a.q_layernorm(q)
        q_nope, q_pe = q.split([a.qk_nope, a.qk_rope], dim=-1)
        compressed = a.kv_a_proj_with_mqa(x)
        c, k_pe = compressed.split([a.kv_lora_rank, a.qk_rope], dim=-1)
        k_pe = k_pe.view(B, S, 1, a.qk_rope).transpose(1, 2)
        q_pe, k_pe = M.apply_rope(q_pe, k_pe, cos, sin)
        c = a.kv_a_layernorm(c)
        kv = a.kv_b_proj(c).view(B, S, a.n_head, a.qk_nope + a.v_head_dim).transpose(1, 2)
        k_nope, v = kv.split([a.qk_nope, a.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, a.n_head, -1, -1)], dim=-1)
        y = F.scaled_dot_product_attention(torch.cat([q_nope, q_pe], -1), k, v, is_causal=True, scale=a.scale)
        out = a.o_proj(y.transpose(1, 2).reshape(B, S, -1))
        ref = a(x, cos, sin)
    print(f"tiny_mla: D={cfg.hidden_size}, H={cfg.n_head}, Dc={cfg.kv_lora_rank}, Dr={cfg.qk_rope_head_dim}, "
          f"Dn={cfg.qk_nope_head_dim}, Dv={cfg.v_head_dim}; scale = (Dn+Dr)^-0.5 = {a.scale:.4f}")
    for name, t in (("x", x), ("q", q), ("q_nope", q_nope), ("q_pe", q_pe), ("compressed", compressed), ("c", c),
                    ("k_pe", k_pe), ("kv", kv), ("k", k), ("v", v), ("y", y), ("out", out)):
        print(f"  {name:10s} {shape(t)}")
    print(f"recomputed vs MLAttention.forward: {(out - ref).abs().max():.1e}")
    assert torch.allclose(out, ref, atol=1e-5)


def cache():
    cfg, rot, a = build()
    x = torch.randn(1, 6, cfg.hidden_size)
    with torch.no_grad():
        cos, sin = rot(6, x.device, x.dtype)
        full = a(x, cos, sin)                                   # all six tokens, no cache
        kv = {}
        c5, s5 = rot(5, x.device, x.dtype)
        a(x[:, :5], c5, s5, cache=kv)                           # prompt: five tokens
        print(f"after the prompt: c {shape(kv['c'])}, k_pe {shape(kv['k_pe'])}")
        c1, s1 = rot(1, x.device, x.dtype, start=5)
        step = a(x[:, 5:], c1, s1, cache=kv, start=5)           # one new token at position 5
        print(f"after one step:   c {shape(kv['c'])}, k_pe {shape(kv['k_pe'])}")
    print(f"cached step vs uncached forward, last position: {(step[:, 0] - full[:, 5]).abs().max():.1e}")
    assert torch.allclose(step[:, 0], full[:, 5], atol=1e-5)


def absorb():
    cfg, rot, a = build()
    x = torch.randn(1, 5, cfg.hidden_size)
    H, Dn, Dc = a.n_head, a.qk_nope, a.kv_lora_rank
    with torch.no_grad():
        q = a.q_layernorm(a.q_proj(x).view(1, 5, H, a.q_head_dim).transpose(1, 2))
        q_nope = q[..., :Dn]                                    # (1, H, S, Dn)
        c = a.kv_a_layernorm(a.kv_a_proj_with_mqa(x)[..., :Dc]) # (1, S, Dc)
        W = a.kv_b_proj.weight.view(H, Dn + a.v_head_dim, Dc)   # per head: rows for k_nope, then v
        W_uk = W[:, :Dn, :]                                     # (H, Dn, Dc)
        k_nope = torch.einsum("hnc,btc->bhtn", W_uk, c)        # what the repository expands
        repo = q_nope @ k_nope.transpose(-1, -2)                # (1, H, S, T)
        q_lat = torch.einsum("bhsn,hnc->bhsc", q_nope, W_uk)   # the query moved into latent space
        absorbed = q_lat @ c.unsqueeze(1).transpose(-1, -2)     # scores straight from the cache
    print(f"latent part of the scores: expanded keys vs absorbed query, largest difference "
          f"{(repo - absorbed).abs().max():.1e}")
    pos = torch.randn(1, 1, 5, a.qk_rope)
    cos, sin = rot(5, x.device, x.dtype)
    r1, _ = M.apply_rope(pos, pos, cos, sin)
    print(f"a rotated vector depends on its position: the same input at positions 0 and 4 differs by "
          f"{(r1[0, 0, 0] - r1[0, 0, 4]).abs().max():.2f} after rotation (input identical)")


def shift():
    cfg, rot, a = build()
    x = torch.randn(1, 8, cfg.hidden_size)
    with torch.no_grad():
        y0 = a(x, *rot(8, x.device, x.dtype))                   # positions 0-7
        y1 = a(x, *rot(8, x.device, x.dtype, start=100))        # the same tokens at 100-107
    print(f"same tokens at positions 0-7 and 100-107: outputs differ by {(y0 - y1).abs().max():.1e} "
          f"(only relative positions matter)")
    assert torch.allclose(y0, y1, atol=1e-4)


def mutant_shift():
    """A mutant that also rotates the latent, by tables as wide as the latent. The cache test cannot see it;
    the shift test can."""
    cfg, rot, a = build()
    rot_c = M.Rotary(cfg, cfg.kv_lora_rank)
    x = torch.randn(1, 8, cfg.hidden_size)
    H, S = a.n_head, 8

    def forward(start):
        cos, sin = rot(S, x.device, x.dtype, start=start)
        cc, sc = rot_c(S, x.device, x.dtype, start=start)
        q = a.q_layernorm(a.q_proj(x).view(1, S, H, a.q_head_dim).transpose(1, 2))
        q_nope, q_pe = q.split([a.qk_nope, a.qk_rope], dim=-1)
        c, k_pe = a.kv_a_proj_with_mqa(x).split([a.kv_lora_rank, a.qk_rope], dim=-1)
        q_pe, k_pe = M.apply_rope(q_pe, k_pe.view(1, S, 1, a.qk_rope).transpose(1, 2), cos, sin)
        c = a.kv_a_layernorm(c)
        c, _ = M.apply_rope(c[:, None], c[:, None], cc, sc)               # the mutation: rotate the latent
        kv = a.kv_b_proj(c[:, 0]).view(1, S, H, a.qk_nope + a.v_head_dim).transpose(1, 2)
        k_nope, v = kv.split([a.qk_nope, a.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, H, -1, -1)], dim=-1)
        y = F.scaled_dot_product_attention(torch.cat([q_nope, q_pe], -1), k, v, is_causal=True, scale=a.scale)
        return a.o_proj(y.transpose(1, 2).reshape(1, S, -1))

    with torch.no_grad():
        d = (forward(0) - forward(100)).abs().max().item()
        scale = forward(0).abs().max().item()
    print(f"mutant with a rotated latent: positions 0-7 vs 100-107 differ by {d:.3f} (output scale {scale:.3f})")


def arithmetic():
    rows = []
    cm = M.AnuLMConfig.tiny_mla()
    rows.append(("tiny_mla, MLA (cached)", cm.kv_lora_rank + cm.qk_rope_head_dim))
    rows.append(("tiny_mla, same heads as full K and V",
                 cm.n_head * (cm.qk_nope_head_dim + cm.qk_rope_head_dim) + cm.n_head * cm.v_head_dim))
    cg = M.AnuLMConfig.tiny_gqa()
    rows.append(("tiny_gqa, GQA", 2 * cg.n_kv_head * cg.head_dim))
    rows.append(("tiny_gqa, if every head kept its K and V", 2 * cg.n_head * cg.head_dim))
    r = released()
    rows.append(("released 400M, GQA", 2 * r.n_kv_head * r.head_dim))
    rows.append(("released 400M, if every head kept its K and V", 2 * r.n_head * r.head_dim))
    for name, n in rows:
        print(f"  {name:48s} {n:6,} numbers per token per layer")
    mla, full = rows[0][1], rows[1][1]
    print(f"tiny_mla: {full:,} / {mla} = {full / mla:.1f} times smaller; over {cm.n_layer} layers in bf16: "
          f"{mla * cm.n_layer * 2:,} bytes per token")
    T = cm.block_size                                                # the longest context tiny_mla accepts
    Dc, H, Dn, Dv = cm.kv_lora_rank, cm.n_head, cm.qk_nope_head_dim, cm.v_head_dim
    print(f"re-expanding the cache every step, tiny_mla, T = block_size = {T}: kv_b_proj does T*Dc*H*(Dn+Dv) = "
          f"{T * Dc * H * (Dn + Dv):,} multiply-adds per layer, {T * Dc * H * Dn:,} for keys, {T * Dc * H * Dv:,} for values")


def exercises():
    print("exercise 1: Dc + Dr for kv_lora_rank 64, rope 32:", 64 + 32)
    c = M.AnuLMConfig.tiny_mla()
    print("exercise 2: tiny_mla cache over 12 layers, fp32, per token:", (c.kv_lora_rank + c.qk_rope_head_dim) * 12 * 4, "bytes")
    print("exercise 3: q_proj width, H*(Dn+Dr):", c.n_head * (c.qk_nope_head_dim + c.qk_rope_head_dim))


if __name__ == "__main__":
    by_steps()
    cache()
    absorb()
    shift()
    mutant_shift()
    arithmetic()
    exercises()
