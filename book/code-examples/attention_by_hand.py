"""
Book example, not repository code.

Chapter 11: attention from first principles.

    python book/code-examples/attention_by_hand.py

Each part is labelled with the chapter section it feeds:

11.3  The 2-D toy `The dog bit`: every number of the by-hand calculation, then
      the same inputs through PyTorch's SDPA and through `model.py`'s `_sdpa`.
11.4  Why divide by √d: random unit-variance queries against shared random
      keys at widths 2, 8 and 64, with and without the scale (seeded).
11.6  The scale in the real model: a released-size attention layer, initialised
      as AnuLM._init_weights does, where QK-norm bounds every scaled score;
      then the same bound computed from the trained gains in ckpt_base_v2.pt.
11.8  The cost of the score matrix at the released size (arithmetic), and
      which SDPA kernel this machine's PyTorch picks on the CPU.

The toy vectors and the random inputs are made up for teaching. `model.py` is
imported unchanged.
"""

from __future__ import annotations

import math
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


def fmt(t, nd=3):
    if t.dim() == 1:
        return "[" + ", ".join(f"{v:.{nd}f}" for v in t.tolist()) + "]"
    return "\n".join("    " + fmt(r, nd) for r in t)


def toy() -> None:
    print("11.3  The toy: The dog bit (illustrative numbers)")
    words = ["The", "dog", "bit"]
    X = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])          # one row per token
    W_q = torch.tensor([[1., 0.], [0., 2.]])
    W_k = torch.eye(2)
    W_v = torch.tensor([[0., 1.], [1., 0.]])
    Q, K, V = X @ W_q, X @ W_k, X @ W_v                         # row vector times matrix
    d = X.shape[1]
    for name, T in (("X", X), ("Q = X W_Q", Q), ("K = X W_K", K), ("V = X W_V", V)):
        print(f"  {name}:\n{fmt(T, 0)}")
    scores = Q @ K.T
    print(f"  scores Q Kᵀ (row = query, column = key):\n{fmt(scores, 0)}")
    scaled = scores / math.sqrt(d)
    print(f"  ÷ √{d} = ÷ {math.sqrt(d):.3f}:\n{fmt(scaled)}")
    mask = torch.ones(3, 3, dtype=torch.bool).tril()
    masked = scaled.masked_fill(~mask, float("-inf"))
    print(f"  causal mask (−∞ above the diagonal):\n{fmt(masked)}")
    W = masked.softmax(-1)
    print(f"  softmax along each row:\n{fmt(W)}")
    out = W @ V
    print(f"  output = weights · V:\n{fmt(out)}")
    e = scaled[2].exp()
    print(f"  the 'bit' row in full: exp({fmt(scaled[2])}) = {fmt(e)}, sum {e.sum():.3f}, "
          f"weights {fmt(e / e.sum())}")
    print(f"  'bit' output = " + " + ".join(f"{w:.3f}·{fmt(v, 0)}" for w, v in zip(W[2].tolist(), V)) +
          f" = {fmt(out[2])}")
    for i, w in enumerate(words):
        best = words[int(W[i].argmax())]
        print(f"  {w:>4} attends most to {best!r} (weight {W[i].max():.3f})")

    q, k, v = (t[None, None] for t in (Q, K, V))                # (B=1, H=1, S=3, d=2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    rep = M._sdpa(q, k, v, None, 3, d ** -0.5, 0.0)
    print(f"  PyTorch SDPA (is_causal) vs by hand: max |diff| {(ref[0, 0] - out).abs().max():.1e}")
    print(f"  model._sdpa (mask None, S = 3, scale = 1/√2) vs by hand: max |diff| {(rep[0, 0] - out).abs().max():.1e}")


def why_scale() -> None:
    print("\n11.4  Why divide by √d (random unit-variance vectors, seed 0; illustrative)")
    g = torch.Generator().manual_seed(0)
    n_q, n_k = 2000, 256
    print(f"  {n_q} queries against the same {n_k} keys; components drawn from N(0, 1)")
    print(f"  {'d':>3} {'std of q·k':>11} {'√d':>6} | {'largest weight':>15} {'keys in play':>13} {'rows > 0.99':>12}"
          f" | same, ÷√d")
    for d in (2, 8, 64):
        q = torch.randn(n_q, d, generator=g)
        k = torch.randn(n_k, d, generator=g)
        s = q @ k.T
        row = []
        for logits in (s, s / math.sqrt(d)):
            p = logits.softmax(-1)
            top = p.max(-1).values.mean().item()
            in_play = (-(p * p.clamp_min(1e-30).log()).sum(-1)).exp().mean().item()   # exp(entropy)
            # a row whose largest weight is above 0.99 is nearly one-hot: its gradient
            # dp_i/ds_j = p_i(δ_ij − p_j) is close to zero for every key
            saturated = (p.max(-1).values > 0.99).float().mean().item()
            row.append((top, in_play, saturated))
        (a1, b1, c1), (a2, b2, c2) = row
        print(f"  {d:>3} {s.std().item():>11.2f} {math.sqrt(d):>6.2f} | {a1:>15.3f} {b1:>13.1f} {c1:>12.1%}"
              f" | {a2:.3f} {b2:.1f} {c2:.1%}")


def real_scale() -> None:
    print("\n11.6  The scale in the released configuration (fresh weights, random input; illustrative)")
    cfg = released()
    torch.manual_seed(0)
    rot = M.Rotary(cfg, cfg.head_dim)
    att = M.GQAttention(cfg, rot).eval()
    for lin in (att.query_key_value, att.dense):            # AnuLM._init_weights: N(0, initializer_range)
        torch.nn.init.normal_(lin.weight, mean=0.0, std=cfg.initializer_range)
    print(f"  head_dim {cfg.head_dim}: scale = {cfg.head_dim}^-0.5 × attn_mscale² = "
          f"{cfg.head_dim ** -0.5:.4f} × {rot.attn_mscale ** 2:.1f} = {att.scale}")
    B, S, D = 1, 256, cfg.hidden_size
    x = torch.randn(B, S, D)
    with torch.no_grad():
        qkv = att.query_key_value(x).view(B, S, att.n_head + 2 * att.n_kv_head, att.head_dim)
        q, k, _ = qkv.split([att.n_head, att.n_kv_head, att.n_kv_head], dim=-2)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        raw_q = q
        q, k = att.query_layernorm(q), att.key_layernorm(k)
        cos, sin = rot(S, x.device, x.dtype)
        q, k = M.apply_rope(q, k, cos, sin)
        k = k.repeat_interleave(att.n_rep, dim=1)
        s = (q @ k.transpose(-2, -1)) * att.scale
    print(f"  query length before QK-norm: mean {raw_q.norm(dim=-1).mean():.2f}; after QK-norm and RoPE: "
          f"{q.norm(dim=-1).mean():.3f} (√{cfg.head_dim} = {math.sqrt(cfg.head_dim):.3f})")
    bound = q.norm(dim=-1).max() * k.norm(dim=-1).max() * att.scale
    print(f"  scaled scores (all {S}×{S} pairs, unmasked): std {s.std():.3f}, largest |score| {s.abs().max():.3f}, "
          f"bound ‖q‖·‖k‖·scale = {bound:.3f}")


def trained_gains() -> None:
    print("\n11.6  The same bound in the trained model (QK-norm gains read from ckpt_base_v2.pt)")
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False, mmap=True)
    sd, cfg = ck["model"], ck["cfg"]
    scale = cfg.head_dim ** -0.5
    lims = []
    for i in range(cfg.n_layer):
        gq = sd[f"layers.{i}.attn.query_layernorm.weight"].float()
        gk = sd[f"layers.{i}.attn.key_layernorm.weight"].float()
        # ‖g ⊙ x̂‖ ≤ max|g| · ‖x̂‖ = max|g| · √Dh, so |score| ≤ max|g_q| · max|g_k| · Dh · scale
        lim = gq.abs().max().item() * gk.abs().max().item() * cfg.head_dim * scale
        lims.append(lim)
        print(f"  layer {i:2d}: mean gain q {gq.mean():.2f}, k {gk.mean():.2f}; "
              f"largest |gain| q {gq.abs().max():.2f}, k {gk.abs().max():.2f}; limit {lim:.1f}")
    srt = sorted(lims)
    print(f"  limit over the 20 layers: smallest {srt[0]:.1f}, median {(srt[9] + srt[10]) / 2:.1f}, largest {srt[-1]:.1f}")


def kernels() -> None:
    print("\n11.8  Which SDPA implementation this PyTorch picks on this machine's CPU")
    print(f"  torch {torch.__version__}; backends it names: "
          + ", ".join(n for n in dir(torch.nn.attention.SDPBackend) if n.isupper() and n not in ("ERROR", "OVERRIDEABLE")))
    q = torch.randn(1, 16, 256, 64)
    mask = torch.ones(256, 256, dtype=torch.bool).tril()
    names = {int(getattr(torch.nn.attention.SDPBackend, n)): n
             for n in dir(torch.nn.attention.SDPBackend) if n.isupper()}
    for label, m, p, causal in (("causal, no mask", None, 0.0, True), ("boolean mask", mask, 0.0, False),
                                ("causal, dropout 0.1", None, 0.1, True)):
        choice = torch._fused_sdp_choice(q, q, q, m, p, causal)
        print(f"  {label:22s} -> {names.get(int(choice), choice)}")
    import warnings
    from torch.profiler import profile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # the profiler warns about its own event buffer
        with profile() as prof:
            F.scaled_dot_product_attention(q, q, q, is_causal=True)
    print("  profiler, causal call: " + ", ".join(sorted({e.name for e in prof.events() if "flash" in e.name or "math" in e.name})))


def cost() -> None:
    print("\n11.8  The score matrix at the released size (arithmetic)")
    cfg = released()
    H, Dh, D, Hkv = cfg.n_head, cfg.head_dim, cfg.hidden_size, cfg.n_kv_head
    for S in (1024, 4096):
        scores = H * S * S
        print(f"  S = T = {S}: {H} heads × {S}² = {scores:,} scores per layer per sequence = "
              f"{scores * 2 / 2**20:,.0f} MiB in bf16, {scores * 4 / 2**20:,.0f} MiB in fp32")
        core = 2 * (2 * S * S * Dh * H)             # Q Kᵀ and weights · V, 2 FLOPs per multiply-add
        proj = 2 * S * D * (H + 2 * Hkv) * Dh + 2 * S * H * Dh * D
        print(f"    FLOPs per layer: QKᵀ and ·V {core / 1e9:.1f} G; the qkv and output projections {proj / 1e9:.1f} G")


if __name__ == "__main__":
    toy()
    why_scale()
    real_scale()
    trained_gains()
    cost()
    kernels()
