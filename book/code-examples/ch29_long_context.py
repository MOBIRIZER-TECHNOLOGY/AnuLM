"""
Book example, not repository code.

Chapter 29: what RoPE, YaRN and the sliding window do, computed with
`model.py`'s own functions for the released model's configuration, and a
trimmed cache for windowed layers as an EXTENSION, checked against the
repository's untrimmed one on the test model `tiny()` (CPU, a second).

    python book/code-examples/ch29_long_context.py

It runs `model.py` unchanged. The released configuration is `nano_350m` with
the released run's settings (as `trace_forward.released()` builds it); no
weights are needed for anything here except the EXTENSION check, which uses
`tiny()`'s random weights.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def released(**kw) -> M.AnuLMConfig:
    """AnuLM-Base-v2-400M's configuration (Chapter 20), with optional changes."""
    cfg = M.AnuLMConfig.nano_350m(vocab_size=32768, block_size=1024, sliding_window=256,
                                  max_window_layers=10, num_experts=24, num_experts_per_tok=4,
                                  num_shared_experts=0, moe_intermediate_size=192)
    return replace(cfg, **kw)


def tiny(**kw) -> M.AnuLMConfig:
    """test_model.py's tiny(), copied so this script needs no test harness."""
    base = dict(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4,
                head_dim=8, n_kv_head=2, num_experts=8, num_experts_per_tok=2,
                moe_intermediate_size=16, intermediate_size=64, n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


def wavelengths():
    cfg = released()
    inv = M.Rotary._build_inv_freq(cfg, cfg.head_dim)      # (Dh/2,) = (32,) frequencies
    lam = 2 * math.pi / inv                                 # tokens per full turn
    turns = cfg.block_size / lam                            # turns within the trained length
    for i in (0, 4, 8, 12, 16, 24, 31):
        print(f"pair {i:2d}: wavelength {lam[i].item():12,.1f} tokens, "
              f"{turns[i].item():9.3f} turns in 1,024")
    print(f"pair 16 at a distance of 2,000: {2000 / lam[16].item():.2f} of a turn")
    short = int((turns < 1).sum())
    print(f"{short} of {len(lam)} pairs never complete a turn within 1,024 tokens")
    return short


def yarn_bands():
    plain = M.Rotary._build_inv_freq(released(), 64)        # plain RoPE, (32,)
    print(f"{'divisor at pair':26s}", " ".join(f"{p:5d}" for p in (0, 3, 4, 6, 8, 10, 12, 31)))
    for factor in (2.0, 4.0):
        cfg = released(yarn=True, yarn_factor=factor, yarn_original_context=1024)
        low, high = M._yarn_correction_range(cfg.yarn_beta_fast, cfg.yarn_beta_slow,
                                             64, cfg.rope_theta, 1024)
        div = plain / M.Rotary._build_inv_freq(cfg, 64)      # 1 = kept, factor = divided
        print(f"factor {factor:.0f} (low {low}, high {high}):",
              " ".join(f"{d:5.2f}" for d in div[[0, 3, 4, 6, 8, 10, 12, 31]].tolist()))


def temperature():
    for factor in (2.0, 4.0):
        cfg = released(yarn=True, yarn_factor=factor, yarn_original_context=1024)
        rot = M.Rotary(cfg, cfg.head_dim)
        attn = M.GQAttention(cfg, rot)
        ms = M.yarn_mscale(factor, 1.0)                     # 0.1 * ln(s) + 1
        assert abs(attn.scale - cfg.head_dim ** -0.5 * ms ** 2) < 1e-12
        print(f"factor {factor:.0f}: mscale {ms:.4f}, scale x {ms ** 2:.4f} "
              f"({attn.scale:.5f} instead of {cfg.head_dim ** -0.5:.5f}); "
              f"table factor {rot.table_mscale:.1f}")


def window_rows():
    print("prefill of 6, window 3:", "  ".join(
        "".join("x" if v else "." for v in r) for r in M._attn_mask(6, 6, 0, 3, "cpu").tolist()))
    print("decode at 9, window 4: ", "".join(
        "x" if v else "." for v in M._attn_mask(1, 10, 9, 4, "cpu")[0].tolist()))


@torch.no_grad()
def trimmed_cache_extension(steps=20):
    """EXTENSION: windowed layers keep only their last window - 1 cached positions
    before each decode step. Compared with the repository's untrimmed cache."""
    torch.manual_seed(0)
    W = 4
    m = M.AnuLM(tiny(sliding_window=W, max_window_layers=2, block_size=32)).eval()
    windowed = [i for i, b in enumerate(m.layers) if b.window is not None]
    x = torch.randint(0, 64, (1, 6 + steps))
    full, trim = m.new_cache(), m.new_cache()
    m(x[:, :6], cache=full, start=0)
    m(x[:, :6], cache=trim, start=0)
    worst = 0.0
    for t in range(6, 6 + steps):
        ref, _ = m(x[:, t:t + 1], cache=full, start=t)        # the repository's path
        for i in windowed:                                    # keep the last W - 1 keys
            c = trim[i]
            c["k"], c["v"] = c["k"][:, :, -(W - 1):], c["v"][:, :, -(W - 1):]
            m.layers[i].window = None                         # all W keys are in reach
        try:
            got, _ = m(x[:, t:t + 1], cache=trim, start=t)
        finally:
            for i in windowed:
                m.layers[i].window = W
        worst = max(worst, (got - ref).abs().max().item())
    sizes = [trim[i]["k"].shape[2] for i in range(len(trim))]
    print(f"{steps} decode steps: max |logit difference| {worst:.1e}; "
          f"cached positions per layer, trimmed {sizes} vs {full[0]['k'].shape[2]}")
    assert worst < 1e-5
    per = 4 * 64 * (4 + 2)                                    # Hkv*Dh*(fp32 K + bf16 V)
    now = 20 * 1024 * per
    ext = (10 * 1024 + 10 * 256) * per
    print(f"released model at 1,024 tokens: {now / 2**20:.2f} MiB now, "
          f"{ext / 2**20:.2f} MiB with trimmed windowed layers ({1 - ext / now:.1%} less)")


if __name__ == "__main__":
    print("== RoPE wavelengths, released model (theta 1e6, Dh 64)")
    wavelengths()
    print("\n== YaRN's bands (beta_fast 32, beta_slow 1, original context 1,024)")
    yarn_bands()
    print("\n== YaRN's temperature on the attention scale")
    temperature()
    print("exercise 1: factor 8, m^2 =", f"{M.yarn_mscale(8.0, 1.0) ** 2:.4f}")
    print("\n== the window in _attn_mask")
    window_rows()
    print("\n== EXTENSION: a trimmed cache for windowed layers")
    trimmed_cache_extension()
