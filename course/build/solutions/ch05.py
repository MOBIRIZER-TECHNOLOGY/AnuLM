"""
Reference solution for Chapter 5: rotary positions (RoPE) and YaRN.

Matches model.py's rotate_half, apply_rope, yarn_mscale, the YaRN correction
range and Rotary exactly (course/checks/ch05.py proves it). Rotary here takes
the config's fields as arguments instead of the whole config object.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def rotate_half(x):
    """(x1, x2) -> (-x2, x1), on the two halves of the last dimension."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Rotate q and k (B, H, S, Dr) by the angles in cos/sin (S, Dr)."""
    cos, sin = cos[None, None], sin[None, None]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    """YaRN's attention temperature for a context extended `scale` times."""
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


_mscale = yarn_mscale      # Rotary's argument of the same name (the config's) hides the function


def yarn_correction_range(beta_fast, beta_slow, dim, base, orig_ctx):
    """The rotary dimensions (as pair indices) between which YaRN ramps from
    'keep this frequency' to 'divide it by factor'."""
    def dim_for(rotations):           # the pair index that turns `rotations` times in orig_ctx
        return dim * math.log(orig_ctx / (rotations * 2 * math.pi)) / (2 * math.log(base))
    low = math.floor(dim_for(beta_fast))
    high = math.ceil(dim_for(beta_slow))
    return max(low, 0), min(high, dim - 1)


class Rotary(nn.Module):
    def __init__(self, dim: int, rope_theta: float = 10000.0, yarn: bool = False,
                 yarn_factor: float = 8.0, yarn_original_context: int = 256,
                 yarn_beta_fast: float = 32.0, yarn_beta_slow: float = 1.0,
                 yarn_mscale: float = 1.0, yarn_mscale_all_dim: float = 1.0):
        super().__init__()
        self.dim = dim
        idx = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        inv_freq = 1.0 / (rope_theta ** idx)                     # one frequency per pair, fast to slow
        if yarn:
            slow = 1.0 / (yarn_factor * rope_theta ** idx)       # stretched by the extension factor
            low, high = yarn_correction_range(yarn_beta_fast, yarn_beta_slow, dim, rope_theta,
                                              yarn_original_context)
            if low == high:
                high += 0.001
            ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
            inv_freq = slow * ramp + inv_freq * (1 - ramp)       # fast pairs kept, slow ones stretched
            self.attn_mscale = _mscale(yarn_factor, yarn_mscale_all_dim)
            self.table_mscale = _mscale(yarn_factor, yarn_mscale) / _mscale(yarn_factor, yarn_mscale_all_dim)
        else:
            self.attn_mscale = self.table_mscale = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len, self._cached_key = 0, None
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def forward(self, seq_len: int, device, dtype, start: int = 0):
        """cos, sin for absolute positions start .. start + seq_len - 1, shape (seq_len, dim)."""
        key = (device, dtype)
        if start + seq_len > self._cached_len or self._cached_key != key:
            n = max(start + seq_len, self._cached_len)
            t = torch.arange(n, device=device, dtype=torch.float32)      # angles in float32, always
            freqs = torch.outer(t, self.inv_freq.to(device).float())
            emb = torch.cat((freqs, freqs), dim=-1)                       # the split-halves layout
            self.cos_cached = (emb.cos() * self.table_mscale).to(dtype)
            self.sin_cached = (emb.sin() * self.table_mscale).to(dtype)
            self._cached_len, self._cached_key = n, key
        return self.cos_cached[start:start + seq_len], self.sin_cached[start:start + seq_len]
