"""
Reference solution for Chapter 6: attention.

attn_mask, GQAttention and Block match model.py's _attn_mask, GQAttention and
(dense) Block exactly; course/checks/ch06.py proves it. AttnLM is Chapter 4's
skeleton with attention and rotary positions added: AnuLM with every MLP still
dense (Chapter 7 turns all but the first into experts).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ch04 import RMSNorm, SwiGLU  # noqa: E402
from ch05 import Rotary, apply_rope  # noqa: E402


def attn_mask(S: int, T: int, start: int, window: Optional[int], device):
    """(S, T) booleans, True = may attend; or None when is_causal (or nothing) is enough.
    Queries sit at absolute positions start .. start+S-1, keys at 0 .. T-1."""
    if window is None and (start == 0 or S == 1):
        return None
    q_pos = torch.arange(start, start + S, device=device)[:, None]
    k_pos = torch.arange(T, device=device)[None, :]
    mask = k_pos <= q_pos                       # never a key after the query
    if window is not None:
        mask &= k_pos > q_pos - window          # and none more than window - 1 back
    return mask


class GQAttention(nn.Module):
    """Grouped-query attention: n_head query heads share n_kv_head key/value heads."""

    def __init__(self, hidden_size: int, n_head: int, n_kv_head: int, head_dim: int,
                 use_qk_norm: bool = True, eps: float = 1e-6, attn_mscale: float = 1.0):
        super().__init__()
        assert n_head % n_kv_head == 0
        self.n_head, self.n_kv_head, self.head_dim = n_head, n_kv_head, head_dim
        self.n_rep = n_head // n_kv_head
        self.query_key_value = nn.Linear(hidden_size, (n_head + 2 * n_kv_head) * head_dim, bias=False)
        self.dense = nn.Linear(n_head * head_dim, hidden_size, bias=False)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.query_layernorm = RMSNorm(head_dim, eps)
            self.key_layernorm = RMSNorm(head_dim, eps)
        self.scale = head_dim ** -0.5 * attn_mscale ** 2       # YaRN's temperature lands here

    def forward(self, x, cos, sin, cache: Optional[dict] = None, start: int = 0,
                window: Optional[int] = None):
        B, S, _ = x.shape
        qkv = self.query_key_value(x).view(B, S, self.n_head + 2 * self.n_kv_head, self.head_dim)
        q, k, v = qkv.split([self.n_head, self.n_kv_head, self.n_kv_head], dim=-2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)    # (B, heads, S, Dh)
        if self.use_qk_norm:                                   # normalise, THEN rotate
            q, k = self.query_layernorm(q), self.key_layernorm(k)
        q, k = apply_rope(q, k, cos, sin)
        if cache is not None:                                  # keep the n_kv_head heads, not copies
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        T = k.shape[2]
        mask = attn_mask(S, T, start, window, x.device)
        k = k.repeat_interleave(self.n_rep, dim=1)             # each kv head serves n_rep query heads
        v = v.repeat_interleave(self.n_rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None and S > 1),
                                           scale=self.scale)
        return self.dense(y.transpose(1, 2).reshape(B, S, -1))


class Block(nn.Module):
    """x + attn(input_layernorm(x)), then + mlp(post_attention_layernorm(x)): model.py's layer 0."""

    def __init__(self, hidden_size: int, n_head: int, n_kv_head: int, head_dim: int,
                 intermediate_size: int, use_qk_norm: bool = True, eps: float = 1e-6,
                 attn_mscale: float = 1.0, window: Optional[int] = None):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.attn = GQAttention(hidden_size, n_head, n_kv_head, head_dim, use_qk_norm, eps, attn_mscale)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        self.window = window

    def forward(self, x, cos, sin, cache: Optional[dict] = None, start: int = 0):
        x = x + self.attn(self.input_layernorm(x), cos, sin, cache=cache, start=start, window=self.window)
        return x + self.mlp(self.post_attention_layernorm(x))


class AttnLM(nn.Module):
    """Embeddings, rotary positions, n attention blocks, a final norm, an untied head."""

    def __init__(self, vocab_size: int = 259, hidden_size: int = 128, n_layer: int = 4, n_head: int = 4,
                 n_kv_head: int = 2, head_dim: int = 32, intermediate_size: int = 256,
                 rope_theta: float = 10000.0, block_size: int = 256, use_qk_norm: bool = True,
                 sliding_window: Optional[int] = None, max_window_layers: int = 0,
                 initializer_range: float = 0.02):
        super().__init__()
        self.block_size = block_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary = Rotary(head_dim, rope_theta=rope_theta)
        self.layers = nn.ModuleList(
            Block(hidden_size, n_head, n_kv_head, head_dim, intermediate_size, use_qk_norm,
                  attn_mscale=self.rotary.attn_mscale,
                  window=sliding_window if i < max_window_layers else None)
            for i in range(n_layer))
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
        for name, p in self.named_parameters():       # both residual-writing projections, scaled
            if name.endswith(("dense.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))

    def new_cache(self) -> list[dict]:
        return [dict() for _ in self.layers]

    def forward(self, idx, targets=None, cache: Optional[list] = None, start: int = 0):
        B, S = idx.shape
        assert start + S <= self.block_size, f"positions up to {start + S} exceed block_size {self.block_size}"
        x = self.embed_tokens(idx)
        cos, sin = self.rotary(S, x.device, x.dtype, start=start)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cache=None if cache is None else cache[i], start=start)
        logits = self.lm_head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n: int, temperature: float = 1.0, use_cache: bool = True, generator=None):
        """Sample n tokens. With the cache, the prompt is read once and each later step
        feeds one token; without it, every step re-reads everything."""
        cache, pending, start = (self.new_cache() if use_cache else None), idx, 0
        for _ in range(n):
            if use_cache:
                logits, _ = self(pending, cache=cache, start=start)
                start += pending.shape[1]
            else:
                logits, _ = self(idx[:, -self.block_size:])
            p = F.softmax(logits[:, -1] / temperature, dim=-1)
            nxt = torch.multinomial(p, 1, generator=generator)
            idx, pending = torch.cat([idx, nxt], dim=1), nxt
        return idx
