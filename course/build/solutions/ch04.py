"""
Reference solution for Chapter 4: the model's skeleton.

These match model.py's RMSNorm, SwiGLU and initialisation exactly (the
checks in course/checks/ch04.py prove it). SkeletonLM is AnuLM with the
attention left out: Chapter 6 adds it, Chapter 7 swaps the MLPs for experts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Scale each vector so its root-mean-square is 1, then by a learned gain."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))     # gain starts at 1: the identity scale
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()                                     # the statistics in float32, whatever x is
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


class SwiGLU(nn.Module):
    """down( silu(gate(x)) * up(x) ): a gated MLP, 3 * D * I parameters, no biases."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MLPBlock(nn.Module):
    """One pre-norm residual block, attention not yet added: x + mlp(norm(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int, eps: float = 1e-6):
        super().__init__()
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)   # model.py's name for it
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(self, x):
        return x + self.mlp(self.post_attention_layernorm(x))


class SkeletonLM(nn.Module):
    """Embeddings, n residual blocks, a final norm and an untied output head."""

    def __init__(self, vocab_size: int = 259, hidden_size: int = 384, n_layer: int = 4,
                 intermediate_size: int = 768, initializer_range: float = 0.02,
                 tie_word_embeddings: bool = False):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(MLPBlock(hidden_size, intermediate_size) for _ in range(n_layer))
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        # model.py's initialisation: every weight N(0, 0.02), then the projections
        # that write into the residual stream scaled down by sqrt(2 * n_layer).
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
        for name, p in self.named_parameters():
            if name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))

    def forward(self, idx, targets=None):
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None if targets is None else F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, loss
