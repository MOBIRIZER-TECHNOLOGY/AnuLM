"""
Reference solution for Chapter 7: mixture of experts.

Router, MoE (sparse dispatch) and ExpertBlock match model.py's Router, MoE and
MoE layers exactly. MoELM is the whole model: with these, a MoELM's weights
load into model.py's AnuLM (grouped-query attention) and give the same logits,
which course/checks/ch07.py proves. Its defaults are the `tiny-gqa` preset.
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
from ch05 import Rotary  # noqa: E402
from ch06 import GQAttention, Block  # noqa: E402


class Router(nn.Module):
    """Sigmoid scores, a balancing bias that picks but never weights, optional groups, top-k."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, n_group: int = 1,
                 topk_group: int = 1, routed_scaling_factor: float = 2.5,
                 seq_balance_alpha: float = 0.0, router_z_alpha: float = 0.0):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.n_group, self.topk_group = n_group, topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.seq_balance_alpha, self.router_z_alpha = seq_balance_alpha, router_z_alpha
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Buffers, not parameters: no gradient, never touched by the optimizer,
        # but saved in the checkpoint and moved with .to(device).
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.register_buffer("load_counts", torch.zeros(num_experts))

    def forward(self, x_flat, seq_len: Optional[int] = None):
        # Route in float32 with autocast off: top-k is a discrete choice, and
        # bfloat16 rounding flips it (see the chapter).
        with torch.autocast(device_type=x_flat.device.type, enabled=False):
            idx, w, aux = self._gate(x_flat.float(), seq_len)
        return idx, w.to(x_flat.dtype), aux

    def _gate(self, x, seq_len):
        logits = F.linear(x, self.weight.float())                  # (N, E)
        scores = torch.sigmoid(logits)                              # each expert scored on its own
        choice = scores + self.expert_bias[None, :]                 # the bias only steers the choice
        if self.n_group > 1:
            N, per = scores.shape[0], self.num_experts // self.n_group
            grouped = choice.view(N, self.n_group, per)
            g_score = grouped.topk(min(2, per), dim=-1)[0].sum(-1)  # a group is worth its best two
            keep = torch.zeros_like(g_score).scatter_(1, g_score.topk(self.topk_group, dim=-1)[1], 1.0)
            keep = keep.unsqueeze(-1).expand(N, self.n_group, per).reshape(N, -1)
            choice = choice.masked_fill(~keep.bool(), float("-inf"))
        idx = choice.topk(self.top_k, dim=-1)[1]                    # (N, k)
        w = scores.gather(1, idx)                                   # weights from the UNBIASED scores
        if self.top_k > 1:
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        w = w * self.routed_scaling_factor
        aux = None
        if self.training:
            with torch.no_grad():
                self.load_counts += torch.bincount(idx.flatten(), minlength=self.num_experts).to(self.load_counts.dtype)
            if self.seq_balance_alpha > 0 or self.router_z_alpha > 0:
                aux = logits.new_zeros(())
            if self.seq_balance_alpha > 0:
                T = seq_len or scores.shape[0]
                f = F.one_hot(idx, self.num_experts).sum(1).float().view(-1, T, self.num_experts).mean(1) \
                    * (self.num_experts / self.top_k)
                P = (scores / scores.sum(-1, keepdim=True)).view(-1, T, self.num_experts).mean(1)
                aux = aux + self.seq_balance_alpha * (f * P).sum(-1).mean()
            if self.router_z_alpha > 0:
                aux = aux + self.router_z_alpha * torch.logsumexp(logits, dim=-1).pow(2).mean()
        return idx, w, aux


class MoE(nn.Module):
    """num_experts small SwiGLUs, top_k of them per token, plus a shared expert for every token."""

    def __init__(self, hidden_size: int, moe_intermediate_size: int, num_experts: int, top_k: int,
                 num_shared_experts: int = 1, **router_kw):
        super().__init__()
        self.num_experts, self.top_k = num_experts, top_k
        self.router = Router(hidden_size, num_experts, top_k, **router_kw)
        self.experts = nn.ModuleList(SwiGLU(hidden_size, moe_intermediate_size) for _ in range(num_experts))
        self.shared_expert = (SwiGLU(hidden_size, moe_intermediate_size * num_shared_experts)
                              if num_shared_experts > 0 else None)

    def forward(self, x):                                           # (B, S, D) -> (y, aux)
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        idx, w, aux = self.router(x_flat, seq_len=S)
        N = x_flat.shape[0]
        flat_expert = idx.reshape(-1)                               # (N * k,) one entry per slot
        token_of_slot = torch.arange(N, device=x.device).repeat_interleave(self.top_k)
        out = x_flat.new_zeros(N * self.top_k, D)
        for e in range(self.num_experts):                           # each expert runs once, on its tokens
            sel = (flat_expert == e).nonzero(as_tuple=True)[0]
            if sel.numel():
                out[sel] = self.experts[e](x_flat[token_of_slot[sel]]).to(out.dtype)
        y = (out.view(N, self.top_k, D) * w.unsqueeze(-1)).sum(1).view(B, S, D)
        if self.shared_expert is not None:
            y = y + self.shared_expert(x)
        return y, aux


class ExpertBlock(nn.Module):
    """model.py's Block for layers past first_k_dense_replace: attention, then MoE."""

    def __init__(self, hidden_size: int, n_head: int, n_kv_head: int, head_dim: int,
                 moe_kw: dict, use_qk_norm: bool = True, eps: float = 1e-6,
                 attn_mscale: float = 1.0, window: Optional[int] = None):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.attn = GQAttention(hidden_size, n_head, n_kv_head, head_dim, use_qk_norm, eps, attn_mscale)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = MoE(hidden_size, **moe_kw)
        self.window = window

    def forward(self, x, cos, sin, cache=None, start: int = 0):
        x = x + self.attn(self.input_layernorm(x), cos, sin, cache=cache, start=start, window=self.window)
        h, aux = self.mlp(self.post_attention_layernorm(x))
        return x + h, aux


class MoELM(nn.Module):
    """The whole AnuLM (grouped-query attention). Defaults are the `tiny-gqa` preset."""

    def __init__(self, vocab_size: int = 259, block_size: int = 256, n_layer: int = 8, hidden_size: int = 384,
                 n_head: int = 6, head_dim: int = 64, n_kv_head: int = 2, use_qk_norm: bool = True,
                 num_experts: int = 16, num_experts_per_tok: int = 2, num_shared_experts: int = 1,
                 moe_intermediate_size: int = 96, intermediate_size: int = 768, first_k_dense_replace: int = 1,
                 n_group: int = 1, topk_group: int = 1, routed_scaling_factor: float = 2.5,
                 bias_update_rate: float = 1e-3, seq_balance_alpha: float = 0.0, router_z_alpha: float = 0.0,
                 rope_theta: float = 1e6, sliding_window: Optional[int] = None, max_window_layers: int = 0,
                 rms_norm_eps: float = 1e-6, initializer_range: float = 0.02):
        super().__init__()
        assert num_experts_per_tok <= (num_experts // n_group) * topk_group, \
            "top_k is more than the experts reachable through the allowed groups"
        self.block_size, self.bias_update_rate = block_size, bias_update_rate
        self.num_experts, self.num_experts_per_tok = num_experts, num_experts_per_tok
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary = Rotary(head_dim, rope_theta=rope_theta)
        moe_kw = dict(moe_intermediate_size=moe_intermediate_size, num_experts=num_experts,
                      top_k=num_experts_per_tok, num_shared_experts=num_shared_experts, n_group=n_group,
                      topk_group=topk_group, routed_scaling_factor=routed_scaling_factor,
                      seq_balance_alpha=seq_balance_alpha, router_z_alpha=router_z_alpha)
        layers = []
        for i in range(n_layer):
            window = sliding_window if i < max_window_layers else None
            if i < first_k_dense_replace:
                layers.append(Block(hidden_size, n_head, n_kv_head, head_dim, intermediate_size, use_qk_norm,
                                    rms_norm_eps, self.rotary.attn_mscale, window))
            else:
                layers.append(ExpertBlock(hidden_size, n_head, n_kv_head, head_dim, moe_kw, use_qk_norm,
                                          rms_norm_eps, self.rotary.attn_mscale, window))
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=initializer_range)
        for name, p in self.named_parameters():
            if name.endswith(("dense.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))

    def new_cache(self):
        return [dict() for _ in self.layers]

    def forward(self, idx, targets=None, cache=None, start: int = 0):
        B, S = idx.shape
        assert start + S <= self.block_size, f"positions up to {start + S} exceed block_size {self.block_size}"
        x = self.embed_tokens(idx)
        cos, sin = self.rotary(S, x.device, x.dtype, start=start)
        aux_total = None
        for i, layer in enumerate(self.layers):
            out = layer(x, cos, sin, cache=None if cache is None else cache[i], start=start)
            x, aux = out if isinstance(out, tuple) else (out, None)
            if aux is not None:
                aux_total = aux if aux_total is None else aux_total + aux
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)
        return logits, (loss + aux_total if aux_total is not None else loss)

    @torch.no_grad()
    def update_expert_biases(self) -> float:
        """After each optimizer step: push overloaded experts' bias down and starved ones' up
        by a fixed step. Returns the worst layer's max load / mean load (1.0 = perfect)."""
        ratios = []
        for m in self.modules():
            if isinstance(m, Router) and m.load_counts.sum() > 0:
                load, mean = m.load_counts, m.load_counts.mean()
                m.expert_bias += self.bias_update_rate * torch.sign(mean - load)
                ratios.append((load.max() / mean).item())
                load.zero_()
        return max(ratios) if ratios else 1.0

    def num_params(self):
        """(total, active per token): experts a token is not routed to do no work for it."""
        total = sum(p.numel() for p in self.parameters())
        inactive = sum(sum(p.numel() for p in layer.mlp.experts[0].parameters())
                       * (self.num_experts - self.num_experts_per_tok)
                       for layer in self.layers if isinstance(layer.mlp, MoE))
        return total, total - inactive
