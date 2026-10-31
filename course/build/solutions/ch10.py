"""
Reference solution for Chapter 10: LoRA and QLoRA on the real model.

LoRALinear, apply_lora and merge_lora match lora.py exactly (same names, same
initialisation, same arithmetic); adapter_state is lora.py's adapters plus its
router_state_dict, which an MoE model needs (see the chapter). The NF4 code is
Chapter 10's from-scratch QLoRA, now applied to the model you built.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("attn.query_key_value", "attn.dense")     # lora.py's: the attention projections


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable rank-r correction: base(x) + (x A^T) B^T * alpha / r."""

    def __init__(self, base: nn.Module, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))     # zero: start as the base
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + ((x @ self.lora_a.t().to(x.dtype)) @ self.lora_b.t().to(x.dtype)) * self.scale

    @torch.no_grad()
    def merged_weight(self):
        return self.base.weight + (self.lora_b @ self.lora_a) * self.scale


def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, targets=DEFAULT_TARGETS,
               quantize: bool = False) -> int:
    """Freeze every parameter, then wrap each Linear whose full name ends with a target.
    With quantize=True the frozen Linear is first stored in 4-bit NF4 (QLoRA).
    Returns how many layers were wrapped."""
    for p in model.parameters():
        p.requires_grad_(False)
    hits = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in targets):
                setattr(module, child_name, LoRALinear(NF4Linear(child) if quantize else child, r, alpha))
                hits += 1
    if not hits:
        raise ValueError(f"no Linear matched {targets}; nothing would train")
    return hits


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Fold every adapter into its base weight and put the plain Linear back."""
    merged = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                base = child.base
                base.weight.copy_(child.merged_weight())
                setattr(module, child_name, base)
                merged += 1
    return merged


def adapter_state(model: nn.Module) -> dict:
    """What you keep after LoRA: the adapters, AND the experts' balancing biases, which
    kept moving during training although no parameter outside the adapters did."""
    state = {n: t.detach().clone() for n, t in model.state_dict().items() if "lora_" in n}
    state.update({n: b.detach().clone() for n, b in model.named_buffers() if n.endswith("expert_bias")})
    return state


# --- QLoRA: the frozen weights in 4-bit NormalFloat ---------------------------------------

NF4 = torch.tensor([-1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0,
                    0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0])
BLOCK = 64


def nf4_quantize(w: torch.Tensor):
    """(two 4-bit codes per byte, one scale per block of 64): each weight becomes its nearest NF4 level
    after dividing its block by the block's largest absolute value."""
    flat = w.detach().float().reshape(-1, BLOCK)
    absmax = flat.abs().max(dim=1, keepdim=True).values.clamp_min(1e-12)
    codes = torch.bucketize(flat / absmax, (NF4[1:] + NF4[:-1]) / 2).to(torch.uint8).reshape(-1)
    return (codes[0::2] << 4) | codes[1::2], absmax.squeeze(1)


def nf4_dequantize(packed, absmax, shape):
    codes = torch.stack([packed >> 4, packed & 15], dim=1).reshape(-1).long()
    return (NF4[codes].reshape(-1, BLOCK) * absmax[:, None]).reshape(shape)


class NF4Linear(nn.Module):
    """A frozen Linear stored in 4 bits; it rebuilds its weight on every forward pass."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.in_features, self.out_features, self.shape = base.in_features, base.out_features, tuple(base.weight.shape)
        packed, absmax = nf4_quantize(base.weight)
        self.register_buffer("packed", packed)
        self.register_buffer("absmax", absmax)
        self.register_buffer("bias", None if base.bias is None else base.bias.detach().clone())

    @property
    def weight(self):
        return nf4_dequantize(self.packed, self.absmax, self.shape)

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias)
