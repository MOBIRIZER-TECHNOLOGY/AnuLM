"""AdamW with its two moments stored in bf16: 12 bytes per parameter, not 16.

For a 1B-parameter model on a 16 GB card (docs/PLAN_1B.md). Plain AdamW keeps
fp32 weights, fp32 gradients and two fp32 moments: 16 bytes per parameter,
16 GB for 1B, the whole card before a single activation. Muon keeps one moment
(12 bytes) but measured 2.6x slower per step on this GPU (muon.py). Storing the
moments in bf16 also gives 12 bytes and keeps AdamW's arithmetic: each step
reads a moment into fp32, updates it, does the AdamW step in fp32, and writes
the moment back rounded to bf16. The weights stay fp32.

What bf16 costs: 8 bits of mantissa, so a moment is held to about 0.4%.
The second moment is an average of squares that changes slowly, and rounding
it every step biases it by at most that much; the update divides by its
square root, about 0.2% in step size. tests/test_adamw_lowmem.py measures the
gap to torch.optim.AdamW on a small regression and a tiny AnuLM.

Same interface as torch.optim.AdamW (lr, betas, eps, weight_decay, param
groups, state_dict / load_state_dict), so train.py's schedule, which sets
group["lr"], works unchanged.
"""
from __future__ import annotations

import math

import torch


class AdamWLowMem(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                 state_dtype=torch.bfloat16):
        if lr < 0 or eps <= 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("invalid AdamW hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.state_dtype = state_dtype

    def load_state_dict(self, state_dict):
        """torch's loader casts floating state to each parameter's dtype (fp32):
        a resumed run would silently go back to 16 bytes per parameter and to
        fp32 moments. Cast them back to the storage dtype."""
        super().load_state_dict(state_dict)
        for st in self.state.values():
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st and st[k].dtype != self.state_dtype:
                    st[k] = st[k].to(self.state_dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("AdamWLowMem does not support sparse gradients")
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg"] = torch.zeros_like(p, dtype=self.state_dtype)
                    st["exp_avg_sq"] = torch.zeros_like(p, dtype=self.state_dtype)
                st["step"] += 1
                t = st["step"]
                g = p.grad.float()
                m = st["exp_avg"].float().mul_(b1).add_(g, alpha=1 - b1)
                v = st["exp_avg_sq"].float().mul_(b2).addcmul_(g, g, value=1 - b2)
                st["exp_avg"].copy_(m)
                st["exp_avg_sq"].copy_(v)
                if wd:
                    p.mul_(1 - lr * wd)                      # decoupled, as torch.optim.AdamW
                bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
                # Not sqrt_: with fp32 state, v is the stored moment itself, not a copy.
                denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss
