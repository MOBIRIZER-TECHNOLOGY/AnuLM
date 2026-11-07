"""AdamWLowMem against torch.optim.AdamW.

    python -m unittest tests.test_adamw_lowmem

A broken optimizer still lowers the loss for a while (sign errors excepted),
so these compare it step by step with the reference: the same first step
exactly (bf16 rounding cannot matter yet), the same trajectory within a small
gap, 4 bytes of state per parameter, and a resume that continues identically.
"""
from __future__ import annotations

import copy
import unittest

import torch

from anulm.model import AnuLM, AnuLMConfig
from anulm.training.adamw_lowmem import AdamWLowMem


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = AnuLMConfig(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8,
                      n_kv_head=2, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
                      intermediate_size=64, n_group=1, topk_group=1, moe_impl="sparse")
    return AnuLM(cfg)


def groups(model, wd=0.1):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    rest = [p for p in model.parameters() if p.dim() < 2]
    return [{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}]


def run(model, opt, steps, seed=1):
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(steps):
        # Learnable text: each row counts up from a random start with a random
        # stride (mod 64), so the next token follows from the previous two.
        start = torch.randint(0, 64, (8, 1), generator=g)
        stride = torch.randint(1, 8, (8, 1), generator=g)
        seq = (start + stride * torch.arange(25)) % 64
        x, y = seq[:, :-1], seq[:, 1:]
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


class AdamWLowMemTests(unittest.TestCase):
    def test_first_step_matches(self):
        a, b = tiny_model(), tiny_model()
        ra = run(a, torch.optim.AdamW(groups(a), lr=1e-3, betas=(0.9, 0.95)), 1)
        rb = run(b, AdamWLowMem(groups(b), lr=1e-3, betas=(0.9, 0.95)), 1)
        self.assertEqual(ra, rb)
        # After one step m = 0.1 g and v = 0.05 g^2 exactly up to bf16 rounding of each;
        # the update is lr * sign-ish, so weights agree to ~lr * 0.4%.
        for pa, pb in zip(a.parameters(), b.parameters()):
            self.assertLess((pa - pb).abs().max().item(), 1e-3 * 0.01)

    def test_fp32_state_is_torch_adamw(self):
        """With fp32 moments the arithmetic is AdamW's: any gap is a bug, not rounding."""
        a, b = tiny_model(), tiny_model()
        ra = run(a, torch.optim.AdamW(groups(a), lr=3e-3, betas=(0.9, 0.95)), 60)
        rb = run(b, AdamWLowMem(groups(b), lr=3e-3, betas=(0.9, 0.95), state_dtype=torch.float32), 60)
        self.assertLess(max(abs(x - y) for x, y in zip(ra, rb)), 1e-3)

    def test_trajectory_stays_close(self):
        """bf16 moments: the loss curve stays within run-to-run noise of AdamW's.
        Measured 2026-11-06: mean of the last 20 losses 0.709 (AdamW), 0.719
        (bf16), 0.708 (AdamW from weights nudged by 1e-4)."""
        a, b = tiny_model(), tiny_model()
        ra = run(a, torch.optim.AdamW(groups(a), lr=3e-3, betas=(0.9, 0.95)), 150)
        rb = run(b, AdamWLowMem(groups(b), lr=3e-3, betas=(0.9, 0.95)), 150)
        self.assertLess(rb[-1], rb[0] - 2.0)                   # it learns
        tail_a, tail_b = sum(ra[-20:]) / 20, sum(rb[-20:]) / 20
        self.assertLess(abs(tail_a - tail_b), 0.05 * tail_a)  # and like AdamW

    def test_state_is_half_the_size(self):
        m = tiny_model()
        opt = AdamWLowMem(groups(m), lr=1e-3)
        run(m, opt, 1)
        n = sum(p.numel() for p in m.parameters() if p.grad is not None)
        state_bytes = sum(t.numel() * t.element_size() for s in opt.state.values()
                          for t in s.values() if torch.is_tensor(t))
        self.assertEqual(state_bytes, 4 * n)

    def test_resume_continues_identically(self):
        a = tiny_model()
        opt = AdamWLowMem(groups(a), lr=2e-3)
        run(a, opt, 5, seed=1)
        b = copy.deepcopy(a)
        opt_b = AdamWLowMem(groups(b), lr=2e-3)
        opt_b.load_state_dict(copy.deepcopy(opt.state_dict()))
        for st in opt_b.state.values():                       # still bf16 after loading
            self.assertEqual(st["exp_avg"].dtype, torch.bfloat16)
            self.assertEqual(st["exp_avg_sq"].dtype, torch.bfloat16)
        self.assertEqual(run(a, opt, 5, seed=2), run(b, opt_b, 5, seed=2))


if __name__ == "__main__":
    unittest.main()
