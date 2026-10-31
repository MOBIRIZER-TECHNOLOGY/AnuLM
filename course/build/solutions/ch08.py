"""
Reference solution for Chapter 8: generation, and the machinery of pre-training.

MoELM.generate matches model.py's AnuLM.generate token for token; lr_at and
make_optimizer match train.py's. accumulate, train_step, save_checkpoint and
load_checkpoint are the training loop's moving parts, small enough to read
whole, and course/checks/ch08.py holds them to exact gradient accumulation and
exact resume.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ch07  # noqa: E402
from ch07 import Router, MoE, ExpertBlock  # noqa: E402,F401


class MoELM(ch07.MoELM):
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k=None, eos_id=None,
                 use_cache: bool = True, repetition_penalty: float = 1.0):
        """Sample max_new_tokens tokens after idx (B, T): the same steps, in the same order, as
        model.py's AnuLM.generate, so the same random seed gives the same text."""
        B, bs = idx.shape[0], self.block_size
        done = torch.zeros(B, dtype=torch.bool, device=idx.device)
        cache, start, pending = None, 0, idx
        for _ in range(max_new_tokens):
            if not use_cache:
                logits, _ = self(idx[:, -bs:])
            else:
                if cache is None or start + pending.shape[1] > bs:      # full: rebuild from the last bs tokens
                    cache, start, pending = self.new_cache(), 0, idx[:, -bs:]
                logits, _ = self(pending, cache=cache, start=start)
                start += pending.shape[1]
            logits = logits[:, -1, :].float()
            if repetition_penalty != 1.0:                             # CTRL's rule, on every token so far
                seen = logits.gather(1, idx)
                seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
                logits = logits.scatter(1, idx, seen)
            logits = logits / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            if eos_id is not None:                                    # finished rows keep receiving EOS
                nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt)
            idx = torch.cat((idx, nxt), dim=1)
            pending = nxt
            if eos_id is not None:
                done |= nxt[:, 0] == eos_id
                if bool(done.all()):
                    break
        return idx


def lr_at(step: int, steps: int, lr: float, min_lr: float, warmup: int, schedule: str = "cosine",
          decay_frac: float = 0.2) -> float:
    """Linear warmup, then cosine down to min_lr, or warmup-stable-decay: flat at lr, then a
    straight line down to min_lr over the last decay_frac of the run."""
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    if schedule == "wsd":
        decay_start = int(steps * (1 - decay_frac))
        if step < decay_start:
            return lr
        return lr + (min_lr - lr) * min((step - decay_start) / max(steps - decay_start, 1), 1.0)
    progress = min((step - warmup) / max(steps - warmup, 1), 1.0)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


def make_optimizer(model, lr: float, weight_decay: float = 0.1):
    """AdamW. Weight decay on the matrices only: never on norms' gains or biases."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=lr, betas=(0.9, 0.95), eps=1e-8)


def accumulate(model, batches) -> float:
    """Gradients of the MEAN loss over several micro-batches, as if they were one big batch.
    Returns that mean loss."""
    model.zero_grad(set_to_none=True)
    total = 0.0
    for x, y in batches:
        _, loss = model(x, y)
        (loss / len(batches)).backward()                  # each micro-batch's share of the mean
        total += loss.item() / len(batches)
    return total


def train_step(model, opt, batches, lr: float, clip: float = 1.0) -> float:
    """One optimizer step over a list of (x, y) micro-batches."""
    for g in opt.param_groups:
        g["lr"] = lr
    loss = accumulate(model, batches)
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    model.update_expert_biases()                          # after the step, never part of the loss
    return loss


def save_checkpoint(path, model, opt, step: int, **extra) -> None:
    """Everything needed to continue exactly: weights AND buffers (the experts' balancing bias),
    the optimizer's moments, the step, and whatever else (a data-order generator's state).
    Written beside the target and renamed into place, so a crash mid-write cannot destroy
    the previous checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, **extra}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, opt) -> dict:
    """Restore model and optimizer in place; return the checkpoint (its "step" is the last
    step completed, so training continues at step + 1)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    return ck
