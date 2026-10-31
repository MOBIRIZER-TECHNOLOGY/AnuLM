"""
Reference solution for Chapter 19: reinforcement learning with checkable rewards.

GRPO (Shao et al., 2024, DeepSeekMath): sample a group of answers per prompt,
score each with a program, and push each answer's probability up or down by
how much better it did than its own group. No value model, no reward model.
The repository has no RL code; course/checks/ch19.py holds these pieces to the
formulas and to the bugs that break them quietly.
"""

from __future__ import annotations

import torch


def reward(reply: str, answer: str) -> float:
    """1 if the reply, with surrounding spaces removed, is exactly the right answer, else 0."""
    return 1.0 if reply.strip() == answer else 0.0


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """rewards: (prompts, group). Each answer's reward minus its group's mean, divided by its group's
    standard deviation. A group whose answers all scored the same carries no information: 0."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return torch.where(std > 0, (rewards - mean) / (std + eps), torch.zeros_like(rewards))


def grpo_loss(logp, ref_logp, lengths, advantages, beta: float = 0.04):
    """logp, ref_logp: the summed log-probability of each sampled answer under the policy and under
    the frozen reference; lengths: answer ids per answer; advantages: from group_advantages.
    Policy-gradient term: -mean(advantage * logp / length). KL term: beta * mean(exp(d) - d - 1),
    d = (ref_logp - logp) / length, which is 0 when the two agree and never negative."""
    per_id = logp / lengths
    pg = -(advantages.detach() * per_id).mean()
    d = (ref_logp.detach() - logp) / lengths
    kl = (torch.exp(d) - d - 1).mean()
    return pg + beta * kl
