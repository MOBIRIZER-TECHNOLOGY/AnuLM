"""
Reference solution for Chapter 11: preference tuning with DPO.

Direct Preference Optimization (Rafailov et al., 2023) trains on pairs of
answers to one prompt, one preferred ("chosen") and one not ("rejected"). It
needs no reward model and no sampling during training: only the log-probability
of each answer under the model being trained and under a frozen copy of where
it started (the reference).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def answer_logprobs(model, prompts, answers, pad: int = 0) -> torch.Tensor:
    """Sum of log p(answer id | everything before it) for each (prompt, answer) pair, as a (B,)
    tensor. Prompt ids are read but not counted. Rows are right-padded to one length; padding is
    never counted either, and since attention is causal it cannot change the earlier positions."""
    seqs = [list(p) + list(a) for p, a in zip(prompts, answers)]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    mask = torch.zeros(len(seqs), T - 1)
    for i, (p, a) in enumerate(zip(prompts, answers)):
        ids[i, :len(seqs[i])] = torch.tensor(seqs[i])
        mask[i, len(p) - 1: len(p) - 1 + len(a)] = 1.0       # position t predicts id t + 1
    logits, _ = model(ids[:, :-1])
    logp = F.log_softmax(logits.float(), dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return (logp * mask).sum(-1)


def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta: float = 0.1):
    """-log sigmoid(beta * ((pol_chosen - ref_chosen) - (pol_rejected - ref_rejected))), averaged.
    Returns (loss, reward margin per pair). The reference terms carry no gradient."""
    margin = beta * ((pol_chosen - ref_chosen.detach()) - (pol_rejected - ref_rejected.detach()))
    return -F.logsigmoid(margin).mean(), margin.detach()


def dpo_step(policy, reference, opt, prompts, chosen, rejected, beta: float = 0.1, clip: float = 1.0,
             sft_weight: float = 0.0):
    """One optimizer step of DPO on a batch of preference pairs. The reference is a frozen copy of
    the policy's starting point, evaluated without gradients. sft_weight > 0 adds that much of the
    chosen answers' ordinary fine-tuning loss (mean negative log-probability per answer id), which
    holds the chosen answers up while DPO pushes the rejected ones down. Returns (loss, mean
    margin, share of pairs where the policy already prefers the chosen answer more than the
    reference does)."""
    with torch.no_grad():
        ref_c = answer_logprobs(reference, prompts, chosen)
        ref_r = answer_logprobs(reference, prompts, rejected)
    pol_c = answer_logprobs(policy, prompts, chosen)
    pol_r = answer_logprobs(policy, prompts, rejected)
    loss, margin = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
    if sft_weight:
        loss = loss + sft_weight * -pol_c.sum() / sum(len(c) for c in chosen)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), clip)
    opt.step()
    if hasattr(policy, "update_expert_biases"):              # Chapter 8: the balancer moves after every step
        policy.update_expert_biases()
    return loss.item(), margin.mean().item(), (margin > 0).float().mean().item()
