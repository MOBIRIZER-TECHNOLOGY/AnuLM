"""
Reference solution for Chapter 17: distillation.

build_bigram_teacher, teacher_kl and teacher_weight match bigram_teacher.py
(train.py --bigram-teacher), the repository's teacher with no teacher model.
kd_loss is classic distillation from a model teacher (Hinton et al., 2015),
which the repository does not have yet. course/checks/ch17.py proves them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_bigram_teacher(train_ids: torch.Tensor, vocab_size: int, k: int = 32):
    """From one count pass: unigram log-probs (V,), and for every previous token its k most frequent
    next tokens with log P(next | prev) - log P(next). The bigram is blended with the unigram by
    n / (n + 5), so a previous token seen only a few times leans on the unigram."""
    V = vocab_size
    ids = train_ids.long()
    uni = torch.bincount(ids, minlength=V).float() + 0.5
    log_uni = (uni / uni.sum()).log()
    pairs = ids[:-1] * V + ids[1:]
    keys, cnt = torch.unique(pairs, return_counts=True)
    prev, nxt = keys // V, keys % V
    n = torch.bincount(ids[:-1], minlength=V).float()[prev]
    lam = n / (n + 5.0)
    p_bi = lam * cnt.float() / n.clamp(min=1) + (1 - lam) * uni[nxt] / uni.sum()
    resid = p_bi.log() - log_uni[nxt]
    order = torch.argsort(prev * (cnt.max() + 1) - cnt, stable=True)          # by prev, then most frequent first
    prev_s, nxt_s, res_s = prev[order], nxt[order], resid[order]
    start = torch.searchsorted(prev_s, torch.arange(V))
    rank = torch.arange(len(prev_s)) - start[prev_s]
    keep = rank < k
    idx = torch.zeros(V, k, dtype=torch.long)
    val = torch.zeros(V, k)
    idx[prev_s[keep], rank[keep]] = nxt_s[keep]
    val[prev_s[keep], rank[keep]] = res_s[keep]
    return log_uni, idx, val


def teacher_kl(logits: torch.Tensor, x: torch.Tensor, teacher) -> torch.Tensor:
    """Mean over positions of KL(teacher's next-token distribution after x_t || the model's)."""
    log_uni, idx, val = teacher
    z = log_uni.expand(*x.shape, -1).scatter_add(-1, idx[x], val[x])
    return F.kl_div(F.log_softmax(logits.float(), -1), F.log_softmax(z, -1),
                    log_target=True, reduction="none").sum(-1).mean()


def teacher_weight(step: int, steps: int, weight: float, until: float) -> float:
    """Linear fade from `weight` at step 0 to 0 at until * steps, and 0 after."""
    return weight * max(0.0, 1.0 - step / max(until * steps, 1))


def kd_loss(student_logits, teacher_logits, targets, T: float = 2.0, alpha: float = 0.5):
    """alpha * T^2 * KL(teacher at temperature T || student at temperature T)
    + (1 - alpha) * cross-entropy with the targets (positions with target -1 ignored by both).
    The T^2 keeps the soft part's gradients the same size whatever T is."""
    mask = targets.reshape(-1) != -1
    s = student_logits.float().reshape(-1, student_logits.size(-1))[mask]
    t = teacher_logits.float().reshape(-1, teacher_logits.size(-1))[mask].detach()
    soft = F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True,
                    reduction="batchmean") * T * T
    hard = F.cross_entropy(s, targets.reshape(-1)[mask])
    return alpha * soft + (1 - alpha) * hard
