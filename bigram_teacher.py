"""
A free teacher: the corpus's own bigram statistics as soft targets, faded out.

    python train.py ... --bigram-teacher 2.0 --teacher-until 0.5

Early in training the model is told not just "the next token is X" but what
the whole next-token distribution looks like after the previous token, as
counted over the training split. That is distillation without a teacher
model: one count pass over the data, a (V, K) table on the GPU, and a KL term
whose weight falls linearly to zero by `--teacher-until` of the run.

Measured on the 3-language proxy (docs/RESULTS.md, "the lab"): 25M tokens,
held-out loss averaged over Hindi / English / code windows --

    baseline                         4.946  (second seed 5.012)
    bigram teacher, weight 1.0       4.728  (seed 1: 4.796, i.e. -0.216 on that seed)
    bigram teacher, weight 2.0       4.692
    same, faded at 80% not 50%       4.760  -- it must step aside
    bigram ADDED to the logits       5.473  -- the same statistics as a fixed
                                             prior hurt: the model has to un-learn
                                             them wherever they are wrong

The distinction in the last row is the point: soft targets carry information
about every vocabulary entry at every position early on, then get out of the
way; a permanent prior never does.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_bigram_teacher(train_ids: torch.Tensor, vocab_size: int, k: int = 32,
                         device: str = "cuda"):
    """Unigram log-probs (V,) and, per previous token, the top-k next tokens
    with log P(next | prev) - log P(next), from counts over `train_ids`.

    The bigram is interpolated with the unigram by n / (n + 5), so a context
    seen a handful of times leans on the unigram instead of its few counts.
    """
    V = vocab_size
    ids = train_ids.to(device).long()
    uni = torch.bincount(ids, minlength=V).float() + 0.5
    log_uni = (uni / uni.sum()).log()
    pairs = ids[:-1] * V + ids[1:]
    keys, cnt = torch.unique(pairs, return_counts=True)
    prev, nxt = keys // V, keys % V
    n = torch.bincount(ids[:-1], minlength=V).float()[prev]
    lam = n / (n + 5.0)
    p_bi = lam * cnt.float() / n.clamp(min=1) + (1 - lam) * uni[nxt] / uni.sum()
    resid = p_bi.log() - log_uni[nxt]
    order = torch.argsort(prev * (cnt.max() + 1) - cnt, stable=True)       # by prev, then count desc
    prev_s, nxt_s, res_s = prev[order], nxt[order], resid[order]
    start = torch.searchsorted(prev_s, torch.arange(V, device=device))
    rank = torch.arange(len(prev_s), device=device) - start[prev_s]
    keep = rank < k
    idx = torch.zeros(V, k, dtype=torch.long, device=device)
    val = torch.zeros(V, k, device=device)
    idx[prev_s[keep], rank[keep]] = nxt_s[keep]
    val[prev_s[keep], rank[keep]] = res_s[keep]
    return log_uni, idx, val


def teacher_kl(logits: torch.Tensor, x: torch.Tensor, teacher) -> torch.Tensor:
    """Mean over positions of KL(p_bigram(. | x_t) || p_model(. | x_<=t))."""
    log_uni, idx, val = teacher
    z = log_uni.expand(*x.shape, -1).scatter_add(-1, idx[x], val[x])
    return F.kl_div(F.log_softmax(logits.float(), -1), F.log_softmax(z, -1),
                    log_target=True, reduction="none").sum(-1).mean()


def teacher_weight(step: int, steps: int, weight: float, until: float) -> float:
    """Linear fade from `weight` at step 0 to 0 at `until` * steps."""
    return weight * max(0.0, 1.0 - step / max(until * steps, 1))
