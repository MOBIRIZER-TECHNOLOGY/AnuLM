"""
Chapter 17 checks: distillation.

    python ch17.py                  # your course/build/further.py
    python ch17.py --impl solution  # the reference solution
    python ch17.py --self-test      # proves each check catches the bug it is for

A teacher that teaches the wrong thing still lowers the loss for a while: the
wrong next tokens kept, the KL turned around, a fade that never ends, soft
targets whose gradients shrink as the temperature rises.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, with_replaced  # noqa: E402
import bigram_teacher as ref  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


def corpus():
    g = torch.Generator().manual_seed(0)
    base = torch.randint(0, 40, (4000,), generator=g)
    base[1::2] = (base[::2][: len(base[1::2])] * 3 + 1) % 50          # strong pairs, so the bigram matters
    return base


@check
def the_bigram_teacher_matches_bigram_teacher_py(M):
    ids = corpus()
    got = M.build_bigram_teacher(ids, 50, k=8)
    want = ref.build_bigram_teacher(ids, 50, k=8, device="cpu")
    for name, a, b in zip(("unigram", "next tokens", "values"), got, want):
        assert torch.equal(a, b) if name == "next tokens" else torch.allclose(a, b, atol=1e-6), \
            f"the teacher's {name} differ from bigram_teacher.py"


@check
def the_kl_matches_bigram_teacher_py(M):
    ids = corpus()
    t = ref.build_bigram_teacher(ids, 50, k=8, device="cpu")
    torch.manual_seed(1)
    logits, x = torch.randn(2, 9, 50), ids[:18].reshape(2, 9)
    got, want = M.teacher_kl(logits, x, t), ref.teacher_kl(logits, x, t)
    assert abs(got.item() - want.item()) < 1e-5, f"KL {got.item():.5f}, bigram_teacher.py {want.item():.5f}"


@check
def the_teacher_fades_like_train_py(M):
    for step in (0, 100, 250, 499, 500, 750, 1000):
        got, want = M.teacher_weight(step, 1000, 2.0, 0.5), ref.teacher_weight(step, 1000, 2.0, 0.5)
        assert abs(got - want) < 1e-12, f"step {step}: weight {got}, train.py {want}"


def by_hand(s, t, y, T, alpha):
    keep = y.reshape(-1) != -1
    s, t, y = s.reshape(-1, s.shape[-1])[keep], t.reshape(-1, t.shape[-1])[keep], y.reshape(-1)[keep]
    pt, ls, lt = F.softmax(t / T, -1), F.log_softmax(s / T, -1), F.log_softmax(t / T, -1)
    soft = (pt * (lt - ls)).sum(-1).mean() * T * T
    return alpha * soft + (1 - alpha) * F.cross_entropy(s, y)


@check
def distillation_matches_the_formula(M):
    """alpha * T^2 * KL(teacher || student) at temperature T, plus (1 - alpha) * cross-entropy, on the
    positions that have a target."""
    torch.manual_seed(2)
    s, t = torch.randn(3, 5, 20), torch.randn(3, 5, 20) * 2
    y = torch.randint(0, 20, (3, 5))
    y[0, :2] = -1
    for T, alpha in ((1.0, 0.5), (4.0, 0.7), (2.0, 1.0)):
        got, want = M.kd_loss(s, t, y, T, alpha).item(), by_hand(s, t, y, T, alpha).item()
        assert abs(got - want) < 1e-5, f"T {T}, alpha {alpha}: {got:.5f}, want {want:.5f}"


@check
def only_the_student_learns(M):
    torch.manual_seed(3)
    s = torch.randn(2, 4, 10, requires_grad=True)
    t = torch.randn(2, 4, 10, requires_grad=True)
    M.kd_loss(s, t, torch.randint(0, 10, (2, 4)), 2.0, 0.5).backward()
    assert s.grad is not None and (t.grad is None or t.grad.abs().sum() == 0), "the teacher received a gradient"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("build_bigram_teacher", "teacher_kl", "teacher_weight", "kd_loss")


def MUTANTS(sol):
    def raw_bigram(train_ids, vocab_size, k=32):
        ids = train_ids.long()
        V = vocab_size
        log_uni, idx, val = sol.build_bigram_teacher(train_ids, vocab_size, k)
        uni = torch.bincount(ids, minlength=V).float() + 0.5
        cnt = torch.zeros(V, V)
        cnt.index_put_((ids[:-1], ids[1:]), torch.ones(len(ids) - 1), accumulate=True)
        p = cnt / cnt.sum(1, keepdim=True).clamp(min=1)
        val = torch.where(val != 0, (p.gather(1, idx).clamp(min=1e-9)).log() - log_uni[idx], val)
        return log_uni, idx, val

    def rarest_first(train_ids, vocab_size, k=32):
        log_uni, idx, val = sol.build_bigram_teacher(train_ids, vocab_size, vocab_size)     # every next token
        return log_uni, idx.flip(-1)[:, :k], val.flip(-1)[:, :k]

    def reversed_kl(logits, x, teacher):
        log_uni, idx, val = teacher
        z = log_uni.expand(*x.shape, -1).scatter_add(-1, idx[x], val[x])
        return F.kl_div(F.log_softmax(z, -1), F.log_softmax(logits.float(), -1),
                        log_target=True, reduction="none").sum(-1).mean()

    def never_stops(step, steps, weight, until):
        return weight * (1.0 - step / max(until * steps, 1))

    def no_t_squared(s, t, y, T=2.0, alpha=0.5):
        return sol.kd_loss(s, t, y, T, alpha) - alpha * (T * T - 1) * sol.kd_loss(s, t, y, T, 1.0) / (T * T)

    def teacher_only_warm(s, t, y, T=2.0, alpha=0.5):
        keep = y.reshape(-1) != -1
        sf, tf, yf = s.float().reshape(-1, s.size(-1))[keep], t.float().reshape(-1, t.size(-1))[keep].detach(), y.reshape(-1)[keep]
        soft = F.kl_div(F.log_softmax(sf, -1), F.log_softmax(tf / T, -1), log_target=True, reduction="batchmean") * T * T
        return alpha * soft + (1 - alpha) * F.cross_entropy(sf, yf)

    def everywhere(s, t, y, T=2.0, alpha=0.5):
        sf, tf = s.float().reshape(-1, s.size(-1)), t.float().reshape(-1, t.size(-1)).detach()
        soft = F.kl_div(F.log_softmax(sf / T, -1), F.log_softmax(tf / T, -1), log_target=True,
                        reduction="batchmean") * T * T
        return alpha * soft + (1 - alpha) * F.cross_entropy(sf, y.reshape(-1).clamp(min=0))

    def teacher_trains(s, t, y, T=2.0, alpha=0.5):
        keep = y.reshape(-1) != -1
        sf, tf, yf = s.float().reshape(-1, s.size(-1))[keep], t.float().reshape(-1, t.size(-1))[keep], y.reshape(-1)[keep]
        soft = F.kl_div(F.log_softmax(sf / T, -1), F.log_softmax(tf / T, -1), log_target=True,
                        reduction="batchmean") * T * T
        return alpha * soft + (1 - alpha) * F.cross_entropy(sf, yf)

    bugs = [("bigram counts with no unigram blend", dict(build_bigram_teacher=raw_bigram)),
            ("the rarest next tokens kept, not the most frequent", dict(build_bigram_teacher=rarest_first)),
            ("KL the wrong way round", dict(teacher_kl=reversed_kl)),
            ("a fade that goes negative", dict(teacher_weight=never_stops)),
            ("soft loss without T squared", dict(kd_loss=no_t_squared)),
            ("temperature on the teacher only", dict(kd_loss=teacher_only_warm)),
            ("positions without a target distilled too", dict(kd_loss=everywhere)),
            ("the teacher not detached", dict(kd_loss=teacher_trains))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(17, CHECKS, MUTANTS, __doc__, learner="further.py")
