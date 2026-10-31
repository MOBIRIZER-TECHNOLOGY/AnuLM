"""
Chapter 19 checks: reinforcement learning with checkable rewards.

    python ch19.py                  # your course/build/further.py
    python ch19.py --impl solution  # the reference solution
    python ch19.py --self-test      # proves each check catches the bug it is for

Reinforcement learning optimises exactly what you reward, the way you compute
it. A checker that accepts a wrong answer, advantages that push on groups with
nothing to learn, a KL that can go negative: each one trains, towards
something else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, with_replaced  # noqa: E402

CHECKS = []


def check(fn):
    CHECKS.append(fn)
    return fn


@check
def the_checker_rewards_only_the_right_answer(M):
    for reply, ok in ((" 68", 1.0), ("68", 1.0), (" 68 ", 1.0), (" 68 apples", 0.0), (" 6", 0.0),
                      (" 068", 0.0), (" 686", 0.0), ("", 0.0)):
        assert M.reward(reply, "68") == ok, f"reward({reply!r}, '68') = {M.reward(reply, '68')}, want {ok}"


@check
def advantages_are_relative_to_the_group(M):
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    a = M.group_advantages(r)
    assert torch.allclose(a.mean(1), torch.zeros(3), atol=1e-5), "advantages don't average to 0 within each group"
    assert torch.allclose(a.std(1), torch.ones(3), atol=1e-3), "advantages are not scaled by the group's spread"
    assert a[1, 3] < 0 < a[1, 0] and a[2, 2] > a[0, 0] > 0, "the wrong answers don't get the negative advantages"


@check
def a_group_that_all_agree_teaches_nothing(M):
    r = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
    a = M.group_advantages(r)
    assert torch.all(a[:2] == 0), f"all-right and all-wrong groups get advantages {a[:2].tolist()}; want 0"


@check
def the_loss_matches_the_formula(M):
    logp = torch.tensor([-1.0, -3.0, -0.5, -6.0])
    ref = torch.tensor([-1.2, -2.0, -0.5, -5.0])
    n = torch.tensor([2.0, 3.0, 1.0, 4.0])
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    d = (ref - logp) / n
    want = -(adv * logp / n).mean() + 0.1 * (torch.exp(d) - d - 1).mean()
    got = M.grpo_loss(logp, ref, n, adv, 0.1)
    assert abs(got.item() - want.item()) < 1e-6, f"loss {got.item():.6f}, want {want.item():.6f}"


@check
def the_kl_is_zero_at_the_start_and_never_negative(M):
    logp = torch.tensor([-1.0, -3.0, -0.5])
    n = torch.tensor([2.0, 3.0, 1.0])
    zero = torch.zeros(3)
    base = M.grpo_loss(logp, logp.clone(), n, zero, 1.0)
    assert abs(base.item()) < 1e-7, f"policy = reference, no advantage: loss {base.item()}, want 0"
    for shift in (-2.0, -0.3, 0.4, 3.0):
        kl = M.grpo_loss(logp, logp + shift, n, zero, 1.0)
        assert kl.item() > 0, f"KL {kl.item():.4f} for a shift of {shift}: it must be positive"


@check
def the_gradient_raises_the_better_answers(M):
    logp = torch.tensor([-2.0, -2.0, -2.0, -2.0], requires_grad=True)
    adv = M.group_advantages(torch.tensor([[1.0, 0.0, 1.0, 0.0]]))[0]
    M.grpo_loss(logp, logp.detach().clone(), torch.ones(4), adv, 0.04).backward()
    assert logp.grad[0] < 0 < logp.grad[1], "gradient descent would make the WRONG answers more likely"
    ref = torch.tensor([-2.0, -2.0, -2.0, -2.0], requires_grad=True)
    lp = torch.tensor([-1.0, -1.0, -3.0, -3.0], requires_grad=True)
    M.grpo_loss(lp, ref, torch.ones(4), torch.zeros(4), 0.5).backward()
    assert ref.grad is None or ref.grad.abs().sum() == 0, "the reference received a gradient"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("reward", "group_advantages", "grpo_loss")


def MUTANTS(sol):
    def startswith(reply, answer):
        return 1.0 if reply.strip().startswith(answer) else 0.0

    def contains(reply, answer):
        return 1.0 if answer in reply else 0.0

    def raw(rewards, eps=1e-4):
        return rewards - 0.5

    def batch_mean(rewards, eps=1e-4):
        return (rewards - rewards.mean()) / (rewards.std() + eps)

    def mean_only(rewards, eps=1e-4):
        return rewards - rewards.mean(dim=1, keepdim=True)

    def flipped(logp, ref_logp, lengths, advantages, beta=0.04):
        return sol.grpo_loss(logp, ref_logp, lengths, -advantages, beta)

    def plain_log_ratio(logp, ref_logp, lengths, advantages, beta=0.04):
        pg = -(advantages.detach() * logp / lengths).mean()
        return pg + beta * ((logp - ref_logp.detach()) / lengths).mean()

    def no_length(logp, ref_logp, lengths, advantages, beta=0.04):
        return sol.grpo_loss(logp, ref_logp, torch.ones_like(lengths), advantages, beta)

    def ref_trains(logp, ref_logp, lengths, advantages, beta=0.04):
        d = (ref_logp - logp) / lengths
        return -(advantages.detach() * logp / lengths).mean() + beta * (torch.exp(d) - d - 1).mean()

    bugs = [("a reply that starts with the answer is rewarded", dict(reward=startswith)),
            ("a reply that contains the answer is rewarded", dict(reward=contains)),
            ("raw rewards as advantages", dict(group_advantages=raw)),
            ("normalised over the batch, not the group", dict(group_advantages=batch_mean)),
            ("centred but not scaled", dict(group_advantages=mean_only)),
            ("advantage sign flipped", dict(grpo_loss=flipped)),
            ("KL as a plain log ratio (can go negative)", dict(grpo_loss=plain_log_ratio)),
            ("no division by answer length", dict(grpo_loss=no_length)),
            ("the reference not detached", dict(grpo_loss=ref_trains))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(19, CHECKS, MUTANTS, __doc__, learner="further.py")
