"""
Chapter 11 checks: preference tuning with DPO.

    python ch11.py                  # your course/build/post.py
    python ch11.py --impl solution  # the reference solution
    python ch11.py --self-test      # proves each check catches the bug it is for

A wrong DPO still lowers its own loss. It just optimises something else: the
question's probability, the padding, a margin with the sign flipped. These
checks hold each piece to what it is for.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, solution, with_replaced  # noqa: E402

CHECKS = []
KW = dict(vocab_size=64, block_size=32, n_layer=2, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
          num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
          rope_theta=10000.0)
B = solution(8)
PROMPTS = [[5, 6, 7, 8], [9, 10], [11, 12, 13, 14, 15, 16], [17]]
ANSWERS = [[20, 21], [22, 23, 24, 25, 26], [27], [28, 29, 30]]


def check(fn):
    CHECKS.append(fn)
    return fn


def model():
    torch.manual_seed(0)
    return B.MoELM(**KW).eval()


def by_hand(m, p, a):
    """One pair, no padding: the log-probability of each answer id, added up."""
    ids = torch.tensor([p + a])
    with torch.no_grad():
        logp = F.log_softmax(m(ids[:, :-1])[0].float(), -1)[0]
    return sum(logp[len(p) - 1 + i, t].item() for i, t in enumerate(a))


@check
def answer_logprobs_counts_the_answer_only(M):
    """The sum over the answer's ids, each given everything before it: not the prompt, not a mean."""
    m = model()
    for p, a in zip(PROMPTS, ANSWERS):
        with torch.no_grad():
            got = M.answer_logprobs(m, [p], [a]).item()
        want = by_hand(m, p, a)
        assert abs(got - want) < 1e-4, f"prompt {p}: {got:.4f}, want {want:.4f} (the answer's ids only, summed)"


@check
def padding_changes_nothing(M):
    """Pairs of different lengths in one batch give what each gives alone."""
    m = model()
    with torch.no_grad():
        together = M.answer_logprobs(m, PROMPTS, ANSWERS)
        alone = torch.cat([M.answer_logprobs(m, [p], [a]) for p, a in zip(PROMPTS, ANSWERS)])
    torch.testing.assert_close(together, alone, rtol=1e-4, atol=1e-4,
                               msg="a batch of different lengths differs from each pair alone: padding leaks in")


@check
def loss_is_ln2_when_policy_is_the_reference(M):
    """At the start the policy is the reference: every margin is 0 and the loss is ln 2. Its
    gradient is NOT zero: it pushes chosen up and rejected down from the first step."""
    c = torch.tensor([-3.0, -5.0, -1.5], requires_grad=True)
    r = torch.tensor([-2.0, -6.0, -4.0], requires_grad=True)
    loss, margin = M.dpo_loss(c, r, c.detach(), r.detach(), 0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6, f"loss {loss.item():.4f} at the start, want ln 2 = {math.log(2):.4f}"
    assert torch.all(margin == 0), "margins are not zero when the policy is the reference"
    loss.backward()
    want = -0.1 * 0.5 / 3
    assert torch.allclose(c.grad, torch.full((3,), want)) and torch.allclose(r.grad, torch.full((3,), -want)), \
        f"gradient {c.grad.tolist()} on chosen; want {want:.4f} each (beta / 2, averaged)"


@check
def dpo_loss_matches_the_formula(M):
    pc, pr = torch.tensor([-2.0, -9.0]), torch.tensor([-6.0, -3.0])
    rc, rr = torch.tensor([-4.0, -8.0]), torch.tensor([-5.0, -5.0])
    for beta in (0.1, 0.5):
        m = [beta * ((-2 + 4) - (-6 + 5)), beta * ((-9 + 8) - (-3 + 5))]
        want = sum(-math.log(1 / (1 + math.exp(-v))) for v in m) / 2
        loss, margin = M.dpo_loss(pc, pr, rc, rr, beta)
        assert abs(loss.item() - want) < 1e-5, f"beta {beta}: loss {loss.item():.5f}, want {want:.5f}"
        assert torch.allclose(margin, torch.tensor(m)), f"beta {beta}: margins {margin.tolist()}, want {m}"


@check
def the_reference_gets_no_gradient(M):
    ref_c = torch.tensor([-3.0, -1.0], requires_grad=True)
    ref_r = torch.tensor([-2.0, -4.0], requires_grad=True)
    pol_c = torch.tensor([-2.5, -1.5], requires_grad=True)
    pol_r = torch.tensor([-2.5, -3.0], requires_grad=True)
    loss, _ = M.dpo_loss(pol_c, pol_r, ref_c, ref_r, 0.1)
    loss.backward()
    assert ref_c.grad is None and ref_r.grad is None, "the reference received a gradient; it must stay frozen"


@check
def a_few_steps_prefer_the_chosen_answer(M):
    """After DPO steps on fixed pairs, the policy prefers each chosen answer more than the
    reference does, measured independently of dpo_step's own report."""
    policy = model().train()
    reference = copy.deepcopy(policy).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    chosen = ANSWERS
    rejected = [[a[0] + 1] + a[1:] for a in ANSWERS]                      # the first answer id wrong
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-2)
    for _ in range(8):
        M.dpo_step(policy, reference, opt, PROMPTS, chosen, rejected, 0.5)
    policy.eval()
    with torch.no_grad():
        gain_c = M.answer_logprobs(policy, PROMPTS, chosen) - M.answer_logprobs(reference, PROMPTS, chosen)
        gain_r = M.answer_logprobs(policy, PROMPTS, rejected) - M.answer_logprobs(reference, PROMPTS, rejected)
    assert torch.all(gain_c > gain_r), f"chosen gained {gain_c.tolist()}, rejected {gain_r.tolist()}: wrong way round"


@check
def the_step_moves_the_balancer(M):
    """DPO trains an MoE model like any other step: the experts' biases follow the load."""
    policy = model().train()
    reference = copy.deepcopy(policy).eval()
    before = [b.clone() for n, b in policy.named_buffers() if n.endswith("expert_bias")]
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    M.dpo_step(policy, reference, opt, PROMPTS, ANSWERS, [[a[0] + 1] + a[1:] for a in ANSWERS])
    after = [b for n, b in policy.named_buffers() if n.endswith("expert_bias")]
    assert any(not torch.equal(x, y) for x, y in zip(before, after)), "the experts' biases never moved"


@check
def the_sft_term_is_the_chosen_answers_mean_loss(M):
    """dpo_step's sft_weight adds weight x (mean negative log-probability per chosen answer id)."""
    base = model()
    reference = copy.deepcopy(base).eval()
    rejected = [[a[0] + 1] + a[1:] for a in ANSWERS]
    nll = -sum(by_hand(base, p, a) for p, a in zip(PROMPTS, ANSWERS)) / sum(len(a) for a in ANSWERS)
    got = []
    for w in (0.0, 0.5):
        policy = copy.deepcopy(base)                                  # eval mode: no dropout, same numbers
        opt = torch.optim.SGD(policy.parameters(), lr=0.0)
        got.append(M.dpo_step(policy, reference, opt, PROMPTS, ANSWERS, rejected, 0.1, sft_weight=w)[0])
    assert abs((got[1] - got[0]) - 0.5 * nll) < 1e-4,         f"sft_weight 0.5 added {got[1] - got[0]:.4f} to the loss; want 0.5 x {nll:.4f} (mean loss per chosen id)"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("answer_logprobs", "dpo_loss", "dpo_step")


def MUTANTS(sol):
    def masked(rule):
        def answer_logprobs(model, prompts, answers, pad=0):
            seqs = [list(p) + list(a) for p, a in zip(prompts, answers)]
            T = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), T), pad, dtype=torch.long)
            mask = torch.zeros(len(seqs), T - 1)
            for i, (p, a) in enumerate(zip(prompts, answers)):
                ids[i, :len(seqs[i])] = torch.tensor(seqs[i])
                lo, hi = rule(len(p), len(a), T)
                mask[i, lo:hi] = 1.0
            logits, _ = model(ids[:, :-1])
            logp = F.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
            return (logp * mask).sum(-1)
        return answer_logprobs

    def mean_not_sum(model, prompts, answers, pad=0):
        return sol.answer_logprobs(model, prompts, answers, pad) / torch.tensor([len(a) for a in answers])

    def left_padded(model, prompts, answers, pad=0):
        seqs = [list(p) + list(a) for p, a in zip(prompts, answers)]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        mask = torch.zeros(len(seqs), T - 1)
        for i, (p, a) in enumerate(zip(prompts, answers)):
            off = T - len(seqs[i])
            ids[i, off:] = torch.tensor(seqs[i])
            mask[i, off + len(p) - 1: off + len(p) - 1 + len(a)] = 1.0
        logits, _ = model(ids[:, :-1])
        logp = F.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        return (logp * mask).sum(-1)

    def sigmoid_loss(pc, pr, rc, rr, beta=0.1):
        m = beta * ((pc - rc.detach()) - (pr - rr.detach()))
        return -torch.sigmoid(m).mean(), m.detach()

    def no_beta(pc, pr, rc, rr, beta=0.1):
        return sol.dpo_loss(pc, pr, rc, rr, 1.0)

    def ref_not_detached(pc, pr, rc, rr, beta=0.1):
        m = beta * ((pc - rc) - (pr - rr))
        return -F.logsigmoid(m).mean(), m.detach()

    def swapped(policy, reference, opt, prompts, chosen, rejected, beta=0.1, clip=1.0, sft_weight=0.0):
        return sol.dpo_step(policy, reference, opt, prompts, rejected, chosen, beta, clip, sft_weight)

    def no_balancer(policy, reference, opt, prompts, chosen, rejected, beta=0.1, clip=1.0, sft_weight=0.0):
        rate, policy.bias_update_rate = policy.bias_update_rate, 0.0
        try:
            return sol.dpo_step(policy, reference, opt, prompts, chosen, rejected, beta, clip, sft_weight)
        finally:
            policy.bias_update_rate = rate

    def sft_term(rule):
        def dpo_step(policy, reference, opt, prompts, chosen, rejected, beta=0.1, clip=1.0, sft_weight=0.0):
            with torch.no_grad():
                ref_c = sol.answer_logprobs(reference, prompts, chosen)
                ref_r = sol.answer_logprobs(reference, prompts, rejected)
            pol_c = sol.answer_logprobs(policy, prompts, chosen)
            pol_r = sol.answer_logprobs(policy, prompts, rejected)
            loss, margin = sol.dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            loss = loss + rule(sft_weight, pol_c, chosen)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            policy.update_expert_biases()
            return loss.item(), margin.mean().item(), (margin > 0).float().mean().item()
        return dpo_step

    bugs = [("the prompt counted too", dict(answer_logprobs=masked(lambda P, A, T: (0, P - 1 + A)))),
            ("answer positions off by one", dict(answer_logprobs=masked(lambda P, A, T: (P, P + A)))),
            ("padding counted", dict(answer_logprobs=masked(lambda P, A, T: (P - 1, T - 1)))),
            ("a mean per id instead of the sum", dict(answer_logprobs=mean_not_sum)),
            ("left padding (positions shift)", dict(answer_logprobs=left_padded)),
            ("sigmoid instead of log-sigmoid", dict(dpo_loss=sigmoid_loss)),
            ("beta ignored", dict(dpo_loss=no_beta)),
            ("the reference not detached", dict(dpo_loss=ref_not_detached)),
            ("chosen and rejected swapped", dict(dpo_step=swapped)),
            ("the balancer never updated", dict(dpo_step=no_balancer)),
            ("SFT term with the sign flipped", dict(dpo_step=sft_term(lambda w, c, ch: w * c.sum() / sum(map(len, ch))))),
            ("SFT term per answer, not per id", dict(dpo_step=sft_term(lambda w, c, ch: -w * c.mean()))),
            ("sft_weight ignored", dict(dpo_step=sft_term(lambda w, c, ch: 0.0)))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(11, CHECKS, MUTANTS, __doc__, learner="post.py")
