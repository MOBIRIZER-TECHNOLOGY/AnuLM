"""
Book example, not repository code.

Chapter 34: preference optimisation (DPO) and group-relative RL (GRPO), which
the repository's main code does not implement. Everything here is made up for
teaching; it uses model.py and the lab book's DPO solution unchanged.

    python book/code-examples/ch34_preferences.py

1. An answer's log-probability on model.py's AnuLM. The lab book's
   `answer_logprobs` calls the model without targets, and model.py then
   returns the last position only; `answer_logprob` passes targets so every
   position comes back, and equals minus the SFT loss times the answer length.
2. DPO by hand: the worked numbers of 34.3, against the lab book's dpo_loss.
3. Twenty DPO steps on the tiny test model, one preference pair, with the
   router-bias update after each step.
4. GRPO's group advantages by hand, and the KL estimate exp(d) - d - 1:
   never negative, and on average the KL divergence.
5. Sixty GRPO steps on the tiny test model, sampling with model.py's
   generate, with a made-up reward: the share of answer ids that are even.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "course" / "build" / "solutions"))
import ch11  # noqa: E402  the lab book's DPO solution, unchanged
from anulm.model import AnuLM  # noqa: E402
from tests.test_model import tiny  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def answer_token_logprobs(model, prompt, answer):
    """log p(id | everything before it) for each answer id, shape (len(answer),).
    Targets are passed so that model.py returns logits for every position, not just the last."""
    ids = torch.tensor([prompt + answer])
    x, tgt = ids[:, :-1], ids[:, 1:].clone()
    tgt[:, :len(prompt) - 1] = -1                       # position t predicts id t + 1: grade the answer only
    logits, _ = model(x, tgt)                           # (1, S, V)
    logp = F.log_softmax(logits.float(), -1)
    keep = tgt != -1
    return logp[keep].gather(-1, tgt[keep][:, None]).squeeze(-1)


def answer_logprob(model, prompt, answer):
    """log p(answer | prompt): the sum of its ids' log-probabilities, shape (1,)."""
    return answer_token_logprobs(model, prompt, answer).sum()[None]


def dpo_step(policy, opt, prompt, chosen, rejected, ref_c, ref_r, beta=0.1):
    """The lab book's dpo_step for model.py: answer_logprob in place of answer_logprobs."""
    pol_c, pol_r = answer_logprob(policy, prompt, chosen), answer_logprob(policy, prompt, rejected)
    loss, margin = ch11.dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    policy.update_expert_biases()                       # an MoE policy keeps balancing, as in finetune.py
    return loss.item(), margin.item(), pol_c.item(), pol_r.item()


def group_advantages(rewards, eps=1e-4):
    """rewards (prompts, G) -> each answer's reward minus its group's mean, over the group's std."""
    mean, std = rewards.mean(1, keepdim=True), rewards.std(1, keepdim=True)
    return torch.where(std > 0, (rewards - mean) / (std + eps), torch.zeros_like(rewards))


def grpo_loss(logp, ref_logp, adv, beta=0.04):
    """logp, ref_logp (G, n): per-id log-probabilities of G answers; adv (G,). Push every id of an answer
    up by its advantage, plus beta times the per-position KL estimate exp(d) - d - 1."""
    pg = -(adv[:, None] * logp).mean()
    d = ref_logp - logp                                 # log pi_ref - log pi_theta at each position
    return pg + beta * (torch.exp(d) - d - 1).mean()


def grpo_step(policy, reference, opt, prompt, reward_fn, G=8, n_new=4):
    """Sample G answers to one prompt, score them, and take one GRPO step. Returns the mean reward."""
    out = policy.generate(torch.tensor([prompt] * G), n_new, temperature=1.0)     # (G, len(prompt) + n_new)
    answers = [row[len(prompt):] for row in out.tolist()]
    rewards = torch.tensor([[reward_fn(a) for a in answers]])                    # (1, G)
    adv = group_advantages(rewards)[0]                                           # (G,)
    logp = torch.stack([answer_token_logprobs(policy, prompt, a) for a in answers])   # (G, n_new)
    with torch.no_grad():
        ref = torch.stack([answer_token_logprobs(reference, prompt, a) for a in answers])
    loss = grpo_loss(logp, ref, adv)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    policy.update_expert_biases()
    return rewards.mean().item()


def main() -> None:
    torch.manual_seed(0)
    model = AnuLM(tiny()).eval()
    prompt, chosen, rejected = [5, 9, 12, 3], [17, 4, 2], [17, 8, 30]

    # 1. the log-probability of an answer
    try:
        ch11.answer_logprobs(model, [prompt], [chosen])
        print("lab-book answer_logprobs on model.py: ran")
    except RuntimeError as err:
        print(f"lab-book answer_logprobs on model.py: RuntimeError ({str(err).splitlines()[0][:60]}...)")
    with torch.no_grad():
        lp = answer_logprob(model, prompt, chosen)
        ids = torch.tensor([prompt + chosen])
        tgt = ids[:, 1:].clone()
        tgt[:, :len(prompt) - 1] = -1
        _, sft = model(ids[:, :-1], tgt)
    print(f"log p(chosen | prompt) = {lp.item():.4f}; -(SFT loss x {len(chosen)} answer ids) = "
          f"{-sft.item() * len(chosen):.4f}")

    # 2. DPO by hand
    beta = 0.1
    pc, pr, rc, rr = torch.tensor([-12.0]), torch.tensor([-15.0]), torch.tensor([-13.0]), torch.tensor([-14.0])
    margin = beta * ((pc - rc) - (pr - rr))
    by_hand = -math.log(1 / (1 + math.exp(-margin.item())))
    loss, m = ch11.dpo_loss(pc, pr, rc, rr, beta)
    print(f"DPO: margin {m.item():.3f}, loss by hand {by_hand:.4f}, ch11.dpo_loss {loss.item():.4f}; "
          f"at margin 0 the loss is ln 2 = {math.log(2):.4f}")
    pc.requires_grad_(True)
    ch11.dpo_loss(pc, pr, rc, rr, beta)[0].backward()
    print(f"d loss / d log p(chosen) = {pc.grad.item():.4f} = -beta * sigmoid(-margin) "
          f"= {-beta / (1 + math.exp(margin.item())):.4f}")

    # 3. twenty DPO steps on tiny()
    policy, reference = model.train(), copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    with torch.no_grad():
        ref_c, ref_r = answer_logprob(reference, prompt, chosen), answer_logprob(reference, prompt, rejected)
    for step in range(21):
        loss, margin, lc, lr = dpo_step(policy, opt, prompt, chosen, rejected, ref_c, ref_r)
        if step % 10 == 0:
            print(f"step {step:2d}: loss {loss:.4f}  margin {margin:+.3f}  "
                  f"log p chosen {lc:7.3f}  rejected {lr:7.3f}")

    # 4. GRPO's advantages and the KL estimate
    rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    print("group advantages:", [[round(v, 3) for v in row] for row in group_advantages(rewards).tolist()])
    g = torch.Generator().manual_seed(0)
    p_theta = F.softmax(torch.randn(50, generator=g), 0)
    p_ref = F.softmax(torch.randn(50, generator=g), 0)
    kl = float((p_theta * (p_theta / p_ref).log()).sum())
    x = torch.multinomial(p_theta, 200_000, replacement=True, generator=g)
    d = (p_ref[x] / p_theta[x]).log()
    est = torch.exp(d) - d - 1
    se = est.std() / math.sqrt(len(est))
    # 5. GRPO on tiny(): reward = share of answer ids that are even (a stand-in for a checker)
    torch.manual_seed(1)
    policy = AnuLM(tiny())
    reference = copy.deepcopy(policy).eval()
    policy.train()
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    even = lambda a: sum(t % 2 == 0 for t in a) / len(a)
    curve = [grpo_step(policy, reference, opt, prompt, even) for _ in range(60)]
    print("GRPO, mean reward per 10 steps:", [round(sum(curve[i:i + 10]) / 10, 3) for i in range(0, 60, 10)])
    print(f"KL(theta || ref) = {kl:.4f}; mean of exp(d) - d - 1 over 200,000 samples = {est.mean():.4f} "
          f"(standard error {se:.4f}); smallest single estimate {est.min():.2e}")


if __name__ == "__main__":
    main()
