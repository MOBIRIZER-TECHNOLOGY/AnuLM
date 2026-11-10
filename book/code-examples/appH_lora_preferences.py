"""
Book example, not repository code.

Appendix H: DPO and GRPO with a LoRA policy on model.py, the reference being
the same model with its adapters switched off. Chapter 34 runs both methods
with a frozen second copy of the model; this example removes the copy. It
reuses Chapter 34's example and lora.py unchanged.

    python book/code-examples/appH_lora_preferences.py

1. adapters_off: zero every LoRALinear.scale and put back the router biases the
   adapters were trained on top of. The base model comes back exactly: same
   logits, same expert sets. Without the biases the logits differ and tokens
   change experts (Chapter 31's trap).
2. Chapter 34's exercise answers from the same loss functions: margin 0.3 gives
   0.5544, margin 0 gives ln 2, rewards 1,0,0,0 give +1.4997 and -0.4999,
   exp(d) - d - 1 at d = +/-0.5.
3. Twenty DPO steps on tiny() with a LoRA policy, the reference from
   adapters_off, Chapter 34's made-up pair.
4. Sixty GRPO steps with a LoRA policy and the same reference, Chapter 34's
   stand-in reward (the share of even answer ids).
5. The memory arithmetic on the released model: a frozen copy against the
   adapters.
"""

from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import ch34_preferences as ch34  # noqa: E402  Chapter 34's example: answer_logprob, dpo_step, grpo_step
import ch11  # noqa: E402  the lab book's DPO solution (ch34 put it on the path)
from anulm import model as M  # noqa: E402
from anulm.lora import LoRALinear, apply_lora, lora_parameters, router_state_dict  # noqa: E402
from tests.test_model import tiny  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@contextlib.contextmanager
def adapters_off(model, base_biases: dict | None = None):
    """The base model, from a LoRA-wrapped one: every adapter's scale set to 0, and the router
    biases put back to `base_biases` (what the adapters were trained on top of). Both are
    restored on exit, so the policy is untouched."""
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    scales = [m.scale for m in mods]
    now = {n: b.clone() for n, b in model.named_buffers() if n.endswith("expert_bias")}
    for m in mods:
        m.scale = 0.0
    with torch.no_grad():
        for n, b in model.named_buffers():
            if base_biases and n in base_biases:
                b.copy_(base_biases[n])
    try:
        yield model
    finally:
        for m, s in zip(mods, scales):
            m.scale = s
        with torch.no_grad():
            for n, b in model.named_buffers():
                if n in now:
                    b.copy_(now[n])


class Reference:
    """Callable like a model: runs the policy with its adapters off, without gradient."""

    def __init__(self, policy, base_biases):
        self.policy, self.base_biases = policy, base_biases

    def __call__(self, *args, **kwargs):
        was_training = self.policy.training
        self.policy.eval()
        try:
            with torch.no_grad(), adapters_off(self.policy, self.base_biases):
                return self.policy(*args, **kwargs)
        finally:
            self.policy.train(was_training)


def expert_sets(model, x):
    """The top-k expert ids each token reaches in every MoE layer (Chapter 31's helper)."""
    out = []
    hooks = [m.register_forward_hook(lambda mod, i, o: out.append(o[0]))
             for m in model.modules() if isinstance(m, M.Router)]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return torch.stack([o.sort(-1).values for o in out])


ATTN = ("attn.query_key_value", "attn.dense")              # lora.py's DEFAULT_TARGETS
ATTN_HEAD = ATTN + ("lm_head",)


def lora_model(cfg, seed: int, r: int = 4, alpha: float = 8.0, targets=ATTN):
    """A fresh model with adapters, its pristine base, and the router biases at step 0."""
    torch.manual_seed(seed)
    base = M.AnuLM(cfg).eval()
    policy = M.AnuLM(cfg)
    policy.load_state_dict(base.state_dict())
    apply_lora(policy, r=r, alpha=alpha, targets=targets)
    # router_state_dict() detaches; on a CPU model that is a view of the live buffer, which
    # keeps drifting. The snapshot must be a copy.
    return policy, base, {k: v.clone() for k, v in router_state_dict(policy).items()}


def main() -> None:
    prompt, chosen, rejected = [5, 9, 12, 3], [17, 4, 2], [17, 8, 30]

    # 1. the reference for free
    policy, base, biases0 = lora_model(tiny(bias_update_rate=0.1), seed=0)
    x = torch.randint(0, 64, (2, 16))
    opt = torch.optim.AdamW(lora_parameters(policy), lr=1e-2)
    policy.train()
    for _ in range(5):                                       # as finetune.py: a step, then the balancer
        _, loss = policy(x, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        policy.update_expert_biases()
    policy.eval()
    with torch.no_grad():
        want, _ = base(x, x)
        adapted, _ = policy(x, x)
        with adapters_off(policy, biases0):
            off, _ = policy(x, x)
            sets_off = expert_sets(policy, x)
        with adapters_off(policy):                           # scale 0, but the drifted biases stay
            off_nobias, _ = policy(x, x)
            sets_nobias = expert_sets(policy, x)
        after, _ = policy(x, x)
    sets_base = expert_sets(base, x)
    n_ad = sum(p.numel() for p in lora_parameters(policy))
    view = router_state_dict(policy)                         # what a snapshot without clone() would hold
    aliased = all(v.data_ptr() == b.data_ptr() for (n, v), (_, b) in
                  zip(view.items(), ((n, b) for n, b in policy.named_buffers() if n.endswith("expert_bias"))))
    print(f"1. tiny() + LoRA r=4 after 5 steps: {n_ad} adapter numbers; adapted logits differ from the base by "
          f"up to {(adapted - want).abs().max():.2e}; router biases moved by up to "
          f"{max((biases0[n] - b).abs().max().item() for n, b in policy.named_buffers() if n in biases0):.3f}")
    print(f"   router_state_dict() on a CPU model returns views of the live buffers: {aliased} (snapshot with clone())")
    print(f"   adapters off, biases restored: logits differ by {(off - want).abs().max():.1e}; "
          f"tokens whose expert set differs from the base: {100 * (sets_off != sets_base).any(-1).float().mean():.0f}%")
    print(f"   adapters off, drifted biases kept: logits differ by {(off_nobias - want).abs().max():.1e}; "
          f"expert set differs: {100 * (sets_nobias != sets_base).any(-1).float().mean():.0f}%")
    print(f"   policy restored on exit: {torch.equal(after, adapted)}")

    # 2. the exercise answers of Chapter 34
    t = lambda v: torch.tensor([float(v)])
    loss, m = ch11.dpo_loss(t(-10), t(-20), t(-11), t(-18), beta=0.1)
    loss0, _ = ch11.dpo_loss(t(-10), t(-20), t(-10), t(-20), beta=0.1)
    adv = ch34.group_advantages(torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]))
    kl = lambda d: math.exp(d) - d - 1
    print(f"\n2. exercise 1: margin {m.item():.4f}, loss {loss.item():.4f} (below ln 2 = {math.log(2):.4f}); "
          f"exercise 3: policy = reference gives {loss0.item():.4f}")
    print(f"   exercise 2: rewards 1,0,0,0 -> {[round(v, 4) for v in adv[0].tolist()]}; "
          f"all equal -> {adv[1].tolist()}")
    print(f"   exercise 4: exp(d) - d - 1 at d = 0.5 is {kl(0.5):.4f}, at d = -0.5 is {kl(-0.5):.4f}")

    # 3. DPO with a LoRA policy: adapters on the attention only, then on the attention and the head
    print()
    for r, targets, name in [(4, ATTN, "attention"), (4, ATTN_HEAD, "attention + lm_head")]:
        policy, _, biases0 = lora_model(tiny(), seed=0, r=r, alpha=2.0 * r, targets=targets)
        ref = Reference(policy, biases0)
        ref_c, ref_r = ch34.answer_logprob(ref, prompt, chosen), ch34.answer_logprob(ref, prompt, rejected)
        policy.train()
        opt = torch.optim.AdamW(lora_parameters(policy), lr=1e-2)
        print(f"3. DPO, LoRA r={r} on {name} (trainable {sum(p.numel() for p in lora_parameters(policy))} of "
              f"{sum(p.numel() for p in policy.parameters())} numbers), reference = adapters off")
        for step in range(21):
            loss, margin, lc, lr = ch34.dpo_step(policy, opt, prompt, chosen, rejected, ref_c, ref_r)
            if step % 10 == 0:
                print(f"   step {step:2d}: loss {loss:.4f}  margin {margin:+.3f}  "
                      f"log p chosen {lc:7.3f}  rejected {lr:7.3f}")
        with torch.no_grad():
            still_c = ch34.answer_logprob(ref, prompt, chosen)
        print(f"   the reference did not move: log p(chosen) {ref_c.item():.4f} before, {still_c.item():.4f} after")

    # 4. GRPO with a LoRA policy: the same three choices of what to adapt
    even = lambda a: sum(v % 2 == 0 for v in a) / len(a)
    print()
    for r, lr, targets, name in [(4, 1e-2, ATTN, "attention"), (16, 2e-2, ATTN, "attention"),
                                 (4, 1e-2, ATTN_HEAD, "attention + lm_head")]:
        policy, _, biases0 = lora_model(tiny(), seed=1, r=r, alpha=2.0 * r, targets=targets)
        ref = Reference(policy, biases0)
        policy.train()
        opt = torch.optim.AdamW(lora_parameters(policy), lr=lr)
        curve = [ch34.grpo_step(policy, ref, opt, prompt, even) for _ in range(60)]
        print(f"4. GRPO, LoRA r={r:<2} lr {lr:g} on {name:<19} ({sum(p.numel() for p in lora_parameters(policy)):>5} "
              f"trainable), mean reward per 10 steps: {[round(sum(curve[i:i + 10]) / 10, 3) for i in range(0, 60, 10)]}")

    # 5. the arithmetic on the released model
    full, adapters = 397_659_648, 1_474_560                   # Chapter 20; Chapter 31 (r = 16, attention)
    print(f"\n5. released model: a frozen fp32 copy is {full * 4 / 1e9:.2f} GB; the r=16 attention adapters are "
          f"{adapters * 4 / 1e6:.1f} MB ({100 * adapters / full:.2f}%), and adapters_off needs {0} extra weights")


if __name__ == "__main__":
    main()
