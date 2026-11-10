---
title: "Appendix H · DPO and GRPO with LoRA"
part: "Appendices"
accent: slate
pinned: "@2e80b3e"
---

# Appendix H  DPO and GRPO with LoRA

Chapter 34 runs DPO and GRPO on `anulm/model.py` with a frozen second copy of
the model as the reference, and 34.6 lists what adding them to the repository
would take. One line of that list says that `anulm/lora.py` has no switch to
bypass its adapters, so a LoRA policy cannot serve as its own reference. This
appendix supplies the switch from outside, in a dozen lines, and runs both
methods with it: one model in memory, the policy with its adapters on, the
reference with them off. Everything here was run against the repository at the
pin by `book/code-examples/appH_lora_preferences.py` (**Book example, not
repository code**), which reuses Chapter 34's example and `anulm/lora.py`
unchanged and takes about a minute on a CPU.

```bash
python book/code-examples/appH_lora_preferences.py
```

## H.1  The reference for free

A LoRA layer computes the base layer plus a scaled correction:

> **In the repo** · `anulm/lora.py` · `LoRALinear.forward` · lines 71–75 · @2e80b3e

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """base(x) + scale * dropout(x) @ A.T @ B.T, the adapter run in x's dtype."""
        out = self.base(x)
        h = self.drop(x) @ self.lora_a.t().to(x.dtype)
        return out + (h @ self.lora_b.t().to(x.dtype)) * self.scale
```

`scale` is a plain Python attribute (31.2), so setting it to 0 on every adapter
gives back `base(x)` exactly, with no second copy of the weights. That is
almost the whole reference. The rest is Chapter 31's trap: the router's
balancing biases are buffers, `apply_lora` does not freeze them, and the
training loop keeps moving them after every step (31.4). Policy and reference
share those biases, so a reference that only zeroes the scales routes tokens
the way the *policy* now routes them. The switch therefore also puts back the
biases the adapters were trained on top of, and restores everything on exit:

```python
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
```

The biases come from the repository's own helper, taken at step 0:

> **In the repo** · `anulm/lora.py` · `router_state_dict` · lines 126–137 · @2e80b3e

```python
def router_state_dict(model: nn.Module) -> dict:
    # ...
    return {n: b.detach().cpu() for n, b in model.named_buffers()
            if n.endswith("expert_bias")}
```

with one care. `detach().cpu()` on a model that already lives on the CPU
returns a *view* of the live buffer, not a copy, so a snapshot taken this way
drifts with the policy and restores nothing. Clone it. The example takes five
LoRA steps on `tiny(bias_update_rate=0.1)` with the balancer running, as
`anulm/training/finetune.py` does, then compares three models with the
pristine base on the same batch:

```
1. tiny() + LoRA r=4 after 5 steps: 1920 adapter numbers; adapted logits differ from the base by up to 6.52e-01; router biases moved by up to 0.100
   router_state_dict() on a CPU model returns views of the live buffers: True (snapshot with clone())
   adapters off, biases restored: logits differ by 0.0e+00; tokens whose expert set differs from the base: 0%
   adapters off, drifted biases kept: logits differ by 1.7e-02; expert set differs: 45%
   policy restored on exit: True
```

*verified*: printed by the example. With the scales at 0 and the biases put
back, the logits equal the base model's to the bit and every token reaches the
experts it reached before. With the scales at 0 alone, 45% of tokens are
routed differently and the logits are off by 1.7·10⁻²: a reference that is
not the reference, and nothing in the DPO loss would say so.

`IN THE REPO` (`LoRALinear`, `router_state_dict`) · `NOT IN THE REPO` (the switch)

## H.2  The losses, against Chapter 34's exercises

Nothing in the losses changes. The lab book's `dpo_loss` (34.3) and Chapter
34's `group_advantages` and KL estimate (34.5) reproduce the chapter's
exercise answers:

```
2. exercise 1: margin 0.3000, loss 0.5544 (below ln 2 = 0.6931); exercise 3: policy = reference gives 0.6931
   exercise 2: rewards 1,0,0,0 -> [1.4997, -0.4999, -0.4999, -0.4999]; all equal -> [0.0, 0.0, 0.0, 0.0]
   exercise 4: exp(d) - d - 1 at d = 0.5 is 0.1487, at d = -0.5 is 0.1065
```

*verified*: printed by the example; the same numbers as 34.7's answers. The
all-equal group is the row to remember for GRPO: every advantage is 0, the
step teaches nothing and still costs a full round of sampling, which is why
GRPO pipelines choose prompts the model sometimes gets right.

`NOT IN THE REPO`

## H.3  A DPO step with a LoRA policy

Chapter 34's `dpo_step` takes the reference log-probabilities as numbers, so it
needs nothing new; the reference is a small callable that runs the policy with
its adapters off and no gradient:

```python
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
```

`answer_logprob(ref, prompt, answer)` then works unchanged. The example trains
only the adapters (`lora_parameters`, Chapter 31), on Chapter 34's made-up
pair, twice: with `apply_lora`'s default targets, the attention projections,
and with the output head added to them.

```
3. DPO, LoRA r=4 on attention (trainable 1920 of 49808 numbers), reference = adapters off
   step  0: loss 0.6931  margin +0.000  log p chosen -12.687  rejected -12.458
   step 10: loss 0.5718  margin +0.259  log p chosen -11.252  rejected -13.617
   step 20: loss 0.5706  margin +0.262  log p chosen -11.248  rejected -13.642
   the reference did not move: log p(chosen) -12.6873 before, -12.6873 after
3. DPO, LoRA r=4 on attention + lm_head (trainable 2304 of 50192 numbers), reference = adapters off
   step  0: loss 0.6931  margin +0.000  log p chosen -12.687  rejected -12.458
   step 10: loss 0.2454  margin +1.280  log p chosen -10.097  rejected -22.663
   step 20: loss 0.0217  margin +3.820  log p chosen  -7.961  rejected -45.930
   the reference did not move: log p(chosen) -12.6873 before, -12.6873 after
```

*illustrative*: twenty steps on `tiny()`, one pair, printed by the example.
Step 0 is ln 2 in both runs, as it must be when the policy *is* the reference.
Attention adapters alone stall at a margin of 0.26: the pair differs in its
last two ids, and 1,920 numbers inside the attention can barely move the
output distribution of a three-layer model. Adding 384 numbers on `lm_head`
takes the loss to 0.02 in twenty steps, past Chapter 34's full fine-tune at
the same step (0.30). The rejected answer's log-probability falls to −46 on
the way: the loss sees only the difference, and watching the two terms
separately is Chapter 34's warning, not a suggestion.

`NOT IN THE REPO`

## H.4  GRPO with a LoRA policy

Chapter 34's `grpo_step` takes a reference model as an argument and calls it
like one, so `Reference` drops in. The policy samples its group with
`generate` at T = 1 (34.6), scores it with the chapter's stand-in reward, the
share of even answer ids, and updates its adapters; the balancer runs after
each step. Three choices of what to adapt:

```
4. GRPO, LoRA r=4  lr 0.01 on attention           ( 1920 trainable), mean reward per 10 steps: [0.531, 0.559, 0.497, 0.547, 0.597, 0.547]
4. GRPO, LoRA r=16 lr 0.02 on attention           ( 7680 trainable), mean reward per 10 steps: [0.534, 0.55, 0.509, 0.537, 0.597, 0.544]
4. GRPO, LoRA r=4  lr 0.01 on attention + lm_head ( 2304 trainable), mean reward per 10 steps: [0.569, 0.828, 0.966, 1.0, 1.0, 0.981]
```

*illustrative*: sixty steps on `tiny()`, printed by the example. Four times the
rank on the same layers buys nothing: the reward asks for a change in which
ids the model prefers, and the attention projections of a tiny model are the
wrong place to express it. The head, at 384 numbers, gets the group to all-even
answers in forty steps, where Chapter 34's full fine-tune reached 0.956 in
sixty. The lesson is where to put adapters, not how many: choose the targets
from what the reward has to change, and check with a run this short before a
long one. On the released model, with twenty layers and a 32,768-way head, the
right targets are *not measured here*.

`NOT IN THE REPO`

## H.5  On the released model

What the switch saves is the copy. A frozen fp32 reference of the released
model is 397,659,648 × 4 bytes = 1.59 GB (Chapter 20); the r = 16 attention
adapters are 1,474,560 numbers, 5.9 MB, 0.37% of the model (Chapter 31); the
reference pass needs no further weights. It saves no compute: DPO still runs
two passes through the reference and two through the policy per pair, and the
reference pass costs the same as before, since the base weights are the same
weights. The 456 router biases are 1,824 bytes to snapshot. One more constraint
from `anulm/lora.py`: adapting the experts needs `--moe-impl sparse`, because
the grouped kernel stacks expert weights directly and cannot see through an
adapter (31.3); the attention and the head need nothing.

```
5. released model: a frozen fp32 copy is 1.59 GB; the r=16 attention adapters are 5.9 MB (0.37%), and adapters_off needs 0 extra weights
```

*verified*: arithmetic, printed by the example. Whether DPO or GRPO on
preference data improves the released chat model is *not measured here*; this
appendix shows that the machinery runs, and what it costs.

`NOT IN THE REPO`

## H.6  Common mistakes, and how to catch them

- **A reference that zeroes the scales and keeps the policy's router
  biases.** It drifts with the policy. *Catch it:* `expert_sets` of the
  reference against the pristine base, 0% changed (H.1).
- **A bias snapshot that is a view.** `router_state_dict()` on a CPU model
  returns the live buffers. *Catch it:* `data_ptr()` of the snapshot against
  the buffer, or the logits test of H.1, which fails by 1.7·10⁻².
- **Adapters on layers the reward cannot use.** The loss plateaus and a bigger
  rank does not help. *Catch it:* a sixty-step run with the head adapted, as
  H.4; if that learns and the attention-only run does not, the targets are
  the problem.
- **Reading the DPO loss alone.** Both log-probabilities can fall. *Catch it:*
  log them separately, as the example prints them.

## Key points

- With LoRA, the DPO or GRPO reference is the policy with its adapters off:
  every `LoRALinear.scale` to 0, and the router biases put back from a cloned
  step-0 snapshot. No second copy of the model.
- Chapter 34's losses and steps run unchanged with that reference; the
  exercise answers come out the same.
- Which layers you adapt decides what a reward can teach. On `tiny()`,
  7,680 attention numbers learn nothing from the stand-in reward and 384 on
  the head learn it in forty steps.
- The switch saves the reference's memory, not its compute.
