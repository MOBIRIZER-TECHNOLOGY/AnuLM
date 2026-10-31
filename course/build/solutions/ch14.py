"""
Reference solution for Chapter 14: one model for everything, and debugging it.

pick_task and omni_step match omni_train.py's training loop (one task drawn per
batch, in proportion to its share; --accum batches summed into one optimizer
step); score matches its best-checkpoint score; gates matches
experiments/omni_one.py's release gates. course/checks/ch14.py proves it.
"""

from __future__ import annotations

import numpy as np
import torch


def pick_task(shares, rng: np.random.Generator) -> int:
    """The index of the task the next batch comes from, drawn in proportion to the shares."""
    w = np.asarray(shares, dtype=float)
    return int(rng.choice(len(w), p=w / w.sum()))


def omni_step(model, opt, draw, accum: int = 1, clip: float = 1.0) -> list[float]:
    """One optimizer step over `accum` batches: draw() returns (x, y) for whichever task was picked;
    each batch's loss is divided by accum before backward, so the step follows their mean. Then
    clip, step, clear the gradients, and move the experts' balancer. Returns each batch's loss."""
    losses = []
    for _ in range(accum):
        x, y = draw()
        _, loss = model(x, y)
        (loss / accum).backward()
        losses.append(loss.item())
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    model.update_expert_biases()
    return losses


def score(losses: dict, first: dict) -> float:
    """omni_train's checkpoint score: the mean over tasks of held-out loss / its value at step 0.
    1.0 at the start; lower is better. A task with a large loss can't drown out one with a small one,
    but a mean still hides a task that got worse."""
    return float(np.mean([losses[k] / first[k] for k in losses]))


def gates(results: dict, rules: dict) -> dict:
    """omni_one.gates: for each rule name -> (op, threshold), True when the measured value exists
    and satisfies it ("<=" or ">="). A measurement that is missing FAILS its gate. If all three
    Hindi WERs are present, their mean is added as wer_hi_mean first."""
    res = dict(results)
    hi = [res.get(k) for k in ("wer_hi_fleurs", "wer_hi_kathbath", "wer_hi_mucs")]
    if all(v is not None for v in hi):
        res["wer_hi_mean"] = round(sum(hi) / 3, 1)
    out = {}
    for k, (op, thr) in rules.items():
        v = res.get(k)
        out[k] = v is not None and (v <= thr if op == "<=" else v >= thr)
    return out
