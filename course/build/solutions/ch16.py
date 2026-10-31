"""
Reference solution for Chapter 16: continued pre-training and longer context.

rewarm matches train.py's --rewarmup; mix_documents follows mix_corpus.py
(draw by remaining budget, every 50th document of each source held out);
loss_by_position is eval_context.py's eval_by_position; extend_context turns
your model into model.py's AnuLM with YaRN, logit for logit.
course/checks/ch16.py proves each one.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def rewarm(lr: float, step: int, start_step: int, rewarmup: int) -> float:
    """train.py: a run resumed at start_step ramps from near zero back up to the schedule's lr over
    `rewarmup` steps, instead of jumping straight to it."""
    if rewarmup and start_step and step < start_step + rewarmup:
        return lr * (step - start_step + 1) / rewarmup
    return lr


def mix_documents(sources: dict, budgets: dict, heldout_every: int = 50, seed: int = 1337):
    """sources: name -> list of documents; budgets: name -> characters to use. Draw the next document
    from a source chosen in proportion to its REMAINING budget, so the mix is even along the file.
    Every heldout_every-th document a source yields goes to that source's held-out set instead, and
    never into the mix. Returns (mixed documents, {name: held-out documents})."""
    rng = random.Random(seed)
    state = {n: {"docs": iter(d), "left": float(budgets[n]), "count": 0} for n, d in sources.items()}
    mixed, held = [], {n: [] for n in sources}
    while True:
        live = [n for n, s in state.items() if s["left"] > 0 and s["docs"] is not None]
        if not live:
            break
        n = rng.choices(live, weights=[state[k]["left"] for k in live])[0]
        s = state[n]
        try:
            doc = next(s["docs"])
        except StopIteration:
            s["docs"] = None
            continue
        s["count"] += 1
        if s["count"] % heldout_every == 0:
            held[n].append(doc)
            continue
        mixed.append(doc)
        s["left"] -= len(doc)
    return mixed, held


@torch.no_grad()
def loss_by_position(model, data: torch.Tensor, length: int, batches: int, bins: int = 4, seed: int = 0):
    """Mean next-token loss in each of `bins` equal stretches of a `length`-long window, averaged over
    `batches` random windows: an extension that fails shows up at the far end, which one average hides."""
    g = torch.Generator().manual_seed(seed)
    edges = [length * b // bins for b in range(bins + 1)]
    totals = [0.0] * bins
    for _ in range(batches):
        i = torch.randint(len(data) - length - 1, (1,), generator=g).item()
        x = data[i: i + length].long()[None]
        y = data[i + 1: i + 1 + length].long()[None]
        logits, _ = model(x)
        tok = F.cross_entropy(logits[0].float(), y[0], reduction="none")
        for b in range(bins):
            totals[b] += tok[edges[b]: edges[b + 1]].mean().item()
    return [t / batches for t in totals]


def extend_context(model, block_size: int, factor: float, original_context: int, rope_theta: float) -> None:
    """Let the model read `block_size` positions with YaRN: a new rotary table whose slow frequencies
    are stretched by `factor` (the fast ones kept), and every attention's temperature adjusted to
    match. The weights don't change."""
    Rotary = type(model.rotary)
    model.block_size = block_size
    model.rotary = Rotary(model.rotary.dim, rope_theta, yarn=True, yarn_factor=factor,
                          yarn_original_context=original_context)
    scale = model.rotary.dim ** -0.5 * model.rotary.attn_mscale ** 2
    for layer in model.layers:
        layer.attn.scale = scale
