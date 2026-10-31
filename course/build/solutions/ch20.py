"""
Reference solution for Chapter 20: training on many GPUs, data-parallel, from scratch.

Every copy of the model (a "rank") holds the same weights, reads a different
part of each batch, and after the backward pass the copies average their
gradients, so every copy takes the same step. `comm` is anything with
`rank`, `world`, `all_reduce_sum(tensor)` and `broadcast(tensor, src)`: in
the lab it wraps torch.distributed, in the checks it is two threads.
The repository trains on one GPU; this is new code.
"""

from __future__ import annotations

import torch


def shard(n: int, rank: int, world: int, seed: int, epoch: int) -> list[int]:
    """This rank's examples for one epoch: one permutation that every rank computes identically (same
    seed and epoch), dealt out in turn, so the ranks' shares never overlap and together cover it."""
    g = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(n, generator=g).tolist()
    return order[rank::world]


@torch.no_grad()
def broadcast_params(model, comm) -> None:
    """Make every rank start from rank 0's weights and buffers."""
    for t in list(model.parameters()) + list(model.buffers()):
        comm.broadcast(t.data, 0)


@torch.no_grad()
def all_reduce_grads(model, comm) -> None:
    """Average every parameter's gradient over the ranks. A parameter that got no gradient on this rank
    (an expert no token was routed to) takes part with zeros: every rank must make the same calls, in
    the same order, or they wait for each other forever."""
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        comm.all_reduce_sum(p.grad)
        p.grad /= comm.world


def ddp_step(model, opt, x, y, comm, clip: float = 1.0) -> float:
    """One data-parallel step on this rank's part of the batch. Gradients are averaged, and so are the
    experts' load counts, so every rank moves its balancing biases the same way. Returns this rank's loss."""
    opt.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    all_reduce_grads(model, comm)
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if name.endswith("load_counts"):
                comm.all_reduce_sum(buf)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    model.update_expert_biases()
    return loss.item()
