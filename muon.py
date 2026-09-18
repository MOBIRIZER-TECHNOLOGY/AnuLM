"""
Muon -- momentum orthogonalised by Newton-Schulz.

Borrowed from Keller Jordan's modded-nanogpt speedruns, which is where most of
the recent wall-clock progress on small-scale pretraining has come from. It is
the one clear gap between AnuLM's training setup and 2026 practice.

Two reasons it is here rather than in a "nice to have" list:

  1. Convergence. On nanoGPT-scale runs Muon reaches a target loss in
     meaningfully fewer steps than AdamW.
  2. Memory, which is what actually forced the issue. AdamW keeps two fp32
     moments per parameter; Muon keeps one momentum buffer. That is 16 vs 12
     bytes per parameter including weights and grads:

         443M params, AdamW -> 7.09 GB   (does not fit an 8 GB card)
         443M params, Muon  -> 5.32 GB   (fits, ~2.3 GB left for activations)

The idea: SGD-momentum produces an update matrix that is often close to
low-rank, so a few directions dominate. Muon orthogonalises it -- replacing the
update with the nearest semi-orthogonal matrix -- so every direction gets
comparable step size. The orthogonalisation is done with a quintic Newton-Schulz
iteration rather than an SVD, so it is a handful of matmuls and runs in bf16.

Applies to 2D hidden weights only. Embeddings, the LM head, norm gains and any
1D parameter keep AdamW -- orthogonalisation is meaningless for those, and the
speedrun results are clear that mixing the two is what works.
"""

from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Approximate the orthogonal factor of G via a quintic Newton-Schulz iteration.

    Returns a matrix close to `U @ V.T` where `G = U S V.T`, without ever forming
    an SVD. The coefficients (3.4445, -4.7750, 2.0315) are tuned so the iteration
    converges fast for singular values in [0, 1] while tolerating the sloppiness
    of bf16 -- it does not converge to machine precision and is not meant to.
    """
    assert G.ndim == 2, "Muon operates on 2D parameters only"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)          # ensure top singular value <= 1
    transposed = G.size(0) > G.size(1)
    if transposed:                     # iterate on the smaller Gram matrix
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon for 2D parameters. Pair with AdamW for everything else.

    lr is NOT comparable to an AdamW lr -- the update is orthogonalised, so its
    scale is set by the shape factor below rather than by gradient magnitude.
    0.02 is the usual starting point, roughly 50-100x a typical AdamW lr.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5, weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - mom)
                # Note: `lerp`, not `lerp_` -- mutating p.grad in place would
                # corrupt gradient clipping and any later inspection of grads.
                d = g.lerp(buf, mom) if group["nesterov"] else buf
                d = zeropower_via_newtonschulz5(d, steps=group["ns_steps"])
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                # Shape correction: a tall matrix needs a larger step to move
                # its output distribution as much as a square one would.
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(d.to(p.dtype), alpha=-lr * scale)
        return loss


def build_optimizers(model, muon_lr: float = 0.02, adamw_lr: float = 3e-4,
                     weight_decay: float = 0.1, betas=(0.9, 0.95), fused: bool = False):
    """Split parameters: 2D hidden weights -> Muon, everything else -> AdamW.

    Returns (optimizers, groups_description). Embeddings and the LM head are 2D
    but are deliberately excluded -- they are lookup tables and a classifier, not
    hidden transforms, and orthogonalising them hurts.
    """
    muon_params, decay, no_decay = [], [], []
    # The router weight is 2D but it is a classifier over experts whose output
    # feeds a discrete top-k, not a hidden transform -- same category as
    # lm_head. Orthogonalising it would rescale every expert logit at once.
    excluded = ("embed_tokens", "lm_head", "router.weight")
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and not any(x in name for x in excluded):
            muon_params.append(p)
        elif p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    opts = []
    if muon_params:
        opts.append(Muon(muon_params, lr=muon_lr, weight_decay=weight_decay))
    adamw_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if decay or no_decay:
        kw = {"fused": True} if fused else {}
        opts.append(torch.optim.AdamW(adamw_groups, lr=adamw_lr, betas=betas,
                                      eps=1e-8, **kw))
    desc = (f"Muon: {len(muon_params)} tensors, "
            f"{sum(p.numel() for p in muon_params)/1e6:.1f}M params | "
            f"AdamW: {len(decay)+len(no_decay)} tensors, "
            f"{sum(p.numel() for p in decay+no_decay)/1e6:.1f}M params")
    return opts, desc
