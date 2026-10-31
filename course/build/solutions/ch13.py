"""
Reference solution for Chapter 13: hearing and seeing.

grow_rows matches speech_vocab._grow; ImageProjector matches
vision_encoder.ImageProjector; build_caption matches VisionLanguage.build;
match_scale matches omni_train.Omni._match_scale. course/checks/ch13.py proves
each one against the repository's own code.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- new words for a new kind of input -----------------------------------------------------

def grow_rows(weight: torch.Tensor, n_new: int, gen: torch.Generator) -> torch.Tensor:
    """Append n_new rows near the mean of the existing ones (noise of a tenth of their spread):
    a fresh N(0, 0.02) row would start far from where the trained rows live."""
    mean = weight.mean(dim=0, keepdim=True)
    std = weight.std().item()
    noise = torch.randn(n_new, weight.shape[1], generator=gen, dtype=weight.dtype) * (std * 0.1)
    return torch.cat([weight, mean.expand(n_new, -1) + noise], dim=0)


@torch.no_grad()
def grow_vocab(model, n_new: int, seed: int = 0) -> int:
    """Widen BOTH the embedding table and the output head by n_new ids (the model's head is not
    tied to its embeddings, so growing one leaves it able to read the new ids but not write them,
    or the reverse). Every existing id keeps its row. Returns the new vocabulary size."""
    gen = torch.Generator().manual_seed(seed)
    old_e, old_h = model.embed_tokens, model.lm_head
    e = grow_rows(old_e.weight.float(), n_new, gen)
    h = grow_rows(old_h.weight.float(), n_new, gen)
    model.embed_tokens = nn.Embedding(e.shape[0], e.shape[1])
    model.embed_tokens.weight.copy_(e)
    model.lm_head = nn.Linear(h.shape[1], h.shape[0], bias=False)
    model.lm_head.weight.copy_(h)
    return e.shape[0]


# --- embeddings in, with no ids ---------------------------------------------------------------

def forward_embeds(model, embeds: torch.Tensor, targets=None):
    """The model's forward pass from (B, S, hidden) input vectors instead of ids: the same
    rotary positions, layers, final norm and head. Returns (logits, loss or None)."""
    x = embeds
    cos, sin = model.rotary(x.shape[1], x.device, x.dtype)
    for layer in model.layers:
        out = layer(x, cos, sin)
        x = out[0] if isinstance(out, tuple) else out
    logits = model.lm_head(model.norm(x))
    if targets is None:
        return logits, None
    return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)


class ImageProjector(nn.Module):
    """vision_encoder.ImageProjector: pool each 2x2 neighbourhood of the patch GRID (concatenating
    the four vectors), then Linear -> GELU -> Linear -> LayerNorm into the model's hidden size."""

    def __init__(self, in_dim: int, out_dim: int, pool: int = 2, hidden: int | None = None):
        super().__init__()
        self.pool = pool
        d = in_dim * pool * pool
        hidden = hidden or max(out_dim, d)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """(P, in_dim) -> (P / pool^2, out_dim). The patches are a square grid read row by row, so
        neighbours are found in 2-D: a flat stride would join one row's end to the next one's start."""
        P, D = patches.shape
        if self.pool > 1:
            g = math.isqrt(P)
            x = patches.reshape(g, g, D)
            gp = g // self.pool
            x = x[: gp * self.pool, : gp * self.pool]
            x = x.reshape(gp, self.pool, gp, self.pool, D).permute(0, 2, 1, 3, 4)
            patches = x.reshape(gp * gp, D * self.pool * self.pool)
        return self.net(patches)


def build_caption(model, proj, patches, image_bos: int, image_eos: int, answer_ids, prompt_ids=None):
    """VisionLanguage.build: embeddings of <|image|>, the projected patches, <|/image|>, the prompt
    (if any) and the answer, as one (S, hidden) tensor; targets -1 everywhere except the answer's
    ids, each placed one position before it. Returns (embeds, targets)."""
    emb = model.embed_tokens
    dev = patches.device
    parts = [emb(torch.tensor([image_bos], device=dev)), proj(patches), emb(torch.tensor([image_eos], device=dev))]
    if prompt_ids:
        parts.append(emb(torch.tensor(prompt_ids, device=dev)))
    parts.append(emb(torch.tensor(answer_ids, device=dev)))
    embeds = torch.cat(parts, dim=0)
    targets = torch.full((embeds.shape[0],), -1, dtype=torch.long, device=dev)
    start = embeds.shape[0] - len(answer_ids)
    targets[start - 1: start - 1 + len(answer_ids)] = torch.tensor(answer_ids, device=dev)
    return embeds, targets


@torch.no_grad()
def match_scale(proj, text_embeddings: torch.Tensor) -> tuple[float, float]:
    """omni_train._match_scale: multiply the projector's final LayerNorm weight AND bias by one
    factor, so its output's expected norm, sqrt(sum(w^2) + sum(b^2)), equals the mean norm of the
    text embeddings. Returns (norm before, norm after)."""
    target = text_embeddings.float().norm(dim=-1).mean()
    ln = proj.net[-1]
    now = (ln.weight.float().pow(2).sum() + ln.bias.float().pow(2).sum()).sqrt()
    ln.weight.mul_(target / now)
    ln.bias.mul_(target / now)
    return now.item(), target.item()
