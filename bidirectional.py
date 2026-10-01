"""
Turn AnuLM's decoder into a bidirectional encoder (the LLM2Vec recipe).

    python bidirectional.py adapt --ckpt ckpt_demo_v2.pt --out ckpt_demo_v2_bidir.pt

A decoder attends only backwards: token i sees tokens 0..i. For decisions
(decide.py) that means an option can be read against the message but never
against the options after it. LLM2Vec (BehnamGhader et al., 2024) showed a
decoder can be made a strong encoder cheaply:

  1. switch the attention mask off -- every token sees every real token
     (model.py's bidir_mask, a key-padding mask);
  2. adapt briefly with masked next-token prediction (MNTP): replace 20% of
     the tokens with a placeholder and predict each original token from the
     hidden state one position earlier, with the model's own lm_head, now
     that the model can also see the right-hand context.

The weights stay this project's own -- no new pretraining, no borrowed
backbone. Data: random 512-token windows of the base-v2 corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import AnuLM, load_checkpoint

HERE = Path(__file__).parent


def bidir_hidden(m: AnuLM, idx: torch.Tensor, attn: torch.Tensor | None = None) -> torch.Tensor:
    """Final hidden states with bidirectional attention. `attn` (B, S) marks
    real tokens; padding is never attended to."""
    if attn is None:
        attn = torch.ones_like(idx, dtype=torch.bool)
    key = attn.bool()[:, None, None, :]
    x = F.embedding(idx, m._table("in"))
    cos, sin = m.rotary(idx.shape[1], x.device, x.dtype)
    for layer in m.layers:
        if m.grad_ckpt and m.training:
            x, _ = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, None, 0, key, use_reentrant=False)
        else:
            x, _ = layer(x, cos, sin, bidir_mask=key)
    return m.norm(x)


def adapt(a):
    from bpe import BPE
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ck = load_checkpoint(a.ckpt, "cpu")
    cfg = replace(ck["cfg"], moe_impl="grouped")
    m = AnuLM(cfg).cuda()
    m.load_state_dict(ck["model"])
    tok = BPE.load(cfg.tokenizer_path)
    mask_id = tok.encode("_")[-1]                       # LLM2Vec's placeholder for models without [MASK]
    meta = json.load(open(HERE / "data" / "v2" / "meta.json"))
    data = np.memmap(HERE / "data" / "v2" / meta["train"], dtype=np.uint16, mode="r")
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    W, B = a.window, a.batch_size
    print(f"MNTP adaptation: {a.steps} steps x {B} x {W} tokens, {a.mask_frac:.0%} masked, "
          f"placeholder id {mask_id}", flush=True)
    t0 = time.time()
    m.train()
    for step in range(a.steps):
        starts = rng.integers(0, len(data) - W - 1, B)
        x = torch.from_numpy(np.stack([data[s:s + W] for s in starts]).astype(np.int64)).cuda()
        masked = torch.rand(x.shape, device=x.device) < a.mask_frac
        masked[:, 0] = False                              # position 0 has no previous position
        inp = x.masked_fill(masked, mask_id)
        lr = a.lr * min(1.0, (step + 1) / 100) * max(0.0, 1 - step / a.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        with ac:
            h = bidir_hidden(m, inp)
            # MNTP: the token at position i is predicted from position i-1
            tgt = x[:, 1:].masked_fill(~masked[:, 1:], -1)
            logits = m.lm_head(h[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        m.update_expert_biases()
        if step % 200 == 0 or step == a.steps - 1:
            print(f"step {step:5d}/{a.steps} | masked-token loss {loss.item():.4f} | {time.time() - t0:.0f}s",
                  flush=True)
    torch.save({**{k: v for k, v in ck.items() if k not in ("model", "opt", "sampler")},
                "model": m.state_dict(), "cfg": ck["cfg"], "bidirectional": True,
                "bidir_adapt": {"steps": a.steps, "window": W, "mask_frac": a.mask_frac, "from": a.ckpt}}, a.out)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("adapt")
    p.add_argument("--ckpt", default="ckpt_demo_v2.pt")
    p.add_argument("--out", default="ckpt_demo_v2_bidir.pt")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--mask-frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()
    adapt(a)


if __name__ == "__main__":
    main()
