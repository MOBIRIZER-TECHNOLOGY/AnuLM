"""
Images into AnuLM, the same way continuous audio goes in.

    python vision_encoder.py probe --image cat.jpg
    python vision_encoder.py train --ckpt ckpt_speech_init.pt --manifest captions.jsonl \
                                   --out ckpt_caption.pt --epochs 2
    python vision_encoder.py eval  --ckpt ckpt_caption.pt --manifest captions.jsonl

A frozen SigLIP tower turns a 224x224 image into 196 patch vectors, a projector
maps them into the model's hidden size, and they are handed to the backbone as
embeddings with no ids -- exactly the mechanism `speech_encoder.py` uses, which
is why that one had to exist first. The layout mirrors it too:

    <|image|> [projected patches] <|/image|> caption ... EOS

SigLIP rather than ImageNet weights, because a tower trained against text is
already pointing the direction a language model wants; an ImageNet classifier's
features are organised around a thousand labels instead.

**2x2 patch pooling.** 196 patches per image is a quarter of the 2,048-token
context spent on one picture. Pooling 2x2 neighbourhoods brings it to 49, which
is what most small VLMs settle on, and the cost is fine spatial detail this
model has no capacity to use anyway. `--pool 1` turns it off if you want to
measure that claim rather than believe it.

**What is honestly missing: a judge.** The speech work could lean on Whisper --
generate audio, transcribe it, count word errors, done. There is no equivalent
here. Captioning has automatic metrics (CIDEr, BLEU) and they are weak proxies
that reward copying the reference's phrasing, so this file reports held-out loss
and prints samples, and does not pretend to a headline number. A vision result
worth putting in RESULTS.md needs either a real metric harness or a stated,
repeatable human protocol; that decision is not made here.

**Also honestly missing: the data.** There is no image-caption corpus in this
project. The training command takes a manifest you supply. Nothing in this file
has been run against real captions -- it has been shape-checked end to end on a
synthetic image, and that is a different claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

POOL = 2                        # 2x2 patch pooling: 196 patches -> 49
DEFAULT_TOWER = "vit_base_patch16_siglip_224"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


class VisionTower:
    """A frozen SigLIP ViT. Kept out of the trained module for the same reason
    WhisperEncoder is: it is not what we are training, and 86M frozen
    parameters should not appear in an optimiser's param groups by accident."""

    def __init__(self, name: str = DEFAULT_TOWER, device: str | None = None):
        import timm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # num_classes=0 drops the head; this returns patch tokens, not a logit.
        self.model = timm.create_model(name, pretrained=True, num_classes=0).eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**cfg, is_training=False)
        self.dim = self.model.num_features
        self.name = name

    def load(self, path: str | Path) -> torch.Tensor:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def encode(self, image) -> torch.Tensor:
        """path, PIL image or batched tensor -> (patches, dim)."""
        if isinstance(image, (str, Path)):
            image = self.load(image)
        elif not torch.is_tensor(image):
            image = self.transform(image).unsqueeze(0).to(self.device)
        feats = self.model.forward_features(image)       # (1, P, D), no class token in siglip
        return feats[0].float()


class ImageProjector(nn.Module):
    """Patch vectors -> model hidden states, with optional 2x2 pooling.

    Same two-layer shape as the audio projector, and the same final LayerNorm
    for the same reason: the backbone's pre-norm blocks expect a scale, and a
    freshly projected vector that does not have it wastes the first stretch of
    training finding it.
    """

    def __init__(self, in_dim: int, out_dim: int, pool: int = POOL, hidden: int | None = None):
        super().__init__()
        self.pool = pool
        d = in_dim * pool * pool
        hidden = hidden or max(out_dim, d)
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """(P, in_dim) -> (P // pool^2, out_dim). Patches are a square grid, so
        pooling is done in 2-D: a flat stride would join the right edge of one
        row to the left edge of the next."""
        P, D = patches.shape
        if self.pool > 1:
            g = int(math.isqrt(P))
            if g * g != P:
                raise SystemExit(f"{P} patches is not a square grid; use --pool 1")
            x = patches.reshape(g, g, D)
            gp = g // self.pool
            x = x[: gp * self.pool, : gp * self.pool]
            x = x.reshape(gp, self.pool, gp, self.pool, D).permute(0, 2, 1, 3, 4)
            patches = x.reshape(gp * gp, D * self.pool * self.pool)
        return self.net(patches)


class VisionLanguage(nn.Module):
    """Backbone + image projector."""

    def __init__(self, model, vocab, in_dim: int, pool: int = POOL):
        super().__init__()
        self.model = model
        self.vocab = vocab
        self.proj = ImageProjector(in_dim, model.cfg.hidden_size, pool)

    def build(self, patches: torch.Tensor, answer_ids: list[int]):
        dev = patches.device
        emb = self.model.embed_tokens
        img = self.proj(patches)
        pre = emb(torch.tensor([self.vocab.image_bos], device=dev))
        post = emb(torch.tensor([self.vocab.image_eos], device=dev))
        ans = emb(torch.tensor(answer_ids, device=dev))
        embeds = torch.cat([pre, img, post, ans], dim=0)
        targets = torch.full((embeds.shape[0],), -1, dtype=torch.long, device=dev)
        start = pre.shape[0] + img.shape[0] + post.shape[0]
        targets[start - 1: start - 1 + len(answer_ids)] = torch.tensor(answer_ids, device=dev)
        return embeds, targets


def load_manifest(path: str) -> list[dict]:
    """jsonl of {"image": ..., "text": ...}; relative paths resolve beside it."""
    p = Path(path)
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        img = Path(r.get("image") or r.get("path"))
        rows.append({"image": str(img if img.is_absolute() else p.parent / img),
                     "text": r["text"]})
    missing = [r for r in rows if not Path(r["image"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(rows)} images are missing, "
                         f"first: {missing[0]['image']}")
    return rows


def batches(rows, tower: VisionTower, tok, vl: VisionLanguage, device: str,
            batch_size: int, shuffle: bool = True, seed: int = 0):
    import numpy as np
    order = list(range(len(rows)))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for i in range(0, len(order), batch_size):
        built = []
        for j in order[i: i + batch_size]:
            r = rows[j]
            patches = tower.encode(r["image"]).to(device)
            answer = tok.encode(" " + r["text"].strip()) + [tok.eos_id]
            built.append(vl.build(patches, answer))
        if not built:
            continue
        S = max(e.shape[0] for e, _ in built)
        eb = torch.zeros(len(built), S, built[0][0].shape[1], device=device,
                         dtype=built[0][0].dtype)
        tb = torch.full((len(built), S), -1, dtype=torch.long, device=device)
        for k, (e, t) in enumerate(built):
            eb[k, : e.shape[0]] = e
            tb[k, : t.shape[0]] = t
        yield eb, tb


@torch.no_grad()
def evaluate(vl, rows, tower, tok, device, autocast) -> float:
    vl.eval()
    tot = n = 0.0
    for eb, tb in batches(rows, tower, tok, vl, device, 1, shuffle=False):
        with autocast:
            _, loss = vl.model(None, tb, embeds=eb)
        k = int((tb != -1).sum())
        tot += loss.item() * k
        n += k
    vl.train()
    return tot / max(n, 1)


def train(args) -> None:
    from dataclasses import replace

    from bpe import BPE
    from model import AnuLM, load_checkpoint
    from speech_vocab import vocab_of
    from train import atomic_save

    dev = args.device
    ck = load_checkpoint(args.ckpt, dev)
    cfg = replace(ck["cfg"], moe_impl="grouped" if dev.startswith("cuda") else "sparse")
    tok = BPE.load(cfg.tokenizer_path)
    model = AnuLM(cfg).to(dev)
    model.load_state_dict(ck["model"])

    tower = VisionTower(args.tower, dev)
    vl = VisionLanguage(model, vocab_of(ck), tower.dim, args.pool).to(dev)
    print(f"tower: {args.tower}, {tower.dim}-dim, frozen")
    print(f"projector: {tower.dim} x{args.pool*args.pool} -> {cfg.hidden_size}, "
          f"{sum(p.numel() for p in vl.proj.parameters())/1e6:.2f}M params")

    rows = load_manifest(args.manifest)
    cut = max(1, int(len(rows) * 0.02))
    held, rows = rows[:cut], rows[cut:]
    steps = max(1, int(math.ceil(len(rows) / args.batch_size * args.epochs)))
    print(f"{len(rows)} images, {len(held)} held out, {steps} steps")

    opt = torch.optim.AdamW(
        [{"params": vl.proj.parameters(), "lr": args.proj_lr},
         {"params": model.parameters(), "lr": args.lr}],
        weight_decay=0.1, betas=(0.9, 0.95))
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if dev.startswith("cuda") else torch.autocast("cpu", enabled=False))

    vl.train()
    step, t0 = 0, time.time()
    for epoch in range(math.ceil(args.epochs)):
        for eb, tb in batches(rows, tower, tok, vl, dev, args.batch_size, seed=args.seed + epoch):
            if step >= steps:
                break
            with autocast:
                _, loss = model(None, tb, embeds=eb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vl.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % args.log_every == 0:
                print(f"step {step:6d} | loss {loss.item():.4f} | {time.time()-t0:5.0f}s",
                      flush=True)
            step += 1
        if step >= steps:
            break

    val = evaluate(vl, held, tower, tok, dev, autocast)
    atomic_save({"model": model.state_dict(), "proj": vl.proj.state_dict(),
                 "cfg": replace(cfg, moe_impl=ck["cfg"].moe_impl),
                 "speech_text_vocab": ck["speech_text_vocab"],
                 "tower": args.tower, "pool": args.pool,
                 "step": step, "val_loss": val, "base_ckpt": args.ckpt}, args.out)
    print(f"\ndone in {time.time()-t0:.0f}s | held-out loss {val:.4f} | {args.out}")


def load_trained(ckpt: str, device: str):
    from dataclasses import replace

    from bpe import BPE
    from model import AnuLM, load_checkpoint
    from speech_vocab import vocab_of
    ck = load_checkpoint(ckpt, device)
    if "proj" not in ck or "tower" not in ck:
        raise SystemExit(f"{ckpt} has no image projector")
    cfg = replace(ck["cfg"], moe_impl="grouped" if device.startswith("cuda") else "sparse")
    model = AnuLM(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    tower = VisionTower(ck["tower"], device)
    vl = VisionLanguage(model, vocab_of(ck), tower.dim, ck.get("pool", POOL)).to(device).eval()
    vl.proj.load_state_dict(ck["proj"])
    return vl, tower, BPE.load(cfg.tokenizer_path)


@torch.no_grad()
def caption(vl, tower, tok, image: str, max_tokens: int = 40, device: str = "cuda") -> str:
    """Greedy, no KV cache -- same O(n^2) prefix replay as the speech decoder,
    for the same reason (the prompt is a matrix, not ids)."""
    patches = tower.encode(image).to(device)
    emb = vl.model.embed_tokens
    prefix = torch.cat([
        emb(torch.tensor([vl.vocab.image_bos], device=device)), vl.proj(patches),
        emb(torch.tensor([vl.vocab.image_eos], device=device))], dim=0)
    out: list[int] = []
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device.startswith("cuda") else torch.autocast("cpu", enabled=False))
    for _ in range(max_tokens):
        cur = prefix if not out else torch.cat([prefix, emb(torch.tensor(out, device=device))])
        with autocast:
            logits, _ = vl.model(None, embeds=cur[None])
        nxt = int(logits[0, -1].argmax())
        if nxt == tok.eos_id or nxt >= vl.vocab.text:
            break
        out.append(nxt)
    return tok.decode(out).strip().split("\n")[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Images into AnuLM.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="shapes and context cost for one image")
    pr.add_argument("--image", required=True)
    pr.add_argument("--tower", default=DEFAULT_TOWER)
    pr.add_argument("--pool", type=int, default=POOL)

    t = sub.add_parser("train", help="train the projector and backbone to caption")
    t.add_argument("--ckpt", required=True)
    t.add_argument("--manifest", required=True, help="jsonl of {image, text}")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--proj-lr", type=float, default=1e-3)
    t.add_argument("--tower", default=DEFAULT_TOWER)
    t.add_argument("--pool", type=int, default=POOL)
    t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    e = sub.add_parser("eval", help="held-out loss and sample captions")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--manifest", required=True)
    e.add_argument("--limit", type=int, default=20)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    if args.cmd == "probe":
        tower = VisionTower(args.tower)
        patches = tower.encode(args.image)
        proj = ImageProjector(tower.dim, 1024, args.pool).to(patches.device)
        out = proj(patches)
        print(f"{Path(args.image).name}")
        print(f"  tower     {tuple(patches.shape)}  ({args.tower})")
        print(f"  projected {tuple(out.shape)}  pool {args.pool}x{args.pool}")
        print(f"  one image costs {out.shape[0]} of 2048 positions "
              f"({100*out.shape[0]/2048:.1f}%)")
    elif args.cmd == "train":
        train(args)
    else:
        vl, tower, tok = load_trained(args.ckpt, args.device)
        rows = load_manifest(args.manifest)[: args.limit]
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if args.device.startswith("cuda") else torch.autocast("cpu", enabled=False))
        for i, r in enumerate(rows[:5]):
            print(f"  [{i}] reference {r['text']}")
            print(f"      model     {caption(vl, tower, tok, r['image'], device=args.device) or '(nothing)'}")
        print(f"\nheld-out loss {evaluate(vl, rows, tower, tok, args.device, autocast):.4f} "
              f"over {len(rows)} images (no automatic caption metric -- read the samples)")


if __name__ == "__main__":
    main()
