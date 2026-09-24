"""
The other way in: audio as continuous states rather than as tokens.

    python speech_encoder.py train --ckpt ckpt_speech_init.pt --prefix data/speech_ls100 \
                                   --out ckpt_asr_ct.pt --epochs 2
    python speech_encoder.py eval  --ckpt ckpt_asr_ct.pt --prefix data/speech_ls100
    python speech_encoder.py probe --prefix data/speech_ls100

`speech_data.py` + `finetune.py --speech --task asr` teaches the model to read
audio as 28,672 discrete SNAC ids. This file does the same job the other way:
Whisper's encoder emits 50 Hz of 768-dim states, a small projector maps four
of them at a time into the model's hidden size, and those vectors are fed
straight into the backbone with no id and no softmax over them. It is the input
half of how Qwen3-Omni is built, and the point of having both is to find out
which one actually wins at 232M active parameters -- a question this project
can answer by running it, which is cheaper than having an opinion.

    discrete    wav -> SNAC -> 83.3 ids/s -> embed_tokens -> backbone
    continuous  wav -> Whisper enc -> 50 states/s -> stack 4 -> project -> backbone

**What is trained.** The projector only, plus the backbone. The Whisper encoder
is frozen -- ctranslate2 runs it as a C++ graph and no gradient can flow back
through it anyway, which happens to be the right call at this scale: 88M
encoder parameters would otherwise dominate a 232M model.

**Why stack four.** 50 Hz is 2,048 positions for 41 s of audio, which sounds
generous until you notice it is four times the frame rate SNAC needs for the
same speech. Stacking 4 frames into one 3,072-dim vector brings it to 12.5 Hz
-- 164 s of context, a quarter of the attention cost, and the same rate as
SNAC's coarse level, which makes the two paths comparable per position instead
of per second.

**Why this file has its own loop.** `finetune.py` packs many pairs into one
`block_size` row of *ids*. A continuous prompt has no ids to pack, so rows here
are one clip each, padded, with the loss masked to the transcript. That is less
efficient than packing and it is not a rewrite of the text path: it is the
smallest honest way to train something whose prompt is a matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_codec import ASR_RATE, WhisperEncoder, read_wav

STACK = 4                      # 50 Hz -> 12.5 Hz
MAX_AUDIO_FRAMES = 400         # 400 stacked frames = 32 s, well inside 2048

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


class AudioProjector(nn.Module):
    """Whisper states -> model hidden states. Two layers, not one.

    A single Linear is the common choice and it underfits here: the encoder's
    space and the backbone's have nothing to do with each other, and the
    projector is the only thing allowed to learn the mapping (the encoder is
    frozen). The final LayerNorm matters more than the width -- without it the
    projected vectors arrive at a scale the pre-norm blocks were never trained
    for and the first hundred steps are spent renormalising.
    """

    def __init__(self, in_dim: int, out_dim: int, stack: int = STACK, hidden: int | None = None):
        super().__init__()
        self.stack = stack
        d = in_dim * stack
        hidden = hidden or max(out_dim, d)
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """(B, T, in_dim) -> (B, T // stack, out_dim). A partial final group is
        dropped rather than zero-padded: a padded group looks like silence the
        model must learn to ignore, for no gain."""
        B, T, D = states.shape
        T = T - T % self.stack
        if T == 0:
            return states.new_zeros(B, 0, self.net[-1].normalized_shape[0])
        return self.net(states[:, :T].reshape(B, T // self.stack, D * self.stack))


class ContinuousSpeech(nn.Module):
    """Backbone + projector. The frozen ear lives outside, in WhisperEncoder,
    because it is not a torch module here at all."""

    def __init__(self, model, vocab, in_dim: int, stack: int = STACK):
        super().__init__()
        self.model = model
        self.vocab = vocab
        self.proj = AudioProjector(in_dim, model.cfg.hidden_size, stack)

    def build(self, states: torch.Tensor, text_ids: list[int], answer_ids: list[int]):
        """One example -> (embeds, targets).

        Layout mirrors the discrete ASR pair exactly, so the two runs differ in
        the audio representation and in nothing else:

            <|speech|> [projected audio] <|text|> transcript ... EOS
        """
        dev = states.device
        emb = self.model.embed_tokens
        audio = self.proj(states[None])[0]                       # (Ta, D)
        pre = emb(torch.tensor([self.vocab.speech_bos], device=dev))
        mid = emb(torch.tensor([self.vocab.text_bos], device=dev))
        ans = emb(torch.tensor(answer_ids, device=dev))
        embeds = torch.cat([pre, audio, mid, ans], dim=0)
        # -1 everywhere the loss must not look: the prompt, and the audio that
        # has no id to predict in the first place.
        targets = torch.full((embeds.shape[0],), -1, dtype=torch.long, device=dev)
        start = pre.shape[0] + audio.shape[0] + mid.shape[0]
        targets[start - 1: start - 1 + len(answer_ids)] = torch.tensor(answer_ids, device=dev)
        return embeds, targets


def load_manifest(prefix: str) -> list[dict]:
    rows = [json.loads(l) for l in
            Path(prefix).with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    missing = [r for r in rows if not r.get("audio") or not Path(r["audio"]).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(rows)} rows have no readable `audio` path. "
            f"The continuous path needs the waveforms; re-run speech_data.py encode "
            f"(manifests written before the path was recorded do not carry it).")
    return rows


def batches(rows, enc: WhisperEncoder, tok, sm: ContinuousSpeech, device: str,
            batch_size: int, shuffle: bool = True, seed: int = 0):
    """Pad-to-longest batches of (embeds, targets). Whisper runs per clip and
    is the slow part; it is frozen, so its output could be cached per epoch if
    a run ever becomes encoder-bound."""
    order = list(range(len(rows)))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [rows[j] for j in order[i: i + batch_size]]
        built = []
        for r in chunk:
            wav = read_wav(r["audio"], ASR_RATE)
            states = torch.from_numpy(enc.encode(wav)).to(device)
            if states.shape[0] // sm.proj.stack > MAX_AUDIO_FRAMES:
                continue
            answer = tok.encode(" " + r["text"].strip()) + [tok.eos_id]
            built.append(sm.build(states, [], answer))
        if not built:
            continue
        S = max(e.shape[0] for e, _ in built)
        D = built[0][0].shape[1]
        eb = torch.zeros(len(built), S, D, device=device, dtype=built[0][0].dtype)
        tb = torch.full((len(built), S), -1, dtype=torch.long, device=device)
        for k, (e, t) in enumerate(built):
            eb[k, : e.shape[0]] = e
            tb[k, : t.shape[0]] = t
        yield eb, tb


def train(args) -> None:
    from dataclasses import replace

    from bpe import BPE
    from model import AnuLM, load_checkpoint
    from speech_vocab import vocab_of
    from train import atomic_save

    dev = args.device
    ck = load_checkpoint(args.ckpt, dev)
    cfg = replace(ck["cfg"], moe_impl="grouped" if dev.startswith("cuda") else "sparse")
    vocab = vocab_of(ck)
    tok = BPE.load(cfg.tokenizer_path)
    model = AnuLM(cfg).to(dev)
    model.load_state_dict(ck["model"])

    enc = WhisperEncoder(args.whisper, dev)
    sm = ContinuousSpeech(model, vocab, enc.dim, args.stack).to(dev)
    n_proj = sum(p.numel() for p in sm.proj.parameters())
    print(f"projector: {enc.dim} x{args.stack} -> {cfg.hidden_size}, {n_proj/1e6:.2f}M params "
          f"(whisper-{args.whisper} frozen)")

    rows = load_manifest(args.prefix)
    cut = max(1, int(len(rows) * 0.02))
    held, rows = rows[:cut], rows[cut:]
    steps = max(1, int(math.ceil(len(rows) / args.batch_size * args.epochs)))
    print(f"{len(rows)} clips, {len(held)} held out, {steps} steps at batch {args.batch_size}")

    # The projector starts from nothing and wants a livelier rate than a
    # backbone that is already trained; one group each rather than one rate.
    # fused=True on CUDA: one kernel for the whole step instead of a launch per
    # parameter tensor. finetune.py has always logged "AdamW (fused)"; these two
    # trainers were written without it and paid for the omission on a 456M model.
    opt = torch.optim.AdamW(
        [{"params": sm.proj.parameters(), "lr": args.proj_lr},
         {"params": model.parameters(), "lr": args.lr}],
        weight_decay=0.1, betas=(0.9, 0.95), fused=dev.startswith("cuda"))
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if dev.startswith("cuda") else torch.autocast("cpu", enabled=False))

    sm.train()
    step = 0
    t0 = time.time()
    for epoch in range(math.ceil(args.epochs)):
        for eb, tb in batches(rows, enc, tok, sm, dev, args.batch_size, seed=args.seed + epoch):
            if step >= steps:
                break
            with autocast:
                # Targets go *in*: without them forward takes its inference
                # shortcut and returns only the last position, and the loss it
                # returns carries the router regulariser an MoE needs.
                _, loss = model(None, tb, embeds=eb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sm.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % args.log_every == 0:
                print(f"step {step:6d} | loss {loss.item():.4f} | {time.time()-t0:5.0f}s",
                      flush=True)
            step += 1
        if step >= steps:
            break

    val = evaluate(sm, held, enc, tok, dev, autocast)
    atomic_save({"model": model.state_dict(), "proj": sm.proj.state_dict(),
                 "cfg": replace(cfg, moe_impl=ck["cfg"].moe_impl),
                 "speech_text_vocab": ck["speech_text_vocab"],
                 "whisper": args.whisper, "stack": args.stack,
                 "step": step, "val_loss": val, "base_ckpt": args.ckpt}, args.out)
    print(f"\ndone in {time.time()-t0:.0f}s | held-out loss {val:.4f} | {args.out}")


@torch.no_grad()
def evaluate(sm, rows, enc, tok, dev, autocast) -> float:
    sm.eval()
    tot = n = 0.0
    for eb, tb in batches(rows, enc, tok, sm, dev, 1, shuffle=False):
        with autocast:
            _, loss = sm.model(None, tb, embeds=eb)
        k = int((tb != -1).sum())
        tot += loss.item() * k
        n += k
    sm.train()
    return tot / max(n, 1)


def load_trained(ckpt: str, device: str):
    """A checkpoint written by `train` -> (ContinuousSpeech, encoder, tokeniser)."""
    from dataclasses import replace

    from bpe import BPE
    from model import AnuLM, load_checkpoint
    from speech_vocab import vocab_of
    ck = load_checkpoint(ckpt, device)
    if "proj" not in ck:
        raise SystemExit(f"{ckpt} has no projector; it is a discrete-path checkpoint")
    cfg = replace(ck["cfg"], moe_impl="grouped" if device.startswith("cuda") else "sparse")
    model = AnuLM(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    enc = WhisperEncoder(ck.get("whisper", "small"), device)
    sm = ContinuousSpeech(model, vocab_of(ck), enc.dim, ck.get("stack", STACK)).to(device).eval()
    sm.proj.load_state_dict(ck["proj"])
    return sm, enc, BPE.load(cfg.tokenizer_path)


@torch.no_grad()
def transcribe(sm, enc, tok, wav_path: str, max_tokens: int = 64, device: str = "cuda") -> str:
    """Greedy decode from a continuous prompt, one token at a time.

    No KV cache: the prompt is embeddings rather than ids, and `generate` takes
    ids. Re-running the prefix each step is O(n^2) and fine for evaluation --
    caching it is the obvious optimisation if this ever goes in a demo.
    """
    states = torch.from_numpy(enc.encode(read_wav(wav_path, ASR_RATE))).to(device)
    emb = sm.model.embed_tokens
    audio = sm.proj(states[None])[0]
    prefix = torch.cat([
        emb(torch.tensor([sm.vocab.speech_bos], device=device)), audio,
        emb(torch.tensor([sm.vocab.text_bos], device=device))], dim=0)
    out: list[int] = []
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if device.startswith("cuda") else torch.autocast("cpu", enabled=False))
    for _ in range(max_tokens):
        cur = (prefix if not out else
               torch.cat([prefix, emb(torch.tensor(out, device=device))], dim=0))
        with autocast:
            logits, _ = sm.model(None, embeds=cur[None])
        nxt = int(logits[0, -1].argmax())
        if nxt == tok.eos_id or nxt >= sm.vocab.text:
            break
        out.append(nxt)
    return tok.decode(out).strip().split("\n")[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Audio into AnuLM as continuous states.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the projector and the backbone for ASR")
    t.add_argument("--ckpt", required=True, help="a checkpoint grown by speech_vocab.py")
    t.add_argument("--prefix", required=True, help="an encoded corpus (needs audio paths)")
    t.add_argument("--out", required=True)
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--proj-lr", type=float, default=1e-3)
    t.add_argument("--whisper", default="small")
    t.add_argument("--stack", type=int, default=STACK)
    t.add_argument("--log-every", type=int, default=25)
    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    e = sub.add_parser("eval", help="WER for the continuous path")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--prefix", required=True)
    e.add_argument("--limit", type=int, default=200)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    pr = sub.add_parser("probe", help="shapes and rates for one clip, no training")
    pr.add_argument("--prefix", required=True)
    pr.add_argument("--whisper", default="small")
    pr.add_argument("--stack", type=int, default=STACK)

    args = p.parse_args()

    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        from eval_speech import wer
        sm, enc, tok = load_trained(args.ckpt, args.device)
        rows = load_manifest(args.prefix)[: args.limit]
        edits = length = 0
        for i, r in enumerate(rows):
            hyp = transcribe(sm, enc, tok, r["audio"], device=args.device)
            d, n = wer(r["text"], hyp)
            edits, length = edits + d, length + n
            if i < 3:
                print(f"  [{i}] reference {r['text']}\n      model     {hyp or '(nothing)'}")
        print(f"\nASR (continuous)  WER {100*edits/max(length,1):.1f}%  over {len(rows)} clips")
    else:
        rows = load_manifest(args.prefix)
        enc = WhisperEncoder(args.whisper)
        r = rows[0]
        states = enc.encode(read_wav(r["audio"], ASR_RATE))
        stacked = states.shape[0] // args.stack
        print(f"{Path(r['audio']).name}  {r['seconds']}s")
        print(f"  whisper   {states.shape[0]} x {states.shape[1]}  "
              f"({states.shape[0]/r['seconds']:.1f} Hz)")
        print(f"  stacked   {stacked} positions  ({stacked/r['seconds']:.1f} Hz)")
        print(f"  discrete  {r['length']} ids  ({r['length']/r['seconds']:.1f} Hz) for the same clip")
        print(f"\n  2048 positions = {2048/(stacked/r['seconds']):.0f}s continuous, "
              f"{2048/(r['length']/r['seconds']):.0f}s discrete")


if __name__ == "__main__":
    main()
