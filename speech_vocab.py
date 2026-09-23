"""
Making room in AnuLM's vocabulary for speech.

    python speech_vocab.py --ckpt toonist/AnuLM-Base-400M --out ckpt_speech_init.pt
    python speech_vocab.py --ckpt ckpt_speech_init.pt --describe

A text model cannot emit audio, so give it the words for it. SNAC turns speech
into 7 codes per frame (`audio_codec.py`), each from a 4,096-entry codebook, and
this file appends all 28,672 of them to the vocabulary along with three control
tokens. Everything the checkpoint already knew keeps its id, so the model that
comes out of this is exactly the model that went in, plus some rows it has not
learned to use yet.

    [0, V)                     the BPE tokens the checkpoint was trained on
    [V, V + 28672)             SNAC codes, slot-major: V + slot*4096 + code
    V + 28672 + 0              <|speech|>   audio follows
    V + 28672 + 1              <|/speech|>  audio ends
    V + 28672 + 2              <|text|>     transcript follows

Slot-major matters. The seven slots of a frame come from three codec levels at
three different rates, and giving each slot its own block means position in the
frame identifies the level -- the model is never asked to work out whether an
id is coarse or fine, which is the kind of thing a 400M model spends capacity
on if you let it.

**The cost, and it is the number that decides your schedule.** 28,675 new ids at
hidden size 1024, untied (Sarvam unties, and so does AnuLM), is two matrices:
29.4M parameters in `embed_tokens` and another 29.4M in `lm_head`. Measured on
`ckpt_ctx2k.pt`, that takes the checkpoint from 398M/173.5M to **456M total and
232M active** -- 15% more parameters but **34% more active parameters**, because
embeddings are dense and the experts they join are not. Active parameters are
what set tokens/second, so budget the audio runs at roughly three quarters of
the 10.2k tok/s the text runs managed, not at parity.

Halving that bill is possible and untried here: tie `lm_head` to
`embed_tokens` for the audio block only, or factorise the audio rows as
7 slot vectors plus 4,096 code vectors. Both are worth a RESULTS.md line if
the dense embedding turns out to be what limits the run.

New rows are initialised from the mean of the existing embeddings plus small
noise, not from scratch: a fresh N(0, 0.02) row starts far outside the region
the trained model's norms occupy, and the first few hundred steps are then spent
travelling rather than learning.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

from audio_codec import AUDIO_VOCAB, FRAME_SLOTS, LEVEL_VOCAB

# Image tokens are here, in the speech file, because the vocabulary is grown
# exactly once: adding them later would shift every id after the audio block
# and invalidate every checkpoint trained before the change. Two ids cost
# 4,096 parameters and buy the option.
CONTROL = ["<|speech|>", "<|/speech|>", "<|text|>", "<|image|>", "<|/image|>"]
N_CONTROL = len(CONTROL)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


class SpeechVocab:
    """Where everything lives once the vocabulary has grown.

    `text` is the original vocab size, which is also the base of the audio
    block; a checkpoint that has already been grown carries `speech_text_vocab`
    so this stays right on the second pass.
    """

    def __init__(self, text_vocab: int):
        self.text = text_vocab
        self.audio_base = text_vocab
        self.speech_bos = text_vocab + AUDIO_VOCAB
        self.speech_eos = self.speech_bos + 1
        self.text_bos = self.speech_bos + 2
        self.image_bos = self.speech_bos + 3
        self.image_eos = self.speech_bos + 4
        self.total = text_vocab + AUDIO_VOCAB + N_CONTROL

    def audio(self, ids) -> torch.Tensor:
        """SNAC flat ids (0..28671) -> vocabulary ids."""
        return torch.as_tensor(ids, dtype=torch.long) + self.audio_base

    def to_snac(self, ids) -> torch.Tensor:
        """Vocabulary ids -> SNAC flat ids, dropping anything that is not audio."""
        t = torch.as_tensor(ids, dtype=torch.long) - self.audio_base
        return t[(t >= 0) & (t < AUDIO_VOCAB)]

    def is_audio(self, ids) -> torch.Tensor:
        t = torch.as_tensor(ids, dtype=torch.long)
        return (t >= self.audio_base) & (t < self.audio_base + AUDIO_VOCAB)

    def describe(self) -> str:
        return (f"text [0, {self.text})  audio [{self.audio_base}, "
                f"{self.audio_base + AUDIO_VOCAB})  "
                f"speech {self.speech_bos}/{self.speech_eos}  "
                f"text_bos {self.text_bos}  "
                f"image {self.image_bos}/{self.image_eos}  total {self.total}")


def _grow(weight: torch.Tensor, n_new: int, gen: torch.Generator) -> torch.Tensor:
    """Append rows drawn around the existing distribution's mean."""
    mean = weight.mean(dim=0, keepdim=True)
    std = weight.std().item()
    noise = torch.randn(n_new, weight.shape[1], generator=gen,
                        dtype=weight.dtype) * (std * 0.1)
    return torch.cat([weight, mean.expand(n_new, -1) + noise], dim=0)


def grow_checkpoint(ckpt: str, out: str, seed: int = 0, device: str = "cpu") -> SpeechVocab:
    """Load a checkpoint, widen both embedding matrices, save it back.

    Both matrices: AnuLM unties `lm_head` from `embed_tokens`, so growing only
    one leaves the model able to read audio and unable to write it (or the
    reverse), which fails as a shape error several minutes into a training run
    rather than here.
    """
    from model import load_checkpoint
    from train import atomic_save

    ck = load_checkpoint(ckpt, device)
    cfg, sd = ck["cfg"], ck["model"]
    text_vocab = ck.get("speech_text_vocab") or cfg.vocab_size
    v = SpeechVocab(text_vocab)

    if cfg.vocab_size == v.total:
        print(f"already grown: {v.describe()}")
        return v
    if cfg.vocab_size != text_vocab:
        raise SystemExit(f"checkpoint vocab {cfg.vocab_size} is neither the text vocab "
                         f"{text_vocab} nor the grown size {v.total}; refusing to guess")

    gen = torch.Generator().manual_seed(seed)
    n_new = v.total - text_vocab
    for key in ("embed_tokens.weight", "lm_head.weight"):
        full = next((k for k in sd if k.endswith(key)), None)
        if full is None:
            raise SystemExit(f"no {key} in the checkpoint; keys look like "
                             f"{list(sd)[:3]}")
        before = sd[full].shape
        sd[full] = _grow(sd[full].float(), n_new, gen).to(sd[full].dtype)
        print(f"  {full}: {tuple(before)} -> {tuple(sd[full].shape)}")

    from dataclasses import replace
    ck["cfg"] = replace(cfg, vocab_size=v.total)
    ck["speech_text_vocab"] = text_vocab
    ck["base_ckpt"] = ckpt
    atomic_save(ck, out)
    added = n_new * cfg.hidden_size * 2
    print(f"\n{v.describe()}\n"
          f"+{n_new} ids, +{added/1e6:.1f}M parameters -> {out}")
    return v


def vocab_of(ck: dict) -> SpeechVocab:
    """The layout for an already-grown checkpoint."""
    text = ck.get("speech_text_vocab")
    if text is None:
        raise SystemExit("this checkpoint has no audio vocabulary; run speech_vocab.py first")
    return SpeechVocab(text)


def main() -> None:
    p = argparse.ArgumentParser(description="Add SNAC codes to a checkpoint's vocabulary.")
    p.add_argument("--ckpt", required=True, help="a .pt, an exported folder, or a Hub repo id")
    p.add_argument("--out", help="where to write the grown checkpoint")
    p.add_argument("--describe", action="store_true", help="print the layout and exit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.describe:
        from model import load_checkpoint
        ck = load_checkpoint(args.ckpt, "cpu")
        text = ck.get("speech_text_vocab") or ck["cfg"].vocab_size
        v = SpeechVocab(text)
        grown = ck["cfg"].vocab_size == v.total
        print(f"{Path(args.ckpt).name}: vocab {ck['cfg'].vocab_size} "
              f"({'grown' if grown else 'text only'})")
        print(v.describe())
        print(f"\n{FRAME_SLOTS} slots x {LEVEL_VOCAB} codes = {AUDIO_VOCAB} audio ids")
        return

    if not args.out:
        p.error("--out is required unless --describe")
    grow_checkpoint(args.ckpt, args.out, args.seed)


if __name__ == "__main__":
    main()
