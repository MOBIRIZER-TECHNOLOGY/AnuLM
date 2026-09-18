"""
Does YaRN actually buy anything? Measure a model past its training context.

    python eval_context.py --ckpt ckpt_105b.pt

The 105B's whole long-context story is YaRN: trained at 4096, served at 128K.
This reproduces that experiment at nano scale. Take a checkpoint trained at
block_size B with plain RoPE, then evaluate at 2B and 4B two ways:

  naive   -- same rotary, just feed longer sequences. The slow frequency dims
             are now at phases never seen in training, so this should degrade.
  yarn    -- rebuild the rotary with NTK-by-parts band blending and the logit
             temperature term, factor = length/B. Same weights, no retraining.

No fine-tuning at the longer length. This is zero-shot extension, which is the
strongest form of the claim and the hardest case for it.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import torch

from model import AnuLM

HERE = Path(__file__).parent


def load_val(path: Path, cfg, frac: float = 0.1) -> torch.Tensor:
    """The last `frac` of the file as ids in the checkpoint's own vocabulary:
    raw bytes for a byte model, BPE ids (plain encode, no EOS) otherwise."""
    if getattr(cfg, "tokenizer_path", None):
        from bpe import BPE, DOC_SEP
        text = path.read_text(encoding="utf-8").replace(DOC_SEP, "\n\n")
        text = text[int((1 - frac) * len(text)):]
        return torch.tensor(BPE.load(cfg.tokenizer_path).encode(text), dtype=torch.long)
    raw = path.read_bytes()
    data = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return data[int((1 - frac) * len(data)):]


DEVICE = "cpu"


@torch.no_grad()
def eval_at(state, cfg, data, length: int, batches: int, seed: int = 0) -> float:
    """Mean loss over fixed windows of `length` tokens."""
    m = AnuLM(cfg).to(DEVICE).eval()
    m.load_state_dict(state)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        i = torch.randint(len(data) - length - 1, (1,), generator=g).item()
        x = data[i: i + length].long().unsqueeze(0).to(DEVICE)
        y = data[i + 1: i + 1 + length].long().unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.startswith("cuda")):
            _, loss = m(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def eval_by_position(state, cfg, data, length: int, batches: int, bins: int = 4,
                     seed: int = 0):
    """Loss bucketed by position in the window -- extension failures show up at
    the far end, which a single averaged number hides."""
    m = AnuLM(cfg).to(DEVICE).eval()
    m.load_state_dict(state)
    g = torch.Generator().manual_seed(seed)
    edges = [length * b // bins for b in range(bins + 1)]
    totals = [0.0] * bins
    for _ in range(batches):
        i = torch.randint(len(data) - length - 1, (1,), generator=g).item()
        x = data[i: i + length].long().unsqueeze(0).to(DEVICE)
        y = data[i + 1: i + 1 + length].long().unsqueeze(0).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.startswith("cuda")):
            logits, _ = m(x, y)
        tok = torch.nn.functional.cross_entropy(
            logits[0].float(), y[0], reduction="none")
        for b in range(bins):
            totals[b] += tok[edges[b]: edges[b + 1]].mean().item()
    return [t / batches for t in totals]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ckpt_105b.pt")
    p.add_argument("--data", type=str, default=str(HERE / "data" / "input.txt"))
    p.add_argument("--batches", type=int, default=24)
    p.add_argument("--multipliers", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--device", default="cpu",
                   help="cuda for the 350M checkpoints; CPU is fine for the 17M/47M")
    args = p.parse_args()
    global DEVICE
    DEVICE = args.device

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    base_cfg, state = ck["cfg"], ck["model"]
    trained_at = base_cfg.block_size
    data = load_val(Path(args.data), base_cfg)
    unit = "tokens" if getattr(base_cfg, "tokenizer_path", None) else "bytes"
    print(f"{args.ckpt}: attn={base_cfg.attn}, trained at block_size={trained_at}, "
          f"val loss {ck['val_loss']:.4f}"
          + (f", window {base_cfg.sliding_window} on {base_cfg.max_window_layers} layers"
             if getattr(base_cfg, "sliding_window", None) else ""))
    print(f"val {unit} {len(data)/1e6:.2f}M   {args.batches} windows per setting\n")

    print(f"{'context':>8}  {'naive':>8}  {'yarn':>8}  {'delta':>8}   verdict")
    print("-" * 52)
    rows = []
    for mult in args.multipliers:
        length = trained_at * mult
        naive_cfg = replace(base_cfg, block_size=length, yarn=False)
        naive = eval_at(state, naive_cfg, data, length, args.batches)
        if mult == 1:
            print(f"{length:>8}  {naive:>8.4f}  {'-':>8}  {'-':>8}   training length")
            rows.append((length, naive, None))
            continue
        yarn_cfg = replace(base_cfg, block_size=length, yarn=True,
                           yarn_factor=float(mult), yarn_original_context=trained_at)
        yarned = eval_at(state, yarn_cfg, data, length, args.batches)
        delta = yarned - naive
        verdict = "YaRN helps" if delta < -0.01 else (
            "YaRN hurts" if delta > 0.01 else "no difference")
        print(f"{length:>8}  {naive:>8.4f}  {yarned:>8.4f}  {delta:>+8.4f}   {verdict}")
        rows.append((length, naive, yarned))

    # Where does it break? Average hides a cliff at the far end of the window.
    longest = trained_at * max(args.multipliers)
    if longest > trained_at:
        print(f"\nloss by position within a {longest}-{unit[:-1]} window "
              f"(model trained at {trained_at}):")
        naive_cfg = replace(base_cfg, block_size=longest, yarn=False)
        yarn_cfg = replace(base_cfg, block_size=longest, yarn=True,
                           yarn_factor=float(longest // trained_at),
                           yarn_original_context=trained_at)
        nb = eval_by_position(state, naive_cfg, data, longest, args.batches)
        yb = eval_by_position(state, yarn_cfg, data, longest, args.batches)
        q = longest // 4
        print(f"  {'positions':>14}  {'naive':>8}  {'yarn':>8}")
        for b in range(4):
            lo, hi = b * q, (b + 1) * q
            tag = "  <- in-distribution" if hi <= trained_at else ""
            print(f"  {f'{lo}-{hi}':>14}  {nb[b]:>8.4f}  {yb[b]:>8.4f}{tag}")


if __name__ == "__main__":
    main()
