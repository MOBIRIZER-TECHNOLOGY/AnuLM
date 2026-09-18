"""
Score checkpoints on ONE fixed held-out file, each with its own tokenizer.

    python eval_bench.py ckpt_hindi_v3.pt ckpt_hindi_v4.pt --bench data/bench_hindi.txt

Why this exists: every run's own val split is different text, so own-val
numbers cannot be compared across corpora or tokenizers (docs/RESULTS.md
section 8 learned that the hard way). bits/byte on a shared file is the axis
that compares a byte model, a 16k-BPE model and a 32k-BPE model honestly --
the loss per token is meaningless across vocabularies, the loss per byte of
the same text is not.

The file is scored in non-overlapping windows of the checkpoint's block_size,
plain `encode` (no EOS -- older models never saw one), full-sequence loss.
Also reports bits/char, which is what to quote for Devanagari (2.5 bytes/char).
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from bpe import BPE, DOC_SEP
from model import AnuLM


@torch.no_grad()
def score(ckpt: str, text: str, device: str, batch_size: int = 8) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = AnuLM(cfg).to(device).eval()
    m.load_state_dict(ck["model"])

    n_bytes = len(text.encode("utf-8"))
    n_chars = len(text)
    if getattr(cfg, "tokenizer_path", None):
        tok = BPE.load(cfg.tokenizer_path)
        ids = tok.encode(text)
        tokenizer = f"{Path(cfg.tokenizer_path).stem} ({cfg.vocab_size})"
    else:
        ids = list(text.encode("utf-8"))
        tokenizer = "bytes (259)"
    ids = torch.tensor(ids, dtype=torch.long)

    bs = cfg.block_size
    n_win = (len(ids) - 1) // bs
    ids = ids[: n_win * bs + 1]
    x = ids[:-1].view(n_win, bs)
    y = ids[1:].view(n_win, bs)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    nll_sum, n_tok, t0 = 0.0, 0, time.time()
    for i in range(0, n_win, batch_size):
        xb, yb = x[i: i + batch_size].to(device), y[i: i + batch_size].to(device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda")):
            _, loss = m(xb, yb)
        nll_sum += loss.item() * xb.numel()
        n_tok += xb.numel()
    nats_per_token = nll_sum / n_tok
    # The scored span covers n_tok tokens; convert with the file's own ratio.
    bytes_per_token = n_bytes / len(ids)
    bits_per_byte = nats_per_token / math.log(2) / bytes_per_token
    return {
        "ckpt": ckpt, "step": ck.get("step"), "own_val": ck.get("val_loss"),
        "tokenizer": tokenizer, "block": bs, "attn": cfg.attn,
        "params": sum(p.numel() for p in m.parameters()),
        "tokens": n_tok, "bytes_per_token": bytes_per_token,
        "nats_per_token": nats_per_token, "bits_per_byte": bits_per_byte,
        "bits_per_char": bits_per_byte * n_bytes / n_chars,
        "seconds": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--bench", default="data/bench_hindi.txt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    text = Path(args.bench).read_text(encoding="utf-8")
    n_docs = sum(1 for d in text.split(DOC_SEP) if d.strip())
    text = text.replace(DOC_SEP, "\n\n")        # plain text for every model alike
    print(f"benchmark {args.bench}: {len(text.encode('utf-8'))/1e6:.2f} MB, "
          f"{len(text)/1e6:.2f} M chars, {n_docs} docs, device {args.device}\n")
    print(f"{'checkpoint':26s} {'params':>7s} {'tokenizer':>16s} {'B/tok':>6s} "
          f"{'nats/tok':>8s} {'bits/byte':>9s} {'bits/char':>9s}")
    print("-" * 90)
    for ckpt in args.ckpts:
        r = score(ckpt, text, args.device, args.batch_size)
        print(f"{Path(ckpt).name:26s} {r['params']/1e6:6.0f}M {r['tokenizer']:>16s} "
              f"{r['bytes_per_token']:6.2f} {r['nats_per_token']:8.4f} "
              f"{r['bits_per_byte']:9.4f} {r['bits_per_char']:9.4f}   ({r['seconds']:.0f}s)")


if __name__ == "__main__":
    main()
