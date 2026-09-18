"""
Score checkpoints on the golden set (make_golden.py).

    python eval_golden.py ckpt_hindi_mixed18k.pt ckpt_hindi_full.pt --device cuda

Two numbers per checkpoint, each broken down by source and item type:

  EM   generative exact match. Greedy-decode from the prefix, take the first
       whitespace-delimited word, strip punctuation, compare to the gold word.
       Strict: a synonym or an inflection scores zero.
  MC   4-way multiple choice. Score each choice by the total log-probability
       of its tokens given the prefix (the LAMBADA / HellaSwag protocol);
       correct if the gold word ranks first. Chance is 25%.

Works for BPE and byte checkpoints alike: the prefix ends on a word and every
choice is encoded as " " + word, which the chunk-based BPE tokenises
identically whether or not the prefix is present.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from model import AnuLM

STRIP_PUNCT = re.compile(r"^[\"'(\[“‘]+|[\"')\]”’,;:।.!?]+$")


class Codec:
    """One encode/decode pair per checkpoint: BPE or raw bytes."""

    def __init__(self, cfg):
        self.tok = None
        if getattr(cfg, "tokenizer_path", None):
            from bpe import BPE
            self.tok = BPE.load(cfg.tokenizer_path)

    def encode(self, s: str) -> list[int]:
        return self.tok.encode(s) if self.tok else list(s.encode("utf-8"))

    def decode(self, ids) -> str:
        if self.tok:
            return self.tok.decode(ids)
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


@torch.no_grad()
def score(ckpt: str, items: list[dict], device: str, max_gen: int = 12) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if device.startswith("cuda"):
        cfg = replace(cfg, moe_impl="grouped")
    m = AnuLM(cfg).to(device).eval()
    m.load_state_dict(ck["model"])
    codec = Codec(cfg)
    bs = cfg.block_size
    if codec.tok is None:
        # A byte model needs ~3 bytes per Devanagari character, so the same
        # word budget is 4x the tokens; otherwise EM is capped by truncation.
        max_gen *= 4
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))

    hits = defaultdict(lambda: [0, 0, 0])          # key -> [em, mc, n]
    t0 = time.time()
    for it in items:
        prefix = codec.encode(it["prefix"])
        # --- multiple choice: one batch, the 4 choices right-padded ------
        cands = [codec.encode(" " + c) for c in it["choices"]]
        longest = max(len(c) for c in cands)
        keep = bs - longest
        pre = prefix[-keep:]
        rows = [pre + c + [0] * (longest - len(c)) for c in cands]
        x = torch.tensor(rows, dtype=torch.long, device=device)
        with ac:
            logits, _ = m(x[:, :-1], x[:, 1:])
        logp = F.log_softmax(logits.float(), dim=-1)
        totals = []
        for r, c in enumerate(cands):
            # tokens of the candidate sit at positions len(pre) .. len(pre)+len(c)-1
            # and are predicted from the position before each.
            pos = torch.arange(len(pre) - 1, len(pre) - 1 + len(c), device=device)
            tgt = x[r, len(pre): len(pre) + len(c)]
            totals.append(logp[r, pos, tgt].sum().item())
        mc = int(max(range(4), key=totals.__getitem__) == it["answer_idx"])

        # --- generative exact match ---------------------------------------
        pre = prefix[-(bs - max_gen):]
        ids = torch.tensor([pre], dtype=torch.long, device=device)
        with ac:
            out = m.generate(ids, max_gen, temperature=1.0, top_k=1)
        text = codec.decode(out[0, len(pre):].tolist())
        word = text.strip().split()[0] if text.strip() else ""
        em = int(STRIP_PUNCT.sub("", word) == it["answer"])

        for key in ("all", it["source"], it["type"], f"{it['source']}/{it['type']}"):
            hits[key][0] += em
            hits[key][1] += mc
            hits[key][2] += 1
    return {"ckpt": ckpt, "seconds": time.time() - t0,
            "cells": {k: {"em": v[0] / v[2], "mc": v[1] / v[2], "n": v[2]} for k, v in hits.items()}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--golden", default="golden/golden_hindi.jsonl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="score only the first N items")
    args = p.parse_args()

    # split("\n"): splitlines() also breaks on U+2028 and friends, which
    # json.dumps(ensure_ascii=False) leaves raw inside strings (see finetune.py).
    items = [json.loads(l) for l in Path(args.golden).read_text(encoding="utf-8").split("\n") if l.strip()]
    if args.limit:
        items = items[: args.limit]
    print(f"golden set {args.golden}: {len(items)} items, device {args.device}\n")
    keys = ["all", "wikipedia", "wikisource", "in_context", "novel",
            "wikipedia/in_context", "wikipedia/novel", "wikisource/in_context", "wikisource/novel"]
    header = f"{'checkpoint':24s} " + " ".join(f"{k:>22s}" for k in keys)
    print(header)
    print(" " * 25 + " ".join(f"{'EM%   MC%':>22s}" for _ in keys))
    print("-" * len(header))
    for ckpt in args.ckpts:
        r = score(ckpt, items, args.device)
        cells = r["cells"]
        row = f"{Path(ckpt).name:24s} " + " ".join(
            f"{100 * cells[k]['em']:9.1f} {100 * cells[k]['mc']:6.1f}      " if k in cells else f"{'-':>22s}"
            for k in keys)
        print(row + f"  ({r['seconds']:.0f}s)")
    print("\nchance: EM ~0%, MC 25%")


if __name__ == "__main__":
    main()
