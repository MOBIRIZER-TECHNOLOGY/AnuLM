"""
Parallel corpora -> translation pairs in finetune.py's jsonl format.

    python convert_translate.py --samanantar "data/raw/ai4bharat__samanantar/hi/*.parquet" \
        --iitb "data/raw/cfilt__iitb-english-hindi/data/train-*.parquet" \
        --max-pairs 2000000 --out data/tr_pairs.jsonl --heldout data/tr_heldout.jsonl
    python convert_translate.py --flores data/raw/flores200 --out data/flores_devtest.jsonl

Every sentence pair yields two items, one per direction, with `lang`
"en-hi" or "hi-en" so the fine-tune wraps each in its own template. Pairs
are filtered for length (3-200 words, ratio under 3:1) and for script (the
Hindi side must be mostly Devanagari), and de-duplicated on the English
side. FLORES devtest becomes the scoring file, one item per direction per
sentence, never used for training.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from pathlib import Path

_DEVA = re.compile(r"[ऀ-ॿ]")


def devanagari_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum(1 for c in letters if _DEVA.match(c)) / max(len(letters), 1)


def good_pair(en: str, hi: str) -> bool:
    ne, nh = len(en.split()), len(hi.split())
    if not (3 <= ne <= 200 and 3 <= nh <= 200) or max(ne, nh) > 3 * min(ne, nh):
        return False
    if devanagari_ratio(hi) < 0.8 or _DEVA.search(en):
        return False
    return "http" not in en and "\n" not in en and "\n" not in hi


def parquet_pairs(patterns):
    import pyarrow.parquet as pq
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=4096):
                for r in batch.to_pylist():
                    if "translation" in r:                     # IITB layout
                        yield r["translation"]["en"], r["translation"]["hi"]
                    else:                                      # Samanantar: src (en), tgt (hi)
                        yield r.get("src") or r.get("en"), r.get("tgt") or r.get("hi")


def write(path, items):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def both_directions(en: str, hi: str):
    return [{"title": en[:40], "question": en, "answer": hi, "lang": "en-hi"},
            {"title": hi[:40], "question": hi, "answer": en, "lang": "hi-en"}]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samanantar", nargs="*", default=[])
    p.add_argument("--iitb", nargs="*", default=[])
    p.add_argument("--flores", default=None, help="extracted flores200_dataset directory")
    p.add_argument("--max-pairs", type=int, default=2_000_000, help="sentence pairs (x2 items)")
    p.add_argument("--heldout-pairs", type=int, default=1000)
    p.add_argument("--out", required=True)
    p.add_argument("--heldout", default=None)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    if args.flores:
        root = next(Path(args.flores).glob("**/devtest"))
        en = (root / "eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
        hi = (root / "hin_Deva.devtest").read_text(encoding="utf-8").splitlines()
        assert len(en) == len(hi)
        items = [it for e, h in zip(en, hi) for it in both_directions(e.strip(), h.strip())]
        write(args.out, items)
        print(f"flores devtest: {len(en)} sentences -> {len(items)} items -> {args.out}")
        return

    rng = random.Random(args.seed)
    seen: set[bytes] = set()
    pairs = []
    scanned = 0
    for en, hi in parquet_pairs(args.samanantar + args.iitb):
        scanned += 1
        if not en or not hi:
            continue
        en, hi = " ".join(en.split()), " ".join(hi.split())
        if not good_pair(en, hi):
            continue
        h = hashlib.sha1(en.lower().encode("utf-8")).digest()
        if h in seen:
            continue
        seen.add(h)
        pairs.append((en, hi))
        if len(pairs) % 200000 == 0:
            print(f"\r  {scanned} scanned, {len(pairs)} kept", end="", flush=True)
        if len(pairs) >= args.max_pairs + args.heldout_pairs:
            break
    rng.shuffle(pairs)
    held, train = pairs[:args.heldout_pairs], pairs[args.heldout_pairs:]
    train_items = [it for e, h in train for it in both_directions(e, h)]
    rng.shuffle(train_items)
    write(args.out, train_items)
    if args.heldout:
        write(args.heldout, [it for e, h in held for it in both_directions(e, h)])
    mean_en = sum(len(e.split()) for e, _ in train) / max(len(train), 1)
    print(f"\n{scanned} scanned -> {len(pairs)} pairs kept; train {len(train_items)} items "
          f"(both directions), held-out {2 * len(held)}; mean {mean_en:.1f} English words")
    for e, h in train[:3]:
        print(f"  {e}\n  {h}\n")


if __name__ == "__main__":
    main()
