"""
Build the fixed held-out benchmark every checkpoint is scored on.

    python make_bench.py --raw data/bench_raw.txt --out data/bench_hindi.txt

`fetch_hindi.py --seed 2024 --keep-prob 0.05` draws articles independently of
the training corpora (seed 1337), so ~30% of them were also drawn for v2/v3/v4.
Drop any document whose body appears in ANY corpus a checkpoint here has
trained on, then the benchmark is clean for old and new models alike.
Own-val numbers cannot be compared across corpora (docs/RESULTS.md section 8);
this file is the shared axis.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from bpe import DOC_SEP

CORPORA = ["data/hindi.txt", "data/hindi_big.txt", "data/hindi_v2.txt",
           "data/hindi_v3.txt", "data/hindi_v4.txt"]

_NOT_DEVANAGARI = re.compile(r"[^ऀ-ॿ]+")


def normalise(s: str) -> str:
    """Devanagari code points only -- the signature that survives scrubbing."""
    return _NOT_DEVANAGARI.sub("", s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/bench_raw.txt")
    p.add_argument("--out", default="data/bench_hindi.txt")
    p.add_argument("--probe-chars", type=int, default=80,
                   help="a doc is 'seen' if a window of this many Devanagari "
                        "letters from its body occurs in any training corpus")
    args = p.parse_args()

    # Compare on Devanagari letters only. The v1 corpora were cleaned by an
    # older scrubber that left markup remnants in the text, so an exact
    # substring probe misses articles those models DID train on -- which is
    # exactly the leak that makes a benchmark lie. Markup is ASCII; the prose
    # underneath is the same, so a script-only signature matches across
    # scrubber versions. Three windows spread through the body, any hit drops.
    seen_in = [normalise(Path(c).read_text(encoding="utf-8")) for c in CORPORA if Path(c).exists()]
    docs = [d for d in Path(args.raw).read_text(encoding="utf-8").split(DOC_SEP) if d.strip()]
    kept, dropped = [], 0
    for d in docs:
        body = normalise(d.split("\n\n", 1)[-1])
        n = args.probe_chars
        probes = [body[i: i + n] for i in (len(body) // 10, len(body) // 2, max(len(body) - n, 0))]
        if any(len(p) >= n // 2 and p in corpus for p in probes for corpus in seen_in):
            dropped += 1
            continue
        kept.append(d)
    text = DOC_SEP.join(kept) + DOC_SEP
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    n_bytes = len(text.encode("utf-8"))
    print(f"{len(docs)} candidate docs -> kept {len(kept)}, dropped {dropped} seen in training corpora")
    print(f"wrote {args.out}: {n_bytes/1e6:.2f} MB, {len(text)/1e6:.2f} M chars, {len(kept)} docs")


if __name__ == "__main__":
    main()
