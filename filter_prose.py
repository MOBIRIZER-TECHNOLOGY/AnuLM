"""
Prose filter for a DOC_SEP corpus.

    python filter_prose.py --data data/hindi_full.txt --out data/hindi_prose.txt

The whole Hindi dump is a quarter list-shaped: infoboxes that survived the
scrubber as bare lines, road and station lists, filmographies. Measured on
hindi_full: the median article has 64% of its lines ending in a sentence mark
(। . ! ?), the bottom quarter under 34%. A model trained on that mix learns to
emit headings and lists -- docs/RESULTS.md section 11's sample did exactly that.

Two passes, both cheap:
  * line level: drop short lines that do not end a sentence (headings, list
    items, table cells). Long sentence-less lines are kept -- they are usually
    prose with a missing danda.
  * document level: keep a document only if, after that, at least
    `--min-sentence-frac` of its lines end sentences and `--min-chars` remain.
Titles are kept as the first line regardless (they never end in a danda).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from bpe import DOC_SEP

SENTENCE_END = re.compile(r"[।.!?]['\")\]]*\s*$")


def filter_doc(doc: str, min_line_chars: int, min_sentence_frac: float, min_chars: int):
    title, _, body = doc.partition("\n\n")
    kept = []
    for line in body.split("\n"):
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if not SENTENCE_END.search(s) and len(s) < min_line_chars:
            continue                          # heading / list item / cell
        kept.append(s)
    lines = [l for l in kept if l]
    if not lines:
        return None
    frac = sum(1 for l in lines if SENTENCE_END.search(l)) / len(lines)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    if frac < min_sentence_frac or len(text) < min_chars:
        return None
    return title.strip() + "\n\n" + text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-line-chars", type=int, default=40)
    p.add_argument("--min-sentence-frac", type=float, default=0.4)
    p.add_argument("--min-chars", type=int, default=600)
    args = p.parse_args()

    docs = [d for d in Path(args.data).read_text(encoding="utf-8").split(DOC_SEP) if d.strip()]
    kept = []
    for d in docs:
        f = filter_doc(d, args.min_line_chars, args.min_sentence_frac, args.min_chars)
        if f:
            kept.append(f)
    text = DOC_SEP.join(kept) + DOC_SEP
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    in_bytes = sum(len(d.encode("utf-8")) for d in docs)
    out_bytes = len(text.encode("utf-8"))
    print(f"{len(docs)} docs, {in_bytes/1e6:.1f} MB  ->  {len(kept)} docs ({100*len(kept)/len(docs):.0f}%), "
          f"{out_bytes/1e6:.1f} MB ({100*out_bytes/in_bytes:.0f}%)  -> {args.out}")


if __name__ == "__main__":
    main()
