"""
Build a Hindi training corpus from the Hindi Wikipedia dump.

    python fetch_hindi.py --mb 8 --out data/hindi.txt

Streams the bz2 article dump and stops as soon as it has enough clean prose, so
it downloads a fraction of the 240 MB file rather than all of it. Static dumps
are not rate-limited, unlike the API (which returns HTTP 429 quickly).

Stdlib only: urllib + bz2 + re.
"""

from __future__ import annotations

import argparse
import bz2
import html
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from bpe import DOC_SEP

DUMP ="https://dumps.wikimedia.org/hiwiki/latest/hiwiki-latest-pages-articles.xml.bz2"
UA = "nanosarvam-corpus-builder/0.1 (educational; local LLM training)"

# One capture per <page>: the title, then the wikitext body. The title is kept
# as the first line of each document -- a natural prompt format ("मुंबई\n\n...")
# and the model learns that an article opens with its subject.
TEXT_RE = re.compile(r"<title>(.*?)</title>.*?<text\b[^>]*>(.*?)</text>", re.DOTALL)
REDIRECT_RE = re.compile(r"^\s*#\s*(REDIRECT|पुनर्प्रेषित)", re.IGNORECASE)


def strip_wikitext(s: str) -> str:
    """Good-enough wikitext -> plain text. Not a parser; a scrubber."""
    # Entity-decode FIRST. The dump stores tags as &lt;ref&gt;; the original
    # version stripped tags before decoding entities, so escaped refs survived
    # as bare "ref name=..." text -- which the BPE model then learned as a genre
    # feature and reproduced in generation. Twice, for double-escaped content.
    s = html.unescape(html.unescape(s))
    s = re.sub(r"<!--.*?-->", "", s, flags=re.DOTALL)
    # Media/markup blocks that survived v1 and leaked into samples verbatim:
    s = re.sub(r"<gallery[^>]*>.*?</gallery>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<timeline[^>]*>.*?</timeline>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"^\s*(?:File|Image|चित्र|दस्तावेज़)\s*:.*$", "", s,
               flags=re.MULTILINE | re.IGNORECASE)
    s = re.sub(r"<ref[^>]*/>", "", s)
    s = re.sub(r"<ref.*?</ref>", "", s, flags=re.DOTALL)
    s = re.sub(r"<math.*?</math>", "", s, flags=re.DOTALL)

    # Nested {{templates}} and {|tables|}: peel innermost repeatedly.
    for _ in range(12):
        new = re.sub(r"\{\{[^{}]*\}\}", "", s)
        new = re.sub(r"\{\|[^{}]*?\|\}", "", new, flags=re.DOTALL)
        if new == s:
            break
        s = new

    # Media and category links go entirely; other links keep their display text.
    s = re.sub(r"\[\[(?:File|Image|चित्र|श्रेणी|Category)\s*:[^\]]*\]\]", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\s([^\]]*)\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\]", "", s)

    s = re.sub(r"</?[a-zA-Z][^>]*>", "", s)          # leftover HTML
    s = re.sub(r"'{2,}", "", s)                       # bold / italic marks
    # v3: leaks the v2 model amplified in generation --
    s = re.sub(r"\[\d+\]", "", s)                    # [3] numeric citations
    s = re.sub(r"^-{4,}\.?\s*$", "", s, flags=re.MULTILINE)   # ---- rules
    # headings: v2 required ==...== and missed single-= (`= सन्दर्भ =`)
    s = re.sub(r"^\s*=+[^=\n]*=+\s*$", "", s, flags=re.MULTILINE)
    # unbalanced template/table remnants the 12-pass peeler left behind
    s = re.sub(r"\{\{|\}\}|\{\||\|\}|\[\[|\]\]", "", s)
    s = re.sub(r"^[*#:;]+\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"&[a-z]+;|&#\d+;", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def devanagari_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(1 for ch in s if "ऀ" <= ch <= "ॿ") / len(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mb", type=float, default=8.0)
    p.add_argument("--out", type=str, default="data/hindi.txt")
    p.add_argument("--min-chars", type=int, default=600)
    p.add_argument("--min-devanagari", type=float, default=0.55)
    p.add_argument("--keep-prob", type=float, default=1.0,
                   help="keep each qualifying article with this probability. "
                        "<1 spreads the sample across the whole dump instead of "
                        "taking the front -- the front-loaded v1 corpus had "
                        "monuments but no cricket")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dump", type=str, default=DUMP,
                   help="any Wikimedia pages-articles bz2 dump; e.g. the Hindi "
                        "Wikisource one for an out-of-register benchmark")
    args = p.parse_args()
    rng = random.Random(args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    target = int(args.mb * 1024 * 1024)

    req = urllib.request.Request(args.dump, headers={"User-Agent": UA})
    dec = bz2.BZ2Decompressor()
    buf, chunks = "", []
    total = kept = seen = downloaded = 0
    t0 = time.time()

    print(f"streaming {args.dump.rsplit('/', 1)[1]}, stopping at {args.mb} MB of prose")
    with urllib.request.urlopen(req, timeout=60) as resp:
        while total < target:
            raw = resp.read(1 << 20)
            if not raw:
                break
            downloaded += len(raw)
            try:
                buf += dec.decompress(raw).decode("utf-8", errors="ignore")
            except EOFError:
                break

            last = 0
            for m in TEXT_RE.finditer(buf):
                seen += 1
                last = m.end()
                title, body = html.unescape(m.group(1)).strip(), m.group(2)
                if REDIRECT_RE.match(body):
                    continue
                if args.keep_prob < 1.0 and rng.random() >= args.keep_prob:
                    continue          # decide BEFORE cleaning; cleaning is the cost
                text = strip_wikitext(body)
                if len(text) < args.min_chars or devanagari_ratio(text) < args.min_devanagari:
                    continue
                chunks.append(title + "\n\n" + text + DOC_SEP)
                total += len(chunks[-1].encode("utf-8"))
                kept += 1
            buf = buf[last:] if last else buf[-1_000_000:]   # keep a partial tail

            print(f"\r  downloaded {downloaded/1e6:5.1f} MB -> prose {total/1e6:5.2f}/{args.mb:.1f} MB"
                  f"   kept {kept}/{seen}   {time.time()-t0:4.0f}s", end="", flush=True)

    # newline="\n": no CRLF translation on Windows, so DOC_SEP is the same
    # bytes on every platform and a byte model pays one byte per newline.
    out.write_text("".join(chunks), encoding="utf-8", newline="\n")
    raw_b = out.read_bytes()
    txt = raw_b.decode("utf-8")
    print(f"\n\nwrote {out}  ({len(raw_b)/1e6:.2f} MB, {len(txt)/1e6:.2f} M chars, {kept} articles; "
          f"title + body per document, DOC_SEP between)")
    print(f"downloaded {downloaded/1e6:.1f} MB of the 240 MB dump "
          f"({100*downloaded/240e6:.0f}%)")
    print(f"bytes per character: {len(raw_b)/len(txt):.2f}   (ASCII 1.00, Devanagari 3.00)")
    print(f"Devanagari fraction: {devanagari_ratio(txt):.1%}")
    print(f"\nA byte-level model sees {len(raw_b)/len(txt):.2f}x more tokens per character "
          f"here than on English.\nThat is the fertility problem Sarvam's 262k-token "
          f"tokenizer exists to solve.")


if __name__ == "__main__":
    main()
