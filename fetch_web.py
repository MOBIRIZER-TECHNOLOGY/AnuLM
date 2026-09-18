"""
Build an English web-text corpus from C4 (allenai/c4, public).

    python fetch_web.py --mb 350 --out data/english_web.txt

Streams one shard of the English split (~320 MB .json.gz, ~800 MB of text),
keeps documents that read as prose, de-duplicates by content, and stops at
--mb. Same document convention as fetch_hindi.py: a title line (here the
page URL), a blank line, the body, DOC_SEP after. C4 is already
English-only, deduplicated and boilerplate-stripped; the filters here only
drop the very short, the very long and anything that is mostly non-ASCII.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path

from bpe import DOC_SEP

BASE = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/"
UA = "nanosarvam-corpus-builder/0.1 (educational; local LLM training)"
_BLANKS = re.compile(r"\n{3,}")


def looks_like_prose(text: str, min_bytes: int, max_bytes: int) -> bool:
    n = len(text.encode("utf-8"))
    if n < min_bytes or n > max_bytes or "\x00" in text:
        return False
    if sum(1 for ch in text if ord(ch) < 128) / len(text) < 0.95:
        return False
    lines = [l for l in text.split("\n") if l.strip()]
    # Prose has sentences: most lines should end in a sentence mark.
    ended = sum(1 for l in lines if l.rstrip()[-1:] in ".!?\"'")
    return ended / max(len(lines), 1) >= 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mb", type=float, default=350.0)
    p.add_argument("--out", default="data/english_web.txt")
    p.add_argument("--shards", type=int, nargs="+", default=[0])
    p.add_argument("--min-bytes", type=int, default=1000)
    p.add_argument("--max-bytes", type=int, default=30_000)
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    target = int(args.mb * 1024 * 1024)

    seen: set[bytes] = set()
    chunks, total, kept, scanned, downloaded = [], 0, 0, 0, 0
    t0 = time.time()
    for shard in args.shards:
        if total >= target:
            break
        name = f"c4-train.{shard:05d}-of-01024.json.gz"
        print(f"streaming {name}, stopping at {args.mb} MB of prose")
        req = urllib.request.Request(BASE + name, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp, gzip.GzipFile(fileobj=resp) as gz:
            for raw in gz:
                scanned += 1
                downloaded += len(raw)
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                text = rec.get("text", "")
                if not looks_like_prose(text, args.min_bytes, args.max_bytes):
                    continue
                h = hashlib.sha1(text.encode("utf-8")).digest()
                if h in seen:
                    continue
                seen.add(h)
                body = _BLANKS.sub("\n\n", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
                doc = f"{rec.get('url', '?')}\n\n{body}" + DOC_SEP
                chunks.append(doc)
                total += len(doc.encode("utf-8"))
                kept += 1
                if kept % 5000 == 0:
                    print(f"\r  scanned {scanned} records ({downloaded/1e6:.0f} MB) -> "
                          f"{total/1e6:.1f}/{args.mb:.0f} MB, kept {kept}   {time.time()-t0:.0f}s",
                          end="", flush=True)
                if total >= target:
                    break

    out.write_text("".join(chunks), encoding="utf-8", newline="\n")
    raw_b = out.read_bytes()
    print(f"\n\nwrote {out}  ({len(raw_b)/1e6:.2f} MB, {kept} pages of {scanned} scanned; "
          f"title + body per document, DOC_SEP between)")


if __name__ == "__main__":
    main()
