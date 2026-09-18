"""
Build a Python code corpus from codeparrot-clean (public, ungated).

    python fetch_code.py --mb 250 --out data/python.txt

Streams one shard of the dataset (~1 GB of Python per 246 MB .json.gz), keeps
files that look like hand-written source rather than data or generated code,
de-duplicates by content, and stops at --mb of text. Same document convention
as fetch_hindi.py: a title line (here `# repo/path.py`), a blank line, the
body, DOC_SEP after. Runs of three or more newlines are collapsed to two,
because DOC_SEP is three newlines and PEP 8 puts exactly that between
top-level definitions -- the tokenizer's document splitter would otherwise
cut every module into pieces.
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

BASE = "https://huggingface.co/datasets/codeparrot/codeparrot-clean/resolve/main/"
UA = "nanosarvam-corpus-builder/0.1 (educational; local LLM training)"
_BLANKS = re.compile(r"\n{3,}")


def looks_like_source(text: str, min_bytes: int, max_bytes: int) -> bool:
    n = len(text.encode("utf-8"))
    if n < min_bytes or n > max_bytes or "\x00" in text:
        return False
    lines = text.split("\n")
    longest = max(len(l) for l in lines)
    mean = n / max(len(lines), 1)
    if longest > 400 or mean > 120:              # minified, data tables, base64
        return False
    if sum(1 for ch in text if ord(ch) < 128) / len(text) < 0.9:
        return False
    if text.count("def ") + text.count("class ") + text.count("import ") == 0:
        return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mb", type=float, default=250.0)
    p.add_argument("--out", default="data/python.txt")
    p.add_argument("--shards", type=int, nargs="+", default=[1])
    p.add_argument("--min-bytes", type=int, default=512)
    p.add_argument("--max-bytes", type=int, default=50_000)
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
        url = f"{BASE}file-{shard:012d}.json.gz"
        print(f"streaming {url.rsplit('/', 1)[1]}, stopping at {args.mb} MB of code")
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as resp, gzip.GzipFile(fileobj=resp) as gz:
            for raw in gz:
                scanned += 1
                downloaded += len(raw)
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                path = rec.get("path", "")
                text = rec.get("content", "")
                if not path.endswith(".py") or not looks_like_source(text, args.min_bytes, args.max_bytes):
                    continue
                h = hashlib.sha1(text.encode("utf-8")).digest()
                if h in seen:
                    continue
                seen.add(h)
                body = _BLANKS.sub("\n\n", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
                doc = f"# {rec.get('repo_name', '?')}/{path}\n\n{body}" + DOC_SEP
                chunks.append(doc)
                total += len(doc.encode("utf-8"))
                kept += 1
                if kept % 2000 == 0:
                    print(f"\r  scanned {scanned} records ({downloaded/1e6:.0f} MB) -> "
                          f"{total/1e6:.1f}/{args.mb:.0f} MB, kept {kept}   {time.time()-t0:.0f}s",
                          end="", flush=True)
                if total >= target:
                    break

    out.write_text("".join(chunks), encoding="utf-8", newline="\n")
    raw_b = out.read_bytes()
    print(f"\n\nwrote {out}  ({len(raw_b)/1e6:.2f} MB, {kept} files of {scanned} scanned; "
          f"title + body per document, DOC_SEP between)")


if __name__ == "__main__":
    main()
