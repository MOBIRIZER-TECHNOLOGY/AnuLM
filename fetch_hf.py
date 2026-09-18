"""
Download files from a public Hugging Face dataset repo into data/raw/.

    python fetch_hf.py HuggingFaceFW/fineweb-edu --prefix sample/10BT/000 sample/10BT/001
    python fetch_hf.py openai/openai_humaneval --prefix openai_humaneval/

Lists the repo through the Hub API, downloads every file whose path starts
with one of the prefixes, and skips files already present at the right size,
so a rerun after a dropped connection only fetches what is missing. Plain
urllib, streamed to disk in 8 MB pieces; no token, so gated repos fail.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

UA = "anulm-corpus-builder/0.1 (educational; local LLM training)"


def listing(repo: str) -> list[str]:
    req = urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}", headers={"User-Agent": UA})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    if d.get("gated"):
        raise SystemExit(f"{repo} is gated; this script has no token")
    return [s["rfilename"] for s in d.get("siblings", [])]


def download(repo: str, name: str, dest: Path) -> int:
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{name}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        size = int(resp.headers.get("Content-Length", 0))
        if dest.exists() and size and dest.stat().st_size == size:
            return 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        got, t0 = 0, time.time()
        with tmp.open("wb") as f:
            while True:
                b = resp.read(8 << 20)
                if not b:
                    break
                f.write(b)
                got += len(b)
                print(f"\r  {name}: {got/1e6:.0f}/{size/1e6:.0f} MB  {got/1e6/max(time.time()-t0,1e-9):.0f} MB/s",
                      end="", flush=True)
        tmp.replace(dest)
        print()
        return got


def main():
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--prefix", nargs="+", default=[""])
    p.add_argument("--out", default="data/raw")
    args = p.parse_args()
    files = [f for f in listing(args.repo) if any(f.startswith(x) for x in args.prefix)
             and not f.endswith((".md", ".gitattributes"))]
    if not files:
        raise SystemExit(f"no files in {args.repo} match {args.prefix}")
    root = Path(args.out) / args.repo.replace("/", "__")
    total = 0
    for name in files:
        total += download(args.repo, name, root / name)
    print(f"{args.repo}: {len(files)} files, {total/1e6:.0f} MB downloaded -> {root}")


if __name__ == "__main__":
    main()
