#!/bin/bash
# Build the Wikipedia + Wikisource training corpus, tokenizer and encoding.
# Detached: the tokenizer pass over 1.3 GB outlives the tool's 10-minute cap.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8

python - <<'PY'
# Wikisource benchmark pages are named exactly (पृष्ठ:book.pdf/N), so a title
# match is a complete de-duplication here -- no need for the signature probe,
# which is O(pages x signatures) and took the first attempt past an hour.
import random, time
from pathlib import Path
from bpe import DOC_SEP
t0 = time.time()
bench_titles = {d.split("\n", 1)[0].strip()
                for d in Path("data/bench_wikisource.txt").read_text(encoding="utf-8").split(DOC_SEP) if d.strip()}
ws = [d for d in Path("data/wikisource_full.txt").read_text(encoding="utf-8").split(DOC_SEP) if d.strip()]
kept = [d for d in ws if d.split("\n", 1)[0].strip().startswith("पृष्ठ:")
        and d.split("\n", 1)[0].strip() not in bench_titles]
print(f"wikisource: {len(ws)} -> {len(kept)} book pages after dropping non-page namespaces and "
      f"{sum(1 for d in ws if d.split(chr(10),1)[0].strip() in bench_titles)} benchmark pages", flush=True)
wiki = [d for d in Path("data/hindi_full.txt").read_text(encoding="utf-8").split(DOC_SEP) if d.strip()]
docs = wiki + kept
random.Random(1337).shuffle(docs)
text = DOC_SEP.join(docs) + DOC_SEP
Path("data/mixed.txt").write_text(text, encoding="utf-8", newline="\n")
wb = sum(len(d.encode("utf-8")) for d in wiki); sb = sum(len(d.encode("utf-8")) for d in kept)
print(f"mixed: {len(docs)} docs, {len(text.encode('utf-8'))/1e6:.1f} MB = wikipedia {wb/1e6:.0f} MB + "
      f"wikisource {sb/1e6:.0f} MB, interleaved; {time.time()-t0:.0f}s", flush=True)
PY

python -u bpe.py train --data data/mixed.txt --vocab 16384 --out data/hi16k_mixed.json 2>&1 | grep -E "trained|bytes/token|tokens/word|roundtrip"
python -u bpe.py encode --tokenizer data/hi16k_mixed.json --data data/mixed.txt --out data/mixed.hi16k.bin 2>&1 | tail -1
echo "=== BUILD DONE ==="
