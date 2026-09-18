#!/bin/bash
# Build the Hindi + English + Python corpus, a 32k tokenizer, and the encoding.
#
#   1. English from two sources: Wikipedia through fetch_hindi.py with the
#      Devanagari filter off (the scrubber is language-neutral; the dump
#      server streams ~7 MB of prose a minute and dropped the connection at
#      167 MB the first time), and C4 web text via fetch_web.py, which is
#      fast. Python from codeparrot-clean via fetch_code.py. All fetches run
#      concurrently and are skipped when their output already exists, so a
#      rerun only fetches what is missing. The last 2% of each language
#      becomes its benchmark file, never trained on.
#   2. Interleave with data/mixed.txt (Hindi Wikipedia + Wikisource). Sized
#      for roughly 45% Hindi / 35% English / 20% code by tokens: Hindi runs
#      ~9.4 bytes/token on a BPE, English ~4.5, Python ~3.5, so 1.32 GB :
#      500 MB : 250 MB lands near 140M : 110M : 70M tokens.
#   3. Tokenizer: 32k, trained on a 300 MB Hindi + 150 MB English + 100 MB
#      Python sample. Three scripts need the room; a 32k vocab does not need
#      2 GB to converge, and the pure-Python trainer is O(corpus).
#   4. Encode the whole corpus with it.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
set -o pipefail

echo "=== 1. FETCH ENGLISH + PYTHON ($(date +%H:%M)) ==="
[ -s data/english_raw.txt ] || python -u fetch_hindi.py --mb 500 --keep-prob 0.25 --seed 2026 --min-devanagari 0 \
    --dump https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 \
    --out data/english_raw.txt 2>&1 | tr '\r' '\n' | grep -E "wrote|downloaded [0-9.]+ MB of|streaming|Error|Traceback" &
[ -s data/english_web_raw.txt ] || python -u fetch_web.py --mb 350 --out data/english_web_raw.txt 2>&1 | tr '\r' '\n' | grep -E "wrote|streaming|Error|Traceback" &
[ -s data/python_raw.txt ] || python -u fetch_code.py --mb 250 --out data/python_raw.txt 2>&1 | tr '\r' '\n' | grep -E "wrote|streaming|Error|Traceback" &
wait
for f in english_raw english_web_raw python_raw; do
  [ -s data/$f.txt ] || { echo "$f fetch produced nothing"; exit 1; }
done

echo "=== 2. SPLIT + INTERLEAVE ($(date +%H:%M)) ==="
python - <<'PY'
import random, time
from pathlib import Path
from bpe import DOC_SEP
t0 = time.time()
def docs(p): return [d for d in Path(p).read_text(encoding="utf-8").split(DOC_SEP) if d.strip()]
def write(p, ds): Path(p).write_text(DOC_SEP.join(ds) + DOC_SEP, encoding="utf-8", newline="\n")
train = {}
for name, raws in (("english", ("data/english_raw.txt", "data/english_web_raw.txt")),
                   ("python", ("data/python_raw.txt",))):
    ds = [d for r in raws for d in docs(r)]
    random.Random(len(ds)).shuffle(ds)                         # bench draws from every source
    n_bench = max(1, len(ds) // 50)                            # last 2% held out
    write(f"data/bench_{name}.txt", ds[-n_bench:]); write(f"data/{name}.txt", ds[:-n_bench])
    train[name] = ds[:-n_bench]
    print(f"{name}: {len(ds)} docs -> {len(train[name])} train + {n_bench} bench")
hi = docs("data/mixed.txt")
all_docs = hi + train["english"] + train["python"]
random.Random(1337).shuffle(all_docs)
write("data/multi.txt", all_docs)
mb = lambda ds: sum(len(d.encode("utf-8")) for d in ds) / 1e6
print(f"multi: {len(all_docs)} docs, {mb(all_docs):.0f} MB = hindi {mb(hi):.0f} + english {mb(train['english']):.0f} + python {mb(train['python']):.0f}; {time.time()-t0:.0f}s")
def take(ds, budget):
    out, n = [], 0
    for d in ds:
        if n >= budget: break
        out.append(d); n += len(d.encode("utf-8"))
    return out
rng = random.Random(7)
parts = []
for ds, budget in ((hi, 300e6), (train["english"], 150e6), (train["python"], 100e6)):
    ds = ds[:]; rng.shuffle(ds); parts += take(ds, budget)
rng.shuffle(parts)
write("data/multi_toksample.txt", parts)
print(f"tokenizer sample: {len(parts)} docs, {mb(parts):.0f} MB")
PY

echo "=== 3. TOKENIZER 32k ($(date +%H:%M)) ==="
python -u bpe.py train --data data/multi_toksample.txt --vocab 32768 --out data/multi32k.json 2>&1 | grep -E "trained|bytes/token|tokens/word|roundtrip|unique chunks"
echo "--- fertility per language, multi32k"
for f in data/bench_hindi.txt data/bench_wikisource.txt data/bench_english.txt data/bench_python.txt; do
  echo "$f:"; python -u bpe.py stats --tokenizer data/multi32k.json --data $f 2>&1 | grep -E "bytes/token|tokens/word"
done
echo "--- the old 16k Hindi tokenizer on English and Python, for contrast"
for f in data/bench_english.txt data/bench_python.txt; do
  echo "$f:"; python -u bpe.py stats --tokenizer data/hi16k_mixed.json --data $f 2>&1 | grep -E "bytes/token|tokens/word"
done

echo "=== 4. ENCODE ($(date +%H:%M)) ==="
python -u bpe.py encode --tokenizer data/multi32k.json --data data/multi.txt --out data/multi.multi32k.bin 2>&1 | tail -1
echo "=== BUILD DONE ($(date +%H:%M)) ==="
