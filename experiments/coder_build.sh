#!/bin/bash
# Phase 2 of docs/CODER_PLAN.md: convert, mix, tokenizer, encode.
# CPU only, ~2 hours. Safe to rerun; each step skips work already done.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
set -o pipefail

echo "=== 1. CONVERT ($(date +%H:%M)) ==="
[ -s data/english_edu.txt ] || python -u convert_parquet.py fineweb \
    --in "data/raw/HuggingFaceFW__fineweb-edu/sample/10BT/*.parquet" --out data/english_edu.txt --mb 4000
[ -s data/exercises.txt ] || python -u convert_parquet.py exercises \
    --in "data/raw/jinaai__code_exercises/data/*.parquet" --out data/exercises.txt --jsonl data/sft_exercises.jsonl
[ -s data/sft_opencode.jsonl ] || python -u convert_parquet.py pairs \
    --in "data/raw/nvidia__OpenCodeInstruct/data/*.parquet" --q-col input --a-col output --python-only \
    --score-col average_test_score --min-pair-score 0.9 --jsonl data/sft_opencode.jsonl
[ -s data/sft_glaive.jsonl ] || python -u convert_parquet.py pairs \
    --in "data/raw/glaiveai__glaive-code-assistant/*.json" --q-col question --a-col answer --python-only \
    --jsonl data/sft_glaive.jsonl

echo "=== 2. MIX ($(date +%H:%M)) ==="
# ~65% Python / 30% English / 5% Hindi by tokens. Python ~4.4 bytes/token,
# English ~4.3, Hindi ~9.4, so by bytes: 8800 : 3900 : 1300 MB.
[ -s data/coder.txt ] || python -u mix_corpus.py \
    --source python data/python_big_raw.txt 7200 \
    --source exercises data/exercises.txt 1000 \
    --source english data/english_edu.txt 3900 \
    --source hindi data/mixed.txt 1300 \
    --out data/coder.txt --sample-out data/coder_toksample.txt --sample-mb 250 50 100 100

echo "=== 3. TOKENIZER 32k ($(date +%H:%M)) ==="
[ -s data/code32k.json ] || python -u bpe.py train --data data/coder_toksample.txt --vocab 32768 --out data/code32k.json 2>&1 | grep -E "trained|bytes/token|tokens/word|roundtrip"
for f in data/bench_python_clean.txt data/bench_english.txt data/bench_hindi.txt; do
  echo "$f:"; python -u bpe.py stats --tokenizer data/code32k.json --data $f 2>&1 | grep -E "bytes/token|tokens/word"
done

echo "=== 4. ENCODE ($(date +%H:%M)) ==="
[ -s data/coder.code32k.bin ] || python -u bpe.py encode --tokenizer data/code32k.json --data data/coder.txt --out data/coder.code32k.bin --workers 20 2>&1 | tr '\r' '\n' | tail -1
echo "=== BUILD DONE ($(date +%H:%M)) ==="
