#!/bin/bash
# Translation demo, phase 1: pairs and the test set. CPU, minutes.
# Needs data/raw/ai4bharat__samanantar, data/raw/cfilt__iitb-english-hindi and
# data/raw/flores200 (fetch_hf.py / curl, see docs/TRANSLATE_PLAN.md).
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
PAIRS=${PAIRS:-2000000}

echo "=== PAIRS ($(date +%H:%M)) ==="
[ -s data/tr_pairs.jsonl ] || python -u convert_translate.py \
    --samanantar "data/raw/ai4bharat__samanantar/hi/*.parquet" \
    --iitb "data/raw/cfilt__iitb-english-hindi/data/train-*.parquet" \
    --max-pairs $PAIRS --out data/tr_pairs.jsonl --heldout data/tr_heldout.jsonl 2>&1 | tr '\r' '\n' | tail -8
echo "=== FLORES ($(date +%H:%M)) ==="
[ -s data/flores_devtest.jsonl ] || python -u convert_translate.py --flores data/raw/flores200 --out data/flores_devtest.jsonl
echo "=== BUILD DONE ($(date +%H:%M)) ==="
