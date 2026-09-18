#!/bin/bash
# Translation demo, phase 3: chrF on FLORES-200 devtest, base vs tuned.
#     bash experiments/translate_eval.sh          # 300 sentences per direction, ~40 min
#     LIMIT= bash experiments/translate_eval.sh   # all 1012, ~2.5 h -- the number to quote
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
OUT=${OUT:-ckpt_translate.pt}
LIMIT=${LIMIT-300}
LIMITARG=""; [ -n "$LIMIT" ] && LIMITARG="--limit $LIMIT"
python -u eval_translate.py ckpt_multi36k.pt $OUT --device cuda $LIMITARG 2>&1 | grep -v -i "warning\|triton" | tr '\r' '\n' | grep -v "^  ckpt.*/.*(.*s)$"
