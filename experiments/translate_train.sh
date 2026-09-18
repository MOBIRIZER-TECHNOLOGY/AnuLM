#!/bin/bash
# Translation demo, phase 2: fine-tune the three-language base on the pairs.
#
#     bash experiments/translate_train.sh 20000    # to step 20k, then exit
#     bash experiments/translate_train.sh          # resume, run to the end
#
# One pass over 2M sentence pairs in both directions is ~70k steps at batch
# 8 (~8 hours). Each call resumes from ckpt_translate.pt.last if it exists
# and stops at the step given after an eval and a save, so the machine can
# sleep between calls; a killed call loses at most --eval-every steps.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
BASE=${BASE:-ckpt_multi36k.pt}
OUT=${OUT:-ckpt_translate.pt}
STOP=${1:-}
RESUME=""; [ -s $OUT.last ] && RESUME="--resume"
STOPARG=""; [ -n "$STOP" ] && STOPARG="--stop-at $STOP"

echo "=== TRANSLATE PHASE: ${RESUME:-fresh start} ${STOPARG:-to the end} ($(date +%H:%M)) ==="
python -u finetune.py --ckpt $BASE --qa data/tr_pairs.jsonl --heldout data/tr_heldout.jsonl --out $OUT \
    --epochs 1 --lr 1e-4 --min-lr 1e-5 --warmup 500 --eval-every 2000 --log-every 100 \
    --grad-ckpt --workers 16 $RESUME $STOPARG 2>&1 | grep -v -i "warning\|triton"
echo "=== PHASE DONE ($(date +%H:%M)) ==="
