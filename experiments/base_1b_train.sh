#!/bin/bash
# AnuLM 1B base, from scratch, on the clean corpus (docs/PLAN_1B.md). One RTX 5070 Ti
# class GPU (16 GB). Resumable: rerun the same command after any stop.
#
#   bash experiments/base_1b_train.sh            # fresh, or resume from ckpt_base_1b.pt.last
#   bash experiments/base_1b_train.sh 2000       # a first test: stop at step 2,000, eval and save
#
# Before it: the corpus, data/v3 (docs/TRAIN_FROM_SCRATCH.md, route D):
#   python experiments/build_v2.py                   # English + Hindi (no Python sources needed)
#   python -I experiments/build_code_clean.py --out data/code_clean --target-gb 3.4
#   python -I experiments/build_v3.py --v2 data/v2 --code data/code_clean --out data/v3
#
# Model: --preset 1b = 1,008M parameters, 284M active per token (anulm/model.py,
# AnuLMConfig.base_1b): 24 layers, 24 experts 512 wide with 4 per token, sliding-window
# attention on the first 10 layers, 1,024-token context. To fit 16 GB it needs both
#   --grad-ckpt                  recompute activations (about 30% slower, far less memory)
#   --optimizer adamw-lowmem     AdamW with bf16 moments: 12 bytes/parameter, not 16
# Measured on the training machine (2026-11-06): 13.8 GB of 16 used, 5,360 tokens/s,
# 760 ms per step; loss 10.63 at step 0 and 6.32 at step 1,000.
# Plan: 2,770,000 steps x 4 x 1,024 = 11.3B tokens, one pass over data/v3, ~24 days.
# The schedule is warmup-stable-decay, so the run can be shortened by restarting with a
# smaller --steps any time before the decay (the last 16%) begins.
export PATH="/usr/bin:/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
STOP=${1:-}
DATA=${ANULM_V3:-data/v3/meta.json}
COMMON="--preset 1b --device cuda --moe-impl grouped --data $DATA --out ckpt_base_1b.pt \
  --block-size 1024 --batch-size 4 --steps 2770000 --optimizer adamw-lowmem --grad-ckpt \
  --lr 6e-4 --min-lr 6e-5 --warmup 2000 --schedule wsd --decay-frac 0.16 \
  --eval-every 10000 --eval-windows 400 --log-every 1000 --seed 2026 \
  --seq-balance-alpha 1e-4 --router-z-alpha 1e-3 --report-gpu-memory"
RESUME=""; [ -s ckpt_base_1b.pt.last ] && RESUME="--resume"
STOPARG=""; [ -n "$STOP" ] && STOPARG="--stop-at $STOP"
mkdir -p logs/base_1b
echo "=== BASE 1B: ${RESUME:-fresh start} ${STOPARG:-to the end} ($(date '+%Y-%m-%d %H:%M')) ===" | tee -a logs/base_1b/train.log
python -u -m anulm train $COMMON $RESUME $STOPARG >> logs/base_1b/train.log 2>> logs/base_1b/train.err
