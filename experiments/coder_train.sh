#!/bin/bash
# Phases 3-6 of docs/CODER_PLAN.md: pretraining, in resumable pieces.
#
#     bash experiments/coder_train.sh 100000     # train to step 100k, then exit
#     bash experiments/coder_train.sh 200000     # later: resume, train to 200k
#     bash experiments/coder_train.sh            # no argument: run to the end
#
# The full schedule is 700,000 steps (2.87B tokens at batch 8 x 512, one pass, ~3.3
# days of GPU time). Each call resumes from ckpt_coder.pt.last if it exists
# and stops at the step given, after an eval and a save, so the machine can
# sleep between calls. If a call dies mid-way (sleep, crash), just call it
# again: at most --eval-every steps (~35 min) are lost.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
STOP=${1:-}
COMMON="--preset 350m --device cuda --grad-ckpt --steps 700000 --batch-size 8 --block-size 512 --lr 6e-4 --min-lr 6e-5 --warmup 2000 --moe-impl grouped --eval-every 5000 --log-every 500 --seed 1337 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3 --data data/coder.code32k.bin --out ckpt_coder.pt"
PRESET="num_experts=24 moe_intermediate_size=192 num_experts_per_tok=4 num_shared_experts=0 sliding_window=256 max_window_layers=10 bias_update_rate=3e-3"
RESUME=""; [ -s ckpt_coder.pt.last ] && RESUME="--resume"
STOPARG=""; [ -n "$STOP" ] && STOPARG="--stop-at $STOP"

echo "=== CODER PHASE: ${RESUME:-fresh start} ${STOPARG:-to the end} ($(date +%H:%M)) ==="
python -u train.py $COMMON $RESUME $STOPARG --cfg $PRESET 2>&1 | grep -v -i "warning\|triton"
echo "=== PHASE DONE ($(date +%H:%M)) ==="
