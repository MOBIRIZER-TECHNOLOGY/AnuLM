#!/bin/bash
# RESULTS.md 31, phase 2: the textbook-corpus probe.
#
#     bash experiments/textbook_train.sh 250000
#
# Every argument below is copied from experiments/coder_train.sh -- the same
# preset, the same 24-expert config, the same lr 6e-4 -> 6e-5 cosine over a
# declared 700,000 steps, the same batch 8 x 512, the same seed 1337. Only
# --data and --out differ. That is the whole point: section 25 measured this
# recipe on a corpus that is 53% raw GitHub Python, and the probe at step
# 250,000 scored MBPP 5.1% / HumanEval 0.6%. If the textbook corpus lands
# somewhere else at the same step, the data moved it.
#
# The cosine is declared over 700,000 steps and stopped at 250,000 on
# purpose. Re-declaring it over 250,000 would decay the learning rate to its
# floor by the probe and flatter the comparison with a schedule section 25
# never ran.
#
# 250,000 steps is ~1.02B tokens, ~28 h on one RTX 5070 Ti. Resumable: each
# call picks up from ckpt_textbook.pt.last, so a crash or a sleep costs at
# most one --eval-every interval.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
STOP=${1:-250000}
COMMON="--preset 350m --device cuda --grad-ckpt --steps 700000 --batch-size 8 --block-size 512 --lr 6e-4 --min-lr 6e-5 --warmup 2000 --moe-impl grouped --eval-every 5000 --log-every 500 --seed 1337 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3 --data data/textbook_mix.code32k.bin --out ckpt_textbook.pt"
PRESET="num_experts=24 moe_intermediate_size=192 num_experts_per_tok=4 num_shared_experts=0 sliding_window=256 max_window_layers=10 bias_update_rate=3e-3"
RESUME=""; [ -s ckpt_textbook.pt.last ] && RESUME="--resume"

echo "=== TEXTBOOK PHASE: ${RESUME:-fresh start} to $STOP ($(date +%H:%M)) ==="
python -u train.py $COMMON $RESUME --stop-at "$STOP" --cfg $PRESET 2>&1 | grep -v -i "warning\|triton"
echo "=== PHASE DONE ($(date +%H:%M)) ==="
