#!/bin/bash
# Base v2 pretraining: ~11.3B tokens of Hindi / English / Python (experiments/build_v2.py),
# one pass, on the RTX 5070 Ti. Resumable: rerun the same command after any stop.
#
#   bash experiments/v2_train.sh            # fresh, or resume from ckpt_base_v2.pt.last
#   bash experiments/v2_train.sh 500000     # stop after step 500,000 (a phase)
#
# Architecture: the combo preset the released base and the coder used (24 experts
# top-4, width 192, no shared expert, sliding window 256 on layers 0-9; 398M total,
# 174M active). What changed since, each measured (docs/RESULTS.md):
#   * 1,024-token windows: the retrieval reader reads three passages at once
#   * no --grad-ckpt: the 16 GB card fits it, +30% tokens/s (section 32)
#   * --schedule wsd: a run of days can be extended without re-declaring the cosine
#   * validation held out by document hash at build time, all sources in proportion
#   * plain AdamW: Muon + the lab ideas gave 0.044 at 400M for 43% of the speed
# 2,750,000 steps x 4 x 1,024 = 11.26B tokens, ~0.99 of an epoch; at the measured
# 13.8k tok/s, ~9.5 days.
export PATH="/usr/bin:/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
STOP=${1:-}
COMMON="--preset 350m --device cuda --moe-impl grouped --data data/v2/meta.json --out ckpt_base_v2.pt \
  --block-size 1024 --batch-size 4 --steps 2750000 --lr 6e-4 --min-lr 6e-5 --warmup 2000 \
  --schedule wsd --decay-frac 0.2 --eval-every 10000 --eval-windows 400 --log-every 1000 \
  --seed 2026 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3"
PRESET="num_experts=24 moe_intermediate_size=192 num_experts_per_tok=4 num_shared_experts=0 sliding_window=256 max_window_layers=10 bias_update_rate=3e-3"
RESUME=""; [ -s ckpt_base_v2.pt.last ] && RESUME="--resume"
STOPARG=""; [ -n "$STOP" ] && STOPARG="--stop-at $STOP"
echo "=== BASE V2: ${RESUME:-fresh start} ${STOPARG:-to the end} ($(date '+%Y-%m-%d %H:%M')) ==="
python -u train.py $COMMON $RESUME $STOPARG --cfg $PRESET
