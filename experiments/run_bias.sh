#!/bin/bash
# bias_update_rate for the fine-grained preset: 24 experts finished at 1.76x
# imbalance against 1.42x for 12. Same 2000-step budget as the section-10
# ablations, combo preset, full corpus, three rates.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
COMMON="--preset 350m --device cuda --grad-ckpt --steps 2000 --batch-size 8 --block-size 512 --lr 6e-4 --moe-impl grouped --eval-every 500 --log-every 100 --seed 1337 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3 --data data/hindi_full.hi16k.bin"
PRESET="num_experts=24 moe_intermediate_size=192 num_experts_per_tok=4 num_shared_experts=0 sliding_window=256 max_window_layers=10"

for rate in 1e-3 3e-3 1e-2; do
  echo "=== BIAS RATE $rate ==="
  python -u train.py $COMMON --out ckpt_bias_$rate.pt --cfg $PRESET bias_update_rate=$rate 2>&1 | grep -v -i "warning\|triton" | grep -E "eval @|imbalance|done in"
done

echo "=== BENCH: bias rates ==="
python -u eval_bench.py ckpt_bias_1e-3.pt ckpt_bias_3e-3.pt ckpt_bias_1e-2.pt --device cuda 2>&1 | grep -v -i "warning\|triton"
echo "=== DONE ==="
