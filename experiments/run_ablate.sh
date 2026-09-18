#!/bin/bash
# Architecture ablations on the v4 recipe at a 2000-step budget (~12 min each),
# then everything scored on the shared benchmark.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
COMMON="--preset 350m --device cuda --grad-ckpt --steps 2000 --batch-size 8 --block-size 512 --lr 6e-4 --moe-impl grouped --eval-every 500 --log-every 100 --seed 1337 --data data/hindi_v4.hi16k.bin --seq-balance-alpha 1e-4 --router-z-alpha 1e-3"

run() {  # name, then --cfg overrides
  name=$1; shift
  echo "=== ABLATION $name: $* ==="
  python -u train.py $COMMON --out ckpt_abl_$name.pt --cfg "$@" 2>&1 | grep -v -i "warning\|triton"
}

run baseline
run noshared     num_shared_experts=0
run finegrained  num_experts=24 moe_intermediate_size=192 num_experts_per_tok=6
run top2         num_experts_per_tok=2
run window256    sliding_window=256 max_window_layers=10

echo "=== BENCH: ablations ==="
python -u eval_bench.py ckpt_abl_baseline.pt ckpt_abl_noshared.pt ckpt_abl_finegrained.pt ckpt_abl_top2.pt ckpt_abl_window256.pt --device cuda 2>&1 | grep -v -i "warning\|triton"
echo "=== DONE ==="
