#!/bin/bash
# More steps on more data: the mixed corpus at 3x the budget (18k steps, ~0.5
# epochs of 140.6M tokens). Fresh schedule, so it is a clean 3x-compute
# comparison against ckpt_hindi_mixed (6k). Combo preset + bias rate 3e-3.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
COMMON="--preset 350m --device cuda --grad-ckpt --steps 18000 --batch-size 8 --block-size 512 --lr 6e-4 --moe-impl grouped --eval-every 1000 --log-every 200 --seed 1337 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3"
PRESET="num_experts=24 moe_intermediate_size=192 num_experts_per_tok=4 num_shared_experts=0 sliding_window=256 max_window_layers=10 bias_update_rate=3e-3"

echo "=== LONG: combo preset, mixed corpus, 18k steps ==="
python -u train.py $COMMON --data data/mixed.hi16k.bin --out ckpt_hindi_mixed18k.pt --cfg $PRESET 2>&1 | grep -v -i "warning\|triton"

echo "=== BENCH WIKIPEDIA ==="
python -u eval_bench.py ckpt_hindi_full.pt ckpt_hindi_mixed.pt ckpt_hindi_mixed18k.pt --device cuda 2>&1 | grep -v -i "warning\|triton"
echo "=== BENCH WIKISOURCE ==="
python -u eval_bench.py ckpt_hindi_full.pt ckpt_hindi_mixed.pt ckpt_hindi_mixed18k.pt --bench data/bench_wikisource.txt --device cuda 2>&1 | grep -v -i "warning\|triton"

echo "=== SAMPLES ==="
for prompt in "मुंबई" "उसने कहा" "भारतीय क्रिकेट टीम"; do
  echo "--- prompt: $prompt"
  python sample.py --ckpt ckpt_hindi_mixed18k.pt --prompt "$(printf '%s\n\n' "$prompt")" --tokens 200 --device cuda --seed 7 2>&1 | grep -v -i "warning\|triton"
done
echo "=== DONE ==="
