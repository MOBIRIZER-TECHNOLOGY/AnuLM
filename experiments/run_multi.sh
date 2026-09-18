#!/bin/bash
# Pretrain on the Hindi + English + Python corpus with the 32k tokenizer.
# Same recipe, preset, budget and batch as ckpt_hindi_mixed36k (the control),
# so the changes are tokenizer, data and the fp32 router fix. Then score
# both checkpoints on all four benchmarks.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
COMMON="--preset 350m --device cuda --grad-ckpt --steps 36000 --batch-size 8 --block-size 512 --lr 6e-4 --moe-impl grouped --eval-every 1000 --log-every 200 --seed 1337 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3"
PRESET="num_experts=24 moe_intermediate_size=192 num_experts_per_tok=4 num_shared_experts=0 sliding_window=256 max_window_layers=10 bias_update_rate=3e-3"

echo "=== MULTI: combo preset, 36k steps ($(date +%H:%M)) ==="
python -u train.py $COMMON --data data/multi.multi32k.bin --out ckpt_multi36k.pt --cfg $PRESET 2>&1 | grep -v -i "warning\|triton"

for b in bench_hindi bench_wikisource bench_english bench_python; do
  echo "=== BENCH $b ==="
  python -u eval_bench.py ckpt_hindi_mixed36k.pt ckpt_multi36k.pt --bench data/$b.txt --device cuda 2>&1 | grep -v -i "warning\|triton"
done

echo "=== SAMPLES ==="
for prompt in "मुंबई" "Mumbai" "The history of India" "def fibonacci(n):" "import numpy as np"; do
  echo "--- prompt: $prompt"
  python sample.py --ckpt ckpt_multi36k.pt --prompt "$(printf '%s\n\n' "$prompt")" --tokens 150 --device cuda --seed 7 2>&1 | grep -v -i "warning\|triton"
done
echo "=== DONE ($(date +%H:%M)) ==="
