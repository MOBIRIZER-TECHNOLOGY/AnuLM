#!/bin/bash
# The register gap: train on Wikipedia + Wikisource (interleaved), combo
# preset, same 6000-step budget; score on BOTH benchmarks against the
# Wikipedia-only combo and full checkpoints.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
COMMON="--preset 350m --device cuda --grad-ckpt --steps 6000 --batch-size 8 --block-size 512 --lr 6e-4 --moe-impl grouped --eval-every 500 --log-every 100 --seed 1337 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3"
PRESET="num_experts=24 moe_intermediate_size=192 num_experts_per_tok=4 num_shared_experts=0 sliding_window=256 max_window_layers=10"

echo "=== MIXED: combo preset on Wikipedia + Wikisource ==="
python -u train.py $COMMON --data data/mixed.hi16k.bin --out ckpt_hindi_mixed.pt --cfg $PRESET 2>&1 | grep -v -i "warning\|triton"

echo "=== BENCH WIKIPEDIA ==="
python -u eval_bench.py ckpt_hindi_full.pt ckpt_hindi_combo.pt ckpt_hindi_mixed.pt --device cuda 2>&1 | grep -v -i "warning\|triton"
echo "=== BENCH WIKISOURCE ==="
python -u eval_bench.py ckpt_hindi_full.pt ckpt_hindi_combo.pt ckpt_hindi_mixed.pt --bench data/bench_wikisource.txt --device cuda 2>&1 | grep -v -i "warning\|triton"

echo "=== SAMPLES ==="
for prompt in "मुंबई" "उसने कहा"; do
  echo "--- prompt: $prompt"
  python sample.py --ckpt ckpt_hindi_mixed.pt --prompt "$(printf '%s\n\n' "$prompt")" --tokens 200 --device cuda --seed 7 2>&1 | grep -v -i "warning\|triton"
done
echo "=== DONE ==="
