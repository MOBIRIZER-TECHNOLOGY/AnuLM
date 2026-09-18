#!/bin/bash
# The ablation winners together: fine-grained experts, top-4, no shared expert,
# window on the lower half. v4 recipe, full corpus, same 6000-step budget as
# ckpt_hindi_full, which is the control.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
COMMON="--preset 350m --device cuda --grad-ckpt --steps 6000 --batch-size 8 --block-size 512 --lr 6e-4 --moe-impl grouped --eval-every 500 --log-every 100 --seed 1337 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3 --data data/hindi_full.hi16k.bin"

echo "=== COMBO: 24x192 top-4, no shared, window 256 on layers 0-9 ==="
python -u train.py $COMMON --out ckpt_hindi_combo.pt --cfg num_experts=24 moe_intermediate_size=192 num_experts_per_tok=4 num_shared_experts=0 sliding_window=256 max_window_layers=10 2>&1 | grep -v -i "warning\|triton"

echo "=== BENCH: full vs combo ==="
python -u eval_bench.py ckpt_hindi_full.pt ckpt_hindi_combo.pt --device cuda 2>&1 | grep -v -i "warning\|triton"

echo "=== SAMPLES ==="
for prompt in "मुंबई" "भारतीय क्रिकेट टीम" "हिन्दी साहित्य"; do
  echo "--- prompt: $prompt"
  python sample.py --ckpt ckpt_hindi_combo.pt --prompt "$(printf '%s\n\n' "$prompt")" --tokens 200 --device cuda --seed 7 2>&1 | grep -v -i "warning\|triton"
done
echo "=== DONE ==="
