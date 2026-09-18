#!/bin/bash
# The data lever: v4 recipe, same 6000-step budget, on the whole dump (3.3x the
# text, 0.4 epochs instead of 1.4) -- then scored beside ckpt_hindi_v4.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
COMMON="--preset 350m --device cuda --grad-ckpt --steps 6000 --batch-size 8 --block-size 512 --lr 6e-4 --moe-impl grouped --eval-every 500 --log-every 100 --seed 1337 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3"

echo "=== FULL: v4 recipe on hindi_full (3.3x data) ==="
python -u train.py $COMMON --data data/hindi_full.hi16k.bin --out ckpt_hindi_full.pt 2>&1 | grep -v -i "warning\|triton"

echo "=== BENCH: v4 vs full ==="
python -u eval_bench.py ckpt_hindi_v4.pt ckpt_hindi_full.pt --device cuda 2>&1 | grep -v -i "warning\|triton"

echo "=== SAMPLE ==="
python sample.py --ckpt ckpt_hindi_full.pt --prompt "$(printf 'मुंबई\n\n')" --tokens 250 --device cuda 2>&1 | grep -v -i "warning\|triton"
echo "=== DONE ==="
