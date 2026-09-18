#!/bin/bash
# A/B: old training recipe vs the new one, same corpus text, same trunk, same
# compute. Then score everything on the shared benchmark.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
COMMON="--preset 350m --device cuda --grad-ckpt --steps 6000 --batch-size 8 --block-size 512 --lr 6e-4 --moe-impl grouped --eval-every 500 --log-every 50 --seed 1337"

echo "=== PATH check: $(which grep)"

echo "=== A: old recipe (no EOS, with replacement, no regularisers) ==="
python -u train.py $COMMON --data data/hindi_v4_noeos.hi16k.bin --sample-with-replacement --out ckpt_hindi_v4_base.pt 2>&1 | grep -v -i "warning\|triton"

echo "=== B: new recipe (EOS docs, no-replacement sampler, seq-balance 1e-4, z 1e-3) ==="
python -u train.py $COMMON --data data/hindi_v4.hi16k.bin --seq-balance-alpha 1e-4 --router-z-alpha 1e-3 --out ckpt_hindi_v4.pt 2>&1 | grep -v -i "warning\|triton"

echo "=== BENCH: all ==="
python -u eval_bench.py ckpt_hindi_2k.pt ckpt_hindi_350m.pt ckpt_hindi_bpe.pt ckpt_hindi_v2.pt ckpt_hindi_v3.pt ckpt_hindi_v4_base.pt ckpt_hindi_v4.pt --device cuda 2>&1 | grep -v -i "warning\|triton"

echo "=== SAMPLES ==="
for ck in ckpt_hindi_v4_base.pt ckpt_hindi_v4.pt; do
  echo "--- $ck  prompt: 'मुंबई\\n\\n'"
  python sample.py --ckpt $ck --prompt "$(printf 'मुंबई\n\n')" --tokens 250 --device cuda 2>&1 | grep -v -i "warning\|triton"
done
echo "=== DONE ==="
