#!/bin/bash
# Data-quality lever: the prose-filtered full corpus, same tokenizer, same
# recipe and budget as ckpt_hindi_full (the control).
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
COMMON="--preset 350m --device cuda --grad-ckpt --steps 6000 --batch-size 8 --block-size 512 --lr 6e-4 --moe-impl grouped --eval-every 500 --log-every 100 --seed 1337 --seq-balance-alpha 1e-4 --router-z-alpha 1e-3"

echo "=== PROSE: v4 recipe on hindi_prose (list lines dropped, 89% of the full corpus) ==="
python -u train.py $COMMON --data data/hindi_prose.hi16k.bin --out ckpt_hindi_prose.pt 2>&1 | grep -v -i "warning\|triton"

echo "=== BENCH: full vs prose vs combo ==="
python -u eval_bench.py ckpt_hindi_full.pt ckpt_hindi_prose.pt ckpt_hindi_combo.pt --device cuda 2>&1 | grep -v -i "warning\|triton"

echo "=== SAMPLES ==="
for prompt in "मुंबई" "भारतीय क्रिकेट टीम"; do
  echo "--- prompt: $prompt"
  python sample.py --ckpt ckpt_hindi_prose.pt --prompt "$(printf '%s\n\n' "$prompt")" --tokens 200 --device cuda --seed 7 2>&1 | grep -v -i "warning\|triton"
done
echo "=== BENCH OUT-OF-REGISTER: Hindi Wikisource (literature), every 386M checkpoint ==="
python -u eval_bench.py ckpt_hindi_bpe.pt ckpt_hindi_v2.pt ckpt_hindi_v3.pt ckpt_hindi_v4_base.pt ckpt_hindi_v4.pt ckpt_hindi_full.pt ckpt_hindi_prose.pt ckpt_hindi_combo.pt --bench data/bench_wikisource.txt --device cuda 2>&1 | grep -v -i "warning\|triton"

echo "=== CONTEXT: does the 256-window cost anything past 512? (baseline vs window256 ablations) ==="
for ck in ckpt_abl_baseline.pt ckpt_abl_window256.pt; do
  python -u eval_context.py --ckpt $ck --data data/hindi_v4.txt --device cuda --multipliers 1 2 4 --batches 24 2>&1 | grep -v -i "warning\|triton"
done
echo "=== DONE ==="
