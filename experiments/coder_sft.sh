#!/bin/bash
# Phase 7 of docs/CODER_PLAN.md: instruction-tune the coder and measure it.
#
#     bash experiments/coder_sft.sh                 # from ckpt_coder.pt
#     EX_CAP=200000 bash experiments/coder_sft.sh   # 1.5 h of tuning, not 6.4 h
#     EX_CAP=0 bash experiments/coder_sft.sh        # all 1.29M exercise pairs
#     BASE=ckpt_coder.pt.best bash experiments/coder_sft.sh
#
# Pairs: the code_exercises problems (phi-1 style, plain Python answers),
# OpenCodeInstruct and glaive answers reduced to their Python block, plus the
# docstring pairs from make_qa_py.py. One epoch, then pass@1 on HumanEval
# and MBPP for base and tuned.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
BASE=${BASE:-ckpt_coder.pt}
OUT=${OUT:-ckpt_coder_sft.pt}

echo "=== 1. PAIRS ($(date +%H:%M)) ==="
if [ -s data/sft_code.jsonl ] && [ -s $OUT.last ]; then
  echo "  resuming: keeping the existing data/sft_code.jsonl ($(wc -l < data/sft_code.jsonl) pairs)"
else
python - <<PY
import itertools, json, random
from pathlib import Path

# Read lazily and cap on the way in. sft_exercises.jsonl is 1.1 GB / 1.29M
# pairs; slurping it builds several GB of dicts, and all of it is 6.4 h of
# tuning. EX_CAP trades that against wall clock -- 200k pairs is 1.5 h and
# still more instruction data per epoch than phi-1 used.
def load(path, cap=None):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in itertools.islice((l for l in f if l.strip()), cap)]

rng = random.Random(7)
parts = {"exercises": load("data/sft_exercises.jsonl", ${EX_CAP:-200000} or None),
         "opencode": load("data/sft_opencode.jsonl", 60000),
         "glaive": load("data/sft_glaive.jsonl", 40000),
         "docstrings": load("data/qa_python.jsonl")}
train = [it for v in parts.values() for it in v]
rng.shuffle(train)
held, train = train[:800], train[800:]
for path, rows in (("data/sft_code.jsonl", train), ("data/sft_code_heldout.jsonl", held)):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for it in rows: f.write(json.dumps(it, ensure_ascii=False) + "\n")
print("sft pairs:", {k: len(v) for k, v in parts.items()}, "-> train", len(train), "held-out", len(held))
PY
fi

echo "=== 2. FINE-TUNE ($(date +%H:%M)) ==="
# Resume if a previous invocation left a checkpoint: the tune is ~1.5 h at
# EX_CAP=200000 and longer uncapped, which is long enough to be interrupted.
RESUME=""; [ -s $OUT.last ] && RESUME="--resume"
echo "  ${RESUME:-fresh start}"
python -u finetune.py --ckpt $BASE --qa data/sft_code.jsonl --heldout data/sft_code_heldout.jsonl \
    --out $OUT --epochs 1 --lr 5e-5 --eval-every 500 --grad-ckpt $RESUME 2>&1 | grep -v -i "warning\|triton"

echo "=== 3. PASS@1 ($(date +%H:%M)) ==="
# Concurrently: decoding is launch-bound, so two eval processes raise GPU
# utilisation rather than halving each other's speed (8% -> 61% measured),
# and the pair finishes in ~1.2 h instead of ~1.9 h sequentially.
python -u eval_code.py $BASE $OUT --bench humaneval --device cuda > eval_humaneval_sft.log 2>&1 &
python -u eval_code.py $BASE $OUT --bench mbpp --device cuda > eval_mbpp_sft.log 2>&1 &
wait
for f in eval_humaneval_sft.log eval_mbpp_sft.log; do
  echo "--- $f"; tr '\r' '\n' < $f | grep -E "^checkpoint|^ckpt_|^reference|Traceback|Error"
done
echo "=== DONE ($(date +%H:%M)) ==="
