#!/bin/bash
# Question answering in Hindi, English and Python, from the three-language
# base checkpoint. Builds the English and Python pair sets (the Hindi one is
# make_qa.py's, already built), merges them with Hindi capped so no language
# dominates, fine-tunes once, and scores base vs tuned per language.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
BASE=${BASE:-ckpt_multi36k.pt}
OUT=${OUT:-ckpt_multi_qa.pt}

echo "=== 1. PAIRS ($(date +%H:%M)) ==="
[ -s data/qa_hindi.jsonl ] || python -u make_qa.py --data data/hindi_full.txt --out data/qa_hindi.jsonl --heldout data/qa_hindi_heldout.jsonl 2>&1 | head -3
python -u make_qa_en.py --data data/english.txt --out data/qa_english.jsonl --heldout data/qa_english_heldout.jsonl 2>&1 | head -2
python -u make_qa_py.py --data data/python.txt --out data/qa_python.jsonl --heldout data/qa_python_heldout.jsonl --max 20000 2>&1 | head -1
python - <<'PY'
import json, random
from pathlib import Path
# split("\n"): splitlines() also breaks on U+2028, which json.dumps leaves raw
load = lambda p: [json.loads(l) for l in Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]
hi, en, py = load("data/qa_hindi.jsonl"), load("data/qa_english.jsonl"), load("data/qa_python.jsonl")
for it in hi: it.setdefault("lang", "hi")
rng = random.Random(7)
rng.shuffle(hi); hi = hi[:20000]                      # cap: 55k Hindi would drown the rest
train = hi + en + py; rng.shuffle(train)
held = load("data/qa_hindi_heldout.jsonl") + load("data/qa_english_heldout.jsonl") + load("data/qa_python_heldout.jsonl")
for it in held: it.setdefault("lang", "hi")
for path, rows in (("data/qa_multi.jsonl", train), ("data/qa_multi_heldout.jsonl", held)):
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for it in rows: f.write(json.dumps(it, ensure_ascii=False) + "\n")
print(f"merged: train {len(train)} = hindi {len(hi)} + english {len(en)} + python {len(py)}; held-out {len(held)}")
PY

echo "=== 2. FINE-TUNE ($(date +%H:%M)) ==="
python -u finetune.py --ckpt $BASE --qa data/qa_multi.jsonl --heldout data/qa_multi_heldout.jsonl \
    --out $OUT --epochs 2 --grad-ckpt 2>&1 | grep -v -i "warning\|triton"

echo "=== 3. EVAL PER LANGUAGE ($(date +%H:%M)) ==="
for lang in hindi english python; do
  echo "--- $lang"
  python -u eval_qa.py $BASE $OUT --heldout data/qa_${lang}_heldout.jsonl --device cuda --limit 300 --gen 60 2>&1 | grep -v -i "warning\|triton"
done
echo "=== DONE ($(date +%H:%M)) ==="
