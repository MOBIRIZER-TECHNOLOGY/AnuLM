#!/bin/bash
# Phase 1 of docs/CODER_PLAN.md: fetch everything the Python-coder run needs.
# Every download skips files already present, so rerun after any interruption.
export PATH="/usr/bin:/bin:$PATH"   # before dirname: a detached Git Bash has no MSYS tools on PATH
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8
set -o pipefail

echo "=== PYTHON: codeparrot-clean, 10 shards (~9 GB) ($(date +%H:%M)) ==="
[ -s data/python_big_raw.txt ] || python -u fetch_code.py --mb 9000 --shards 1 2 3 4 5 6 7 8 9 10 \
    --out data/python_big_raw.txt 2>&1 | tr '\r' '\n' | grep -E "wrote|streaming|Error|Traceback"

echo "=== ENGLISH: fineweb-edu, 2 shards (~4.3 GB) ($(date +%H:%M)) ==="
python -u fetch_hf.py HuggingFaceFW/fineweb-edu --prefix sample/10BT/000_ sample/10BT/001_ 2>&1 | tr '\r' '\n' | grep -E "files|Error|Traceback"

echo "=== PYTHON EXERCISES: jinaai/code_exercises (~0.5 GB) ($(date +%H:%M)) ==="
python -u fetch_hf.py jinaai/code_exercises --prefix data/ 2>&1 | tr '\r' '\n' | grep -E "files|Error|Traceback"

echo "=== INSTRUCTIONS: OpenCodeInstruct 1 shard + glaive ($(date +%H:%M)) ==="
python -u fetch_hf.py nvidia/OpenCodeInstruct --prefix data/train-00000- 2>&1 | tr '\r' '\n' | grep -E "files|Error|Traceback"
python -u fetch_hf.py glaiveai/glaive-code-assistant --prefix "" 2>&1 | tr '\r' '\n' | grep -E "files|Error|Traceback"

echo "=== TESTS: HumanEval + MBPP ($(date +%H:%M)) ==="
python -u fetch_hf.py openai/openai_humaneval --prefix openai_humaneval/ 2>&1 | tr '\r' '\n' | grep -E "files|Error|Traceback"
python -u fetch_hf.py google-research-datasets/mbpp --prefix sanitized/ 2>&1 | tr '\r' '\n' | grep -E "files|Error|Traceback"

echo "=== FETCH DONE ($(date +%H:%M)) ==="
du -sh data/python_big_raw.txt data/raw/* 2>/dev/null
