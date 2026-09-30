"""The same questions to the old QA model and the new chat model, side by side."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os.chdir(ROOT)
from serve import Engine  # noqa: E402

QUESTIONS = [
    "Who are you?",
    "Give me three tips for staying healthy.",
    "What is the difference between a list and a tuple in Python?",
    "Write a short poem about the monsoon.",
    "Explain photosynthesis in simple words.",
    "आप कौन हैं?",
    "स्वस्थ रहने के तीन उपाय बताइए।",
    "ताजमहल के बारे में बताइए।",
    "बारिश के मौसम पर एक छोटी कविता लिखिए।",
]
for name, path in (("OLD QA model", "release/AnuLM-Hindi-QA-400M"), ("NEW chat model (v2 base)", "ckpt_chat_v2.pt")):
    if not (ROOT / path).exists():
        print(f"{name}: {path} missing")
        continue
    e = Engine(str(ROOT / path), "cuda")
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    for q in QUESTIONS:
        out = e.generate(q, 120, 0.3, 40, 0, mode="question", repetition_penalty=1.15)
        print(f"\nQ: {q}\nA: {' '.join(out['completion'].split())[:400]}", flush=True)
    del e
