"""
Chat v5: the human-written chat data with a small share of clean Indian data (RESULTS 40's
next step), then recipe-F Decide adapters on it. Unattended; nothing touches the app.

    python experiments/chat_v5.py [--root <checkout>]

  1. data/chat_train_v5.jsonl = chat_train.jsonl (23,022 human-written pairs) plus ~20%:
     3,000 WikiHow Hindi guides (native Hindi; questions percent-decoded -- in v4 they were
     "कैसे %e0%a4%b5..."), 1,500 English->Hindi sentences, 1,250 English<->Hinglish pairs.
     No machine-translated or LLM-written answers this time.
  2. the post_v2 chat recipe -> ckpt_chat_v5.pt
  3. the same prompts to v3 and v5, plus how-to, translation and Hinglish prompts
  4. recipe-F adapters on v5, banking and Hindi, each scored on its test set
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "chat_v5"
F_ARGS = ["--lora-targets", "dense", "--lora-rank", "64", "--lr", "2e-4"]
KEYS = ("eval @", "done in", "held-out", "accuracy", "calibrated", "latency", "Traceback", "Error")
SHARE = {"wikihow": 3000, "nmt_en_hi": 1500, "hinglish_translate": 1250}

PROMPTS = [
    "Who are you?", "Give me three tips for staying healthy.", "Write a short poem about the monsoon.",
    "Explain photosynthesis in simple words.", "आप कौन हैं?", "स्वस्थ रहने के तीन उपाय बताइए।",
    "ताजमहल के बारे में बताइए।", "बारिश के मौसम पर एक छोटी कविता लिखिए।",
    "कैसे चाय बनाएँ?", "इस वाक्य का हिंदी में अनुवाद करें: The train to Delhi leaves at six in the morning.",
    "Translate to Hinglish: Please call me when you reach home.", "Aaj mausam kaisa rahega?",
]


def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(name, args, cap_h):
    lf = LOGS / f"{name}.log"
    log(f"--- {name} (cap {cap_h} h) -> {lf.name}")
    with open(lf, "w", encoding="utf-8") as f:
        p = subprocess.Popen([PY, "-u"] + args, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        try:
            rc = p.wait(timeout=cap_h * 3600)
        except subprocess.TimeoutExpired:
            p.terminate()
            p.wait(120)
            rc = "cap reached"
    for l in [l for l in lf.read_text(encoding="utf-8", errors="replace").splitlines()
              if any(k in l for k in KEYS)][-8:]:
        log("    " + l.strip()[:220])
    log(f"--- {name}: exit {rc}")


def prep():
    rng = random.Random(0)
    base = [json.loads(l) for l in open(ROOT / "data" / "chat_train.jsonl", encoding="utf-8")]
    held = {" ".join(json.loads(l)["question"].lower().split())
            for l in open(ROOT / "data" / "chat_heldout.jsonl", encoding="utf-8")}
    wiki = [json.loads(l) for l in open(ROOT / "data" / "chat_train_v4.jsonl", encoding="utf-8")
            if '"indic-instruct/wikihow"' in l]
    for r in wiki:
        r["question"] = urllib.parse.unquote(r["question"])
    pools = {"wikihow": wiki}
    for name in ("nmt_en_hi", "hinglish_translate"):
        pools[name] = [json.loads(l) for l in open(ROOT / "data" / "extra_hi" / f"{name}.jsonl", encoding="utf-8")]
    added = []
    for name, n in SHARE.items():
        pool = [r for r in pools[name] if " ".join(r["question"].lower().split()) not in held]
        rng.shuffle(pool)
        added += pool[:n]
    rows = base + added
    rng.shuffle(rows)
    out = ROOT / "data" / "chat_train_v5.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    still_encoded = sum("%e0%" in r["question"].lower() for r in rows)
    log(f"chat_train_v5: {len(rows)} pairs = {len(base)} human-written + {len(added)} added "
        f"({len(added) / len(rows):.0%}); {still_encoded} questions still percent-encoded")


def samples():
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    from serve import Engine
    with open(LOGS / "samples.log", "w", encoding="utf-8") as f:
        for name in ("ckpt_chat_v3.pt", "ckpt_chat_v5.pt"):
            e = Engine(str(ROOT / name), "cuda")
            f.write(f"\n{'=' * 70}\n{name}\n{'=' * 70}\n")
            for q in PROMPTS:
                out = e.generate(q, 120, 0.3, 40, 0, mode="question", repetition_penalty=1.15)
                f.write(f"\nQ: {q}\nA: {' '.join(out['completion'].split())[:400]}\n")
                f.flush()
            del e
    log(f"samples -> {LOGS / 'samples.log'}")


def main():
    global ROOT, LOGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--samples-only", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    ROOT, LOGS = a.root.resolve(), a.root.resolve() / "logs" / "chat_v5"
    LOGS.mkdir(parents=True, exist_ok=True)
    if a.samples_only:
        samples()
        return
    log("=== CHAT V5 start ===")
    prep()
    run("2_chat_sft", ["finetune.py", "--ckpt", "ckpt_base_v2.pt", "--qa", "data/chat_train_v5.jsonl",
                       "--heldout", "data/chat_heldout.jsonl", "--out", "ckpt_chat_v5.pt", "--epochs", "2",
                       "--batch-size", "4", "--lr", "5e-5", "--grad-ckpt", "--eval-every", "500",
                       "--log-every", "200"], 2.0)
    if not (ROOT / "ckpt_chat_v5.pt").exists():
        log("ckpt_chat_v5.pt missing; stopping")
        return
    run("3_samples", [str(Path(__file__).resolve()), "--root", str(ROOT), "--samples-only"], 0.5)
    for name, out, extra in (("4_F_bank_v5", "ckpt_decide_lora_F_v5.pt", []),
                             ("4_F_hi_v5", "ckpt_decide_lora_hi_F_v5.pt", ["--task", "massive-hi"])):
        run(f"{name}_train", ["decide.py", "train", "--ckpt", "ckpt_chat_v5.pt", "--out", out, "--mode", "lora",
                              *F_ARGS, *extra], 1.5)
        if (ROOT / out).exists():
            run(f"{name}_eval", ["decide.py", "eval", "--ckpt", out], 0.5)
    log("=== CHAT V5 DONE ===")


if __name__ == "__main__":
    main()
