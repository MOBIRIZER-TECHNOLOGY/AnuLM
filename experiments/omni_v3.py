"""
Speech and vision decoders on the final base (TASKS.md "After base v2", step 7), unattended.

    python experiments/omni_v3.py [--root C:/workspace/AnuLM]

  0. grow ckpt_base_v2.pt with the audio/image vocabulary -> ckpt_mm_init_v2.pt
  1. English ASR, LibriSpeech train-clean-100, 3 epochs   -> ckpt_asr_en_v3.pt
  2. Hindi ASR from 1, FLEURS + IndicTTS mix, 6 epochs     -> ckpt_asr_hi_v3.pt
  3. captioner, Flickr30k, 2 epochs                        -> ckpt_caption_v3.pt
  4. VQA from 3, A-OKVQA, 3 epochs                         -> ckpt_vqa_v3.pt
  5. the served checkpoints and the new ones on the same evals

Measured on the free GPU: 0.29 s/step ASR, 0.26 s/step vision (batch 8), so
about 6.5 h. The old chain descends from ckpt_ctx2k.pt through ckpt_mm_init.pt;
this one from ckpt_base_v2.pt. Nothing touches the app: swapping a model in
(app_omni.py CKPTS) is step 8, by hand, after reading logs/omni_v3/pipeline.log.

Safe to re-launch: a finished step is recorded in state.json and skipped, and
a second copy exits while one is running. A step cut off by a reboot reruns
from its start (the trainers do not resume).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "omni_v3"

# name -> (args, cap in hours, the file it must produce)
STEPS = [
    ("0_grow", ["speech_vocab.py", "--ckpt", "ckpt_base_v2.pt", "--out", "ckpt_mm_init_v2.pt"], 0.3,
     "ckpt_mm_init_v2.pt"),
    ("1_asr_en", ["speech_encoder.py", "train", "--ckpt", "ckpt_mm_init_v2.pt", "--prefix", "data/speech_ls100",
                  "--out", "ckpt_asr_en_v3.pt", "--epochs", "3", "--batch-size", "4",
                  "--eval-every", "1000", "--log-every", "500"], 3.0, "ckpt_asr_en_v3.pt"),
    ("2_asr_hi", ["speech_encoder.py", "train", "--ckpt", "ckpt_asr_en_v3.pt", "--prefix", "data/hindi_mix/train",
                  "--out", "ckpt_asr_hi_v3.pt", "--epochs", "6", "--batch-size", "4",
                  "--eval-every", "1000", "--log-every", "500"], 2.0, "ckpt_asr_hi_v3.pt"),
    ("3_caption", ["vision_encoder.py", "train", "--ckpt", "ckpt_mm_init_v2.pt",
                   "--manifest", "data/flickr30k_train.jsonl", "--cache", "data/flickr30k_feats.f16",
                   "--out", "ckpt_caption_v3.pt", "--epochs", "2", "--batch-size", "8",
                   "--eval-every", "2000", "--log-every", "1000"], 4.0, "ckpt_caption_v3.pt"),
    ("4_vqa", ["vision_encoder.py", "train", "--ckpt", "ckpt_caption_v3.pt",
               "--manifest", "data/aokvqa_train.jsonl", "--cache", "data/aokvqa_feats.f16",
               "--out", "ckpt_vqa_v3.pt", "--epochs", "3", "--batch-size", "8",
               "--eval-every", "1000", "--log-every", "500"], 1.0, "ckpt_vqa_v3.pt"),
]

# served (app_omni.py CKPTS) vs new, each pair on the same eval
EVALS = [
    ("asr_en", "ckpt_asr_ls100_e3.pt", "ckpt_asr_en_v3.pt",
     ["speech_encoder.py", "eval", "--prefix", "data/speech_ls100", "--limit", "100"]),
    ("asr_hi", "ckpt_asr_hi_mix2.pt", "ckpt_asr_hi_v3.pt",
     ["speech_encoder.py", "eval", "--prefix", "data/fleurs_hi/dev", "--limit", "50"]),
    ("caption", "ckpt_caption_f30k_e2.pt", "ckpt_caption_v3.pt",
     ["vision_encoder.py", "eval", "--manifest", "data/flickr30k_val.jsonl", "--limit", "20"]),
    ("vqa", "ckpt_vqa_f30k.pt", "ckpt_vqa_v3.pt",
     ["vision_encoder.py", "vqa-eval", "--manifest", "data/aokvqa_val.jsonl"]),
]
KEYS = ("eval @", "done in", "held-out", "WER", "wer", "accuracy", "Traceback", "Error")


def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def state() -> dict:
    p = LOGS / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def mark(name: str, rc):
    s = state()
    s[name] = {"rc": rc, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = LOGS / "state.json.tmp"
    tmp.write_text(json.dumps(s, indent=2))
    os.replace(tmp, LOGS / "state.json")


def run(name: str, args: list[str], cap_h: float):
    if name in state():
        log(f"--- {name}: already done ({state()[name]}), skipping")
        return
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
    tail = [l for l in lf.read_text(encoding="utf-8", errors="replace").splitlines() if any(k in l for k in KEYS)]
    for l in tail[-8:]:
        log("    " + l.strip()[:220])
    log(f"--- {name}: exit {rc}")
    mark(name, rc)


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return f'"{pid}"' in out


def main():
    global ROOT, LOGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT, help="the checkout holding data/ and the checkpoints")
    a = ap.parse_args()
    ROOT, LOGS = a.root.resolve(), a.root.resolve() / "logs" / "omni_v3"
    LOGS.mkdir(parents=True, exist_ok=True)

    if (LOGS / "done.marker").exists():
        return
    pidf = LOGS / "pid"
    if pidf.exists() and alive(int(pidf.read_text() or 0)):
        return                                              # another copy is running
    pidf.write_text(str(os.getpid()))

    log(f"=== OMNI V3 start (pid {os.getpid()}) ===")
    for name, args, cap, out in STEPS:
        if name != "0_grow" and not (ROOT / args[args.index("--ckpt") + 1]).exists():
            log(f"--- {name}: its --ckpt is missing, skipping")
            continue
        if name == "0_grow" and (ROOT / out).exists():
            mark(name, "already on disk")
        run(name, args, cap)

    for key, old, new, args in EVALS:
        for tag, ck in (("served", old), ("v3", new)):
            if (ROOT / ck).exists():
                run(f"5_eval_{key}_{tag}", args[:2] + ["--ckpt", ck] + args[2:], 0.75)

    (LOGS / "done.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("=== OMNI V3 DONE ===")


if __name__ == "__main__":
    main()
