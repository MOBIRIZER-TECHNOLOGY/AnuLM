"""
More Indian data, one night, unattended (2026-10-18).

    python experiments/indic_v4.py [--root C:/workspace/AnuLM]

  1. prep: the unused IndicTTS-Hindi speaker (gender=1) as 16 kHz clips;
     data/hindi_mix2/train.jsonl = hindi_mix (FLEURS train + speaker 0) + speaker 1
  2. prep: FLEURS hi test (418 clips) extracted; data/fleurs_hi/devtest = dev + test,
     so Hindi WER is scored on ~650 clips instead of 50
  3. Hindi ASR: the omni_v3 recipe (from ckpt_asr_en_v3.pt, 6 epochs) on hindi_mix2
     -> ckpt_asr_hi_v4.pt; only the data differs from ckpt_asr_hi_v3.pt
  4. prep: data/chat_train_v4.jsonl = chat_train.jsonl + the Hindi subsets of
     ai4bharat/indic-instruct-data-v0.1 (anudesh, wikihow, dolly, oasst1; first
     user/assistant turn), anything matching a held-out question removed
  5. chat: the post_v2 recipe on the new data -> ckpt_chat_v4.pt; samples vs v3
  6. Decide adapters, recipe F (dense, rank 64, lr 2e-4): banking and Hindi on
     ckpt_chat_v4.pt, and Hindi on ckpt_chat_v3.pt (banking F on v3 exists)
  7. evals: Hindi WER v3 vs v4 on devtest; every adapter on its test set

Nothing touches the app. Re-launch is safe: finished steps are skipped.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "indic_v4"
F_ARGS = ["--lora-targets", "dense", "--lora-rank", "64", "--lr", "2e-4"]
KEYS = ("eval @", "done in", "held-out", "WER", "accuracy", "calibrated", "latency", "Traceback", "Error")


def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def state() -> dict:
    p = LOGS / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def mark(name, rc):
    s = state()
    s[name] = {"rc": rc, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (LOGS / "state.json.tmp").write_text(json.dumps(s, indent=2))
    os.replace(LOGS / "state.json.tmp", LOGS / "state.json")


def step(name):
    """Run a Python prep function once, recording it like a subprocess step."""
    def wrap(fn):
        def run_once():
            if name in state():
                log(f"--- {name}: already done, skipping")
                return
            log(f"--- {name}")
            try:
                msg = fn()
                log(f"    {msg}")
                mark(name, 0)
            except Exception as e:
                log(f"    FAILED: {type(e).__name__}: {e}")
                mark(name, f"failed: {type(e).__name__}")
        return run_once
    return wrap


def run(name, args, cap_h):
    if name in state():
        log(f"--- {name}: already done, skipping")
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


# ------------------------------------------------------------------ prep
@step("1_prep_tts_speaker1")
def prep_tts():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    sys.path.insert(0, str(ROOT))
    from audio_codec import resample
    from speech_data import MAX_SECONDS, MIN_SECONDS

    clips = ROOT / "data" / "speech_hi_tts_m_clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for shard in sorted((ROOT / "data" / "indictts_hi" / "data").glob("*.parquet")):
        pf = pq.ParquetFile(str(shard))
        genders = pf.read(columns=["gender"]).column("gender").to_pylist()
        if 1 not in genders:
            continue
        for batch in pf.iter_batches(batch_size=64):
            for r in batch.to_pylist():
                if r.get("gender") != 1:
                    continue
                text, audio = (r.get("text") or "").strip().lower(), r.get("audio")
                raw = audio.get("bytes") if isinstance(audio, dict) else audio
                if not text or not raw:
                    skipped += 1
                    continue
                data, sr = sf.read(io.BytesIO(raw), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                dur = len(data) / sr
                if not (MIN_SECONDS <= dur <= MAX_SECONDS):
                    skipped += 1
                    continue
                dest = clips / f"m{len(rows):06d}.wav"
                sf.write(str(dest), np.asarray(resample(data, sr, 16000), dtype=np.float32), 16000,
                         subtype="PCM_16")
                rows.append({"audio": str(dest.resolve()), "text": text})
    if not rows:
        raise RuntimeError("no gender=1 rows found")
    (ROOT / "data" / "speech_hi_tts_m.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    base = [json.loads(l) for l in open(ROOT / "data" / "hindi_mix" / "train.jsonl", encoding="utf-8")]
    mix = base + rows
    random.Random(0).shuffle(mix)
    (ROOT / "data" / "hindi_mix2").mkdir(exist_ok=True)
    (ROOT / "data" / "hindi_mix2" / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mix), encoding="utf-8")
    return f"speaker 1: {len(rows)} clips ({skipped} skipped); hindi_mix2: {len(mix)} clips (was {len(base)})"


@step("2_prep_fleurs_test")
def prep_fleurs():
    src = ROOT / "data" / "fleurs" / "data" / "hi_in"
    out = ROOT / "data" / "fleurs_hi" / "test"
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src / "audio" / "test.tar.gz") as t:
        for m in t.getmembers():
            if m.isfile() and m.name.endswith(".wav"):
                (out / Path(m.name).name).write_bytes(t.extractfile(m).read())
    rows = []
    for line in open(src / "test.tsv", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 2 and (out / parts[1]).exists():
            rows.append({"audio": str((out / parts[1]).resolve()), "text": parts[2]})
    fh = ROOT / "data" / "fleurs_hi"
    (fh / "test.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    dev = (fh / "dev.jsonl").read_text(encoding="utf-8")
    (fh / "devtest.jsonl").write_text(dev + (fh / "test.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    return f"test: {len(rows)} clips; devtest: {len(dev.splitlines()) + len(rows)} clips"


def _norm_q(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


@step("4_prep_chat_data")
def prep_chat():
    import pyarrow.parquet as pq
    base = [json.loads(l) for l in open(ROOT / "data" / "chat_train.jsonl", encoding="utf-8")]
    held = {_norm_q(json.loads(l)["question"]) for l in open(ROOT / "data" / "chat_heldout.jsonl", encoding="utf-8")}
    seen = {_norm_q(r["question"]) for r in base}
    added, counts, dropped = [], {}, 0
    for name in ("anudesh", "wikihow", "dolly", "oasst1"):
        n = 0
        for r in pq.read_table(ROOT / "data" / "indic_instruct" / f"{name}-hi.parquet").to_pylist():
            if "instruction" in r:                       # dolly: instruction / context / response columns
                ctx = (r.get("context") or "").strip()
                q = (r.get("instruction") or "").strip() + (f"\n\n{ctx}" if ctx else "")
                a = (r.get("response") or "").strip()
            else:                                        # the others: a messages list
                msgs = r.get("messages") or []
                if len(msgs) < 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
                    dropped += 1
                    continue
                q, a = (msgs[0].get("content") or "").strip(), (msgs[1].get("content") or "").strip()
            k = _norm_q(q)
            if not q or not a or k in held or k in seen or len(q) > 1500 or len(a) > 3000:
                dropped += 1
                continue
            seen.add(k)
            added.append({"question": q, "answer": a, "lang": "hi", "source": f"indic-instruct/{name}"})
            n += 1
        counts[name] = n
    rows = base + added
    random.Random(0).shuffle(rows)
    (ROOT / "data" / "chat_train_v4.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    hi = sum(r.get("lang") == "hi" for r in rows)
    return (f"added {len(added)} ({counts}), dropped {dropped}; chat_train_v4 {len(rows)} pairs, "
            f"{hi / len(rows):.0%} Hindi (was {sum(r.get('lang') == 'hi' for r in base) / len(base):.0%})")


def alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    return f'"{pid}"' in out


def main():
    global ROOT, LOGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args()
    ROOT, LOGS = a.root.resolve(), a.root.resolve() / "logs" / "indic_v4"
    LOGS.mkdir(parents=True, exist_ok=True)
    if (LOGS / "done.marker").exists():
        return
    pidf = LOGS / "pid"
    if pidf.exists() and pidf.read_text().strip() and alive(int(pidf.read_text())):
        return
    pidf.write_text(str(os.getpid()))
    log(f"=== INDIC V4 start (pid {os.getpid()}) ===")

    # Hindi speech
    prep_tts()
    prep_fleurs()
    if (ROOT / "data" / "hindi_mix2" / "train.jsonl").exists():
        run("3_asr_hi", ["speech_encoder.py", "train", "--ckpt", "ckpt_asr_en_v3.pt",
                         "--prefix", "data/hindi_mix2/train", "--out", "ckpt_asr_hi_v4.pt", "--epochs", "6",
                         "--batch-size", "4", "--eval-every", "2000", "--log-every", "1000"], 3.5)

    # Hindi chat
    prep_chat()
    if (ROOT / "data" / "chat_train_v4.jsonl").exists():
        run("5_chat_sft", ["finetune.py", "--ckpt", "ckpt_base_v2.pt", "--qa", "data/chat_train_v4.jsonl",
                           "--heldout", "data/chat_heldout.jsonl", "--out", "ckpt_chat_v4.pt", "--epochs", "2",
                           "--batch-size", "4", "--lr", "5e-5", "--grad-ckpt", "--eval-every", "1000",
                           "--log-every", "200"], 2.5)
        if (ROOT / "ckpt_chat_v4.pt").exists():
            run("5_chat_samples", [str(ROOT / "experiments" / "chat_samples.py"), "ckpt_chat_v3.pt",
                                   "ckpt_chat_v4.pt"], 0.5)

    # Decide adapters, recipe F
    jobs = [("6_F_bank_v4", "ckpt_chat_v4.pt", "ckpt_decide_lora_F_v4.pt", []),
            ("6_F_hi_v4", "ckpt_chat_v4.pt", "ckpt_decide_lora_hi_F_v4.pt", ["--task", "massive-hi"]),
            ("6_F_hi_v3", "ckpt_chat_v3.pt", "ckpt_decide_lora_hi_F.pt", ["--task", "massive-hi"])]
    for name, chat, out, extra in jobs:
        if not (ROOT / chat).exists():
            log(f"--- {name}: {chat} missing, skipping")
            continue
        run(f"{name}_train", ["decide.py", "train", "--ckpt", chat, "--out", out, "--mode", "lora",
                              *F_ARGS, *extra], 1.5)
        if (ROOT / out).exists():
            run(f"{name}_eval", ["decide.py", "eval", "--ckpt", out], 0.5)
    if (ROOT / "ckpt_decide_lora_F.pt").exists():
        run("6_F_bank_v3_eval", ["decide.py", "eval", "--ckpt", "ckpt_decide_lora_F.pt"], 0.5)

    # Hindi WER on ~650 clips
    for tag, ck in (("v3", "ckpt_asr_hi_v3.pt"), ("v4", "ckpt_asr_hi_v4.pt")):
        if (ROOT / ck).exists() and (ROOT / "data" / "fleurs_hi" / "devtest.jsonl").exists():
            run(f"7_wer_hi_{tag}", ["speech_encoder.py", "eval", "--ckpt", ck, "--prefix", "data/fleurs_hi/devtest",
                                    "--limit", "1000"], 1.0)

    (LOGS / "done.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("=== INDIC V4 DONE ===")


if __name__ == "__main__":
    main()
