"""
Overnight 2026-10-18: Hindi speech recognition on many real speakers, and Indian-accented
English measured. Unattended; nothing touches the app.

    python experiments/speech_v5.py [--root <checkout>]

  1. prep (CPU): Kathbath Hindi (ai4bharat, gated, m4a in parquet) -> 16 kHz clips; train split
     -> data/kathbath_hi/train.jsonl, valid -> data/kathbath_hi/test.jsonl (held out)
  2. prep (CPU): MUCS 2021 Hindi-English (OpenSLR 104, CC BY-SA 4.0; Kaldi segments over long
     recordings) -> utterance clips; code-switched text, Hindi in Devanagari, English in Latin
  3. prep (CPU): Monsoon en-IN (CC BY 4.0, test only) -> clips + text. Only audio and text are
     kept: the parquet also holds speakers' personal details, which are never read out.
  4. wait for the GPU; record the Indian-banking benchmark suite on the API (if its adapter exists)
  5. Hindi ASR: from ckpt_asr_hi_v3.pt, one epoch of Kathbath + MUCS + hindi_mix (FLEURS train +
     IndicTTS speaker 0; speaker 1 is left out, RESULTS 40) -> ckpt_asr_hi_v5.pt
  6. evals: Hindi v3 vs v5 on FLEURS dev+test (657), Kathbath test, MUCS test; English v3 on
     Monsoon en-IN (the Indian-accent baseline)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "speech_v5"
KEYS = ("eval @", "done in", "held-out", "WER", "Traceback", "Error", "acc ", "p50")
MIN_S, MAX_S = 0.4, 20.0


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


def once(name, fn):
    if name in state():
        log(f"--- {name}: already done, skipping")
        return
    log(f"--- {name}")
    try:
        log(f"    {fn()}")
        mark(name, 0)
    except Exception as e:
        log(f"    FAILED: {type(e).__name__}: {e}")
        mark(name, f"failed: {type(e).__name__}")


def run(name, args, cap_h, env_extra=None):
    if name in state():
        log(f"--- {name}: already done, skipping")
        return
    lf = LOGS / f"{name}.log"
    log(f"--- {name} (cap {cap_h} h) -> {lf.name}")
    with open(lf, "w", encoding="utf-8") as f:
        p = subprocess.Popen([PY, "-u"] + args, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env_extra or {})})
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
    mark(name, rc)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# ------------------------------------------------------------------ Kathbath
def _decode_m4a(raw: bytes):
    import av
    import numpy as np
    c = av.open(io.BytesIO(raw))
    rs = av.AudioResampler(format="s16", layout="mono", rate=16000)
    out = []
    for frame in c.decode(audio=0):
        out += [f.to_ndarray().reshape(-1) for f in rs.resample(frame)]
    out += [f.to_ndarray().reshape(-1) for f in rs.resample(None)]
    c.close()
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)


def _kathbath_shard(args):
    shard, clip_dir = args
    import pyarrow.parquet as pq
    import soundfile as sf
    clip_dir = Path(clip_dir)
    rows, bad = [], 0
    for batch in pq.ParquetFile(shard).iter_batches(batch_size=128):
        for r in batch.to_pylist():
            try:
                pcm = _decode_m4a(r["audio_filepath"]["bytes"])
            except Exception:
                bad += 1
                continue
            dur = len(pcm) / 16000
            text = (r.get("text") or "").strip()
            if not text or not (MIN_S <= dur <= MAX_S):
                bad += 1
                continue
            dest = clip_dir / (Path(r["fname"]).stem + ".wav")
            sf.write(str(dest), pcm, 16000, subtype="PCM_16")
            rows.append({"audio": str(dest.resolve()), "text": text, "speaker": r.get("speaker_id"),
                         "seconds": round(dur, 2)})
    return rows, bad


def prep_kathbath():
    src = ROOT / "data" / "kathbath_hi"
    out = {}
    for split, pattern in (("test", "valid-*.parquet"), ("train", "train-*.parquet")):
        shards = sorted(src.glob(pattern))
        if split == "train" and len(shards) < 32:
            raise RuntimeError(f"only {len(shards)} of 32 train shards downloaded")
        clip_dir = ROOT / "data" / "kathbath_hi_clips" / split
        clip_dir.mkdir(parents=True, exist_ok=True)
        rows, bad = [], 0
        with ProcessPoolExecutor(8) as ex:
            for r, b in ex.map(_kathbath_shard, [(str(s), str(clip_dir)) for s in shards]):
                rows += r
                bad += b
        write_jsonl(src / f"{split}.jsonl", rows)
        out[split] = (len(rows), bad, round(sum(r["seconds"] for r in rows) / 3600, 1),
                      len({r["speaker"] for r in rows}))
    tr = {r["speaker"] for r in map(json.loads, open(src / "train.jsonl", encoding="utf-8"))}
    te = {r["speaker"] for r in map(json.loads, open(src / "test.jsonl", encoding="utf-8"))}
    return (f"kathbath: " + "; ".join(f"{k} {n} clips ({b} skipped), {h} h, {s} speakers" for k, (n, b, h, s) in out.items())
            + f"; test speakers also in train: {len(te & tr)} of {len(te)}")


# ------------------------------------------------------------------ MUCS
def prep_mucs(split: str):
    import numpy as np
    import soundfile as sf
    sys.path.insert(0, str(ROOT))
    from audio_codec import read_wav
    base = ROOT / "data" / "mucs_hi_en"
    tar = base / f"Hindi-English_{split}.tar.gz"
    deadline = time.time() + 3 * 3600
    while not tar.exists():                                   # the train archive downloads slowly
        if time.time() > deadline:
            raise RuntimeError(f"{tar.name} never arrived")
        time.sleep(60)
    ex_dir = base / split
    if not (ex_dir / "transcripts" / "segments").exists():
        with tarfile.open(tar) as t:
            t.extractall(base, filter="data")
    tdir = ex_dir / "transcripts"
    text = dict(l.split(" ", 1) for l in (tdir / "text").read_text(encoding="utf-8").splitlines() if " " in l)
    scp = {k: v for k, v in (l.split(None, 1) for l in (tdir / "wav.scp").read_text(encoding="utf-8").splitlines() if l.strip())}
    segs = {}
    for l in (tdir / "segments").read_text(encoding="utf-8").splitlines():
        p = l.split()
        if len(p) == 4:
            segs.setdefault(p[1], []).append((p[0], float(p[2]), float(p[3])))
    clip_dir = base / f"{split}_clips"
    clip_dir.mkdir(exist_ok=True)
    rows, bad = [], 0
    for rec, items in segs.items():
        wav = ex_dir / Path(scp.get(rec, rec + ".wav").strip()).name
        if not wav.exists():
            bad += len(items)
            continue
        audio = read_wav(str(wav), 16000)
        for utt, s, e in items:
            t_ = (text.get(utt) or "").strip()
            clip = audio[int(s * 16000): int(e * 16000)]
            dur = len(clip) / 16000
            if not t_ or not (MIN_S <= dur <= MAX_S):
                bad += 1
                continue
            dest = clip_dir / f"{utt}.wav"
            sf.write(str(dest), np.clip(clip, -1, 1), 16000, subtype="PCM_16")
            rows.append({"audio": str(dest.resolve()), "text": t_.lower(), "seconds": round(dur, 2)})
    write_jsonl(base / f"{split}.jsonl", rows)
    return f"mucs {split}: {len(rows)} clips ({bad} skipped), {sum(r['seconds'] for r in rows) / 3600:.1f} h"


# ------------------------------------------------------------------ Monsoon
def prep_monsoon():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    sys.path.insert(0, str(ROOT))
    from audio_codec import resample
    base = ROOT / "data" / "monsoon_en_in"
    clip_dir = base / "clips"
    clip_dir.mkdir(exist_ok=True)
    rows = []
    for shard in sorted(base.glob("data__test-*.parquet")):
        for batch in pq.ParquetFile(str(shard)).iter_batches(batch_size=64, columns=["audio", "id", "text"]):
            for r in batch.to_pylist():                       # audio and text only, nothing personal
                data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if sr != 16000:
                    data = resample(data, sr, 16000)
                dest = clip_dir / f"{r['id']}.wav"
                sf.write(str(dest), np.clip(data, -1, 1), 16000, subtype="PCM_16")
                rows.append({"audio": str(dest.resolve()), "text": (r.get("text") or "").strip()})
    rows = [r for r in rows if r["text"]]
    write_jsonl(base / "test.jsonl", rows)
    return f"monsoon en-IN: {len(rows)} clips"


# ------------------------------------------------------------------ GPU
def gpu_busy() -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "ForEach-Object { $_.CommandLine }"], capture_output=True, text=True).stdout
    return any(k in out for k in ("decide.py", "finetune.py", "speech_encoder.py", "vision_encoder.py",
                                  "app_omni.py", "chat_v5.py", "queue_india.py"))


def benchmark_india():
    if not (ROOT / "ckpt_decide_lora_india_F.pt").exists():
        return "no Indian-banking adapter; skipped"
    import urllib.request
    api = subprocess.Popen([PY, "-u", str(ROOT / "api.py"), "--port", "8000"], cwd=str(ROOT),
                           stdout=open(LOGS / "api.log", "w"), stderr=subprocess.STDOUT,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        r = subprocess.run([PY, "-u", str(ROOT / "examples" / "api_demo" / "benchmark.py"), "--only", "decide_india"],
                           cwd=str(ROOT), capture_output=True, text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        return (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    finally:
        api.terminate()
        api.wait(30)


def main():
    global ROOT, LOGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    a = ap.parse_args()
    ROOT, LOGS = a.root.resolve(), a.root.resolve() / "logs" / "speech_v5"
    LOGS.mkdir(parents=True, exist_ok=True)
    if (LOGS / "done.marker").exists():
        return
    log(f"=== SPEECH V5 start (pid {os.getpid()}) ===")

    once("1_prep_monsoon", prep_monsoon)
    once("1_prep_mucs_test", lambda: prep_mucs("test"))
    # Kathbath: wait for the download (34 files), then decode
    for _ in range(180):
        if len(list((ROOT / "data" / "kathbath_hi").glob("train-*-of-00032.parquet"))) == 32:
            break
        time.sleep(30)
    once("2_prep_kathbath", prep_kathbath)
    once("2_prep_mucs_train", lambda: prep_mucs("train"))

    parts = []
    for p, name in ((ROOT / "data" / "kathbath_hi" / "train.jsonl", "kathbath"),
                    (ROOT / "data" / "mucs_hi_en" / "train.jsonl", "mucs"),
                    (ROOT / "data" / "hindi_mix" / "train.jsonl", "hindi_mix")):
        if p.exists():
            rows = [json.loads(l) for l in open(p, encoding="utf-8")]
            parts.append((name, rows))
    mix = [r for _, rows in parts for r in rows]
    random.Random(0).shuffle(mix)
    write_jsonl(ROOT / "data" / "hindi_mix5" / "train.jsonl", mix)
    log("hindi_mix5: " + ", ".join(f"{n} {len(r)}" for n, r in parts) + f" = {len(mix)} clips")

    log("waiting for the GPU")
    while gpu_busy():
        time.sleep(30)
    once("4_benchmark_india", benchmark_india)

    run("5_asr_hi", ["speech_encoder.py", "train", "--ckpt", "ckpt_asr_hi_v3.pt", "--prefix", "data/hindi_mix5/train",
                     "--out", "ckpt_asr_hi_v5.pt", "--epochs", "1", "--batch-size", "4",
                     "--eval-every", "10000", "--log-every", "2000"], 5.0)

    evals = [("fleurs", "data/fleurs_hi/devtest", "1000"), ("kathbath", "data/kathbath_hi/test", "800"),
             ("mucs", "data/mucs_hi_en/test", "800")]
    for tag, ck in (("v3", "ckpt_asr_hi_v3.pt"), ("v5", "ckpt_asr_hi_v5.pt")):
        for name, prefix, limit in evals:
            if (ROOT / ck).exists() and (ROOT / f"{prefix}.jsonl").exists():
                run(f"6_wer_{name}_{tag}", ["speech_encoder.py", "eval", "--ckpt", ck, "--prefix", prefix,
                                            "--limit", limit], 1.0)
    if (ROOT / "data" / "monsoon_en_in" / "test.jsonl").exists():
        run("6_wer_monsoon_en_v3", ["speech_encoder.py", "eval", "--ckpt", "ckpt_asr_en_v3.pt",
                                    "--prefix", "data/monsoon_en_in/test", "--limit", "3000"], 1.5)
    (LOGS / "done.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log("=== SPEECH V5 DONE ===")


if __name__ == "__main__":
    main()
