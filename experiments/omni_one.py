"""
One model for every task, start to finish, unattended -- and safe to re-run.

    python experiments/omni_one.py                 # train, evaluate, export, report
    python experiments/omni_one.py --publish       # ... and upload if every gate passes

  1. data    omni_data.py (chat, reading, intents, English + Hindi speech) and
             omni_images.py (captions, VQA), each from its primary source; only
             sources that allow commercial use (docs/DATASETS.md, NOTICE)
  2. chat    the chat model: ckpt_chat_v3.pt if it is here, else the released
             base (toonist/AnuLM-Base-v2-400M) fine-tuned on data/chat_train.jsonl
             exactly as experiments/post_v2.py made chat v3
  3. grow    speech_vocab.py adds the audio and image ids -> ckpt_chat_mm.pt
  4. wait    for the GPU (no other process holding more than 1 GB of it)
  5. base    the text report card of the chat model it starts from (omni_eval.py)
  6. train   omni_train.py: one decoder on seven tasks at once -> ckpt_omni.pt;
             resumes from ckpt_omni.pt.last after any interruption
  7. eval    WER (LibriSpeech held-out, FLEURS hi dev+test, Kathbath, MUCS),
             reading F1, intent accuracy, chat held-out loss, VQA, captions
  8. report  logs/omni_one/REPORT.md: every number beside the model it
             replaces, and the release gates, pass or fail
  9. export  release/AnuLM-Omni-400M (export_hf.py --kind omni)
 10. publish only with --publish and only if every gate passed:
             toonist/AnuLM-Omni-400M on Hugging Face

**Re-running.** Every step records success in logs/omni_one/state.json and is
skipped next time; a failed step is not recorded, so a re-run retries it and
everything after it. Training resumes from its last eval (every 10,000 steps,
about 50 minutes). After a reboot, run the same command again -- that is the
whole recovery procedure. `experiments/run_omni_one.cmd` starts it detached.

**What is deliberately not in it.** Flickr30k and A-OKVQA's COCO photos (not
for commercial use), IndicTTS (terms unclear), the Indian-banking Decide set
(written with an AI assistant whose terms forbid training models on its
output without authorization), and the vision projector the Flickr30k models
learned -- the image projector starts fresh. The audio projector starts from
the English speech model's, which learned from LibriSpeech (CC BY 4.0) only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "omni_one"
OUT = "ckpt_omni.pt"
REPO = "toonist/AnuLM-Omni-400M"
KEYS = ("eval @", "done in", "held-out", "WER", "Traceback", "Error", "F1", "acc", "resumed", "task ")

STEPS = 100_000
# Share of steps per task (not of examples): Hindi speech is the largest and the
# weakest skill, chat the one that already works and must only be kept.
TASKS = [
    ("chat", "text", "data/chat_train.jsonl", 0.07, 8),
    ("read", "text", "data/rc_train.jsonl", 0.14, 4),
    ("intent", "text", "data/omni/intent_train.jsonl", 0.06, 8),
    ("asr_en", "asr", "data/speech_ls100", 0.12, 4),
    ("asr_hi", "asr", "data/omni/hindi_train", 0.32, 4),
    ("caption", "image", "data/omni/caption_train.jsonl", 0.17, 8),
    ("vqa", "image", "data/omni/vqa_train.jsonl", 0.12, 8),
]
HELD = {"chat": "data/chat_heldout.jsonl", "read": "data/rc_heldout.jsonl"}

# Release gates: the joint model may cost a little on each task against the
# separate model it replaces, not more. Baselines in the comments.
GATES = {
    "wer_en": ("<=", 9.0),                 # asr_en_v3: 7.1% on the same 100 clips
    "wer_hi_fleurs": ("<=", 40.0),         # asr_hi_v3: 36.5% on FLEURS dev+test (657)
    "chat_heldout_loss": ("<=", 3.30),     # chat_v3: 3.146 by omni_eval.py
    "read_squad_en": (">=", 30.0),         # generative; the pointer reader was F1 57.2
    "intent_banking77": (">=", 80.0),      # the Decide adapter: 89.0%
    "vqa_acc": (">=", 10.0),               # new data; no earlier model to compare
}


def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def state() -> dict:
    p = LOGS / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def mark(name, info=0):
    s = state()
    s[name] = {"ok": info, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (LOGS / "state.json.tmp").write_text(json.dumps(s, indent=2))
    os.replace(LOGS / "state.json.tmp", LOGS / "state.json")


class Failed(Exception):
    pass


def run(name, args, cap_h, done=None):
    """A subprocess step. Recorded only if it exits 0 (and `done()` agrees);
    anything else stops the pipeline so a re-run retries from here."""
    if name in state():
        log(f"--- {name}: done earlier, skipping")
        return
    lf = LOGS / f"{name}.log"
    log(f"--- {name} (cap {cap_h} h) -> {lf.name}")
    with open(lf, "a", encoding="utf-8") as f:
        p = subprocess.Popen([PY, "-u"] + args, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        try:
            rc = p.wait(timeout=cap_h * 3600)
        except subprocess.TimeoutExpired:
            p.terminate()
            p.wait(300)
            rc = "cap reached"
    for l in [l for l in lf.read_text(encoding="utf-8", errors="replace").splitlines()
              if any(k in l for k in KEYS)][-8:]:
        log("    " + l.strip()[:220])
    if rc != 0 or (done and not done()):
        log(f"--- {name}: FAILED (exit {rc}); stopping. Re-run the same command to retry from here.")
        raise Failed(name)
    log(f"--- {name}: ok")
    mark(name)


def once(name, fn):
    if name in state():
        log(f"--- {name}: done earlier, skipping")
        return
    log(f"--- {name}")
    try:
        out = fn()
    except Exception as e:
        log(f"--- {name}: FAILED ({type(e).__name__}: {e}); stopping.")
        raise Failed(name) from e
    log(f"    {out}")
    mark(name, out if isinstance(out, (int, float, str)) else 0)


def gpu_used_mb() -> int:
    """MiB in use on the card. Per-process figures read "[N/A]" under Windows'
    WDDM driver, so this is the whole card -- called only while the pipeline
    itself holds none of it."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return sum(int(l) for l in out.split() if l.strip().isdigit())


def wait_for(script: str):
    """Let another copy of `script` (started by hand) finish before running it."""
    def running():
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                              "ForEach-Object { $_.CommandLine }"], capture_output=True, text=True).stdout \
            if os.name == "nt" else subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
        return script in out
    while running():
        time.sleep(60)


def wait_gpu():
    t0 = time.time()
    while (mb := gpu_used_mb()) > 1024:
        if time.time() - t0 > 600 and int(time.time() - t0) % 1800 < 60:
            log(f"    still waiting for the GPU ({mb} MiB in use by others)")
        time.sleep(60)
    return f"GPU free after {(time.time() - t0) / 60:.0f} min"


def wer(name, ckpt, prefix, limit):
    """speech_encoder.py eval -> the WER it prints, kept in results."""
    lf = LOGS / f"7_{name}.log"
    if not (ROOT / f"{prefix}.jsonl").exists():
        return None
    r = subprocess.run([PY, "-u", "speech_encoder.py", "eval", "--ckpt", ckpt, "--prefix", prefix,
                        "--limit", str(limit)], cwd=str(ROOT), capture_output=True, text=True,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lf.write_text(r.stdout + r.stderr, encoding="utf-8")
    m = re.search(r"WER ([\d.]+)%\s+over (\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"no WER in {lf.name}")
    return float(m.group(1))


def results() -> dict:
    p = LOGS / "results.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_results(d):
    (LOGS / "results.json").write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate():
    res = results()
    for name, prefix, limit in (("wer_en", "data/speech_ls100", 100),
                                ("wer_hi_fleurs", "data/fleurs_hi/devtest", 1000),
                                ("wer_hi_kathbath", "data/kathbath_hi/test", 800),
                                ("wer_hi_mucs", "data/mucs_hi_en/test", 800)):
        if name not in res:
            res[name] = wer(name, OUT, prefix, limit)
            log(f"    {name}: {res[name]}%")
            save_results(res)
    if "chat_heldout_loss" not in res:
        r = subprocess.run([PY, "-u", "omni_eval.py", "--ckpt", OUT, "--vision", "--out",
                            str(LOGS / "eval_text_vision.json")], cwd=str(ROOT), capture_output=True,
                           text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        (LOGS / "7_eval_text_vision.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        if r.returncode:
            raise RuntimeError("omni_eval.py failed; see 7_eval_text_vision.log")
        res.update({k: v for k, v in json.loads((LOGS / "eval_text_vision.json").read_text(encoding="utf-8")).items()
                    if k not in ("ckpt", "seconds")})
        save_results(res)
    return {k: v for k, v in res.items() if not isinstance(v, list)}


def gates(res) -> dict:
    out = {}
    for k, (op, thr) in GATES.items():
        v = res.get(k)
        out[k] = v is not None and (v <= thr if op == "<=" else v >= thr)
    return out


def report():
    res = results()
    base = {}
    bp = LOGS / "eval_base_chat.json"
    if bp.exists():
        base = json.loads(bp.read_text(encoding="utf-8"))
    g = gates(res)
    earlier = {"wer_en": "7.1 (asr_en_v3)", "wer_hi_fleurs": "36.5 (asr_hi_v3)",
               "intent_banking77": "89.0 (Decide adapter)", "read_squad_en": "57.2 (pointer reader)"}
    lines = ["# AnuLM-Omni-400M: report", "",
             f"Written {time.strftime('%Y-%m-%d %H:%M')} by experiments/omni_one.py. One decoder, seven tasks.", "",
             "| measure | omni | chat model it started from | earlier separate model | gate | pass |",
             "|---|---|---|---|---|---|"]
    keys = ["chat_heldout_loss", "read_squad_en", "read_mlqa_hi", "intent_banking77", "intent_massive_hi",
            "wer_en", "wer_hi_fleurs", "wer_hi_kathbath", "wer_hi_mucs", "vqa_acc", "caption_heldout_loss"]
    for k in keys:
        gate = GATES.get(k)
        lines.append(f"| {k} | {res.get(k, '-')} | {base.get(k, '-')} | {earlier.get(k, '-')} | "
                     f"{f'{gate[0]} {gate[1]}' if gate else '-'} | {('yes' if g[k] else '**no**') if k in g else '-'} |")
    lines += ["", "WER in %, lower is better; F1, accuracy higher is better; losses lower is better.", ""]
    if res.get("caption_samples"):
        lines += ["## Captions, read them", ""]
        for s in res["caption_samples"]:
            lines += [f"- reference: {s['reference']}", f"  - model: {s['model']}"]
    lines += ["", f"Release gates: **{'all pass' if all(g.values()) else 'NOT all pass -- not published'}**"]
    (LOGS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "all gates pass" if all(g.values()) else f"gates failed: {[k for k, v in g.items() if not v]}"


def card() -> str:
    """The model card: docs/MODEL_CARD_OMNI.md plus this run's measured table."""
    body = (ROOT / "docs" / "MODEL_CARD_OMNI.md").read_text(encoding="utf-8")
    rep = (LOGS / "REPORT.md").read_text(encoding="utf-8")
    table = rep[rep.index("| measure"): rep.index("\nWER in %")]
    p = LOGS / "card.md"
    p.write_text(body.replace("<!-- MEASURED -->", table), encoding="utf-8")
    return str(p)


def publish():
    res = results()
    if not all(gates(res).values()):
        return "gates did not all pass: not published (see REPORT.md)"
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT / "release" / "AnuLM-Omni-400M"), repo_id=REPO,
                      repo_type="model", commit_message="AnuLM-Omni-400M: one model, seven tasks")
    return f"published https://huggingface.co/{REPO}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--publish", action="store_true", help="upload to the Hub if every release gate passes")
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    log(f"=== OMNI ONE (pid {os.getpid()}) ===")
    try:
        run("1_data_text", ["omni_data.py", "fetch", "all"], 12.0)
        wait_for("omni_images.py")
        run("1_data_images", ["omni_images.py", "fetch"], 12.0)
        if not (ROOT / "ckpt_chat_v3.pt").exists():
            run("2_base", ["-c", "from huggingface_hub import snapshot_download as s; "
                                 "s('toonist/AnuLM-Base-v2-400M', local_dir='release/AnuLM-Base-v2-400M')"], 2.0)
            run("2_chat", ["finetune.py", "--ckpt", "release/AnuLM-Base-v2-400M", "--qa", "data/chat_train.jsonl",
                           "--heldout", "data/chat_heldout.jsonl", "--out", "ckpt_chat_v3.pt", "--epochs", "2",
                           "--batch-size", "4", "--lr", "5e-5", "--grad-ckpt", "--eval-every", "500",
                           "--log-every", "100"], 3.0)
        run("3_grow", ["speech_vocab.py", "--ckpt", "ckpt_chat_v3.pt", "--out", "ckpt_chat_mm.pt"], 0.5,
            done=lambda: (ROOT / "ckpt_chat_mm.pt").exists())
        once("4_wait_gpu", wait_gpu)
        run("5_base_eval", ["omni_eval.py", "--ckpt", "ckpt_chat_v3.pt", "--out", str(LOGS / "eval_base_chat.json")], 2.0)
        args = ["omni_train.py", "--ckpt", "ckpt_chat_mm.pt", "--out", OUT, "--steps", str(a.steps),
                "--eval-every", "10000", "--eval-n", "200", "--log-every", "1000", "--resume",
                "--grad-ckpt"]     # unattended: an out-of-memory at 3 a.m. would stop the night
        if (ROOT / "ckpt_asr_en_v3.pt").exists():      # LibriSpeech-only projector; a clone starts fresh
            args += ["--audio-proj-from", "ckpt_asr_en_v3.pt"]
        for name, kind, path, w, b in TASKS:
            args += ["--task", f"{name}={kind}:{path}:{w}:{b}"]
        for name, path in HELD.items():
            args += ["--held", f"{name}={path}"]
        # A cap, not a deadline: reaching it leaves ckpt_omni.pt.last, and the
        # re-run resumes. The pipeline re-launches training up to three times.
        for attempt in range(3):
            try:
                run("6_train", args, 30.0, done=lambda: (ROOT / OUT).exists() and not (ROOT / (OUT + ".last")).exists())
                break
            except Failed:
                if attempt == 2 or not (ROOT / (OUT + ".last")).exists():
                    raise
                log("    resuming training from its last save")
        once("7_eval", evaluate)
        once("8_report", report)
        run("9_export", ["export_hf.py", OUT, "release/AnuLM-Omni-400M", "--kind", "omni", "--license", "cc-by-sa-4.0",
                         "--card", card(), "--base-model", "toonist/AnuLM-Base-v2-400M", "--repo", REPO], 0.5)
        if a.publish:
            once("10_publish", publish)
        log("=== OMNI ONE DONE: logs/omni_one/REPORT.md ===")
        (LOGS / "done.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    except Failed as e:
        log(f"=== stopped at {e}; fix the cause if there is one, then run the same command again ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
