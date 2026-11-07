"""
One model for every task, start to finish, unattended -- and safe to re-run.

    python experiments/omni_one.py                 # train, evaluate, export, report
    python experiments/omni_one.py --publish       # ... and upload if every gate passes

  1. data    omni_data.py (chat, reading, intents, English + Hindi speech) and
             omni_images.py (captions, VQA), each from its primary source; only
             sources that allow commercial use (docs/DATASETS.md, NOTICE)
  2. chat    the chat model: ckpt_chat_v3.pt if it is here, else the released
             base (toonist/AnuLM-Base-v2-400M) fine-tuned on data/chat_train.jsonl
             exactly as experiments/archive/post_v2.py made chat v3
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
 10. eval    the exported folder, repeating every release-gate measurement
 11. publish only with --publish and only if every exported-model gate passed:
             toonist/AnuLM-Omni-400M on Hugging Face

**Re-running.** Every step records success in logs/omni_one/state.json and is
skipped next time; a failed step is not recorded, so a re-run retries it and
everything after it. Training resumes from its last eval (every 625 steps of
8 batches, about 50 minutes). A content fingerprint of the training code,
command and data invalidates training and all downstream state when they change,
archiving generated outputs under logs/omni_one/archive rather than deleting them;
an unchanged interrupted run keeps its resume file. After a reboot, run the same
command again -- that is the whole recovery procedure.
`experiments/run_omni_one.cmd` starts it detached.

**What is deliberately not in it.** Flickr30k and A-OKVQA's COCO photos (not
for commercial use), IndicTTS (terms unclear), the Indian-banking Decide set
(written with an AI assistant whose terms forbid training models on its
output without authorization), and the vision projector the Flickr30k models
learned -- the image projector starts fresh. The audio projector starts from
the English speech model's, which learned from LibriSpeech (CC BY 4.0) only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs" / "omni_one"
OUT = "ckpt_omni.pt"
RELEASE = "release/AnuLM-Omni-400M"
REPO = "toonist/AnuLM-Omni-400M"
KEYS = ("eval @", "done in", "held-out", "WER", "Traceback", "Error", "F1", "acc", "resumed", "task ")

# Each optimizer step sums ACCUM batches. One batch per step (the first run,
# 100,000 steps) let the ~30-token intent and VQA batches each move the shared
# decoder a full AdamW step: chat held-out went 3.27 -> 5.39. Summing 8 kept
# every task better at equal batches (logs/ab_projscale, 2026-10-30).
ACCUM = 8
# The 2026-11-03 run took 12,500 steps (100,000 batches, 24.2 h) and passed
# every gate but chat: chat held-out rose from the first eval (3.27 -> 3.41 by
# step 2,500, 3.58 at 12,500) while the other tasks pulled the shared decoder.
# Distilling chat from its starting model did not slow that (probe at step 625:
# 3.358 vs 3.351, experiments/probe_distill.py). So: a quarter of the steps,
# with the learning rate fully decayed by the end, and never keep a
# checkpoint whose chat loss is more than MAX_RATIO times its first eval.
STEPS = 3_125                   # x ACCUM = 25,000 batches, about 6 hours
MAX_RATIO = "chat=1.049"        # 3.30 (the chat gate) / 3.146 (the chat model)
# 3,000-batch probes, chat held-out loss on the gate's metric (start 3.335):
# 15% chat at 5e-5 3.50, 25% at 5e-5 3.457, 15% at 2.5e-5 3.43. Both levers
# are used together here (not probed as a pair).
LR = 2.5e-5
# Share of steps per task (not of examples): Hindi speech is the largest and the
# weakest skill, chat the one that already works and must only be kept.
TASKS = [
    # 0.07 let the other tasks pull chat off course even with --accum; 0.25 with
    # half the learning rate (LR below) kept it closest to where it started.
    # chat_train_v6 = the chat model's 23,022 pairs + 13,163 human-written English
    # pairs (oasst2 later turns, Aya English; experiments/make_chat_v6.py): run b
    # read the old pairs twice over and its chat loss still rose to 3.309.
    ("chat", "text", "data/chat_train_v6.jsonl", 0.25, 8),
    ("read", "text", "data/rc_train.jsonl", 0.14, 4),
    # 0.06 (5%) reached BANKING77 87.7% in 12,500 steps but only 69.7% in
    # 3,125 (2026-11-05); batches are ~30 answer tokens, so 15% costs little time.
    ("intent", "text", "data/omni/intent_train.jsonl", 0.18, 8),
    ("asr_en", "asr", "data/speech_ls100", 0.12, 4),
    ("asr_hi", "asr", "data/omni/hindi_train", 0.26, 4),
    ("caption", "image", "data/omni/caption_train.jsonl", 0.13, 8),   # no gate
    ("vqa", "image", "data/omni/vqa_train.jsonl", 0.12, 8),
]
# --data v2 (run 2, 2026-11-06): the same shares on more data, from data/next/
# (experiments/make_omni_v2_data.py; terms in docs/DATASETS.md): LibriSpeech
# train-clean-360, VizWiz answerable questions, 60k more Localized Narratives
# photos, MLQA Hindi reading.
DATA_V2 = {"read": "data/rc_train_v2.jsonl", "asr_en": "data/speech_ls460",
           "caption": "data/omni/caption_train_v2.jsonl", "vqa": "data/omni/vqa_train_v2.jsonl"}
HELD = {"chat": "data/chat_heldout.jsonl", "read": "data/rc_heldout.jsonl"}

# Release gates: the joint model may cost a little on each task against the
# separate model it replaces, not more. Baselines in the comments.
GATES = {
    "wer_en": ("<=", 9.0),                 # asr_en_v3: 7.1% on the same 100 clips
    # Hindi: the real-speaker data (Kathbath, MUCS) trades FLEURS for everything
    # else -- speech_v5 on 2026-10-29: FLEURS 36.5 -> 39.6, Kathbath 37.0 -> 24.9,
    # MUCS 79.2 -> 55.6. So the gate is the mean of the three, no worse than
    # asr_hi_v3's 50.9, and FLEURS alone allowed that much slack and no more.
    "wer_hi_mean": ("<=", 50.9),
    "wer_hi_fleurs": ("<=", 43.0),
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
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(s):
    (LOGS / "state.json.tmp").write_text(json.dumps(s, indent=2), encoding="utf-8")
    os.replace(LOGS / "state.json.tmp", LOGS / "state.json")


def mark(name, info=0):
    s = state()
    s[name] = {"ok": info, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_state(s)


RECIPE_CODE = ("experiments/omni_one.py", "anulm/multimodal/omni_train.py",
               "anulm/model.py", "anulm/training/train.py", "anulm/training/finetune.py",
               "anulm/data/make_qa.py", "anulm/multimodal/speech_vocab.py",
               "anulm/multimodal/speech_encoder.py", "anulm/multimodal/vision_encoder.py",
               "anulm/multimodal/audio_codec.py", "anulm/hf_tok.py", "anulm/bpe.py",
               "anulm/hf/configuration_anulm.py")
RECIPE_STEPS = ("6_train", "7_eval", "8_report", "9_export", "10_export_eval", "10_publish", "11_publish")


def training_fingerprint(args) -> str:
    """Content-hash the code, command and source data that produce OUT."""
    h = hashlib.sha256(b"AnuLM omni training recipe v1\0")
    seen = set()

    def add(path, media_base=None):
        p = Path(path)
        if not p.is_absolute():
            p = ROOT / p
        key = str(p.resolve())
        if key in seen:
            return
        seen.add(key)
        try:
            label = p.relative_to(ROOT).as_posix()
        except ValueError:
            label = str(p)
        h.update(f"path\0{label}\0".encode())
        if not p.exists():
            h.update(b"missing\0")
            return
        if p.is_dir():
            for f in sorted(x for x in p.rglob("*") if x.is_file()):
                add(f)
            return
        h.update(b"file\0")
        with p.open("rb") as f:
            while chunk := f.read(8 << 20):
                h.update(chunk)
        h.update(b"\0")
        if p.suffix == ".jsonl":
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for kind in ("audio", "image"):
                        if row.get(kind):
                            add((media_base or p.parent) / row[kind])

    for arg in args:
        h.update(f"arg\0{arg}\0".encode())
    for path in RECIPE_CODE:
        add(path)
    add("ckpt_chat_mm.pt")
    for flag in ("--audio-proj-from", "--vision-proj-from"):
        if flag in args:
            add(args[args.index(flag) + 1])
    for _, kind, path, _, _ in TASKS:
        if kind == "asr":
            add(Path(path).with_suffix(".jsonl"), ROOT)
        else:
            add(path)
    for path in HELD.values():
        add(path)
    return h.hexdigest()


def archive_stale(previous):
    """Move generated outputs aside without overwriting an earlier archive."""
    curve = Path(OUT).with_name(Path(OUT).stem + "_curve.csv")
    generated = [ROOT / name for name in (OUT, OUT + ".last", OUT + ".tmp", OUT + ".last.tmp", curve)]
    generated += [LOGS / name for name in
                  ("results.json", "export_results.json", "eval_text_vision.json", "export_eval_text_vision.json",
                   "REPORT.md", "card.md", "done.marker")]
    existing = [p for p in generated if p.exists()]
    if not existing:
        return None
    name = re.sub(r"[^A-Za-z0-9_-]", "-", previous) if previous else time.strftime("%Y%m%d-%H%M%S")
    base = LOGS / "archive" / name
    archive, suffix = base, 2
    while True:
        try:
            archive.mkdir(parents=True)
            break
        except FileExistsError:
            archive = base.with_name(f"{base.name}-{suffix}")
            suffix += 1
    for p in existing:
        os.replace(p, archive / p.name)
    return archive


def prepare_training(args, retrain: bool = False):
    """Archive a changed recipe; an unchanged interrupted run keeps .last.

    A *finished* training is never thrown away silently: the fingerprint
    covers shared files (model.py, train.py, ...), so an unrelated edit or a
    branch merge would otherwise archive a day of GPU work and start again.
    That takes `--retrain`; without it the pipeline stops and says why."""
    fingerprint = training_fingerprint(args)
    s = state()
    previous = s.get("training_fingerprint")
    if previous == fingerprint:
        return
    if previous is not None and "6_train" in s and not retrain:
        raise Failed("6_train: the training code, command or data changed since "
                     f"{OUT} was trained. Run again with --retrain to archive it and train "
                     "anew, or restore what changed to keep using it")
    had_training = (previous is not None or any(k in s for k in RECIPE_STEPS)
                    or any((ROOT / p).exists() for p in (OUT, OUT + ".last")))
    archive = archive_stale(previous)
    for name in RECIPE_STEPS:
        s.pop(name, None)
    s["training_fingerprint"] = fingerprint
    save_state(s)
    log(f"--- training recipe {fingerprint[:12]}: " +
        ("changed; training and all downstream steps invalidated" if had_training else "recorded") +
        (f"; archived to {archive}" if archive else ""))


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


def wer(name, ckpt, prefix, limit, stage="7"):
    """speech_encoder.py eval -> the WER it prints, kept in results."""
    lf = LOGS / f"{stage}_{name}.log"
    if not (ROOT / f"{prefix}.jsonl").exists():
        return None
    r = subprocess.run([PY, "-u", "-m", "anulm.multimodal.speech_encoder", "eval", "--ckpt", ckpt, "--prefix", prefix,
                        "--limit", str(limit)], cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace",     # Hindi transcripts; Windows' default is cp1252
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    lf.write_text(r.stdout + r.stderr, encoding="utf-8")
    m = re.search(r"WER ([\d.]+)%\s+over (\d+)", r.stdout)
    if not m:
        raise RuntimeError(f"no WER in {lf.name}")
    return float(m.group(1))


def results(name="results.json") -> dict:
    p = LOGS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_results(d, name="results.json"):
    (LOGS / name).write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate(ckpt=OUT, exported=False):
    result_file = "export_results.json" if exported else "results.json"
    detail = "export_eval_text_vision" if exported else "eval_text_vision"
    stage = "10" if exported else "7"
    res = results(result_file)
    for name, prefix, limit in (("wer_en", "data/speech_ls100", 100),
                                ("wer_hi_fleurs", "data/fleurs_hi/devtest", 1000),
                                ("wer_hi_kathbath", "data/kathbath_hi/test", 800),
                                ("wer_hi_mucs", "data/mucs_hi_en/test", 800)):
        if name not in res:
            res[name] = wer(name, ckpt, prefix, limit, stage)
            log(f"    {name}: {res[name]}%")
            save_results(res, result_file)
    if "chat_heldout_loss" not in res:
        r = subprocess.run([PY, "-u", "-m", "anulm.multimodal.omni_eval", "--ckpt", ckpt, "--vision", "--out",
                            str(LOGS / f"{detail}.json")], cwd=str(ROOT), capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        (LOGS / f"{stage}_eval_text_vision.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        if r.returncode:
            raise RuntimeError(f"omni_eval.py failed; see {stage}_eval_text_vision.log")
        res.update({k: v for k, v in json.loads((LOGS / f"{detail}.json").read_text(encoding="utf-8")).items()
                    if k not in ("ckpt", "seconds")})
        save_results(res, result_file)
    return {k: v for k, v in res.items() if not isinstance(v, list)}


# The separate model each task used before, as measured then (also shown by app_omni.py).
EARLIER = {"wer_en": "7.1 (asr_en_v3)", "wer_hi_fleurs": "36.5 (asr_hi_v3); 39.6 (v5)",
           "wer_hi_kathbath": "37.0 (asr_hi_v3); 24.9 (v5)", "wer_hi_mucs": "79.2 (asr_hi_v3); 55.6 (v5)",
           "wer_hi_mean": "50.9 (asr_hi_v3); 40.0 (v5)",
           "intent_banking77": "89.0 (Decide adapter)", "read_squad_en": "57.2 (pointer reader)"}


def gates(res) -> dict:
    hi = [res.get(k) for k in ("wer_hi_fleurs", "wer_hi_kathbath", "wer_hi_mucs")]
    if all(v is not None for v in hi):
        res["wer_hi_mean"] = round(sum(hi) / 3, 1)
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
    earlier = EARLIER
    lines = ["# AnuLM-Omni-400M: report", "",
             f"Written {time.strftime('%Y-%m-%d %H:%M')} by experiments/omni_one.py. One decoder, seven tasks.", "",
             "| measure | omni | chat model it started from | earlier separate model | gate | pass |",
             "|---|---|---|---|---|---|"]
    keys = ["chat_heldout_loss", "read_squad_en", "read_mlqa_hi", "intent_banking77", "intent_massive_hi",
            "wer_en", "wer_hi_fleurs", "wer_hi_kathbath", "wer_hi_mucs", "wer_hi_mean", "vqa_acc",
            "caption_heldout_loss"]
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
    res = results("export_results.json")
    if not all(gates(res).values()):
        return "exported-model gates did not all pass: not published (see export_results.json)"
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT / RELEASE), repo_id=REPO,
                      repo_type="model", commit_message="AnuLM-Omni-400M: one model, seven tasks")
    return f"published https://huggingface.co/{REPO}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--publish", action="store_true", help="upload to the Hub if every release gate passes")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--retrain", action="store_true",
                    help="if the recipe changed after training finished, archive that model and train again")
    ap.add_argument("--data", choices=("v1", "v2"), default="v1", help="v2: add the data/next sets (DATA_V2)")
    a = ap.parse_args()
    if a.data == "v2":
        TASKS[:] = [(n, k, DATA_V2.get(n, p), w, b) for n, k, p, w, b in TASKS]
    LOGS.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    log(f"=== OMNI ONE (pid {os.getpid()}) ===")
    try:
        run("1_data_text", ["-m", "anulm.multimodal.omni_data", "fetch", "all"], 12.0)
        wait_for("omni_images")
        run("1_data_images", ["-m", "anulm.multimodal.omni_images", "fetch"], 12.0)
        if not (ROOT / "ckpt_chat_v3.pt").exists():
            run("2_base", ["-c", "from huggingface_hub import snapshot_download as s; "
                                 "s('toonist/AnuLM-Base-v2-400M', local_dir='release/AnuLM-Base-v2-400M')"], 2.0)
            run("2_chat", ["-m", "anulm.training.finetune", "--ckpt", "release/AnuLM-Base-v2-400M", "--qa", "data/chat_train.jsonl",
                           "--heldout", "data/chat_heldout.jsonl", "--out", "ckpt_chat_v3.pt", "--epochs", "2",
                           "--batch-size", "4", "--lr", "5e-5", "--grad-ckpt", "--eval-every", "500",
                           "--log-every", "100"], 3.0)
        run("3_grow", ["-m", "anulm.multimodal.speech_vocab", "--ckpt", "ckpt_chat_v3.pt", "--out", "ckpt_chat_mm.pt"], 0.5,
            done=lambda: (ROOT / "ckpt_chat_mm.pt").exists())
        once("4_wait_gpu", wait_gpu)
        run("5_base_eval", ["-m", "anulm.multimodal.omni_eval", "--ckpt", "ckpt_chat_v3.pt", "--out", str(LOGS / "eval_base_chat.json")], 2.0)
        args = ["-m", "anulm.multimodal.omni_train", "--ckpt", "ckpt_chat_mm.pt", "--out", OUT, "--steps", str(a.steps),
                "--accum", str(ACCUM), "--warmup", str(max(1, 500 // ACCUM)), "--lr", str(LR),
                "--eval-every", "625", "--eval-n", "200", "--log-every", "125", "--resume",
                "--grad-ckpt", "--max-ratio", MAX_RATIO]     # unattended: an out-of-memory at 3 a.m. would stop the night
        if (ROOT / "ckpt_asr_en_v3.pt").exists():      # LibriSpeech-only projector; a clone starts fresh
            args += ["--audio-proj-from", "ckpt_asr_en_v3.pt"]
        for name, kind, path, w, b in TASKS:
            args += ["--task", f"{name}={kind}:{path}:{w}:{b}"]
        for name, path in HELD.items():
            args += ["--held", f"{name}={path}"]
        prepare_training(args, retrain=a.retrain)
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
        run("9_export", ["-m", "anulm.hf.export_hf", OUT, RELEASE, "--kind", "omni", "--license", "cc-by-sa-4.0",
                         "--card", card(), "--base-model", "toonist/AnuLM-Base-v2-400M", "--repo", REPO], 0.5)
        once("10_export_eval", lambda: evaluate(RELEASE, exported=True))
        if a.publish:
            once("11_publish", publish)
        log("=== OMNI ONE DONE: logs/omni_one/REPORT.md ===")
        (LOGS / "done.marker").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    except Failed as e:
        log(f"=== stopped at {e}; fix the cause if there is one, then run the same command again ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
