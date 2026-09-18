"""
Refresh the *Live status* block of TASKS.md, and the assistant's live-state
memory file, from what is on disk.

    python update_tasks.py                # once
    python update_tasks.py --loop 3600    # every hour, until killed

Two outputs, because the machine can lose power at any moment and both
readers matter. TASKS.md is for the human: what is running, which step,
what to type to resume. The memory file is for a *fresh assistant
session*, which otherwise starts knowing nothing -- it is written to the
project memory directory, so the next session reads the current step,
the resume command and the open decisions without re-deriving them.

Reads the step, val loss and best val from `ckpt_coder.pt.last` (and any
other `<name>.pt.last` given with --ckpt), whether a training process is
alive, the GPU's memory and utilisation, and the tail of the newest phase
log; writes them between the `<!-- live-status -->` markers; appends one
line to `tasks_history.log` so the curve survives even if TASKS.md is
edited. Pure stdlib apart from torch for the checkpoint read; safe to run
while training writes the checkpoint (it retries once after 30 s).
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE / "TASKS.md"
HISTORY = HERE / "tasks_history.log"
CURVE = HERE / "coder_curve.csv"          # the 700k-step coder run's record


def curve_path(ckpt_name: str) -> Path:
    """`ckpt_coder.pt` -> `coder_curve.csv`, `ckpt_ctx2k.pt` -> `ctx2k_curve.csv`.

    One file per run. They used to share `coder_curve.csv`, so the first
    refresh after a second run started wrote that run's step 500 into the
    middle of the coder's 138-point curve -- a record of a finished run that
    docs/MODEL_CARD.md and RESULTS.md 25 both cite.
    """
    stem = Path(ckpt_name).stem
    if stem.startswith("ckpt_"):
        stem = stem[len("ckpt_"):]
    return HERE / f"{stem}_curve.csv"
START, END = "<!-- live-status -->", "<!-- /live-status -->"

# The assistant's per-project memory directory. Written every refresh so a
# session started after a power cut inherits the live state instead of
# re-deriving it from checkpoints and logs.
MEMORY = (Path.home() / ".claude" / "projects" / "C--workspace-AnuLM"
          / "memory" / "live-training-state.md")
MEMORY_INDEX = MEMORY.parent / "MEMORY.md"
INDEX_LINE = ("- [Live training state](live-training-state.md) — rewritten hourly by "
              "update_tasks.py; current step, resume command, open decisions\n")


def _read(path: Path) -> dict | None:
    import torch  # noqa: local import so --help works without torch
    ck = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    return {"step": int(ck.get("step", -1)),
            "val": ck.get("val_loss", ck.get("best")),
            "best": ck.get("best_val", ck.get("best")),
            "saved": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}


def ckpt_state(path: Path) -> dict | None:
    """Read the resume point, falling back to the best-val checkpoint.

    This line is the canary for a failure mode that otherwise hides: a
    truncated `.last` makes every restart die in seconds while the
    scheduled task keeps firing, so the machinery looks healthy and the
    step silently stops moving. Reporting "unreadable" plus the last good
    step is what makes that visible -- a step that does not move between
    two hourly entries means the loop is spinning, not training.
    """
    if not path.exists():
        return None
    for attempt in range(2):
        try:
            return _read(path)
        except Exception as e:  # mid-write on the first try; truncated on the second
            if attempt == 0:
                time.sleep(30)
                continue
            best = Path(str(path)[:-5])          # drop ".last"
            note = f"{type(e).__name__}"
            if best.exists():
                try:
                    st = _read(best)
                    st["error"] = f"{path.name} UNREADABLE ({note}); showing {best.name}"
                    return st
                except Exception:
                    pass
            return {"error": f"{path.name} unreadable: {str(e)[:90]}"}
    return None


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def gpu() -> str:
    out = run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"])
    return out or "nvidia-smi unavailable"


def python_procs() -> int:
    if sys.platform != "win32":
        return run(["pgrep", "-c", "python"]).strip().isdigit() and int(run(["pgrep", "-c", "python"])) or 0
    out = run(["tasklist"])
    return sum(1 for l in out.splitlines() if l.lower().startswith("python.exe")) - 1  # minus this process


def newest_log() -> tuple[str, str]:
    # *_phase*.log is the coder run's shape; ctx_train.log is the
    # long-context run's. Take whichever training log is newest.
    logs = sorted(glob.glob(str(HERE / "*_phase*.log")) + glob.glob(str(HERE / "ctx_train.log")),
                  key=os.path.getmtime)
    if not logs:
        return "", ""
    p = logs[-1]
    lines = [l for l in Path(p).read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    return Path(p).name, "\n".join(lines[-4:])


def render(ckpts: list[str]) -> tuple[str, str]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    procs = python_procs()
    rows, hist, states = [], [], []
    for name in ckpts:
        st = ckpt_state(HERE / f"{name}.last")
        states.append((name, st))
        if st is None:
            rows.append(f"| `{name}` | not on disk | | | |")
            continue
        if "error" in st and "step" not in st:
            rows.append(f"| `{name}` | {st['error']} | | | |")
            continue
        v = f"{st['val']:.4f}" if isinstance(st["val"], float) else str(st["val"])
        b = f"{st['best']:.4f}" if isinstance(st["best"], float) else str(st["best"])
        note = f"  **{st['error']}**" if "error" in st else ""
        rows.append(f"| `{name}` | {st['step'] + 1:,}{note} | {v} | {b} | {st['saved']} |")
        hist.append(f"{now}\t{name}\tstep {st['step'] + 1}\tval {v}\tbest {b}\tsaved {st['saved']}\tprocs {procs}")
    logname, tail = newest_log()
    alive = "A training process is running" if procs > 0 else "**No training process running**"
    block = "\n".join([
        START,
        "## Live status",
        "",
        f"Written {now} by `update_tasks.py`. {alive}; GPU: {gpu()}.",
        "",
        "| checkpoint (`.last`) | step | val loss | best val | saved |",
        "| --- | --- | --- | --- | --- |",
        *rows,
        "",
        f"Tail of `{logname}`:" if logname else "No phase log found.",
        "",
        "```",
        tail,
        "```" if logname else "",
        END,
    ])
    return block, "\n".join(hist), states


def _bpb(rows: dict, val: float) -> str:
    """bits/byte for a point taken from a checkpoint rather than a log line.

    The log prints it directly; a checkpoint does not carry bytes_per_token,
    so solve it from any row that came from a log (val / ln2 / bits) and
    reuse that. Returns "" when no such row exists yet.
    """
    for v, b in rows.values():
        try:
            if float(b) > 0:
                return f"{val / math.log(2) / (float(v) / math.log(2) / float(b)):.3f}"
        except (ValueError, ZeroDivisionError):
            continue
    return ""


# The "+/- 0.0272" is the standard error train.py started printing with
# RESULTS.md 26. It is optional here so this still reads every log written
# before that, including the coder run's -- which is the whole reason the
# curve CSV exists.
EVAL_RE = re.compile(r"eval @\s*(\d+)\s*\|\s*val loss\s*([\d.]+)"
                     r"(?:\s*\+/-\s*[\d.]+)?\s*\|\s*bits/byte\s*([\d.]+)")


def save_curve(ckpts_state=None) -> int:
    """Mirror every eval point from the phase logs into a small CSV.

    The val-loss curve is the most valuable artefact of a 50-hour run and it
    exists in exactly one place: `coder_train_phase1.log`. That is a single
    file, appended to by a wrapper whose `>>` handle a crash can leave in any
    state, and this project has already lost two eval logs to a task that
    restarted and truncated them. A few KB of CSV, rebuilt from scratch on
    every refresh and written atomically, removes that single point of
    failure -- and unlike the log it survives the log being rotated, pruned
    or clobbered, because old rows already in the CSV are kept.
    """
    # Which training log belongs to which run. A checkpoint with no entry
    # here still gets a curve, built from the checkpoint alone.
    LOGS = {"ckpt_coder": "coder_train_phase*.log", "ckpt_ctx2k": "ctx_train.log"}
    total = 0
    for name, st in ckpts_state or []:
        total += _save_one(curve_path(name), [(name, st)],
                           log_glob=LOGS.get(Path(name).stem))
    return total


def _save_one(curve: Path, ckpts_state, log_glob: str | None) -> int:
    rows: dict[int, tuple[str, str]] = {}
    if curve.exists():                       # keep points whose log line is gone
        try:
            with curve.open(encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    rows[int(r["step"])] = (r["val_loss"], r["bits_per_byte"])
        except Exception:
            pass                             # a damaged CSV is rebuilt, not fatal
    # The checkpoint itself, always readable, so the curve keeps growing even
    # while the log cannot be opened (below). Refreshes run every 30 min and
    # evals every ~33 min, so this alone captures nearly every point.
    for _, st in ckpts_state or []:
        if st and "step" in st and isinstance(st.get("val"), float):
            rows.setdefault(st["step"] + 1, (f"{st['val']:.4f}", _bpb(rows, st["val"])))

    locked = []
    for log in (sorted(glob.glob(str(HERE / log_glob))) if log_glob else []):
        try:
            text = Path(log).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            # Windows: `cmd`'s `>>` takes an EXCLUSIVE handle, so while the
            # trainer's wrapper is appending, no other process can even open
            # the file. Swallowing this silently is what let the curve sit at
            # 103 points for 18 hours while the run reached step 700,000 --
            # the count looked stable, so nothing looked wrong. Say so.
            locked.append(f"{Path(log).name} ({type(e).__name__})")
            continue
        for m in EVAL_RE.finditer(text.replace("\r", "\n")):
            # +1 here, NOT at write time: the log prints the 0-indexed step.
            # Adding it on write instead shifts every row by one on each
            # refresh, so rows accumulate duplicates and the CSV doubles.
            rows[int(m.group(1)) + 1] = (m.group(2), m.group(3))
    if locked:
        print(f"  (log unreadable, using checkpoints only: {', '.join(locked)})", flush=True)
    if not rows:
        return 0
    try:
        tmp = curve.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "val_loss", "bits_per_byte"])
            for step in sorted(rows):
                w.writerow([step, *rows[step]])
        os.replace(tmp, curve)
    except Exception as e:
        print(f"  (curve write failed: {type(e).__name__}: {e})", flush=True)
    return len(rows)


def write_memory(rows_state: list[tuple[str, dict]], procs: int) -> None:
    """Rewrite the live-state memory file, and make sure MEMORY.md points at it.

    Everything here is re-derived from disk on every call, so it can never
    drift from reality the way a hand-written note does. Keep it short:
    it is loaded into a session's context, and the durable background
    (what the project is, which traps were hit) lives in the other memory
    files and in TASKS.md.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    alive = "training IS running" if procs else "**nothing is running**"
    lines = []
    for name, st in rows_state:
        if st and "step" in st:
            v = f"{st['val']:.4f}" if isinstance(st["val"], float) else st["val"]
            b = f"{st['best']:.4f}" if isinstance(st["best"], float) else st["best"]
            lines.append(f"- `{name}`: step **{st['step'] + 1:,}**, val {v}, best {b}, "
                         f"saved {st['saved']}" + (f" ({st['error']})" if "error" in st else ""))
        elif st:
            lines.append(f"- `{name}`: {st.get('error', 'unreadable')}")
        else:
            lines.append(f"- `{name}`: not on disk")
    body = f"""---
name: live-training-state
description: Live coder-run state, rewritten hourly by update_tasks.py — current step, how to resume, what is still open
metadata:
  type: project
---

**Rewritten {now} by `update_tasks.py`; {alive}.** Everything below is read
from disk each hour, so it is current as of that timestamp. If the machine
lost power, the step below is where to resume from.

{chr(10).join(lines)}

GPU: {gpu()}.

**To resume after any interruption** (power cut, reboot, crash): the
scheduled task `anulm_coder` fires every 30 min and restarts training
by itself, so usually do nothing. To force it:
`schtasks /run /tn anulm_coder`. To check it is really training and
not crash-looping, compare the step above against the previous hourly line
in `C:\\workspace\\AnuLM\\tasks_history.log` — a step that has not
moved means the loop is spinning.

**Target:** step 700,000 of the coder pretraining, ~0.4 s/step. Then
`EX_CAP=0 bash experiments/coder_sft.sh` (instruction tune + pass@1),
then `python serve.py --ckpt ckpt_coder_sft.pt` for the demo.

**Known at the last probe (step 250,000, tuned):** MBPP 5.1% (13/257),
HumanEval 0.0%. Val loss is noisy at the default `--eval-iters 20`
(+/-0.125), so only moves larger than ~0.03 mean anything.

Full detail: `C:\\workspace\\AnuLM\\TASKS.md` and
`docs/CODER_PLAN.md`. Related: [[training-state-2026-09-10]],
[[project-moved-from-transformers]].
"""
    try:
        MEMORY.parent.mkdir(parents=True, exist_ok=True)
        tmp = MEMORY.with_suffix(".md.tmp")
        tmp.write_text(body, encoding="utf-8", newline="\n")
        os.replace(tmp, MEMORY)
        if MEMORY_INDEX.exists():
            idx = MEMORY_INDEX.read_text(encoding="utf-8")
            if "live-training-state.md" not in idx:
                MEMORY_INDEX.write_text(idx.rstrip("\n") + "\n" + INDEX_LINE,
                                        encoding="utf-8", newline="\n")
        else:
            MEMORY_INDEX.write_text(INDEX_LINE, encoding="utf-8", newline="\n")
    except Exception as e:                      # never let this kill the loop
        print(f"  (memory write failed: {type(e).__name__}: {e})", flush=True)


def update(ckpts: list[str]) -> str:
    text = TASKS.read_text(encoding="utf-8")
    a, b = text.index(START), text.index(END) + len(END)
    block, hist, states = render(ckpts)
    write_memory(states, python_procs())
    n_curve = save_curve(states)
    # Atomic, because two writers can overlap: the long-running --loop process
    # and the every-30-min scheduled one-shot. Both produce near-identical
    # content, so last-writer-wins is fine -- a half-written file is not.
    tmp = TASKS.with_suffix(".md.tmp")
    tmp.write_text(text[:a] + block + text[b:], encoding="utf-8", newline="\n")
    os.replace(tmp, TASKS)
    if hist:
        with HISTORY.open("a", encoding="utf-8", newline="\n") as f:
            f.write(hist + "\n")
    return (hist or f"{datetime.now():%H:%M} no checkpoint on disk") + f"\tcurve {n_curve} pts"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="*", default=["ckpt_coder.pt", "ckpt_ctx2k.pt"],
                   help="checkpoint names whose .last to read; missing ones are skipped")
    p.add_argument("--loop", type=int, default=0, help="seconds between refreshes; 0 = once")
    a = p.parse_args()
    while True:
        print(update(a.ckpt), flush=True)
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
