"""
Train AnuLM-Decide in the GPU window after the demo branch, then resume pretraining.

    python experiments/decide_branch.py        # launched detached

Waits for the demo branch (experiments/demo_branch.py) to finish. That script's
last act restarts the main run from ckpt_base_v2.pt.last; this one pauses it
again at once -- it has done no work yet, and it will restart from the same
resume point -- trains the decision model on the decayed demo base, evaluates
it, and ALWAYS re-enables and restarts anulm_v2. ~30 min of GPU.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs" / "decide"
PY = sys.executable
BRANCH_LOG = ROOT / "logs" / "demo_branch" / "pipeline.log"


def log(m):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOGS / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout


def run(name, args, cap_h):
    lf = LOGS / f"{name}.log"
    log(f"--- {name} (cap {cap_h} h)")
    with open(lf, "w", encoding="utf-8") as f:
        p = subprocess.Popen([PY, "-u"] + args, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=cap_h * 3600)
        except subprocess.TimeoutExpired:
            p.terminate()
            p.wait(120)
            rc = "cap reached"
    for l in lf.read_text(encoding="utf-8", errors="replace").splitlines():
        if any(k in l for k in ("eval @", "done in", "accuracy", "answers", "latency", "Traceback", "Error")):
            log("    " + l.strip()[:220])
    log(f"--- {name}: exit {rc}")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    log("waiting for the demo branch to finish")
    while "DEMO BRANCH DONE" not in (BRANCH_LOG.read_text(encoding="utf-8", errors="replace")
                                     if BRANCH_LOG.exists() else ""):
        time.sleep(60)
    try:
        subprocess.run(["schtasks", "/change", "/tn", "anulm_v2", "/disable"], capture_output=True)
        time.sleep(30)
        ps("Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
           "Where-Object { $_.CommandLine -like '*train.py*' -and $_.CommandLine -like '*ckpt_base_v2*' } | "
           "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")
        time.sleep(30)
        log("main run paused again (it restarts from the same resume point)")
        if not (ROOT / "ckpt_demo_v2.pt").exists():
            log("ckpt_demo_v2.pt missing; nothing to train on")
            return
        run("1_train", ["decide.py", "train", "--ckpt", "ckpt_demo_v2.pt", "--out", "ckpt_decide.pt",
                        "--epochs", "3", "--batch-size", "8", "--lr", "3e-5", "--eval-every", "500"], 1.5)
        run("2_eval", ["decide.py", "eval", "--ckpt", "ckpt_decide.pt"], 0.5)
    finally:
        subprocess.run(["schtasks", "/change", "/tn", "anulm_v2", "/enable"], capture_output=True)
        subprocess.run(["schtasks", "/run", "/tn", "anulm_v2"], capture_output=True)
        log("=== main run re-enabled and restarted; DECIDE DONE ===")


if __name__ == "__main__":
    main()
