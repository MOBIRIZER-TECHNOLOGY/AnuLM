"""
A/B probe for the chat fix: the release recipe, with and without --distill chat.

    python experiments/probe_distill.py            # both arms, one after the other
    python experiments/probe_distill.py distill    # one arm

Each arm is omni_one.py's own training command (same tasks, shares, LR,
accumulation, seed and 12,500-step schedule) stopped at step 625, the first
evaluation, written to its own checkpoint so ckpt_omni.pt is never touched.
The 2026-11-03 run's chat held-out loss had already risen 3.27 -> 3.34 by step
625, so one interval is enough to see whether distillation holds it.
Logs: logs/probe_distill/<arm>.log; the per-task losses are each arm's
<out>_curve.csv.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("omni_one", ROOT / "experiments" / "omni_one.py")
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)

ARMS = {"base": [], "distill": ["--distill", "chat", "--distill-alpha", "1.0"]}
STOP = 625


def command(arm: str) -> list[str]:
    out = f"ckpt_probe_nov_{arm}.pt"
    args = [sys.executable, "-u", "-m", "anulm.multimodal.omni_train", "--ckpt", "ckpt_chat_mm.pt",
            "--out", out, "--steps", str(O.STEPS), "--stop-after", str(STOP),
            "--accum", str(O.ACCUM), "--warmup", str(max(1, 500 // O.ACCUM)), "--lr", str(O.LR),
            "--eval-every", "625", "--eval-n", "200", "--log-every", "125", "--grad-ckpt"]
    if (ROOT / "ckpt_asr_en_v3.pt").exists():
        args += ["--audio-proj-from", "ckpt_asr_en_v3.pt"]
    for name, kind, path, w, b in O.TASKS:
        args += ["--task", f"{name}={kind}:{path}:{w}:{b}"]
    for name, path in O.HELD.items():
        args += ["--held", f"{name}={path}"]
    return args + ARMS[arm]


def main() -> None:
    arms = sys.argv[1:] or list(ARMS)
    logs = ROOT / "logs" / "probe_distill"
    logs.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        curve = ROOT / f"ckpt_probe_nov_{arm}_curve.csv"
        if curve.exists():
            print(f"{arm}: {curve.name} exists; delete it to rerun this arm", flush=True)
            continue
        print(f"=== {arm}: {' '.join(command(arm)[3:])}", flush=True)
        with open(logs / f"{arm}.log", "w", encoding="utf-8") as f:
            r = subprocess.run(command(arm), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        print(f"=== {arm}: exit {r.returncode}", flush=True)
        if r.returncode:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
