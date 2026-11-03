"""Launch the fixed text-only overnight pipeline with bounded checkpoint resume."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # experiments/archive/ -> the repository
DATA = ROOT / "data"          # held-out prompts also in training are dropped by omni_train
WORK_DIR = ROOT / "runs" / "overnight-text-v1"
CONSOLE_LOG = WORK_DIR / "console.log"
LAUNCHER_STATUS = WORK_DIR / "launcher_status.json"
RESUME_CHECKPOINTS = (WORK_DIR / "pretrained.pt.last", WORK_DIR / "omni.pt.last")


def command(resume: bool = False) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "anulm.training.train_from_scratch",
        "--corpus", str(DATA / "ctx_mix.txt"),
        "--work-dir", str(WORK_DIR),
        "--vocab", "32768",
        "--preset", "350m",
        "--block-size", "1024",
        "--batch-size", "4",
        "--grad-accum", "4",
        "--pretrain-steps", "12000",
        "--warmup", "500",
        "--eval-every", "250",
        "--eval-iters", "20",
        "--test-iters", "50",
        "--log-every", "25",
        "--grad-ckpt",
        "--moe-impl", "grouped",
        "--device", "cuda",
        "--report-gpu-memory",
        "--stage-timeout", "43200",
        "--min-free-gib", "12",
        "--task", f"chat=text:{DATA / 'chat_train.jsonl'}:25:8",
        "--task", f"read=text:{DATA / 'rc_train.jsonl'}:75:4",
        "--held", f"chat={DATA / 'chat_heldout.jsonl'}",
        "--held", f"read={DATA / 'rc_heldout.jsonl'}",
        "--omni-steps", "2000",
        "--omni-accum", "8",
        "--omni-warmup", "100",
        "--omni-eval-every", "250",
        "--omni-eval-n", "200",
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def command_text(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd) if os.name == "nt" else shlex.join(cmd)


def write_status(status: str, attempt: int, cmd: list[str], returncode: int | None = None) -> None:
    value = {
        "status": status,
        "attempt": attempt,
        "max_attempts": 3,
        "command": cmd,
        "returncode": returncode,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = LAUNCHER_STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, LAUNCHER_STATUS)


def run_attempt(cmd: list[str], log) -> int:
    line = f"\n$ {command_text(cmd)}\n"
    print(line, end="", flush=True)
    log.write(line)
    log.flush()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as error:
        line = f"launcher could not start pipeline: {error}\n"
        print(line, end="", file=sys.stderr, flush=True)
        log.write(line)
        log.flush()
        return 127
    assert process.stdout is not None
    for output in process.stdout:
        print(output, end="", flush=True)
        log.write(output)
        log.flush()
    return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the initial command without reading inputs or creating files")
    parser.add_argument("--resume-at-start", action="store_true",
                        help="add --resume to the first attempt for manual recovery")
    args = parser.parse_args()

    initial = command(args.resume_at_start)
    if args.dry_run:
        print(command_text(initial))
        return 0

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    returncode = 1
    with CONSOLE_LOG.open("a", encoding="utf-8") as log:
        for attempt in range(1, 4):
            resume = args.resume_at_start or attempt > 1
            cmd = command(resume)
            write_status("running", attempt, cmd)
            returncode = run_attempt(cmd, log)
            if returncode == 0:
                write_status("completed", attempt, cmd, returncode)
                return 0
            if attempt == 3 or not any(path.is_file() for path in RESUME_CHECKPOINTS):
                break
            message = f"pipeline failed with exit code {returncode}; retrying with --resume\n"
            print(message, end="", flush=True)
            log.write(message)
            log.flush()
            write_status("retrying", attempt, cmd, returncode)

        write_status("failed", attempt, cmd, returncode)
    return returncode or 1


if __name__ == "__main__":
    sys.exit(main())
