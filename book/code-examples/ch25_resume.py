"""
Book example, not repository code.

Chapter 25: kill a training run and bring it back, with train.py unchanged.

    python book/code-examples/ch25_resume.py          # CPU, under a minute
    python book/code-examples/ch25_resume.py --save   # also refresh book/measurements/part_v_b_resume_curves.json

Part 1 runs train.py three ways on the same small byte corpus, each in its
own temporary folder:

  A  60 steps without a break;
  B  the same command with --stop-at 30, then again with --resume;
  C  as B, but the resume point is damaged before the resume, so train.py
     takes its fallback: the best-val file, without optimizer moments or
     sampler position.

It then compares B's final resume point with A's, tensor by tensor, and
reads every logged training loss for Figure 25.3.

Part 2 simulates a kill in the middle of a checkpoint write: torch.save is
replaced by a writer that stops halfway, once for a plain in-place save and
once inside train.atomic_save.

Part 3 is the other kind of bad resume point: readable, but holding a Python
object the weights-only loader refuses (here an argparse.Namespace). train.py
must stop with UnsafeCheckpointError rather than fall back, and resume only
when ANULM_TRUST_PICKLE=1 says the file is trusted.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.model import safe_load  # noqa: E402
from anulm.training import train as T  # noqa: E402  the repository's train.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Results go to book/measurements/ only with --save, so a reader's rerun never overwrites them.
SAVE = "--save" in sys.argv[1:]
OUT = (ROOT / "book" / "measurements" if SAVE else Path.cwd()) / "part_v_b_resume_curves.json"
CMD = ["--preset", "tiny-gqa", "--steps", "60", "--batch-size", "8", "--block-size", "64",
       "--eval-every", "10", "--eval-iters", "4", "--log-every", "1", "--warmup", "10",
       "--lr", "1e-3", "--device", "cpu"]


def run(folder: Path, *extra: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "anulm.training.train", *CMD,
                           "--data", str(folder.parent / "corpus.txt"), "--out", str(folder / "ckpt.pt"), *extra],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONPATH": str(ROOT), **(env or {})})


def train(folder: Path, *extra: str) -> str:
    r = run(folder, *extra)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def losses(log: str) -> dict[int, float]:
    return {int(s): float(v) for s, v in re.findall(r"step\s+(\d+) \| loss (\d+\.\d+)", log)}


def same(a: dict, b: dict) -> bool:
    """Equal nested dicts/lists of tensors and numbers, bit for bit."""
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def part1(tmp: Path) -> dict:
    (tmp / "corpus.txt").write_text(T.FALLBACK * 120, encoding="utf-8")
    A, B, C = (tmp / n for n in "ABC")
    for d in (A, B, C):
        d.mkdir()
    log_a = train(A)
    log_b1 = train(B, "--stop-at", "30")
    log_b2 = train(B, "--resume")
    log_c1 = train(C, "--stop-at", "30")
    last = C / "ckpt.pt.last"
    last.write_bytes(last.read_bytes()[: last.stat().st_size // 2])      # a resume point cut in half
    log_c2 = train(C, "--resume")

    print("Part 1. train.py, 60 steps, stopped at step 30 and resumed")
    short = lambda line: line.replace(str(tmp), "<tmp>")
    for line in ([l for l in log_b1.splitlines()[-3:] if l.strip()][:1]
                 + [l for l in log_b2.splitlines() if l.startswith("resuming")]):
        print("  B> " + short(line))
    for line in log_c2.splitlines():
        if line.startswith("WARNING"):
            print("  C> " + short(line))
    a = safe_load(A / "ckpt.pt.last")
    b = safe_load(B / "ckpt.pt.last")
    c = safe_load(C / "ckpt.pt.last")
    for k in ("model", "opt", "sampler", "step", "best_val"):
        print(f"  A and B agree on {k:8s}: {same(a[k], b[k])}")
    print(f"  C, after the fallback, agrees with A on model: {same(a['model'], c['model'])}")
    la, lb, lc = losses(log_a), {**losses(log_b1), **losses(log_b2)}, {**losses(log_c1), **losses(log_c2)}
    print(f"  logged loss, steps 30-59 (the resumed half): A {sum(la[s] for s in range(30, 60)) / 30:.4f}  "
          f"B {sum(lb[s] for s in range(30, 60)) / 30:.4f}  C {sum(lc[s] for s in range(30, 60)) / 30:.4f}")
    print(f"  step 30: A {la[30]:.4f}  B {lb[30]:.4f}  C {lc[30]:.4f}")
    va = [float(v) for v in re.findall(r"val loss (\d+\.\d+)", log_a)]
    vc = [float(v) for v in re.findall(r"val loss (\d+\.\d+)", log_c1 + log_c2)]
    print(f"  final val loss: A {va[-1]:.4f}  C {vc[-1]:.4f}")
    return {"A": la, "B": lb, "C": lc, "val_A": va, "val_C": vc,
            "agree": {k: same(a[k], b[k]) for k in ("model", "opt", "sampler", "step", "best_val")}}


def part2(tmp: Path) -> dict:
    old, new = {"w": torch.zeros(200_000)}, {"w": torch.ones(200_000)}
    real_save = torch.save

    def killed_save(obj, path):                     # writes half the file, then the process "dies"
        real_save(obj, str(path) + ".full")
        data = Path(str(path) + ".full").read_bytes()
        os.remove(str(path) + ".full")
        Path(path).write_bytes(data[: len(data) // 2])
        raise KeyboardInterrupt("killed mid-write")

    print("\nPart 2. a kill in the middle of a checkpoint write")
    out = {}
    for how in ("torch.save in place", "train.atomic_save"):
        path = tmp / f"{how.split('.')[1].split()[0]}.pt"
        real_save(old, path)
        torch.save = killed_save
        try:
            (T.atomic_save if how.startswith("train") else torch.save)(new, path)
        except KeyboardInterrupt:
            pass
        finally:
            torch.save = real_save
        try:
            w = torch.load(path, weights_only=True)["w"]
            result = f"loads; holds the {'old' if w.sum() == 0 else 'new'} checkpoint"
        except Exception as e:
            result = f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"
        print(f"  {how:20s} -> {path.name} {result}")
        out[how] = result
    return out


def part3(tmp: Path) -> dict:
    import argparse
    D = tmp / "D"
    D.mkdir()
    train(D, "--stop-at", "30")
    last = D / "ckpt.pt.last"
    ck = safe_load(last)                                     # the real resume point loads safely
    ck["args"] = argparse.Namespace(lr=1e-3)                  # a Python object outside the allow-list
    torch.save(ck, last)

    print("\nPart 3. a readable resume point that holds a Python object")
    refused = run(D, "--resume")
    err = [l for l in refused.stderr.splitlines() if "UnsafeCheckpointError" in l]
    print(f"  --resume                      -> exit {refused.returncode}, "
          f"fell back: {'falling back' in refused.stdout}")
    print("    " + err[-1].replace(str(tmp), "<tmp>")[:150] + " ...")
    trusted = run(D, "--resume", env={"ANULM_TRUST_PICKLE": "1"})
    resumed = [l for l in trusted.stdout.splitlines() if l.startswith("resuming")]
    print(f"  --resume, ANULM_TRUST_PICKLE=1 -> exit {trusted.returncode}, "
          f"{resumed[0].split(' (')[0].replace(str(tmp), '<tmp>') if resumed else 'did not resume'}")
    assert refused.returncode != 0 and err and "falling back" not in refused.stdout
    assert trusted.returncode == 0 and resumed
    return {"refused_exit": refused.returncode, "trusted_exit": trusted.returncode}


def main():
    with tempfile.TemporaryDirectory() as d:
        r1 = part1(Path(d))
        r2 = part2(Path(d))
        r3 = part3(Path(d))
    OUT.write_text(json.dumps({"part1": r1, "part2": r2, "part3": r3}, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
