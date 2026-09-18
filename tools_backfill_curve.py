"""Backfill curve points whose log lines were lost, from values recorded elsewhere.

save_curve() keeps rows already in the CSV, so these persist once written.

  5,000 / 10,000   the phase-1 table in docs/CODER_PLAN.md (log overwritten when
                   coder_train_phase1b.log was created by a Start-Process redirect)
  105,000          read off ckpt_coder.pt.last at the time of the Ctrl+C kill
  265,000          read off ckpt_coder.pt after the truncated-resume incident

255,000 and 260,000 are genuinely gone: they were written only to the log of the
run that was killed at 265,000, and nothing else recorded them.

bits/byte = val / ln(2) / bytes_per_token, with bytes_per_token = 4.930 solved
from an intact row (step 525,000: 2.9214 -> 0.855).
"""
import csv
import math
from pathlib import Path

CURVE = Path(r"C:\workspace\AnuLM\coder_curve.csv")
BPT = 4.930
RECOVERED = {5000: 4.586, 10000: 4.048, 105000: 3.4436, 265000: 3.2617}

rows = {}
with CURVE.open(encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        rows[int(r["step"])] = (r["val_loss"], r["bits_per_byte"])

# sanity-check the constant against a row that is definitely intact
v, b = rows[525000]
solved = float(v) / math.log(2) / float(b)
assert abs(solved - BPT) < 0.01, f"bytes_per_token looks wrong: {solved:.3f}"
print(f"bytes_per_token check: {solved:.3f}")

added = []
for step, val in RECOVERED.items():
    if step in rows:
        continue
    rows[step] = (f"{val:.4f}", f"{val / math.log(2) / BPT:.3f}")
    added.append(step)

tmp = CURVE.with_suffix(".csv.tmp")
with tmp.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "val_loss", "bits_per_byte"])
    for step in sorted(rows):
        w.writerow([step, *rows[step]])
tmp.replace(CURVE)
print("backfilled:", added or "nothing", "-> total", len(rows), "points")
