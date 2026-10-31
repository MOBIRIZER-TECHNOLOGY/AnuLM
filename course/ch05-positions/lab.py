"""
Chapter 5 lab: measure what rotary positions do, with model.py's own code.

    python lab.py                      # model.py's Rotary
    python lab.py --impl build         # yours, from course/build/anulm.py
    python lab.py --impl solution      # the reference solution

No training; a few seconds on a CPU. Three measurements, saved to results.json:

  1. relative:  q·k after rotation, as a function of the distance between two
                positions, measured starting at three different places in the text;
  2. precision: how far the angles drift when the rotary frequencies are held in
                bfloat16 (what happens if the whole model is cast with .to(bfloat16));
  3. YaRN:      which frequency pairs complete a turn within the trained context,
                and which ones YaRN stretches.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "course" / "checks"))
import model as ref  # noqa: E402
from _common import load, BUILD  # noqa: E402

CPU = torch.device("cpu")


def rotary_factory(impl):
    if impl == "model":
        return lambda dim, **kw: ref.Rotary(ref.AnuLMConfig(**kw), dim), ref.apply_rope
    M = load(BUILD / ("solutions/ch05.py" if impl == "solution" else "anulm.py"))
    return M.Rotary, M.apply_rope


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["model", "build", "solution"], default="model")
    p.add_argument("--dim", type=int, default=64, help="rotary dimensions per head (the 400M model: 64)")
    p.add_argument("--theta", type=float, default=1e6, help="rope_theta (the 30b preset and the 400M: 1e6)")
    a = p.parse_args()
    try:
        Rotary, apply_rope = rotary_factory(a.impl)
        Rotary(a.dim)
    except NotImplementedError as e:
        sys.exit(f"course/build/anulm.py is not finished ({e}). Run course/checks/ch05.py, "
                 f"or use --impl model / solution.")
    torch.manual_seed(0)
    out = {"dim": a.dim, "theta": a.theta, "impl": a.impl}

    # 1. relative: the same content rotated to every position; read q·k at three starting points
    n = 1200
    cos, sin = Rotary(a.dim, rope_theta=1e4)(n, CPU, torch.float32)
    q = torch.ones(1, 1, 1, a.dim)            # all-ones q and k: q·k(distance) = sum of cos(distance · freq)
    qq, kk = apply_rope(q.expand(1, 1, n, a.dim).clone(), q.expand(1, 1, n, a.dim).clone(), cos, sin)
    rel = {}
    for start in (0, 300, 900):
        rel[str(start)] = [round((qq[0, 0, start + d] @ kk[0, 0, start]).item() / a.dim, 5) for d in range(0, 257, 2)]
    spread = max(max(abs(x - y) for x, y in zip(rel["0"], rel[s])) for s in ("300", "900"))
    out["relative"] = {"distances": list(range(0, 257, 2)), "by_start": rel, "max_spread": spread}
    print(f"1. q·k depends only on distance: largest difference between start 0, 300, 900 = {spread:.2e}")

    # 2. precision: frequencies held in bfloat16, as after model.to(torch.bfloat16)
    good = Rotary(a.dim, rope_theta=a.theta)
    bad = Rotary(a.dim, rope_theta=a.theta).to(torch.bfloat16)
    positions = [50, 100, 200, 500, 1000, 2000, 4000]
    c1, s1 = good(4001, CPU, torch.float32)
    c2, s2 = bad(4001, CPU, torch.float32)
    err = (torch.atan2(s2, c2) - torch.atan2(s1, c1) + math.pi) % (2 * math.pi) - math.pi
    drift = {p_: round(err[p_].abs().max().item(), 4) for p_ in positions}
    out["bf16_drift"] = drift
    print("2. largest angle error (radians) with frequencies stored in bfloat16:")
    print("   " + "  ".join(f"pos {k}: {v:.3f}" for k, v in drift.items()))

    # 3. YaRN: wavelength of each pair vs the trained context, and what YaRN does to it
    ctx, factor = 256, 4.0
    plain = Rotary(a.dim, rope_theta=1e4).inv_freq
    yarned = Rotary(a.dim, rope_theta=1e4, yarn=True, yarn_factor=factor, yarn_original_context=ctx).inv_freq
    out["yarn"] = {"context": ctx, "factor": factor, "theta": 1e4,
                   "wavelength": [round(2 * math.pi / f, 2) for f in plain.tolist()],
                   "stretch": [round(x, 4) for x in (plain / yarned).tolist()],
                   "attn_mscale": Rotary(a.dim, rope_theta=1e4, yarn=True, yarn_factor=factor,
                                         yarn_original_context=ctx).attn_mscale}
    kept = sum(1 for r in out["yarn"]["stretch"] if r < 1.001)
    full = sum(1 for r in out["yarn"]["stretch"] if r > factor - 0.001)
    print(f"3. YaRN at {factor:g}x a {ctx}-position context: {kept} pairs kept, {full} stretched {factor:g}x, "
          f"{a.dim // 2 - kept - full} in between; attention temperature {out['yarn']['attn_mscale']:.4f} "
          f"(logits scaled by its square, {out['yarn']['attn_mscale'] ** 2:.4f})")
    (HERE / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
