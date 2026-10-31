"""Draws Chapter 5's figures; the evidence ones from the lab's results.json.

    python lab.py && python make_figures.py
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from fig import Fig, P  # noqa: E402

OUT = HERE / "figures"
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))


def fig_rotation():
    """5.1: one pair of numbers, turned further at every position; fast pairs turn more."""
    f = Fig(720, 360)
    for k, (cx, title, step) in enumerate(((180, "a fast pair (pair 0)", 1.0), (500, "a slow pair", 0.12))):
        r = 92
        cy = 160
        f.text(cx, 36, title, 13, P.ink, "middle", 700)
        f.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{P.mist}" stroke-width="1.5"/>')
        f.line(cx - r - 10, cy, cx + r + 10, cy, P.mist, 1)
        f.line(cx, cy - r - 10, cx, cy + r + 10, P.mist, 1)
        base = 0.35
        for pos in range(4):
            a = base + pos * step
            x2, y2 = cx + r * math.cos(a), cy - r * math.sin(a)
            col = P.saffron if pos == 3 else P.indigo
            f.arrow(cx, cy, x2, y2, col, 2.0 if pos in (0, 3) else 1.4)
            lx, ly = cx + (r + 18) * math.cos(a), cy - (r + 18) * math.sin(a)
            f.text(lx, ly + 4, f"pos {pos}", 11, col, "middle", 600)
        f.text(cx, cy + r + 40, f"turns {step:g} rad per position" if k else "turns 1 rad per position", 12,
               P.stone, "middle")
    f.caption("Figure 5.1  RoPE treats each pair of numbers in a query or key as a 2-D vector and turns it by "
              "position × frequency. Pairs run from fast to slow, so together they encode position at many scales.")
    f.save(OUT / "fig5_1_rotation.svg")


def fig_relative():
    """5.2: q·k is the same function of distance wherever in the text you measure it."""
    rel = R["relative"]
    d = rel["distances"]
    cols = {"0": P.indigo, "300": P.saffron, "900": P.teal}
    f = Fig(720, 372)
    series = [("", list(zip(d, rel["by_start"][s])), cols[s]) for s in ("0", "300", "900")]
    X, Y = f.line_chart(90, 30, 580, 240, series, xlabel="distance between the two positions",
                        ylabel="q·k (all-ones q and k, per dimension)")
    f.text(X(140), Y(0.9), "measured from position 0, 300 and 900:", 12, P.ink, weight=600)
    f.text(X(140), Y(0.9) + 18, "three lines drawn, one visible; they coincide", 12, P.ink)
    f.caption(f"Figure 5.2  After rotation, q·k depends only on how far apart two tokens are. The three "
              f"curves differ by at most {rel['max_spread']:.0e}, which is float rounding. Measured in the lab.")
    f.save(OUT / "fig5_2_relative.svg")


def fig_drift():
    """5.3: angles drift when the frequencies are stored in bfloat16."""
    dr = {int(k): v for k, v in R["bf16_drift"].items()}
    theta = f"1e{round(math.log10(R['theta'])):d}" if R["theta"] >= 1e4 else f"{R['theta']:g}"
    pts = sorted(dr.items())
    f = Fig(720, 372)
    X, Y = f.line_chart(90, 30, 580, 240, [("", pts, P.rose)], xlabel="position in the text",
                        ylabel="largest angle error (radians)", ymin=0, ymax=3.2)
    for p in (200, 1000):
        f.parts.append(f'<circle cx="{X(p):.1f}" cy="{Y(dr[p]):.1f}" r="5" fill="{P.rose}"/>')
    # 200 sits in the corner by the axis: label it in the empty space above, with a leader
    f.line(X(200), Y(dr[200]) - 6, X(200), Y(1.6) + 6, P.rose, 1)
    f.text(X(200) + 6, Y(1.6), f"{dr[200]:.2f} rad at 200", 12, P.rose, weight=600)
    # 1000: below the rising line, which is higher to its right
    f.text(X(1000) + 12, Y(dr[1000]) + 18, f"{dr[1000]:.2f} rad at 1,000", 12, P.rose, weight=600)
    f.text(X(1000) + 12, Y(dr[1000]) + 34, "(the 400M model's context ends here)", 11, P.stone)
    f.caption(f"Figure 5.3  Angle error when model.to(bfloat16) also casts the rotary frequencies, with the "
              f"400M model's settings (head dim {R['dim']}, theta {theta}). Measured in the lab.")
    f.save(OUT / "fig5_3_bf16_drift.svg")


def fig_yarn():
    """5.4: which pairs complete a turn within the trained context, and what YaRN stretches."""
    y = R["yarn"]
    wl, st = y["wavelength"], y["stretch"]
    n = len(wl)
    f = Fig(720, 380)
    x0, y0, w, h = 90, 40, 560, 230
    lo, hi = math.floor(math.log10(min(wl))), math.log10(max(wl))      # floor: the shortest bar still shows
    Y = lambda v: y0 + h - (math.log10(v) - lo) / (hi - lo) * h
    bw = w / n
    for t in (1, 10, 100, 1000, 10000):
        if lo <= math.log10(t) <= hi:
            f.line(x0, Y(t), x0 + w, Y(t), P.mist, 1)
            f.text(x0 - 8, Y(t) + 4, f"{t:,}", 11, P.stone, "end")
    for i, (v, s) in enumerate(zip(wl, st)):
        col = P.indigo if s < 1.001 else (P.saffron if s > y["factor"] - 0.001 else P.saffron_soft)
        f.rect(x0 + i * bw + 1, Y(v), bw - 2, y0 + h - Y(v), col, r=1, sw=0)
    f.line(x0, Y(y["context"]), x0 + w, Y(y["context"]), P.rose, 1.6, dash="5 4")
    f.text(x0 + 6, Y(y["context"]) - 6, f"trained context: {y['context']} positions", 12, P.rose, weight=600)
    f.line(x0, y0 + h, x0 + w, y0 + h, P.ink, 1.2)
    f.text(x0 + w / 2, y0 + h + 22, "pair index, fastest (0) to slowest", 12, P.ink, "middle")
    f.parts.append(f'<text transform="translate({x0 - 52},{y0 + h / 2}) rotate(-90)" font-family="Inter, sans-serif" '
                   f'font-size="12" fill="{P.ink}" text-anchor="middle">wavelength (positions per turn)</text>')
    ly = y0 + h + 46
    for k, (col, label) in enumerate(((P.indigo, "kept"), (P.saffron_soft, "partly stretched (the ramp)"),
                                      (P.saffron, f"stretched {y['factor']:g}x"))):
        f.rect(x0 + k * 190, ly - 10, 14, 12, col, r=2, sw=0)
        f.text(x0 + k * 190 + 20, ly, label, 12, P.ink)
    f.caption(f"Figure 5.4  YaRN at {y['factor']:g}x: pairs that turn many times within the trained context keep "
              f"their frequency; pairs that never complete a turn there are slowed {y['factor']:g}x, so longer "
              f"text maps onto angles the model has seen. Theta {y['theta']:g}, {2 * n} dimensions.")
    f.save(OUT / "fig5_4_yarn.svg")


if __name__ == "__main__":
    for fn in (fig_rotation, fig_relative, fig_drift, fig_yarn):
        fn()
    print("figures written to", OUT)
