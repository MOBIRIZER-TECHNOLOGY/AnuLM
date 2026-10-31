"""Draws Chapter 7's figures; the evidence ones from the lab's results.json.

    python lab.py && python make_figures.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from fig import Fig, P  # noqa: E402

OUT = HERE / "figures"
RES = HERE / "results.json"
R = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else None
COL = {"dense": P.stone, "balanced": P.saffron, "unbalanced": P.rose}


def fig_moe():
    """7.1: a token goes to the shared expert and to the top 3 of 8 routed experts."""
    f = Fig(720, 420)
    f.box(30, 150, 90, 40, "token x", P.teal_soft, P.teal, 13)
    f.box(170, 150, 100, 40, "router", P.indigo_soft, P.indigo, 13)
    f.arrow(120, 170, 168, 170, P.stone)
    chosen = {1: 0.45, 4: 0.33, 6: 0.22}
    for e in range(8):
        y = 40 + e * 36
        on = e in chosen
        label = f"expert {e}  × {chosen[e] * 2.5:.2f}" if on else f"expert {e}"    # its weight, inside the box
        f.box(340, y, 120, 28, label, P.indigo_soft if on else P.paper, P.indigo if on else P.mist, 12,
              color=P.ink if on else P.stone)
        if on:
            f.arrow(270, 170, 338, y + 14, P.indigo, 1.4)
            f.arrow(460, y + 14, 548, 170, P.stone, 1.2)
    f.box(340, 346, 120, 28, "shared expert", P.indigo_soft, P.indigo, 12)      # below expert 7, clear of it
    f.line(145, 190, 145, 360, P.stone, 1.2)
    f.arrow(145, 360, 338, 360, P.stone, 1.2)
    f.arrow(460, 360, 552, 192, P.stone, 1.2)
    f.rect(550, 148, 44, 44, P.paper, P.ink, r=22)
    f.text(572, 177, "+", 22, P.ink, "middle", 700)
    f.arrow(594, 170, 640, 170, P.stone)
    f.text(648, 175, "y", 16, P.indigo, weight=700, mono=True)
    f.caption("Figure 7.1  A mixture-of-experts layer. The router picks 3 of 8 experts for this token and weights "
              "them (× weight); the shared expert runs for every token. Weights here are illustrative.")
    f.save(OUT / "fig7_1_moe.svg")


def fig_router():
    """7.2: scores choose with the bias added, but weight without it."""
    scores = [0.62, 0.71, 0.40, 0.55, 0.69, 0.33, 0.58, 0.47]
    bias = [0.00, -0.12, 0.05, 0.10, 0.02, 0.00, 0.06, -0.03]
    f = Fig(720, 340)
    x0, base, bw, sc = 60, 230, 34, 220
    f.text(x0, 34, "sigmoid score (solid) and score + balancing bias (outline), per expert", 12, P.ink, weight=700)
    biased = [s + b for s, b in zip(scores, bias)]
    top = sorted(range(8), key=lambda e: -biased[e])[:3]
    for e in range(8):
        x = x0 + e * (bw + 26)
        f.rect(x, base - scores[e] * sc, bw, scores[e] * sc, P.indigo if e in top else P.indigo_soft, r=2, sw=0)
        f.rect(x - 3, base - biased[e] * sc, bw + 6, 2.5, P.saffron, r=0, sw=0)
        f.text(x + bw / 2, base + 16, f"{e}", 12, P.ink, "middle", 600)
        if e in top:
            f.text(x + bw / 2, base - biased[e] * sc - 8, "picked", 10, P.saffron, "middle", 600)
    f.text(x0, base + 36, "expert", 11, P.stone)
    w = [scores[e] for e in top]
    tot = sum(w)
    tx = 548
    f.text(tx, 74, "Picked by score + bias:", 12, P.ink, weight=700)
    f.text(tx, 94, f"experts {', '.join(map(str, sorted(top)))}", 12, P.ink)
    f.text(tx, 124, "Weighted by score alone:", 12, P.ink, weight=700)
    for k, e in enumerate(sorted(top)):
        f.text(tx, 144 + 18 * k, f"expert {e}: {scores[e] / tot * 2.5:.2f}", 12, P.ink, mono=True)
    f.text(tx, 210, "(normalised, × 2.5)", 11, P.stone)
    best = max(range(8), key=lambda e: scores[e])             # say what the numbers show
    if best not in top:
        rank = sorted(scores, reverse=True).index(scores[best]) + 1
        for k, line in enumerate([f"Expert {best} has the highest" if rank == 1 else f"Expert {best} ranks {rank}",
                                  "score, but its bias (it has", "been overloaded) drops it", "out of the top 3."]):
            f.text(tx, 236 + 16 * k, line, 11, P.stone)
    f.caption("Figure 7.2  The router. The balancing bias (saffron) decides which experts are picked; the weights "
              "come from the unbiased scores. Illustrative numbers.")
    f.save(OUT / "fig7_2_router.svg")


def fig_imbalance():
    f = Fig(720, 372)
    series = [("", R[n]["imbalance"], COL[n]) for n in ("unbalanced", "balanced")]
    X, Y = f.line_chart(90, 30, 580, 250, series, xlabel="training step",
                        ylabel="worst layer: busiest expert ÷ average", ymin=1.0)
    for n, dy in (("unbalanced", -12), ("balanced", -12)):   # above each line, clear of the axis
        s, v = R[n]["imbalance"][-1]
        f.text(X(s) - 6, Y(v) + dy, f"{'no balancing' if n == 'unbalanced' else 'bias rule on'}: {v:.2f}x",
               12, COL[n], "end", 600)
    f.caption("Figure 7.3  Load imbalance while training (1.0 = every expert equally busy). Without the bias rule "
              "the router settles on favourites; with it, the load spreads. Measured in the lab.")
    f.save(OUT / "fig7_3_imbalance.svg")


def fig_share():
    f = Fig(720, 330)
    n = len(R["balanced"]["expert_share"])
    x0, base, h = 90, 250, 190
    top = max(max(R[k]["expert_share"]) for k in ("balanced", "unbalanced"))
    for e in range(n):
        for k, name in enumerate(("balanced", "unbalanced")):
            v = R[name]["expert_share"][e]
            x = x0 + e * 70 + k * 24
            f.rect(x, base - v / top * h, 22, v / top * h, COL[name], r=2, sw=0)
        f.text(x0 + e * 70 + 23, base + 16, f"{e}", 12, P.ink, "middle", 600)
    even = 1 / n
    ey = base - even / top * h
    f.line(x0 - 10, ey, x0 + n * 70 - 16, ey, P.ink, 1.2, dash="5 4")
    f.text(x0 + n * 70 - 8, ey - 2, "even:", 11, P.ink)                   # right of the bars, clear of them
    f.text(x0 + n * 70 - 8, ey + 12, f"{even:.1%}", 11, P.ink)
    f.text(x0, base + 36, "expert (first expert layer)", 11, P.stone)
    for k, (name, label) in enumerate((("balanced", "bias rule on"), ("unbalanced", "no balancing"))):
        f.rect(x0 + 300 + k * 160, base + 26, 14, 12, COL[name], r=2, sw=0)
        f.text(x0 + 320 + k * 160, base + 36, label, 11, P.ink)
    f.caption("Figure 7.4  Each expert's share of the routing choices on validation text, after training. "
              "Measured in the lab.")
    f.save(OUT / "fig7_4_share.svg")


def fig_curves():
    f = Fig(720, 372)
    series = [("", [p for p in R[n]["curve"] if p[0] >= 100], COL[n]) for n in ("dense", "unbalanced", "balanced")]
    X, Y = f.line_chart(90, 30, 580, 250, series, xlabel="training step",
                        ylabel="validation loss (nats per byte)", ymin=1.3, ymax=2.6)
    for k, n in enumerate(("dense", "balanced", "unbalanced")):
        f.text(X(700), Y(2.45) + 18 * k, f"{n}: {R[n]['final']:.3f} ({R[n]['total'] / 1e6:.2f}M parameters, "
                                         f"{R[n]['active'] / 1e6:.2f}M active)", 12, COL[n], weight=600)
    f.caption("Figure 7.5  The same attention, data and steps; the same active MLP width per token. Measured in "
              "the lab, starting from step 100.")
    f.save(OUT / "fig7_5_curves.svg")


if __name__ == "__main__":
    for fn in (fig_moe, fig_router):
        fn()
    if R:
        for fn in (fig_imbalance, fig_share, fig_curves):
            fn()
    print("figures written to", OUT)
