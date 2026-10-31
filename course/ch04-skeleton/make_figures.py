"""Draws Chapter 4's figures; the evidence one from the lab's results.

    python lab.py && python make_figures.py
"""

import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from fig import Fig, P  # noqa: E402

OUT = HERE / "figures"


def fig_skeleton():
    """4.1: what this chapter builds, and the gap Chapter 6 fills."""
    f = Fig(720, 470)
    cx, w = 250, 270
    f.box(cx - w / 2, 396, w, 40, "token ids", P.teal_soft, P.teal, 13)
    f.box(cx - w / 2, 340, w, 40, "embed_tokens", P.indigo_soft, P.indigo, 13, sub=None)
    f.rect(cx - w / 2 - 16, 120, w + 32, 204, "none", P.indigo, r=12, sw=1.4, dash="6 5")
    f.text(cx + w / 2 + 26, 226, "× n_layer", 14, P.indigo, weight=700)
    f.box(cx - w / 2, 266, w, 44, "attention", P.paper, P.stone, 13, sub="Chapter 6", color=P.stone,
          dash="5 4")
    f.box(cx - w / 2, 196, w, 36, "post_attention_layernorm", P.indigo_soft, P.indigo, 12)
    f.box(cx - w / 2, 136, w, 46, "mlp: SwiGLU", P.indigo_soft, P.indigo, 13, sub="Chapter 7 makes it experts")
    f.box(cx - w / 2, 70, w, 36, "norm", P.indigo_soft, P.indigo, 12)
    f.box(cx - w / 2, 18, w, 36, "lm_head → logits", P.indigo_soft, P.indigo, 12)
    for y1, y2 in ((396, 382), (340, 312), (266, 234), (196, 184), (136, 108), (70, 56)):
        f.arrow(cx, y1, cx, y2, P.stone)
    # the residual path
    f.line(cx - w / 2 - 34, 300, cx - w / 2 - 34, 128, P.saffron, 2.2)
    f.arrow(cx - w / 2 - 34, 128, cx - w / 2 - 34, 122, P.saffron)
    f.text(cx - w / 2 - 42, 214, "residual", 11, P.saffron, "end", 600)
    tx = 495
    f.text(tx, 40, "What you build in this chapter", 13, P.ink, weight=700)
    for k, line in enumerate(["The model's frame, with names", "matching model.py exactly.", "",
                              "Each block adds what its MLP", "makes of the normalised input", "back onto x (the residual).",
                              "", "Attention, the part that lets a", "position see earlier ones,", "is left out on purpose:",
                              "the lab measures what the", "frame can do without it."]):
        f.text(tx, 66 + 20 * k, line, 12, P.ink)
    f.caption("Figure 4.1  The skeleton: embeddings, n residual blocks of norm and SwiGLU, a final norm and the "
              "output head. Attention slots in at Chapter 6.")
    f.save(OUT / "fig4_1_skeleton.svg")


def fig_rmsnorm():
    """4.2: RMSNorm divides a vector by its root-mean-square."""
    x = [2.6, -1.2, 0.4, 3.8, -2.9, 0.9, -0.3, 1.7]
    rms = math.sqrt(sum(v * v for v in x) / len(x))
    y = [v / rms for v in x]
    f = Fig(720, 318)
    base = 150

    def bars(x0, vals, col, title, scale):
        f.text(x0 + 4 * 26, 34, title, 12, P.ink, "middle", 700)
        f.line(x0 - 6, base, x0 + 8 * 26, base, P.stone, 1)
        for i, v in enumerate(vals):
            h = v * scale
            f.rect(x0 + i * 26, base - max(h, 0), 18, abs(h), col, r=2, sw=0)
            # label at the bar's tip: above a bar that rises, below one that falls
            f.text(x0 + i * 26 + 9, base - h - 5 if v >= 0 else base - h + 12, f"{v:.1f}", 9, P.stone, "middle")

    bars(30, x, P.teal, "a vector x", 22)
    f.arrow(258, base - 30, 300, base - 30, P.stone)
    f.text(279, base - 40, f"÷ {rms:.2f}", 11, P.ink, "middle", 600)
    bars(316, y, P.indigo, "x / rms(x)", 22 * rms / 2.2)
    f.text(316 + 4 * 26, 262, "root-mean-square is now exactly 1", 11, P.stone, "middle")
    tx = 560
    f.text(tx, 60, "rms(x) =", 12, P.ink, weight=700)
    f.text(tx, 80, "√(mean(x²) + ε)", 13, P.ink, mono=True)
    f.text(tx, 110, f"here {rms:.2f}", 12, P.ink)
    f.text(tx, 140, "Then each of the D", 12, P.ink)
    f.text(tx, 158, "numbers is multiplied", 12, P.ink)
    f.text(tx, 176, "by its own learned", 12, P.ink)
    f.text(tx, 194, "gain (weight), which", 12, P.ink)
    f.text(tx, 212, "starts at 1.", 12, P.ink)
    f.caption(f"Figure 4.2  RMSNorm rescales a vector to a root-mean-square of 1; its direction is unchanged. "
              f"No mean is subtracted, which is what separates it from LayerNorm.")
    f.save(OUT / "fig4_2_rmsnorm.svg")


def fig_swiglu():
    """4.3: the gated MLP."""
    f = Fig(720, 330)
    f.text(30, 155, "x", 18, P.teal, weight=700, mono=True)
    f.text(30, 175, "D", 11, P.stone, mono=True)
    f.line(52, 150, 80, 150, P.stone)
    f.line(80, 90, 80, 210, P.stone)
    f.arrow(80, 90, 110, 90, P.stone)
    f.arrow(80, 210, 110, 210, P.stone)
    f.box(112, 68, 150, 44, "gate_proj", P.indigo_soft, P.indigo, 13, sub="D → I")
    f.box(112, 188, 150, 44, "up_proj", P.indigo_soft, P.indigo, 13, sub="D → I")
    f.arrow(262, 90, 300, 90, P.stone)
    f.box(302, 70, 80, 40, "SiLU", P.paper, P.ink, 13)
    f.arrow(382, 90, 430, 135, P.stone)
    f.arrow(262, 210, 430, 165, P.stone)
    f.rect(432, 128, 44, 44, P.paper, P.ink, r=22)
    f.text(454, 157, "×", 20, P.ink, "middle", 700)
    f.arrow(476, 150, 512, 150, P.stone)
    f.box(514, 128, 140, 44, "down_proj", P.indigo_soft, P.indigo, 13, sub="I → D")
    f.arrow(654, 150, 690, 150, P.stone)
    # SiLU, as a small labelled inset with its own axes
    ox, oy, sx, sy = 560, 196, 120, 44
    f.line(ox, oy + sy * 0.9, ox + sx, oy + sy * 0.9, P.stone, 1)          # g axis at SiLU = 0
    f.line(ox + sx / 2, oy - 4, ox + sx / 2, oy + sy + 4, P.stone, 1)       # SiLU axis at g = 0
    f.text(ox + sx + 6, oy + sy * 0.9 + 4, "g", 10, P.stone)
    f.text(ox + sx / 2 + 6, oy + 4, "SiLU(g)", 10, P.saffron, weight=600)
    pts = []
    for i in range(41):
        v = -4 + 8 * i / 40
        s = v / (1 + math.exp(-v))
        pts.append((ox + (v + 4) / 8 * sx, oy + sy - (s + 0.5) / 4.5 * sy))
    d = " ".join(f"{'M' if i == 0 else 'L'}{a:.1f},{b:.1f}" for i, (a, b) in enumerate(pts))
    f.parts.append(f'<path d="{d}" fill="none" stroke="{P.saffron}" stroke-width="2"/>')
    f.text(30, 262, "The gate branch decides, per number, how much of the up branch gets through:", 12, P.ink)
    f.text(30, 280, "SiLU(g) = g · sigmoid(g) is near 0 for negative g and near g for positive g.", 12, P.ink)
    f.caption("Figure 4.3  SwiGLU, the MLP in every AnuLM layer (and in every expert): two projections up, "
              "one gated by SiLU, multiplied, and one projection down. 3·D·I weights, no biases.")
    f.save(OUT / "fig4_3_swiglu.svg")


def fig_ceiling():
    """4.4: without attention, the skeleton stops at the bigram table."""
    R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(HERE / "curve.csv", encoding="utf-8")))
    pts = [(int(r["step"]), float(r["val_loss"])) for r in rows]
    f = Fig(720, 360)
    hl = [(R["bigram_loss"], f"Chapter 1's table of counts {R['bigram_loss']:.2f}", P.teal)]
    if R.get("ch1_transformer_loss"):
        hl.append((R["ch1_transformer_loss"], f"Chapter 1's transformer {R['ch1_transformer_loss']:.2f}",
                   P.indigo))
    f.line_chart(90, 30, 580, 250, [("", pts, P.saffron)], xlabel="training step",
                 ylabel="validation loss (nats per byte)", ymin=1.5, ymax=5.8, hlines=hl)
    f.text(420, 120, f"the skeleton: {R['params']:,} parameters,", 12, P.saffron, weight=600)
    f.text(420, 138, f"{R['layers']} blocks, no attention, ends at {R['final_loss']:.2f}", 12, P.saffron,
           weight=600)
    f.caption(f"Figure 4.4  Without attention every position sees only its own byte, so the skeleton learns "
              f"the bigram table by gradient descent and stops there. Measured in the lab ({R['seconds']} s).")
    f.save(OUT / "fig4_4_ceiling.svg")


if __name__ == "__main__":
    for fn in (fig_skeleton, fig_rmsnorm, fig_swiglu, fig_ceiling):
        fn()
    print("figures written to", OUT)
