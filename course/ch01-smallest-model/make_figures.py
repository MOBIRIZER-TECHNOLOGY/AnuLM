"""Draws Chapter 1's figures from the lab's own output (results.json, curve.csv).

    python lab.py && python make_figures.py
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from fig import Fig, P  # noqa: E402

R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
OUT = HERE / "figures"


def show(ch: str) -> str:
    return {" ": "␣", "\n": "↵"}.get(ch, ch)


def fig_next_byte():
    """1.1: the model's whole job -- a probability for every possible next byte."""
    f = Fig(720, 280)
    f.text(24, 34, "The text so far", 12, P.stone, weight=600)
    x = 24
    for ch in R["next_after"]:
        x += f.chip(x, 46, show(ch)) + 3
    f.box(150, 130, 200, 64, "the model", P.indigo_soft, P.indigo, 15,
          sub=f"{R['params'] / 1e6:.1f}M parameters")
    f.arrow(250, 76, 250, 128, P.stone)
    f.text(470, 112, "What comes next?", 12, P.stone, weight=600)
    rows = [(show(c), p, P.saffron if i == 0 else P.indigo_soft) for i, (c, p) in enumerate(R["next_top"][:4])]
    f.arrow(352, 162, 446, 162, P.stone)
    f.hbars(450, 124, 260, rows, vmax=1.0, label_w=30, bar_h=18, gap=8)
    f.caption("Figure 1.1  Given the bytes so far, the model returns a probability for each of the 256 "
              "possible next bytes. Measured on the lab's trained model.")
    f.save(OUT / "fig1_1_next_byte.svg")


def fig_bigram():
    """1.2: a table of counts already knows a lot about English."""
    letters = R["bigram_letters"]
    sub = R["bigram_sub"]
    n = len(letters)
    cell = 30
    f = Fig(720, 120 + n * cell)
    x0, y0 = 120, 70
    f.text(x0 + n * cell / 2, 32, "next byte", 12, P.stone, "middle", 600)
    f.text(30, y0 + n * cell / 2, "this byte", 12, P.stone, weight=600)
    hi = max(max(r) for r in sub)
    for j, c in enumerate(letters):
        f.text(x0 + j * cell + cell / 2, y0 - 10, show(c), 13, P.ink, "middle", 600, mono=True)
    for i, c in enumerate(letters):
        f.text(x0 - 12, y0 + i * cell + cell * 0.65, show(c), 13, P.ink, "end", 600, mono=True)
        for j in range(n):
            v = sub[i][j] / hi
            f.rect(x0 + j * cell + 1, y0 + i * cell + 1, cell - 2, cell - 2, P.teal, r=3, sw=0)
            f.parts[-1] = f.parts[-1].replace('fill="#1F9E89"', f'fill="#1F9E89" fill-opacity="{0.06 + 0.94 * v:.2f}"')
    tx = x0 + n * cell + 28
    f.text(tx, 90, "How to read it", 13, P.ink, weight=700)
    for k, line in enumerate([
            "Each row is a byte; each square,",
            "how often another byte follows it.",
            "Darker means more often.",
            "",
            "After a space (␣), t is likeliest:",
            "the, to, that. After e, a space:",
            "many words end in e.",
            "",
            f"Validation loss {R['bigram_val_loss']:.2f} nats/byte,",
            f"against {R['uniform_loss']:.2f} for a blind guess."]):
        f.text(tx, 114 + 20 * k, line, 12, P.ink)
    f.caption("Figure 1.2  The bigram model is nothing but this table of counts, normalised row by row. "
              "Counted on the lab's training text.")
    f.save(OUT / "fig1_2_bigram.svg")


def fig_transformer():
    """1.3: the tiny transformer, bottom to top."""
    f = Fig(720, 476)
    cx, w = 230, 260
    layers = [("bytes in", P.teal_soft, P.teal, "one number 0-255 per byte", 380),
              ("embedding + position", P.indigo_soft, P.indigo, "each byte becomes 128 numbers", 318),
              ("attention", P.indigo_soft, P.indigo, "look back at earlier bytes", 238),
              ("MLP", P.indigo_soft, P.indigo, "think about what it found", 186),
              ("output head", P.indigo_soft, P.indigo, "a score for every next byte", 100),
              ("probabilities out", P.saffron_soft, P.saffron, "softmax of the scores", 36)]
    for label, fill, stroke, sub, y in layers:
        f.box(cx - w / 2, y, w, 44, label, fill, stroke, 14, sub=sub)
    f.rect(cx - w / 2 - 18, 170, w + 36, 124, "none", P.indigo, r=12, sw=1.4, dash="6 5")
    f.text(cx + w / 2 + 30, 236, f"× {4} layers", 14, P.indigo, weight=700)
    for y1, y2 in ((380, 362), (318, 300), (238, 230), (186, 166), (100, 80)):
        f.arrow(cx, y1, cx, y2, P.stone)
    tx = 470
    f.text(tx, 60, "Sizes in the lab", 13, P.ink, weight=700)
    for k, line in enumerate([f"{R['params']:,} parameters", "128 numbers per byte (d)", "4 layers, 4 heads",
                              "sees 128 bytes at a time", "", "Attention is causal:",
                              "position t sees bytes 0..t only,", "so it cannot cheat by reading",
                              "the answer."]):
        f.text(tx, 84 + 20 * k, line, 12, P.ink)
    f.caption("Figure 1.3  The whole tiny transformer. Chapter 4 opens each block; here it is enough that "
              "each layer adds what it finds to a running representation.")
    f.save(OUT / "fig1_3_transformer.svg")


def fig_curve():
    """1.4: the transformer passes the table of counts within a few hundred steps."""
    rows = list(csv.DictReader(open(HERE / "curve.csv", encoding="utf-8")))
    pts = [(int(r["step"]), float(r["val_loss"])) for r in rows]
    f = Fig(720, 360)
    f.line_chart(90, 30, 580, 250, [("tiny transformer", pts, P.saffron)],
                 xlabel="training step", ylabel="validation loss (nats per byte)", ymin=1.0, ymax=5.8,
                 hlines=[(R["uniform_loss"], f"blind guess {R['uniform_loss']:.2f}", P.stone),
                         (R["bigram_val_loss"], f"bigram table {R['bigram_val_loss']:.2f}", P.teal)])
    f.caption(f"Figure 1.4  Validation loss while training on a laptop CPU ({R['seconds'] // 60} min). "
              f"Lower is better; it ends at {R['tiny_val_loss']:.2f}.")
    f.save(OUT / "fig1_4_curve.svg")


if __name__ == "__main__":
    for fn in (fig_next_byte, fig_bigram, fig_transformer, fig_curve):
        fn()
    print("figures written to", OUT)
