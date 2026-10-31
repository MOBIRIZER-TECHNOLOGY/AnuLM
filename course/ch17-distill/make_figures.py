"""Draws Chapter 17's figures; the evidence ones from the lab's results.json.

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
COLS = {"alone": P.stone, "bigram teacher": P.saffron, "bigram as a prior": P.rose, "model teacher": P.teal}


def fig_soft():
    """17.1: one hard target against a teacher's whole distribution."""
    f = Fig(720, 300)
    f.text(16, 34, "after \"thou art a\", the next byte is ...", 13, P.ink, weight=700)
    hard = [("n", 1.0), ("g", 0), ("b", 0), ("s", 0), ("f", 0), ("m", 0)]
    soft = [("n", 0.38), ("g", 0.17), ("b", 0.14), ("s", 0.12), ("f", 0.10), ("m", 0.09)]
    for col, (title, rows, colr) in enumerate((("the text says: \"n\" (and nothing else)", hard, P.stone),
                                              ("a teacher says: these, this likely", soft, P.teal))):
        x0 = 30 + col * 350
        f.text(x0, 66, title, 12, colr, weight=600)
        for k, (ch, v) in enumerate(rows):
            x = x0 + k * 50
            h = 120 * v
            f.rect(x, 214 - h, 34, max(h, 1.5), colr, r=3, sw=0)
            f.text(x + 17, 232, ch, 13, P.ink, "middle", 600, mono=True)
            if v:
                f.text(x + 17, 208 - h, f"{v:.0%}", 11, P.ink, "middle")
    f.text(30, 262, "Every position now teaches about every byte, not only the one that came next.", 12, P.ink)
    f.caption("Figure 17.1  Hard and soft targets. The numbers are illustrative; the lab measures what the "
              "soft ones are worth.")
    f.save(OUT / "fig17_1_soft.svg")


def curves(key, title, num, fname, ymax):
    f = Fig(720, 330)
    series = [("", R[key][n]["curve"], COLS[n]) for n in R[key]]
    X, Y = f.line_chart(90, 30, 430, 220, series, xlabel="training step", ylabel="validation loss",
                        ymin=1.6, ymax=ymax)
    ends = sorted((Y(R[key][n]["final"]), n) for n in R[key])
    placed = []
    for y, n in ends:
        y = max(y, placed[-1] + 15) if placed else y
        placed.append(y)
        f.text(X(R["steps"]) + 12, y + 4, f"{n} {R[key][n]['final']:.3f}", 12, COLS[n], weight=600)
    f.caption(f"Figure {num}  {title} Measured in the lab.")
    f.save(OUT / fname)


if __name__ == "__main__":
    fig_soft()
    if R:
        curves("arms", f"A student with {R['student_params']:,} parameters on all of Shakespeare. The teacher, "
                       f"{R['teacher_params']:,}, scores {R['teacher_loss']:.3f}.", "17.2", "fig17_2_full.svg", 2.7)
        curves("small", f"The same student on the first {R['small_bytes']:,} bytes only. Alone, it memorises them.",
               "17.3", "fig17_3_small.svg", 3.4)
    print("figures written to", OUT)
