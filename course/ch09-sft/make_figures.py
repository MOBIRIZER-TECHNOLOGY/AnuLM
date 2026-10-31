"""Draws Chapter 9's figures; the evidence ones from the lab's results.json.

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


def fig_row():
    """9.1: one packed row, what the model reads and what it is graded on."""
    f = Fig(720, 300)
    cells = ([("Q", "p")] * 1 + [("…", "p")] + [("?", "p"), ("\\n", "p"), ("A:", "p")]
             + [("␣1", "a"), ("5", "a"), ("7", "a"), ("EOS", "a")]
             + [("Q", "p"), ("…", "p"), ("A:", "p")] + [("␣6", "a"), ("8", "a"), ("EOS", "a")]
             + [("EOS", "x"), ("EOS", "x")])
    x0, w, gap = 20, 30, 3
    f.text(22, 34, "input row (block_size ids)", 12, P.stone, weight=600)
    f.text(22, 206, "target at each position: the NEXT id, or -1 (not graded)", 12, P.stone, weight=600)
    for i, (s, kind) in enumerate(cells):
        x = x0 + i * (w + gap)
        fill, stroke = {"p": (P.mist, P.stone), "a": (P.saffron_soft, P.saffron), "x": (P.stone_soft, P.stone)}[kind]
        f.rect(x, 46, w, 34, fill, stroke, r=4, sw=1)
        f.text(x + w / 2, 68, s, 11, P.ink, "middle", 600, mono=True)
        nxt = cells[i + 1] if i + 1 < len(cells) else ("EOS", "x")
        graded = nxt[1] == "a"
        f.rect(x, 150, w, 34, P.saffron_soft if graded else P.paper, P.saffron if graded else P.stone_soft, r=4, sw=1)
        f.text(x + w / 2, 172, nxt[0] if graded else "-1", 11, P.ink if graded else P.stone, "middle",
               600 if graded else 400, mono=True)
        if graded:
            f.arrow(x + w + gap / 2, 82, x + w / 2 + 2, 148, P.saffron, 1.1, 5)
    lx = 22
    for label, fill, stroke in (("prompt: read, never graded", P.mist, P.stone),
                                ("answer + EOS: graded", P.saffron_soft, P.saffron),
                                ("padding: EOS ids, never graded", P.stone_soft, P.stone)):
        f.rect(lx, 228, 14, 14, fill, stroke, r=3, sw=1)
        f.text(lx + 20, 240, label, 12, P.ink)
        lx += 20 + 7 * len(label) + 30
    f.caption("Figure 9.1  Two question–answer pairs packed into one row. Every position predicts the next id; "
              "only answer ids count. The last prompt id (“A:”) is graded on the answer's first id.")
    f.save(OUT / "fig9_1_row.svg")


ARMS = [("answer-only", P.saffron), ("whole text", P.indigo), ("no base", P.stone), ("with replay", P.teal)]


def fig_curves():
    """9.2: exact-match accuracy on held-out sums during training, per arm."""
    f = Fig(720, 340)
    series = [("", [(s, 100 * v) for s, v in R[name]["curve"]], col) for name, col in ARMS if name in R]
    X, Y = f.line_chart(90, 30, 470, 220, series, xlabel="fine-tuning step", ylabel="sums answered exactly (%)",
                        ymin=0, ymax=100, yticks=[0, 25, 50, 75, 100])
    last = {name: R[name]["curve"][-1] for name, _ in ARMS if name in R}
    ys = sorted(((Y(100 * v), name) for name, (s, v) in last.items()))
    placed = []
    for y, name in ys:                                       # labels at the right, never on top of each other
        y = max(y, placed[-1] + 18) if placed else y
        placed.append(y)
        col = dict(ARMS)[name]
        f.text(X(R["steps"]) + 12, y + 4, f"{name}  {100 * last[name][1]:.1f}%", 12, col, weight=600)
    f.caption(f"Figure 9.2  Same data, same steps, same settings. Accuracy on {R['test']} held-out sums, "
              "greedy decoding, every 250 steps. Measured in the lab.")
    f.save(OUT / "fig9_2_curves.svg")


def fig_forgetting():
    """9.3: what the base-started models still know of Shakespeare afterwards."""
    f = Fig(720, 250)
    before = R["answer-only"]["shakespeare_before"]
    rows = [("base, before", before, P.stone)]
    rows += [(name, R[name]["shakespeare_after"], col) for name, col in ARMS
             if name in R and R[name]["shakespeare_after"] is not None]
    f.text(22, 30, "validation loss on Shakespeare (lower: it still writes it)", 12, P.stone, weight=600)
    f.hbars(22, 46, 660, rows, label_w=150, fmt="{:.2f}")
    f.caption("Figure 9.3  The cost of a narrow fine-tune. Each model's loss on Shakespeare it never trained on, "
              "before and after 1,500 steps of sums. Measured in the lab.")
    f.save(OUT / "fig9_3_forgetting.svg")


if __name__ == "__main__":
    fig_row()
    if R:
        fig_curves()
        fig_forgetting()
    print("figures written to", OUT)
