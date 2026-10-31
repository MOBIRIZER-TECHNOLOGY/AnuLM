"""Draws Chapter 2's figures; the evidence ones from the lab's results.json.

    python lab.py && python make_figures.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from fig import Fig, P  # noqa: E402

OUT = HERE / "figures"
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))


def fig_pipeline():
    """2.1: from files on disk to a corpus you can train on."""
    f = Fig(720, 250)
    steps = [("your files", "text, exports, dumps", P.teal_soft, P.teal),
             ("documents", "split on DOC_SEP", P.paper, P.ink),
             ("dedup", "first copy only", P.paper, P.ink),
             ("filter", "prose, script", P.paper, P.ink),
             ("split", "interleaved", P.paper, P.ink),
             ("decontaminate", "13-word windows", P.paper, P.ink)]
    x, w, gap = 12, 104, 12
    for i, (label, sub, fill, stroke) in enumerate(steps):
        f.box(x, 50, w, 56, label, fill, stroke, 12, sub=sub)
        if i < len(steps) - 1:
            f.arrow(x + w + 1, 78, x + w + gap - 1, 78, P.stone)
        x += w + gap
    f.box(150, 150, 420, 44, "and before any of it: may you train on this, and sell what you train?",
          P.saffron_soft, P.saffron, 12)
    f.caption("Figure 2.1  The pipeline. Each step has a function in course/build/data.py and a check that "
              "catches its silent failure.")
    f.save(OUT / "fig2_1_pipeline.svg")


def fig_split():
    s = R["split"]
    f = Fig(720, 230)
    rows = [("whole corpus", s["corpus"], P.stone), ("last 10% (the tail)", s["tail"], P.rose),
            ("interleaved 10%", s["interleaved"], P.teal)]
    f.text(30, 34, "share of Devanagari characters: Hindi articles first, then Shakespeare", 12, P.ink, weight=700)
    f.hbars(30, 54, 660, rows, vmax=1.0, label_w=170, bar_h=24, gap=12)
    f.caption("Figure 2.2  A validation set taken from the tail of a corpus built source by source contains "
              "one source only. Interleaving keeps the corpus's own mix. Measured in the lab.")
    f.save(OUT / "fig2_2_split.svg")


def fig_contamination():
    c = R["contamination"]
    f = Fig(720, 280)
    x0, base, h = 120, 190, 130
    groups = [("Hindi passages", "hindi"), ("English passages", "english")]
    for g, (label, key) in enumerate(groups):
        gx = x0 + g * 280
        f.text(gx + 90, base + 24, label, 12, P.ink, "middle", 600)
        for k, (who, col) in enumerate((("planted", P.stone_soft), ("yours", P.teal), ("repository", P.rose))):
            v = c["planted"][key] if who == "planted" else c[who][key]
            bx = gx + k * 64
            f.rect(bx, base - v / 5 * h, 48, max(v / 5 * h, 2), col, r=3, sw=0)
            f.text(bx + 24, base - v / 5 * h - 6, str(v), 12, P.ink, "middle", 700)
            f.text(bx + 24, base + 42, who, 11, P.stone, "middle")
    f.line(x0 - 20, base, x0 + 540, base, P.ink, 1.2)
    f.caption("Figure 2.3  Benchmark passages hidden in the training text, and how many each matcher found. "
              "The repository's word pattern sees only Latin letters and digits. Measured in the lab.")
    f.save(OUT / "fig2_3_contamination.svg")


if __name__ == "__main__":
    for fn in (fig_pipeline, fig_split, fig_contamination):
        fn()
    print("figures written to", OUT)
