"""Draws Chapter 3's figures from the lab's results.json.

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


def show(t):
    return t.replace(" ", "␣")


def fig_split():
    """3.1: real words, as bytes and as your tokenizer's tokens."""
    f = Fig(720, 250)
    f.text(30, 34, "word", 11, P.stone, weight=600)
    f.text(170, 34, "bytes", 11, P.stone, weight=600)
    f.text(250, 34, f"your tokens ({R['vocab']}-id vocabulary)", 11, P.stone, weight=600)
    for k, (word, ex) in enumerate(R["examples"].items()):
        y = 52 + k * 52
        f.text(30, y + 18, show(word), 15, P.ink, weight=600)
        f.text(190, y + 18, str(ex["bytes"]), 15, P.stone, "middle", 600)
        x = 250
        for t in ex["tokens"]:
            x += f.chip(x, y, show(t), P.saffron_soft, P.saffron, 14) + 6
        f.text(x + 10, y + 18, f"{len(ex['tokens'])} token{'s' if len(ex['tokens']) > 1 else ''}", 12, P.ink)
    f.caption("Figure 3.1  Real words split by the tokenizer the lab trains. A common word becomes one token; "
              "the 13 bytes of भारत become one token too.")
    f.save(OUT / "fig3_1_split.svg")


def fig_window():
    """3.2: how much text fits in 1,024 tokens."""
    fe = R["fertility"]
    names = list(fe["English"])
    cols = [P.stone, P.saffron, P.indigo]
    f = Fig(720, 340)
    x0, base, h = 110, 236, 176
    top = max(fe[l][n]["chars_in_1024"] for l in fe for n in names)
    for g, lang in enumerate(("English", "Hindi")):
        gx = x0 + g * 300
        f.text(gx + 110, base + 24, lang, 13, P.ink, "middle", 700)
        for k, n in enumerate(names):
            v = fe[lang][n]["chars_in_1024"]
            bx = gx + k * 76
            f.rect(bx, base - v / top * h, 60, v / top * h, cols[k], r=3, sw=0)
            f.text(bx + 30, base - v / top * h - 6, f"{v:,}", 12, P.ink, "middle", 700)
    f.line(x0 - 20, base, x0 + 560, base, P.ink, 1.2)
    for k, n in enumerate(names):
        f.rect(x0 + k * 190, base + 40, 14, 12, cols[k], r=2, sw=0)
        f.text(x0 + k * 190 + 20, base + 50, n, 11, P.ink)
    f.text(30, 30, "characters of held-out text that fit in a 1,024-token window", 12, P.ink, weight=700)
    f.caption("Figure 3.2  The same window holds ten times more Hindi with the 400M model's tokenizer than with "
              "raw bytes. Measured in the lab.")
    f.save(OUT / "fig3_2_window.svg")


if __name__ == "__main__":
    for fn in (fig_split, fig_window):
        fn()
    print("figures written to", OUT)
