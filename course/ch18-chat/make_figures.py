"""Draws Chapter 18's figures; the evidence one from the lab's results.json.

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


def fig_turns():
    """18.1: a two-turn conversation, and which parts are graded."""
    f = Fig(720, 230)
    parts = [("User: what is 23 plus 45?\\nAssistant:", "p", 272), (" 68", "a", 40), ("EOS", "a", 44),
             ("User: and plus 17?\\nAssistant:", "p", 232), (" 85", "a", 40), ("EOS", "a", 44)]
    x = 16
    for label, kind, w in parts:
        fill, stroke = (P.mist, P.stone) if kind == "p" else (P.saffron_soft, P.saffron)
        f.rect(x, 50, w, 36, fill, stroke, r=4, sw=1)
        f.text(x + w / 2, 73, label, 10.5, P.ink, "middle", 600, mono=True)
        x += w + 3
    f.text(16, 36, "one conversation, one row: the model reads all of it", 12, P.stone, weight=600)
    f.rect(16, 112, 14, 14, P.mist, P.stone, r=3, sw=1)
    f.text(36, 124, "the user's words and the template: read, never graded", 12, P.ink)
    f.rect(16, 138, 14, 14, P.saffron_soft, P.saffron, r=3, sw=1)
    f.text(36, 150, "every reply and its EOS: graded, in every turn", 12, P.ink)
    f.text(16, 184, "In a real chat each reply is the last one when it is written: grade them all.", 12, P.rose,
           weight=600)
    f.caption("Figure 18.1  Chapter 9's mask, over several turns.")
    f.save(OUT / "fig18_1_turns.svg")


def fig_arms():
    """18.2: first and second replies right, per grading choice."""
    f = Fig(720, 330)
    arms = list(R["arms"])
    x0, gw = 160, 170
    for k, name in enumerate(arms):
        r = R["arms"][name]
        x = x0 + k * gw
        for j, (key, col, lab) in enumerate((("first", P.indigo, "1st"), ("second", P.teal, "2nd"))):
            h = 170 * r[key]
            bx = x + j * 52
            f.rect(bx, 220 - h, 44, max(h, 1.5), col, r=3, sw=0)
            f.text(bx + 22, 214 - h, f"{100 * r[key]:.1f}%", 11, P.ink, "middle", 600)
            f.text(bx + 22, 238, lab, 11, P.stone, "middle")
        f.text(x + 48, 260, name, 12, P.ink, "middle", 700)
    f.text(16, 60, "held-out", 12, P.stone)
    f.text(16, 76, "conversations", 12, P.stone)
    f.text(16, 92, "answered right", 12, P.stone)
    f.caption("Figure 18.2  Chapter 9's model, taught two-turn conversations three ways. The second reply is read "
              "after the model's own first one. Measured in the lab.")
    f.save(OUT / "fig18_2_arms.svg")


if __name__ == "__main__":
    fig_turns()
    if R:
        fig_arms()
    print("figures written to", OUT)
