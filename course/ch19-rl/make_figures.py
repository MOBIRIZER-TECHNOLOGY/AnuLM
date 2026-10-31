"""Draws Chapter 19's figures; the evidence one from the lab's results.json.

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


def fig_group():
    """19.1: one prompt, a group of answers, rewards and advantages."""
    f = Fig(720, 330)
    f.box(16, 110, 150, 50, "what is 47 plus 38?", P.mist, P.stone, 12)
    answers = [("85", 1), ("85", 1), ("75", 0), ("85", 1), ("84", 0), ("85", 1), ("85", 1), ("95", 0)]
    mean = sum(r for _, r in answers) / len(answers)
    sd = (sum((r - mean) ** 2 for _, r in answers) / (len(answers) - 1)) ** 0.5
    f.text(206, 34, "8 samples", 11, P.stone, weight=600)
    f.text(300, 34, "checker", 11, P.stone, weight=600)
    f.text(410, 34, "advantage: (reward − group mean) / group spread", 11, P.stone, weight=600)
    for k, (a, r) in enumerate(answers):
        y = 46 + k * 28
        f.arrow(168, 135, 200, y + 10, P.mist, 1, 4)
        f.rect(206, y, 60, 22, P.paper, P.stone, r=4, sw=1)
        f.text(236, y + 16, a, 12, P.ink, "middle", 600, mono=True)
        f.text(316, y + 16, "1" if r else "0", 12, P.teal if r else P.rose, "middle", 700)
        adv = (r - mean) / sd
        w = 60 * abs(adv)
        x = 520 if adv > 0 else 520 - w
        f.rect(x, y + 3, w, 16, P.teal if adv > 0 else P.rose, r=3, sw=0)
        f.text(520 + w + 8 if adv > 0 else 520 - w - 8, y + 15, f"{adv:+.2f}", 11, P.ink,
               "start" if adv > 0 else "end")
    f.line(520, 44, 520, 272, P.ink, 1)
    f.caption("Figure 19.1  GRPO on one prompt. Right answers are pushed up, wrong ones down, by how they did "
              "against their own group. A group that is all right teaches nothing.")
    f.save(OUT / "fig19_1_group.svg")


def fig_curves():
    """19.2: sampled accuracy, and the share of groups that still had something to teach."""
    f = Fig(720, 420)
    cols = {"GRPO": P.teal, "rejection sampling": P.indigo}
    acc = [("", [(c[0], 100 * c[2]) for c in R["arms"][n]["curve"]], cols[n]) for n in R["arms"]]
    X, Y = f.line_chart(90, 30, 450, 140, acc, ylabel="sampled right (%)", ymin=88, ymax=100, yticks=[90, 95, 100])
    for n in R["arms"]:
        last = R["arms"][n]["curve"][-1]
        f.text(X(R["steps"]) + 10, Y(100 * last[2]) + (4 if n == "GRPO" else -6), f"{n} {100 * last[2]:.1f}%", 12,
               cols[n], weight=600)
    sig = [("", [(c[0], 100 * c[3]) for c in R["arms"][n]["curve"] if c[3] is not None], cols[n]) for n in R["arms"]]
    X2, Y2 = f.line_chart(90, 230, 450, 110, sig, xlabel="step", ylabel="groups with signal (%)", ymin=0, ymax=25,
                          yticks=[0, 10, 20])
    f.caption(f"Figure 19.2  Chapter 9's model, {R['prompts']} sums × {R['group']} samples a step. Top: 4 samples per "
              "held-out sum at temperature 1. Bottom: groups not all right or all wrong. Measured in the lab.")
    f.save(OUT / "fig19_2_curves.svg")


if __name__ == "__main__":
    fig_group()
    if R:
        fig_curves()
    print("figures written to", OUT)
