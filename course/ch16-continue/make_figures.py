"""Draws Chapter 16's figures; the evidence ones from the lab's results.json.

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


def fig_mix():
    """16.1: one source after another, against a mix that is even along the file."""
    f = Fig(720, 250)
    cols = {"h": (P.saffron, "Hindi"), "e": (P.indigo, "English"), "p": (P.teal, "Python")}
    f.text(16, 34, "one after another: a run stopped at 60% has never seen Python", 12, P.rose, weight=600)
    x = 16
    for k, w in (("h", 270), ("e", 270), ("p", 140)):
        f.rect(x, 46, w, 30, cols[k][0], r=3, sw=0)
        f.text(x + 12, 66, cols[k][1], 12, "#FFFFFF", "start", 600)
        x += w + 2
    f.line(16 + 0.6 * 684, 40, 16 + 0.6 * 684, 84, P.ink, 1.4, dash="4 3")
    f.text(16 + 0.6 * 684 + 4, 96, "stopped here", 11, P.ink)
    f.text(16, 132, "mix_corpus.py: the next document from a source chosen by its remaining budget", 12, P.teal,
           weight=600)
    import random
    rnd = random.Random(3)
    x, left = 16, {"h": 270.0, "e": 270.0, "p": 140.0}
    while x < 700 and sum(left.values()) > 0:
        k = rnd.choices(list(left), weights=list(left.values()))[0]
        w = min(rnd.randint(5, 12), 700 - x)
        f.rect(x, 144, w, 30, cols[k][0], r=1, sw=0)
        left[k] = max(0.0, left[k] - w)
        x += w
    f.text(16, 196, "every 50th document of each source goes to a held-out file instead, and never into the mix",
           12, P.ink)
    f.caption("Figure 16.1  Mixing sources. Hindi 40%, English 40%, Python 20%, in both bars.")
    f.save(OUT / "fig16_1_mix.svg")


def fig_resume():
    """16.2: resuming a finished model at the peak learning rate, with and without rewarmup."""
    f = Fig(720, 330)
    series = [("", [(s, v) for s, v in R["resume"]["jump"]["curve"]], P.rose),
              ("", [(s, v) for s, v in R["resume"]["rewarm 100"]["curve"]], P.teal)]
    start = R["resume"]["jump"]["curve"][0][1]
    X, Y = f.line_chart(90, 30, 480, 220, series, xlabel="steps after resuming", ylabel="validation loss",
                        ymin=1.48, ymax=1.64, yticks=[1.5, 1.55, 1.6],
                        hlines=[(start, f"the finished model {start:.3f}", P.stone)])
    f.text(X(300) + 10, Y(R["resume"]["jump"]["final"]) + 14, "jump to the peak", 12, P.rose, weight=600)
    f.text(X(300) + 10, Y(R["resume"]["rewarm 100"]["final"]) - 4, "rewarm over 100", 12, P.teal, weight=600)
    f.caption("Figure 16.2  Continuing Chapter 8's model at its peak learning rate. Rewarmup softens the first "
              "steps; neither keeps what the final decay bought. Measured in the lab.")
    f.save(OUT / "fig16_2_resume.svg")


def fig_positions():
    """16.3: loss in each quarter of a window four times the trained length."""
    pos = R["by_position"]
    cols = {"plain RoPE": P.rose, "YaRN, no training": P.saffron, "trained, plain": P.stone, "trained, YaRN": P.teal}
    f = Fig(720, 340)
    series = [("", [(q + 1, v) for q, v in enumerate(vals)], cols[n]) for n, vals in pos.items()]
    X, Y = f.line_chart(90, 30, 430, 220, series, xlabel="quarter of the 512-byte window", ylabel="loss",
                        xticks=[1, 2, 3, 4], ymin=1.4, ymax=2.6)
    f.line(X(1.5), 30, X(1.5), 250, P.mist, 1)
    f.text(X(1) + 4, 44, "trained", 11, P.stone)
    f.text(X(1) + 4, 58, "length", 11, P.stone)
    ends = sorted((Y(v[-1]), n) for n, v in pos.items())
    placed = []
    for y, n in ends:
        y = max(y, placed[-1] + 15) if placed else y
        placed.append(y)
        f.text(X(4) + 12, y + 4, f"{n} {pos[n][-1]:.2f}", 12, cols[n], weight=600)
    f.caption("Figure 16.3  Chapter 8's model, trained on 128 bytes, reading 512. Each point is the mean loss over "
              "a quarter of the window. Measured in the lab.")
    f.save(OUT / "fig16_3_positions.svg")


if __name__ == "__main__":
    fig_mix()
    if R:
        fig_resume()
        fig_positions()
    print("figures written to", OUT)
