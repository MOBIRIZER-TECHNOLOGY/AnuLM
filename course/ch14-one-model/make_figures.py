"""Draws Chapter 14's figures: two from the repository's own logs, one from the lab.

    python lab.py && python make_figures.py
"""

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "figures"))
from fig import Fig, P  # noqa: E402

OUT = HERE / "figures"
RES = HERE / "results.json"
R = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else None
AB = ROOT / "logs" / "ab_projscale"


def fig_hidden():
    """14.1: the first AnuLM-Omni run, as logged: the score improves, chat gets worse."""
    rows = list(csv.DictReader(open(ROOT / "ckpt_omni_curve.csv", encoding="utf-8")))
    score = [(int(r["step"]) / 1000, float(r["score"])) for r in rows]
    chat = [(int(r["step"]) / 1000, float(r["chat"])) for r in rows]
    f = Fig(720, 420)
    X, Y = f.line_chart(100, 34, 500, 120, [("", score, P.indigo)], ylabel="score", ymin=0.5, ymax=1.05,
                        yticks=[0.6, 0.8, 1.0])
    f.text(X(100) + 10, Y(score[-1][1]) + 4, f"score {score[-1][1]:.3f}: better", 12, P.indigo, weight=600)
    X2, Y2 = f.line_chart(100, 214, 500, 120, [("", chat, P.rose)], xlabel="thousands of batches",
                          ylabel="chat held-out loss", ymin=3.0, ymax=6.0, yticks=[3, 4, 5, 6],
                          hlines=[(3.30, "release gate 3.30", P.teal)])
    f.text(X2(100) + 10, Y2(chat[-1][1]) + 4, f"chat {chat[-1][1]:.2f}: worse", 12, P.rose, weight=600)
    f.text(X2(0) + 8, Y2(chat[0][1]) + 18, f"start {chat[0][1]:.2f}", 11, P.rose)
    f.caption("Figure 14.1  The first AnuLM-Omni run (ckpt_omni_curve.csv). The score, a mean over seven tasks, "
              "improved by 41% while chat, the skill it was meant to keep, got far worse.")
    f.save(OUT / "fig14_1_hidden.svg")


def final_chat(log):
    hits = re.findall(r"eval @\s+\d+.*?chat ([0-9.]+)", (AB / log).read_text(encoding="utf-8", errors="replace"))
    return float(hits[-1])


def fig_probes():
    """14.2: the probes, as logged: what moved chat's held-out loss."""
    rows = [("one batch a step, chat only", final_chat("probe_chat_only.log"), P.rose),
            ("accumulate 4, chat only", final_chat("c2_accum4.log"), P.teal),
            ("one batch, lr ÷ 5, chat only", final_chat("c3_lr1e5.log"), P.saffron),
            ("all tasks, one batch a step", final_chat("ln.log"), P.rose),
            ("… with projectors at a word's scale", final_chat("embed.log"), P.rose),
            ("all tasks, accumulate 8, chat 7%", final_chat("mix_accum8.log"), P.teal),
            ("… chat 25% of batches", final_chat("probe_chat25.log"), P.teal),
            ("… chat 15%, lr ÷ 2", final_chat("probe_chat15_lr25.log"), P.teal)]
    f = Fig(720, 400)
    f.text(16, 30, "chat held-out loss at the end of each probe (it started at 3.265; the gate is 3.30)", 12,
           P.stone, weight=600)
    x0, label_w, w = 16, 250, 660
    vmax = 5.2
    for i, (label, v, col) in enumerate(rows):
        y = 46 + i * 34
        f.text(x0 + label_w - 10, y + 16, label, 12, P.ink, "end", 500)
        bw = (w - label_w - 60) * v / vmax
        f.rect(x0 + label_w, y, bw, 22, col, r=4)
        f.text(x0 + label_w + bw + 8, y + 16, f"{v:.3f}", 12, P.ink)
    gx = x0 + label_w + (w - label_w - 60) * 3.30 / vmax
    f.line(gx, 40, gx, 46 + len(rows) * 34, P.ink, 1.2, dash="4 4")
    f.text(gx + 4, 46 + len(rows) * 34 + 14, "gate", 11, P.ink)
    f.caption("Figure 14.2  The probes of 30 October 2026 (logs/ab_projscale). Chat only: 1,000 batches; all tasks: "
              "3,000. One batch a step drifted even with chat alone.")
    f.save(OUT / "fig14_2_probes.svg")


ARMS = [("one batch a step", P.rose), ("accumulate 4", P.teal), ("one batch, lr / 4", P.saffron)]


def fig_lab():
    """14.3: the lab's three runs: the kept skill, and the score."""
    f = Fig(720, 450)
    sh = [("", [(b, h["shakespeare"]) for b, h, _ in R[n]["curve"]], c) for n, c in ARMS]
    X, Y = f.line_chart(100, 34, 470, 140, sh, ylabel="Shakespeare loss")
    ends = sorted((Y(R[n]["curve"][-1][1]["shakespeare"]), n) for n, _ in ARMS)
    placed = []
    for y, n in ends:
        y = max(y, placed[-1] + 15) if placed else y
        placed.append(y)
        f.text(X(R["batches"]) + 10, y + 4, f"{n} {R[n]['curve'][-1][1]['shakespeare']:.2f}", 11, dict(ARMS)[n], weight=600)
    sc = [("", [(b, s) for b, _, s in R[n]["curve"]], c) for n, c in ARMS]
    X2, Y2 = f.line_chart(100, 230, 470, 120, sc, xlabel="batches seen", ylabel="score (lower: better)")
    ends = sorted((Y2(R[n]["curve"][-1][2]), n) for n, _ in ARMS)
    placed = []
    for y, n in ends:
        y = max(y, placed[-1] + 15) if placed else y
        placed.append(y)
        f.text(X2(R["batches"]) + 10, y + 4, f"{n} {R[n]['curve'][-1][2]:.3f}", 11, dict(ARMS)[n], weight=600)
    f.caption("Figure 14.3  The lab: the same 2,400 batches, three ways. Top: the skill to keep; bottom: the run's "
              "one number. Measured in the lab.")
    f.save(OUT / "fig14_3_lab.svg")


if __name__ == "__main__":
    fig_hidden()
    fig_probes()
    if R:
        fig_lab()
    print("figures written to", OUT)
