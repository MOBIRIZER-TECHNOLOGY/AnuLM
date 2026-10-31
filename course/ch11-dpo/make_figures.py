"""Draws Chapter 11's figures; the evidence ones from the lab's results.json.

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
ARMS = [("dpo", P.rose), ("dpo, lr / 10", P.saffron), ("dpo + sft", P.teal), ("sft on chosen", P.indigo)]


def fig_dpo():
    """11.1: one preference pair through the policy and the frozen reference."""
    f = Fig(720, 320)
    f.box(16, 40, 200, 50, "prompt", P.mist, P.stone, 13, sub="Question: what is 44 plus 56?")
    f.box(16, 110, 200, 44, "chosen: 100", P.teal_soft, P.teal, 13)
    f.box(16, 170, 200, 44, "rejected: 110", P.rose_soft, P.rose, 13)
    f.box(280, 70, 170, 56, "policy", P.saffron_soft, P.saffron, 13, sub="trains")
    f.box(280, 160, 170, 56, "reference", P.stone_soft, P.stone, 13, sub="a frozen copy of the start")
    for y in (132, 192):
        f.arrow(218, y, 278, 100 if y == 132 else 186, P.stone, 1.2, 6)
    f.text(486, 70, "chosen: log p − log p_ref", 13, P.teal, weight=700)
    f.text(486, 88, "how much MORE likely than at the start", 11, P.stone)
    f.text(486, 122, "rejected: log p − log p_ref", 13, P.rose, weight=700)
    f.text(486, 140, "the same, for the wrong answer", 11, P.stone)
    f.arrow(452, 98, 480, 80, P.stone, 1.2, 6)
    f.arrow(452, 188, 480, 132, P.stone, 1.2, 6)
    f.rect(486, 168, 218, 66, P.indigo_soft, P.indigo, r=6)
    f.text(595, 194, "margin = β × (chosen − rejected)", 12, P.ink, "middle", 600)
    f.text(595, 218, "loss = −log sigmoid(margin)", 12, P.ink, "middle", 600)
    f.text(16, 262, "Nothing in the loss says the chosen answer must stay likely: only that it gains on the rejected one.",
           12, P.rose, weight=600)
    f.caption("Figure 11.1  DPO on one pair. At the start the policy is the reference, every margin is 0, and the "
              "loss is ln 2 = 0.693.")
    f.save(OUT / "fig11_1_dpo.svg")


def fig_accuracy():
    """11.2: greedy accuracy on held-out sums while each arm trains on the same pairs."""
    f = Fig(720, 340)
    series = [("", [(s, 100 * c["greedy"]) for s, c in R[n]["curve"]], col) for n, col in ARMS]
    X, Y = f.line_chart(90, 30, 470, 220, series, xlabel="step", ylabel="sums answered exactly (%)",
                        ymin=0, ymax=100, yticks=[0, 25, 50, 75, 100])
    ends = sorted((Y(100 * R[n]["final"]["greedy"]), n) for n, _ in ARMS)
    placed = []
    for y, n in ends:
        y = max(y, placed[-1] + 16) if placed else y
        placed.append(y)
        f.text(X(R["steps"]) + 12, y + 4, f"{n}  {100 * R[n]['final']['greedy']:.1f}%", 12, dict(ARMS)[n], weight=600)
    f.caption(f"Figure 11.2  Four ways to use the same {R['pairs']} (right, wrong) pairs. Greedy accuracy on "
              f"{200} held-out sums; all start at {100 * R['start']['greedy']:.1f}%. Measured in the lab.")
    f.save(OUT / "fig11_2_accuracy.svg")


def fig_logp():
    """11.3: held-out log-probabilities of the chosen and the rejected answers."""
    f = Fig(720, 360)
    show = [("dpo", P.rose), ("dpo + sft", P.teal)]
    series = []
    for n, col in show:
        series.append(("", [(s, c["logp_chosen"]) for s, c in R[n]["curve"]], col))
    X, Y = f.line_chart(90, 30, 470, 240, series, xlabel="step", ylabel="mean log-probability (held-out pairs)",
                        ymin=-45, ymax=2, yticks=[-40, -30, -20, -10, 0])
    for n, col in show:
        pts = [(s, c["logp_rejected"]) for s, c in R[n]["curve"]]
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(b):.1f}" for i, (a, b) in enumerate(pts))
        f.parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" stroke-dasharray="6 4"/>')
    lab = [(R["dpo"]["final"]["logp_chosen"], "dpo: chosen", P.rose), (R["dpo"]["final"]["logp_rejected"], "dpo: rejected", P.rose),
           (R["dpo + sft"]["final"]["logp_chosen"], "dpo + sft: chosen", P.teal),
           (R["dpo + sft"]["final"]["logp_rejected"], "dpo + sft: rejected", P.teal)]
    placed = []
    for v, text, col in sorted(lab, key=lambda t: -t[0]):
        y = Y(v)
        y = max(y, placed[-1] + 16) if placed else y
        placed.append(y)
        f.text(X(R["steps"]) + 12, y + 4, f"{text} {v:.1f}", 12, col, weight=600)
    f.caption("Figure 11.3  Solid: the right answers; dashed: the wrong ones. Plain DPO widens the gap by pushing "
              "BOTH down. Measured in the lab.")
    f.save(OUT / "fig11_3_logp.svg")


if __name__ == "__main__":
    fig_dpo()
    if R:
        fig_accuracy()
        fig_logp()
    print("figures written to", OUT)
