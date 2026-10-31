"""Draws Chapter 12's figures; the evidence ones from the lab's results.json.

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


def fig_decide():
    """12.1: one pass, a score at the last token of every option."""
    f = Fig(720, 300)
    toks = [("Message: How do I locate my card?", P.mist, P.stone, 232),
            ("- card swallowed", P.indigo_soft, P.indigo, 128), ("- card arrival", P.indigo_soft, P.indigo, 118),
            ("- wrong exchange rate", P.indigo_soft, P.indigo, 160)]
    x = 16
    ends = []
    for label, fill, stroke, w in toks:
        f.rect(x, 50, w, 34, fill, stroke, r=5, sw=1.2)
        f.text(x + w / 2, 72, label, 12, P.ink, "middle", 500)
        if label.startswith("-"):
            ends.append(x + w - 14)
        x += w + 6
    f.text(16, 36, "one forward pass over the message and every option's name", 12, P.stone, weight=600)
    probs = [0.06, 0.88, 0.06]
    for k, (cx, p) in enumerate(zip(ends, probs)):
        f.parts.append(f'<circle cx="{cx:.1f}" cy="84" r="4.5" fill="{P.saffron}"/>')
        f.arrow(cx, 90, cx, 122, P.saffron, 1.3, 6)
        f.box(cx - 46, 124, 92, 30, "head → score", P.saffron_soft, P.saffron, 11)
        f.arrow(cx, 156, cx, 186, P.stone, 1.2, 6)
        h = 60 * p
        f.rect(cx - 18, 250 - h, 36, h, P.teal if p == max(probs) else P.teal_soft, r=3, sw=0)
        f.text(cx, 244 - h, f"{p:.0%}", 12, P.ink, "middle", 600)
    f.text(16, 210, "softmax over", 12, P.stone)
    f.text(16, 226, "the options", 12, P.stone)
    f.caption("Figure 12.1  Decide. The hidden state at the last token of each option's name has read the message "
              "and that option; one linear head scores it.")
    f.save(OUT / "fig12_1_decide.svg")


def fig_reliability():
    """12.3: confidence against accuracy, before and after the temperature."""
    d = R["decide"]
    f = Fig(720, 380)
    X, Y = f.line_chart(110, 30, 300, 260, [("", [(0, 0), (1, 1)], P.stone_soft)],
                        xlabel="confidence", ylabel="share right", ymin=0, ymax=1,
                        xticks=[0, 0.25, 0.5, 0.75, 1], yticks=[0, 0.25, 0.5, 0.75, 1])
    f.text(X(0.66), Y(0.42), "the diagonal: perfectly calibrated", 11, P.stone, "start")
    n_all = sum(n for _, n, _, _ in d["reliability_after"])
    for b, n, conf, acc in d["reliability_after"]:
        r = 3 + 12 * (n / n_all) ** 0.5
        f.parts.append(f'<circle cx="{X(conf):.1f}" cy="{Y(acc):.1f}" r="{r:.1f}" fill="{P.saffron}" '
                       f'fill-opacity="0.75"/>')
    tx = 470
    a, c = d["before"], d["after"]
    f.text(tx, 54, f"test accuracy {a['accuracy']:.1%}", 14, P.ink, weight=700)
    f.text(tx, 74, f"chance with {d['options']} options: {d['chance']:.0%}", 12, P.stone)
    f.text(tx, 112, f"temperature fitted: {d['temperature']}", 12, P.ink, weight=600)
    f.text(tx, 132, f"ECE {a['ece']:.3f} → {c['ece']:.3f}", 12, P.ink)
    f.text(tx, 150, f"mean confidence {a['mean_confidence']:.1%} → {c['mean_confidence']:.1%}", 12, P.ink)
    rows = [(0.5, "≥ 0.5"), (0.7, "≥ 0.7"), (0.9, "≥ 0.9")]
    f.text(tx, 190, "answer only when confidence is", 12, P.stone, weight=600)
    for k, (tau, label) in enumerate(rows):
        cov = c[f"answer_if_conf>={tau}"]
        acc = "–" if cov["coverage"] == 0 else f"{cov['accuracy']:.0%} right"
        f.text(tx, 212 + 20 * k, f"{label}: answers {cov['coverage']:.0%}, {acc}", 12, P.ink)
    f.caption("Figure 12.3  Each dot is a bin of test messages (its area: how many): their mean confidence against "
              "how many were right, after the temperature. Measured in the lab.")
    f.save(OUT / "fig12_3_reliability.svg")


def fig_lookup():
    """12.2: R@1 per retriever, Hindi and English."""
    L = R["lookup"]
    labels = ["BM25", "BM25, \\w+ words", "BM25, b = 0", "word overlap, no idf"]
    f = Fig(720, 330)
    for col, (name, title) in enumerate((("xquad_hi", "Hindi (XQuAD)"), ("squad_en", "English (SQuAD)"))):
        x0 = 16 + col * 352
        f.text(x0, 34, f"{title}: {L[name]['passages']} passages, {L[name]['questions']:,} questions", 13, P.ink,
               weight=700)
        rows = [(lab, L[name][lab]["r1"], P.saffron if lab == "BM25" else P.stone) for lab in labels]
        f.hbars(x0, 52, 340, rows, vmax=1.0, label_w=150, fmt="{:.1%}")
    f.text(16, 210, "R@1: the passage a question was written from comes back first.", 12, P.ink)
    f.text(16, 228, "\\w+ is Python's word pattern. It stops at every Devanagari vowel sign, so it cuts Hindi words in two.",
           12, P.ink)
    f.caption("Figure 12.2  Four ways to score passages, on the same questions. Measured in the lab.")
    f.save(OUT / "fig12_2_lookup.svg")


if __name__ == "__main__":
    fig_decide()
    if R:
        fig_reliability()
        fig_lookup()
    print("figures written to", OUT)
