"""Draws Chapter 10's figures; the evidence ones from the lab's results.json.

    python lab.py && python make_figures.py
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from fig import Fig, P  # noqa: E402

OUT = HERE / "figures"
R = json.loads((HERE / "results.json").read_text(encoding="utf-8")) if (HERE / "results.json").exists() else None
NAMES = {"full": "full fine-tune", "lora": "LoRA", "lora + rows": "LoRA + rows", "qlora + rows": "QLoRA + rows"}
COLS = {"full": P.indigo, "lora": P.rose, "lora + rows": P.saffron, "qlora + rows": P.teal}


def fig_lora():
    """10.2: a frozen weight matrix and the thin trainable pair beside it."""
    f = Fig(720, 372)
    f.text(24, 36, "x", 18, P.teal, weight=700, mono=True)
    f.arrow(44, 31, 96, 31, P.stone)
    f.line(96, 31, 96, 230, P.stone, 1.4)
    f.arrow(96, 110, 130, 110, P.stone)
    f.arrow(96, 230, 130, 230, P.stone)
    # frozen W
    f.rect(132, 50, 150, 150, P.stone_soft, P.stone, r=6)
    f.text(207, 120, "W", 30, P.stone, "middle", 700)
    f.text(207, 148, "frozen", 12, P.stone, "middle")
    f.text(207, 216, "d × d", 12, P.stone, "middle", mono=True)
    # A then B
    f.rect(132, 218, 150, 22, P.saffron_soft, P.saffron, r=4)
    f.text(207, 233, "A   (r × d)", 12, P.ink, "middle", 600, mono=True)
    f.arrow(284, 229, 316, 229, P.stone)
    f.rect(318, 168, 22, 122, P.saffron_soft, P.saffron, r=4)
    f.text(329, 306, "B", 13, P.ink, "middle", 700, mono=True)
    f.text(329, 322, "d × r", 11, P.stone, "middle", mono=True)
    f.arrow(284, 125, 398, 125, P.stone)
    f.arrow(342, 229, 398, 150, P.stone)
    f.rect(400, 108, 44, 44, P.paper, P.ink, r=22)
    f.text(422, 137, "+", 22, P.ink, "middle", 700)
    f.arrow(446, 130, 486, 130, P.stone)
    f.text(494, 136, "y", 18, P.indigo, weight=700, mono=True)
    tx = 520
    f.text(tx, 60, "y = W x + (α/r)·B A x", 12, P.ink, weight=700, mono=True)
    for k, line in enumerate(["W stays as pre-training", "left it. Only A and B learn.", "",
                              "With d = 1,024 and r = 16:", "W has 1,048,576 numbers,", "A and B 32,768 (3%).",
                              "", "B starts at zero, so", "training begins at the", "base model."]):
        f.text(tx, 88 + 20 * k, line, 12, P.ink)
    f.caption("Figure 10.2  LoRA leaves the weight frozen (stone) and learns a thin correction "
              "B·A beside it (saffron: what learns).")
    f.save(OUT / "fig10_2_lora.svg")


def fig_nf4():
    """10.3: NF4's sixteen levels sit where the weights are."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ch10", HERE.parent / "build" / "solutions" / "ch10.py")
    sol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sol)
    NF4 = sol.NF4                                     # the exact table the checks hold you to
    f = Fig(720, 336)
    x0, x1, base, top = 70, 650, 220, 70
    X = lambda v: x0 + (v + 1.15) / 2.3 * (x1 - x0)
    pts = []
    for i in range(201):
        v = -1.15 + 2.3 * i / 200
        pts.append((X(v), base - (base - top) * math.exp(-v * v / (2 * 0.35 ** 2))))
    d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))
    f.parts.append(f'<path d="{d} L{x1:.1f},{base} L{x0:.1f},{base} Z" fill="{P.teal_soft}" stroke="none"/>')
    f.parts.append(f'<path d="{d}" fill="none" stroke="{P.teal}" stroke-width="2"/>')
    f.line(x0, base, x1, base, P.ink, 1.2)
    for v in NF4.tolist():
        f.line(X(v), base - 6, X(v), base + 14, P.saffron, 2.4)
    for v in (-1, -0.5, 0, 0.5, 1):
        f.text(X(v), base + 32, f"{v:g}", 11, P.stone, "middle")
    f.text(X(0), 52, "how weights are spread (scaled to −1..1 per block of 64)", 12, P.teal, "middle", 600)
    f.text(X(0.62), base + 54, "the 16 NF4 levels (saffron): closest together near zero,", 12, P.ink, "middle")
    f.text(X(0.62), base + 70, "where most weights are", 12, P.ink, "middle")
    f.caption("Figure 10.3  NormalFloat-4 stores each weight as one of 16 levels, chosen as quantiles of a "
              "normal distribution, plus one scale per block.")
    f.save(OUT / "fig10_3_nf4.svg")


def fig_memory():
    """10.1: what fine-tuning the 400M model holds in memory, three ways."""
    # 398M parameters. fp32 weights 4 B, gradients 4 B, AdamW moments 8 B per trained
    # parameter (lora.py and docs/RESULTS.md section 29). NF4: 0.5 B per weight plus
    # one fp32 scale per 64 weights. Adapters: r=16 on attention saved a 5.7 MB file
    # in section 29, i.e. ~1.43M fp32 numbers; each also needs a gradient and two moments.
    n, ad = 398e6, 5.7e6 / 4
    rows = [("full fine-tune", [(n * 4, P.indigo, "weights"), (n * 4, P.rose, "gradients"),
                                (n * 8, P.rose_soft, "AdamW moments")]),
            ("LoRA", [(n * 4, P.stone, "frozen weights"), (ad * 16, P.saffron, "adapters + their optimizer")]),
            ("QLoRA", [(n * 0.5 + n / 64 * 4, P.stone, "frozen 4-bit weights"),
                       (ad * 16, P.saffron, "adapters + their optimizer")])]
    f = Fig(720, 280)
    x, w, scale = 150, 440, 440 / (n * 16)
    for i, (label, parts) in enumerate(rows):
        y = 50 + i * 66
        f.text(x - 12, y + 22, label, 13, P.ink, "end", 600)
        xx = x
        for v, col, _ in parts:
            bw = max(v * scale, 3)
            f.rect(xx, y, bw, 32, col, r=3)
            xx += bw
        total = sum(v for v, _, _ in parts)
        f.text(xx + 10, y + 21, f"{total / 1e9:.{1 if total >= 1e9 else 2}f} GB", 13, P.ink, weight=700)
        f.text(x, y + 50, " · ".join(f"{name} {v / 1e9:.2f} GB" if v >= 5e7 else f"{name} {v / 1e6:.0f} MB"
                                     for v, _, name in parts), 11, P.stone)
    f.caption("Figure 10.1  Memory to fine-tune the 398M AnuLM model before activations, worked out from its "
              "parameter count (bytes per number × how many).")
    f.save(OUT / "fig10_1_memory.svg")


def fig_learn_forget():
    """10.5: learned the new language (left is better) vs forgot the old text (down is better)."""
    m = R["methods"]
    f = Fig(720, 380)
    xs = [m[k]["new_loss"] for k in m]
    ys = [R["base_old"]] + [m[k]["old_loss"] for k in m]
    x0, x1 = 0.5, max(xs) + 0.6
    y0, y1 = min(ys) - 0.15, max(ys) + 0.15
    X, Y = f.line_chart(110, 40, 540, 250, [("", [(x0, y0), (x1, y1)], P.paper)],
                        xlabel="loss on Hindi after 1,000 steps (lower: learned more)",
                        ylabel="loss on Shakespeare (lower: forgot less)")
    f.line(110, Y(R["base_old"]), 650, Y(R["base_old"]), P.stone, 1.2, dash="5 4")
    f.text(646, Y(R["base_old"]) - 6, f"base model on Shakespeare: {R['base_old']:.2f}", 11, P.stone, "end", 600)
    side = {"full": (12, "start", 4), "lora": (-12, "end", 4), "lora + rows": (12, "start", -8),
            "qlora + rows": (12, "start", 16)}
    for k in m:
        a, b = m[k]["new_loss"], m[k]["old_loss"]
        f.parts.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="6" fill="{COLS[k]}"/>')
        dx, anchor, dy = side[k]
        f.text(X(a) + dx, Y(b) + dy, f"{NAMES[k]}  ({a:.2f}, {b:.2f})", 12, COLS[k], anchor, 600)
    f.caption(f"Figure 10.5  What each method learned and forgot. The base model's Hindi loss, "
              f"{R['base_new']:.1f}, is off the chart to the right. Measured in the lab.")
    f.save(OUT / "fig10_5_learn_forget.svg")


def fig_curves():
    """10.4: held-out Hindi loss from step 100; all four start at the base model's loss."""
    m = R["methods"]
    f = Fig(720, 350)
    X, Y = f.line_chart(90, 30, 470, 240, [("", [p for p in m[k]["curve"] if p[0] >= 100], COLS[k]) for k in m],
                        xlabel="fine-tuning step", ylabel="loss on Hindi", ymin=0.5, ymax=4.5,
                        yticks=[1, 2, 3, 4])
    ends = sorted(((Y(m[k]["curve"][-1][1]), k) for k in m))
    placed = []
    for y, k in ends:
        y = max(y, placed[-1] + 16) if placed else y
        placed.append(y)
        f.text(X(R["steps"]) + 12, y + 4, f"{NAMES[k]}  {m[k]['curve'][-1][1]:.2f}", 12, COLS[k], weight=600)
    f.caption(f"Figure 10.4  Held-out loss on Hindi from step 100; all four start at {R['base_new']:.1f}. "
              "Attention-only LoRA cannot write bytes the model never learned. Measured in the lab.")
    f.save(OUT / "fig10_4_curves.svg")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    for fn in (fig_lora, fig_nf4, fig_memory):
        fn()
    if R:
        fig_learn_forget()
        fig_curves()
    print("figures written to", OUT)
