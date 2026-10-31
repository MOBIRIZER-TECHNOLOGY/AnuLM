"""Draws Chapter 13's figures; the evidence one from the lab's results.json.

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


def fig_vocab():
    """13.1: growing both tables; old ids keep their rows."""
    f = Fig(720, 280)
    for col, (title, x0) in enumerate((("embed_tokens (reads ids)", 40), ("lm_head (writes ids)", 400))):
        f.text(x0, 36, title, 13, P.ink, weight=700, mono=True)
        rows = [("0 – 255", "the bytes", P.mist, P.stone, 70), ("256 – 258", "BOS, EOS, PAD", P.mist, P.stone, 30),
                ("259", "<|image|>", P.saffron_soft, P.saffron, 24), ("260", "<|/image|>", P.saffron_soft, P.saffron, 24)]
        y = 50
        for ids, what, fill, stroke, h in rows:
            f.rect(x0, y, 180, h - 3, fill, stroke, r=3, sw=1)
            f.text(x0 + 8, y + h / 2 + 2, ids, 11, P.ink, mono=True)
            f.text(x0 + 172, y + h / 2 + 2, what, 11, P.ink, "end", 600 if fill == P.saffron_soft else 400)
            y += h
    f.text(40, 230, "New rows start at the mean of the trained rows, plus a little noise: a fresh random row would", 12, P.ink)
    f.text(40, 248, "start far from where the trained ones live. Grow both tables, or the model can read the new ids but never write them.",
           12, P.ink)
    f.caption("Figure 13.1  Two new ids for pictures. Every old id keeps its row, so on text the model is exactly "
              "what it was.")
    f.save(OUT / "fig13_1_vocab.svg")


def fig_layout():
    """13.2: 196 patches pooled to 49, projected, and placed before the caption."""
    f = Fig(720, 370)
    g, s, x0, y0 = 7, 9, 24, 50
    for i in range(g * 2):
        for j in range(g * 2):
            f.rect(x0 + j * s, y0 + i * s, s - 1, s - 1, P.indigo_soft if (i // 2 + j // 2) % 2 else P.teal_soft,
                   "none", r=1)
    side = 2 * g * s
    f.text(x0, 36, "196 SigLIP patches (14 × 14)", 12, P.ink, weight=600)
    f.text(x0, y0 + side + 18, "each 2 × 2 square → one vector", 11, P.stone)
    px = x0 + side + 60
    f.arrow(x0 + side + 10, y0 + side / 2, px - 2, y0 + side / 2, P.stone)
    f.box(px, y0 + side / 2 - 26, 150, 52, "projector", P.saffron_soft, P.saffron, 12, sub="Linear · GELU · Linear · LN")
    f.text(px + 166, y0 + side / 2 - 4, "49 vectors, each as wide", 11, P.stone)
    f.text(px + 166, y0 + side / 2 + 12, "as the model's hidden state", 11, P.stone)
    y = 262
    seq = [("<|image|>", P.saffron_soft, P.saffron, 70), ("49 picture vectors", P.indigo_soft, P.indigo, 180),
           ("<|/image|>", P.saffron_soft, P.saffron, 76), ("A man is standing …", P.mist, P.stone, 180),
           ("EOS", P.mist, P.stone, 44)]
    x = 24
    for label, fill, stroke, w in seq:
        f.rect(x, y, w, 30, fill, stroke, r=4, sw=1)
        f.text(x + w / 2, y + 20, label, 11, P.ink, "middle", 600)
        x += w + 4
    f.text(24, y + 50, "one sequence of vectors, fed in with no ids", 11, P.ink, weight=600)
    f.text(24 + 70 + 180 + 76 + 12, y + 50, "graded: the caption and its EOS only", 11, P.saffron, weight=600)
    f.parts.append(f'<path d="M {px + 75} {y0 + side / 2 + 28} C {px + 75} 210, 188 210, 188 {y - 2}" fill="none" '
                   f'stroke="{P.stone}" stroke-width="1.3" stroke-dasharray="4 4"/>')
    f.caption("Figure 13.2  vision_encoder.py's layout. The picture is read, never graded; the caption is graded "
              "one position early, as in Chapter 9.")
    f.save(OUT / "fig13_2_layout.svg")


def fig_looks():
    """13.3: held-out caption loss with the right picture, a wrong one, or none."""
    f = Fig(720, 300)
    modes = [("right picture", P.teal), ("wrong picture", P.rose), ("no picture", P.stone)]
    for col, name in enumerate(("matched", "unit gain")):
        x0 = 16 + col * 352
        r = R[name]
        f.text(x0, 34, f"{name}: picture vectors start at norm {r['projector_norm_at_start']:.2f}", 13, P.ink,
               weight=700)
        rows = [(m, r[m], c) for m, c in modes]
        lo = min(v for _, v, _ in rows)
        f.hbars(x0, 50, 340, rows, vmax=max(v for _, v, _ in rows) * 1.05, label_w=120, fmt="{:.3f}")
    f.text(16, 170, f"A word's embedding has norm {R['matched']['word_norm']:.2f}. Loss per byte of caption on "
                    f"{R['test']} pictures neither model saw;", 12, P.ink)
    f.text(16, 188, "\"wrong picture\" gives each caption another test picture. Lower is better.", 12, P.ink)
    f.caption("Figure 13.3  Does it look? If the model uses the picture, the right one must beat a wrong one. "
              "Measured in the lab.")
    f.save(OUT / "fig13_3_looks.svg")


if __name__ == "__main__":
    fig_vocab()
    fig_layout()
    if R:
        fig_looks()
    print("figures written to", OUT)
