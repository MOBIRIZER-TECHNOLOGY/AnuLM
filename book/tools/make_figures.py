"""
Draw the book's figures as SVG, from the same data the chapters quote.

    python book/tools/make_figures.py        # -> book/figures/*.svg

Every number a figure shows is computed here from its source, and asserted
against the value the chapter prints:

- the toy `The dog bit` (Ch 11) is recomputed with the chapter's matrices;
- the √d experiment (Ch 11) repeats attention_by_hand.py's seeded draw;
- the trained heads and expert loads come from book/measurements/attention.json;
- the parameter breakdown (Ch 20) is the eight-term sum of trace_forward.py.

Colours follow STYLE.md. The SVG carries light-theme colours as attributes and
class names that the HTML preview restyles for dark mode. Each mark has a
<title>, which browsers show on hover.
"""

from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

import torch

BOOK = Path(__file__).resolve().parent.parent
ROOT = BOOK.parent
OUT = BOOK / "figures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BOOK / "code-examples"))
from anulm import model as M  # noqa: E402
from trace_forward import released  # noqa: E402

INK, MUTED, RULE, CARD, PAPER, NIGHT = "#17262E", "#5E6E72", "#DCE1DF", "#F0F2F1", "#FFFFFF", "#13232A"
GOLD, GOLDTEXT, CORAL, TEAL, SLATE = "#D9A13B", "#8F6A1E", "#C4532F", "#0E9488", "#56676C"
HEAT = "#A8701A"          # heatmap ramp end: deeper than GOLD so the top of the scale stands out on white
SANS = "Lato, 'Segoe UI', system-ui, sans-serif"
MONO = "'JetBrains Mono', Consolas, monospace"
SERIF = "Lora, Georgia, serif"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def text(x, y, s, size=13, cls="f-ink", fill=INK, anchor="start", weight=400, family=SANS, extra=""):
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" fill="{fill}" font-size="{size}" font-family="{family}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{extra}>{esc(s)}</text>')


def rect(x, y, w, h, fill, cls="", rx=0, extra="", tip=None):
    t = f"<title>{esc(tip)}</title>" if tip else ""
    c = f' class="{cls}"' if cls else ""
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{c}{extra}>{t}</rect>'


def line(x1, y1, x2, y2, stroke=RULE, cls="f-rule", width=1, extra=""):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" class="{cls}" '
            f'stroke-width="{width}"{extra}/>')


def svg(name, w, h, title, desc, body):
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-labelledby="t d" '
           f'class="figure-svg">\n<title id="t">{esc(title)}</title>\n<desc id="d">{esc(desc)}</desc>\n'
           f'<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
           f'<line x1="0" y1="0" x2="0" y2="6" stroke="{RULE}" class="f-rule" stroke-width="2"/></pattern>'
           f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
           f'<path d="M0,0 L10,5 L0,10 z" fill="{MUTED}" class="f-muted-fill"/></marker></defs>\n'
           + "\n".join(body) + "\n</svg>\n")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(doc, encoding="utf-8")
    return name


def close(a, b, tol, what):
    assert abs(a - b) <= tol, f"{what}: computed {a}, chapter says {b}"


# --------------------------------------------------------------------------- Ch 11
def toy():
    X = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])
    W_q, W_k, W_v = torch.tensor([[1., 0.], [0., 2.]]), torch.eye(2), torch.tensor([[0., 1.], [1., 0.]])
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    S = Q @ K.T
    mask = torch.ones(3, 3, dtype=torch.bool).tril()
    W = (S / math.sqrt(2)).masked_fill(~mask, float("-inf")).softmax(-1)
    return Q, K, V, S, W, W @ V


def fig_flow():
    w, h = 760, 300
    b = [text(20, 28, "Attention for one head, as a pipeline", 15, weight=700, family=SERIF),
         text(20, 48, "S queries, T keys and values, head width d · the shapes are those of 11.2", 12, "f-muted", MUTED)]

    def node(x, y, ww, title, shape, accent=False):
        out = [rect(x, y, ww, 50, CARD, "f-card", 8,
                    f' stroke="{GOLD}" stroke-width="2.5"' if accent else f' stroke="{RULE}" stroke-width="1"')]
        out.append(text(x + ww / 2, y + 22, title, 13, anchor="middle", weight=700))
        out.append(text(x + ww / 2, y + 40, shape, 11.5, "f-muted", MUTED, "middle", family=MONO))
        return out

    def arrow(x1, y1, x2, y2):
        return line(x1, y1, x2, y2, MUTED, "f-muted-stroke", 1.5, ' marker-end="url(#arrow)"')

    b += node(20, 136, 92, "x", "(S, D)")
    for i, (t, s) in enumerate((("Q = x W_Q", "(S, d)"), ("K = x W_K", "(T, d)"), ("V = x W_V", "(T, d)"))):
        y = 72 + i * 72
        b += node(150, y, 118, t, s)
        b.append(arrow(112, 161, 148, y + 25))
    b += node(306, 90, 104, "Q Kᵀ", "(S, T) scores")
    b.append(arrow(268, 97, 304, 112))
    b.append(arrow(268, 169, 304, 122))
    b += node(430, 90, 70, "÷ √d", "scale", accent=True)
    b.append(arrow(410, 115, 428, 115))
    b += node(520, 90, 96, "+ mask", "−∞ on future")
    b.append(arrow(500, 115, 518, 115))
    b += node(636, 90, 104, "softmax", "each row → 1")
    b.append(arrow(616, 115, 634, 115))
    b += node(560, 218, 180, "weights · V", "output (S, d)")
    b.append(arrow(688, 140, 670, 216))
    b.append(arrow(268, 241, 558, 243))
    return svg("fig11-1-flow.svg", w, h, "Figure 11.1 · Scaled dot-product attention, one head",
               "x is projected to Q, K and V. Q times K-transpose gives an S by T score matrix, which is divided by the "
               "square root of d, masked, and turned into weights by a row-wise softmax; the weights average V into an "
               "S by d output.", b)


def fig_lookup():
    Q, K, V, S, W, O = toy()
    words = ["The", "dog", "bit"]
    close(W[2, 1].item(), 0.620, 5e-4, "bit→dog weight")
    close(O[2, 0].item(), 1.546, 5e-4, "bit output[0]")
    w, h = 760, 380
    b = [text(20, 28, "One query, three keys: the bit row of the toy, step by step", 15, weight=700, family=SERIF)]
    # query card
    b.append(rect(20, 128, 120, 74, CARD, "f-card", 6))
    b.append(text(80, 152, "query of bit", 12, "f-muted", MUTED, "middle"))
    b.append(text(80, 180, "q = [1, 2]", 15, anchor="middle", weight=700, family=MONO))
    cols = {"key": 214, "score": 330, "scaled": 408, "weight": 478, "value": 640}
    for k, x in cols.items():
        label = {"key": "key k", "score": "q·k", "scaled": "÷ √2", "weight": "softmax weight", "value": "value v"}[k]
        b.append(text(x, 64, label, 11, "f-muted", MUTED, "start" if k in ("key", "weight", "value") else "middle",
                      700, extra=' letter-spacing="0.6"'))
    b.append(line(170, 74, 740, 74))
    for i, word in enumerate(words):
        y = 112 + i * 74
        wt = W[2, i].item()
        b.append(line(140, 165, 206, y, MUTED, "f-muted-stroke", 1.2, ' marker-end="url(#arrow)"'))
        b.append(text(214, y - 6, word, 12, "f-muted", MUTED, weight=700))
        b.append(text(214, y + 14, f"[{K[i, 0]:.0f}, {K[i, 1]:.0f}]", 14, family=MONO))
        b.append(text(330, y + 14, f"{S[2, i]:.0f}", 14, anchor="middle", family=MONO))
        b.append(text(408, y + 14, f"{S[2, i] / math.sqrt(2):.3f}", 14, anchor="middle", family=MONO))
        bw = 130 * wt
        b.append(rect(478, y + 1, 130, 18, CARD, "f-card", 4))
        b.append(rect(478, y + 1, max(bw, 2), 18, GOLD, "f-gold", 4,
                      tip=f"{word}: weight {wt:.3f}"))
        b.append(text(478 + bw + 6 if bw < 90 else 478 + bw - 6, y + 14, f"{wt:.3f}", 12,
                      "f-ink" if bw < 90 else "f-onfill", INK if bw < 90 else NIGHT, "start" if bw < 90 else "end", 700, MONO))
        b.append(text(640, y + 14, f"[{V[i, 0]:.0f}, {V[i, 1]:.0f}]", 14, family=MONO))
    b.append(line(740, 74, 740, 74))
    b.append(text(20, 316, "output = Σ weight × value", 12, "f-muted", MUTED, weight=700))
    b.append(text(20, 340, f"0.074·[0, 1] + 0.620·[2, 0] + 0.306·[1, 1] = [{O[2, 0]:.3f}, {O[2, 1]:.3f}]",
                  14, family=MONO))
    b.append(text(20, 366, "bit's query is long in the second component; dog's key is too, so dog wins 62% of the weight.",
                  12, "f-muted", MUTED))
    return svg("fig11-2-soft-lookup.svg", w, h, "Figure 11.2 · One attention row by hand",
               "The query of 'bit' is scored against the keys of The, dog and bit; the scores are divided by √2, "
               "turned into weights 0.074, 0.620 and 0.306 by the softmax, and used to average the values.", b)


def fig_grid():
    *_, W, _ = toy()
    words = ["The", "dog", "bit"]
    w, h, c, x0, y0 = 470, 300, 72, 120, 78
    b = [text(20, 28, "The toy's attention weights", 15, weight=700, family=SERIF),
         text(20, 48, "rows: queries · columns: keys · after the mask and the softmax", 12, "f-muted", MUTED)]
    for j, word in enumerate(words):
        b.append(text(x0 + j * c + c / 2, y0 - 10, word, 12, "f-muted", MUTED, "middle", 700))
    for i, qw in enumerate(words):
        b.append(text(x0 - 12, y0 + i * c + c / 2 + 5, qw, 12, "f-muted", MUTED, "end", 700))
        for j in range(3):
            x, y = x0 + j * c, y0 + i * c
            if j > i:
                b.append(rect(x + 1, y + 1, c - 2, c - 2, "url(#hatch)", "", 4, tip=f"{qw} → {words[j]}: masked"))
                b.append(text(x + c / 2, y + c / 2 + 5, "masked", 11, "f-muted", MUTED, "middle"))
                continue
            v = W[i, j].item()
            b.append(rect(x + 1, y + 1, c - 2, c - 2, CARD, "f-card", 4))
            b.append(rect(x + 1, y + 1, c - 2, c - 2, HEAT, "f-heat", 4,
                          f' fill-opacity="{0.08 + 0.92 * v:.3f}"', f"{qw} → {words[j]}: {v:.3f}"))
            dark = v > 0.55
            b.append(text(x + c / 2, y + c / 2 + 5, f"{v:.3f}", 14, "f-onheat" if dark else "f-ink",
                          PAPER if dark else INK, "middle", 700, MONO))
    b.append(text(x0 + 3 * c + 14, y0 + 2 * c + c / 2 - 4, "each row", 11, "f-muted", MUTED))
    b.append(text(x0 + 3 * c + 14, y0 + 2 * c + c / 2 + 11, "sums to 1", 11, "f-muted", MUTED))
    return svg("fig11-3-toy-weights.svg", w, h, "Figure 11.3 · The toy's attention weights",
               "A 3 by 3 grid. The row: 1.000. The dog row: 0.003, 0.997. The bit row: 0.074, 0.620, 0.306. "
               "Cells above the diagonal are masked.", b)


def sqrt_d_data():
    """attention_by_hand.why_scale's draw, repeated: same generator, seed and order."""
    g = torch.Generator().manual_seed(0)
    out = {}
    for d in (2, 8, 64):
        q = torch.randn(2000, d, generator=g)
        k = torch.randn(256, d, generator=g)
        s = q @ k.T
        row = []
        for logits in (s, s / math.sqrt(d)):
            p = logits.softmax(-1)
            row.append((-(p * p.clamp_min(1e-30).log()).sum(-1)).exp().mean().item())
        out[d] = row
    return out


def fig_sqrt_d():
    data = sqrt_d_data()
    close(data[64][0], 2.5, 0.05, "keys in play d=64 unscaled")
    close(data[64][1], 157.0, 0.05, "keys in play d=64 scaled")
    w, h = 720, 290
    x0, x1, y0 = 120, 640, 96
    lx = lambda v: x0 + (x1 - x0) * math.log2(v) / 8          # 1 … 256 on a log scale
    b = [text(20, 28, "Keys in play, out of 256, with and without ÷√d", 15, weight=700, family=SERIF),
         text(20, 48, "random unit-variance queries and keys · mean over 2,000 rows · illustrative", 12, "f-muted", MUTED)]
    # legend
    b.append(f'<circle cx="{x0 + 4}" cy="70" r="5" fill="{CORAL}" class="f-coral"/>')
    b.append(text(x0 + 14, 74, "unscaled", 12))
    b.append(f'<circle cx="{x0 + 104}" cy="70" r="5" fill="{TEAL}" class="f-teal"/>')
    b.append(text(x0 + 114, 74, "divided by √d", 12))
    for t in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        b.append(line(lx(t), y0 - 6, lx(t), y0 + 160))
        b.append(text(lx(t), y0 + 178, f"{t}", 11, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y0 + 196, "keys in play (e to the entropy of a row; log scale)", 11, "f-muted", MUTED, "middle"))
    for r, d in enumerate((2, 8, 64)):
        y = y0 + 22 + r * 52
        un, sc = data[d]
        b.append(text(x0 - 16, y + 4, f"d = {d}", 13, anchor="end", weight=700, family=MONO))
        b.append(line(lx(un), y, lx(sc), y, MUTED, "f-muted-stroke", 2))
        b.append(f'<circle cx="{lx(un):.1f}" cy="{y}" r="6" fill="{CORAL}" stroke="{PAPER}" stroke-width="2" '
                 f'class="f-coral f-ring"><title>d = {d}, unscaled: {un:.1f} keys in play</title></circle>')
        b.append(f'<circle cx="{lx(sc):.1f}" cy="{y}" r="6" fill="{TEAL}" stroke="{PAPER}" stroke-width="2" '
                 f'class="f-teal f-ring"><title>d = {d}, divided by √d: {sc:.1f} keys in play</title></circle>')
        b.append(text(lx(un) - 10, y + 4, f"{un:.1f}", 12, anchor="end", weight=700, family=MONO))
        b.append(text(lx(sc) + 10, y + 4, f"{sc:.1f}", 12, weight=700, family=MONO))
    b.append(text(x1 + 6, y0 + 4, "uniform", 10, "f-muted", MUTED))
    return svg("fig11-4-sqrt-d.svg", w, h, "Figure 11.4 · Why divide by √d",
               "Unscaled, the keys in play fall from 131.7 at d = 2 to 27.8 at d = 8 and 2.5 at d = 64. "
               "Divided by √d they stay near 160 at every width: 172.2, 161.8, 157.0.", b)


def heat_panel(b, x0, y0, M_, title, best_label):
    L, H = len(M_), len(M_[0])
    c = 14
    b.append(text(x0, y0 - 30, title, 13, weight=700))
    for hd in range(0, H, 5):
        b.append(text(x0 + hd * c + c / 2, y0 - 8, f"{hd}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text(x0 + H * c / 2, y0 + L * c + 18, "head", 10, "f-muted", MUTED, "middle"))
    bl, bh = max(((l, hh) for l in range(L) for hh in range(H)), key=lambda t: M_[t[0]][t[1]])
    for l in range(L):
        if x0 < 200:
            b.append(text(x0 - 8, y0 + l * c + c / 2 + 4, f"{l}", 10, "f-muted", MUTED, "end", family=MONO))
        for hh in range(H):
            v = M_[l][hh]
            b.append(rect(x0 + hh * c + 1, y0 + l * c + 1, c - 2, c - 2, CARD, "f-card", 2))
            b.append(rect(x0 + hh * c + 1, y0 + l * c + 1, c - 2, c - 2, HEAT, "f-heat", 2,
                          f' fill-opacity="{0.04 + 0.96 * min(v, 1):.3f}"', f"layer {l}, head {hh}: {v:.3f}"))
    b.append(f'<rect x="{x0 + bh * c - 1}" y="{y0 + bl * c - 1}" width="{c + 2}" height="{c + 2}" rx="3" fill="none" '
             f'stroke="{INK}" class="f-ink-stroke" stroke-width="2"/>')
    b.append(text(x0 + H * c + 8, y0 + bl * c + c / 2 + 4, best_label.format(l=bl, h=bh, v=M_[bl][bh]), 11, weight=700))
    return L * c


def fig_heads():
    r = json.loads((BOOK / "measurements" / "attention.json").read_text(encoding="utf-8"))
    prev, first = r["prev"], r["first_where_visible"]
    close(prev[9][14], 0.881, 5e-4, "L9.h14 previous-token")
    close(first[17][15], 0.860, 5e-4, "L17.h15 first position")
    w, h = 760, 400
    x0, x1, y0 = 64, 430, 92
    b = [text(20, 28, "What the trained model's 320 heads attend to", 15, weight=700, family=SERIF),
         text(20, 48, "AnuLM-Base-v2-400M · 64 held-out windows of 512 tokens · queries 64–511 · measured", 12, "f-muted", MUTED)]
    b.append(text(x0 - 46, y0 - 8, "layer", 10, "f-muted", MUTED))
    hgt = heat_panel(b, x0, y0, prev, "Weight on the previous token", "L{l}·h{h} {v:.2f}")
    heat_panel(b, x1, y0, first, "Weight on the first position, where visible", "L{l}·h{h} {v:.2f}")
    # window band
    c = 14
    b.append(line(x0 - 30, y0, x0 - 30, y0 + 10 * c, GOLD, "f-gold-stroke", 3))
    b.append(text(x0 - 36, y0 + 5 * c, "windowed", 10, "f-muted", MUTED, "middle",
                  extra=f' transform="rotate(-90 {x0 - 36} {y0 + 5 * c})"'))
    # scale
    sy = y0 + hgt + 34
    b.append(text(x0, sy + 4, "0", 10, "f-muted", MUTED, "end", family=MONO))
    for i in range(10):
        b.append(rect(x0 + 6 + i * 16, sy - 6, 15, 12, CARD, "f-card", 1))
        b.append(rect(x0 + 6 + i * 16, sy - 6, 15, 12, HEAT, "f-heat", 1, f' fill-opacity="{0.04 + 0.96 * (i + .5) / 10:.3f}"'))
    b.append(text(x0 + 6 + 160 + 6, sy + 4, "1   mean attention weight", 10, "f-muted", MUTED, family=MONO))
    return svg("fig11-5-trained-heads.svg", w, h, "Figure 11.5 · What the trained heads attend to",
               "Two heatmaps of 20 layers by 16 heads. Left: weight on the previous token; a few heads in layers 3–5, 9 "
               "and 14 are bright, the strongest layer 9 head 14 at 0.88. Right: weight on the first position, over "
               "queries that can see it; layers 0–7 are dark, layers 8–18 bright, the strongest layer 17 head 15 at 0.86.", b)


# --------------------------------------------------------------------------- Ch 20
def fig_anatomy():
    w, h = 760, 620
    b = [text(20, 28, "AnuLM-Base-v2-400M, one forward pass", 15, weight=700, family=SERIF),
         text(20, 48, "the residual stream runs down the left; every sublayer reads it and adds to it", 12, "f-muted", MUTED)]
    rail = 132

    def box(x, y, ww, hh, title, sub, cls="f-card", fill=CARD, accent=False):
        out = [rect(x, y, ww, hh, fill, cls, 8,
                    f' stroke="{GOLD}" stroke-width="2.5"' if accent else f' stroke="{RULE}" stroke-width="1"')]
        out.append(text(x + 14, y + 22, title, 13, weight=700))
        if sub:
            out.append(text(x + 14, y + 40, sub, 11.5, "f-muted", MUTED, family=MONO))
        return out

    def plus(y):
        return [f'<circle cx="{rail}" cy="{y}" r="9" fill="{PAPER}" stroke="{INK}" stroke-width="1.5" class="f-paper f-ink-stroke"/>',
                line(rail - 5, y, rail + 5, y, INK, "f-ink-stroke", 1.5), line(rail, y - 5, rail, y + 5, INK, "f-ink-stroke", 1.5)]

    b += box(60, 66, 144, 46, "token ids", "(B, S)")
    b.append(line(rail, 112, rail, 128, INK, "f-ink-stroke", 2, ' marker-end="url(#arrow)"'))
    b += box(60, 130, 144, 46, "embedding", "V × D table")
    b.append(line(rail, 176, rail, 556, INK, "f-ink-stroke", 4))
    b.append(text(rail - 14, 300, "residual stream (B, S, D)", 11, "f-muted", MUTED, "middle", family=MONO,
                  extra=f' transform="rotate(-90 {rail - 14} 300)"'))

    def sub(y, hh, title, subtitle, accent=False, ww=300):
        out = box(300, y, ww, hh, title, subtitle, accent=accent)
        cy = y + 26
        out.append(line(rail + 9, cy, 298, cy, MUTED, "f-muted-stroke", 1.2, ' marker-end="url(#arrow)"'))
        out += plus(cy)
        return out

    # layer 0: attention, then a dense SwiGLU, stacked
    b.append(rect(176, 194, 560, 134, CARD, "f-card", 10, ' fill-opacity="0.55"'))
    b.append(text(190, 214, "layer 0", 12, "f-muted", MUTED, weight=700))
    b += sub(206, 52, "RMSNorm → attention", "16 query heads share 4 key/value · window 256")
    b += sub(266, 52, "RMSNorm → SwiGLU", "dense, inner width I = 2,816")
    # layers 1-19, drawn as a stack
    for k in (2, 1):
        b.append(rect(176 + 6 * k, 346 - 6 * k, 560, 170, CARD, "f-card", 10, ' fill-opacity="0.35"'))
    b.append(rect(176, 346, 560, 170, CARD, "f-card", 10, ' fill-opacity="0.55"'))
    b.append(text(190, 366, "layers 1–19", 12, "f-muted", MUTED, weight=700))
    b.append(text(190, 382, "(× 19)", 11, "f-muted", MUTED))
    b += sub(358, 52, "RMSNorm → attention", "same shape · window on layers 1–9")
    b += sub(420, 84, "RMSNorm → router → experts", "4 of 24 experts per token", accent=True, ww=420)
    for i in range(24):
        cx = 314 + (i % 12) * 17
        cy = 482 + (i // 12) * 12
        on = i in (3, 9, 14, 20)
        b.append(rect(cx, cy - 8, 13, 9, GOLD if on else RULE, "f-gold" if on else "f-rule-fill", 2,
                      tip="an expert chosen for this token" if on else "an idle expert"))
    b.append(text(530, 488, "gold: the 4 chosen", 11, "f-muted", MUTED))
    # rotary tables feed every attention
    b.append(rect(560, 66, 176, 46, CARD, "f-card", 8, f' stroke="{RULE}" stroke-dasharray="4 3"'))
    b.append(text(574, 88, "rotary cos, sin", 13, weight=700))
    b.append(text(574, 104, "(S, Dh), cached", 11.5, "f-muted", MUTED, family=MONO))
    b.append(line(648, 112, 560, 204, MUTED, "f-muted-stroke", 1, ' stroke-dasharray="4 3" marker-end="url(#arrow)"'))
    # out
    b += box(60, 558, 144, 46, "final RMSNorm", None)
    b.append(line(204, 581, 268, 581, INK, "f-ink-stroke", 2, ' marker-end="url(#arrow)"'))
    b += box(270, 558, 180, 46, "lm_head", "→ (B, S, V) logits")
    b.append(line(450, 581, 504, 581, INK, "f-ink-stroke", 2, ' marker-end="url(#arrow)"'))
    b += box(506, 558, 230, 46, "loss, or the next token", "softmax · cross-entropy")
    return svg("fig20-1-anatomy.svg", w, h, "Figure 20.1 · The whole model",
               "Token ids become a (B, S, D) residual stream through the embedding table. Layer 0 adds attention and a "
               "dense SwiGLU; layers 1 to 19 add attention and a mixture of experts whose router picks 4 of 24. A final "
               "RMSNorm and lm_head produce (B, S, V) logits for the loss or for sampling.", b)


def fig_params():
    cfg = released()
    D, V, L, H, Hkv, Dh = cfg.hidden_size, cfg.vocab_size, cfg.n_layer, cfg.n_head, cfg.n_kv_head, cfg.head_dim
    moe = L - cfg.first_k_dense_replace
    terms = [("routed experts", moe * cfg.num_experts * 3 * D * cfg.moe_intermediate_size),
             ("embedding + output head", 2 * V * D),
             ("attention", L * ((H + 2 * Hkv) * Dh * D + H * Dh * D)),
             ("dense layer 0", 3 * D * cfg.intermediate_size),
             ("routers", moe * cfg.num_experts * D),
             ("RMSNorm gains", 2 * L * D + D),
             ("QK-norm gains", L * 2 * Dh)]
    total = sum(v for _, v in terms)
    tot, act = M.AnuLM(cfg).num_params()
    assert total == tot == 397_659_648 and act == 173_526_528, (total, tot, act)
    idle = moe * (cfg.num_experts - cfg.num_experts_per_tok) * 3 * D * cfg.moe_intermediate_size
    w, h = 760, 420
    x0, x1 = 220, 640
    sx = lambda v: (x1 - x0) * v / 270e6
    b = [text(20, 28, "Where the 397.7M parameters are", 15, weight=700, family=SERIF),
         text(20, 48, "AnuLM-Base-v2-400M · computed from the configuration · verified against num_params()", 12, "f-muted", MUTED)]
    for t in (0, 50e6, 100e6, 150e6, 200e6, 250e6):
        b.append(line(x0 + sx(t), 70, x0 + sx(t), 70 + 7 * 30))
        b.append(text(x0 + sx(t), 70 + 7 * 30 + 16, f"{t / 1e6:.0f}M", 10, "f-muted", MUTED, "middle", family=MONO))
    for i, (name, v) in enumerate(terms):
        y = 78 + i * 30
        b.append(text(x0 - 12, y + 13, name, 12.5, anchor="end"))
        bw = max(sx(v), 2)
        b.append(f'<path d="M{x0},{y} h{bw - 4:.1f} a4,4 0 0 1 4,4 v10 a4,4 0 0 1 -4,4 h{-(bw - 4):.1f} z" '
                 f'fill="{GOLD}" class="f-gold"><title>{esc(name)}: {v:,}</title></path>')
        label = f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v:,}"
        b.append(text(x0 + bw + 8, y + 13, f"{label}  {100 * v / total:.1f}%" if v >= 1e6 else label, 12, weight=700, family=MONO))
    # per-token split
    y = 330
    b.append(text(20, y - 12, "For one token", 13, weight=700))
    used = act
    sx2 = lambda v: (x1 - 20) * v / total
    b.append(rect(20, y, sx2(used) - 1, 22, GOLD, "f-gold", 4, tip=f"active by num_params(): {used:,}"))
    b.append(rect(20 + sx2(used) + 1, y, sx2(idle) - 1, 22, RULE, "f-rule-fill", 4, tip=f"idle experts: {idle:,}"))
    b.append(text(20 + sx2(used) / 2, y + 15, f"active 173.5M (43.6%)", 12, "f-onfill", NIGHT, "middle", 700))
    b.append(text(20 + sx2(used) + sx2(idle) / 2, y + 15, "idle experts 224.1M", 12, anchor="middle", weight=700))
    b.append(text(20, y + 44, "Of the active 173.5M, 33.6M is the input embedding table, of which a token reads one row:",
                  11.5, "f-muted", MUTED))
    b.append(text(20, y + 60, "about 140M weights are actually multiplied per token.", 11.5, "f-muted", MUTED))
    return svg("fig20-2-parameters.svg", w, h, "Figure 20.2 · Where the parameters are",
               "Routed experts 269.0M (67.6%), embedding and output head 67.1M, attention 52.4M, dense layer 0 8.7M, "
               "routers 0.47M, norms under 0.05M. Per token, 173.5M are active by num_params() and 224.1M of expert "
               "weights sit idle.", b)


def fig_loads():
    r = json.loads((BOOK / "measurements" / "attention.json").read_text(encoding="utf-8"))
    loads = {int(k): v for k, v in r["expert_load"].items()}
    close(max(loads[2]), 0.084, 5e-4, "layer 2 busiest")
    close(max(loads[3]), 0.095, 5e-4, "layer 3 busiest")
    w, h = 760, 520
    x0, x1, y0, rh = 110, 700, 84, 21
    sx = lambda v: x0 + (x1 - x0) * v / 0.10
    b = [text(20, 28, "How each MoE layer spreads its tokens over 24 experts", 15, weight=700, family=SERIF),
         text(20, 48, "share of routing slots per expert · 64 held-out windows of 512 tokens · measured", 12, "f-muted", MUTED)]
    for t in range(0, 11, 2):
        b.append(line(sx(t / 100), y0 - 8, sx(t / 100), y0 + 19 * rh))
        b.append(text(sx(t / 100), y0 + 19 * rh + 16, f"{t}%", 10, "f-muted", MUTED, "middle", family=MONO))
    even = 1 / 24
    b.append(line(sx(even), y0 - 14, sx(even), y0 + 19 * rh, INK, "f-ink-stroke", 1.5, ' stroke-dasharray="3 3"'))
    b.append(text(sx(even) + 6, y0 - 16, "even split, 4.2%", 11, weight=700))
    for i, l in enumerate(sorted(loads)):
        y = y0 + i * rh + rh / 2
        b.append(text(x0 - 12, y + 4, f"layer {l}", 11, "f-muted", MUTED, "end", family=MONO))
        for e, v in enumerate(loads[l]):
            b.append(f'<circle cx="{sx(v):.1f}" cy="{y:.1f}" r="4.5" fill="{GOLD}" fill-opacity="0.85" stroke="{PAPER}" '
                     f'stroke-width="1.5" class="f-gold f-ring"><title>layer {l}, expert {e}: {100 * v:.2f}%</title></circle>')
    return svg("fig20-3-expert-loads.svg", w, h, "Figure 20.3 · Expert loads",
               "For each of the 19 MoE layers, 24 dots show each expert's share of routing slots. Most layers cluster "
               "between 2.4% and 6.6% around the even 4.2%; layers 2 and 3 spread from 0.7% to 9.5%.", b)


def part_modules():
    """Each part's figures live in tools/figures/<part>.py, written by that part's author; every function
    named fig_* there is a figure. They import the helpers above from this module."""
    import importlib.util
    for path in sorted((Path(__file__).parent / "figures").glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"figures_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield path.stem, [getattr(mod, n) for n in sorted(dir(mod)) if n.startswith("fig_") and callable(getattr(mod, n))]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    only = sys.argv[1:]                      # e.g. part_v: just that module's figures
    if not only or "core" in only:
        for f in (fig_flow, fig_lookup, fig_grid, fig_sqrt_d, fig_heads, fig_anatomy, fig_params, fig_loads):
            print("wrote", OUT / f())
    for name, funcs in part_modules():
        if only and name not in only:
            continue
        for f in funcs:
            print("wrote", OUT / f())
