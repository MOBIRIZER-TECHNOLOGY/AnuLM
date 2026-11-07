"""
Figures for Part IV, second half: Chapters 16 (SwiGLU), 17 (the router), 18 (dispatch and
balance) and 19 (latent attention).

    python book/tools/make_figures.py part_iv_b

Every number is computed here from its source and asserted against the value the chapter
prints: measured numbers from book/measurements/part_iv_b_trained.json (written by
code-examples/ch16_measure_trained.py), arithmetic from model.py's own configurations.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from make_figures import (BOOK, CARD, CORAL, GOLD, GOLDTEXT, INK, MONO, MUTED, NIGHT, PAPER, RULE, SERIF,  # noqa: E402
                          SLATE, TEAL, M, close, esc, line, rect, svg, text, released)

MEAS = BOOK / "measurements" / "part_iv_b_trained.json"


def _lines(obj) -> str:
    """'first–last' line of a class in the model code, read from the source, so a
    caption cannot fall behind the code it names."""
    import inspect
    src, first = inspect.getsourcelines(obj)
    return f"{first}–{first + len(src) - 1}"


def _meas():
    return json.loads(MEAS.read_text(encoding="utf-8"))


def _box(x, y, w, h, title, sub=None, accent=False, dashed=False):
    stroke = f' stroke="{GOLD}" stroke-width="2.5"' if accent else f' stroke="{RULE}" stroke-width="1"'
    if dashed:
        stroke += ' stroke-dasharray="5 4"'
    out = [rect(x, y, w, h, CARD, "f-card", 8, stroke)]
    ty = y + (h / 2 + 5 if sub is None else h / 2 - 3)
    out.append(text(x + w / 2, ty, title, 13, anchor="middle", weight=700))
    if sub:
        out.append(text(x + w / 2, ty + 17, sub, 11, "f-muted", MUTED, "middle", family=MONO))
    return out


def _arrow(x1, y1, x2, y2, label=None, lx=None, ly=None, anchor="middle"):
    out = [line(x1, y1, x2, y2, MUTED, "f-muted-stroke", 1.5, ' marker-end="url(#arrow)"')]
    if label:
        out.append(text(lx if lx is not None else (x1 + x2) / 2, ly if ly is not None else (y1 + y2) / 2 - 6,
                        label, 10.5, "f-muted", MUTED, anchor, family=MONO))
    return out


# --------------------------------------------------------------------------- Ch 16
def fig_16_swiglu():
    cfg = released()
    D, I = cfg.hidden_size, cfg.intermediate_size
    assert (D, I) == (1024, 2816)
    w, h = 760, 300
    b = [text(20, 28, "SwiGLU: two projections, one gate, one projection back", 15, weight=700, family=SERIF),
         text(20, 48, f"model.py lines {_lines(M.SwiGLU)} · shapes for the released layer 0: D = {D:,}, I = {I:,}", 12,
              "f-muted", MUTED)]
    b += _box(20, 130, 110, 54, "u", "(B, S, D)")
    b += _box(190, 76, 130, 54, "gate_proj", "D → I")
    b += _box(190, 186, 130, 54, "up_proj", "D → I")
    b += _box(380, 76, 110, 54, "SiLU", "g · sigmoid(g)", accent=True)
    b += _arrow(130, 150, 188, 104)
    b += _arrow(130, 164, 188, 212)
    b += _arrow(320, 103, 378, 103, "g (B,S,I)", ly=95)
    # product
    cx, cy = 560, 157
    b.append(f'<circle cx="{cx}" cy="{cy}" r="17" fill="{CARD}" stroke="{INK}" stroke-width="1.5" class="f-card f-ink-stroke"/>')
    b.append(text(cx, cy + 6, "⊙", 18, anchor="middle", weight=700))
    b += _arrow(490, 110, cx - 13, cy - 11, "gates", lx=530, ly=118)
    b += _arrow(320, 213, cx - 13, cy + 11, "values (B, S, I)", lx=430, ly=228)
    b += _box(612, 130, 128, 54, "down_proj", "I → D")
    b += _arrow(cx + 17, cy, 610, cy)
    b.append(text(676, 212, "F(u)  (B, S, D)", 12, anchor="middle", weight=700, family=MONO))
    b += _arrow(676, 184, 676, 198)
    b.append(text(20, 276, f"Three bias-free matrices: 3 · D · I = 3 · {D:,} · {I:,} = {3 * D * I:,} parameters. "
                           "Each expert is the same module with I = 192.", 12, "f-muted", MUTED))
    return svg("fig16-1-swiglu.svg", w, h, "Figure 16.1 · The SwiGLU, one token",
               "The normalised token u is projected twice from D to I. One projection passes through SiLU to become "
               "the gates; the gates multiply the other projection element by element, and down_proj maps the "
               "product back to width D.", b)


def fig_16_silu():
    xs = torch.linspace(-6, 4, 401, dtype=torch.float64)
    ys = F.silu(xs)
    i = ys.argmin()
    close(ys[i].item(), -0.2785, 5e-4, "SiLU minimum")
    close(xs[i].item(), -1.2785, 0.03, "SiLU argmin")
    w, h = 640, 350
    x0, x1, y0, y1 = 80, 600, 290, 70          # data box, y0 bottom
    X = lambda v: x0 + (x1 - x0) * (v + 6) / 10
    Y = lambda v: y0 - (y0 - y1) * (v + 0.5) / 4.5
    b = [text(20, 28, "SiLU(g) = g · sigmoid(g)", 15, weight=700, family=SERIF),
         text(20, 48, "the gate a SwiGLU channel applies to its value · verified with F.silu", 12, "f-muted", MUTED)]
    for t in range(-6, 5, 2):
        b.append(line(X(t), y1, X(t), y0))
        b.append(text(X(t), y0 + 16, f"{t}", 10, "f-muted", MUTED, "middle", family=MONO))
    for t in (0, 1, 2, 3, 4):
        b.append(line(x0, Y(t), x1, Y(t)))
        b.append(text(x0 - 8, Y(t) + 4, f"{t}", 10, "f-muted", MUTED, "end", family=MONO))
    b.append(line(x0, Y(0), x1, Y(0), INK, "f-ink-stroke", 1.2))
    b.append(line(X(0), y1, X(0), y0, INK, "f-ink-stroke", 1.2))
    b.append(line(X(-0.5), Y(-0.5), X(4), Y(4), MUTED, "f-muted-stroke", 1, ' stroke-dasharray="4 4"'))
    b.append(text(X(3.0) - 10, Y(3.6), "y = g", 10.5, "f-muted", MUTED, "end"))
    pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs.tolist(), ys.tolist()))
    b.append(f'<polyline points="{pts}" fill="none" stroke="{GOLD}" stroke-width="3" class="f-gold-stroke">'
             f'<title>SiLU(g) for g from -6 to 4</title></polyline>')
    fine = torch.linspace(-6, 0, 600_001, dtype=torch.float64)       # the example's grid, for the marked point
    j = F.silu(fine).argmin()
    mx, my = fine[j].item(), F.silu(fine[j]).item()
    close(mx, -1.2785, 5e-4, "SiLU argmin, fine grid")
    b.append(f'<circle cx="{X(mx):.1f}" cy="{Y(my):.1f}" r="5" fill="{INK}" class="f-ink-fill">'
             f'<title>minimum {my:.4f} at g = {mx:.4f}</title></circle>')
    b.append(line(X(mx) - 4, Y(my) - 5, X(-2.6), Y(0.9) + 4, MUTED, "f-muted-stroke", 1))
    b.append(text(X(-2.6), Y(0.9), f"minimum {my:.4f} at g = {mx:.4f}", 11.5, anchor="middle", weight=700))
    b.append(text(X(-5), Y(0.35), "gate nearly shut", 11, "f-muted", MUTED, "middle"))
    b.append(text(X(2.2), Y(1.1), "gate open, ≈ g", 11, "f-muted", MUTED, "start"))
    b.append(text((x0 + x1) / 2, y0 + 36, "gate input g", 11, "f-muted", MUTED, "middle"))
    return svg("fig16-2-silu.svg", w, h, "Figure 16.2 · SiLU, the gate function",
               "SiLU is close to zero for very negative g, dips to a minimum of −0.2785 at g = −1.2785, crosses zero "
               "at g = 0 and approaches the line y = g for large g.", b)


def fig_16_channels():
    s = _meas()["layer0_swiglu"]
    c = s["cumulative_share_top_n"]
    close(c[0], 0.596, 5e-4, "largest channel share")
    close(s["channels_for_90pct"], 53.4, 0.05, "channels for 90%")
    close(s["tokens_whose_largest_is_top1"], 0.022, 5e-4, "most-used channel")
    w, h = 680, 340
    x0, x1, y0, y1 = 80, 640, 270, 70
    X = lambda n: x0 + (x1 - x0) * math.log2(n) / 8                # 1 … 256
    Y = lambda v: y0 - (y0 - y1) * v
    b = [text(20, 28, "Layer 0: how few of 2,816 channels carry a token", 15, weight=700, family=SERIF),
         text(20, 48, "mean share of a token's Σh² in its largest n channels · 32,768 held-out tokens · measured", 12,
              "f-muted", MUTED)]
    for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        b.append(line(X(n), y1, X(n), y0))
        b.append(text(X(n), y0 + 16, f"{n}", 10, "f-muted", MUTED, "middle", family=MONO))
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        b.append(line(x0, Y(v), x1, Y(v)))
        b.append(text(x0 - 8, Y(v) + 4, f"{int(v * 100)}%", 10, "f-muted", MUTED, "end", family=MONO))
    pts = " ".join(f"{X(n):.1f},{Y(c[n - 1]):.1f}" for n in range(1, 257))
    b.append(f'<polyline points="{pts}" fill="none" stroke="{GOLD}" stroke-width="3" class="f-gold-stroke">'
             f'<title>share of sum h squared in the largest n channels</title></polyline>')
    for n, dx, dy in ((1, 10, 18), (10, -8, -14), (100, -8, -14)):
        v = c[n - 1]
        b.append(f'<circle cx="{X(n):.1f}" cy="{Y(v):.1f}" r="5" fill="{INK}" class="f-ink-fill">'
                 f'<title>n = {n}: {v:.3f}</title></circle>')
        b.append(text(X(n) + dx, Y(v) + dy, f"n = {n}: {100 * v:.1f}%", 11.5,
                      anchor="start" if dx > 0 else "end", weight=700))
    b.append(text((x0 + x1) / 2, y0 + 34, "n, a token's largest channels (log scale)", 11, "f-muted", MUTED, "middle"))
    b.append(text(x1, y0 - 10, "the most-used channel is a token's largest for only "
                  f"{100 * s['tokens_whose_largest_is_top1']:.1f}% of tokens", 11, "f-muted", MUTED, "end"))
    return svg("fig16-3-channels.svg", w, h, "Figure 16.3 · How few channels carry a token",
               "On held-out text, a token's largest layer-0 channel holds 59.6% of its sum of squared inner values on "
               "average, its largest 10 hold 83.5% and its largest 100 hold 93.9%.", b)


# --------------------------------------------------------------------------- Ch 17
def _cell(x, y, w, h, value, fill=CARD, cls="f-card", stroke=None, tip=None, bold=False, color=INK, tcls="f-ink"):
    extra = f' stroke="{stroke}" stroke-width="2.5"' if stroke else ""
    out = [rect(x, y, w, h, fill, cls, 5, extra, tip)]
    out.append(text(x + w / 2, y + h / 2 + 5, value, 13, tcls, color, "middle", 700 if bold else 400, MONO))
    return out


def fig_17_router():
    logits = torch.tensor([0.0, 1.0, -1.0, 0.5, 2.0, -0.5])
    bias = torch.tensor([0.0, -0.2, 0.0, 0.15, 0.0, 0.0])
    s = torch.sigmoid(logits)
    choice = (s + bias).topk(2).indices
    w = s[choice] / s[choice].sum() * 2.5
    assert choice.tolist() == [4, 3]
    close(w[0].item(), 1.465, 5e-4, "w expert 4")
    close(w[1].item(), 1.035, 5e-4, "w expert 3")
    weights = {int(e): float(v) for e, v in zip(choice, w)}
    W, H = 760, 420
    x0, cw, ch, gap = 250, 70, 34, 8
    X = lambda i: x0 + i * (cw + gap)
    b = [text(20, 28, "The router, one token: the bias chooses, the unbiased scores weight", 15, weight=700,
              family=SERIF),
         text(20, 48, "E = 6 experts, A = 2 chosen · the numbers of ch17_router.py · illustrative", 12, "f-muted", MUTED)]
    for i in range(6):
        b.append(text(X(i) + cw / 2, 82, f"expert {i}", 11, "f-muted", MUTED, "middle", 700))
    rows = [("logit  u · rᵢ", [f"{v:.1f}" for v in logits.tolist()], 92),
            ("score  sᵢ = sigmoid", [f"{v:.3f}" for v in s.tolist()], 152),
            ("for choosing  sᵢ + bᵢ", [f"{v:.3f}" for v in (s + bias).tolist()], 232),
            ("weight  2.5 · sᵢ / Σ chosen", [f"{weights[i]:.3f}" if i in weights else "·" for i in range(6)], 330)]
    for label, vals, y in rows:
        b.append(text(x0 - 16, y + ch / 2 + 5, label, 12.5, anchor="end", weight=700))
        for i, v in enumerate(vals):
            chosen = i in weights and y >= 232
            b += _cell(X(i), y, cw, ch, v, GOLD if chosen else CARD, "f-gold" if chosen else "f-card",
                       tip=f"expert {i}: {label.split('  ')[0]} {v}", bold=chosen,
                       color=NIGHT if chosen else INK, tcls="f-onfill" if chosen else "f-ink")
    for i, bv in enumerate(bias.tolist()):
        if bv:
            b.append(text(X(i) + cw / 2, 220, f"bias {bv:+.2f}", 11, anchor="middle", weight=700, family=MONO))
    b.append(text(x0 - 16, 300, "top-2 of the biased row", 11.5, "f-muted", MUTED, "end"))
    for i in weights:
        b.append(line(X(i) + cw / 2, 268, X(i) + cw / 2, 326, MUTED, "f-muted-stroke", 1.5, ' marker-end="url(#arrow)"'))
    b.append(text(X(1) + cw / 2, 300, "0.731 − 0.2: loses", 11, "f-muted", MUTED, "middle"))
    b.append(text(20, 396, "The weights come from the unbiased scores 0.881 and 0.622 (row 2), normalised to "
                           "0.586 and 0.414, then × 2.5; they sum to 2.5.", 12, "f-muted", MUTED))
    return svg("fig17-1-router.svg", W, H, "Figure 17.1 · The router, one token",
               "Six experts. Logits 0, 1, −1, 0.5, 2, −0.5 become sigmoid scores; a bias of −0.2 on expert 1 and "
               "+0.15 on expert 3 makes experts 4 and 3 the top two; their unbiased scores give weights 1.465 and "
               "1.035.", b)


def fig_17_groups():
    logits = torch.tensor([3.0, -3., -3., -3., 1.0, 1.0, -3., -3., 0.8, 0.8, -3., -3., -.5, -.5, -3., -3.])
    s = torch.sigmoid(logits)
    g = s.view(4, 4).topk(2, dim=-1).values.sum(-1)
    keep = g.topk(2).indices.tolist()
    allowed = torch.tensor([i // 4 in keep for i in range(16)])
    chosen = s.masked_fill(~allowed, float("-inf")).topk(2).indices.tolist()
    assert sorted(keep) == [1, 2] and sorted(chosen) == [4, 5] and s.argmax().item() == 0
    close(g[0].item(), 1.000, 5e-4, "group 0 score")
    close(g[1].item(), 1.462, 5e-4, "group 1 score")
    W, H = 760, 300
    cw, ch, gx = 38, 40, 186
    b = [text(20, 28, "Group-limited top-k: 16 experts in 4 groups, 2 groups kept, 2 experts chosen", 15,
              weight=700, family=SERIF),
         text(20, 48, "a group is worth its best two scores · the example of ch17_router.py · illustrative", 12,
              "f-muted", MUTED)]
    for gi in range(4):
        gx0 = 26 + gi * gx
        kept = gi in keep
        b.append(rect(gx0 - 8, 70, 4 * cw + 3 * 4 + 16, 136, PAPER if kept else "url(#hatch)",
                      "f-paper" if kept else "", 8,
                      f' stroke="{GOLD if kept else RULE}" stroke-width="{2.5 if kept else 1}"',
                      tip=f"group {gi}: worth {g[gi]:.3f}, {'kept' if kept else 'masked'}"))
        b.append(text(gx0, 92, f"group {gi}", 12, weight=700))
        b.append(text(gx0 + 4 * cw + 12, 92, "kept" if kept else "masked", 11, "f-muted", MUTED, "end", 700))
        for j in range(4):
            e = gi * 4 + j
            on = e in chosen
            b += _cell(gx0 + j * (cw + 4), 104, cw, ch, f"{s[e]:.2f}", GOLD if on else CARD,
                       "f-gold" if on else "f-card", tip=f"expert {e}: score {s[e]:.3f}", bold=on,
                       color=NIGHT if on else INK, tcls="f-onfill" if on else "f-ink")
            b.append(text(gx0 + j * (cw + 4) + cw / 2, 160, f"{e}", 10, "f-muted", MUTED, "middle", family=MONO))
        b.append(text(gx0, 192, f"worth {g[gi]:.3f}", 12, weight=700, family=MONO))
    b.append(text(26, 244, "Expert 0 has the best score of all 16, 0.953, but its group's second best is 0.047,",
                  12, "f-muted", MUTED))
    b.append(text(26, 262, "so group 0 is worth 1.000 and loses to groups 1 (1.462) and 2 (1.380). "
                           "Chosen: experts 4 and 5.", 12, "f-muted", MUTED))
    return svg("fig17-2-groups.svg", W, H, "Figure 17.2 · Group-limited routing",
               "Four groups of four sigmoid scores. Groups are worth 1.000, 1.462, 1.380 and 0.755; groups 1 and 2 are "
               "kept, so experts 4 and 5 are chosen and expert 0, the best single score, is masked.", b)


def fig_17_scores():
    r = _meas()["routers"]
    a, c = r["3"]["score_by_rank"], r["10"]["score_by_rank"]
    close(a[0], 0.283, 5e-4, "layer 3 top score")
    close(c[0], 0.162, 5e-4, "layer 10 top score")
    close(r["3"]["w_max"], 1.735, 5e-4, "layer 3 largest weight")
    W, H = 720, 340
    x0, x1, y0, y1 = 80, 680, 270, 80
    X = lambda k: x0 + (x1 - x0) * (k - 1) / 23
    Y = lambda v: y0 - (y0 - y1) * v / 0.3
    b = [text(20, 28, "Trained router scores, sorted, mean over 32,768 held-out tokens", 15, weight=700, family=SERIF),
         text(20, 48, "AnuLM-Base-v2-400M · sigmoid score of each token's 1st to 24th expert · measured", 12,
              "f-muted", MUTED)]
    b.append(rect(X(1) - 10, y1, X(4) - X(1) + 20, y0 - y1, CARD, "f-card", 4))
    b.append(text((X(1) + X(4)) / 2, y0 - 8, "chosen (A = 4)", 11, "f-muted", MUTED, "middle", 700))
    for t in (0, 0.1, 0.2, 0.3):
        b.append(line(x0 - 10, Y(t), x1, Y(t)))
        b.append(text(x0 - 16, Y(t) + 4, f"{t:.1f}", 10, "f-muted", MUTED, "end", family=MONO))
    for k in (1, 4, 8, 12, 16, 20, 24):
        b.append(text(X(k), y0 + 16, f"{k}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y0 + 34, "rank of the expert for the token", 11, "f-muted", MUTED, "middle"))
    for vals, col, cls, name in ((a, CORAL, "f-coral", "layer 3"), (c, TEAL, "f-teal", "layer 10")):
        pts = " ".join(f"{X(k + 1):.1f},{Y(v):.1f}" for k, v in enumerate(vals))
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5" class="{cls}-stroke"/>')
        for k, v in enumerate(vals):
            b.append(f'<circle cx="{X(k + 1):.1f}" cy="{Y(v):.1f}" r="3.5" fill="{col}" class="{cls}">'
                     f'<title>{name}, rank {k + 1}: {v:.3f}</title></circle>')
    b.append(text(X(2) + 4, Y(a[0]) + 4, f"layer 3: {a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}, {a[3]:.3f}, then near 0",
                  12, weight=700))
    b.append(text(X(5) + 10, Y(c[3]) - 26, f"layer 10: {c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, {c[3]:.3f}, then slowly down",
                  12, weight=700))
    b.append(f'<circle cx="{x1 - 170}" cy="{y1 - 24}" r="5" fill="{CORAL}" class="f-coral"/>')
    b.append(text(x1 - 160, y1 - 20, "layer 3", 12))
    b.append(f'<circle cx="{x1 - 90}" cy="{y1 - 24}" r="5" fill="{TEAL}" class="f-teal"/>')
    b.append(text(x1 - 80, y1 - 20, "layer 10", 12))
    return svg("fig17-3-scores.svg", W, H, "Figure 17.3 · Trained router scores by rank",
               f"Layer 3's sorted mean scores fall from {a[0]:.3f} to {a[3]:.3f} at rank 4 and near zero after; layer "
               f"10's fall from {c[0]:.3f} to {c[3]:.3f} and then slowly. All scores are far below 0.5.", b)


# --------------------------------------------------------------------------- Ch 18
def fig_18_dispatch():
    idx = torch.tensor([[0, 2], [2, 3], [0, 1], [2, 0], [3, 2]])
    w = torch.tensor([[1.5, 1.0], [2.0, 0.5], [1.25, 1.25], [1.0, 1.5], [0.5, 2.0]])
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=4)
    offs = counts.cumsum(0).tolist()
    assert offs == [3, 4, 8, 10] and counts.tolist() == [3, 1, 4, 2]
    HL = 2                                                   # the expert followed through all three panels
    W, H = 760, 560
    b = [text(20, 28, "Three ways to dispatch the same ten slots", 15, weight=700, family=SERIF),
         text(20, 48, "five tokens, A = 2, E = 4 · the example of ch18_dispatch_balance.py · expert 2 in gold · "
                      "illustrative", 12, "f-muted", MUTED)]

    def chip(x, y, label, on, wdt=54):
        return _cell(x, y, wdt, 26, label, GOLD if on else CARD, "f-gold" if on else "f-card",
                     tip=label, bold=on, color=NIGHT if on else INK, tcls="f-onfill" if on else "f-ink")

    # sparse
    y = 74
    b.append(text(20, y + 12, "sparse", 13, weight=700))
    b.append(text(20, y + 28, "nonzero per expert", 11, "f-muted", MUTED))
    b.append(text(20, y + 42, "one GEMM each", 11, "f-muted", MUTED))
    for e in range(4):
        toks = [n for n in range(5) for s in range(2) if idx[n, s] == e]
        yy = y + e * 32
        b.append(text(170, yy + 18, f"expert {e}", 12, weight=700, family=MONO))
        for j, n in enumerate(toks):
            b += chip(250 + j * 60, yy, f"t{n}", e == HL)
        b.append(text(250 + 4 * 60 + 10, yy + 18, f"→ ({len(toks)}, D) through expert {e}", 11.5, "f-muted", MUTED,
                      family=MONO))
    # dense
    y = 224
    b.append(line(20, y - 10, 740, y - 10))
    b.append(text(20, y + 12, "dense", 13, weight=700))
    b.append(text(20, y + 28, "every expert, every token", 11, "f-muted", MUTED))
    b.append(text(20, y + 42, "× an (N, E) weight matrix", 11, "f-muted", MUTED))
    dense = torch.zeros(5, 4).scatter_(1, idx, w)
    for e in range(4):
        b.append(text(250 + e * 64 + 27, y + 14, f"e{e}", 11, "f-muted", MUTED, "middle", 700))
    for n in range(5):
        b.append(text(236, y + 22 + n * 28 + 18, f"t{n}", 11, "f-muted", MUTED, "end", 700, MONO))
        for e in range(4):
            v = dense[n, e].item()
            b += chip(250 + e * 64, y + 22 + n * 28, f"{v:g}" if v else "0", e == HL and v > 0)
    b.append(text(520, y + 60, "each expert runs on all 5 tokens;", 11.5, "f-muted", MUTED))
    b.append(text(520, y + 78, "a 0 weight removes its output", 11.5, "f-muted", MUTED))
    b.append(text(520, y + 96, f"work: E/A = {4 // 2}× the sparse path's", 11.5, "f-muted", MUTED))
    # grouped
    y = 404
    b.append(line(20, y - 10, 740, y - 10))
    b.append(text(20, y + 12, "grouped", 13, weight=700))
    b.append(text(20, y + 28, "sort slots by expert", 11, "f-muted", MUTED))
    b.append(text(20, y + 42, "one grouped_mm", 11, "f-muted", MUTED))
    b.append(text(20, y + 56, "per projection", 11, "f-muted", MUTED))
    for j, s_ in enumerate(order.tolist()):
        n, e = s_ // 2, flat[s_].item()
        b += chip(200 + j * 52, y + 16, f"t{n}·e{e}", e == HL, 48)
    for o in [0] + offs:
        x = 200 + o * 52 - 2
        b.append(line(x, y + 8, x, y + 50, INK, "f-ink-stroke", 1.5))
    for o in offs:
        b.append(text(200 + o * 52 - 2, y + 66, f"{o}", 11, anchor="middle", weight=700, family=MONO))
    b.append(text(200, y + 66, "offs:", 11, "f-muted", MUTED, "end"))
    b.append(text(200, y + 96, "Each run of rows meets its own expert's matrix in one kernel call; "
                               "shapes depend on N, A and E only.", 11.5, "f-muted", MUTED))
    return svg("fig18-1-dispatch.svg", W, H, "Figure 18.1 · Three ways to dispatch",
               "Sparse: each expert gathers its own tokens (expert 2: tokens 0, 1, 3, 4) and runs once. Dense: every "
               "expert runs on all five tokens and a weight matrix with zeros removes the unchosen outputs. Grouped: "
               "the ten slots are sorted by expert into runs ending at offsets 3, 4, 8 and 10, and one grouped matrix "
               "multiply runs them all.", b)


def fig_18_imbalance():
    r = _meas()["routers"]
    layers = sorted(r, key=int)
    on = [r[l]["imbalance_stored"] for l in layers]
    off = [r[l]["imbalance_no_bias"] for l in layers]
    close(r["3"]["imbalance_stored"], 2.29, 5e-3, "layer 3 with bias")
    close(r["3"]["imbalance_no_bias"], 6.00, 5e-3, "layer 3 without bias")
    close(sorted(on)[9], 1.31, 5e-3, "median with bias")
    close(sorted(off)[9], 1.61, 5e-3, "median without bias")
    W, H = 720, 560
    x0, x1, y0, rh = 110, 680, 92, 22
    X = lambda v: x0 + (x1 - x0) * (v - 1) / 5.2
    b = [text(20, 28, "Imbalance per MoE layer, with and without the stored biases", 15, weight=700, family=SERIF),
         text(20, 48, "busiest expert's load / mean load · 32,768 held-out tokens · 1.0 is perfect · measured", 12,
              "f-muted", MUTED)]
    b.append(f'<circle cx="{x0 + 4}" cy="70" r="5" fill="{TEAL}" class="f-teal"/>')
    b.append(text(x0 + 14, 74, "with the bias (as trained)", 12))
    b.append(f'<circle cx="{x0 + 214}" cy="70" r="5" fill="{CORAL}" class="f-coral"/>')
    b.append(text(x0 + 224, 74, "bias removed", 12))
    for t in (1, 2, 3, 4, 5, 6):
        b.append(line(X(t), y0 - 6, X(t), y0 + 19 * rh))
        b.append(text(X(t), y0 + 19 * rh + 16, f"{t}×", 10, "f-muted", MUTED, "middle", family=MONO))
    for i, l in enumerate(layers):
        y = y0 + i * rh + rh / 2
        b.append(text(x0 - 12, y + 4, f"layer {l}", 11, "f-muted", MUTED, "end", family=MONO))
        b.append(line(X(on[i]), y, X(off[i]), y, MUTED, "f-muted-stroke", 1.5))
        for v, col, cls, name in ((on[i], TEAL, "f-teal", "with the bias"), (off[i], CORAL, "f-coral", "bias removed")):
            b.append(f'<circle cx="{X(v):.1f}" cy="{y:.1f}" r="5.5" fill="{col}" stroke="{PAPER}" stroke-width="1.5" '
                     f'class="{cls} f-ring"><title>layer {l}, {name}: {v:.2f}×</title></circle>')
    b.append(text(X(6.0) - 8, y0 + 2 * rh + rh / 2 + 18, "layer 3: one expert receives every token", 11, anchor="end",
                  weight=700))
    b.append(text(x0, y0 + 19 * rh + 40, "median over layers: 1.31× with the bias, 1.61× without", 12, weight=700))
    return svg("fig18-2-imbalance.svg", W, H, "Figure 18.2 · What the bias buys",
               "For each of the 19 MoE layers, the imbalance with the stored biases (median 1.31×, worst 2.29× in "
               "layer 3) and with them removed (median 1.61×; layers 2 and 3 reach 5.96× and 6.00×).", b)


# --------------------------------------------------------------------------- Ch 19
def fig_19_mla():
    cfg = M.AnuLMConfig.tiny_mla()
    D, H, Dc, Dr, Dn, Dv = (cfg.hidden_size, cfg.n_head, cfg.kv_lora_rank, cfg.qk_rope_head_dim,
                            cfg.qk_nope_head_dim, cfg.v_head_dim)
    assert (D, H, Dc, Dr, Dn, Dv) == (384, 6, 48, 24, 48, 48)
    W, H_ = 760, 520
    b = [text(20, 28, "Latent attention, one layer: cache a latent and one rotary key", 15, weight=700, family=SERIF),
         text(20, 48, f"model.py lines {_lines(M.MLAttention)} · shapes for tiny_mla: D = {D}, H = {H}, Dc = {Dc}, Dr = {Dr}, "
                      f"Dn = {Dn}, Dv = {Dv}", 12, "f-muted", MUTED)]
    b += _box(20, 230, 100, 54, "x", "(B, S, D)")
    # query branch
    b += _box(170, 76, 150, 54, "q_proj", "D → H·(Dn+Dr)")
    b += _box(370, 66, 160, 40, "q_nope", f"(B,H,S,{Dn})")
    b += _box(370, 116, 160, 40, "q_pe → RoPE", f"(B,H,S,{Dr})")
    b += _arrow(120, 245, 168, 108)
    b += _arrow(320, 96, 368, 86)
    b += _arrow(320, 110, 368, 134)
    # kv branch
    b += _box(170, 300, 150, 54, "kv_a_proj_with_mqa", f"D → Dc+Dr = {Dc + Dr}")
    b += _arrow(120, 268, 168, 322)
    b.append(rect(356, 270, 188, 138, PAPER, "f-paper", 10, f' stroke="{GOLD}" stroke-width="2.5"'))
    b.append(text(450, 290, "the KV cache", 12, anchor="middle", weight=700))
    b += _box(370, 300, 160, 40, "c → RMSNorm", f"(B, T, {Dc})", accent=False)
    b += _box(370, 352, 160, 40, "k_pe → RoPE", f"(B, 1, T, {Dr})")
    b += _arrow(320, 320, 368, 320)
    b += _arrow(320, 334, 368, 370)
    # expansion
    b += _box(580, 300, 160, 54, "kv_b_proj", f"Dc → H·(Dn+Dv)")
    b += _arrow(530, 320, 578, 324)
    b += _box(580, 378, 160, 40, "k_pe expanded", f"(B, H, T, {Dr})")
    b += _arrow(530, 372, 578, 396)
    b += _box(580, 200, 160, 54, "k_nope, v", f"(B,H,T,{Dn}), ({Dv})")
    b += _arrow(660, 300, 660, 256)
    # attention
    b += _box(580, 76, 160, 64, "attention", f"q, k: {Dn + Dr} wide · v: {Dv}")
    b += _arrow(530, 100, 578, 104)
    b += _arrow(660, 200, 660, 142)
    b.append(line(740, 398, 752, 398, MUTED, "f-muted-stroke", 1.5))
    b.append(line(752, 398, 752, 120, MUTED, "f-muted-stroke", 1.5))
    b += _arrow(752, 120, 742, 120)
    b += _box(580, 446, 160, 50, "o_proj", "H·Dv → D")
    b.append(text(560, 470, "attention output, (B, S, H·Dv) →", 11, "f-muted", MUTED, "end"))
    b.append(text(20, 440, f"Cached per token and layer: Dc + Dr = {Dc + Dr} numbers.", 12, weight=700))
    b.append(text(20, 460, f"Full per-head keys and values would be H·(Dn+Dr) + H·Dv = "
                           f"{H * (Dn + Dr) + H * Dv}.", 12, "f-muted", MUTED))
    b.append(text(20, 480, "kv_b_proj re-expands every cached latent at every step.", 12, "f-muted", MUTED))
    return svg("fig19-1-mla.svg", W, H_, "Figure 19.1 · Latent attention, one layer",
               "x is projected to queries, split into a non-rotary and a rotary part. A second projection makes a "
               "48-wide latent and a 24-wide rotary key; these two are cached. kv_b_proj expands the latent into "
               "every head's key part and value; the rotary key is shared across heads; attention and o_proj follow.",
               b)


def fig_19_cache():
    cm, cg, r = M.AnuLMConfig.tiny_mla(), M.AnuLMConfig.tiny_gqa(), released()
    rows = [("tiny_mla (MLA)", cm.kv_lora_rank + cm.qk_rope_head_dim,
             cm.n_head * (cm.qk_nope_head_dim + cm.qk_rope_head_dim + cm.v_head_dim)),
            ("tiny_gqa (GQA)", 2 * cg.n_kv_head * cg.head_dim, 2 * cg.n_head * cg.head_dim),
            ("released 400M (GQA)", 2 * r.n_kv_head * r.head_dim, 2 * r.n_head * r.head_dim)]
    assert [(a, f) for _, a, f in rows] == [(72, 720), (256, 768), (512, 2048)]
    W, H = 720, 300
    x0, x1 = 230, 580
    X = lambda v: x0 + (x1 - x0) * math.log10(v) / math.log10(5000)
    b = [text(20, 28, "Numbers cached per token per layer", 15, weight=700, family=SERIF),
         text(20, 48, "what each design caches, against full per-head keys and values · log scale · verified", 12,
              "f-muted", MUTED)]
    b.append(f'<circle cx="{x0 + 4}" cy="70" r="5" fill="{TEAL}" class="f-teal"/>')
    b.append(text(x0 + 14, 74, "cached", 12))
    b.append(f'<circle cx="{x0 + 94}" cy="70" r="5" fill="{CORAL}" class="f-coral"/>')
    b.append(text(x0 + 104, 74, "if every head kept its own K and V", 12))
    for t in (10, 100, 1000):
        b.append(line(X(t), 90, X(t), 250))
        b.append(text(X(t), 266, f"{t:,}", 10, "f-muted", MUTED, "middle", family=MONO))
    for i, (name, cached, full) in enumerate(rows):
        y = 112 + i * 50
        b.append(text(x0 - 12, y + 12, name, 12.5, anchor="end", weight=700))
        b.append(rect(x0, y, X(cached) - x0, 12, TEAL, "f-teal", 3, tip=f"{name}: cached {cached:,}"))
        b.append(rect(x0, y + 15, X(full) - x0, 12, CORAL, "f-coral", 3, tip=f"{name}: full {full:,}"))
        b.append(text(X(cached) + 6, y + 10, f"{cached:,}", 11, weight=700, family=MONO))
        b.append(text(X(full) + 6, y + 25, f"{full:,}  ({full / cached:.1f}×)", 11, weight=700, family=MONO))
    return svg("fig19-2-cache.svg", W, H, "Figure 19.2 · What each design caches",
               "tiny_mla's MLA caches 72 numbers against 720 for full per-head K and V, 10 times fewer; tiny_gqa's "
               "GQA caches 256 against 768; the released GQA model caches 512 against 2,048.", b)
