"""
Figures for Part IV, Chapters 12–15 (masking, grouped-query attention, rotary
positions, normalisation and the residual stream).

    python book/tools/make_figures.py part_iv_a

Each function computes what it draws from `model.py` or from the measurements
in book/measurements/part_iv_a_*.json, and asserts the numbers the chapter
prints.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from make_figures import (BOOK, CARD, CORAL, GOLD, GOLDTEXT, HEAT, INK, MONO, MUTED, NIGHT, PAPER, RULE,  # noqa: E402
                          SANS, SERIF, SLATE, TEAL, M, close, esc, line, rect, svg, text)

MEAS = BOOK / "measurements"


def _meas(name):
    return json.loads((MEAS / f"part_iv_a_{name}.json").read_text(encoding="utf-8"))


def _arrow(x1, y1, x2, y2, dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    return line(x1, y1, x2, y2, MUTED, "f-muted-stroke", 1.4, f' marker-end="url(#arrow)"{d}')


def _box(x, y, w, h, title, sub=None, accent=False, dashed=False):
    stroke = f' stroke="{GOLD}" stroke-width="2.5"' if accent else f' stroke="{RULE}" stroke-width="1"'
    if dashed:
        stroke = f' stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5 4"'
    out = [rect(x, y, w, h, CARD, "f-card", 8, stroke)]
    out.append(text(x + w / 2, y + (h / 2 + 5 if sub is None else 22), title, 13, anchor="middle", weight=700))
    if sub:
        out.append(text(x + w / 2, y + 40, sub, 11, "f-muted", MUTED, "middle", family=MONO))
    return out


# --------------------------------------------------------------------------- Ch 12
def _grid(b, x0, y0, mask, title, sub, c=24, wrong=None):
    """mask: (S, T) bool. wrong: cells to ring (where this mask differs from the right one)."""
    S, T = mask.shape
    b.append(text(x0, y0 - 34, title, 13, weight=700))
    b.append(text(x0, y0 - 18, sub, 11, "f-muted", MUTED, family=MONO))
    for i in range(S):
        for j in range(T):
            x, y = x0 + j * c, y0 + i * c
            on = bool(mask[i, j])
            tip = f"query {i}, key {j}: {'may attend' if on else 'masked'}"
            if on:
                b.append(rect(x + 1, y + 1, c - 2, c - 2, GOLD, "f-gold", 3, tip=tip))
            else:
                b.append(rect(x + 1, y + 1, c - 2, c - 2, "url(#hatch)", "", 3, f' stroke="{RULE}"', tip))
            if wrong is not None and bool(wrong[i, j]):
                b.append(f'<rect x="{x - 1}" y="{y - 1}" width="{c + 2}" height="{c + 2}" rx="4" fill="none" '
                         f'stroke="{INK}" class="f-ink-stroke" stroke-width="2.2"/>')
    return S * c


def fig_masks():
    cpu = torch.device("cpu")
    causal = torch.ones(6, 6, dtype=torch.bool).tril()               # what is_causal builds for S = T = 6
    assert M._attn_mask(6, 6, 0, None, cpu) is None
    offset = M._attn_mask(3, 6, 3, None, cpu)                         # 3 new tokens, 3 cached
    top_left = torch.ones(3, 6, dtype=torch.bool).tril()             # is_causal for (3, 6)
    window = M._attn_mask(6, 6, 0, 3, cpu)
    assert int(window.sum(-1).max()) == 3 and int((offset != top_left).sum()) == 9
    w, h = 760, 330
    b = [text(20, 28, "Which keys each query may see", 15, weight=700, family=SERIF),
         text(20, 48, "rows: queries · columns: keys, positions 0–5 · gold = may attend · hatched = masked",
              12, "f-muted", MUTED)]
    _grid(b, 24, 118, causal, "Causal", "S=T=6, start=0 → None")
    _grid(b, 214, 118, offset, "Offset (explicit)", "S=3, T=6, start=3")
    _grid(b, 404, 118, top_left, "Offset with is_causal", "top-left: wrong", wrong=offset != top_left)
    _grid(b, 594, 118, window, "Window of 3", "S=T=6, window=3")
    b.append(text(214, 210, "queries at positions 3, 4, 5", 11, "f-muted", MUTED))
    b.append(text(404, 210, "ringed: 9 cells that differ", 11, "f-muted", MUTED))
    b.append(text(404, 226, "each query loses its own key", 11, "f-muted", MUTED))
    b.append(text(404, 242, "and the two before it", 11, "f-muted", MUTED))
    b.append(text(24, 300, "_attn_mask returns None for the causal case, so SDPA builds that mask itself; the other "
                  "three are built explicitly.", 12, "f-muted", MUTED))
    return svg("fig12-1-masks.svg", w, h, "Figure 12.1 · Causal, offset and windowed masks",
               "Four 0/1 grids. Causal: lower triangle of a 6 by 6 grid. Offset: three new queries at positions 3 to 5 "
               "against six keys see keys up to their own position. The same call with is_causal sees a top-left "
               "triangle instead, nine cells wrong. Window of 3: a band of width three along the diagonal.", b)


def fig_mask_loss():
    r = _meas("masks")
    close(r["as_trained"]["all"], 3.351, 5e-4, "as-trained loss")
    close(r["no_window"]["pos_256_1023"], 4.380, 5e-4, "no window, positions 256+")
    close(r["no_mask"]["all"], 7.979, 5e-4, "no mask")
    rows = [("as trained", r["as_trained"]), ("window removed", r["no_window"]), ("no mask at all", r["no_mask"])]
    w, h = 760, 300
    x0, x1, y0 = 190, 690, 96
    sx = lambda v: x0 + (x1 - x0) * v / 10.4
    b = [text(20, 28, "Held-out loss of the released model with its masks changed", 15, weight=700, family=SERIF),
         text(20, 48, f"ckpt_base_v2.pt · {r['windows']} windows of {r['seq']:,} tokens, seed {r['seed']} · nats per token "
              "· measured", 12, "f-muted", MUTED)]
    b.append(f'<circle cx="{x0 + 4}" cy="70" r="5" fill="{TEAL}" class="f-teal"/>')
    b.append(text(x0 + 14, 74, "positions 0–255", 12))
    b.append(f'<circle cx="{x0 + 134}" cy="70" r="5" fill="{CORAL}" class="f-coral"/>')
    b.append(text(x0 + 144, 74, "positions 256–1,023", 12))
    for t in range(0, 11, 2):
        b.append(line(sx(t), y0 - 4, sx(t), y0 + 3 * 52))
        b.append(text(sx(t), y0 + 3 * 52 + 16, f"{t}", 10, "f-muted", MUTED, "middle", family=MONO))
    lnv = math.log(32768)
    b.append(line(sx(lnv), y0 - 4, sx(lnv), y0 + 3 * 52, INK, "f-ink-stroke", 1.2, ' stroke-dasharray="3 3"'))
    b.append(text(sx(lnv) - 4, y0 + 3 * 52 + 32, "ln V = 10.397, a uniform guess", 10.5, "f-muted", MUTED, "end"))
    for i, (name, v) in enumerate(rows):
        y = y0 + 12 + i * 52
        b.append(text(x0 - 14, y + 10, name, 12.5, anchor="end", weight=700))
        for k, (key, col, cls) in enumerate((("pos_0_255", TEAL, "f-teal"), ("pos_256_1023", CORAL, "f-coral"))):
            yy = y + k * 16
            bw = sx(v[key]) - x0
            b.append(rect(x0, yy, bw, 13, col, cls, 3, tip=f"{name}, {key.replace('pos_', 'positions ').replace('_', '–')}: "
                          f"{v[key]:.3f}"))
            b.append(text(x0 + bw + 6, yy + 11, f"{v[key]:.3f}", 11, weight=700, family=MONO))
    return svg("fig12-2-mask-loss.svg", w, h, "Figure 12.2 · What the masks are worth to the trained model",
               "As trained: 3.507 on positions 0–255 and 3.299 on 256–1,023. Window removed: 3.506 and 4.380. "
               "No mask at all: 8.243 and 7.891.", b)


# --------------------------------------------------------------------------- Ch 13
def fig_gqa_groups():
    from trace_forward import released
    cfg = released()
    H, Hkv, Dh = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    rep = H // Hkv
    k = torch.arange(float(Hkv)).view(1, Hkv, 1, 1).repeat_interleave(rep, dim=1).flatten().tolist()
    assert k == [h // rep for h in range(H)]
    w, h = 760, 330
    b = [text(20, 28, "One fused projection, 16 query heads, 4 shared key/value heads", 15, weight=700, family=SERIF),
         text(20, 48, f"AnuLM-Base-v2-400M · query_key_value: D = {cfg.hidden_size:,} → "
              f"(H + 2·Hkv)·Dh = ({H} + 2·{Hkv})·{Dh} = {(H + 2 * Hkv) * Dh:,}", 12, "f-muted", MUTED)]
    x0, y0, cw = 20, 80, 720 / (H + 2 * Hkv)
    labels = [f"q{i}" for i in range(H)] + [f"k{j}" for j in range(Hkv)] + [f"v{j}" for j in range(Hkv)]
    for i, lab in enumerate(labels):
        kind = "q" if i < H else "k" if i < H + Hkv else "v"
        group = i // rep if kind == "q" else (i - H) % Hkv
        fill = GOLD if group == 2 else CARD
        cls = "f-gold" if group == 2 else "f-card"
        b.append(rect(x0 + i * cw + 1, y0, cw - 2, 34, fill, cls, 4, f' stroke="{RULE}"',
                      f"{lab}: rows {i * Dh}–{(i + 1) * Dh - 1} of the weight"))
        b.append(text(x0 + i * cw + cw / 2, y0 + 22, lab, 11, "f-ink" if group != 2 else "f-onfill",
                      INK if group != 2 else NIGHT, "middle", 700, MONO))
    for lab, a, z in (("queries: H·Dh = 1,024 wide", 0, H), ("keys: 256", H, H + Hkv), ("values: 256", H + Hkv, H + 2 * Hkv)):
        b.append(line(x0 + a * cw + 2, y0 + 44, x0 + z * cw - 2, y0 + 44, MUTED, "f-muted-stroke", 1))
        b.append(text(x0 + (a + z) * cw / 2, y0 + 60, lab, 11, "f-muted", MUTED, "middle"))
    # grouping
    gy = 210
    b.append(text(20, gy - 18, "repeat_interleave: query head h reads key/value head h // 4", 13, weight=700))
    for i in range(H):
        cx = x0 + 30 + i * 42
        g = i // rep
        b.append(rect(cx - 16, gy, 32, 26, GOLD if g == 2 else CARD, "f-gold" if g == 2 else "f-card", 4,
                      f' stroke="{RULE}"', f"query head {i} → key/value head {g}"))
        b.append(text(cx, gy + 18, f"q{i}", 11, "f-ink" if g != 2 else "f-onfill", INK if g != 2 else NIGHT,
                      "middle", 700, MONO))
        kx = x0 + 30 + (g * rep + 1.5) * 42
        b.append(line(cx, gy + 26, kx, gy + 70, GOLD if g == 2 else MUTED,
                      "f-gold-stroke" if g == 2 else "f-muted-stroke", 1.6 if g == 2 else 1))
    for g in range(Hkv):
        kx = x0 + 30 + (g * rep + 1.5) * 42
        b.append(rect(kx - 40, gy + 70, 80, 28, GOLD if g == 2 else CARD, "f-gold" if g == 2 else "f-card", 4,
                      f' stroke="{RULE}"', f"key/value head {g}, shared by query heads {g * rep}–{g * rep + rep - 1}"))
        b.append(text(kx, gy + 89, f"k{g}, v{g}", 12, "f-ink" if g != 2 else "f-onfill", INK if g != 2 else NIGHT,
                      "middle", 700, MONO))
    b.append(text(20, gy + 116, "gold: one group, query heads 8–11 and the key/value head they share (h // 4 = 2)",
                  11, "f-muted", MUTED))
    return svg("fig13-1-gqa-groups.svg", w, h, "Figure 13.1 · Heads as slices of one projection",
               "The fused projection's 1,536 outputs are 24 slices of 64: query heads q0–q15, key heads k0–k3 and "
               "value heads v0–v3. repeat_interleave lets query heads 4g to 4g+3 read key/value head g.", b)


def fig_group_tv():
    r = _meas("groups")
    close(r["within_all"], 0.494, 5e-4, "within all")
    close(r["across_all"], 0.570, 5e-4, "across all")
    close(r["within"][0], 0.038, 5e-4, "layer 0 within")
    L = len(r["within"])
    w, h = 760, 300
    x0, x1, y0, y1 = 70, 730, 80, 240
    sx = lambda l: x0 + (x1 - x0) * (l + 0.5) / L
    sy = lambda v: y1 - (y1 - y0) * v / 0.8
    b = [text(20, 28, "How differently two heads attend, within a group and across groups", 15, weight=700, family=SERIF),
         text(20, 48, f"total-variation distance between weight rows, 0 = identical, 1 = disjoint · {r['windows']} "
              f"held-out windows of {r['seq']} · measured", 12, "f-muted", MUTED)]
    b.append(f'<circle cx="{x0 + 4}" cy="66" r="5" fill="{TEAL}" class="f-teal"/>')
    b.append(text(x0 + 14, 70, f"same key/value head ({r['pairs_within']} pairs)", 12))
    b.append(f'<circle cx="{x0 + 254}" cy="66" r="5" fill="{CORAL}" class="f-coral"/>')
    b.append(text(x0 + 264, 70, f"different key/value heads ({r['pairs_across']} pairs)", 12))
    for t in (0, 0.2, 0.4, 0.6, 0.8):
        b.append(line(x0, sy(t), x1, sy(t)))
        b.append(text(x0 - 8, sy(t) + 4, f"{t:.1f}", 10, "f-muted", MUTED, "end", family=MONO))
    for l in range(L):
        b.append(text(sx(l), y1 + 16, f"{l}", 10, "f-muted", MUTED, "middle", family=MONO))
        a, c = r["within"][l], r["across"][l]
        b.append(line(sx(l), sy(a), sx(l), sy(c), MUTED, "f-muted-stroke", 1.5))
        for v, col, cls, name in ((c, CORAL, "f-coral", "different groups"), (a, TEAL, "f-teal", "same group")):
            b.append(f'<circle cx="{sx(l):.1f}" cy="{sy(v):.1f}" r="5" fill="{col}" stroke="{PAPER}" stroke-width="1.5" '
                     f'class="{cls} f-ring"><title>layer {l}, {name}: {v:.3f}</title></circle>')
    b.append(text((x0 + x1) / 2, y1 + 34, "layer", 11, "f-muted", MUTED, "middle"))
    return svg("fig13-2-group-tv.svg", w, h, "Figure 13.2 · Do heads that share keys attend alike?",
               "For each of 20 layers, the mean total-variation distance between attention rows of query heads that "
               "share a key/value head, and of heads that do not. Same-group pairs are closer in 19 layers; layer 0 "
               "most of all, 0.038 against 0.238. Over all layers 0.494 against 0.570.", b)


# --------------------------------------------------------------------------- Ch 14
def _toy_rope(start):
    Q = torch.tensor([[1., 0.], [0., 4.], [1., 2.]])
    K = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])
    cos, sin = M.Rotary(M.AnuLMConfig(), dim=2)(3, torch.device("cpu"), torch.float32, start=start)
    q, k = M.apply_rope(Q[None, None], K[None, None], cos, sin)
    return q[0, 0], k[0, 0]


def fig_rope_toy():
    panels = []
    for start in (0, 5):
        q, k = _toy_rope(start)
        panels.append((start, q, k, (q[2] @ k[1]).item()))
    close(panels[0][3], 3.844, 5e-4, "bit·dog with RoPE")
    close(panels[1][3], 3.844, 5e-4, "bit·dog shifted by 5")
    w, h = 760, 336
    b = [text(20, 28, "RoPE on the toy: bit's query and dog's key, turned by their positions", 15, weight=700,
              family=SERIF),
         text(20, 48, "D_h = 2, one pair turning 1 radian per position · illustrative · the score is the same in both panels",
              12, "f-muted", MUTED)]
    for p, (start, q, k, score) in enumerate(panels):
        cx, cy, r = 190 + p * 380, 225, 40
        b.append(line(cx - 110, cy, cx + 110, cy))
        b.append(line(cx, cy - 100, cx, cy + 70))
        b.append(text(cx - 150, 82, f"positions {start}–{start + 2}", 13, weight=700))
        b.append(text(cx - 150, 98, f"bit at {start + 2}, dog at {start + 1}: q·k = {score:.3f}", 11.5, "f-muted",
                      MUTED, family=MONO))
        for vec, col, cls, lab in ((q[2], CORAL, "f-coral", "q bit"), (k[1], TEAL, "f-teal", "k dog")):
            x2, y2 = cx + r * vec[0].item(), cy - r * vec[1].item()
            b.append(f'<line x1="{cx}" y1="{cy}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="3"><title>{lab}: [{vec[0]:.3f}, {vec[1]:.3f}]</title></line>')
            b.append(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="4" fill="{col}" class="{cls}"/>')
            b.append(text(x2 + (8 if vec[0] >= 0 else -8), y2 - 6, lab, 11.5, weight=700,
                          anchor="start" if vec[0] >= 0 else "end"))
    b.append(text(20, 322, "Shifting both positions by 5 turns both vectors by 5 radians; the angle between them, "
                  "and so their dot product, stays.", 11.5, "f-muted", MUTED))
    return svg("fig14-1-rope-toy.svg", w, h, "Figure 14.1 · A rotation by position",
               "Two panels. At positions 0 to 2 and at positions 5 to 7, bit's query and dog's key point in "
               "different directions, but the angle between them is the same, so their dot product is 3.844 in both.", b)


def fig_rope_waves():
    rows = {}
    for theta in (1e4, 1e6):
        f = M.Rotary._build_inv_freq(M.AnuLMConfig(rope_theta=theta), 64)
        rows[theta] = (2 * math.pi / f).tolist()
    assert sum(v <= 1024 for v in rows[1e6]) == 12 and sum(v <= 1024 for v in rows[1e4]) == 18
    close(rows[1e6][-1], 4_080_185, 1, "slowest wavelength, θ = 1e6")
    w, h = 760, 300
    x0, x1, y0, y1 = 80, 730, 80, 240
    sx = lambda i: x0 + (x1 - x0) * (i + 0.5) / 32
    sy = lambda v: y1 - (y1 - y0) * math.log10(v) / 7
    b = [text(20, 28, "Tokens per full turn, for each of a head's 32 pairs", 15, weight=700, family=SERIF),
         text(20, 48, "wavelength 2π/ω_i, ω_i = θ^(−2i/64) · log scale · verified from Rotary._build_inv_freq",
              12, "f-muted", MUTED)]
    b.append(f'<circle cx="{x0 + 4}" cy="64" r="5" fill="{CORAL}" class="f-coral"/>')
    b.append(text(x0 + 14, 68, "θ = 10,000", 12))
    b.append(f'<circle cx="{x0 + 114}" cy="64" r="5" fill="{TEAL}" class="f-teal"/>')
    b.append(text(x0 + 124, 68, "θ = 1,000,000 (released)", 12))
    for e in range(0, 8):
        b.append(line(x0, sy(10 ** e), x1, sy(10 ** e)))
        b.append(text(x0 - 8, sy(10 ** e) + 4, f"10^{e}" if e > 1 else f"{10 ** e}", 10, "f-muted", MUTED, "end",
                      family=MONO))
    b.append(line(x0, sy(1024), x1, sy(1024), INK, "f-ink-stroke", 1.3, ' stroke-dasharray="4 3"'))
    b.append(text(x0 + 6, sy(1024) - 6, "1,024: the released context", 11, weight=700))
    for theta, col, cls in ((1e4, CORAL, "f-coral"), (1e6, TEAL, "f-teal")):
        for i, v in enumerate(rows[theta]):
            b.append(f'<circle cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="4" fill="{col}" class="{cls}">'
                     f'<title>θ = {theta:,.0f}, pair {i}: {v:,.1f} tokens per turn</title></circle>')
    for i in range(0, 32, 4):
        b.append(text(sx(i), y1 + 16, f"{i}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "pair i (dimensions i and i + 32)", 11, "f-muted", MUTED, "middle"))
    return svg("fig14-2-wavelengths.svg", w, h, "Figure 14.2 · How fast each pair turns",
               "Both bases start at 6.28 tokens per turn for pair 0. At θ = 10,000 the slowest pair turns once in "
               "47,117 tokens and 18 pairs complete a turn within 1,024 tokens; at θ = 1,000,000 the slowest takes "
               "4,080,185 tokens and 12 pairs complete a turn.", b)


# --------------------------------------------------------------------------- Ch 15
def fig_prenorm():
    w, h = 760, 385
    b = [text(20, 28, "Where the norm sits: pre-norm (model.py) and post-norm", 15, weight=700, family=SERIF),
         text(20, 48, "one block; the residual stream x runs down the left of each panel", 12, "f-muted", MUTED)]

    def plus(cx, cy):
        return [f'<circle cx="{cx}" cy="{cy}" r="9" fill="{PAPER}" stroke="{INK}" stroke-width="1.5" class="f-paper f-ink-stroke"/>',
                line(cx - 5, cy, cx + 5, cy, INK, "f-ink-stroke", 1.5), line(cx, cy - 5, cx, cy + 5, INK, "f-ink-stroke", 1.5)]

    # pre-norm panel
    x0, rail = 30, 70
    b.append(text(x0, 80, "Pre-norm · Block.forward", 13, weight=700))
    b.append(line(rail, 96, rail, 330, INK, "f-ink-stroke", 3))
    for i, (norm, sub, y) in enumerate((("input_layernorm", "attention", 130), ("post_attention_layernorm", "MLP / MoE", 230))):
        b += _box(rail + 50, y - 20, 170, 40, norm, accent=True)
        b += _box(rail + 240, y - 20, 100, 40, sub)
        b.append(_arrow(rail, y, rail + 48, y))
        b.append(_arrow(rail + 220, y, rail + 238, y))
        b.append(line(rail + 290, y + 20, rail + 290, y + 50, MUTED, "f-muted-stroke", 1.4))
        b.append(_arrow(rail + 290, y + 50, rail + 10, y + 50))
        b += plus(rail, y + 50)
    b.append(text(rail + 14, 372, "the stream itself is never normalised", 11, "f-muted", MUTED))
    # post-norm panel: the norm sits on the rail itself
    x1, rail2 = 420, 470
    b.append(text(x1, 80, "Post-norm · NOT IN THE REPO", 13, weight=700))
    dash = ' stroke-dasharray="6 4"'
    top = 96
    for sub, y in (("attention", 120), ("FFN", 250)):
        b.append(line(rail2, top, rail2, y + 41, MUTED, "f-muted-stroke", 3, dash))
        b.append(line(rail2, y + 59, rail2, y + 70, MUTED, "f-muted-stroke", 3, dash))
        b += _box(rail2 + 70, y - 20, 110, 40, sub, dashed=True)
        b.append(_arrow(rail2, y, rail2 + 68, y, dash=True))
        b.append(line(rail2 + 125, y + 20, rail2 + 125, y + 50, MUTED, "f-muted-stroke", 1.4, ' stroke-dasharray="4 3"'))
        b.append(_arrow(rail2 + 125, y + 50, rail2 + 10, y + 50, dash=True))
        b += plus(rail2, y + 50)
        b += _box(rail2 - 55, y + 70, 110, 32, "LayerNorm", dashed=True)
        top = y + 102
    b.append(line(rail2, top, rail2, top + 10, MUTED, "f-muted-stroke", 3, dash))
    b.append(text(rail2 + 70, 372, "the stream itself is normalised", 11, "f-muted", MUTED))
    return svg("fig15-1-prenorm.svg", w, h, "Figure 15.1 · Pre-norm and post-norm",
               "Left, as in model.py: each sublayer reads a normalised copy of the stream and its output is added "
               "back; the stream is never normalised. Right, drawn dashed because it is not in the repository: the "
               "sublayer's output is added and the sum is normalised.", b)


def fig_stream():
    r = _meas("stream")
    close(r["stream"][0], 0.023, 5e-4, "stream into layer 0")
    close(r["stream"][20], 4.381, 5e-4, "stream into the final norm")
    close(r["mlp_pos0"][7], 101.29, 5e-3, "layer 7 MLP at position 0")
    L = 20
    w, h = 760, 330
    x0, x1, y0, y1 = 70, 730, 84, 260
    sx = lambda i: x0 + (x1 - x0) * (i + 0.5) / (L + 1)
    sy = lambda v: y1 - (y1 - y0) * v / 4.5
    b = [text(20, 28, "The residual stream grows through the layers", 15, weight=700, family=SERIF),
         text(20, 48, "RMS per token over 1,024 dimensions, positions 1–511 · 16 held-out windows of 512 · bf16 experts · measured",
              12, "f-muted", MUTED)]
    b.append(rect(x0, 60, 12, 10, GOLD, "f-gold", 2))
    b.append(text(x0 + 18, 69, "stream entering the layer", 12))
    b.append(f'<circle cx="{x0 + 214}" cy="65" r="5" fill="{TEAL}" class="f-teal"/>')
    b.append(text(x0 + 224, 69, "attention adds", 12))
    b.append(f'<circle cx="{x0 + 344}" cy="65" r="5" fill="{CORAL}" class="f-coral"/>')
    b.append(text(x0 + 354, 69, "feed-forward adds", 12))
    for t in (0, 1, 2, 3, 4):
        b.append(line(x0, sy(t), x1, sy(t)))
        b.append(text(x0 - 8, sy(t) + 4, f"{t}", 10, "f-muted", MUTED, "end", family=MONO))
    bw = (x1 - x0) / (L + 1) * 0.6
    for i in range(L + 1):
        v = r["stream"][i]
        b.append(rect(sx(i) - bw / 2, sy(v), bw, y1 - sy(v), GOLD, "f-gold", 2,
                      tip=f"{'final norm' if i == L else f'layer {i}'}: stream {v:.3f}"))
        b.append(text(sx(i), y1 + 16, "out" if i == L else f"{i}", 10, "f-muted", MUTED, "middle", family=MONO))
        if i < L:
            for key, col, cls, name in (("attn", TEAL, "f-teal", "attention adds"), ("mlp", CORAL, "f-coral", "feed-forward adds")):
                b.append(f'<circle cx="{sx(i):.1f}" cy="{sy(r[key][i]):.1f}" r="4" fill="{col}" stroke="{PAPER}" '
                         f'stroke-width="1.2" class="{cls} f-ring"><title>layer {i}, {name}: {r[key][i]:.3f}</title></circle>')
    b.append(text((x0 + x1) / 2, y1 + 34, "layer (out = the input to the final norm)", 11, "f-muted", MUTED, "middle"))
    lo, hi = min(r["stream_pos0_min"][8:19]), max(r["stream_pos0_max"][8:19])
    close(lo, 81.5, 0.05, "position 0, smallest window, layers 8-18")
    close(hi, 130.2, 0.05, "position 0, largest window, layers 8-18")
    b.append(text(20, 318, f"Position 0, not drawn: layer 7's feed-forward adds 101.3 there; entering layers 8–18 the "
                  f"stream is about 100 ({lo:.0f}–{hi:.0f} across windows).", 11.5, "f-muted", MUTED))
    return svg("fig15-2-stream.svg", w, h, "Figure 15.2 · The residual stream of the released model",
               "The stream's RMS per token grows from 0.023 entering layer 0 to 4.381 entering the final norm. What "
               "attention and the feed-forward half add grows with it, from under 0.4 in the early layers to between "
               "1.0 and 1.8 in layers 18 and 19.", b)


def fig_split_halves():
    n, c, x0 = 16, 40, 60                                            # a 16-wide head; the rule is the same at 64
    rot = M.Rotary(M.AnuLMConfig(), dim=n)
    cos, _ = rot(3, torch.device("cpu"), torch.float32)
    assert torch.equal(cos[:, 1], cos[:, 1 + n // 2])                # dims i and i + D/2 share an angle
    w, h = 760, 230
    b = [text(20, 28, "Which dimensions turn together", 15, weight=700, family=SERIF),
         text(20, 48, "a 16-wide head for the drawing; the released head is 64 wide, pairs (i, i + 32)", 12,
              "f-muted", MUTED)]
    for row, (name, pair) in enumerate((("split halves (model.py)", lambda i: i % (n // 2)),
                                         ("consecutive pairs (RoFormer)", lambda i: i // 2))):
        y = 90 + row * 80
        b.append(text(20, y - 10, name, 12.5, weight=700))
        for i in range(n):
            p = pair(i)
            accent = p == 1
            b.append(rect(x0 + i * c + 2, y, c - 4, 30, GOLD if accent else CARD, "f-gold" if accent else "f-card", 4,
                          f' stroke="{RULE}"', f"dimension {i}: pair {p}"))
            b.append(text(x0 + i * c + c / 2, y + 20, f"{i}", 11, "f-onfill" if accent else "f-ink",
                          NIGHT if accent else INK, "middle", 700, MONO))
    b.append(text(20, 222, "gold: pair 1, which turns at ω_1. rotate_half pairs dimension i with i + D_h/2; the paper "
                  "pairs 2i with 2i + 1.", 11.5, "f-muted", MUTED))
    return svg("fig14-3-split-halves.svg", w, h, "Figure 14.3 · Split halves against consecutive pairs",
               "Top: model.py rotates dimension i with dimension i + 8 of a 16-wide head, so pair 1 is dimensions 1 and "
               "9. Bottom: the RoFormer paper rotates consecutive dimensions, so pair 1 is dimensions 2 and 3.", b)
