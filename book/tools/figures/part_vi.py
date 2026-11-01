"""
Figures for Part VI, Chapters 27–29 (the KV cache, choosing the next token,
long context).

    python book/tools/make_figures.py part_vi

Each function computes what it draws from `model.py` or from the measurements
in book/measurements/part_vi_*.json, and asserts the numbers the chapter
prints. Teal is the part's accent.
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
    return json.loads((MEAS / f"part_vi_{name}.json").read_text(encoding="utf-8"))


def _tiny(**kw):
    base = dict(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
                num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
                n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


def _title(b, title, sub):
    b.append(text(20, 28, title, 15, weight=700, family=SERIF))
    b.append(text(20, 48, sub, 12, "f-muted", MUTED))


def _cell(b, x, y, c, on, tip, accent=TEAL):
    if on:
        b.append(rect(x + 1, y + 1, c - 2, c - 2, accent, "f-teal", 4, tip=tip))
    else:
        b.append(rect(x + 1, y + 1, c - 2, c - 2, "url(#hatch)", "", 4, f' stroke="{RULE}" stroke-width="1"', tip=tip))


# --------------------------------------------------------------------------- Ch 27
def fig_offset_mask():
    right = M._attn_mask(3, 5, 2, None, "cpu").tolist()
    top_left = torch.ones(3, 5, dtype=torch.bool).tril().tolist()
    rows = lambda g: ["".join("x" if v else "." for v in r) for r in g]
    assert rows(right) == ["xxx..", "xxxx.", "xxxxx"], rows(right)
    assert rows(top_left) == ["x....", "xx...", "xxx.."], rows(top_left)
    w, h, c = 760, 290, 40
    b = []
    _title(b, "Three new queries against a cache of two", "rows: queries at absolute positions 2–4 · columns: keys 0–4 · "
           "filled = may attend")
    for gi, (grid, name, sub) in enumerate(((right, "_attn_mask(3, 5, start=2)", "correct: the corner of the full triangle"),
                                            (top_left, "is_causal=True", "top-left: the cached keys are hidden"))):
        x0, y0 = 110 + gi * 340, 112
        b.append(text(x0, y0 - 34, name, 13, weight=700, family=MONO))
        b.append(text(x0, y0 - 18, sub, 11.5, "f-muted", MUTED))
        b.append(rect(x0, y0 - 6, 2 * c, 4, SLATE, "", 2))
        b.append(text(x0 + c, y0 + 5 * 0 - 9, "", 10))
        for j in range(5):
            b.append(text(x0 + j * c + c / 2, y0 + 3 * c + 18, f"k{j}", 11, "f-muted", MUTED, "middle", family=MONO))
        b.append(text(x0 + c, y0 + 3 * c + 34, "cached", 11, "f-muted", MUTED, "middle"))
        b.append(text(x0 + 3.5 * c, y0 + 3 * c + 34, "new", 11, "f-muted", MUTED, "middle"))
        for i in range(3):
            b.append(text(x0 - 10, y0 + i * c + c / 2 + 4, f"q{i + 2}", 11, "f-muted", MUTED, "end", family=MONO))
            for j in range(5):
                on = grid[i][j]
                _cell(b, x0 + j * c, y0 + i * c, c, on, f"query at {i + 2}, key {j}: {'attends' if on else 'hidden'}",
                      TEAL if gi == 0 else CORAL)
    return svg("fig27-1-offset-mask.svg", w, h, "Figure 27.1 · The offset mask",
               "Left: the mask _attn_mask builds for 3 queries at positions 2, 3, 4 against 5 keys; rows xxx.., xxxx., "
               "xxxxx. Right: is_causal's top-left mask, x...., xx..., xxx.., which hides the two cached keys.", b)


def fig_cache_timeline():
    torch.manual_seed(0)
    m = M.AnuLM(_tiny(block_size=8)).eval()
    calls = []
    hook = m.register_forward_pre_hook(lambda mod, a, kw: calls.append((kw.get("start", 0), a[0].shape[1])),
                                       with_kwargs=True)
    m.generate(torch.randint(0, 64, (1, 6)), 5, top_k=1, use_cache=True)
    hook.remove()
    assert calls == [(0, 6), (6, 1), (7, 1), (0, 8), (0, 8)], calls
    w, h, c = 760, 330, 34
    x0, y0, rh = 150, 92, 42
    b = []
    _title(b, "What each call of generate runs, block_size = 8", "prompt of 6 tokens, 5 new · teal: run in this call · "
           "grey: read from the cache · the first 3 calls fill the context, then every call rebuilds")
    for t in range(11):
        b.append(text(x0 + t * c + c / 2, y0 - 8, f"t{t}", 10.5, "f-muted", MUTED, "middle", family=MONO))
    length = 6
    for i, (start, S) in enumerate(calls):
        y = y0 + i * rh
        kind = "prefill" if i == 0 else ("decode" if start > 0 else "rebuild")
        b.append(text(x0 - 12, y + c / 2 + 4, f"call {i + 1}: {kind}", 12, anchor="end", weight=700))
        first = length - (start + S)                              # sequence index of position 0
        for t in range(length):
            pos = t - first
            x = x0 + t * c
            if pos < 0:
                b.append(rect(x + 1, y + 1, c - 2, c - 2, PAPER, "f-paper", 4, f' stroke="{RULE}" stroke-width="1"',
                              tip=f"token t{t}: outside the context"))
            elif pos >= start:
                b.append(rect(x + 1, y + 1, c - 2, c - 2, TEAL, "f-teal", 4, tip=f"token t{t} at position {pos}: run"))
                b.append(text(x + c / 2, y + c / 2 + 4, f"{pos}", 10.5, "f-onfill", PAPER, "middle", 700, MONO))
            else:
                b.append(rect(x + 1, y + 1, c - 2, c - 2, CARD, "f-card", 4, f' stroke="{RULE}" stroke-width="1"',
                              tip=f"token t{t} at position {pos}: cached"))
                b.append(text(x + c / 2, y + c / 2 + 4, f"{pos}", 10.5, "f-muted", MUTED, "middle", 400, MONO))
        b.append(text(x0 + 11 * c + 14, y + c / 2 + 4, f"start={start}, S={S}", 11.5, family=MONO))
        length += 1
    b.append(text(x0, y0 + 5 * rh + 14, "numbers are positions; the rebuilds re-run the last 8 tokens at positions 0–7",
                  11.5, "f-muted", MUTED))
    return svg("fig27-2-cache-timeline.svg", w, h, "Figure 27.2 · Prefill, decode and the slide",
               "Five calls: a prefill of 6 tokens, decode steps at positions 6 and 7, then two rebuilds that each run the "
               "last 8 tokens at positions 0 to 7.", b)


# --------------------------------------------------------------------------- Ch 28
def _arrow(x1, y1, x2, y2, dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    return line(x1, y1, x2, y2, MUTED, "f-muted-stroke", 1.4, f' marker-end="url(#arrow)"{d}')


def _box(b, x, y, w, h, title, sub, accent=False, dashed=False):
    stroke = f' stroke="{TEAL}" stroke-width="2.5"' if accent else f' stroke="{RULE}" stroke-width="1"'
    if dashed:
        stroke = f' stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5 4"'
    b.append(rect(x, y, w, h, PAPER if dashed else CARD, "f-paper" if dashed else "f-card", 8, stroke))
    b.append(text(x + w / 2, y + 22, title, 12.5, anchor="middle", weight=700))
    b.append(text(x + w / 2, y + 40, sub, 10.5, "f-muted", MUTED, "middle", family=MONO))


def fig_sampling_pipeline():
    import inspect
    src = inspect.getsource(M.AnuLM.generate)
    marks = ["logits[:, -1, :].float()", "self.text_vocab:", "repetition_penalty != 1.0",
             "max(temperature, 1e-5)", "torch.topk", "F.softmax", "torch.multinomial"]
    at = [src.index(k) for k in marks]
    assert at == sorted(at), "the chapter's order of operations no longer matches generate"
    assert "top_p" not in src, "generate now has top_p: the figure's EXTENSION box is wrong"
    w, h = 760, 300
    b = []
    _title(b, "From the last logits to one token, in generate's order",
           "AnuLM.generate lines 1091–1105 · shapes and rules in each box · dashed: not in the repository")
    bw, bh = 150, 52
    row1 = [("last row, fp32", "(B, V)"), ("text-vocab mask", "audio, image → −∞"),
            ("repetition penalty", "÷ or × seen ids"), ("÷ temperature", "max(T, 1e-5)")]
    row2 = [("top-k", "below k-th → −∞"), ("top-p", "EXTENSION"), ("softmax", "(B, V) probs"),
            ("multinomial", "(B, 1) ids")]
    xs = [30 + i * 185 for i in range(4)]
    for i, (t, s) in enumerate(row1):
        _box(b, xs[i], 80, bw, bh, t, s, accent=(t == "÷ temperature"))
        if i:
            b.append(_arrow(xs[i - 1] + bw, 106, xs[i] - 2, 106))
    b.append(_arrow(xs[3] + bw / 2, 80 + bh, xs[0] + bw / 2 + 40, 186))
    for i, (t, s) in enumerate(row2):
        _box(b, xs[i], 190, bw, bh, t, s, dashed=(t == "top-p"))
        if i:
            b.append(_arrow(xs[i - 1] + bw, 216, xs[i] - 2, 216, dash=(i in (1, 2))))
    b.append(text(xs[1] + bw / 2, 262, "NOT IN THE REPO · Chapter 28.7's patch", 10.5, "f-muted", MUTED, "middle"))
    return svg("fig28-1-pipeline.svg", w, h, "Figure 28.1 · From logits to a token",
               "Seven steps in generate's order: last row in fp32, text-vocabulary mask, repetition penalty, division by "
               "the temperature, top-k, softmax, multinomial. Top-p, dashed between top-k and softmax, is not in the "
               "repository.", b)


def fig_nucleus():
    r = _meas("ch28_sampling")
    nuc = r["nucleus90_size_T1"]
    close(nuc["median"], 341, 0.01, "nucleus median")
    close(nuc["share_le_40"], 0.394, 0.0005, "share within 40")
    close(nuc["p90"], 7650, 5, "nucleus p90")
    counts = r["nucleus90_log2_histogram"]["counts"]
    total = sum(counts)
    assert total == r["positions"] == 8192, total
    w, h = 760, 330
    x0, x1, y0, y1 = 70, 720, 80, 250
    bw = (x1 - x0) / len(counts)
    top = max(counts)
    sy = lambda v: y1 - (y1 - y0) * v / 1600
    b = []
    _title(b, "Tokens in the p = 0.9 nucleus, at 8,192 positions of the released model",
           "temperature 1 · bf16 autocast · held-out windows · bins double in width · measured")
    for v in range(0, 1601, 400):
        b.append(line(x0, sy(v), x1, sy(v)))
        b.append(text(x0 - 8, sy(v) + 4, f"{v}", 10, "f-muted", MUTED, "end", family=MONO))
    for k, c in enumerate(counts):
        x = x0 + k * bw
        lo, hi = 2 ** k, 2 ** (k + 1) - 1
        b.append(rect(x + 3, sy(c), bw - 6, y1 - sy(c), GOLD, "f-gold", 3,
                      tip=f"nucleus of {lo}–{hi} tokens: {c} positions ({100 * c / total:.1f}%)"))
        b.append(text(x + 3, y1 + 16, f"{lo:,}", 9.5, "f-muted", MUTED, family=MONO))
    xk = x0 + math.log2(40) * bw
    b.append(line(xk, y0 - 10, xk, y1, INK, "f-ink-stroke", 1.5, ' stroke-dasharray="4 3"'))
    b.append(text(xk + 6, y0 - 2, "top-k = 40", 11.5, weight=700))
    b.append(text(xk - 6, y0 + 14, f"{100 * nuc['share_le_40']:.1f}% of positions", 11, "f-muted", MUTED, "end"))
    b.append(text(xk + 6, y0 + 14, f"{100 * (1 - nuc['share_le_40']):.1f}%", 11, "f-muted", MUTED))
    b.append(text((x0 + x1) / 2, y1 + 36, "tokens in the nucleus (bin start; each bin twice as wide as the last)", 11,
                  "f-muted", MUTED, "middle"))
    b.append(text(x0 - 46, y0 - 14, "positions", 11, "f-muted", MUTED))
    return svg("fig28-2-nucleus.svg", w, h, "Figure 28.2 · How many tokens a 0.9 nucleus holds",
               f"Histogram over 8,192 positions: 1,436 positions need one token; the median nucleus is 341 tokens; "
               f"the ninetieth percentile about 7,650. 39.4% of positions fit within top-k's 40.", b)


# --------------------------------------------------------------------------- Ch 29
def _released(**kw):
    from dataclasses import replace
    cfg = M.AnuLMConfig.nano_350m(vocab_size=32768, block_size=1024, sliding_window=256, max_window_layers=10,
                                  num_experts=24, num_experts_per_tok=4, num_shared_experts=0,
                                  moe_intermediate_size=192)
    return replace(cfg, **kw)


def fig_yarn_bands():
    plain = M.Rotary._build_inv_freq(_released(), 64)
    divs = {}
    for f in (2.0, 4.0):
        cfg = _released(yarn=True, yarn_factor=f, yarn_original_context=1024)
        divs[f] = (plain / M.Rotary._build_inv_freq(cfg, 64)).tolist()
        lo, hi = M._yarn_correction_range(cfg.yarn_beta_fast, cfg.yarn_beta_slow, 64, cfg.rope_theta, 1024)
        assert (lo, hi) == (3, 12), (lo, hi)
    close(divs[2.0][4], 1.06, 0.005, "pair 4 at factor 2")
    close(divs[4.0][4], 1.09, 0.005, "pair 4 at factor 4")
    turns = (1024 * plain / (2 * math.pi)).tolist()
    assert sum(t < 1 for t in turns) == 20
    w, h = 760, 330
    x0, x1, y0, y1 = 80, 660, 84, 250
    sx = lambda i: x0 + (x1 - x0) * i / 31
    sy = lambda v: y1 - (y1 - y0) * (v - 1) / 3
    b = []
    _title(b, "How much YaRN divides each rotary pair's frequency", "released configuration: θ = 1e6, Dh = 64, "
           "original context 1,024 · β_fast 32, β_slow 1 · verified")
    b.append(rect(sx(3), y0 - 6, sx(12) - sx(3), y1 - y0 + 6, CARD, "f-card", 0, tip="pairs 3–12: the ramp"))
    for lab, a, c in (("kept", 0, 3), ("blended", 3, 12), ("divided by the factor", 12, 31)):
        b.append(text((sx(a) + sx(c)) / 2, y0 - 12, lab, 11.5, "f-muted", MUTED, "middle", 700))
    for v in (1, 2, 3, 4):
        b.append(line(x0, sy(v), x1, sy(v)))
        b.append(text(x0 - 8, sy(v) + 4, f"{v}×", 10, "f-muted", MUTED, "end", family=MONO))
    for i in list(range(0, 32, 4)) + [31]:
        b.append(text(sx(i), y1 + 16, f"{i}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "rotary pair i (fast to slow); pairs 12–31 never complete a turn in 1,024 tokens",
                  11, "f-muted", MUTED, "middle"))
    for f, col, cls in ((2.0, TEAL, "f-teal"), (4.0, CORAL, "f-coral")):
        pts = [(sx(i), sy(d)) for i, d in enumerate(divs[f])]
        b.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
                 f'" fill="none" stroke="{col}" stroke-width="2"/>')
        for i, (x, y) in enumerate(pts):
            b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{col}" class="{cls}">'
                     f'<title>factor {f:.0f}, pair {i}: divided by {divs[f][i]:.3f}</title></circle>')
        b.append(text(x1 + 10, sy(f) + 4, f"factor {f:.0f}", 11.5, weight=700))
    return svg("fig29-1-yarn-bands.svg", w, h, "Figure 29.1 · Which pairs YaRN stretches",
               "Pairs 0 to 3 keep their frequency, pairs 12 to 31 are divided by the factor (2 or 4), and pairs 4 to 11 "
               "ramp between; at factor 2 pair 4 is divided by 1.06, at factor 4 by 1.09.", b)


def fig_context_loss():
    r = _meas("ch29_context")["settings"]
    close(r["2048 naive"]["quarters"][3], 5.229, 0.0005, "2048 naive Q4")
    close(r["4096 yarn"]["mean"], 3.170, 0.0005, "4096 yarn mean")
    close(r["4096 naive"]["quarters"][3], 6.272, 0.0005, "4096 naive Q4")
    w, h = 760, 360
    x0, x1, y0, y1 = 80, 580, 80, 290
    sx = lambda p: x0 + (x1 - x0) * p / 4096
    sy = lambda v: y1 - (y1 - y0) * (v - 2.8) / 3.8
    b = []
    _title(b, "Loss by position, released model, no retraining", "nats per token, each point a quarter of a window · "
           "16 held-out windows · bf16 · measured")
    for v in (3, 4, 5, 6):
        b.append(line(x0, sy(v), x1, sy(v)))
        b.append(text(x0 - 8, sy(v) + 4, f"{v}", 10, "f-muted", MUTED, "end", family=MONO))
    for p in (0, 1024, 2048, 3072, 4096):
        b.append(text(sx(p), y1 + 16, f"{p:,}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "position in the window", 11, "f-muted", MUTED, "middle"))
    b.append(line(sx(1024), y0 - 6, sx(1024), y1, INK, "f-ink-stroke", 1.2, ' stroke-dasharray="4 3"'))
    b.append(text(sx(1024) + 6, y0 + 2, "trained length", 11, weight=700))
    series = (("1024 trained", SLATE, "", "1,024 as trained"), ("2048 naive", CORAL, "5 4", "2,048 plain RoPE"),
              ("2048 yarn", TEAL, "5 4", "2,048 YaRN ×2"), ("4096 naive", CORAL, "", "4,096 plain RoPE"),
              ("4096 yarn", TEAL, "", "4,096 YaRN ×4"))
    for k, (name, col, dash, label) in enumerate(series):
        L = r[name]["length"]
        pts = [(sx((j + 0.5) * L / 4), sy(v)) for j, v in enumerate(r[name]["quarters"])]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        b.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
                 f'" fill="none" stroke="{col}" stroke-width="2"{d}/>')
        for j, (x, y) in enumerate(pts):
            b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{col}" stroke="{PAPER}" stroke-width="1.5" '
                     f'class="f-ring"><title>{label}, quarter {j + 1}: {r[name]["quarters"][j]:.3f}</title></circle>')
        ly = y0 + 10 + k * 22
        b.append(line(x1 + 20, ly, x1 + 46, ly, col, "", 2, d))
        b.append(text(x1 + 52, ly + 4, label, 11.5))
    return svg("fig29-2-context-loss.svg", w, h, "Figure 29.2 · Loss by position past the training length",
               "With plain RoPE the loss rises past position 1,024, to 5.23 in the last quarter at 2,048 and 6.27 at "
               "4,096; with YaRN it stays between 3.07 and 3.33, mean 3.170 at 4,096.", b)


def fig_window_cache():
    cfg = _released()
    per = cfg.n_kv_head * cfg.head_dim * (4 + 2)
    now = cfg.n_layer * 1024 * per
    ext = sum(cfg.sliding_window if i < cfg.max_window_layers else 1024 for i in range(cfg.n_layer)) * per
    close(now / 2 ** 20, 30.0, 1e-9, "cache now")
    close(ext / 2 ** 20, 18.75, 1e-9, "cache trimmed")
    w, h = 760, 320
    x0, x1, y0, y1 = 80, 720, 90, 240
    bw = (x1 - x0) / cfg.n_layer
    sy = lambda v: y1 - (y1 - y0) * v / 1024
    b = []
    _title(b, "Positions each layer caches at a full 1,024-token context", "released model · bars: the repository "
           "today · dashed: windowed layers trimmed to 256 (EXTENSION)")
    for v in (0, 256, 512, 768, 1024):
        b.append(line(x0, sy(v), x1, sy(v)))
        b.append(text(x0 - 8, sy(v) + 4, f"{v}", 10, "f-muted", MUTED, "end", family=MONO))
    for i in range(cfg.n_layer):
        x = x0 + i * bw
        b.append(rect(x + 4, sy(1024), bw - 8, y1 - sy(1024), GOLD, "f-gold", 2,
                      f' fill-opacity="{0.35 if i < cfg.max_window_layers else 0.9}"', tip=f"layer {i}: 1,024 positions"))
        if i < cfg.max_window_layers:
            b.append(rect(x + 4, sy(256), bw - 8, y1 - sy(256), "none", "", 2,
                          f' stroke="{INK}" stroke-width="1.5" stroke-dasharray="4 3" class="f-ink-stroke"',
                          tip=f"layer {i}, trimmed: 256 positions"))
        b.append(text(x + bw / 2, y1 + 16, f"{i}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "layer (0–9 windowed, 256 tokens)", 11, "f-muted", MUTED, "middle"))
    b.append(text(x0, y1 + 58, f"cache per 1,024-token sequence: {now / 2**20:.2f} MiB now, {ext / 2**20:.2f} MiB "
                  f"trimmed ({1 - ext / now:.1%} less), fp32 keys and bf16 values", 11.5, weight=700))
    return svg("fig29-3-window-cache.svg", w, h, "Figure 29.3 · Cache per layer, with and without trimming",
               "All 20 layers cache 1,024 positions today: 30.00 MiB. Trimming layers 0 to 9 to 256 positions, an "
               "extension, would make it 18.75 MiB, 37.5% less.", b)


def fig_step_time():
    r = _meas("ch27_cache")
    rows = {d["context"]: d for d in r["step_ms_bf16"]}
    close(rows[65]["cached_ms"], 30.0, 0.051, "cached at 65")
    close(rows[65]["uncached_ms"], 39.9, 0.051, "uncached at 65")
    close(rows[1024]["cached_ms"], 28.6, 0.051, "cached at 1024")
    close(rows[1024]["uncached_ms"], 38.6, 0.051, "uncached at 1024")
    close(r["rebuild_ms_bf16"], 38.0, 0.051, "rebuild")
    w, h = 760, 330
    x0, x1, y0, y1 = 90, 560, 80, 270
    sx = lambda v: x0 + (x1 - x0) * v / 1024
    sy = lambda v: y1 - (y1 - y0) * v / 60
    b = []
    _title(b, "Milliseconds per step, released model, batch 1", "bf16 autocast · median of 25 calls, lowest of 3 sweeps "
           "· RTX 5070 Ti on a shared machine · measured")
    for v in range(0, 61, 10):
        b.append(line(x0, sy(v), x1, sy(v)))
        b.append(text(x0 - 8, sy(v) + 4, f"{v}", 10, "f-muted", MUTED, "end", family=MONO))
    for t in (0, 256, 512, 768, 1024):
        b.append(text(sx(t), y1 + 16, f"{t}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "context length (tokens)", 11, "f-muted", MUTED, "middle"))
    b.append(text(x0 - 40, y0 - 14, "ms", 11, "f-muted", MUTED))
    for key, col, cls, name in (("uncached_ms", CORAL, "f-coral", "uncached forward"),
                                ("cached_ms", TEAL, "f-teal", "cached decode step")):
        pts = [(sx(c), sy(d[key])) for c, d in sorted(rows.items())]
        b.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) +
                 f'" fill="none" stroke="{col}" stroke-width="2"/>')
        for (c, d), (x, y) in zip(sorted(rows.items()), pts):
            b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}" stroke="{PAPER}" stroke-width="2" '
                     f'class="{cls} f-ring"><title>{name}, context {c}: {d[key]:.1f} ms</title></circle>')
        lx, ly = pts[-1]
        b.append(text(lx + 12, ly + 8, f"{name}: {sorted(rows.items())[-1][1][key]:.1f}", 11.5, weight=700))
    rb = r["rebuild_ms_bf16"]
    b.append(f'<rect x="{sx(1024) - 5:.1f}" y="{sy(rb) - 5:.1f}" width="10" height="10" fill="none" stroke="{INK}" '
             f'class="f-ink-stroke" stroke-width="1.5"><title>rebuild of 1,024 tokens: {rb:.1f} ms</title></rect>')
    b.append(text(sx(1024) + 12, sy(rb) - 14, f"rebuild of 1,024: {rb:.1f}", 11.5, weight=700))
    b.append(text(x0 + 6, sy(rows[65]["cached_ms"]) + 22, "a decode step costs about 29 ms whatever the context",
                  11.5, "f-muted", MUTED))
    return svg("fig27-3-step-time.svg", w, h, "Figure 27.3 · Time per step on the released model",
               "Cached decode steps take 30.0 ms at context 65 and 28.6 ms at 1,024; uncached forwards 39.9 and 38.6 ms; "
               "the rebuild at 1,024 tokens 38.0 ms.", b)
