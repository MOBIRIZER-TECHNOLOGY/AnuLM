"""
Figures for Part I (Chapters 1-3) and Part II (Chapters 4-7).

    python book/tools/make_figures.py part_i_ii

Every number drawn here is computed from its source (the repository at the pin,
a book example, or book/measurements/part_i_ii_released.json) and asserted
against the value the chapter prints.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_figures import (BOOK, ROOT, INK, MUTED, RULE, CARD, PAPER, NIGHT, GOLD, GOLDTEXT, CORAL, TEAL,  # noqa: E402
                          SLATE, HEAT, SANS, MONO, SERIF, esc, text, rect, line, svg, close)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BOOK / "code-examples"))


def _title(b, t, sub):
    b.append(text(20, 28, t, 15, weight=700, family=SERIF))
    b.append(text(20, 48, sub, 12, "f-muted", MUTED))


def _box(x, y, w, h, title, sub, accent=False):
    out = [rect(x, y, w, h, CARD, "f-card", 8,
                f' stroke="{SLATE}" stroke-width="2.5"' if accent else f' stroke="{RULE}" stroke-width="1"')]
    out.append(text(x + w / 2, y + h / 2 - 3, title, 13, anchor="middle", weight=700))
    out.append(text(x + w / 2, y + h / 2 + 15, sub, 11.5, "f-muted", MUTED, "middle", family=MONO))
    return out


def _arrow(x1, y1, x2, y2):
    return line(x1, y1, x2, y2, MUTED, "f-muted-stroke", 1.5, ' marker-end="url(#arrow)"')


def _bars(b, rows, x0, x1, y0, rh, vmax, fmt, highlight=None, ticks=(), tfmt=None):
    tfmt = tfmt or fmt
    sx = lambda v: (x1 - x0) * v / vmax
    for t in ticks:
        b.append(line(x0 + sx(t), y0 - 6, x0 + sx(t), y0 + len(rows) * rh))
        b.append(text(x0 + sx(t), y0 + len(rows) * rh + 15, tfmt(t), 10, "f-muted", MUTED, "middle", family=MONO))
    for i, (name, v) in enumerate(rows):
        y = y0 + i * rh
        b.append(text(x0 - 10, y + rh / 2 + 4, name, 12, anchor="end"))
        fill = HEAT if name == highlight else GOLD
        b.append(rect(x0, y + 4, max(sx(v), 2), rh - 8, fill, "f-gold", 3, tip=f"{name}: {fmt(v)}"))
        b.append(text(x0 + max(sx(v), 2) + 6, y + rh / 2 + 4, fmt(v), 11.5, weight=700, family=MONO))


# --------------------------------------------------------------------------- Ch 1
def fig_01_loop():
    from anulm import model as M
    from tests.test_model import tiny
    torch.manual_seed(0)
    m = M.AnuLM(tiny()).eval()
    ids = torch.randint(0, 64, (1, 4))
    with torch.no_grad():
        logits, _ = m(ids)
    probs = logits[:, -1, :].softmax(-1)
    nxt = probs.argmax(-1, keepdim=True)
    assert tuple(logits.shape) == (1, 1, 64) and tuple(probs.shape) == (1, 64) and tuple(nxt.shape) == (1, 1)
    w, h = 760, 330
    b = []
    _title(b, "The four-step loop in AnuLM.generate", "shapes printed on tiny() (V = 64) by ch01_four_steps.py · B = 1, S = 4")
    b += _box(30, 120, 130, 70, "1 · read", "self(ids)", accent=True)
    b += _box(230, 120, 130, 70, "2 · score", "softmax(logits)")
    b += _box(430, 120, 130, 70, "3 · pick", "multinomial")
    b += _box(610, 120, 130, 70, "4 · repeat", "cat(idx, nxt)")
    for x1, x2, lab in ((160, 228, "(1, 1, 64)"), (360, 428, "(1, 64)"), (560, 608, "(1, 1)")):
        b.append(_arrow(x1, 155, x2, 155))
        b.append(text((x1 + x2) / 2, 145, lab, 10.5, "f-muted", MUTED, "middle", family=MONO))
    b.append(f'<path d="M675,190 C675,270 95,270 95,192" fill="none" stroke="{MUTED}" class="f-muted-stroke" '
             f'stroke-width="1.5" marker-end="url(#arrow)"/>')
    b.append(text(388, 262, "ids grows by one: (1, 4) → (1, 5) → …   the cache keeps the rest (Ch 27)", 11.5,
                  "f-muted", MUTED, "middle", family=MONO))
    b.append(text(95, 108, "ids (1, 4)", 11, "f-muted", MUTED, "middle", family=MONO))
    b.append(text(388, 300, "logits are (B, 1, V): without targets the forward pass projects only the last position",
                  11.5, "f-muted", MUTED, "middle"))
    return svg("fig01-1-loop.svg", w, h, "Figure 1.1 · The four-step loop",
               "Read: the model runs on ids (1, 4) and returns logits (1, 1, 64). Score: softmax gives probabilities "
               "(1, 64). Pick: one id (1, 1). Repeat: the id is appended and the loop runs again.", b)


def fig_01_repo_map():
    from ch01_four_steps import repo_map, ROLES, PIN
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        groups = repo_map()
    rows = [(r, sum(n for _, n in groups.get(r, []))) for r, _ in ROLES]
    total_files = sum(len(v) for v in groups.values())
    # The numbers Chapter 1.5 prints; a changed count must change the chapter too.
    assert total_files == 90 and dict(rows)["the model"] == 1276, (total_files, rows)
    total = sum(v for _, v in rows)
    assert total == 23918 and round(100 * 1276 / total) == 5, total
    w, h = 760, 40 * len(rows) + 110
    b = []
    _title(b, "The repository at the pin, by role",
           f"lines of Python in the {total_files} files of anulm/, tests/, tools/ and the root · total {total:,} · "
           f"verified with git at {PIN}")
    _bars(b, rows, 210, 640, 70, 30, 5000, lambda v: f"{v:,.0f}", highlight="the model",
          ticks=(0, 1000, 2000, 3000, 4000, 5000))
    return svg("fig01-2-repo-map.svg", w, h, "Figure 1.2 · The repository by role",
               "Lines of Python per role: " + ", ".join(f"{r} {v:,}" for r, v in rows) + ".", b)


# --------------------------------------------------------------------------- Ch 2
def fig_02_ladder():
    r = json.loads((BOOK / "measurements" / "part_i_ii_ladder.json").read_text(encoding="utf-8"))
    rows = [("uniform over 256 bytes", r["uniform"]), ("unigram counts", r["unigram"]),
            ("bigram counts", r["bigram_counts"]), ("bigram_teacher.py table", r["bigram_teacher"]),
            ("AnuLM, zero blocks", r["zero_block_anulm"]), ("lab book's tiny transformer", r["course_tiny_transformer"])]
    for (name, v), want in zip(rows, (5.5452, 3.3475, 2.4931, 2.6264, 2.4871, 1.7505)):
        close(v, want, 5e-5, name)
    w, h = 760, 300
    b = []
    _title(b, "Held-out loss on Shakespeare, nats per byte (lower is better)",
           "data/input.txt, last 10% held out · ch02_counts_to_attention.py, seed 0 · the last bar: the lab book's run, measured differently")
    _bars(b, rows, 230, 650, 70, 32, 6.0, lambda v: f"{v:.3f}", highlight="lab book's tiny transformer",
          ticks=(0, 1, 2, 3, 4, 5, 6), tfmt=lambda v: f"{v:.0f}")
    return svg("fig02-1-ladder.svg", w, h, "Figure 2.1 · The ladder",
               "Held-out loss: " + ", ".join(f"{n} {v:.4f}" for n, v in rows) + ".", b)


def fig_02_timeline():
    reg = (BOOK / "RESEARCH_REFERENCES.md").read_text(encoding="utf-8")
    events = [("R03", 2013, "word2vec", "word vectors from large corpora"),
              ("R02", 2014, "attention for translation", "soft search of the source"),
              ("R01", 2017, "the transformer", "scaled dot-product attention"),
              ("R04", 2019, "GPT-2", "decoder-only language model"),
              ("R35", 2024, "LLM2Vec", "a decoder made bidirectional")]
    for rid, year, _, _ in events:
        row = next(l for l in reg.splitlines() if l.startswith(f"| {rid} |"))
        cells = [c.strip() for c in row.split("|")]
        assert cells[5] == str(year) and "CHECKED" in row, (rid, cells[5])
    w, h = 760, 250
    b = []
    _title(b, "From word vectors to AnuLM's two ways of reading", "dated by each source's first version")
    x0, x1, y = 60, 700, 130
    sx = lambda yr: x0 + (x1 - x0) * (yr - 2012) / (2025 - 2012)
    b.append(line(x0, y, x1, y, SLATE, "f-slate-stroke", 2))
    for yr in range(2012, 2026):
        b.append(line(sx(yr), y - 4, sx(yr), y + 4, SLATE, "f-slate-stroke", 1))
        if yr % 2 == 1 or yr in (2012,):
            pass
    for i, (rid, yr, name, sub) in enumerate(events):
        x = sx(yr)
        up = i % 2 == 0
        b.append(f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="{SLATE if rid != "R01" else INK}" class="f-slate"><title>{esc(name)} ({yr}) [{rid}]</title></circle>')
        ty = y - 52 if up else y + 34
        b.append(line(x, y - 8 if up else y + 8, x, ty + (14 if up else -14), RULE, "f-rule", 1))
        b.append(text(x, ty, f"{yr} · {name}", 12.5, anchor="middle", weight=700))
        b.append(text(x, ty + 16, f"{sub} [{rid}]", 11, "f-muted", MUTED, "middle"))
    b.append(text(sx(2012), y + 90, "Recurrent networks predate all of these.",
                  11, "f-muted", MUTED))
    return svg("fig02-2-timeline.svg", w, h, "Figure 2.2 · Five dated steps",
               "2013 word2vec; 2014 attention for translation; 2017 the transformer; 2019 GPT-2, decoder-only; "
               "2024 LLM2Vec, a decoder made bidirectional.", b)


def fig_02_masks():
    from anulm import model as M
    S = 5
    causal = M._attn_mask(S, S, 0, S, "cpu")           # a window as wide as the sequence: plain causal
    assert torch.equal(causal, torch.ones(S, S, dtype=torch.bool).tril())
    w, h = 760, 290
    b = []
    _title(b, "Which keys each query may see", "rows are queries, columns are keys · five positions · True = may attend")
    toks = ["t0", "t1", "t2", "t3", "t4"]
    for k, (title, m, x0) in enumerate((("decoder: causal (AnuLM)", causal, 110),
                                        ("encoder: bidirectional (bidirectional.py)", torch.ones(S, S, dtype=torch.bool), 450))):
        b.append(text(x0 + 90, 82, title, 12.5, anchor="middle", weight=700))
        for i in range(S):
            b.append(text(x0 - 10, 112 + i * 34, toks[i], 11, "f-muted", MUTED, "end", family=MONO))
            for j in range(S):
                on = bool(m[i, j])
                b.append(rect(x0 + j * 36, 94 + i * 34, 32, 30, SLATE if on else CARD, "f-slate" if on else "f-card", 4,
                              tip=f"query {toks[i]}, key {toks[j]}: {'sees' if on else 'hidden'}"))
        for j in range(S):
            b.append(text(x0 + j * 36 + 16, 94 + S * 34 + 14, toks[j], 11, "f-muted", MUTED, "middle", family=MONO))
    return svg("fig02-3-masks.svg", w, h, "Figure 2.3 · Who may see whom",
               "Left: causal attention, each query sees itself and earlier keys. Right: bidirectional, every query sees "
               "every key.", b)


# --------------------------------------------------------------------------- Ch 3
def _released_json():
    return json.loads((BOOK / "measurements" / "part_i_ii_released.json").read_text(encoding="utf-8"))


def fig_03_surprise():
    r = _released_json()
    hist = r["histogram_half_nat"]
    n = r["graded_tokens"]
    close(r["loss"], 3.379, 5e-4, "released loss")
    close(hist[0] / n, 0.239, 5e-4, "first bin share")
    close(r["frac_above_lnV"], 0.033, 5e-4, "share above ln V")
    w, h = 760, 340
    b = []
    _title(b, "Per-token surprise of the released model on held-out text",
           f"{n:,} graded tokens · 64 windows of 512 · bins of 0.5 nat · measured")
    x0, x1, y0, y1 = 70, 720, 70, 270
    sx = lambda v: x0 + (x1 - x0) * v / 18.0
    top = 8000
    sy = lambda c: y1 - (y1 - y0) * c / top
    for c in (0, 2000, 4000, 6000, 8000):
        b.append(line(x0, sy(c), x1, sy(c)))
        b.append(text(x0 - 8, sy(c) + 4, f"{c:,}", 10, "f-muted", MUTED, "end", family=MONO))
    for i, c in enumerate(hist):
        if c:
            b.append(rect(sx(i * 0.5) + 1, sy(c), sx(0.5) - x0 - 2, y1 - sy(c), GOLD, "f-gold", 1,
                          tip=f"{i * 0.5:.1f}–{i * 0.5 + 0.5:.1f} nats: {int(c):,} tokens"))
    for v in range(0, 19, 2):
        b.append(text(sx(v), y1 + 16, str(v), 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "surprise, −log p(true next token), nats", 11.5, "f-muted", MUTED, "middle"))
    for v, lab in ((r["loss"], f"mean {r['loss']:.3f}"), (r["ln_vocab"], f"ln V = {r['ln_vocab']:.3f}")):
        b.append(line(sx(v), y0 - 6, sx(v), y1, INK, "f-ink-stroke", 1.5, ' stroke-dasharray="4 3"'))
        b.append(text(sx(v) + 5, y0 + 4, lab, 11.5, weight=700))
    b.append(text(sx(r["ln_vocab"]) + 5, y0 + 20, f"{100 * r['frac_above_lnV']:.1f}% of tokens beyond", 11, "f-muted", MUTED))
    return svg("fig03-1-surprise.svg", w, h, "Figure 3.1 · How surprised the released model is",
               f"Histogram of per-token surprise: {100 * hist[0] / n:.1f}% of tokens below 0.5 nats, median "
               f"{r['quantiles']['p50']:.2f}, mean {r['loss']:.3f}, {100 * r['frac_above_lnV']:.1f}% above ln V.", b)


def fig_03_init():
    from ch03_one_job import initial_loss
    from trace_forward import released
    from anulm import model as M
    from tests.test_model import tiny
    rows = []
    for name, cfg, want in (("tiny()", tiny(), 4.1704), ("nano_30b", M.AnuLMConfig.nano_30b(), 5.6242),
                            ("released 400M", released(), 10.6166)):
        got, pred = initial_loss(cfg)
        close(got, want, 5e-4, name)
        rows.append((name, got - math.log(cfg.vocab_size), pred - math.log(cfg.vocab_size), cfg.hidden_size))
    w, h = 760, 250
    b = []
    _title(b, "A fresh model's first loss, above ln V", "predicted D · 0.02² / 2 (teal) and measured on random targets (coral) · verified")
    x0, x1, y0, rh = 200, 520, 80, 46
    sx = lambda v: x0 + (x1 - x0) * v / 0.30
    for t in (0, 0.1, 0.2, 0.3):
        b.append(line(sx(t), y0 - 10, sx(t), y0 + len(rows) * rh))
        b.append(text(sx(t), y0 + len(rows) * rh + 16, f"+{t:.1f}", 10, "f-muted", MUTED, "middle", family=MONO))
    for i, (name, got, pred, D) in enumerate(rows):
        y = y0 + i * rh + rh / 2
        b.append(text(x0 - 12, y + 4, f"{name} (D = {D:,})", 12, anchor="end"))
        b.append(line(sx(0), y, sx(max(got, pred)), y, RULE, "f-rule", 1))
        b.append(f'<circle cx="{sx(pred):.1f}" cy="{y:.1f}" r="6" fill="{TEAL}" class="f-teal"><title>{esc(name)} predicted +{pred:.4f}</title></circle>')
        b.append(f'<circle cx="{sx(got):.1f}" cy="{y:.1f}" r="6" fill="{CORAL}" class="f-coral"><title>{esc(name)} measured +{got:.4f}</title></circle>')
        b.append(text(540, y + 4, f"+{got:.3f} measured, +{pred:.3f} predicted", 11, "f-muted", MUTED, family=MONO))
    return svg("fig03-2-init.svg", w, h, "Figure 3.2 · Where a fresh model starts",
               "First loss minus ln V: " + "; ".join(f"{n} measured +{g:.3f}, predicted +{p:.3f}" for n, g, p, _ in rows) + ".", b)


# --------------------------------------------------------------------------- Ch 4
def fig_04_qkv_layout():
    from tests.test_model import tiny
    cfg = tiny()
    H, Hkv, Dh = cfg.n_head, cfg.n_kv_head, cfg.head_dim
    width = (H + 2 * Hkv) * Dh
    assert (H, Hkv, Dh, width) == (4, 2, 8, 64)
    w, h = 760, 260
    b = []
    _title(b, "One position's 64 outputs of query_key_value, cut into heads",
           "tiny(): H = 4 query heads, Hkv = 2 key and 2 value heads, Dh = 8 · view(B, S, 8, 8), then split([4, 2, 2])")
    x0, cw, y = 40, 10.6, 110
    names = [f"q{h}" for h in range(H)] + [f"k{j}" for j in range(Hkv)] + [f"v{j}" for j in range(Hkv)]
    for s, name in enumerate(names):
        accent = name.startswith("q")
        for d in range(Dh):
            col = s * Dh + d
            b.append(rect(x0 + col * cw, y, cw - 1, 30, SLATE if accent else CARD, "f-slate" if accent else "f-card", 1,
                          tip=f"column {col}: {name}, dimension {d}"))
        cx = x0 + (s * Dh + Dh / 2) * cw
        b.append(text(cx, y - 8, name, 12, anchor="middle", weight=700, family=MONO))
        b.append(text(x0 + s * Dh * cw, y + 48, str(s * Dh), 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text(x0 + width * cw, y + 48, str(width), 10, "f-muted", MUTED, "middle", family=MONO))
    for lab, a, z in (("queries: H · Dh = 32", 0, H * Dh), ("keys: 16", H * Dh, (H + Hkv) * Dh),
                      ("values: 16", (H + Hkv) * Dh, width)):
        b.append(line(x0 + a * cw + 2, y + 62, x0 + z * cw - 3, y + 62, MUTED, "f-muted-stroke", 1.5))
        b.append(text(x0 + (a + z) / 2 * cw, y + 80, lab, 11.5, "f-muted", MUTED, "middle"))
    b.append(text(x0, y + 112, "query head h = x @ W[h·Dh : (h+1)·Dh].T ; key head j = x @ W[(H+j)·Dh : (H+j+1)·Dh].T",
                  11.5, family=MONO))
    return svg("fig04-1-qkv-layout.svg", w, h, "Figure 4.1 · One row of qkv, cut into heads",
               "Columns 0-31 are four query heads of 8, columns 32-47 two key heads, columns 48-63 two value heads.", b)


def fig_04_matmul():
    w, h = 760, 270
    b = []
    _title(b, "q @ k.transpose(-2, -1): batch axes pass through, the inner axis disappears",
           "shapes from ch04_shapes.py on tiny() · B = 2, H = 4, S = T = 5, Dh = 8")

    def block(x, y, ww, hh, lab, shape, accent=False):
        out = [rect(x, y, ww, hh, CARD, "f-card", 4, f' stroke="{SLATE if accent else RULE}" stroke-width="{2 if accent else 1}"')]
        out.append(text(x + ww / 2, y + hh + 18, lab, 12.5, anchor="middle", weight=700))
        out.append(text(x + ww / 2, y + hh + 34, shape, 11, "f-muted", MUTED, "middle", family=MONO))
        return out
    b += block(60, 90, 80, 50, "q", "(2, 4, 5, 8)")
    b.append(text(170, 120, "@", 20, anchor="middle", weight=700))
    b += block(200, 75, 50, 80, "kᵀ", "(2, 4, 8, 5)")
    b.append(text(285, 120, "=", 20, anchor="middle", weight=700))
    b += block(320, 90, 50, 50, "scores", "(2, 4, 5, 5)", accent=True)
    b.append(text(60, 82, "S × Dh", 10.5, "f-muted", MUTED, family=MONO))
    b.append(text(200, 67, "Dh × T", 10.5, "f-muted", MUTED, family=MONO))
    b.append(text(320, 82, "S × T", 10.5, "f-muted", MUTED, family=MONO))
    notes = ["(B, H) = (2, 4): batch axes, matched one to one", "Dh = 8: the inner axis, summed over",
             "result: one S × T score matrix per batch and head", "cost: 2 · B · H · S · T · Dh = 3,200 FLOPs here"]
    assert 2 * 2 * 4 * 5 * 5 * 8 == 3200
    for i, n in enumerate(notes):
        b.append(text(440, 92 + i * 24, n, 12))
    return svg("fig04-2-matmul.svg", w, h, "Figure 4.2 · The batched matrix multiply",
               "q (2, 4, 5, 8) times k transposed (2, 4, 8, 5) gives scores (2, 4, 5, 5): the batch and head axes "
               "pass through and the head width is summed over.", b)


# --------------------------------------------------------------------------- Ch 5
def fig_05_temperature():
    z = torch.tensor([2.0, 1.0, 0.1])
    temps = (0.5, 1.0, 2.0)
    P = {T: torch.softmax(z / T, -1).tolist() for T in temps}
    close(P[0.5][0], 0.8638, 5e-5, "T=0.5 top")
    close(P[2.0][2], 0.194, 5e-4, "T=2 last")
    w, h = 760, 300
    b = []
    _title(b, "One set of logits, three temperatures", "logits (2.0, 1.0, 0.1), illustrative · softmax(z / T) · bar height = probability")
    shades = {0.5: HEAT, 1.0: GOLD, 2.0: "#ECCB8A"}
    x0, y1, ht = 90, 250, 170
    for i in range(3):
        gx = x0 + i * 220
        b.append(text(gx + 75, y1 + 20, f"token {i}  (z = {z[i].item():.1f})", 12, anchor="middle", weight=700))
        for j, T in enumerate(temps):
            v = P[T][i]
            x = gx + j * 50
            b.append(rect(x, y1 - ht * v, 44, ht * v, shades[T], "f-gold", 2, tip=f"T = {T}, token {i}: p = {v:.4f}"))
            b.append(text(x + 22, y1 - ht * v - 6, f"{v:.2f}", 10.5, anchor="middle", family=MONO))
    for j, T in enumerate(temps):
        b.append(rect(560 + j * 62, 64, 14, 14, shades[T], "f-gold", 2))
        b.append(text(578 + j * 62, 76, f"T = {T}", 11.5))
    b.append(line(x0 - 10, y1, 720, y1, MUTED, "f-muted-stroke", 1))
    return svg("fig05-1-temperature.svg", w, h, "Figure 5.1 · Temperature reshapes, never reorders",
               "Probabilities of three tokens with logits 2.0, 1.0, 0.1: at T = 0.5 they are "
               + ", ".join(f"{v:.3f}" for v in P[0.5]) + "; at T = 1, " + ", ".join(f"{v:.3f}" for v in P[1.0])
               + "; at T = 2, " + ", ".join(f"{v:.3f}" for v in P[2.0]) + ".", b)


def fig_05_released_temperature():
    r = _released_json()["temperature"]
    Ts = sorted(float(t) for t in r)
    loss = [r[str(t) if str(t) in r else f"{t}"]["loss"] for t in Ts]
    ent = [r[str(t)]["entropy"] for t in Ts]
    close(r["1.0"]["loss"], 3.379, 5e-4, "loss at T=1")
    close(r["0.8"]["loss"], 3.499, 5e-4, "loss at T=0.8")
    best = min(range(len(Ts)), key=lambda i: loss[i])
    assert Ts[best] == 0.96, Ts[best]
    close(loss[best], 3.372, 5e-4, "lowest loss")
    w, h = 760, 350
    b = []
    _title(b, "The released model's held-out loss and entropy at each temperature",
           "32,732 graded tokens · loss of softmax(z / T) on the true token (coral) · mean entropy of that distribution (teal) · measured")
    x0, x1, y0, y1 = 70, 700, 86, 290
    sx = lambda t: x0 + (x1 - x0) * (t - 0.4) / (2.1 - 0.4)
    sy = lambda v: y1 - (y1 - y0) * v / 10.5
    for v in (0, 2, 4, 6, 8, 10):
        b.append(line(x0, sy(v), x1, sy(v)))
        b.append(text(x0 - 8, sy(v) + 4, str(v), 10, "f-muted", MUTED, "end", family=MONO))
    for t in (0.5, 0.8, 1.0, 1.25, 1.5, 2.0):
        b.append(text(sx(t), y1 + 16, f"{t:g}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "temperature T", 11.5, "f-muted", MUTED, "middle"))
    b.append(text(x0 - 8, y0 - 10, "nats", 11, "f-muted", MUTED, "end"))
    for series, col, cls, name in ((loss, CORAL, "f-coral", "loss"), (ent, TEAL, "f-teal", "entropy")):
        pts = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(Ts, series))
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" class="{cls}-stroke" stroke-width="2.5"/>')
        for t, v in zip(Ts, series):
            b.append(f'<circle cx="{sx(t):.1f}" cy="{sy(v):.1f}" r="4.5" fill="{col}" class="{cls}"><title>T = {t:g}: {name} {v:.3f}</title></circle>')
    b.append(text(sx(2.0) - 6, sy(loss[-1]) + 20, "loss", 12, anchor="end", weight=700))
    b.append(text(sx(2.0) - 6, sy(ent[-1]) - 10, "entropy", 12, anchor="end", weight=700))
    b.append(line(sx(0.96), sy(loss[best]) + 6, sx(1.0) + 14, sy(2.3) - 4, INK, "f-ink-stroke", 1))
    b.append(text(sx(1.0) + 16, sy(2.3) + 8, f"lowest near T = 0.96: {loss[best]:.3f} (T = 1: 3.379)", 11.5, weight=700))
    return svg("fig05-2-released-temperature.svg", w, h, "Figure 5.2 · The released model under temperature",
               "Loss by temperature: " + ", ".join(f"T {t:g}: {v:.3f}" for t, v in zip(Ts, loss))
               + ". Entropy: " + ", ".join(f"T {t:g}: {v:.3f}" for t, v in zip(Ts, ent)) + ".", b)


def fig_05_gradient():
    z = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
    loss = torch.nn.functional.cross_entropy(z[None], torch.tensor([0]))
    loss.backward()
    p = torch.softmax(z.detach(), -1)
    y = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(z.grad, p - y, atol=1e-7)
    g = (p - y).tolist()
    close(g[0], -0.341, 5e-4, "p - y for the true token")
    w, h = 760, 300
    b = []
    _title(b, "∂ loss / ∂ z = p − y", "logits (2.0, 1.0, 0.1), true token 0 · illustrative · checked against autograd")
    x0, y_mid, sc = 300, 175, 160
    b.append(line(x0 - 20, y_mid, 720, y_mid, MUTED, "f-muted-stroke", 1))
    for i in range(3):
        x = x0 + i * 140
        v = g[i]
        col, cls = (TEAL, "f-teal") if v > 0 else (CORAL, "f-coral")
        top = y_mid - sc * v if v > 0 else y_mid
        b.append(rect(x, top, 70, abs(sc * v), col, cls, 3, tip=f"token {i}: p = {p[i]:.4f}, y = {y[i]:.0f}, p - y = {v:+.4f}"))
        b.append(text(x + 35, (y_mid - sc * v - 8) if v > 0 else (y_mid - sc * v + 18), f"{v:+.3f}", 12, anchor="middle", weight=700, family=MONO))
        b.append(text(x + 35, 280, f"token {i}: p {p[i]:.3f}, y {y[i]:.0f}", 11.5, "f-muted", MUTED, "middle"))
    b.append(text(20, 120, "positive: a descent step", 12))
    b.append(text(20, 138, "lowers this logit", 12))
    b.append(text(20, 215, "negative: a descent step", 12))
    b.append(text(20, 233, "raises the true logit", 12))
    return svg("fig05-3-gradient.svg", w, h, "Figure 5.3 · The gradient of cross-entropy",
               f"p - y for logits 2.0, 1.0, 0.1 and true token 0: {g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f}.", b)


# --------------------------------------------------------------------------- Ch 6
def fig_06_swiglu_graph():
    from anulm import model as M
    f = M.SwiGLU(8, 16)
    assert tuple(f.gate_proj.weight.shape) == (16, 8) and tuple(f.down_proj.weight.shape) == (8, 16)
    w, h = 760, 330
    b = []
    _title(b, "SwiGLU forward (top arrows) and backward (bottom arrows)",
           "ch06_backprop.py: N = 5 tokens, D = 8, I = 16 · each weight's gradient has the weight's shape")
    nodes = {"x": (30, 140, 80, "x", "(N, D)"), "a": (190, 80, 110, "a = x Wgᵀ", "(N, I)"),
             "b": (190, 200, 110, "b = x Wuᵀ", "(N, I)"), "s": (350, 80, 110, "s = silu(a)", "(N, I)"),
             "h": (500, 140, 100, "h = s ⊙ b", "(N, I)"), "y": (650, 140, 90, "y = h Wdᵀ", "(N, D)")}
    for k, (x, y, ww, t, sh) in nodes.items():
        b += _box(x, y, ww, 56, t, sh, accent=(k == "x"))
    edges = [("x", "a"), ("x", "b"), ("a", "s"), ("s", "h"), ("b", "h"), ("h", "y")]
    for u, v in edges:
        xu, yu, wu = nodes[u][:3]
        xv, yv, _ = nodes[v][:3]
        b.append(_arrow(xu + wu, yu + 22, xv - 2, yv + 22))
        b.append(f'<line x1="{xv:.1f}" y1="{yv + 38:.1f}" x2="{xu + wu + 2:.1f}" y2="{yu + 38:.1f}" stroke="{CORAL}" '
                 f'class="f-coral-stroke" stroke-width="1.5" stroke-dasharray="4 3" marker-end="url(#arrow)"/>')
    b.append(text(30, 300, "dashed: gradients flowing back. The two that reach x, through a and through b, add:"
                  "  dL/dx = dL/da · Wg + dL/db · Wu", 11.5, "f-muted", MUTED))
    return svg("fig06-1-swiglu-graph.svg", w, h, "Figure 6.1 · Backpropagation through SwiGLU",
               "x (N, D) feeds a = x Wg^T and b = x Wu^T (N, I); s = silu(a); h = s times b; y = h Wd^T (N, D). "
               "Gradients flow back along each edge; the two reaching x add.", b)


def fig_06_causal_grad():
    r = json.loads((BOOK / "measurements" / "part_i_ii_causal_grad.json").read_text(encoding="utf-8"))
    g = r["max_abs_grad"]
    assert all(v == 0.0 for v in g[6:]) and all(v > 0 for v in g[:6])
    close(g[5], 16.0, 0.5, "own position")
    w, h = 760, 300
    b = []
    _title(b, "How much each input position moves position 5's logits", "tiny(), untrained · largest |∂ logits[5] / ∂ embedding[j]| · log scale · verified")
    x0, y1, y0 = 80, 240, 80
    lo, hi = -1, 2
    sy = lambda v: y1 - (y1 - y0) * (math.log10(v) - lo) / (hi - lo)
    for e in range(lo, hi + 1):
        b.append(line(x0 - 6, sy(10 ** e), 720, sy(10 ** e)))
        b.append(text(x0 - 10, sy(10 ** e) + 4, f"1e{e}", 10, "f-muted", MUTED, "end", family=MONO))
    for j, v in enumerate(g):
        x = x0 + 10 + j * 52
        if v > 0:
            b.append(rect(x, sy(v), 38, y1 - sy(v), HEAT if j == 5 else GOLD, "f-gold", 2, tip=f"position {j}: {v:.2e}"))
        else:
            b.append(text(x + 19, y1 - 6, "0", 12, anchor="middle", weight=700, family=MONO))
        b.append(text(x + 19, y1 + 16, str(j), 11, "f-muted", MUTED, "middle", family=MONO))
    b.append(text(x0 + 10 + 6 * 52, y0 + 10, "positions 6–11: exactly zero (the future)", 12, weight=700))
    b.append(text(400, y1 + 36, "input position j", 11.5, "f-muted", MUTED, "middle"))
    return svg("fig06-2-causal-grad.svg", w, h, "Figure 6.2 · Gradients respect causality",
               "Largest gradient of position 5's logits per input position: " + ", ".join(f"{j}: {v:.2g}" for j, v in enumerate(g)) + ".", b)


# --------------------------------------------------------------------------- Ch 7
def fig_07_bias_correction():
    b1, b2 = 0.9, 0.95
    T = 40
    m = [1 - b1 ** t for t in range(1, T + 1)]          # raw first moment, constant gradient 1
    v = [1 - b2 ** t for t in range(1, T + 1)]          # raw second moment
    close(m[0], 0.1, 1e-12, "m after 1 step")
    close(m[4], 0.410, 5e-4, "m after 5 steps")
    w, h = 760, 320
    b = []
    _title(b, "Adam's moments with a constant gradient of 1", "raw m (coral) and raw v (teal) start at 0; m / (1 − β₁ᵗ) and v / (1 − β₂ᵗ) are 1 at every step · β = (0.9, 0.95)")
    x0, x1, y0, y1 = 70, 700, 80, 270
    sx = lambda t: x0 + (x1 - x0) * (t - 1) / (T - 1)
    sy = lambda val: y1 - (y1 - y0) * val / 1.05
    for val in (0, 0.25, 0.5, 0.75, 1.0):
        b.append(line(x0, sy(val), x1, sy(val)))
        b.append(text(x0 - 8, sy(val) + 4, f"{val:.2f}", 10, "f-muted", MUTED, "end", family=MONO))
    for t in (1, 10, 20, 30, 40):
        b.append(text(sx(t), y1 + 16, str(t), 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "step t", 11.5, "f-muted", MUTED, "middle"))
    b.append(line(x0, sy(1.0), x1, sy(1.0), INK, "f-ink-stroke", 2, ' stroke-dasharray="5 3"'))
    b.append(text(x1, sy(1.0) - 8, "both corrected estimates: 1.0", 11.5, anchor="end", weight=700))
    for series, col, cls, name in ((m, CORAL, "f-coral", "raw m"), (v, TEAL, "f-teal", "raw v")):
        pts = " ".join(f"{sx(t + 1):.1f},{sy(val):.1f}" for t, val in enumerate(series))
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" class="{cls}-stroke" stroke-width="2.5"><title>{name}</title></polyline>')
    b.append(text(sx(22), sy(m[21]) + 20, "raw m = 1 − 0.9ᵗ", 11.5, weight=700))
    b.append(text(sx(14), sy(v[13]) + 20, "raw v = 1 − 0.95ᵗ", 11.5, weight=700))
    return svg("fig07-1-bias-correction.svg", w, h, "Figure 7.1 · Bias correction",
               f"Raw first moment after 1, 5 and 40 steps: {m[0]:.3f}, {m[4]:.3f}, {m[-1]:.3f}; raw second moment "
               f"{v[0]:.3f}, {v[4]:.3f}, {v[-1]:.3f}. Both corrected estimates equal 1.", b)


def fig_07_schedules():
    from anulm.training.train import lr_at
    wsd = SimpleNamespace(lr=6e-4, min_lr=6e-5, warmup=2000, steps=1_900_000, schedule="wsd",
                          decay_frac=0.1631579, rewarmup=0)
    cos = SimpleNamespace(**{**vars(wsd), "schedule": "cosine"})
    close(lr_at(1_000_000, cos), 3.081e-4, 5e-8, "cosine at 1M")
    close(lr_at(1_700_000, wsd), 4.084e-4, 5e-8, "wsd at 1.7M")
    steps = list(range(0, 1_900_000, 5000)) + [1_899_999]
    w, h = 760, 320
    b = []
    _title(b, "Learning rate over the released run: WSD as trained (teal), cosine for comparison (coral)",
           "train.py lr_at · lr 6e-4, min_lr 6e-5, warmup 2,000, 1.9M steps, decay over the last 16.3% · verified")
    x0, x1, y0, y1 = 80, 700, 76, 266
    sx = lambda s: x0 + (x1 - x0) * s / 1_900_000
    sy = lambda v: y1 - (y1 - y0) * v / 6.5e-4
    for v in (0, 2e-4, 4e-4, 6e-4):
        b.append(line(x0, sy(v), x1, sy(v)))
        b.append(text(x0 - 8, sy(v) + 4, f"{v * 1e4:.0f}e-4" if v else "0", 10, "f-muted", MUTED, "end", family=MONO))
    for s in (0, 500_000, 1_000_000, 1_500_000, 1_900_000):
        b.append(text(sx(s), y1 + 16, f"{s / 1e6:.1f}M", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "step", 11.5, "f-muted", MUTED, "middle"))
    for args, col, cls, name in ((cos, CORAL, "f-coral", "cosine"), (wsd, TEAL, "f-teal", "WSD")):
        pts = " ".join(f"{sx(s):.1f},{sy(lr_at(s, args)):.1f}" for s in steps)
        b.append(f'<polyline points="{pts}" fill="none" stroke="{col}" class="{cls}-stroke" stroke-width="2.5"><title>{name}</title></polyline>')
    ds = int(wsd.steps * (1 - wsd.decay_frac))
    b.append(line(sx(ds), y0, sx(ds), y1, INK, "f-ink-stroke", 1, ' stroke-dasharray="3 3"'))
    b.append(text(sx(ds) - 6, y0 + 12, f"decay starts at {ds:,}", 11, anchor="end", weight=700))
    return svg("fig07-2-schedules.svg", w, h, "Figure 7.2 · Two schedules",
               "WSD stays at 6e-4 until step 1,589,999 and falls linearly to 6e-5; cosine falls from 6e-4 from step "
               "2,000, reaching 3.08e-4 at 1M steps.", b)
