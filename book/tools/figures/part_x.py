"""
Part X figures (Chapter 41, the capstone).

    python book/tools/make_figures.py part_x

- fig41-1: the four checkpoints, each with the number it should print;
- fig41-2: the recorded nano_30b run (README.md "Status: trained and verified"),
  with ln V and the measured loss before any step;
- fig41-3: how many tokens top-p keeps on the released model, measured by
  code-examples/ch41_checkpoints.py (book/measurements/part_x_released.json).

Every number drawn is computed here or read from its source, and asserted
against the value the chapter prints.
"""

from __future__ import annotations

import json
import math

from make_figures import (BOOK, CARD, CORAL, INK, MONO, MUTED, PAPER, RULE, SERIF, TEAL, close,
                          line, rect, svg, text)

# The recorded run, as README.md prints it at @773b6a8 ("Status: trained and verified").
README_LOG = [(0, 5.5869), (50, 2.7887), (250, 1.6047), (500, 1.3917), (999, 1.2182)]
README_VAL = (999, 1.5190)
CI_VAL_50 = 2.7868          # docs/TUTORIAL.md: the 50-step CI run, Ubuntu
FIRST_VAL = 5.6090          # ch41_checkpoints.py first-loss, before any step
FIRST_STEP0 = 5.6203        # ch41_checkpoints.py first-loss: today's train.py, first batch


def arrow(x1, y1, x2, y2):
    return line(x1, y1, x2, y2, MUTED, "f-muted-stroke", 1.5, ' marker-end="url(#arrow)"')


def fig_checkpoints():
    tests = json.loads((BOOK / "measurements" / "part_x_tests.json").read_text(encoding="utf-8"))
    assert tests["summary"] == "66 passed, 0 failed, 66 total", tests["summary"]
    rel = json.loads((BOOK / "measurements" / "part_x_released.json").read_text(encoding="utf-8"))
    close(rel["num_params"][0] / 1e6, 397.7, 0.05, "released total")
    close(FIRST_STEP0, 5.62, 0.005, "step-0 loss")
    close(math.log(259), 5.56, 0.005, "ln 259")
    w, h = 760, 300
    b = [text(20, 28, "The capstone in four checkpoints", 15, weight=700, family=SERIF),
         text(20, 48, "each one prints a number you can compare against the book's", 12, "f-muted", MUTED)]
    cards = [
        ("1 · train", "train.py --preset 30b", ["loss 5.62 at step 0", "(ln 259 = 5.56)", "val 1.52 ± 0.03"]),
        ("2 · trace", "ch41_shapes.py", ["(2, 5) ids", "→ (2, 5, 32) stream", "→ (2, 1, 64) logits"]),
        ("3 · extend", "ch41_top_p.py", ["top-p: 3 tests pass", "off by one: fails 3", "no renorm: fails 1"]),
        ("4 · test", "test_model.py", ["66 passed, 0 failed", "then the released", "model, 397.7M"]),
    ]
    x, cw = 20, 170
    for i, (title, cmd, lines_) in enumerate(cards):
        cx = x + i * (cw + 10)
        ext = i == 2
        extra = (f' stroke="{TEAL}" stroke-width="2" stroke-dasharray="6 4"' if ext
                 else f' stroke="{RULE}" stroke-width="1"')
        b.append(rect(cx, 70, cw, 190, CARD, "f-card", 8, extra,
                      tip=f"checkpoint {title}: {cmd}"))
        b.append(text(cx + 14, 96, title, 14, weight=700))
        b.append(text(cx + 14, 118, cmd, 11, "f-muted", MUTED, family=MONO))
        b.append(line(cx + 14, 132, cx + cw - 14, 132))
        for j, s in enumerate(lines_):
            b.append(text(cx + 14, 158 + j * 22, s, 12.5, family=MONO if j == 0 else "Lato, 'Segoe UI', sans-serif"))
        if ext:
            b.append(text(cx + 14, 246, "EXTENSION", 10.5, "f-muted", MUTED, weight=700))
        if i < 3:
            b.append(arrow(cx + cw + 1, 165, cx + cw + 9, 165))
    b.append(text(20, 286, "Dashed: code this chapter adds; it is not in the repository.", 11.5, "f-muted", MUTED))
    return svg("fig41-1-checkpoints.svg", w, h, "Figure 41.1 · The four checkpoints",
               "Four cards in a row. Train: loss 5.62 at step 0 against ln 259 = 5.56, val 1.52 ± 0.03 after 1,000 steps. "
               "Trace: (2, 5) ids become a (2, 5, 32) stream and (2, 1, 64) logits. Extend, dashed as an extension: "
               "top-p passes three tests, the off-by-one version fails all three and the unnormalised one fails one. "
               "Test: 66 passed, then the released 397.7M model.", b)


def fig_loss():
    close(math.log(259), 5.5568, 5e-5, "ln 259")
    w, h = 760, 380
    x0, x1, y0, y1 = 80, 700, 70, 320
    sx = lambda s: x0 + (x1 - x0) * s / 1000
    sy = lambda v: y1 - (y1 - y0) * (v - 1.0) / (6.0 - 1.0)
    b = [text(20, 28, "nano_30b on tinyshakespeare: the recorded run", 15, weight=700, family=SERIF),
         text(20, 48, "README.md's log at @773b6a8 · batch 16 × 256 bytes · CPU · 1,000 steps", 12, "f-muted", MUTED)]
    for v in (1, 2, 3, 4, 5, 6):
        b.append(line(x0, sy(v), x1, sy(v)))
        b.append(text(x0 - 10, sy(v) + 4, f"{v:.0f}", 10, "f-muted", MUTED, "end", family=MONO))
    for s in (0, 250, 500, 750, 1000):
        b.append(text(sx(s), y1 + 18, f"{s}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 36, "step", 11, "f-muted", MUTED, "middle"))
    b.append(text(24, (y0 + y1) / 2, "loss, nats per byte", 11, "f-muted", MUTED, "middle",
                  extra=f' transform="rotate(-90 24 {(y0 + y1) / 2})"'))
    lv = math.log(259)
    b.append(line(x0, sy(lv), x1, sy(lv), INK, "f-ink-stroke", 1.2, ' stroke-dasharray="4 3"'))
    b.append(text(x1 - 4, sy(lv) - 8, "ln 259 = 5.557: a uniform guess", 11, anchor="end", weight=700))
    pts = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in README_LOG)
    b.append(f'<polyline points="{pts}" fill="none" stroke="{CORAL}" stroke-width="2" class="f-coral-stroke"/>')
    for s, v in README_LOG:
        b.append(f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="4.5" fill="{CORAL}" stroke="{PAPER}" stroke-width="1.5" '
                 f'class="f-coral f-ring"><title>training loss at step {s}: {v}</title></circle>')
        dx, dy = ((10, 22) if s == 0 else (-12, 18) if s == 999 else (8, -8))
        label = f"{v:.4f}" + (" train · 2.7868 val, the 50-step CI run" if s == 50 else "")
        b.append(text(sx(s) + dx, sy(v) + dy, label, 10.5, anchor="end" if s == 999 else "start", family=MONO))
    for s, v, tip in ((README_VAL[0], README_VAL[1], "val loss after 1,000 steps (README.md)"),
                      (49, CI_VAL_50, "val loss of the 50-step CI run (docs/TUTORIAL.md)"),
                      (0, FIRST_VAL, "val loss before any step (ch41_checkpoints.py)")):
        b.append(rect(sx(s) - 5, sy(v) - 5, 10, 10, TEAL, "f-teal", 1, tip=f"{tip}: {v}"))
    b.append(text(sx(999) - 10, sy(README_VAL[1]) - 8, "val 1.5190", 10.5, anchor="end", family=MONO))
    b.append(text(sx(0) + 10, sy(FIRST_VAL) - 8, "val 5.6090, and 5.6203 on the first batch, with today's train.py",
                  10.5, family=MONO))
    b.append(f'<circle cx="{sx(0):.1f}" cy="{sy(FIRST_STEP0):.1f}" r="4.5" fill="{PAPER}" stroke="{CORAL}" '
             f'stroke-width="2" class="f-paper f-coral-stroke"><title>first-batch loss with today\'s train.py: '
             f'{FIRST_STEP0}</title></circle>')
    # legend
    lx, ly = x0 + 380, sy(3.6)
    b.append(f'<circle cx="{lx}" cy="{ly}" r="4.5" fill="{CORAL}" class="f-coral"/>')
    b.append(text(lx + 12, ly + 4, "training loss, one batch (README.md)", 11))
    b.append(rect(lx - 5, ly + 15, 10, 10, TEAL, "f-teal", 1))
    b.append(text(lx + 12, ly + 24, "validation loss", 11))
    return svg("fig41-2-loss.svg", w, h, "Figure 41.2 · The run you are reproducing",
               "Training loss of the recorded nano_30b run falls from 5.5869 at step 0, close to ln 259 = 5.557, "
               "to 2.7887 at step 50, 1.6047 at 250, 1.3917 at 500 and 1.2182 at 999. With today's train.py the "
               "first batch scores 5.6203; validation loss is 5.6090 "
               "before any step, 2.7868 after the 50-step CI run and 1.5190 after 1,000 steps.", b)


def fig_nucleus():
    r = json.loads((BOOK / "measurements" / "part_x_released.json").read_text(encoding="utf-8"))
    n9, n5 = r["nucleus"]["0.9"], r["nucleus"]["0.5"]
    assert n9["percentiles"]["50"] == 501 and n5["percentiles"]["50"] == 3, (n9["percentiles"], n5["percentiles"])
    assert r["precision"].startswith("fp32") and r["data"] == "data/v2/val.u16", r["precision"]
    close(n9["share_size_1"], 0.131, 5e-4, "p=0.9 size 1")
    close(n9["share_over_50"], 0.644, 5e-4, "p=0.9 over 50")
    assert r["positions"] == 4092
    edges = n9["hist_upper_edges"]
    w, h = 760, 400
    x0, x1, y0, y1 = 70, 730, 80, 320
    n = len(edges)
    bw = (x1 - x0) / n
    top = max(max(n9["hist"]), max(n5["hist"]))
    sy = lambda c: y1 - (y1 - y0) * c / 1500
    assert top <= 1500
    b = [text(20, 28, "How many tokens top-p keeps, position by position", 15, weight=700, family=SERIF),
         text(20, 48, f"AnuLM-Base-v2-400M · {r['positions']:,} positions of data/v2/val.u16 (8 windows × 512) · fp32 · measured",
              12, "f-muted", MUTED)]
    for c in (0, 500, 1000, 1500):
        b.append(line(x0, sy(c), x1, sy(c)))
        b.append(text(x0 - 8, sy(c) + 4, f"{c}", 10, "f-muted", MUTED, "end", family=MONO))
    for i, e in enumerate(edges):
        lo = e // 2 + 1 if e > 1 else 1
        label = f"{lo}" if lo == e else f"{lo}–{e}"
        for j, (series, col, cls, p) in enumerate(((n5, CORAL, "f-coral", 0.5), (n9, TEAL, "f-teal", 0.9))):
            c = series["hist"][i]
            bx = x0 + i * bw + 3 + j * (bw - 6) / 2
            b.append(rect(bx, sy(c), (bw - 6) / 2 - 1, max(y1 - sy(c), 0.5), col, cls, 1,
                          tip=f"top-p {p}: {c} positions keep {label} tokens"))
        if i % 2 == 0 or i == n - 1:
            b.append(text(x0 + (i + 0.5) * bw, y1 + 16, str(e), 9.5, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "tokens kept (upper edge of each doubling bin: 1, 2, 3–4, 5–8, …)",
                  11, "f-muted", MUTED, "middle"))
    # sample.py's top-k 50 falls in the bin 33-64
    k = next(i for i, e in enumerate(edges) if e >= 50)
    xk = x0 + (k + (50 - edges[k] // 2) / (edges[k] / 2)) * bw
    b.append(line(xk, y0 - 6, xk, y1, INK, "f-ink-stroke", 1.2, ' stroke-dasharray="4 3"'))
    b.append(text(xk + 6, y0 + 4, "top-k 50, sample.py's default", 11, weight=700))
    b.append(rect(x0 + 380, y0 + 22, 10, 10, CORAL, "f-coral", 1))
    b.append(text(x0 + 396, y0 + 31, "p = 0.5: median 3 tokens", 11))
    b.append(rect(x0 + 380, y0 + 42, 10, 10, TEAL, "f-teal", 1))
    b.append(text(x0 + 396, y0 + 51, "p = 0.9: median 501, 13.1% need just 1", 11))
    return svg("fig41-3-nucleus.svg", w, h, "Figure 41.3 · Nucleus sizes on the released model",
               "Paired histograms of how many tokens top-p keeps at each of 4,092 held-out positions. At p = 0.5 most "
               "positions keep 1 to 4 tokens, median 3. At p = 0.9 the spread runs from 1 token (13.1% of positions) "
               "to over 16,000, median 501; 64.4% of positions keep more than the 50 tokens sample.py's top-k allows.", b)
