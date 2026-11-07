"""
Figures for Part V, second half: Chapters 24 (measuring progress), 25 (checkpoints)
and 26 (debugging).

    python book/tools/make_figures.py part_v_b

Every number drawn here is read from the measurement files the chapters quote
(book/measurements/part_v_b_*.json, written by the book's scripts in
code-examples/) and asserted against the value the chapter prints.
"""

from __future__ import annotations

import json
import math

from make_figures import (BOOK, CARD, CORAL, GOLD, INK, MONO, MUTED, PAPER, RULE, SERIF, TEAL,  # noqa: F401
                          close, esc, line, rect, svg, text)

MEAS = BOOK / "measurements"


def _load(name):
    return json.loads((MEAS / f"part_v_b_{name}.json").read_text(encoding="utf-8"))


def _dot(x, y, r, fill, cls, tip):
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{PAPER}" stroke-width="1.2" '
            f'class="{cls} f-ring"><title>{esc(tip)}</title></circle>')


def _xaxis(b, x0, x1, y, lo, hi, ticks, fmt, label=None):
    sx = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    for t in ticks:
        b.append(line(sx(t), y - 4, sx(t), y))
        b.append(text(sx(t), y + 14, fmt(t), 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(line(x0, y, x1, y))
    if label:
        b.append(text((x0 + x1) / 2, y + 32, label, 11, "f-muted", MUTED, "middle"))
    return sx


# --------------------------------------------------------------------------- Ch 24
def fig_24_windows():
    r = _load("val_noise")
    pos, edges = r["positions"], r["positions"]["edges"]
    n = r["val_tokens"]
    assert n == 113_221_976 and len(pos["random80_seed0"]) == 80 and len(pos["strided80"]) == 80
    w, h = 760, 280
    x0, x1 = 30, 730
    sx = lambda v: x0 + (x1 - x0) * v / n
    b = [text(20, 28, "Where 80 evaluation windows land in the released model's held-out split", 15, weight=700, family=SERIF),
         text(20, 48, "data/v2/val.u16, 113.2M tokens, written one source after another · positions computed by train.py",
              12, "f-muted", MUTED)]
    names = {"english": "English 53.0%", "hindi_web": "Hindi web 40.4%", "hindi_wiki": "", "python": "Python 6.1%"}
    for i, (k, (a, z)) in enumerate(edges.items()):
        fill, cls = (CARD, "f-card") if i % 2 == 0 else (RULE, "f-rule-fill")
        b.append(rect(sx(a), 120, max(sx(z) - sx(a), 1.5), 70, fill, cls,
                      tip=f"{k}: tokens {a:,} to {z:,} ({100 * (z - a) / n:.1f}%)"))
        if names[k]:
            b.append(text((sx(a) + sx(z)) / 2, 112, names[k], 11.5, anchor="middle", weight=700))
    b.append(text(sx(edges["hindi_wiki"][0]) - 2, 205, "Hindi Wikipedia 0.5% ↑", 10.5, "f-muted", MUTED, "end"))
    b.append(text(x0, 92, "token 0", 10, "f-muted", MUTED, family=MONO))
    b.append(text(x1, 92, f"token {n:,}", 10, "f-muted", MUTED, "end", family=MONO))
    b.append(rect(x0, 222, 10, 10, CORAL, "f-coral"))
    b.append(text(x0 + 16, 231, "upper row: random, eval_windows(seed 0), 20 batches × 4", 11.5, weight=700))
    b.append(rect(x0 + 380, 222, 10, 10, TEAL, "f-teal"))
    b.append(text(x0 + 396, 231, "lower row: strided, strided_windows(n = 80)", 11.5, weight=700))
    for p in pos["random80_seed0"]:
        b.append(rect(sx(p) - 1, 126, 2.2, 26, CORAL, "f-coral", tip=f"random window at token {p:,}"))
    for p in pos["strided80"]:
        b.append(rect(sx(p) - 1, 158, 2.2, 26, TEAL, "f-teal", tip=f"strided window at token {p:,}"))
    cnt = {k: sum(1 for p in pos["random80_seed0"] if a <= p < z) for k, (a, z) in edges.items()}
    cs = {k: sum(1 for p in pos["strided80"] if a <= p < z) for k, (a, z) in edges.items()}
    b.append(text(x0, 258, "windows per source · random: " + ", ".join(f"{k} {v}" for k, v in cnt.items())
                  + " · strided: " + ", ".join(f"{k} {v}" for k, v in cs.items()), 10.5, "f-muted", MUTED))
    return svg("fig24-1-windows.svg", w, h, "Figure 24.1 · Random and strided windows",
               "The held-out split drawn as four source bands, English, Hindi web, Hindi Wikipedia and Python. "
               "Eighty random windows (seed 0) fall unevenly across it; eighty strided windows are evenly spaced, so "
               "each source receives windows in proportion to its share. Window counts per source are printed below.", b)


def fig_24_noise():
    r = _load("val_noise")
    close(r["strided400"]["mean"], 3.3523, 5e-5, "strided 400 mean")
    close(r["strided400"]["sem"], 0.0491, 5e-5, "strided 400 sem")
    close(r["random80"]["min"], 3.2508, 5e-5, "random80 min")
    close(r["random80"]["max"], 3.4436, 5e-5, "random80 max")
    close(r["stratified"]["mean"], 3.3680, 5e-5, "stratified mean")
    w, h = 760, 360
    x0, x1, lo, hi = 250, 730, 3.15, 3.55
    b = [text(20, 28, "The released model's val loss on fifteen window sets", 15, weight=700, family=SERIF),
         text(20, 48, "ckpt_base_v2.pt on its own held-out split · 1,024-token windows · bf16 · each bar is ± one standard "
                      "error as evaluate() reports it", 12, "f-muted", MUTED)]
    rows = [("random · 80 windows · 10 seeds", [(s["mean"], s["sem"], f"seed {s['seed']}") for s in r["random80"]["sets"]], CORAL, "f-coral"),
            ("random · 400 windows · 3 seeds", [(s["mean"], s["sem"], f"seed {s['seed']}") for s in r["random400"]["sets"]], CORAL, "f-coral"),
            ("strided · 400 windows (recorded)", [(r["strided400"]["mean"], r["strided400"]["sem"], "batch 4")], TEAL, "f-teal"),
            ("per source · 1,100 windows", [(r["stratified"]["mean"], r["stratified"]["sem"], "stratified")], TEAL, "f-teal")]
    y0 = 82
    sx = _xaxis(b, x0, x1, y0 + 4 * 52 + 6, lo, hi, [3.15, 3.20, 3.25, 3.30, 3.35, 3.40, 3.45, 3.50, 3.55], lambda t: f"{t:.2f}",
                "val loss, nats per token")
    for i, (label, pts, col, cls) in enumerate(rows):
        y = y0 + i * 52 + 20
        b.append(text(x0 - 14, y + 4, label, 12, anchor="end"))
        for j, (mu, se, tip) in enumerate(pts):
            yy = y + (j - (len(pts) - 1) / 2) * 3.6 if len(pts) > 1 else y
            b.append(line(sx(mu - se), yy, sx(mu + se), yy, MUTED, "f-muted-stroke", 1))
            b.append(_dot(sx(mu), yy, 4.5, col, cls, f"{label}, {tip}: {mu:.4f} ± {se:.4f}"))
    b.append(text(20, h - 10, "80-window sets span 3.2508 to 3.4436; the recorded number is 3.3523 ± 0.0491.",
                  11, "f-muted", MUTED))
    return svg("fig24-2-noise.svg", w, h, "Figure 24.2 · How much a val loss moves with its windows",
               "Ten random sets of 80 windows give val losses from 3.2508 to 3.4436, each with about ± 0.067. "
               "Three random sets of 400 windows give 3.353 to 3.391 with about ± 0.029. The recorded strided set gives "
               "3.3523 ± 0.0491, and a per-source stratified estimate 3.3680 ± 0.0169.", b)


def fig_24_sources():
    r = _load("val_noise")
    per = r["per_source"]
    close(per["english"]["bits_per_byte"], 1.175, 5e-4, "english bpb")
    close(per["hindi_web"]["bits_per_byte"], 0.497, 5e-4, "hindi web bpb")
    close(r["bits_per_byte_train_py"], 0.7560, 5e-5, "train.py bpb")
    w, h = 760, 300
    names = {"english": "English", "hindi_web": "Hindi web", "hindi_wiki": "Hindi Wikipedia", "python": "Python"}
    b = [text(20, 28, "Per token or per byte: the same model, four sources", 15, weight=700, family=SERIF),
         text(20, 48, "ckpt_base_v2.pt · strided windows inside each source · bytes per token from data/v2/meta.json",
              12, "f-muted", MUTED)]

    def panel(px, title, key, hi, ticks, fmt, total):
        b.append(text(px, 78, title, 13, weight=700))
        bx0, bx1 = px + 120, px + 330
        sx = lambda v: bx0 + (bx1 - bx0) * v / hi
        for t in ticks:
            b.append(line(sx(t), 90, sx(t), 90 + 4 * 34))
            b.append(text(sx(t), 90 + 4 * 34 + 14, fmt(t), 10, "f-muted", MUTED, "middle", family=MONO))
        b.append(line(sx(total), 86, sx(total), 90 + 4 * 34, INK, "f-ink-stroke", 1.5, ' stroke-dasharray="3 3"'))
        for i, (k, v) in enumerate(per.items()):
            y = 96 + i * 34
            b.append(text(bx0 - 10, y + 14, names[k], 12, anchor="end"))
            b.append(rect(bx0, y, max(sx(v[key]) - bx0, 1), 20, GOLD, "f-gold", 3, tip=f"{names[k]}: {v[key]:.3f}"))
            b.append(rect(sx(v[key]) + 3, y + 2, 40, 16, PAPER, "f-paper"))
            b.append(text(sx(v[key]) + 6, y + 14, f"{v[key]:.3f}", 11.5, weight=700, family=MONO))
        b.append(text(sx(total), 90 + 4 * 34 + 30, f"whole split {total:.3f}", 10.5, weight=700, anchor="middle"))

    panel(20, "nats per token", "loss", 4.5, [0, 1, 2, 3, 4], lambda t: f"{t:.0f}", r["strided400"]["mean"])
    panel(390, "bits per byte", "bits_per_byte", 1.4, [0, 0.4, 0.8, 1.2], lambda t: f"{t:.1f}", r["bits_per_byte_train_py"])
    b.append(text(20, h - 18, "bytes per token: English 4.43, Hindi web 9.37, Hindi Wikipedia 9.34, Python 4.04; whole split 6.40",
                  11, "f-muted", MUTED))
    return svg("fig24-3-sources.svg", w, h, "Figure 24.3 · Loss per token and per byte, by source",
               "Per token, Hindi web (3.231) scores better than English (3.605) and Python is lowest (2.173). Per byte the "
               "order changes: Hindi web 0.497, Python 0.776, Hindi Wikipedia 0.619, English 1.175, because a Hindi token "
               "covers about twice the bytes of an English one. The whole split reads 3.352 nats per token, 0.756 bits "
               "per byte.", b)


# --------------------------------------------------------------------------- Ch 25
def fig_25_atomic():
    w, h = 760, 330
    b = [text(20, 28, "Writing a checkpoint so that a kill cannot destroy it", 15, weight=700, family=SERIF),
         text(20, 48, "two lanes, the same kill at the same moment · the file names are train.py's", 12, "f-muted", MUTED)]

    def box(x, y, ww, title, sub, fill=CARD, cls="f-card", stroke=RULE):
        out = [rect(x, y, ww, 46, fill, cls, 6, f' stroke="{stroke}" stroke-width="1.2"')]
        out.append(text(x + ww / 2, y + 20, title, 12.5, anchor="middle", weight=700))
        out.append(text(x + ww / 2, y + 37, sub, 10.5, "f-muted", MUTED, "middle", family=MONO))
        return out

    def arrow(x1, y1, x2, y2):
        return line(x1, y1, x2, y2, MUTED, "f-muted-stroke", 1.5, ' marker-end="url(#arrow)"')

    for lane, y, title in ((0, 80, "torch.save(obj, path)  — in place"), (1, 200, "atomic_save(obj, path)  — write beside, then rename")):
        b.append(text(20, y - 8, title, 13, weight=700, family=MONO))
    b += box(20, 90, 150, "ckpt.pt.last", "old, complete")
    b.append(arrow(170, 113, 208, 113))
    b += box(210, 90, 190, "ckpt.pt.last", "opened for writing")
    b.append(arrow(400, 113, 438, 113))
    b += box(440, 90, 130, "half written", "kill arrives", stroke=CORAL)
    b.append(arrow(570, 113, 598, 113))
    b += box(600, 90, 140, "unreadable", "no central directory", stroke=CORAL)
    b += box(20, 210, 150, "ckpt.pt.last", "old, complete")
    b.append(arrow(170, 233, 208, 233))
    b += box(210, 210, 190, "ckpt.pt.last.tmp", "torch.save writes here")
    b.append(arrow(400, 233, 438, 233))
    b += box(440, 210, 130, "half written", "kill arrives", stroke=CORAL)
    b.append(arrow(570, 233, 598, 233))
    b += box(600, 210, 140, "ckpt.pt.last", "old, still complete", stroke=TEAL)
    b.append(text(210, 280, "without the kill: os.replace(tmp, path) swaps the finished file in, in one step",
                  11.5, "f-muted", MUTED))
    b.append(text(210, 298, "(atomic within a volume, as the code's comment says)", 11.5, "f-muted", MUTED))
    return svg("fig25-2-atomic.svg", w, h, "Figure 25.2 · In-place and atomic checkpoint writes",
               "Top lane: torch.save opens the existing checkpoint for writing, a kill leaves it half written and "
               "unreadable. Bottom lane: atomic_save writes a .tmp file beside it, so the same kill leaves the old "
               "checkpoint complete; only a finished write is moved into place by os.replace.", b)


def fig_25_sizes():
    r = _load("resume_point")
    last, best = r["ckpt_base_v2.pt.last"], r["ckpt_base_v2.pt"]
    close(last["file_bytes"] / 1e9, 4.86, 5e-3, ".last size")
    close(best["file_bytes"] / 1e9, 1.59, 5e-3, "best size")
    close(last["tensor_bytes"]["opt"] / 1e9, 3.18, 5e-3, "optimizer bytes")
    w, h = 760, 270
    x0, x1 = 150, 730
    sx = lambda v: (x1 - x0) * v / 5.0e9
    b = [text(20, 28, "What the released run keeps in its two checkpoint files", 15, weight=700, family=SERIF),
         text(20, 48, "ckpt_base_v2.pt (best val, --out) and ckpt_base_v2.pt.last (resume point) · step 1,900,000 · read "
                      "memory-mapped", 12, "f-muted", MUTED)]
    for t in (0, 1e9, 2e9, 3e9, 4e9, 5e9):
        b.append(line(x0 + sx(t), 70, x0 + sx(t), 180))
        b.append(text(x0 + sx(t), 196, f"{t / 1e9:.0f} GB", 10, "f-muted", MUTED, "middle", family=MONO))
    segs = {"best": [("weights + buffers", best["tensor_bytes"]["model"], GOLD, "f-gold")],
            "last": [("weights + buffers", last["tensor_bytes"]["model"], GOLD, "f-gold"),
                     ("AdamW exp_avg", last["opt_bytes_by_kind"]["exp_avg"], TEAL, "f-teal"),
                     ("AdamW exp_avg_sq", last["opt_bytes_by_kind"]["exp_avg_sq"], CORAL, "f-coral"),
                     ("sampler order", last["tensor_bytes"]["sampler"], INK, "f-ink")]}
    for i, (name, label) in enumerate((("best", "ckpt_base_v2.pt"), ("last", "….pt.last"))):
        y = 84 + i * 50
        b.append(text(x0 - 12, y + 19, label, 12, anchor="end", family=MONO))
        x = x0
        for seg, v, col, cls in segs[name]:
            ww = max(sx(v), 2)
            b.append(rect(x, y, ww - 1, 28, col, cls, 2, tip=f"{seg}: {v / 1e9:.2f} GB"))
            if ww > 70:
                b.append(text(x + ww / 2, y + 18, f"{seg} {v / 1e9:.2f}", 11, "f-onfill", "#13232A", "middle", 700))
            x += ww
    b.append(text(x0, 222, "The sampler's order tensor is 84 MiB, too thin to label: 11,047,015 window starts, of which "
                           "7,600,000 had been used.", 11, "f-muted", MUTED))
    b.append(text(x0, 240, "AdamW keeps two moments per weight, so the optimizer state is twice the model.",
                  11, "f-muted", MUTED))
    return svg("fig25-1-sizes.svg", w, h, "Figure 25.1 · What a resume point holds",
               "The best-val file holds 1.59 GB of weights and buffers. The resume point holds the same 1.59 GB plus "
               "AdamW's two moment tensors, 1.59 GB each, and the sampler's 84 MiB window order: 4.86 GB on disk.", b)


def fig_25_curves():
    r = _load("resume_curves")["part1"]
    A, B, C = ({int(k): v for k, v in r[n].items()} for n in "ABC")
    assert all(A[s] == B[s] for s in A), "B must reproduce A exactly"
    close(A[30], 1.7326, 5e-5, "A at step 30")
    close(C[30], 2.1005, 5e-5, "C at step 30")
    w, h = 760, 360
    x0, x1, y0, y1 = 70, 730, 116, 300
    lo, hi = 1.0, 5.6
    sx = lambda s: x0 + (x1 - x0) * s / 59
    sy = lambda v: y1 - (y1 - y0) * (v - lo) / (hi - lo)
    b = [text(20, 28, "Resuming exactly, and resuming without the optimizer", 15, weight=700, family=SERIF),
         text(20, 48, "train.py --preset tiny-gqa on a byte corpus, CPU · logged training loss per step · book example "
                      "ch25_resume.py", 12, "f-muted", MUTED)]
    for t in (1, 2, 3, 4, 5):
        b.append(line(x0, sy(t), x1, sy(t)))
        b.append(text(x0 - 8, sy(t) + 4, f"{t}", 10, "f-muted", MUTED, "end", family=MONO))
    for s in (0, 10, 20, 30, 40, 50, 59):
        b.append(text(sx(s), y1 + 16, f"{s}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "step", 11, "f-muted", MUTED, "middle"))
    b.append(line(sx(29.5), y0 - 6, sx(29.5), y1, INK, "f-ink-stroke", 1, ' stroke-dasharray="3 3"'))
    b.append(text(sx(29.5) + 6, y0 - 8, "stopped after step 29, resumed at 30", 11, weight=700))
    for s in range(60):
        b.append(_dot(sx(s), sy(A[s]), 3.2, TEAL, "f-teal", f"A and B (identical), step {s}: {A[s]:.4f}"))
    for s in range(30, 60):
        b.append(_dot(sx(s), sy(C[s]), 3.2, CORAL, "f-coral", f"C, after the fallback, step {s}: {C[s]:.4f}"))
    b.append(_dot(26, 70, 4, TEAL, "f-teal", "legend"))
    b.append(text(36, 74, "A uninterrupted, and B stopped and resumed: the same numbers", 11.5))
    b.append(_dot(26, 88, 4, CORAL, "f-coral", "legend"))
    b.append(text(36, 92, "C resumed from the best-val file (A's weights at step 29): fresh moments, restarted data order",
                  11.5))
    return svg("fig25-3-curves.svg", w, h, "Figure 25.3 · Exact resume, and the fallback's dent",
               "Training loss of each step's batch before its update, for 60 steps. Run B, stopped after step 29 and "
               "resumed, logs exactly run A's numbers. Run C starts step 30 with A's weights but a restarted sampler, so "
               "its step-30 loss of 2.10 against A's 1.73 is a different batch; its fresh optimizer moments act from "
               "step 31, and it ends with a worse val loss on the same windows.", b)


# --------------------------------------------------------------------------- Ch 26
def fig_26_leak():
    r = _load("silent_failures")["curves"]
    ok, leak = r["committed"], r["leaky"]
    close(leak[-1], 0.218, 5e-4, "leaky final")
    close(ok[-1], 4.158, 5e-4, "committed final")
    w, h = 760, 320
    x0, x1, y0, y1 = 70, 730, 70, 262
    hi = 4.6
    sx = lambda s: x0 + (x1 - x0) * s / (len(ok) - 1)
    sy = lambda v: y1 - (y1 - y0) * v / hi
    b = [text(20, 28, "A model that cheats has the better loss", 15, weight=700, family=SERIF),
         text(20, 48, "tiny() trained on uniform random ids, where no model can beat ln 64 = 4.159 honestly · book "
                      "example ch26_silent_failures.py", 12, "f-muted", MUTED)]
    for t in (0, 1, 2, 3, 4):
        b.append(line(x0, sy(t), x1, sy(t)))
        b.append(text(x0 - 8, sy(t) + 4, f"{t}", 10, "f-muted", MUTED, "end", family=MONO))
    for s in (0, 50, 100, 150, 199):
        b.append(text(sx(s), y1 + 16, f"{s + 1}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 34, "training step", 11, "f-muted", MUTED, "middle"))
    b.append(line(x0, sy(math.log(64)), x1, sy(math.log(64)), INK, "f-ink-stroke", 1, ' stroke-dasharray="3 3"'))
    for s in range(0, len(ok), 2):
        b.append(_dot(sx(s), sy(ok[s]), 2.6, TEAL, "f-teal", f"as committed, step {s + 1}: {ok[s]:.3f}"))
        b.append(_dot(sx(s), sy(leak[s]), 2.6, CORAL, "f-coral", f"causal mask removed, step {s + 1}: {leak[s]:.3f}"))
    b.append(text(x1, sy(math.log(64)) - 8, "as committed: stays at ln V", 11.5, anchor="end", weight=700))
    b.append(text(sx(8), sy(1.6), "mask removed: each position reads", 11.5, weight=700))
    b.append(text(sx(8), sy(1.6) + 16, "the id it must predict, and \"learns\"", 11.5, weight=700))
    return svg("fig26-1-leak.svg", w, h, "Figure 26.1 · The loss rewards a leaking mask",
               "Two training curves on random ids. The committed model stays at ln 64 = 4.159, the best any honest model "
               "can do. The same model with its causal mask removed falls to 0.218 by step 200, because each position can "
               "read the id it is asked to predict.", b)


def fig_26_router():
    r = _load("router_bf16")
    per = {int(k): v["set_differs"] for k, v in r["per_layer"].items()}
    close(r["all_layers_set_differs"], 0.0317, 5e-5, "all layers")
    close(per[1], 0.1095, 5e-5, "layer 1")
    close(per[19], 0.0169, 5e-5, "layer 19")
    w, h = 760, 330
    x0, x1, y0, y1 = 70, 730, 70, 250
    hi = 0.12
    bw = (x1 - x0) / 19
    sy = lambda v: y1 - (y1 - y0) * v / hi
    b = [text(20, 28, "Expert choices that bf16 autocast would change, per layer", 15, weight=700, family=SERIF),
         text(20, 48, "ckpt_base_v2.pt · 32 held-out windows of 1,024 tokens · share of tokens whose top-4 set differs "
                      "from the fp32 gate", 12, "f-muted", MUTED)]
    for t in (0, 0.04, 0.08, 0.12):
        b.append(line(x0, sy(t), x1, sy(t)))
        b.append(text(x0 - 8, sy(t) + 4, f"{100 * t:.0f}%", 10, "f-muted", MUTED, "end", family=MONO))
    for i, (l, v) in enumerate(sorted(per.items())):
        x = x0 + i * bw + 4
        b.append(rect(x, sy(v), bw - 8, y1 - sy(v), GOLD, "f-gold", 2, tip=f"layer {l}: {100 * v:.2f}%"))
        b.append(text(x + (bw - 8) / 2, y1 + 14, f"{l}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 32, "layer", 11, "f-muted", MUTED, "middle"))
    avg = r["all_layers_set_differs"]
    b.append(line(x0, sy(avg), x1, sy(avg), INK, "f-ink-stroke", 1.5, ' stroke-dasharray="3 3"'))
    b.append(text(x1, sy(avg) - 6, f"all layers {100 * avg:.2f}%", 11.5, anchor="end", weight=700))
    b.append(text(x0 + bw + 4, sy(per[1]) + 4, f"  layer 1: {100 * per[1]:.2f}%", 11.5, weight=700))
    return svg("fig26-2-router.svg", w, h, "Figure 26.2 · bf16 routing, measured on the released model",
               "For each of the 19 MoE layers, the share of tokens whose four chosen experts change when the gate runs "
               "under bf16 autocast. Layer 1 changes 10.95%, the deepest layer 1.69%; over all layers 3.17% of expert "
               "sets change.", b)
