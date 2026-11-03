"""
Figures for Part V, first half: Chapters 21 (data), 22 (the training loop) and
23 (precision, memory and speed). Accent: teal.

    python book/tools/make_figures.py part_v_a

Every number drawn here is read from its source and asserted against the value
the chapter prints:

- corpus shares and the tail split: book/measurements/part_v_a_corpus.json
  (written by code-examples/ch21_measure_corpus.py from data/v2/meta.json);
- the window grid: train.WindowSampler, run here unchanged;
- the released run's curve: book/measurements/part_v_a_run.json
  (parsed from the run's own log by code-examples/ch22_read_run_log.py);
- number formats: torch.finfo; memory: arithmetic on the released configuration
  and book/measurements/part_v_a_memory.json (code-examples/ch23_memory.py).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from make_figures import (BOOK, CARD, CORAL, INK, MONO, MUTED, NIGHT, PAPER, ROOT, RULE, SERIF,  # noqa: E402
                          SLATE, TEAL, close, esc, line, rect, svg, text)

sys.path.insert(0, str(ROOT))
MEAS = BOOK / "measurements"


def _box(x, y, w, h, title, sub=None, accent=False, dashed=False):
    stroke = f' stroke="{TEAL}" stroke-width="2.5"' if accent else f' stroke="{RULE}" stroke-width="1"'
    if dashed:
        stroke += ' stroke-dasharray="4 3"'
    out = [rect(x, y, w, h, CARD, "f-card", 8, stroke)]
    out.append(text(x + w / 2, y + (20 if sub else h / 2 + 5), title, 12.5, anchor="middle", weight=700))
    if sub:
        for i, s in enumerate(sub if isinstance(sub, list) else [sub]):
            out.append(text(x + w / 2, y + 37 + 14 * i, s, 11, "f-muted", MUTED, "middle", family=MONO))
    return out


def _arrow(x1, y1, x2, y2, label=None, dx=0, dy=-6):
    out = [line(x1, y1, x2, y2, MUTED, "f-muted-stroke", 1.5, ' marker-end="url(#arrow)"')]
    if label:
        out.append(text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, label, 10.5, "f-muted", MUTED, "middle", family=MONO))
    return out


# --------------------------------------------------------------------------- Ch 21
def fig_21_1_pipeline():
    w, h = 760, 330
    b = [text(20, 28, "From documents to a batch", 15, weight=700, family=SERIF),
         text(20, 48, "experiments/build_v2.py runs once; train.py reads its files for every step", 12, "f-muted", MUTED)]
    b += _box(20, 78, 150, 62, "documents", ["4 sources", "in a fixed order"])
    b += _box(210, 70, 200, 78, "build_v2.py", ["encode, append EOS", "crc32(first 200 chars)", "lowest 1% → val"],
              accent=True)
    b += _box(452, 64, 140, 40, "train.u16", None)
    b += _box(452, 116, 140, 40, "val.u16", None)
    b += _box(612, 88, 128, 44, "meta.json", None, dashed=True)
    b += _arrow(170, 109, 208, 109)
    b += _arrow(410, 98, 450, 84)
    b += _arrow(410, 120, 450, 136)
    b.append(text(672, 150, "counts, vocab, bytes/token", 10.5, "f-muted", MUTED, "middle"))
    # train.py side
    y = 214
    b.append(line(20, 186, 740, 186, RULE, "f-rule", 1, ' stroke-dasharray="2 4"'))
    b.append(text(20, 204, "train.py, every step", 11, "f-muted", MUTED, weight=700))
    b += _box(20, y, 150, 62, "load_memmap", ["np.memmap, uint16", "no RAM up front"])
    b += _box(210, y, 160, 62, "WindowSampler", ["B start offsets", "no replacement"], accent=True)
    b += _box(410, y, 150, 62, "make_batch", ["S + 1 ids per row", "x = row[:-1]"])
    b += _box(600, y, 140, 62, "model(x, y)", ["loss"])
    b += _arrow(522, 156, 95, 212)
    b += _arrow(170, 245, 208, 245)
    b += _arrow(370, 245, 408, 245, "(B,)")
    b += _arrow(560, 245, 598, 245, "(B, S) ×2")
    b.append(text(20, 312, "y = row[1:] is x shifted by one: the target at each position is the next id.", 11.5,
                  "f-muted", MUTED))
    return svg("fig21-1-pipeline.svg", w, h, "Figure 21.1 · From documents to a batch",
               "build_v2.py encodes each document, appends EOS, and sends it to train.u16 or val.u16 by a hash of its "
               "first 200 characters. train.py memory-maps the files, draws B window starts without replacement, and "
               "make_batch cuts (B, S) inputs and targets shifted by one.", b)


def fig_21_2_splits():
    r = json.loads((MEAS / "part_v_a_corpus.json").read_text(encoding="utf-8"))
    s = r["sources"]
    close(r["tail_split"]["python"], 0.659, 5e-4, "tail split python share")
    close(r["tail_split"]["english"], 0.0, 1e-9, "tail split english share")
    close(s["english"]["train_share"], 0.5305, 5e-5, "english train share")
    close(s["english"]["val_share"], 0.5296, 5e-5, "english val share")
    groups = {"English": ["english"], "Hindi": ["hindi_web", "hindi_wiki"], "Python": ["python"]}
    colours = {"English": TEAL, "Hindi": CORAL, "Python": SLATE}
    cls = {"English": "f-teal", "Hindi": "f-coral", "Python": "f-muted-fill"}
    rows = [("training split (11.31B tokens)", {g: sum(s[k]["train_share"] for k in ks) for g, ks in groups.items()}),
            ("tail 10% of the same file", {g: sum(r["tail_split"][k] for k in ks) for g, ks in groups.items()}),
            ("hash split, val.u16 (113M)", {g: sum(s[k]["val_share"] for k in ks) for g, ks in groups.items()})]
    w, h = 760, 270
    x0, x1 = 250, 690
    b = [text(20, 28, "What each way of holding out base v2 would test on", 15, weight=700, family=SERIF),
         text(20, 48, "share of tokens by language · data/v2/meta.json · measured", 12, "f-muted", MUTED)]
    for i, (name, comp) in enumerate(rows):
        y = 78 + i * 52
        b.append(text(x0 - 12, y + 20, name, 12.5, anchor="end", weight=700 if i == 1 else 400))
        x = x0
        for g in groups:
            v = comp[g]
            ww = (x1 - x0) * v
            if ww <= 0:
                continue
            b.append(rect(x, y, ww - 1.5, 30, colours[g], cls[g], 3, tip=f"{name}: {g} {100 * v:.1f}%"))
            if ww > 46:
                b.append(rect(x + ww / 2 - 22, y + 7, 44, 16, PAPER, "f-paper", 3, ' fill-opacity="0.9"'))
                b.append(text(x + ww / 2, y + 19, f"{100 * v:.1f}%", 11, anchor="middle", weight=700, family=MONO))
            elif g == "Python":
                b.append(text(x1 + 6, y + 19, f"{100 * v:.1f}%", 11, weight=700, family=MONO))
            x += ww
    ly = 246
    for j, g in enumerate(groups):
        lx = x0 + j * 130
        b.append(rect(lx, ly - 11, 14, 14, colours[g], cls[g], 3))
        b.append(text(lx + 20, ly, g, 12))
    b.append(text(20, ly, "English 0.0% in the tail", 11.5, "f-muted", MUTED))
    return svg("fig21-2-splits.svg", w, h, "Figure 21.2 · Three ways to hold data out",
               "The base-v2 training file is 53.0% English, 40.4% Hindi and 6.6% Python. Its last 10% would be 65.9% "
               "Python, 34.1% Hindi and no English. The hash split the build used holds out 53.0% English, 40.9% Hindi "
               "and 6.1% Python.", b)


def fig_21_3_windows():
    import torch
    from anulm.training import train as T
    n, S, B = 640, 64, 3
    s = T.WindowSampler(n, S, B, seed=3)
    epochs = []
    for _ in range(2):
        order = s.order.tolist()
        epochs.append((int(min(order)), order))
        while s.epoch == len(epochs) - 1:
            s.next()
    assert epochs[0][0] != epochs[1][0], "the offsets should move between epochs"
    g = torch.Generator().manual_seed(3)
    rand = torch.randint(n - S - 1, (len(epochs[0][1]),), generator=g).tolist()
    w, h = 760, 300
    x0, x1 = 120, 730
    sx = lambda t: x0 + (x1 - x0) * t / n
    b = [text(20, 28, "Windows over 640 tokens, block size 64", 15, weight=700, family=SERIF),
         text(20, 48, "train.WindowSampler, seed 3 · the number in each window is its place in the visit order",
              12, "f-muted", MUTED)]
    for t in range(0, n + 1, 64):
        b.append(text(sx(t), 270, str(t), 10, "f-muted", MUTED, "middle", family=MONO))
    for e, (off, order) in enumerate(epochs):
        y = 74 + e * 62
        b.append(text(x0 - 12, y + 20, f"epoch {e}", 12, anchor="end", weight=700))
        b.append(text(x0 - 12, y + 35, f"offset {off}", 10.5, "f-muted", MUTED, "end", family=MONO))
        b.append(rect(sx(0), y, sx(n) - sx(0), 30, CARD, "f-card", 3))
        for rank, st in enumerate(order):
            b.append(rect(sx(st) + 1, y, sx(st + S) - sx(st) - 2, 30, TEAL, "f-teal", 3, ' fill-opacity="0.85"',
                          tip=f"epoch {e}: window at {st}, visited {rank + 1}th"))
            b.append(rect(sx(st) + (sx(st + S) - sx(st)) / 2 - 10, y + 7, 20, 16, PAPER, "f-paper", 3, ' fill-opacity="0.9"'))
            b.append(text(sx(st) + (sx(st + S) - sx(st)) / 2, y + 19, str(rank + 1), 11, anchor="middle", weight=700,
                          family=MONO))
    y = 74 + 2 * 62
    b.append(text(x0 - 12, y + 12, "with", 12, anchor="end", weight=700))
    b.append(text(x0 - 12, y + 27, "replacement", 12, anchor="end", weight=700))
    covered = [False] * n
    for k, st in enumerate(sorted(rand)):
        yy = y + (k % 3) * 11
        b.append(rect(sx(st), yy, sx(st + S) - sx(st), 9, CORAL, "f-coral", 2, ' fill-opacity="0.8"',
                      tip=f"a random window at {st}"))
        for t in range(st, st + S):
            covered[t] = True
    unseen = covered.count(False)
    b.append(text(x0, y + 50, f"{len(rand)} random starts: {unseen} of {n} tokens never drawn this 'epoch'", 11.5,
                  "f-muted", MUTED))
    return svg("fig21-3-windows.svg", w, h, "Figure 21.3 · Windows without and with replacement",
               "Two epochs of WindowSampler over 640 tokens: each lays a grid of 64-token windows from a new random "
               "offset and visits them in a new random order. Below, the same number of windows drawn with "
               "replacement overlap and leave gaps.", b)


# --------------------------------------------------------------------------- Ch 22
def fig_22_1_loop():
    w, h = 760, 512
    b = [text(20, 28, "train.py main(), from arguments to checkpoints", 15, weight=700, family=SERIF),
         text(20, 48, "line numbers at @773b6a8 · the teal box runs M times per step", 12, "f-muted", MUTED)]
    setup = [("parse arguments", "400–496"), ("seed, autocast", "498–503"), ("data, split, sampler", "509–550"),
             ("preset + --cfg → AnuLM", "553–587"), ("compile probe", "589–610"), ("optimizer", "612"),
             ("resume from .last", "614–675")]
    for i, (t, ln) in enumerate(setup):
        y = 70 + i * 58
        b.append(rect(20, y, 220, 40, CARD, "f-card", 8, f' stroke="{RULE}" stroke-width="1"'))
        b.append(text(34, y + 25, t, 12.5, weight=700))
        b.append(text(228, y + 25, ln, 10.5, "f-muted", MUTED, "end", family=MONO))
        if i:
            b += _arrow(130, y - 12, 130, y - 1)
    b += _arrow(240, 438, 290, 438)
    b.append(text(130, 490, "once per invocation", 11, "f-muted", MUTED, "middle"))
    # the loop
    b.append(rect(290, 64, 450, 432, CARD, "f-card", 12, f' fill-opacity="0.45" stroke="{RULE}"'))
    b.append(text(306, 86, "for step in range(start_step, end)   684", 12, weight=700, family=MONO))
    rows = [("lr_at → set_lr_scale", "686–687"), ("zero_grad(set_to_none=True)", "689")]
    y = 100
    for t, ln in rows:
        b.append(rect(310, y, 410, 32, PAPER, "f-paper", 6, f' stroke="{RULE}"'))
        b.append(text(324, y + 21, t, 12, family=MONO))
        b.append(text(710, y + 21, ln, 10.5, "f-muted", MUTED, "end", family=MONO))
        y += 40
    b.append(rect(310, y, 410, 112, PAPER, "f-paper", 8, f' stroke="{TEAL}" stroke-width="2.5"'))
    b.append(text(324, y + 20, "× M micro-batches   691–706", 12, weight=700, family=MONO))
    for k, t in enumerate(("sampler.next() → make_batch → (B, S) ×2", "forward under autocast → loss",
                           "(loss / M).backward()   grads add up", "loss_acc, aux_acc, tokens_seen")):
        b.append(text(338, y + 42 + 19 * k, t, 11.5, family=MONO))
    y += 120
    for t, ln in (("clip_grad_norm_(…, 1.0)", "708–709"), ("opt.step()", "710"),
                  ("update_expert_biases()  out of band", "712"), ("log every --log-every", "716–723"),
                  ("eval → best to --out, always .last", "725–742")):
        b.append(rect(310, y, 410, 32, PAPER, "f-paper", 6, f' stroke="{RULE}"'))
        b.append(text(324, y + 21, t, 12, family=MONO))
        b.append(text(710, y + 21, ln, 10.5, "f-muted", MUTED, "end", family=MONO))
        y += 36
    assert y <= 496, y
    return svg("fig22-1-loop.svg", w, h, "Figure 22.1 · The training loop",
               "Setup runs once: arguments, precision, data and sampler, model, compile probe, optimizer and resume. "
               "Each step sets the learning rate, zeroes the gradients, accumulates M micro-batches, clips, steps "
               "the optimizer, updates the expert biases, logs, and every --eval-every steps evaluates and saves.", b)


def fig_22_2_run():
    r = json.loads((MEAS / "part_v_a_run.json").read_text(encoding="utf-8"))
    ev = {int(k): v for k, v in r["eval"].items()}
    tr = {int(k): v for k, v in r["train"].items()}
    close(ev[1_899_999]["val"], 3.3523, 1e-9, "final val")
    close(ev[r["stable_end"]]["val"], 3.9779, 1e-9, "val at the end of the flat phase")
    close(r["decay_drop"], 0.6256, 5e-5, "decay drop")
    w, h = 760, 400
    x0, x1, y0, y1 = 70, 730, 70, 330
    lo, hi = 2.75, 5.0
    sx = lambda s: x0 + (x1 - x0) * s / 1_900_000
    sy = lambda v: y1 - (y1 - y0) * (min(max(v, lo), hi) - lo) / (hi - lo)
    b = [text(20, 28, "The released run: 1.9M steps of 4 × 1,024 tokens", 15, weight=700, family=SERIF),
         text(20, 48, "v2_train_run.log · measured · train loss is one step's batch; val is 400 fixed windows",
              12, "f-muted", MUTED)]
    b.append(rect(sx(1_590_000), y0, sx(1_900_000) - sx(1_590_000), y1 - y0, CARD, "f-card", 0))
    b.append(text(sx(1_590_000) + 6, y0 + 16, "linear decay", 11, "f-muted", MUTED, weight=700))
    b.append(text(sx(1_590_000) + 6, y0 + 30, "6e-4 → 6e-5", 10.5, "f-muted", MUTED, family=MONO))
    for v in (3.0, 3.5, 4.0, 4.5, 5.0):
        b.append(line(x0, sy(v), x1, sy(v)))
        b.append(text(x0 - 8, sy(v) + 4, f"{v:.1f}", 10.5, "f-muted", MUTED, "end", family=MONO))
    for s in range(0, 1_900_001, 250_000):
        b.append(text(sx(s), y1 + 18, f"{s / 1e6:.2f}M" if s else "0", 10.5, "f-muted", MUTED, "middle", family=MONO))
    b.append(text(x0 - 46, (y0 + y1) / 2, "loss (nats/token)", 11, "f-muted", MUTED, "middle",
                  extra=f' transform="rotate(-90 {x0 - 46} {(y0 + y1) / 2})"'))
    outside = [s for s, v in tr.items() if not lo <= v["loss"] <= hi]
    assert len(outside) == 9, len(outside)
    for s, v in sorted(tr.items()):
        if s in outside:
            continue
        b.append(f'<circle cx="{sx(s):.1f}" cy="{sy(v["loss"]):.1f}" r="1.6" fill="{CORAL}" fill-opacity="0.45" '
                 f'class="f-coral"><title>step {s:,}: train loss {v["loss"]:.4f}</title></circle>')
    pts = " ".join(f"{sx(s):.1f},{sy(v['val']):.1f}" for s, v in sorted(ev.items()))
    b.append(f'<polyline points="{pts}" fill="none" stroke="{TEAL}" stroke-width="2.5" class="f-teal-stroke"/>')
    for s, v in sorted(ev.items()):
        b.append(f'<circle cx="{sx(s):.1f}" cy="{sy(v["val"]):.1f}" r="2.2" fill="{TEAL}" class="f-teal">'
                 f'<title>step {s:,}: val {v["val"]:.4f} ± {v["sem"]:.4f}</title></circle>')
    for s, dy in ((9_999, -10), (r["stable_end"], -12), (1_899_999, 18)):
        v = ev[s]["val"]
        b.append(text(sx(s) + (6 if s < 1_800_000 else -6), sy(v) + dy, f"{v:.4f}", 11.5,
                      anchor="start" if s < 1_800_000 else "end", weight=700, family=MONO))
    ly = 372
    b.append(f'<circle cx="80" cy="{ly - 4}" r="4" fill="{CORAL}" class="f-coral"/>')
    b.append(text(90, ly, "train loss, every 1,000 steps (9 of 1,901 off the chart)", 11.5))
    b.append(line(470, ly - 4, 494, ly - 4, TEAL, "f-teal-stroke", 2.5))
    b.append(text(502, ly, "val loss, every 10,000 steps", 11.5))
    return svg("fig22-2-run.svg", w, h, "Figure 22.2 · The released run's loss",
               "Validation loss falls from 4.6271 at step 9,999 to 3.9779 by the end of the flat learning rate at step "
               "1,589,999, then to 3.3523 during the final linear decay. Train-loss points scatter about a nat around it.",
               b)


# --------------------------------------------------------------------------- Ch 23
def fig_23_1_formats():
    import torch
    rows = [("float32", torch.float32, 8, 23), ("bfloat16", torch.bfloat16, 8, 7), ("float16", torch.float16, 5, 10)]
    for _, dt, e, m in rows:
        assert torch.finfo(dt).bits == 1 + e + m
    close(torch.finfo(torch.bfloat16).eps, 0.0078125, 1e-12, "bf16 eps")
    close(torch.finfo(torch.float16).max, 65504.0, 1e-9, "fp16 max")
    w, h = 760, 290
    x0, unit = 120, 17
    b = [text(20, 28, "Three ways to store a number", 15, weight=700, family=SERIF),
         text(20, 48, "sign · exponent (range) · mantissa (precision) · values from torch.finfo · verified", 12,
              "f-muted", MUTED)]
    for i, (name, dt, e, m) in enumerate(rows):
        y = 74 + i * 64
        f = torch.finfo(dt)
        b.append(text(x0 - 12, y + 19, name, 13, anchor="end", weight=700, family=MONO))
        x = x0
        for k, (n, fill, cls, label) in enumerate(((1, SLATE, "f-muted-fill", "s"), (e, TEAL, "f-teal", "exponent"),
                                                   (m, CORAL, "f-coral", "mantissa"))):
            b.append(rect(x, y, n * unit - 2, 28, fill, cls, 3, tip=f"{name}: {n} {label} bit{'s' if n > 1 else ''}"))
            if n * unit > 70 or k == 0:
                b.append(rect(x + (n * unit - 2) / 2 - (24 if k else 7), y + 6, 48 if k else 14, 16, PAPER, "f-paper", 3,
                              ' fill-opacity="0.9"'))
                b.append(text(x + (n * unit - 2) / 2, y + 18, str(n) if k else "1", 11, anchor="middle", weight=700,
                              family=MONO))
            x += n * unit
        b.append(text(x0, y + 46, f"eps {f.eps:.3g}   max {f.max:.3g}   smallest normal {f.tiny:.3g}", 11.5,
                      "f-muted", MUTED, family=MONO))
    ly = 272
    for j, (fill, cls, label) in enumerate(((TEAL, "f-teal", "exponent bits: range"),
                                            (CORAL, "f-coral", "mantissa bits: precision"))):
        b.append(rect(x0 + j * 230, ly - 11, 14, 14, fill, cls, 3))
        b.append(text(x0 + j * 230 + 20, ly, label, 12))
    return svg("fig23-1-formats.svg", w, h, "Figure 23.1 · fp32, bf16 and fp16",
               "float32 has 1 sign, 8 exponent and 23 mantissa bits. bfloat16 keeps the 8 exponent bits and 7 mantissa "
               "bits, so the same range with eps 0.0078. float16 has 5 exponent and 10 mantissa bits: finer steps but a "
               "maximum of 65,504 and a smallest normal number of 6.1e-05.", b)


def fig_23_2_memory():
    r = json.loads((MEAS / "part_v_a_memory.json").read_text(encoding="utf-8"))
    st = r["steps"]
    close(st["plain"]["peak_gb"], 12.26, 5e-3, "plain peak")
    close(st["grad_ckpt"]["peak_gb"], 6.64, 5e-3, "checkpointing peak")
    close(st["ce_chunk 2048"]["peak_gb"], 11.34, 5e-3, "chunked peak")
    close(st["both"]["peak_gb"], 6.18, 5e-3, "both peak")
    n = 397_659_648
    fixed = 12 * n / 2**30                       # resident between steps: zero_grad(set_to_none=True) frees gradients
    close(fixed, 4.44, 5e-3, "weights + moments")
    close(st["plain"]["peak_gb"] - st["plain"]["first_step_peak_gb"], 2.99, 5e-3, "moments appear after step 0")
    card = st["plain"]["card_gb"]
    w, h = 760, 330
    x0, x1 = 200, 720
    sx = lambda v: x0 + (x1 - x0) * v / 16.5
    b = [text(20, 28, "One training step of the released model, 4 × 1,024 tokens", 15, weight=700, family=SERIF),
         text(20, 48, "peak memory allocated, GiB · RTX 5070 Ti, bf16 autocast, fused AdamW · measured", 12,
              "f-muted", MUTED)]
    for v in range(0, 17, 2):
        b.append(line(sx(v), 66, sx(v), 252))
        b.append(text(sx(v), 268, str(v), 10.5, "f-muted", MUTED, "middle", family=MONO))
    labels = {"plain": "plain (the released run)", "grad_ckpt": "--grad-ckpt", "ce_chunk 2048": "--ce-chunk 2048",
              "both": "both"}
    for i, (k, lab) in enumerate(labels.items()):
        y = 78 + i * 44
        v = st[k]
        b.append(text(x0 - 12, y + 19, lab, 12.5, anchor="end", weight=700 if k == "plain" else 400))
        b.append(rect(sx(0), y, sx(fixed) - sx(0) - 1, 28, SLATE, "f-muted-fill", 3,
                      tip=f"weights and AdamW moments, resident between steps: {fixed:.2f} GiB"))
        b.append(rect(sx(fixed), y, sx(v["peak_gb"]) - sx(fixed), 28, TEAL, "f-teal", 3,
                      tip=f"{lab}: gradients, activations, logits and workspace {v['peak_gb'] - fixed:.2f} GiB"))
        b.append(text(sx(v["peak_gb"]) + 8, y + 19, f"{v['peak_gb']:.2f}", 12, weight=700, family=MONO))
        b.append(f'<circle cx="{sx(v["reserved_gb"]):.1f}" cy="{y + 14}" r="4" fill="{PAPER}" stroke="{INK}" '
                 f'stroke-width="1.5" class="f-paper f-ink-stroke"><title>reserved by the allocator: '
                 f'{v["reserved_gb"]:.2f} GiB</title></circle>')
    for v, lab in ((card, f"card {card:.2f}"), (0.92 * card, f"warning at {0.92 * card:.2f}")):
        b.append(line(sx(v), 62, sx(v), 252, CORAL, "", 1.5, ' stroke-dasharray="4 3"'))
    b.append(text(sx(card) + 4, 300, f"card {card:.2f}", 11, "f-muted", MUTED, "end", weight=700))
    b.append(text(sx(0.92 * card) - 4, 286, f"warn_if_spilling at {0.92 * card:.2f}", 11, "f-muted", MUTED, "end",
                  weight=700))
    ly = 318
    b.append(rect(20, ly - 11, 14, 14, SLATE, "f-muted-fill", 3))
    b.append(text(40, ly, f"weights + AdamW moments, 12 B × 397.7M = {fixed:.2f}", 11.5))
    b.append(rect(330, ly - 11, 14, 14, TEAL, "f-teal", 3))
    b.append(text(350, ly, "gradients, activations, logits, workspace", 11.5))
    b.append(f'<circle cx="640" cy="{ly - 4}" r="4" fill="{PAPER}" stroke="{INK}" stroke-width="1.5" '
             f'class="f-paper f-ink-stroke"/>')
    b.append(text(650, ly, "reserved", 11.5))
    return svg("fig23-2-memory.svg", w, h, "Figure 23.2 · Memory per training step",
               "Peak allocated memory: plain 12.26 GiB, gradient checkpointing 6.64, chunked loss 11.34, both 6.18. Of each, "
               "4.44 GiB is the weights and AdamW moments, resident between steps; the rest is gradients, activations, "
               "logits and workspace. The plain run's allocator reserves 15.12 GiB, past the "
               "14.65 GiB warning line and below the 15.92 GiB card.", b)
