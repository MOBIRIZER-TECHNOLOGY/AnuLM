"""
Figures for Part VII, After pre-training (Chapters 30-34).

    python book/tools/make_figures.py part_vii

Every number drawn here is recomputed from the book example that prints it, or
copied from the docs/RESULTS.md table the chapter quotes and asserted equal to
the value in the chapter's text.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_figures import (BOOK, CARD, CORAL, GOLD, INK, MONO, MUTED, NIGHT, PAPER, RULE, SERIF,  # noqa: E402
                          SLATE, TEAL, close, esc, line, rect, svg, text)

sys.path.insert(0, str(BOOK / "code-examples"))
PLUM = "#7B5E8E"


def arrow(x1, y1, x2, y2):
    return line(x1, y1, x2, y2, MUTED, "f-muted-stroke", 1.5, ' marker-end="url(#arrow)"')


def legend_swatch(x, y, fill, label, cls=""):
    return [rect(x, y - 10, 12, 12, fill, cls, 2), text(x + 18, y, label, 11.5, "f-muted", MUTED)]


# --------------------------------------------------------------------------- Ch 30
def fig_30_1_packing():
    import ch30_sft_packing as E
    from anulm.bpe import BPE
    from anulm.training.finetune import _pair_ids, pack
    text_ = ("Question: What is the capital of India?\nAnswer: New Delhi is the capital of India.\n"
             "Question: What is two plus two?\nAnswer: Two plus two is four.\n") * 30
    tok = BPE.train(text_, 300, verbose=False)
    items = [{"question": "What is the capital of India?", "answer": "New Delhi.", "lang": "en"},
             {"question": "What is two plus two?", "answer": "Four.", "lang": "en"}]
    pairs = [_pair_ids(tok, it) for it in items]
    lens = [(len(p), len(a)) for p, a in pairs]
    assert lens == [(10, 12), (9, 7)], lens
    n = sum(a + b for a, b in lens)
    x, y = pack(pairs, n, tok.eos_id)
    xb, yb = E.pack_by_hand(pairs, n, tok.eos_id)
    assert torch.equal(x.long(), xb) and int((y != -1).sum()) == 19
    seg = []                                       # segment of each id in the row of n + 1
    for k, (p, a) in enumerate(lens):
        seg += [f"prompt {k + 1}"] * p + [f"answer {k + 1}"] * a
    seg += ["pad"]
    w, h = 780, 300
    c, x0 = 18.5, 30
    b = [text(20, 28, "Two pairs packed into one row, prompts masked", 15, weight=700, family=SERIF),
         text(20, 48, "the Chapter 30 example · block of 38 · each square is one position", 12, "f-muted", MUTED)]
    fills = {"prompt": RULE, "answer": SLATE, "pad": PAPER}

    def row(y0, label, kinds, tips):
        out = [text(x0, y0 - 8, label, 12, weight=700)]
        for i, k in enumerate(kinds):
            fill = PLUM if k == "graded" else fills.get(k.split()[0], CARD)
            cls = "" if k == "graded" else ("f-rule-fill" if k.startswith("prompt") or k == "masked" else "")
            if k == "masked":
                out.append(rect(x0 + i * c + 1, y0, c - 2, 26, "url(#hatch)", "", 3, tip=tips[i]))
            else:
                out.append(rect(x0 + i * c + 1, y0, c - 2, 26, fill, cls, 3,
                                ' stroke="#DCE1DF" stroke-width="1"' if k == "pad" else "", tips[i]))
        return out

    ids_row = seg[:n + 1]
    b += row(84, "the row: n + 1 = 39 ids (prompt, answer + EOS, prompt, answer + EOS)", ids_row,
             [f"id {i}: {s}" for i, s in enumerate(ids_row)])
    b.append(text(x0, 132, "x = row[:-1]: 38 inputs", 11.5, "f-muted", MUTED))
    b.append(text(x0 + 200, 132, "y = targets[1:]: 38 targets, position t predicts id t + 1", 11.5, "f-muted", MUTED))
    tk = ["graded" if v != -1 else "masked" for v in y[0].tolist()]
    b += row(160, "y: the target at each input position (−1 is never graded)", tk,
             [f"position {i}: {'graded' if t == 'graded' else 'target −1'}" for i, t in enumerate(tk)])
    # brackets for the two answers
    pos = 0
    for k, (p, a) in enumerate(lens):
        s, e = pos + p - 1, pos + p + a - 1          # graded positions of answer k in y
        b.append(line(x0 + s * c + 2, 198, x0 + e * c - 2, 198, INK, "f-ink-stroke", 1.2))
        b.append(text(x0 + (s + e) * c / 2, 214, f"answer {k + 1}: {a} targets", 11.5, anchor="middle"))
        pos += p + a
    yy = 252
    b += legend_swatch(30, yy, RULE, "prompt id", "f-rule-fill")
    b += legend_swatch(130, yy, SLATE, "answer id (EOS last)")
    b += legend_swatch(290, yy, PLUM, "graded target")
    b.append(rect(410, yy - 10, 12, 12, "url(#hatch)", "", 2))
    b.append(text(428, yy, "target −1", 11.5, "f-muted", MUTED))
    b.append(text(30, yy + 24, "19 of 38 positions graded: exactly the 12 + 7 answer ids. The last position's "
                  "target is padding.", 11.5, "f-muted", MUTED))
    return svg("fig30-1-packing.svg", w, h, "Figure 30.1 · Packing with masked prompts",
               "Two pairs, 10 + 12 and 9 + 7 ids, packed into one row of 39 ids. Inputs are the first 38; targets "
               "are shifted by one. Prompt targets are -1; the 19 answer targets are graded.", b)


RESULTS_20 = {"MC (chance 25%)": (43.0, 50.2), "token F1": (6.3, 21.8), "names the subject (%)": (12.0, 81.0)}


def fig_30_2_results():
    close(RESULTS_20["names the subject (%)"][1], 81.0, 0, "RESULTS 20 subject")
    w, h = 760, 270
    x0, x1 = 230, 700
    sx = lambda v: (x1 - x0) * v / 100
    b = [text(20, 28, "What instruction tuning bought, RESULTS §20", 15, weight=700, family=SERIF),
         text(20, 48, "Hindi Wikipedia questions held out by article · base vs fine-tuned · measured", 12,
              "f-muted", MUTED)]
    for t in range(0, 101, 20):
        b.append(line(x0 + sx(t), 66, x0 + sx(t), 216))
        b.append(text(x0 + sx(t), 232, str(t), 10, "f-muted", MUTED, "middle", family=MONO))
    for i, (name, (a, c_)) in enumerate(RESULTS_20.items()):
        y = 74 + i * 48
        b.append(text(x0 - 12, y + 18, name, 12.5, anchor="end"))
        b.append(rect(x0, y, sx(a), 16, CORAL, "f-coral", 3, tip=f"{name}, base: {a}"))
        b.append(text(x0 + sx(a) + 6, y + 12, f"{a:.1f}", 11.5, weight=700, family=MONO))
        b.append(rect(x0, y + 19, sx(c_), 16, TEAL, "f-teal", 3, tip=f"{name}, fine-tuned: {c_}"))
        b.append(text(x0 + sx(c_) + 6, y + 31, f"{c_:.1f}", 11.5, weight=700, family=MONO))
    b += legend_swatch(230, 256, CORAL, "base ckpt_hindi_mixed18k", "f-coral")
    b += legend_swatch(440, 256, TEAL, "fine-tuned ckpt_hindi_qa", "f-teal")
    return svg("fig30-2-results.svg", w, h, "Figure 30.2 · What instruction tuning bought",
               "Multiple choice 43.0 to 50.2 percent, token F1 6.3 to 21.8, names the subject 12 to 81 percent.", b)


# --------------------------------------------------------------------------- Ch 31
def fig_31_1_lora():
    from trace_forward import released
    cfg = released()
    D, out = cfg.hidden_size, (cfg.n_head + 2 * cfg.n_kv_head) * cfg.head_dim
    assert (D, out) == (1024, 1536)
    r = 16
    w, h = 760, 330
    b = [text(20, 28, "LoRA beside a frozen weight, then merged", 15, weight=700, family=SERIF),
         text(20, 48, "query_key_value of the released model, r = 16, α = 32 · shapes per token", 12, "f-muted", MUTED)]

    def box(x, y, ww, hh, title, sub, accent=False, frozen=False):
        st = f' stroke="{PLUM}" stroke-width="2.5"' if accent else f' stroke="{RULE}" stroke-width="1"'
        o = [rect(x, y, ww, hh, CARD, "f-card", 8, st)]
        o.append(text(x + ww / 2, y + hh / 2 - 3, title, 13, anchor="middle", weight=700))
        o.append(text(x + ww / 2, y + hh / 2 + 14, sub, 11, "f-muted", MUTED, "middle", family=MONO))
        return o

    b += box(20, 120, 90, 56, "x", f"(B, S, {D})")
    b += box(160, 74, 230, 56, "W  frozen", f"{out} × {D} = {out * D:,}")
    b += box(160, 166, 120, 56, "A  trains", f"{r} × {D}", accent=True)
    b += box(318, 166, 160, 56, "B  trains, 0 at start", f"{out} × {r}", accent=True)
    b += box(500, 166, 64, 56, "× α/r", "= 2", accent=True)
    b.append(f'<circle cx="612" cy="148" r="16" fill="{CARD}" class="f-card" stroke="{MUTED}" stroke-width="1.5"/>')
    b.append(text(612, 154, "+", 18, anchor="middle", weight=700))
    b += box(664, 120, 86, 56, "y", f"(B, S, {out})")
    b.append(arrow(110, 140, 158, 104))
    b.append(arrow(110, 156, 158, 192))
    b.append(arrow(280, 194, 316, 194))
    b.append(text(299, 158, f"(B, S, {r})", 9.5, "f-muted", MUTED, "middle", family=MONO))
    b.append(arrow(478, 194, 498, 194))
    b.append(arrow(564, 186, 600, 160))
    b.append(arrow(390, 102, 596, 142))
    b.append(arrow(628, 148, 662, 148))
    n_ad = r * (D + out)
    b.append(text(160, 250, f"trainable: r·(in + out) = {r}·({D} + {out}) = {n_ad:,}, "
                  f"{100 * n_ad / (out * D):.1f}% of W", 12, weight=700))
    b.append(line(20, 270, 740, 270))
    b.append(text(20, 296, "merge_lora:  W ← W + (α/r)·B A", 13, weight=700, family=MONO))
    b.append(text(20, 316, "one matrix again, the same shape as W: no extra work per token after merging", 12,
                  "f-muted", MUTED))
    return svg("fig31-1-lora.svg", w, h, "Figure 31.1 · LoRA beside a frozen weight",
               "x goes through the frozen W and, in parallel, through A (16 by 1024) and B (1536 by 16) scaled by "
               "alpha over r; the two are added. Merging replaces W with W plus (alpha/r) B A.", b)


RESULTS_29 = [("LoRA r=16, attention, 1 epoch", 0.260, 67, "5.7 MB"),
              ("LoRA r=16, + all experts, 1 epoch", 0.309, 349, "108.7 MB"),
              ("LoRA r=64, attention, 3 epochs", 0.334, 189, "22.5 MB"),
              ("full fine-tune, 1 epoch", 0.512, 131, "1,517.6 MB")]


def fig_31_2_tradeoff():
    close(RESULTS_29[-1][1], 0.512, 0, "RESULTS 29 full")
    w, h = 760, 290
    x0, x1 = 270, 560
    sx = lambda v: (x1 - x0) * v / 0.6
    b = [text(20, 28, "LoRA against a full fine-tune at 398M, RESULTS §29", 15, weight=700, family=SERIF),
         text(20, 48, "fall in held-out answer loss from 3.9028 · same data, same base · measured", 12,
              "f-muted", MUTED)]
    b.append(text(x1 + 30, 70, "wall clock", 11.5, "f-muted", MUTED, weight=700))
    b.append(text(x1 + 120, 70, "kept on disk", 11.5, "f-muted", MUTED, weight=700))
    for t in (0, 0.2, 0.4, 0.6):
        b.append(line(x0 + sx(t), 78, x0 + sx(t), 250))
        b.append(text(x0 + sx(t), 266, f"{t:.1f}", 10, "f-muted", MUTED, "middle", family=MONO))
    for i, (name, d, sec, size) in enumerate(RESULTS_29):
        y = 88 + i * 40
        full = i == 3
        b.append(text(x0 - 12, y + 15, name, 12, anchor="end", weight=700 if full else 400))
        b.append(rect(x0, y, sx(d), 20, PLUM if full else SLATE, "", 3, tip=f"{name}: −{d:.3f}"))
        b.append(text(x0 + sx(d) + 6, y + 15, f"−{d:.3f}", 11.5, weight=700, family=MONO))
        b.append(text(x1 + 30, y + 15, f"{sec} s", 11.5, family=MONO))
        b.append(text(x1 + 120, y + 15, size, 11.5, family=MONO))
    return svg("fig31-2-tradeoff.svg", w, h, "Figure 31.2 · LoRA against a full fine-tune",
               "Held-out loss falls by 0.260, 0.309 and 0.334 for the three LoRA runs and by 0.512 for the full "
               "fine-tune, which also took less time than two of them; LoRA's files are far smaller.", b)


# --------------------------------------------------------------------------- Ch 32
def fig_32_1_options():
    from anulm.tasks import decide as D
    from anulm.bpe import BPE
    tok = BPE.train("Message: which of these does it ask for options card arrival "
                    "lost card refund exchange rate\n- " * 30, 300, verbose=False)
    opts = ["card_arrival", "lost_card", "refund"]
    ids, pos = D.encode(tok, "where is my card", opts)
    assert pos == [42, 47, 51] and len(ids) == 52, (pos, len(ids))
    w, h = 760, 300
    b = [text(20, 28, "Scoring K options in one forward pass", 15, weight=700, family=SERIF),
         text(20, 48, "decide.encode on the Chapter 32 example · 52 ids · causal backbone", 12, "f-muted", MUTED)]
    starts = [38] + [p + 1 for p in pos[:-1]]
    segs = [("EOS", 0, 0, 44), ("Message: where is my card … Options:", 1, 37, 250)]
    segs += [(f"- {D.readable(o)}", s_, e_, 130) for o, s_, e_ in zip(opts, starts, pos)]
    x = 20
    for k, (name, s_, e_, ww) in enumerate(segs):
        b.append(rect(x, 80, ww - 4, 34, CARD, "f-card", 4, ' stroke="#DCE1DF" stroke-width="1"',
                      f"{name}: ids {s_}–{e_}"))
        b.append(text(x + (ww - 4) / 2 - (10 if k >= 2 else 0), 101, name, 11.5, anchor="middle"))
        b.append(text(x + 2, 132, f"{s_}" if s_ == e_ else f"{s_}–{e_}", 10, "f-muted", MUTED, family=MONO))
        if k >= 2:
            b.append(rect(x + ww - 26, 80, 20, 34, PLUM, "", 3, tip=f"position {e_}: last token of option {k - 2}"))
            b.append(text(x + ww - 16, 132, str(e_), 10.5, anchor="middle", family=MONO, weight=700))
        x += ww
    b.append(text(20, 172, "hidden states (1, 52, D)  →  gather at [42, 47, 51]  →  (1, 3, D)", 12, family=MONO))
    b.append(text(20, 192, "→  Linear(D, 1)  →  (1, 3) logits  →  softmax(logits / τ)", 12, family=MONO))
    b.append(text(20, 222, "Each marked position has read the message and its own option's name; under causal "
                  "attention it has", 12, "f-muted", MUTED))
    b.append(text(20, 240, "not read the options after it. One pass serves all K options.", 12, "f-muted", MUTED))
    b += legend_swatch(20, 278, PLUM, "an option's last token: where its score is read")
    b += legend_swatch(360, 278, CARD, "other ids", "f-card")
    return svg("fig32-1-options.svg", w, h, "Figure 32.1 · Scoring K options in one pass",
               "A start token, the message, then three option names. The hidden states at each option's last token, "
               "positions 42, 47 and 51, are gathered and scored by one linear head; a softmax over the three "
               "gives the probabilities.", b)


def reliability_data():
    from anulm.tasks import decide as D
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(0)
    true_logits = torch.randn(2000, 10, generator=g) * 1.5
    y = torch.multinomial(F.softmax(true_logits, 1), 1, generator=g).squeeze(1)
    logits = 3.0 * true_logits
    tau = D.fit_temperature(logits, y)
    out = {}
    for name, t in (("raw", 1.0), ("calibrated", tau)):
        p = F.softmax(logits / t, 1)
        conf, pred = p.max(1)
        right = (pred == y).float()
        rows = []
        for k in range(10):
            m = (conf > k / 10) & (conf <= (k + 1) / 10)
            if int(m.sum()) >= 20:
                rows.append((float(conf[m].mean()), float(right[m].mean()), int(m.sum())))
        out[name] = (rows, D.metrics(p, y)["ece"])
    return tau, out


def fig_32_2_reliability():
    tau, d = reliability_data()
    close(tau, 3.0, 1e-9, "fitted temperature")
    close(d["raw"][1], 0.3350, 5e-5, "raw ECE")
    close(d["calibrated"][1], 0.0306, 5e-5, "calibrated ECE")
    w, h = 760, 400
    x0, y0, sz = 90, 70, 280
    sx = lambda v: x0 + sz * v
    sy = lambda v: y0 + sz * (1 - v)
    b = [text(20, 28, "Reliability: confidence against accuracy", 15, weight=700, family=SERIF),
         text(20, 48, "made-up, overconfident logits · 2,000 examples, 10 classes, seed 0 · illustrative", 12,
              "f-muted", MUTED)]
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        b.append(line(sx(t), y0, sx(t), y0 + sz))
        b.append(line(x0, sy(t), x0 + sz, sy(t)))
        b.append(text(sx(t), y0 + sz + 16, f"{t:.2f}", 10, "f-muted", MUTED, "middle", family=MONO))
        b.append(text(x0 - 8, sy(t) + 4, f"{t:.2f}", 10, "f-muted", MUTED, "end", family=MONO))
    b.append(line(sx(0), sy(0), sx(1), sy(1), INK, "f-ink-stroke", 1.2, ' stroke-dasharray="4 3"'))
    b.append(text(x0 + sz / 2, y0 + sz + 36, "mean confidence in the bin", 11.5, "f-muted", MUTED, "middle"))
    b.append(f'<text x="30" y="{y0 + sz / 2:.1f}" class="f-muted" fill="{MUTED}" font-size="11.5" '
             f'font-family="Lato, sans-serif" text-anchor="middle" transform="rotate(-90 30 {y0 + sz / 2:.1f})">'
             f'accuracy in the bin</text>')
    for name, colour, cls in (("raw", CORAL, "f-coral"), ("calibrated", TEAL, "f-teal")):
        rows, _ = d[name]
        pts = " ".join(f"{sx(c_):.1f},{sy(a):.1f}" for c_, a, _ in rows)
        b.append(f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for c_, a, n in rows:
            b.append(f'<circle cx="{sx(c_):.1f}" cy="{sy(a):.1f}" r="5" fill="{colour}" class="{cls} f-ring" '
                     f'stroke="{PAPER}" stroke-width="1.5"><title>{name}: confidence {c_:.2f}, accuracy {a:.2f}, '
                     f'{n} examples</title></circle>')
    tx = 430
    b += legend_swatch(tx, 110, CORAL, "raw, τ = 1", "f-coral")
    b.append(text(tx, 132, f"ECE {100 * d['raw'][1]:.2f}%: confident far beyond its accuracy", 12))
    b += legend_swatch(tx, 172, TEAL, f"divided by the fitted τ = {tau:.2f}", "f-teal")
    b.append(text(tx, 194, f"ECE {100 * d['calibrated'][1]:.2f}%: close to the diagonal", 12))
    b.append(text(tx, 234, "The dashed diagonal is perfect calibration.", 12, "f-muted", MUTED))
    b.append(text(tx, 252, "Bins with fewer than 20 examples are not drawn.", 12, "f-muted", MUTED))
    b.append(text(tx, 270, "Accuracy is 43.8% in both: τ never changes", 12, "f-muted", MUTED))
    b.append(text(tx, 288, "which option is largest.", 12, "f-muted", MUTED))
    return svg("fig32-2-reliability.svg", w, h, "Figure 32.2 · Reliability before and after temperature scaling",
               "Raw confidences lie far below the diagonal, ECE 33.50 percent; after dividing the logits by the "
               "fitted temperature 3.00 the points lie near it, ECE 3.06 percent.", b)



# --------------------------------------------------------------------------- Ch 33
def fig_33_1_pipeline():
    w, h = 780, 330
    b = [text(20, 28, "Looking facts up: retrieve, then read", 15, weight=700, family=SERIF),
         text(20, 48, "rag.py and rc_pointer.py · the reader that learned to read is highlighted", 12, "f-muted", MUTED)]

    def box(x, y, ww, hh, title, sub, accent=False, dashed=False):
        st = (f' stroke="{PLUM}" stroke-width="2.5"' if accent else
              f' stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5 4"' if dashed else
              f' stroke="{RULE}" stroke-width="1"')
        o = [rect(x, y, ww, hh, CARD, "f-card", 8, st)]
        o.append(text(x + ww / 2, y + 22, title, 12.5, anchor="middle", weight=700))
        for k, line_ in enumerate(sub):
            o.append(text(x + ww / 2, y + 40 + 15 * k, line_, 10.5, "f-muted", MUTED, "middle", family=MONO))
        return o

    b += box(20, 130, 100, 60, "question", ["text"])
    b += box(160, 70, 170, 74, "BM25", ["Retriever.bm25", "932,648 chunks, numpy"])
    b += box(160, 176, 170, 74, "dense, borrowed", ["Dense.search", "multilingual-e5-small"], dashed=True)
    b += box(370, 123, 120, 74, "fusion", ["search(mode=", "'hybrid'): RRF"])
    b += box(530, 70, 230, 80, "generative reader", ["RagQA: passages, then", "question; generate"])
    b += box(530, 176, 230, 80, "pointer reader", ["rc_pointer: question first;", "start/end score per token"],
             accent=True)
    b.append(arrow(120, 152, 158, 112))
    b.append(arrow(120, 168, 158, 210))
    b.append(arrow(330, 107, 368, 148))
    b.append(arrow(330, 213, 368, 172))
    b.append(arrow(490, 150, 528, 112))
    b.append(arrow(490, 170, 528, 212))
    b.append(text(645, 167, "either reads the top 3 passages", 10.5, "f-muted", MUTED, "middle"))
    b.append(text(20, 290, "Dashed: a borrowed model, not this project's. Fusion is optional: rag.py defaults to dense when "
                  "its index exists,", 11.5, "f-muted", MUTED))
    b.append(text(20, 307, "else BM25. Each reader is a fine-tuned AnuLM; the pointer reads only text that is in the "
                  "passages.", 11.5, "f-muted", MUTED))
    return svg("fig33-1-pipeline.svg", w, h, "Figure 33.1 · Retrieve, then read",
               "A question goes to BM25, to a borrowed dense retriever, or to both fused by reciprocal rank; the top "
               "three passages go to a generative reader or to a pointer reader that scores answer spans.", b)


RESULTS_33_DEMO = {"BM25": (65, 0), "dense (e5)": (90, 5), "hybrid (RRF)": (75, 2)}


def fig_33_2_retrievers():
    assert RESULTS_33_DEMO["dense (e5)"] == (90, 5)
    w, h = 760, 250
    b = [text(20, 28, "Retrievers on the 20 demo questions, RESULTS §33", 15, weight=700, family=SERIF),
         text(20, 48, "left: answer in the top passage · right: RESULTS' AnuLM 400M reader, right end to end · "
              "measured", 12, "f-muted", MUTED)]
    panels = [("answer in the top 1 passage (%)", 0, 100, lambda v: v[0], "{:d}%"),
              ("answered correctly, of 20", 380, 20, lambda v: v[1], "{:d}/20")]
    for title, px, top, get, fmt in panels:
        b.append(text(px + 20, 82, title, 12, weight=700))
        x0, x1 = px + 130, px + 340
        for i, (name, v) in enumerate(RESULTS_33_DEMO.items()):
            y = 100 + i * 40
            val = get(v)
            ww = (x1 - x0) * val / top
            hi = name.startswith("dense")
            b.append(text(x0 - 10, y + 15, name, 12, anchor="end", weight=700 if hi else 400))
            b.append(rect(x0, y, max(ww, 2), 20, PLUM if hi else SLATE, "", 3, tip=f"{name}: {fmt.format(val)}"))
            b.append(text(x0 + max(ww, 2) + 6, y + 15, fmt.format(val), 11.5, weight=700, family=MONO))
    b.append(text(20, 236, "Hybrid recalls as well as dense at 10 passages (MLQA: better) yet answers worse: BM25's "
                  "near-misses confuse the reader.", 11.5, "f-muted", MUTED))
    return svg("fig33-2-retrievers.svg", w, h, "Figure 33.2 · Retrievers on the demo questions",
               "Answer in the top passage: BM25 65%, dense 90%, hybrid 75%. Answered correctly end to end by AnuLM's "
               "reader: 0, 5 and 2 of 20.", b)


# --------------------------------------------------------------------------- Ch 34
def fig_34_1_dpo_loss():
    f = lambda m: math.log1p(math.exp(-m))                 # -log sigmoid(m)
    close(f(0.0), 0.6931, 5e-5, "ln 2")
    close(f(0.2), 0.5981, 5e-5, "worked margin")
    w, h = 760, 330
    x0, x1, y0, y1 = 80, 700, 70, 270
    lo, hi, top = -3.0, 5.0, 3.2
    sx = lambda m: x0 + (x1 - x0) * (m - lo) / (hi - lo)
    sy = lambda v: y1 - (y1 - y0) * v / top
    b = [text(20, 28, "The DPO loss against the margin", 15, weight=700, family=SERIF),
         text(20, 48, "loss = −log σ(m), m = β·[(log π_θ − log π_ref)(chosen) − (same)(rejected)] · computed",
              12, "f-muted", MUTED)]
    for t in range(-3, 6):
        b.append(line(sx(t), y0, sx(t), y1))
        b.append(text(sx(t), y1 + 16, str(t), 10, "f-muted", MUTED, "middle", family=MONO))
    for v in (0, 1, 2, 3):
        b.append(line(x0, sy(v), x1, sy(v)))
        b.append(text(x0 - 8, sy(v) + 4, str(v), 10, "f-muted", MUTED, "end", family=MONO))
    pts = " ".join(f"{sx(lo + k * 0.05):.1f},{sy(f(lo + k * 0.05)):.1f}" for k in range(int((hi - lo) / 0.05) + 1))
    b.append(f'<polyline points="{pts}" fill="none" stroke="{PLUM}" stroke-width="2.5"/>')
    for m, label, dx in ((0.0, "start: m = 0, loss ln 2 = 0.693", 10), (0.2, "worked example: m = 0.2, loss 0.598", 10)):
        b.append(f'<circle cx="{sx(m):.1f}" cy="{sy(f(m)):.1f}" r="5" fill="{INK}" class="f-ink">'
                 f'<title>{esc(label)}</title></circle>')
    b.append(text(sx(0) + 12, sy(f(0)) - 26, "start: m = 0, loss = ln 2 = 0.693", 11.5, weight=700))
    b.append(text(sx(0.2) + 12, sy(f(0.2)) + 4, "34.3's example: m = 0.2, loss 0.598", 11.5))
    b.append(text(x0 + (x1 - x0) / 2, y1 + 36, "margin m: how much more the policy prefers the chosen answer than "
                  "the reference did", 11.5, "f-muted", MUTED, "middle"))
    b.append(text(sx(1.4), sy(1.6), "the loss never reaches 0: it keeps pushing the margin up", 11.5,
                  "f-muted", MUTED))
    return svg("fig34-1-dpo-loss.svg", w, h, "Figure 34.1 · The DPO loss against the margin",
               "The loss minus log sigmoid of the margin falls from about 3 at margin minus 3 to near 0 at margin 5; "
               "it is ln 2 at margin 0 and 0.598 at margin 0.2.", b)


def fig_34_2_grpo_group():
    import ch34_preferences as E
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    a = E.group_advantages(r)
    close(float(a[0, 0]), 0.866, 5e-4, "advantage")
    w, h = 760, 280
    b = [text(20, 28, "Group-relative advantages, one prompt per panel", 15, weight=700, family=SERIF),
         text(20, 48, "four sampled answers each · reward 1 if right, 0 if wrong · advantage = (R − mean) / std · "
              "illustrative", 12, "f-muted", MUTED)]
    for pnl, (title, rr, aa) in enumerate((("two right, two wrong", r[0], a[0]), ("all four right", r[1], a[1]))):
        px = 30 + pnl * 370
        b.append(text(px, 82, title, 12.5, weight=700))
        mid = 170
        b.append(line(px, mid, px + 320, mid, INK, "f-ink-stroke", 1))
        for k in range(4):
            x = px + 20 + k * 75
            v = float(aa[k])
            hgt = 55 * abs(v)
            col, cls = (TEAL, "f-teal") if v > 0 else (CORAL, "f-coral")
            if hgt > 0:
                b.append(rect(x, mid - hgt if v > 0 else mid, 46, hgt, col, cls, 3,
                              tip=f"answer {k + 1}: reward {float(rr[k]):.0f}, advantage {v:+.3f}"))
            b.append(text(x + 23, (mid - hgt - 8) if v > 0 else (mid + hgt + 16) if v < 0 else mid - 8,
                          f"{v:+.3f}", 11, anchor="middle", weight=700, family=MONO))
            b.append(text(x + 23, 260, f"R = {float(rr[k]):.0f}", 11, "f-muted", MUTED, "middle", family=MONO))
        if pnl == 1:
            b.append(text(px + 160, 205, "every advantage 0: nothing to learn", 11.5, "f-muted", MUTED, "middle"))
    return svg("fig34-2-grpo-group.svg", w, h, "Figure 34.2 · Group-relative advantages",
               "Rewards 1, 0, 0, 1 give advantages +0.866, -0.866, -0.866, +0.866; a group that is all right gives "
               "zeros.", b)
