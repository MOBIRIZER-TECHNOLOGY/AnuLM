"""
Figures for Part VIII (Beyond text, Chapters 35-37) and Part IX (To production, Chapters 38-40).

    python book/tools/make_figures.py part_viii_ix

Every number drawn is computed here from the repository code or read from the
book's measurement files, and asserted against the value the chapter prints.
Part VIII's accent is plum, Part IX's teal (STYLE.md section 1).
"""

from __future__ import annotations

import re

import json
import sys
from pathlib import Path

from make_figures import (BOOK, CARD, CORAL, GOLD, INK, MONO, MUTED, NIGHT, PAPER, ROOT, RULE, SERIF, SLATE,
                          TEAL, close, esc, line, rect, svg, text)

sys.path.insert(0, str(ROOT))
PLUM = "#7B5E8E"
MEAS = BOOK / "measurements"


def _box(x, y, w, h, title, sub=None, accent=None, dashed=False, fill=CARD, size=13):
    stroke = f' stroke="{accent}" stroke-width="2.5"' if accent else f' stroke="{RULE}" stroke-width="1"'
    if dashed:
        stroke = f' stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5 4"'
    out = [rect(x, y, w, h, fill, "f-card", 8, stroke)]
    out.append(text(x + w / 2, y + (h / 2 + 4 if not sub else h / 2 - 3), title, size, anchor="middle", weight=700))
    if sub:
        out.append(text(x + w / 2, y + h / 2 + 14, sub, 11, "f-muted", MUTED, "middle", family=MONO))
    return out


def _arrow(x1, y1, x2, y2, dashed=False):
    d = ' stroke-dasharray="4 3"' if dashed else ""
    return line(x1, y1, x2, y2, MUTED, "f-muted-stroke", 1.5, f' marker-end="url(#arrow)"{d}')



def _pin() -> str:
    """The commit the book cites, from the source map (as every book tool reads it)."""
    return re.search(r"`@([0-9a-f]{7,40})`", (BOOK / "REPOSITORY_SOURCE_MAP.md").read_text(encoding="utf-8")).group(1)

# --------------------------------------------------------------------------- Ch 35
def fig_35_1_vocab():
    from anulm.multimodal import speech_vocab as SV
    from anulm.multimodal.audio_codec import AUDIO_VOCAB, LEVEL_VOCAB
    v = SV.SpeechVocab(32_768)
    assert (v.audio_base, v.speech_bos, v.image_eos, v.total) == (32_768, 61_440, 61_444, 61_445)
    w, h = 760, 300
    x0, x1 = 30, 730
    sx = lambda i: x0 + (x1 - x0) * i / v.total
    b = [text(20, 28, "The grown vocabulary, to scale", 15, weight=700, family=SERIF),
         text(20, 48, "released-model sizes · ids keep their numbers; new ids are appended", 12, "f-muted", MUTED)]
    y = 88
    b.append(rect(sx(0), y, sx(v.text) - sx(0) - 1, 54, CARD, "f-card", 4, f' stroke="{SLATE}" stroke-width="1"',
                  tip=f"text: ids 0 to {v.text - 1:,}"))
    b.append(text((sx(0) + sx(v.text)) / 2, y + 24, "text (BPE)", 13, anchor="middle", weight=700))
    b.append(text((sx(0) + sx(v.text)) / 2, y + 42, f"[0, {v.text:,})", 11, "f-muted", MUTED, "middle", family=MONO))
    for j in range(7):
        a = v.audio_base + j * LEVEL_VOCAB
        b.append(rect(sx(a), y, sx(a + LEVEL_VOCAB) - sx(a) - 1, 54, PLUM, "f-plum", 3,
                      f' fill-opacity="{0.45 + 0.07 * j:.2f}"', tip=f"audio slot {j}: ids {a:,} to {a + LEVEL_VOCAB - 1:,}"))
        b.append(text((sx(a) + sx(a + LEVEL_VOCAB)) / 2, y + 32, str(j), 12, "f-onfill", PAPER, "middle", 700))
    b.append(text((sx(v.audio_base) + sx(v.speech_bos)) / 2, y - 10, f"audio: 7 slots × 4,096 = {AUDIO_VOCAB:,}",
                  12, anchor="middle", weight=700))
    b.append(text(sx(v.audio_base), y + 74, f"{v.audio_base:,}", 11, "f-muted", MUTED, "middle", family=MONO))
    b.append(text(sx(v.speech_bos) - 6, y + 74, f"{v.speech_bos:,}", 11, "f-muted", MUTED, "end", family=MONO))
    b.append(line(sx(v.total), y - 4, sx(v.total), y + 58, INK, "f-ink-stroke", 3))
    # control tokens, magnified
    names = ["<|speech|>", "<|/speech|>", "<|text|>", "<|image|>", "<|/image|>"]
    cy = 196
    b.append(text(20, cy - 12, "the last five ids, magnified: control tokens", 12, weight=700))
    for i, nm in enumerate(names):
        bx = 20 + i * 146
        b.append(rect(bx, cy, 136, 46, CARD, "f-card", 6, f' stroke="{PLUM}" stroke-width="1.5"',
                      tip=f"{nm}: id {v.speech_bos + i:,}"))
        b.append(text(bx + 68, cy + 20, nm, 12, anchor="middle", weight=700, family=MONO))
        b.append(text(bx + 68, cy + 37, f"{v.speech_bos + i:,}", 11, "f-muted", MUTED, "middle", family=MONO))
    b.append(_arrow(744, y + 60, 714, cy - 4))
    b.append(text(20, 272, f"total V = {v.total:,}; generate masks every id from {v.text:,} up when text_vocab is set",
                  11.5, "f-muted", MUTED))
    return svg("fig35-1-vocab-layout.svg", w, h, "Figure 35.1 · The grown vocabulary",
               "A bar from 0 to 61,445. Text ids fill 0 to 32,767. Seven audio blocks of 4,096 ids each, one per SNAC "
               "slot, fill 32,768 to 61,439. Five control tokens follow: <|speech|> 61,440, <|/speech|> 61,441, "
               "<|text|> 61,442, <|image|> 61,443, <|/image|> 61,444.", b)


def fig_35_2_asr_sequence():
    tokens = [("<|speech|>", "emb", -1), ("audio 1", "proj", -1), ("audio 2", "proj", -1),
              ("<|text|>", "emb", "5"), ("id 5", "emb", "6"), ("id 6", "emb", "7"), ("id 7", "emb", "EOS"),
              ("EOS", "emb", -1)]
    want = [-1, -1, -1, 5, 6, 7, 1, -1]                # ch35_wider_vocab.py step 7, EOS = id 1
    assert [(-1 if t == -1 else (1 if t == "EOS" else int(t))) for _, _, t in tokens] == want
    w, h = 760, 300
    b = [text(20, 28, "One speech example: vectors in, targets one position early", 15, weight=700, family=SERIF),
         text(20, 48, "ContinuousSpeech.build · positions 0–7 of ch35_wider_vocab.py's step 7", 12, "f-muted", MUTED)]
    cw, x0, y0 = 86, 30, 92
    b.append(text(x0, y0 - 14, "embeds (S, D)", 12, weight=700))
    for i, (nm, src, tgt) in enumerate(tokens):
        x = x0 + i * cw
        is_proj = src == "proj"
        b.append(rect(x, y0, cw - 6, 46, PLUM if is_proj else CARD, "f-plum" if is_proj else "f-card", 6,
                      "" if is_proj else f' stroke="{RULE}" stroke-width="1"',
                      tip=f"position {i}: {nm}, from {'the projector' if is_proj else 'embed_tokens'}"))
        b.append(text(x + (cw - 6) / 2, y0 + 20, nm, 11.5, "f-onfill" if is_proj else "f-ink",
                      PAPER if is_proj else INK, "middle", 700, MONO))
        b.append(text(x + (cw - 6) / 2, y0 + 36, "projector" if is_proj else "embed_tokens", 10,
                      "f-onfill" if is_proj else "f-muted", PAPER if is_proj else MUTED, "middle"))
        b.append(text(x + (cw - 6) / 2, y0 + 64, str(i), 10, "f-muted", MUTED, "middle", family=MONO))
        ty = y0 + 104
        is_t = tgt != -1
        b.append(rect(x, ty, cw - 6, 34, CARD, "f-card", 6,
                      f' stroke="{PLUM}" stroke-width="2"' if is_t else f' stroke="{RULE}" stroke-width="1"',
                      tip=f"target at position {i}: {tgt}"))
        b.append(text(x + (cw - 6) / 2, ty + 22, str(tgt), 12, anchor="middle", weight=700 if is_t else 400,
                      family=MONO, cls="f-ink" if is_t else "f-muted", fill=INK if is_t else MUTED))
        if is_t and i + 1 < len(tokens):
            b.append(_arrow(x + (cw - 6) / 2 + 10, ty - 2, x + cw + (cw - 6) / 2 - 10, y0 + 50))
    b.append(text(x0, y0 + 96, "targets (S,): −1 is ignored by the loss", 12, weight=700))
    b.append(text(x0, 270, "Each graded position predicts the next input: <|text|> predicts the first word, the last "
                  "word predicts EOS.", 11.5, "f-muted", MUTED))
    return svg("fig35-2-asr-sequence.svg", w, h, "Figure 35.2 · One speech example as the model sees it",
               "Eight positions: <|speech|>, two projected audio vectors, <|text|>, three text ids and EOS. Targets are "
               "-1 except at positions 3 to 6, which hold 5, 6, 7 and EOS: each the id of the next input.", b)


# --------------------------------------------------------------------------- Ch 36
def fig_36_1_projectors():
    import torch
    from anulm.multimodal.speech_encoder import AudioProjector
    from anulm.multimodal.vision_encoder import ImageProjector
    torch.manual_seed(0)
    with torch.no_grad():
        a = AudioProjector(768, 1024, 4)(torch.randn(1, 501, 768))
        v = ImageProjector(768, 1024, 2)(torch.randn(196, 768))
    assert a.shape == (1, 125, 1024) and v.shape == (49, 1024)
    w, h = 760, 330
    b = [text(20, 28, "Two frozen encoders, two trained projectors", 15, weight=700, family=SERIF),
         text(20, 48, "shapes for a 10 s clip and one 224 × 224 image · released width D = 1,024", 12, "f-muted", MUTED)]

    def row(y, items):
        out = []
        x = 14
        for i, (title, sub, kind, ww) in enumerate(items):
            acc = PLUM if kind == "proj" else None
            fill = CARD
            out += _box(x, y, ww, 56, title, sub, accent=acc, fill=fill, size=12.5)
            if kind == "frozen":
                out.append(text(x + ww / 2, y + 72, "frozen", 10.5, "f-muted", MUTED, "middle"))
            if kind == "proj":
                out.append(text(x + ww / 2, y + 72, "trained", 10.5, "f-muted", MUTED, "middle", weight=700))
            if i + 1 < len(items):
                out.append(_arrow(x + ww + 2, y + 28, x + ww + 16, y + 28))
            x += ww + 18
        return out

    b.append(text(20, 86, "audio", 12, weight=700))
    b += row(96, [("16 kHz samples", "160,000", "in", 104), ("Whisper-small", "encoder", "frozen", 108),
                  ("frames", "(501, 768)", "x", 96), ("stack 4", "(125, 3,072)", "x", 100),
                  ("projector", "→ (125, 1,024)", "proj", 116), ("embeds", "125 positions", "x", 104)])
    b.append(text(20, 206, "image", 12, weight=700))
    b += row(216, [("pixels", "224 × 224 × 3", "in", 104), ("SigLIP ViT-B/16", "tower", "frozen", 108),
                   ("patches", "(196, 768)", "x", 96), ("pool 2 × 2", "(49, 3,072)", "x", 100),
                   ("projector", "→ (49, 1,024)", "proj", 116), ("embeds", "49 positions", "x", 104)])
    b.append(text(20, 318, "projector = Linear 3,072→3,072 · GELU · Linear 3,072→1,024 · LayerNorm: 12,589,056 "
                  "parameters each", 11.5, "f-muted", MUTED))
    return svg("fig36-1-projectors.svg", w, h, "Figure 36.1 · From pixels and samples to positions",
               "Audio: 16 kHz samples go through the frozen Whisper-small encoder to 501 frames of 768, are stacked "
               "four at a time to 125 vectors of 3,072, and projected to 125 positions of 1,024. Image: 224 by 224 "
               "pixels go through the frozen SigLIP tower to 196 patches of 768, are pooled 2 by 2 to 49 vectors of "
               "3,072, and projected to 49 positions.", b)


def fig_36_2_scale():
    import math
    r = json.loads((MEAS / "part_viii_ix_scale.json").read_text(encoding="utf-8"))
    close(r["base_v2_text_embedding_norm_mean"], 0.4989, 5e-5, "base text norm")
    close(r["omni_vision_proj_output_norm_mean"], 9.6928, 5e-5, "vision proj norm")
    bars = [("fresh projector (LayerNorm init)", 32.0, PLUM),
            ("SigLIP patch, before the projector", r["siglip_patch_norm_mean"], SLATE),
            ("joint model: image projector output", r["omni_vision_proj_output_norm_mean"], PLUM),
            ("joint model: audio projector, expected", r["omni_audio_proj_ln_expected_norm"], PLUM),
            ("base v2: text embedding row", r["base_v2_text_embedding_norm_mean"], SLATE),
            ("joint model: text embedding row", r["omni_text_embedding_norm_mean"], SLATE),
            ("joint model: grown audio/control row", r["omni_new_row_norm_mean"], SLATE)]
    w, h = 760, 360
    x0, x1 = 300, 700
    lo, hi = math.log10(0.02), math.log10(100)
    sx = lambda v: x0 + (x1 - x0) * (math.log10(v) - lo) / (hi - lo)
    b = [text(20, 28, "How long the vectors are that enter the backbone", 15, weight=700, family=SERIF),
         text(20, 48, "mean Euclidean norm, logarithmic axis · measured on ckpt_base_v2.pt and ckpt_omni.pt",
              12, "f-muted", MUTED)]
    y0 = 70
    for t in (0.1, 1, 10, 100):
        b.append(line(sx(t), y0 - 4, sx(t), y0 + len(bars) * 34))
        b.append(text(sx(t), y0 + len(bars) * 34 + 16, f"{t:g}", 10, "f-muted", MUTED, "middle", family=MONO))
    for i, (name, v, col) in enumerate(bars):
        y = y0 + i * 34
        b.append(text(x0 - 12, y + 16, name, 12, anchor="end"))
        b.append(rect(x0, y + 4, sx(v) - x0, 18, col, "f-plum" if col == PLUM else "f-slate", 3,
                      tip=f"{name}: {v:.4g}"))
        b.append(text(sx(v) + 8, y + 18, f"{v:.3g}", 12, weight=700, family=MONO))
    b.append(text(20, h - 14, "plum: what a projector emits · slate: encoder output and the embedding rows the "
                  "backbone was trained on", 11.5, "f-muted", MUTED))
    return svg("fig36-2-scale.svg", w, h, "Figure 36.2 · The scale a projector hands over",
               f"A fresh projector emits vectors of length 32; SigLIP patches are {r['siglip_patch_norm_mean']:.1f}; "
               f"the joint model's image projector emits {r['omni_vision_proj_output_norm_mean']:.2f} and its audio "
               f"projector is expected to emit {r['omni_audio_proj_ln_expected_norm']:.3g}; text embedding rows are "
               f"{r['base_v2_text_embedding_norm_mean']:.3f} in base v2 and {r['omni_text_embedding_norm_mean']:.3f} "
               f"in the joint model; grown rows {r['omni_new_row_norm_mean']:.3f}.", b)


# --------------------------------------------------------------------------- Ch 37
def fig_37_1_one_step():
    tasks = [("chat", "text", 0.07), ("read", "text", 0.14), ("intent", "text", 0.06), ("asr_en", "asr", 0.12),
             ("asr_hi", "asr", 0.32), ("caption", "image", 0.17), ("vqa", "image", 0.12)]
    m = json.loads((MEAS / "part_viii_ix_omni_log.json").read_text(encoding="utf-8"))
    assert [(t[0], t[1], t[3]) for t in m["tasks"]] == tasks
    w, h = 760, 330
    b = [text(20, 28, "One optimizer step of omni_train.py", 15, weight=700, family=SERIF),
         text(20, 48, "experiments/omni_one.py shares at @773b6a8 · one task's batch per step", 12, "f-muted", MUTED)]
    y0 = 70
    b.append(text(20, y0 + 4, "1. draw one task", 12, weight=700))
    x = 20
    for name, kind, p in tasks:
        ww = 690 * p
        b.append(rect(x, y0 + 14, ww - 2, 40, PLUM if name == "asr_hi" else CARD,
                      "f-plum" if name == "asr_hi" else "f-card", 4,
                      "" if name == "asr_hi" else f' stroke="{RULE}" stroke-width="1"', tip=f"{name} ({kind}): p = {p}"))
        col, cls = (PAPER, "f-onfill") if name == "asr_hi" else (INK, "f-ink")
        b.append(text(x + ww / 2, y0 + 31, name, 11 if ww > 50 else 9.5, cls, col, "middle", 700))
        b.append(text(x + ww / 2, y0 + 46, f"{p:.2f}", 10, cls, col, "middle", family=MONO))
        x += ww
    b.append(text(20, y0 + 72, "e.g. asr_hi, drawn with p = 0.32", 11, "f-muted", MUTED))
    y1 = 170
    b += _box(20, y1, 200, 56, "2. its batch builder", "speech_encoder.batches")
    b.append(_arrow(222, y1 + 28, 262, y1 + 28))
    b += _box(264, y1, 200, 56, "(embeds, targets)", "(B, S, D), (B, S) with −1")
    b.append(_arrow(466, y1 + 28, 506, y1 + 28))
    b += _box(508, y1, 232, 56, "3. one backbone", "model(None, tb, embeds=eb)", accent=PLUM)
    y2 = 254
    b += _box(20, y2, 220, 50, "4. backward, clip at 1.0", None)
    b.append(_arrow(242, y2 + 25, 282, y2 + 25))
    b += _box(284, y2, 220, 50, "AdamW: backbone 5e-5,", "projectors 1e-3")
    b.append(_arrow(506, y2 + 25, 546, y2 + 25))
    b += _box(548, y2, 192, 50, "router bias nudge", "update_expert_biases")
    b.append(_arrow(624, y1 + 58, 130, y2 - 2))
    return svg("fig37-1-one-step.svg", w, h, "Figure 37.1 · One training step",
               "Step 1 draws one of seven tasks with probabilities chat 0.07, read 0.14, intent 0.06, asr_en 0.12, "
               "asr_hi 0.32, caption 0.17, vqa 0.12. Step 2 builds that task's batch as embeddings and targets. Step 3 "
               "runs the single backbone. Step 4 back-propagates, clips, steps AdamW with two learning rates and nudges "
               "the router biases.", b)


def fig_37_2_task_ratios():
    m = json.loads((MEAS / "part_viii_ix_omni_log.json").read_text(encoding="utf-8"))
    ev = m["evals"]
    close(ev[-1]["score"], 0.5873, 1e-9, "final score")
    close(ev[-1]["ratios"]["chat"], 1.651, 5e-4, "final chat ratio")
    names = [t[0] for t in m["tasks"]]
    steps = [0] + [e["step"] for e in ev]
    w, h = 760, 400
    x0, x1, y0, y1 = 70, 600, 70, 340
    sx = lambda s: x0 + (x1 - x0) * s / 100_000
    sy = lambda v: y1 - (y1 - y0) * v / 1.8
    b = [text(20, 28, "Held-out loss ÷ step-0 loss, per task, over the first joint run", 15, weight=700,
              family=SERIF),
         text(20, 48, "logs/omni_one/6_train.log · lower is better · the score is the mean of the seven", 12,
              "f-muted", MUTED)]
    for v in (0, 0.5, 1.0, 1.5):
        b.append(line(x0, sy(v), x1, sy(v), RULE if v != 1.0 else INK, "f-rule" if v != 1.0 else "f-ink-stroke",
                      1, ' stroke-dasharray="3 3"' if v == 1.0 else ""))
        b.append(text(x0 - 8, sy(v) + 4, f"{v:.1f}", 10, "f-muted", MUTED, "end", family=MONO))
    for s in range(0, 100_001, 20_000):
        b.append(text(sx(s), y1 + 18, f"{s // 1000}k", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(text((x0 + x1) / 2, y1 + 36, "training step", 11, "f-muted", MUTED, "middle"))

    def path(vals, col, cls, width, dash=""):
        pts = " ".join(f"{sx(s):.1f},{sy(v):.1f}" for s, v in zip(steps, vals))
        return f'<polyline points="{pts}" fill="none" stroke="{col}" class="{cls}" stroke-width="{width}"{dash}/>'

    for n in names:
        vals = [1.0] + [e["ratios"][n] for e in ev]
        is_chat = n == "chat"
        b.append(path(vals, CORAL if is_chat else SLATE, "f-coral-stroke" if is_chat else "f-slate-stroke",
                      3 if is_chat else 1.2, "" if is_chat else ' stroke-opacity="0.7"'))
        for s, v in zip(steps, vals):
            b.append(f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="{3 if is_chat else 2}" '
                     f'fill="{CORAL if is_chat else SLATE}" class="{"f-coral" if is_chat else "f-slate"}">'
                     f'<title>{esc(n)} at step {s}: {v:.3f}</title></circle>')
        lab_y = sy(vals[-1]) + 4
        b.append(text(x1 + 10, lab_y + {"asr_en": 6, "asr_hi": -2, "caption": 6, "vqa": -6, "read": 0,
                                         "intent": 0, "chat": 0}[n], f"{n} {vals[-1]:.2f}", 11,
                      weight=700 if is_chat else 400))
    sc = [1.0] + [e["score"] for e in ev]
    b.append(path(sc, PLUM, "f-plum-stroke", 3, ' stroke-dasharray="7 4"'))
    b.append(text(x1 + 10, sy(sc[-1]) + 12, f"score {sc[-1]:.3f}", 11, weight=700))
    b.append(text(sx(30_000), sy(1.73) - 12, "chat: 3.27 → 5.39, above 1 at every evaluation", 11.5, weight=700))
    return svg("fig37-2-task-ratios.svg", w, h, "Figure 37.2 · The score improved; chat did not",
               "Seven lines of held-out loss divided by step-0 loss over 100,000 steps. Chat jumps to 1.70 by step "
               "10,000 and ends at 1.65; reading ends at 0.88, VQA 0.62, captions 0.44, Hindi speech 0.33, English "
               "speech 0.16, intents 0.03. The dashed score, their mean, falls from 1.0 to 0.587.", b)


# --------------------------------------------------------------------------- Ch 38
def fig_38_1_utf8_release():
    from anulm.inference import api, serve
    from types import SimpleNamespace
    eng = SimpleNamespace(tok=None)
    eng.decode = lambda ids: serve.Engine.decode(eng, ids)
    eng.decode_complete = lambda ids: serve.Engine.decode_complete(eng, ids)
    g = api.Generation.__new__(api.Generation)
    g.eng, g.stops, g.new, g.sent, g.text, g.finish, g.cancelled = eng, [], [], "", "", "length", False
    g.matched_stop = False
    word = "नमस्ते"
    rel = []
    for b in word.encode("utf-8"):
        g.new.append(b)
        rel.append(g._release(final=False))
    assert "".join(rel) == word and sum(1 for r in rel if r) == 6
    data = word.encode("utf-8")
    w, h = 760, 250
    cw, x0 = 40, 20
    b = [text(20, 28, "One byte per token, one character out when it is complete", 15, weight=700, family=SERIF),
         text(20, 48, "api.Generation._release on serve.Engine's byte decoder · the word नमस्ते, 18 bytes",
              12, "f-muted", MUTED)]
    b.append(text(x0, 80, "token (byte)", 11.5, weight=700))
    b.append(text(x0, 168, "released", 11.5, weight=700))
    for i, (by, r) in enumerate(zip(data, rel)):
        x = x0 + i * cw
        b.append(rect(x, 88, cw - 4, 34, CARD, "f-card", 4, f' stroke="{RULE}" stroke-width="1"',
                      tip=f"token {i}: byte 0x{by:02x}"))
        b.append(text(x + (cw - 4) / 2, 109, f"{by:02x}", 11, anchor="middle", family=MONO))
        if r:
            b.append(rect(x, 176, cw - 4, 40, TEAL, "f-teal", 4, tip=f"after token {i}: {r}"))
            b.append(text(x + (cw - 4) / 2, 202, r, 17, "f-onfill", PAPER, "middle", 700))
            b.append(_arrow(x + (cw - 4) / 2, 124, x + (cw - 4) / 2, 172))
        else:
            b.append(text(x + (cw - 4) / 2, 200, "–", 12, "f-muted", MUTED, "middle"))
    b.append(text(20, 238, "Before a character is complete, decode_complete holds its bytes back; "
                  "the delta is empty and nothing is sent.", 11.5, "f-muted", MUTED))
    return svg("fig38-1-utf8-release.svg", w, h, "Figure 38.1 · Releasing text safely",
               "Eighteen byte tokens, three per Devanagari character. After the first and second byte of each "
               "character nothing is released; after the third the whole character is released. Six releases in all.", b)


def fig_38_2_serving_path():
    w, h = 760, 360
    b = [text(20, 28, "The serving path in api.py", 15, weight=700, family=SERIF),
         text(20, 48, "many connections, one lock, one sequence on the device · dashed: not in the repository",
              12, "f-muted", MUTED)]
    for i, c in enumerate(("client A", "client B", "client C")):
        b += _box(20, 74 + i * 62, 110, 44, c, "HTTP / SSE", size=12)
        b.append(_arrow(132, 96 + i * 62, 168, 96 + i * 62))
    b += _box(170, 74, 150, 168, "", None)
    b.append(text(245, 100, "FastAPI routes", 12.5, anchor="middle", weight=700))
    b.append(text(245, 120, "chat_completions", 11, "f-muted", MUTED, "middle", family=MONO))
    b.append(text(245, 138, "→ Generation", 11, "f-muted", MUTED, "middle", family=MONO))
    b.append(text(245, 166, "one worker thread", 11, "f-muted", MUTED, "middle"))
    b.append(text(245, 182, "per streamed request", 11, "f-muted", MUTED, "middle"))
    b.append(text(245, 210, "queue → events()", 11, "f-muted", MUTED, "middle", family=MONO))
    b.append(text(245, 228, "polls every 5 ms", 11, "f-muted", MUTED, "middle"))
    b.append(_arrow(322, 158, 368, 158))
    b += _box(370, 128, 110, 60, "GPU lock", "threading.Lock", accent=TEAL)
    b.append(_arrow(482, 158, 528, 158))
    b += _box(530, 118, 210, 80, "model.generate", "(1, S) · bf16 autocast · KV cache", size=13)
    b.append(text(635, 214, "on_token → _release → emit", 11, "f-muted", MUTED, "middle", family=MONO))
    b.append(line(635, 220, 635, 262, MUTED, "f-muted-stroke", 1.2, ' marker-end="url(#arrow)"'))
    b.append(text(635, 280, "deltas back through the queue", 11, "f-muted", MUTED, "middle"))
    b += _box(370, 286, 160, 50, "batch scheduler", "EXTENSION", dashed=True, size=12)
    b += _box(548, 296, 192, 50, "paged KV cache", "EXTENSION", dashed=True, size=12)
    b.append(text(20, 300, "A disconnect cancels events();", 11.5, "f-muted", MUTED))
    b.append(text(20, 316, "finally sets gen.cancelled, and", 11.5, "f-muted", MUTED))
    b.append(text(20, 332, "on_token stops at the next token.", 11.5, "f-muted", MUTED))
    return svg("fig38-2-serving-path.svg", w, h, "Figure 38.2 · The serving path",
               "Three clients connect to FastAPI routes, which build a Generation and run it in a worker thread that "
               "reports deltas through a queue. Every generation passes one GPU lock to model.generate on a single "
               "sequence. A batch scheduler and a paged KV cache, drawn dashed, are not in the repository.", b)


# --------------------------------------------------------------------------- Ch 39
def fig_39_1_export():
    r = json.loads((MEAS / "part_viii_ix_export.json").read_text(encoding="utf-8"))
    assert r["release_files"]["model.safetensors"] == 796_522_272
    w, h = 760, 360
    b = [text(20, 28, "From a training checkpoint to two loaders", 15, weight=700, family=SERIF),
         text(20, 48, "export_hf.py at @773b6a8 · sizes from the released folder", 12, "f-muted", MUTED)]
    b += _box(20, 120, 140, 70, "ckpt_base_v2.pt", "pickle · fp32", size=12.5)
    b.append(_arrow(162, 155, 206, 155))
    b += _box(208, 128, 100, 54, "export_hf", "main()", accent=TEAL, size=12.5)
    b.append(_arrow(310, 155, 350, 155))
    b.append(rect(352, 62, 200, 262, CARD, "f-card", 10, f' stroke="{RULE}" stroke-width="1"'))
    b.append(text(452, 84, "release folder", 13, anchor="middle", weight=700))
    files = [("model.safetensors", "796.5 MB · bf16 + fp32"), ("config.json", "the dataclass, nested"),
             ("tokenizer.json", "+ the project's own"), ("README.md", "model card"),
             ("modeling_anulm.py", "the wrapper"), ("configuration_anulm.py", "config adapter"),
             ("model.py", "snapshot at export")]
    for i, (f, sub) in enumerate(files):
        y = 96 + i * 32
        b.append(text(366, y + 12, f, 11.5, weight=700, family=MONO))
        b.append(text(366, y + 25, sub, 10, "f-muted", MUTED))
    b.append(_arrow(554, 120, 594, 104))
    b += _box(596, 74, 144, 60, "load_checkpoint", "native AnuLM", size=12.5)
    b.append(_arrow(554, 250, 594, 236))
    b += _box(596, 206, 144, 70, "AutoModel…", "trust_remote_code", accent=TEAL, size=12.5)
    b.append(text(668, 292, "no beams, no padding,", 10.5, "f-muted", MUTED, "middle"))
    b.append(text(668, 306, "no inputs_embeds", 10.5, "f-muted", MUTED, "middle"))
    b.append(text(20, 346, "Speech, image and adapter checkpoints get extras.safetensors and no auto_map.",
                  11.5, "f-muted", MUTED))
    return svg("fig39-1-export.svg", w, h, "Figure 39.1 · From a training checkpoint to two loaders",
               "A pickled fp32 checkpoint goes through export_hf.main into a folder of seven kinds of file: "
               "model.safetensors (796.5 MB, bf16 and fp32), config.json, the tokenizer, a README, and three code "
               "files. load_checkpoint reads the folder into the native model; AutoModelForCausalLM reads it through "
               "the copied wrapper with trust_remote_code.", b)


def fig_39_2_dtypes():
    r = json.loads((MEAS / "part_viii_ix_export.json").read_text(encoding="utf-8"))
    hd = r["release_header"]
    assert hd["tensors"] == {"F32": 138, "BF16": 1413} or hd["tensors"] == {"BF16": 1413, "F32": 138}
    loss = r["heldout_loss"]
    vals = list(loss.values())
    close(vals[1] - vals[0], -0.00004, 6e-6, "bf16 storage delta")
    close(vals[2] - vals[0], 0.00312, 6e-6, "bias rounding delta")
    close(vals[5] - vals[0], 0.00377, 6e-6, "whole bf16 delta")
    groups = [("expert matrices", 1368, 19 * 24 * 3 * 1024 * 192, "BF16"),
              ("vocabulary tables", 2, 2 * 32768 * 1024, "BF16"),
              ("attention matrices", 40, 20 * (1536 + 1024) * 1024, "BF16"),
              ("layer-0 SwiGLU", 3, 3 * 1024 * 2816, "BF16"),
              ("router weights", 19, 19 * 24 * 1024, "F32"),
              ("norm gains", 81, 20 * 2 * 1024 + 1024 + 20 * 2 * 64, "F32"),
              ("router bias + load counts", 38, 19 * 48, "F32")]
    assert sum(g[2] for g in groups if g[3] == "BF16") == hd["bytes"]["BF16"] // 2
    assert sum(g[2] for g in groups if g[3] == "F32") == hd["bytes"]["F32"] // 4
    w, h = 760, 430
    b = [text(20, 28, "What the released file stores, and in which precision", 15, weight=700, family=SERIF),
         text(20, 48, "model.safetensors header · numbers per group · the rule: ≥ 100,000 numbers → bfloat16",
              12, "f-muted", MUTED)]
    import math
    x0, x1 = 230, 610
    sx = lambda v: x0 + (x1 - x0) * (math.log10(v) - 2) / (math.log10(3e8) - 2)
    for t in (1e3, 1e5, 1e7):
        b.append(line(sx(t), 64, sx(t), 64 + 7 * 30))
        b.append(text(sx(t), 64 + 7 * 30 + 14, f"{t:,.0f}", 10, "f-muted", MUTED, "middle", family=MONO))
    b.append(line(sx(1e5), 60, sx(1e5), 64 + 7 * 30, INK, "f-ink-stroke", 1.5, ' stroke-dasharray="4 3"'))
    for i, (name, n, numel, dt) in enumerate(groups):
        y = 70 + i * 30
        col = TEAL if dt == "BF16" else SLATE
        b.append(text(x0 - 10, y + 13, f"{name} ({n})", 12, anchor="end"))
        b.append(rect(x0, y, sx(numel) - x0, 18, col, "f-teal" if dt == "BF16" else "f-slate", 3,
                      tip=f"{name}: {n} tensors, {numel:,} numbers, {dt}"))
        b.append(text(sx(numel) + 6, y + 13, f"{numel:,} · {dt}", 11, weight=700, family=MONO))
    y = 300
    b.append(text(20, y, "held-out loss (64 × 512 tokens, seed 0), change against the native fp32 model:", 12,
                  weight=700))
    for i, (k, v) in enumerate(loss.items()):
        b.append(text(40, y + 20 + i * 17, f"{v:.5f}  {v - vals[0]:+.5f}   {k.strip()}", 11.5, family=MONO))
    return svg("fig39-2-dtypes.svg", w, h, "Figure 39.2 · What the released file stores",
               "Groups of tensors on a log scale of numbers: expert matrices 268,959,744, vocabulary tables 67,108,864, "
               "attention 52,428,800 and layer-0 SwiGLU 8,650,752 in bfloat16; router weights 466,944, norm gains "
               "44,544 and router bias and load counts 912 in float32. Held-out loss 3.37908 native, 3.37903 exported, "
               "3.38219 with router biases rounded, 3.38572 with them zeroed, 3.37905 centred then rounded, 3.38285 "
               "for the whole model in bfloat16.", b)


# --------------------------------------------------------------------------- Ch 40
def fig_40_1_gap():
    w, h = 760, 380
    b = [text(20, 28, "The repository against a serving stack", 15, weight=700, family=SERIF),
         text(20, 48, f"solid: at @{_pin()} · dashed: EXTENSION, not in the repository", 12, "f-muted", MUTED)]
    rows = [("requests", "one lock: one sequence on the device", "continuous batching"),
            ("KV cache", "dict per layer, grown by torch.cat", "paged blocks (PagedAttention)"),
            ("weights", "fp32 in memory, bf16 autocast", "int8 / int4 quantisation"),
            ("decoding", "one token per forward", "speculative: draft + verify"),
            ("attention", "PyTorch SDPA picks a kernel", "own fused kernels"),
            ("training", "one GPU, AdamW or Muon", "data / tensor / pipeline, ZeRO"),
            ("operations", "shared key, upload caps, error ids", "per-user auth, rate limits, metrics")]
    b.append(text(150, 76, "in the repository", 12, weight=700))
    b.append(text(470, 76, "a large serving stack adds", 12, weight=700))
    for i, (k, have, add) in enumerate(rows):
        y = 88 + i * 40
        b.append(text(20, y + 22, k, 12, weight=700))
        b.append(rect(140, y, 290, 32, CARD, "f-card", 6, f' stroke="{TEAL}" stroke-width="1.5"', tip=have))
        b.append(text(152, y + 21, have, 11.5))
        b.append(_arrow(432, y + 16, 456, y + 16, dashed=True))
        b.append(rect(460, y, 280, 32, PAPER, "f-paper", 6,
                      f' stroke="{MUTED}" stroke-width="1.2" stroke-dasharray="5 4"', tip=f"EXTENSION: {add}"))
        b.append(text(472, y + 21, add, 11.5, "f-muted", MUTED))
    return svg("fig40-1-gap.svg", w, h, "Figure 40.1 · The repository against a serving stack",
               "Seven rows. Requests: one lock against continuous batching. KV cache: a dict per layer grown by "
               "concatenation against paged blocks. Weights: fp32 with bf16 autocast against int8 or int4. Decoding: one "
               "token per forward against speculative decoding. Attention: PyTorch SDPA against own kernels. Training: "
               "one GPU against data, tensor and pipeline parallelism with ZeRO. Operations: an optional key against "
               "rate limits, metrics and filters.", b)


def fig_40_2_memory():
    r = json.loads((MEAS / "part_viii_ix_gap.json").read_text(encoding="utf-8"))
    wb = r["weight_bytes"]
    close(wb["bf16"] / 1e9, 0.795, 5e-4, "bf16 weights GB")
    close(r["adamw_fp32_bytes"] / 1e9, 6.36, 5e-3, "AdamW GB")
    GiB = 2 ** 30
    card = 16 * GiB
    ctx = 1024 * r["kv_bytes_per_token"]["bf16"]
    ctx_served = 1024 * r["kv_bytes_per_token"]["autocast (fp32 keys, bf16 values)"]
    sv = r["served"]
    assert (sv["contexts"], sv["contexts_with_stack"]) == (495, 478)
    bars = [("as served: fp32 + bf16 expert stack", wb["fp32"] + sv["expert_stack_bytes"], SLATE),
            ("hypothetical: weights bf16", wb["bf16"], SLATE),
            ("hypothetical: int8 + scales", wb["int8 (per-row fp16 scales)"], SLATE),
            ("hypothetical: int4 + scales", wb["int4 (fp16 scale per 64 weights)"], SLATE),
            ("1,024-token cache, as served", ctx_served, TEAL),
            ("1,024-token cache, bf16", ctx, TEAL),
            ("scores of one layer, fp32", r["score_matrix_bytes_per_layer"], TEAL),
            ("AdamW training state, fp32", r["adamw_fp32_bytes"], CORAL)]
    w, h = 760, 370
    x0, x1 = 250, 700
    sx = lambda v: x0 + (x1 - x0) * v / card
    b = [text(20, 28, "Sizes against one 16 GiB card", 15, weight=700, family=SERIF),
         text(20, 48, "released 400M model · linear axis to 16 GiB · arithmetic from model.py's shapes", 12,
              "f-muted", MUTED)]
    for t in (0, 4, 8, 12, 16):
        b.append(line(sx(t * GiB), 64, sx(t * GiB), 64 + len(bars) * 32))
        b.append(text(sx(t * GiB), 64 + len(bars) * 32 + 16, f"{t} GiB", 10, "f-muted", MUTED, "middle",
                      family=MONO))
    for i, (name, v, col) in enumerate(bars):
        y = 70 + i * 32
        b.append(text(x0 - 10, y + 14, name, 12, anchor="end"))
        b.append(rect(x0, y, max(sx(v) - x0, 2), 20, col,
                      {SLATE: "f-slate", TEAL: "f-teal", CORAL: "f-coral"}[col], 3, tip=f"{name}: {v:,} bytes"))
        lab = f"{v / 2 ** 20:,.0f} MiB" if v < GiB else f"{v / GiB:.2f} GiB"
        b.append(text(max(sx(v), x0 + 2) + 8, y + 15, lab, 11.5, weight=700, family=MONO))
    b.append(text(20, h - 14, f"Weights and KV cache only: at most {sv['contexts_with_stack']} full contexts beside the "
                  "expert stack (activations not counted); the lock serves one at a time.", 11.5, "f-muted", MUTED))
    return svg("fig40-2-memory.svg", w, h, "Figure 40.2 · A 16 GiB card for the released model",
               "As served, the fp32 weights plus the grouped path's bf16 expert stack take 1.98 GiB; hypothetical bf16, "
               "int8 and int4 weights 0.74, 0.37 and 0.20 GiB. A 1,024-token cache takes 30 MiB as served, 20 MiB in "
               "bf16; one layer's fp32 scores 64 MiB; AdamW's training state 5.93 GiB.", b)
