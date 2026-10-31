"""
The book's front and back matter for build_preview.py: the cover, the part
pages, the contents and the index.

The index is built from the chapters themselves: every term below is looked
up in every numbered section, lab and code tour, so it can't point at a page
that doesn't mention it, and it changes when the text does.
"""

from __future__ import annotations

import html
import random
import re

PARTS = [
    ("I", "Foundations", ["ch01-smallest-model", "ch02-data", "ch03-tokenizer"],
     "The whole idea in one small file, then the two things every model is made of before training starts: "
     "its data and its tokenizer."),
    ("II", "Build AnuLM from scratch", ["ch04-skeleton", "ch05-positions", "ch06-attention", "ch07-experts",
                                        "ch08-pretrain"],
     "Write the model piece by piece until it loads into model.py and gives the same answers, then pre-train it."),
    ("III", "Post-training", ["ch09-sft", "ch10-lora-qlora", "ch11-dpo", "ch12-decide-rag"],
     "Teach the model to answer, cheaply, by preference, and to decide and look things up."),
    ("IV", "Multimodal and shipping", ["ch13-senses", "ch14-one-model", "ch15-shipping"],
     "Give it ears and eyes, put every skill in one model, find what that broke, and release it."),
    ("V", "Going further", ["ch16-continue", "ch17-distill", "ch18-chat", "ch19-rl", "ch20-many-gpus"],
     "Keep pre-training and read longer windows, learn from a teacher, hold a conversation, practise "
     "with a checker as the reward, and train on more than one GPU."),
]

BLURB = {
    "ch01-smallest-model": "Count bytes, then a tiny transformer: the whole idea in one file.",
    "ch02-data": "Documents, cleaning, an honest validation split, decontamination and licences.",
    "ch03-tokenizer": "Byte-pair encoding, trained and held to bpe.py merge for merge.",
    "ch04-skeleton": "Embeddings, RMSNorm, SwiGLU and the residual stream.",
    "ch05-positions": "Rotary positions, why they work, and YaRN past the trained length.",
    "ch06-attention": "The causal mask, grouped-query heads and the KV cache.",
    "ch07-experts": "Routed experts, and keeping them balanced without a loss.",
    "ch08-pretrain": "Generation, schedules, AdamW, accumulation and checkpoints you can resume.",
    "ch09-sft": "Templates, packing, grading only the answer, and forgetting.",
    "ch10-lora-qlora": "LoRA, QLoRA in four bits, and what an adapter on experts must keep.",
    "ch11-dpo": "Preference pairs, and why the DPO loss can fall while the model gets worse.",
    "ch12-decide-rag": "Calibrated decisions in one pass, and BM25 search.",
    "ch13-senses": "New ids for speech, projected vectors for pictures, and whether it looks.",
    "ch14-one-model": "Many tasks in one model, and the regression its score hid.",
    "ch15-shipping": "Release folders, precision, and gates that decide.",
    "ch16-continue": "Resuming a finished run, mixing sources honestly, and YaRN on a trained model.",
    "ch17-distill": "A bigram teacher and a model teacher, and when a teacher is worth having.",
    "ch18-chat": "Conversations of several turns, which replies to grade, and a held-out set that leaked.",
    "ch19-rl": "GRPO with a program as the reward, against rejection sampling.",
    "ch20-many-gpus": "Data parallelism from scratch, and the two bugs only experts have.",
}

# (index entry, regex, case-sensitive?[, only these chapters])
TERMS = [
    ("AdamW", r"\bAdamW\b", True), ("alpha / r (LoRA scale)", r"alpha / r", False),
    ("attention", r"\battention\b", False), ("bfloat16", r"\bb(?:float)?16\b", False),
    ("BM25", r"\bBM25\b", True), ("BPE (byte-pair encoding)", r"\bBPE\b|byte-pair", False),
    ("calibration", r"\bcalibrat", False), ("catastrophic forgetting", r"\bforg(?:et|ot)", False),
    ("causal mask", r"\bcausal\b", False), ("checkpoint", r"\bcheckpoints?\b", False),
    ("clipping, gradient", r"\bclip", False), ("contamination", r"contaminat", False),
    ("cosine schedule", r"\bcosine\b", False), ("cross-entropy", r"cross-entropy", False),
    ("deduplication", r"\bdedup|duplicates?\b", False), ("Devanagari", r"Devanagari", True),
    ("DPO", r"\bDPO\b", True), ("ECE (expected calibration error)", r"\bECE\b", True),
    ("embeddings", r"\bembedding", False), ("EOS (end of text)", r"\bEOS\b", True),
    ("experts", r"\bexperts?\b", False), ("expert biases", r"expert(?:s')? (?:balancing )?bias|expert_bias|balancing bias", False),
    ("gates, release", r"\bgates?\b", False), ("gradient accumulation", r"\baccumulat", False),
    ("grouped-query attention", r"grouped-query|\bGQA\b", False), ("Hindi", r"\bHindi\b", True),
    ("idf", r"\bidf\b", False), ("KV cache", r"\bKV cache", True),
    ("LayerNorm", r"LayerNorm", True), ("learning rate", r"learning rate", False),
    ("licences", r"\blicen[cs]e", False), ("LoRA", r"\bLoRA\b", True),
    ("merging adapters", r"\bmerg(?:e|ed|ing)\b|merge_lora", False, {"9", "10", "11", "12", "13", "14", "15"}),
    ("merges, BPE", r"\bmerges?\b", False, {"3"}), ("mixture of experts (MoE)", r"mixture of experts|\bMoE\b", False),
    ("NF4", r"\bNF4\b", True), ("packing", r"\bpack(?:ed|ing)?\b", False),
    ("projector", r"\bprojectors?\b", False), ("QLoRA", r"\bQLoRA\b", True),
    ("RAG (retrieval)", r"\bRAG\b|retriev", False), ("repetition penalty", r"repetition penalty", False),
    ("replay", r"\breplay", False), ("residual stream", r"\bresidual", False),
    ("RMSNorm", r"RMSNorm", True), ("RoPE (rotary positions)", r"\bRoPE\b|\brotary\b", False),
    ("router", r"\brout(?:er|ing)\b", False), ("safetensors", r"safetensors", False),
    ("score, multi-task", r"\bscore\b", False), ("SigLIP", r"SigLIP", True),
    ("sliding window", r"sliding window", False), ("SNAC audio codes", r"\bSNAC\b", True),
    ("softmax", r"\bsoftmax\b", False), ("SwiGLU", r"SwiGLU", True),
    ("temperature", r"\btemperature\b", False), ("tokenizer", r"\btokeni[sz]", False),
    ("top-k", r"\btop-k\b", False), ("validation split", r"validation", False),
    ("vocabulary, growing", r"\bvocabular", False), ("warmup", r"\bwarm-?up", False),
    ("weight decay", r"weight decay", False), ("Whisper", r"\bWhisper\b", True),
    ("WSD (warmup-stable-decay)", r"\bWSD\b|warmup-stable-decay", False), ("YaRN", r"\bYaRN\b", True),
    ("all-reduce", r"all-reduce|all_reduce", False), ("data parallelism", r"data.parallel|DistributedDataParallel", False),
    ("distillation", r"\bdistil", False), ("GRPO", r"\bGRPO\b", True),
    ("rejection sampling", r"rejection sampling", False), ("rewarmup", r"\brewarm", False),
    ("conversations, multi-turn", r"\bconversations?\b", False), ("reward", r"\brewards?\b", False),
    ("KL divergence", r"\bKL\b", True), ("held-out leak", r"\bleak", False),
]

CSS = """
.cover-sheet { background: #161A2E; color: #F3EEE6; margin: 0; }
.cover-inner { max-width: 60rem; margin: 0 auto; min-height: min(100vh, 62rem); display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 2rem; align-items: center;
  padding-block: 3rem; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.cover-art svg { width: 100%; height: auto; display: block; }
.cover-kicker { font: 600 0.8rem/1.4 var(--sans); letter-spacing: 0.16em; text-transform: uppercase; color: #F0954F; }
.cover-title { font: 700 clamp(2.4rem, 6.4vw, 4.2rem)/1.02 var(--serif); margin: 0.8rem 0 1rem; text-wrap: balance;
  color: #F3EEE6; }
.cover-title em { font-style: italic; color: #F0954F; }
.cover-sub { font: 400 1.1rem/1.5 var(--serif); color: #C9C3B8; max-width: 26rem; margin: 0 0 2rem; }
.cover-by { font: 600 0.9rem/1.5 var(--sans); color: #F3EEE6; border-top: 1px solid #3A4060; padding-top: 1rem;
  max-width: 26rem; }
.cover-by span { display: block; font-weight: 500; color: #9AA0B8; }
@media (max-width: 44rem) { .cover-inner { grid-template-columns: 1fr; min-height: auto; } .cover-art { order: 2; } }
.front { padding-top: 3rem; }
.front h1 { font: 700 2.2rem/1.15 var(--serif); margin: 0 0 1.5rem; }
.contents-part { margin: 2.2rem 0 0.8rem; font: 600 0.8rem var(--sans); letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--saffron); }
.contents-ch { display: grid; grid-template-columns: 2.6rem minmax(0, 1fr); gap: 0 0.6rem; margin: 0.9rem 0; }
.contents-num { font: 700 1.6rem/1.1 var(--serif); color: var(--saffron); }
.contents-ch a.ct { font: 650 1.08rem/1.3 var(--sans); color: var(--ink); text-decoration: none; }
.contents-ch a.ct:hover { color: var(--indigo); }
.contents-blurb { margin: 0.15rem 0 0.3rem; color: var(--muted); font-size: 0.95rem; }
.contents-secs { font: 0.82rem/1.55 var(--sans); color: var(--muted); }
.contents-secs a { color: var(--muted); text-decoration: none; }
.contents-secs a:hover { color: var(--indigo); text-decoration: underline; }
.part-page { margin-top: 4rem; padding: 3rem 0 2rem; border-top: 3px solid var(--ink); }
.part-num { font: 600 0.85rem var(--sans); letter-spacing: 0.2em; text-transform: uppercase; color: var(--saffron); }
.part-page h1 { font: 700 2.6rem/1.1 var(--serif); margin: 0.4rem 0 0.8rem; text-wrap: balance; }
.part-page p { color: var(--muted); font-size: 1.05rem; }
.index-letter { font: 700 1.3rem var(--serif); color: var(--saffron); margin: 1.2rem 0 0.2rem; break-after: avoid; }
.index-cols { columns: 2 17rem; column-gap: 2.5rem; font: 0.9rem/1.6 var(--sans); }
.index-entry { break-inside: avoid; margin: 0.1rem 0; }
.index-entry a { color: var(--indigo); text-decoration: none; font-variant-numeric: tabular-nums; }
.index-entry a:hover { text-decoration: underline; }
.index-note { color: var(--muted); font-size: 0.95rem; }
"""


def cover_art() -> str:
    """Bytes on the left become routes into experts, and the experts open onto sound and sight."""
    rnd = random.Random(7)
    W, H = 520, 640
    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
         f'aria-label="Bytes flowing through routes into experts, then out to sound and pictures">']
    glyphs = list("The quick ") + ["भा", "र", "त", "E0", "A4", "ि", "न"] + list("byte") + ["0x", "ा", "क"]
    for r in range(14):
        for c in range(5):
            x, y = 24 + c * 34, 40 + r * 40
            a = 0.15 + 0.85 * (1 - c / 5) * rnd.random()
            lit = rnd.random() < 0.08
            p.append(f'<rect x="{x}" y="{y}" width="28" height="28" rx="4" fill="{"#F0954F" if lit else "#2B3152"}" '
                     f'fill-opacity="{0.95 if lit else a:.2f}"/>')
            if rnd.random() < 0.55:
                g = html.escape(rnd.choice(glyphs))
                p.append(f'<text x="{x + 14}" y="{y + 19}" text-anchor="middle" font-family="JetBrains Mono, monospace" '
                         f'font-size="11" fill="{"#161A2E" if lit else "#C9C3B8"}" fill-opacity="{0.9 if lit else a:.2f}">{g}</text>')
    experts = [(330, 70 + k * 72) for k in range(8)]
    for k in range(60):
        y0 = 54 + rnd.random() * 540
        ex, ey = experts[rnd.randrange(8)]
        hot = k % 9 == 0
        p.append(f'<path d="M 196 {y0:.0f} C 260 {y0:.0f}, 270 {ey}, {ex - 18} {ey}" fill="none" '
                 f'stroke="{"#F0954F" if hot else "#5B66B8"}" stroke-width="{1.6 if hot else 0.8}" '
                 f'stroke-opacity="{0.9 if hot else 0.35}"/>')
    for k, (ex, ey) in enumerate(experts):
        hot = k in (2, 5)
        p.append(f'<circle cx="{ex}" cy="{ey}" r="18" fill="{"#F0954F" if hot else "#232846"}" '
                 f'stroke="{"#F0954F" if hot else "#5B66B8"}" stroke-width="1.5"/>')
    for k in range(18):                                      # a waveform, for hearing
        x = 400 + k * 6
        h = 6 + 30 * abs(__import__("math").sin(k * 0.7)) * (0.4 + 0.6 * rnd.random())
        p.append(f'<rect x="{x}" y="{200 - h / 2:.1f}" width="3.5" height="{h:.1f}" rx="1.5" fill="#9AA6F2"/>')
    for r in range(4):                                       # a patch grid, for seeing
        for c in range(4):
            p.append(f'<rect x="{404 + c * 24}" y="{380 + r * 24}" width="20" height="20" rx="2" '
                     f'fill="#{"F0954F" if (r + c) % 3 == 0 else "3B4CCA"}" fill-opacity="{0.35 + 0.15 * ((r * 4 + c) % 4):.2f}"/>')
    for (ex, ey) in (experts[2], experts[5]):
        ty = 200 if ey < 320 else 428
        p.append(f'<path d="M {ex + 18} {ey} C 370 {ey}, 370 {ty}, 396 {ty}" fill="none" stroke="#F0954F" '
                 f'stroke-width="1.6" stroke-opacity="0.9"/>')
    for x, label in ((24, "bytes"), (300, "experts"), (404, "hear · see")):
        p.append(f'<text x="{x}" y="626" font-family="Inter, sans-serif" font-size="11" letter-spacing="2" '
                 f'fill="#9AA0B8">{label.upper()}</text>')
    p.append("</svg>")
    return "".join(p)


def cover() -> str:
    return ('<header class="cover-sheet"><div class="cover-inner"><div>'
            '<div class="cover-kicker">The AnuLM lab · a hands-on course</div>'
            '<h1 class="cover-title">Build Your Own LLM, <em>from Scratch to Omni</em></h1>'
            '<p class="cover-sub">Pre-train a mixture-of-experts language model on your own data, post-train it '
            'every way that matters, give it ears and eyes, and ship it, on one consumer GPU or a laptop. '
            'Every figure is drawn from a real run, and every chapter has checks that catch the bugs a falling '
            'loss hides.</p>'
            '<div class="cover-by">AnuLM<span>Draft edition · October 2026</span></div>'
            f'</div><div class="cover-art">{cover_art()}</div></div></header>')


def label(num: str, heading: str) -> str | None:
    """The short name a section gets in the contents and the index, or None to leave it out."""
    m = re.match(r"(\d+\.\d+)\s", heading)
    if m:
        return m.group(1)
    if heading.startswith("Lab"):
        return f"{num} lab"
    if heading.startswith("Code tour"):
        return f"{num} code tour"
    return None


def sections(md: str):
    """(heading, raw text) for every ## section of a chapter."""
    out, head, buf = [], None, []
    for line in md.split("\n"):
        if line.startswith("## "):
            if head:
                out.append((head, "\n".join(buf)))
            head, buf = line[3:].strip(), []
        elif head:
            buf.append(line)
    if head:
        out.append((head, "\n".join(buf)))
    return out


def contents(chapters, anchor) -> str:
    by = {c["name"]: c for c in chapters}
    out = ['<section class="front" id="contents"><h1>Contents</h1>']
    for roman, title, names, _ in PARTS:
        names = [n for n in names if n in by]
        if not names:
            continue
        out.append(f'<div class="contents-part">Part {roman} · {html.escape(title)}</div>')
        for n in names:
            c = by[n]
            secs = [f'<a href="#{anchor(n, h)}">{html.escape(re.sub(r"^(\d+\.\d+)\s+", r"\1 ", h).replace("`", ""))}</a>'
                    for h, _ in c["sections"] if label(c["num"], h)]
            out.append(f'<div class="contents-ch"><div class="contents-num">{c["num"]}</div><div>'
                       f'<a class="ct" href="#{n}">{html.escape(c["title"])}</a>'
                       f'<p class="contents-blurb">{html.escape(BLURB.get(n, ""))}</p>'
                       f'<div class="contents-secs">{" · ".join(secs)}</div></div></div>')
    out.append('<div class="contents-part">Back matter</div>'
               '<div class="contents-ch"><div class="contents-num"></div><div><a class="ct" href="#index">Index</a>'
               '</div></div></section>')
    return "".join(out)


def part_page(name: str) -> str:
    for roman, title, names, intro in PARTS:
        if names and names[0] == name:
            return (f'<section class="part-page" id="part-{roman.lower()}"><div class="part-num">Part {roman}</div>'
                    f"<h1>{html.escape(title)}</h1><p>{html.escape(intro)}</p></section>")
    return ""


def index(chapters, anchor) -> str:
    entries = []
    for term, pat, case, *only in TERMS:
        rx = re.compile(pat, 0 if case else re.I)
        refs = []
        for c in chapters:
            if only and c["num"] not in only[0]:
                continue
            for h, text in c["sections"]:
                lab = label(c["num"], h)
                if lab and (rx.search(text) or rx.search(h)):
                    refs.append((lab, anchor(c["name"], h)))
        if refs:
            entries.append((term, refs))
    entries.sort(key=lambda e: e[0].lower())
    out = ['<section class="front" id="index"><h1>Index</h1>',
           '<p class="index-note">Numbers are sections; "lab" and "code tour" are the end of a chapter. '
           'Built from the text, so every reference mentions its term.</p><div class="index-cols">']
    letter = ""
    for term, refs in entries:
        first = term[0].upper()
        if first != letter:
            letter = first
            out.append(f'<div class="index-letter">{letter}</div>')
        links = ", ".join(f'<a href="#{a}">{html.escape(l)}</a>' for l, a in refs)
        out.append(f'<div class="index-entry">{html.escape(term)} &nbsp;{links}</div>')
    out.append("</div></section>")
    return "".join(out)
