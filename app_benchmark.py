"""
The Benchmark tab of app_omni.py: the recorded API benchmark
(examples/api_demo/benchmark.py) -- every request, its response, its time and
its score, on held-out test data. Read-only: it shows the recording, it does
not re-run it (`python examples/api_demo/benchmark.py` does).
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

import gradio as gr

BENCH = Path(__file__).parent / "examples" / "api_demo" / "benchmark"

ORDER = ["decide_banking", "decide_banking_modernbert", "decide_hindi", "decide_general", "answer",
         "asr_en", "asr_hi", "caption", "vqa", "chat", "tts"]
NAME = {"decide_banking": "Banking intents", "decide_banking_modernbert": "Banking intents (ModernBERT)",
        "decide_hindi": "Hindi intents", "decide_general": "Zero-shot intents (77 options)",
        "answer": "Wikipedia lookup", "asr_en": "Speech to text, English", "asr_hi": "Speech to text, Hindi",
        "caption": "Image captions", "vqa": "Image questions (free-form)", "chat": "Chat", "tts": "Text to speech"}
DATA = {"decide_banking": "BANKING77 test", "decide_banking_modernbert": "BANKING77 test (same 300)",
        "decide_hindi": "MASSIVE hi-IN", "decide_general": "BANKING77 test", "answer": "20 curated questions",
        "asr_en": "LibriSpeech held-out", "asr_hi": "FLEURS Hindi dev", "caption": "Flickr30k val",
        "vqa": "A-OKVQA val", "chat": "held-out prompts, en + hi", "tts": "20 sentences"}


def ms(v) -> str:
    if v is None:
        return "—"
    return f"{v:,.0f} ms" if v < 1000 else f"{v / 1000:,.1f} s"


def load() -> tuple[dict, dict[str, list[dict]]]:
    if not (BENCH / "summary.json").exists():
        return {}, {}
    summary = json.loads((BENCH / "summary.json").read_text(encoding="utf-8"))
    rows = {}
    for name in ORDER:
        p = BENCH / f"{name}.jsonl"
        if p.exists():
            rows[name] = [json.loads(l) for l in p.open(encoding="utf-8")]
    return summary, rows


def score_of(s: dict) -> str:
    if "wer" in s:
        return f"WER {s['wer']:.1%}"
    if "accuracy" in s:
        return f"{s['accuracy']:.1%}"
    if s.get("tokens_per_s_mean"):
        return f"{s['tokens_per_s_mean']:.0f} tokens/s"
    if s.get("rtf_mean"):
        return f"{s['rtf_mean']:.2f} × real time"
    return "read the outputs"


def summary_table(summary: dict) -> list[list]:
    out = []
    for name in ORDER:
        s = summary.get("suites", {}).get(name)
        if s:
            out.append([NAME[name], DATA[name], s["n"], score_of(s), ms(s.get("ms_p50")), ms(s.get("ms_p95")),
                        ms(s.get("cold_ms")), s["errors"]])
    return out


def latency_html(summary: dict) -> str:
    """Median bar and 95th-percentile tick per test, log scale, plain HTML."""
    suites = [summary["suites"][n] for n in ORDER if n in summary.get("suites", {}) and summary["suites"][n].get("ms_p50")]
    if not suites:
        return ""
    lo = math.log10(max(1.0, min(s["ms_p50"] for s in suites) / 2))
    hi = math.log10(max(s["ms_p95"] for s in suites) * 1.3)
    pos = lambda v: 100 * (math.log10(max(v, 1)) - lo) / (hi - lo)
    grid = [10 ** k for k in range(math.ceil(lo), math.floor(hi) + 1)]
    lines = "".join(f'<div style="position:absolute;left:{pos(g):.1f}%;top:0;bottom:0;border-left:1px solid '
                    f'color-mix(in srgb, currentColor 15%, transparent)"></div>' for g in grid)
    labels = "".join(f'<span style="position:absolute;left:{pos(g):.1f}%;transform:translateX(-50%)">{ms(g)}</span>'
                     for g in grid)
    rows = []
    for s in suites:
        rows.append(
            f'<div style="display:contents">'
            f'<div style="text-align:right;padding-right:10px;white-space:nowrap">{html.escape(NAME[s["suite"]])}</div>'
            f'<div style="position:relative;height:22px">{lines}'
            f'<div title="median {ms(s["ms_p50"])}" style="position:absolute;left:0;top:4px;height:14px;'
            f'width:{pos(s["ms_p50"]):.1f}%;background:var(--color-accent, #2f6fdf);border-radius:4px"></div>'
            f'<div title="95th percentile {ms(s["ms_p95"])}" style="position:absolute;left:{pos(s["ms_p95"]):.1f}%;'
            f'top:1px;height:20px;border-left:2px solid currentColor"></div>'
            f'<span style="position:absolute;left:calc({pos(s["ms_p95"]):.1f}% + 8px);top:2px;opacity:.75;'
            f'white-space:nowrap">{ms(s["ms_p50"])}</span></div></div>')
    return (f'<div style="font-size:13px;font-variant-numeric:tabular-nums">'
            f'<div style="display:grid;grid-template-columns:max-content 1fr;row-gap:6px;padding-right:70px">'
            f'{"".join(rows)}<div></div><div style="position:relative;height:18px;opacity:.7">{labels}</div></div>'
            f'<p style="opacity:.75;margin:8px 0 0">Bar: median. Line: 95th percentile. Log scale: each gridline '
            f'is ten times the last. First requests (model loading) are left out; the table lists them.</p></div>')


def row_cells(name: str, i: int, r: dict) -> list:
    q = r["request"]
    if name.startswith("decide"):
        inp = q["input"]
    elif name == "answer":
        inp = q["question"]
    elif name.startswith("asr"):
        inp = f"{q['file']} ({(r.get('audio_seconds') or 0):.1f} s audio)"
    elif name == "caption":
        inp = r["image"]
    elif name == "vqa":
        inp = f"{r['question']} [{r['image']}]"
    elif name == "chat":
        inp = q["messages"][0]["content"]
    else:
        inp = q.get("input", "")
    exp = r.get("expected", "")
    if isinstance(exp, list):
        exp = " | ".join(exp[:2])
    if name.startswith("asr"):
        res = f"{r['word_errors']}/{r['ref_words']} words wrong"
    elif "correct" in r:
        res = ("✓" if r["correct"] else "✗") + (f" {r['confidence']:.2f}" if r.get("confidence") is not None else "")
    elif name == "chat":
        res = f"{r.get('completion_tokens', 0)} tokens"
    elif name == "tts":
        res = f"{r.get('audio_seconds')} s audio"
    else:
        res = ""
    t = ms(r["ms"]) + (" (first)" if r.get("cold") else "")
    return [i + 1, inp, exp, r.get("got", ""), res, t]


def add_tab(summary: dict | None = None, rows: dict | None = None):
    """Call inside a `gr.Blocks` context."""
    if summary is None:
        summary, rows = load()
    with gr.Tab("Benchmark"):
        if not summary:
            gr.Markdown("No recording yet: start `python api.py`, then run "
                        "`python examples/api_demo/benchmark.py`.")
            return
        total = sum(s["n"] for s in summary["suites"].values())
        errors = sum(s["errors"] for s in summary["suites"].values())
        gr.Markdown(
            f"### {total:,} real API requests on held-out test data\n"
            f"Recorded {summary.get('recorded', '')} on the {summary.get('device', '')} "
            f"(RTX 5070 Ti), one request at a time, timed from the client. "
            f"**{errors} server errors.** No test item was seen in training. "
            f"Re-run with `python examples/api_demo/benchmark.py`; "
            f"the files are in `examples/api_demo/benchmark/`.")
        gr.Dataframe(summary_table(summary), headers=["Test", "Data", "Requests", "Score", "Median", "95th pct",
                                                      "First request", "Errors"],
                     interactive=False, wrap=True)
        gr.Markdown("**Scores.** Intents: share right. Lookup: answer contains the expected text. Speech: word "
                    "error rate (lower is better). Image questions: the free-form answer must match exactly, a "
                    "far stricter test than the 43.5% four-way multiple choice. Real time: 0.11 means a 10 s "
                    "clip is transcribed in 1.1 s.")
        gr.HTML(latency_html(summary))

        gr.Markdown("### Every request and response\nPick a test, then click a row to see the exact JSON "
                    "sent and returned.")
        choices = [(f"{NAME[n]} ({len(rows[n])})", n) for n in ORDER if n in rows]
        with gr.Row():
            suite = gr.Dropdown(choices, value=choices[0][1], label="test", scale=2)
            search = gr.Textbox(label="search", placeholder="card, alarm, बुश ...", scale=2)
            misses = gr.Checkbox(label="misses only", value=False, scale=1)
        ids = gr.State([])
        table = gr.Dataframe(headers=["#", "Input", "Expected", "Model output", "Result", "Time"],
                             interactive=False, wrap=True, max_height=420)
        with gr.Row():
            meta = gr.Markdown()
        with gr.Row():
            req = gr.JSON(label="request")
            res = gr.JSON(label="response")

        def filt(name, q, only_miss):
            out, keep = [], []
            q = (q or "").strip().lower()
            for i, r in enumerate(rows.get(name, [])):
                cells = row_cells(name, i, r)
                if only_miss and not (r.get("correct") is False or r.get("word_errors", 0) > 0):
                    continue
                if q and q not in " ".join(str(c) for c in cells[1:4]).lower():
                    continue
                out.append(cells)
                keep.append(i)
            first = show(name, keep[0]) if keep else ("*no requests match*", None, None)
            return (out, keep, *first)

        def show(name, i):
            r = rows[name][i]
            endpoint = r["response"].get("object", "") if isinstance(r["response"], dict) else ""
            m = (f"**{NAME[name]} · request {i + 1}** · HTTP {r['status']} · **{ms(r['ms'])}**"
                 + (" (first request: includes loading the model)" if r.get("cold") else "")
                 + (f" · `{endpoint}`" if endpoint else ""))
            return m, r["request"], r["response"]

        def pick(name, keep, evt: gr.SelectData):
            row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            if row is None or row >= len(keep):
                return gr.update(), gr.update(), gr.update()
            return show(name, keep[row])

        for ev in (suite.change, misses.change, search.submit):
            ev(filt, [suite, search, misses], [table, ids, meta, req, res])
        table.select(pick, [suite, ids], [meta, req, res])
        return suite, search, misses, table, ids, meta, req, res, filt


def wire_initial(demo: gr.Blocks, handles):
    """Fill the request table when the page loads."""
    if not handles:
        return
    suite, search, misses, table, ids, meta, req, res, filt = handles
    demo.load(filt, [suite, search, misses], [table, ids, meta, req, res])
