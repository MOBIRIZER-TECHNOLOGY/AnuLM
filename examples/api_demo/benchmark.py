"""
AnuLM API benchmark: hundreds of real requests on held-out test data, each one
timed, every request and response saved.

    python api.py                                    # terminal 1, repo root
    python examples/api_demo/benchmark.py            # terminal 2, ~30 min on the GPU
    python examples/api_demo/benchmark.py --only asr_en,decide_banking --n-scale 0.2

Writes examples/api_demo/benchmark/<suite>.jsonl (one line per request: the
request, the response, the HTTP status, the client-side milliseconds, and the
score) and benchmark/summary.json (accuracy or WER and latency percentiles per
suite). Test data is read from the repository's data/ folder (--data).

Every item is held out from training: BANKING77 test, MASSIVE hi-IN validation
(the repo's test split), the first 570 LibriSpeech rows (the ASR trainer's
held-out cut), FLEURS hi dev, Flickr30k val, A-OKVQA val, chat_heldout.jsonl.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import re
import statistics
import time
import unicodedata
import wave
from pathlib import Path

import requests

HERE = Path(__file__).parent
OUT = HERE / "benchmark"


# ------------------------------------------------------------------ scoring
def norm_words(s: str) -> list[str]:
    """eval_speech.normalise, so these WERs match the repo's: keep letters, digits
    and the Devanagari marks (Mn, Mc), drop punctuation. A regex `\\w` would not
    do: it excludes matras and viramas, which splits Hindi words into fragments."""
    keep = [c for c in s.lower() if c.isalnum() or c.isspace() or unicodedata.category(c) in ("Mn", "Mc")]
    return "".join(keep).replace("।", " ").split()


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ------------------------------------------------------------------ client
class Client:
    def __init__(self, base: str, key: str | None):
        self.base, self.s = base.rstrip("/"), requests.Session()
        if key:
            self.s.headers["Authorization"] = f"Bearer {key}"

    def post(self, path: str, **kw):
        t0 = time.perf_counter()
        r = self.s.post(self.base + path, timeout=600, **kw)
        ms = (time.perf_counter() - t0) * 1000
        ctype = r.headers.get("content-type", "")
        body = r.json() if ctype.startswith("application/json") else {"bytes": len(r.content)}
        return r.status_code, body, round(ms, 1), r.content


def reply_text(body: dict) -> str:
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


# ------------------------------------------------------------------ suites
def suite_decide(c, data, n, model, rows, options=None, top=3):
    out = []
    for text, label in rows[:n]:
        req = {"model": model, "input": text, "top": top, "threshold": 0.7}
        if options:
            req["options"] = options
        st, body, ms, _ = c.post("/v1/decide", json=req)
        res = body.get("results", [{}])[0] if st == 200 else {}
        choice = res.get("choice")
        out.append({"request": {k: v for k, v in req.items() if k != "options"} | (
                        {"options": f"<{len(options)} labels>"} if options else {}),
                    "response": body if not options else {**body, "results": [res]},
                    "status": st, "ms": ms, "expected": label, "got": choice,
                    "confidence": res.get("confidence"), "action": res.get("action"),
                    "correct": choice == label})
    return out


def banking_rows(data, seed=0):
    rows = [(r["text"], r["category"]) for r in csv.DictReader(open(data / "banking77" / "test.csv", encoding="utf-8"))]
    random.Random(seed).shuffle(rows)
    return rows


def massive_hi_rows(data, seed=0):
    rows = [json.loads(l) for l in open(data / "decide_general" / "massive_hi-IN" / "validation.jsonl", encoding="utf-8")]
    rows = [(r["text"], r["label_text"]) for r in rows]
    random.Random(seed).shuffle(rows)
    return rows


def suite_answer(c, data, n):
    out = []
    for l in list(open(data / "rag_demo_questions.jsonl", encoding="utf-8"))[:n]:
        q = json.loads(l)
        st, body, ms, _ = c.post("/v1/answer", json={"question": q["q"]})
        got = (body.get("answer") or "") if st == 200 else ""
        ok = any(a.lower() in got.lower() for a in q["a"])           # rc_pointer.py's demo rule
        out.append({"request": {"question": q["q"]}, "response": body, "status": st, "ms": ms,
                    "expected": q["a"], "got": got, "correct": ok})
    return out


def wav_seconds(raw: bytes) -> float | None:
    try:
        with wave.open(io.BytesIO(raw)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return None


def suite_asr(c, data, n, lang):
    if lang == "en":
        rows = [json.loads(l) for l in open(data / "speech_ls100.jsonl", encoding="utf-8")][:570][:n]
    else:
        rows = [json.loads(l) for l in open(data / "fleurs_hi" / "dev.jsonl", encoding="utf-8")][:n]
    out = []
    for r in rows:
        raw = Path(r["audio"]).read_bytes()
        st, body, ms, _ = c.post("/v1/audio/transcriptions", files={"file": (Path(r["audio"]).name, raw)},
                                 data={"model": "anulm-asr", "language": lang, "response_format": "verbose_json"})
        hyp = body.get("text", "") if st == 200 else ""
        ref_w, hyp_w = norm_words(r["text"]), norm_words(hyp)
        errs = edit_distance(ref_w, hyp_w)
        dur = body.get("duration") or r.get("seconds")
        out.append({"request": {"file": Path(r["audio"]).name, "language": lang, "model": "anulm-asr"},
                    "response": body, "status": st, "ms": ms, "expected": r["text"], "got": hyp,
                    "word_errors": errs, "ref_words": len(ref_w), "audio_seconds": dur,
                    "rtf": round(ms / 1000 / dur, 3) if dur else None})
    return out


def data_url(path: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def suite_caption(c, data, n):
    rows = [json.loads(l) for l in open(data / "flickr30k_val.jsonl", encoding="utf-8")]
    refs: dict[str, list[str]] = {}
    for r in rows:
        refs.setdefault(r["image"], []).append(r["text"])
    images = list(refs)[::max(1, len(refs) // n)][:n]
    out = []
    for img in images:
        req = {"model": "anulm-vision", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url(img)}}]}]}
        st, body, ms, _ = c.post("/v1/chat/completions", json=req)
        req["messages"][0]["content"][0]["image_url"]["url"] = f"<{Path(img).name}, base64>"
        out.append({"request": req, "response": body, "status": st, "ms": ms,
                    "image": Path(img).name, "expected": refs[img], "got": reply_text(body)})
    return out


def suite_vqa(c, data, n):
    rows = [json.loads(l) for l in open(data / "aokvqa_val.jsonl", encoding="utf-8")][:n]
    out = []
    for r in rows:
        req = {"model": "anulm-vision", "messages": [{"role": "user", "content": [
            {"type": "text", "text": r["question"]},
            {"type": "image_url", "image_url": {"url": data_url(r["image"])}}]}]}
        st, body, ms, _ = c.post("/v1/chat/completions", json=req)
        got = reply_text(body).strip().lower().rstrip(".")
        req["messages"][0]["content"][1]["image_url"]["url"] = f"<{Path(r['image']).name}, base64>"
        out.append({"request": req, "response": body, "status": st, "ms": ms, "image": Path(r["image"]).name,
                    "question": r["question"], "choices": r["choices"], "expected": r["text"], "got": got,
                    "correct": got == r["text"].strip().lower()})
    return out


def suite_chat(c, data, n):
    rows = [json.loads(l) for l in open(data / "chat_heldout.jsonl", encoding="utf-8")]
    en = [r for r in rows if r.get("lang") == "en"][: n // 2]
    hi = [r for r in rows if r.get("lang") == "hi"][: n - n // 2]
    out = []
    for r in en + hi:
        req = {"model": "anulm-chat", "messages": [{"role": "user", "content": r["question"]}],
               "max_tokens": 120, "temperature": 0.7, "repetition_penalty": 1.3, "seed": 7}
        st, body, ms, _ = c.post("/v1/chat/completions", json=req)
        toks = (body.get("usage") or {}).get("completion_tokens") or 0
        out.append({"request": req, "response": body, "status": st, "ms": ms, "lang": r.get("lang"),
                    "expected": r["answer"], "got": reply_text(body), "completion_tokens": toks,
                    "tokens_per_s": round(toks / (ms / 1000), 1) if toks else None})
    return out


TTS_TEXT = [
    "My card was stolen this morning, please block it.", "What is the exchange rate for euros today?",
    "Your transfer has been sent and should arrive within two working days.",
    "Please confirm your date of birth to continue.", "The weather in Delhi will be hot and dry.",
    "Set an alarm for six thirty tomorrow morning.", "Your order has been shipped.",
    "I would like to speak to a customer service agent.", "The meeting has been moved to Friday afternoon.",
    "Thank you for calling, have a great day.", "Your account balance is four thousand two hundred rupees.",
    "The train to Mumbai leaves from platform three.", "Please hold while I check that for you.",
    "We have received your refund request.", "The new card will arrive in five to seven days.",
    "Remember to drink water and take a short walk.", "The library closes at eight in the evening.",
    "Your appointment is confirmed for ten o'clock.", "Press one for English, two for Hindi.",
    "Thank you, your payment was successful.",
]


def suite_tts(c, data, n):
    out = []
    for t in TTS_TEXT[:n]:
        req = {"model": "system-tts", "input": t, "response_format": "wav"}
        st, body, ms, raw = c.post("/v1/audio/speech", json=req)
        secs = wav_seconds(raw) if st == 200 else None
        out.append({"request": req, "response": body, "status": st, "ms": ms,
                    "audio_seconds": round(secs, 2) if secs else None,
                    "rtf": round(ms / 1000 / secs, 3) if secs else None})
    return out


def readable(label: str) -> str:
    """decide.py's readable(): the zero-shot model was trained on 'card arrival', not 'card_arrival'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label).replace("_", " ").lower()


def general_rows(data):
    labels = [readable(l) for l in json.load(open(data / "banking77" / "categories.json", encoding="utf-8"))]
    return [(t, readable(l)) for t, l in banking_rows(data, seed=1)], labels


SUITES = {
    # name: (default n, title, runner)
    "decide_banking": (300, "Decide: BANKING77 intents (AnuLM + LoRA)",
                       lambda c, d, n: suite_decide(c, d, n, "anulm-decide-banking", banking_rows(d))),
    "decide_banking_modernbert": (300, "Decide: BANKING77 intents (ModernBERT reference)",
                                  lambda c, d, n: suite_decide(c, d, n, "modernbert-decide-banking", banking_rows(d))),
    "decide_hindi": (300, "Decide: MASSIVE Hindi intents (AnuLM + LoRA)",
                     lambda c, d, n: suite_decide(c, d, n, "anulm-decide-hindi", massive_hi_rows(d))),
    "decide_general": (50, "Decide: zero-shot over all 77 BANKING77 labels",
                       lambda c, d, n: suite_decide(c, d, n, "anulm-decide-general", *general_rows(d))),
    "answer": (20, "Lookup: Wikipedia answers with sources", suite_answer),
    "asr_en": (100, "Speech to text: English (LibriSpeech held-out)", lambda c, d, n: suite_asr(c, d, n, "en")),
    "asr_hi": (100, "Speech to text: Hindi (FLEURS dev)", lambda c, d, n: suite_asr(c, d, n, "hi")),
    "caption": (50, "Vision: captions (Flickr30k val)", suite_caption),
    "vqa": (200, "Vision: open-ended questions (A-OKVQA val)", suite_vqa),
    "chat": (40, "Chat: held-out prompts, English and Hindi", suite_chat),
    "tts": (20, "Text to speech", suite_tts),
}


def summarize(name: str, title: str, rows: list[dict]) -> dict:
    ok = [r for r in rows if r["status"] == 200]
    # The first request of a suite may load its model (seconds); it is reported as
    # cold_ms and kept out of the latency statistics, which describe a warm server.
    ms = [r["ms"] for r in ok if not r.get("cold")] or [r["ms"] for r in ok]
    s = {"suite": name, "title": title, "n": len(rows), "ok": len(ok),
         "errors": len(rows) - len(ok), "cold_ms": rows[0]["ms"] if rows else None}
    if ms:
        s |= {"ms_mean": round(statistics.mean(ms), 1), "ms_p50": round(pct(ms, 0.5), 1),
              "ms_p95": round(pct(ms, 0.95), 1), "ms_min": round(min(ms), 1), "ms_max": round(max(ms), 1)}
    if ok and "correct" in ok[0]:
        s["accuracy"] = round(sum(r["correct"] for r in ok) / len(ok), 4)
    if name.startswith("decide"):
        auto = [r for r in ok if r["action"] == "auto-route"]
        s["auto_routed"] = len(auto)
        s["accuracy_when_auto_routed"] = round(sum(r["correct"] for r in auto) / len(auto), 4) if auto else None
    if name.startswith("asr"):
        s["wer"] = round(sum(r["word_errors"] for r in ok) / max(1, sum(r["ref_words"] for r in ok)), 4)
        s["audio_seconds"] = round(sum(r["audio_seconds"] or 0 for r in ok), 1)
    if "rtf" in (ok[0] if ok else {}):
        rt = [r["rtf"] for r in ok if r["rtf"]]
        s["rtf_mean"] = round(statistics.mean(rt), 3) if rt else None
    if name == "chat":
        tps = [r["tokens_per_s"] for r in ok if r["tokens_per_s"]]
        s["tokens_per_s_mean"] = round(statistics.mean(tps), 1) if tps else None
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--key")
    p.add_argument("--data", type=Path, default=HERE.parent.parent / "data")
    p.add_argument("--only", default="", help="comma-separated suite names")
    p.add_argument("--n-scale", type=float, default=1.0, help="multiply every suite's size")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    c = Client(a.base_url, a.key)
    health = c.s.get(c.base + "/health", timeout=5).json()
    only = [s for s in a.only.split(",") if s]

    sp = OUT / "summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() and only else {"suites": {}}
    summary |= {"recorded": time.strftime("%Y-%m-%d %H:%M"), "base_url": a.base_url, "device": health.get("device")}
    for name, (n, title, fn) in SUITES.items():
        if only and name not in only:
            continue
        n = max(1, round(n * a.n_scale))
        t0 = time.time()
        print(f"{name:28s} n={n:<4d}", end=" ", flush=True)
        rows = fn(c, a.data, n)
        if rows:
            rows[0]["cold"] = True
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = summarize(name, title, rows)
        s["wall_seconds"] = round(time.time() - t0, 1)
        summary["suites"][name] = s
        sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metric = (f"acc {s['accuracy']:.1%}" if "accuracy" in s else f"WER {s['wer']:.1%}" if "wer" in s else "")
        print(f"{metric:12s} p50 {s.get('ms_p50', 0):7.0f} ms  p95 {s.get('ms_p95', 0):7.0f} ms  "
              f"errors {s['errors']}  ({s['wall_seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
