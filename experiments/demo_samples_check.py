"""
Run every sample in examples/samples/ through a running Omni demo and score it.

    python -m anulm.apps.app_omni                                  # in one terminal (port 7863)
    python experiments/demo_samples_check.py                       # all 117 samples
    python experiments/demo_samples_check.py --url http://127.0.0.1:7864 --limit 3 --label "CPU smoke"

It starts nothing itself: it calls the page's own API (/do_listen, /do_look,
/do_chat, /do_decide, the handlers behind the buttons and the sample
galleries), so it checks what a visitor gets, through Gradio, file upload and
all. For each sample it records the input, the output, the reference, the
time and a deliberately simple score:

    speech      word error rate (the release gate's rule, anulm/eval/eval_speech.py)
    captions    non-empty (a caption has no single right answer; read them)
    image Q     exact match with an annotator's answer, or one contained in the reply
    lookup      an expected answer contained in the reply
    intents     the chosen intent equals the label
    chat        non-empty, in the asked language (Devanagari for Hindi), not the question
                echoed, no character repeated ten times running

These are smoke-level scores on ~10-25 samples a kind, not the benchmark:
logs/omni_one/results.json has the held-out numbers on hundreds to thousands
of rows. Writes examples/samples/results.json and results.md. Exits 1 if any
call raised a server error (a wrong answer is not an error), 2 if the page
cannot be reached.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLES = ROOT / "examples" / "samples"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

KIND_NAMES = {"asr_en": "English speech", "asr_hi": "Hindi speech", "caption": "captions",
              "vqa": "image questions", "lookup": "Wikipedia lookup", "chat": "chat",
              "intent_banking": "banking intents (EN)", "intent_massive_hi": "assistant intents (HI)"}


def norm(s: str) -> str:
    from anulm.eval.eval_speech import normalise
    return normalise(s or "")


def contains(reply: str, answers: list[str]) -> bool:
    """Whole-word containment after normalising, so "4" does not match "14"."""
    r = f" {norm(reply)} "
    return any(a and f" {a} " in r for a in (norm(x) for x in answers))


def script_share(text: str, lang: str) -> float:
    """Share of letters in the language's script: Devanagari for hi, Latin otherwise."""
    letters = [c for c in text if c.isalpha() or unicodedata.category(c) in ("Mn", "Mc")]
    if not letters:
        return 0.0
    if lang == "hi":
        hits = sum("\u0900" <= c <= "\u097f" for c in letters)
    else:
        hits = sum(c.isascii() for c in letters)
    return hits / len(letters)


def call(client, item: dict):
    """(output text, extra) from the page's own handler for this sample."""
    from gradio_client import handle_file

    from anulm.apps.app_omni import DECIDE_OWN, SAMPLE_TASKS
    inp = item["inputs"]
    if item["tab"] == "listen":           # transcript only: the clips are read sentences, not questions
        heard, _said, _voice, meta = client.predict(handle_file(str(SAMPLES / inp["audio"])), inp["lang"], False,
                                                    api_name="/do_listen")
        return heard, {"meta": meta}
    if item["tab"] == "look":
        _q, seen, _voice, meta = client.predict(handle_file(str(SAMPLES / inp["image"])), inp["question"], None,
                                                api_name="/do_look")
        return seen, {"meta": meta}
    if item["tab"] == "chat":
        reply, meta = client.predict(inp["question"], inp["mode"], api_name="/do_chat")
        return reply, {"meta": meta}
    if item["tab"] == "decide":
        _label, verdict, typed = client.predict(inp["message"], 0.7, DECIDE_OWN, SAMPLE_TASKS[inp["task"]],
                                                api_name="/do_decide")
        typed = typed or {}
        return typed.get("choice", ""), {"meta": verdict, "confidence": typed.get("confidence"),
                                         "action": typed.get("action")}
    raise ValueError(f"unknown tab {item['tab']}")


def score(item: dict, out: str) -> dict:
    kind, ref = item["kind"], item.get("reference")
    if kind.startswith("asr"):
        from anulm.eval.eval_speech import wer
        edits, words = wer(ref, out or "")
        return {"edits": edits, "words": words, "wer": round(100 * edits / max(words, 1), 1),
                "pass": edits == 0}
    if kind == "caption":
        return {"pass": bool((out or "").strip()) and out != "(nothing)"}
    if kind == "vqa":
        answers = item.get("answers") or [ref]
        exact = norm(out) in {norm(a) for a in answers}
        return {"exact": exact, "contains": exact or contains(out, answers), "pass": exact or contains(out, answers)}
    if kind == "lookup":
        return {"pass": contains(out, item.get("answers") or [ref]),
                "from_wikipedia": "(from Wikipedia" in (out or "")}
    if kind.startswith("intent"):
        return {"pass": out == ref}
    if kind == "chat":
        share = script_share(out or "", item["language"])
        # Repeating the question back is non-empty and in the right script, and still no answer.
        echo = norm(out) == norm(item["inputs"]["question"])
        stuck = bool(re.search(r"(.)\1{9,}", out or ""))         # "----------": the decoder stuck on one token
        return {"nonempty": bool((out or "").strip()), "script_share": round(share, 2), "echo": echo,
                "stuck": stuck, "pass": bool((out or "").strip()) and share >= 0.6 and not echo and not stuck}
    return {"pass": None}


def summarise(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    out = {}
    for kind, rs in by.items():
        ok = [r for r in rs if not r.get("error")]
        s = {"n": len(rs), "errors": len(rs) - len(ok),
             "median_seconds": round(statistics.median(r["seconds"] for r in rs), 2) if rs else None,
             "passed": sum(bool(r["score"].get("pass")) for r in ok)}
        if kind.startswith("asr") and ok:
            s["wer"] = round(100 * sum(r["score"]["edits"] for r in ok) / max(1, sum(r["score"]["words"] for r in ok)), 1)
        elif ok:
            s["accuracy"] = round(100 * s["passed"] / len(ok), 1)
        out[kind] = s
    return out


def markdown(res: dict) -> str:
    lines = [f"# Demo sample check: {res['label']}", "",
             f"{res['when']} against {res['url']}; {res['n']} samples, {res['errors']} server errors, "
             f"{res['seconds']:.0f} s in all. Written by `experiments/demo_samples_check.py`.", "",
             "Smoke-level scores on a few samples per kind; the held-out benchmark numbers are in "
             "`logs/omni_one/results.json`.", "",
             "| kind | samples | errors | score | median time |", "|---|---|---|---|---|"]
    for kind, s in res["summary"].items():
        what = {"caption": "non-empty", "chat": "non-empty, right script, no echo",
                "vqa": "match an annotator", "lookup": "contain the answer"}.get(kind, "correct")
        if "wer" in s:
            val = f"WER {s['wer']}%"
        elif "accuracy" in s:
            val = f"{s['passed']}/{s['n'] - s['errors']} {what}"
        else:
            val = "–"
        lines.append(f"| {KIND_NAMES.get(kind, kind)} | {s['n']} | {s['errors']} | {val} | "
                     f"{s['median_seconds']} s |")
    lines += ["", "## Every sample", "", "| id | output | reference | score | s |", "|---|---|---|---|---|"]

    def cell(x, n=120):
        x = "" if x is None else str(x).replace("|", "\\|").replace("\n", " ")
        return x if len(x) <= n else x[:n - 1] + "…"
    for r in res["items"]:
        sc = r.get("error") or ("WER " + str(r["score"]["wer"]) + "%" if "wer" in r["score"]
                                else ("pass" if r["score"].get("pass") else "miss"))
        lines.append(f"| {r['id']} | {cell(r['output'])} | {cell(r['reference'])} | {cell(sc, 60)} | "
                     f"{r['seconds']:.1f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:7863")
    p.add_argument("--manifest", default=str(SAMPLES / "samples.json"))
    p.add_argument("--limit", type=int, default=0, help="at most this many samples of each kind (0: all)")
    p.add_argument("--kinds", default="", help="comma-separated kinds to run, e.g. asr_en,vqa (default: all)")
    p.add_argument("--label", default="full run", help="what this run is, e.g. 'CPU smoke check'")
    p.add_argument("--out", default=str(SAMPLES / "results.json"))
    args = p.parse_args()

    items = json.loads(Path(args.manifest).read_text(encoding="utf-8"))["items"]
    if args.kinds:
        items = [i for i in items if i["kind"] in args.kinds.split(",")]
    if args.limit:
        by_kind = defaultdict(list)
        for i in items:
            by_kind[i["kind"]].append(i)
        items = [i for rs in by_kind.values() for i in rs[:args.limit]]

    from gradio_client import Client
    try:
        client = Client(args.url, verbose=False)
    except Exception as e:                      # noqa: BLE001 -- any failure here means no page to test
        print(f"cannot reach the demo at {args.url}: {e}", file=sys.stderr)
        return 2

    rows, t_all = [], time.time()
    for n, item in enumerate(items, 1):
        t0 = time.time()
        row = {"id": item["id"], "tab": item["tab"], "kind": item["kind"], "language": item["language"],
               "input": item["inputs"], "reference": item.get("reference")}
        try:
            out, extra = call(client, item)
            row.update(output=out, **extra, score=score(item, out))
        except Exception as e:                  # noqa: BLE001 -- recorded, and the exit code says so
            row.update(output=None, error=f"{type(e).__name__}: {e}"[:500], score={})
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        mark = row.get("error") or json.dumps(row["score"], ensure_ascii=False)
        print(f"[{n}/{len(items)}] {item['id']} {row['seconds']:.1f}s  {str(row['output'])[:70]!r}  {mark}",
              flush=True)

    res = {"label": args.label, "url": args.url, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
           "manifest": str(Path(args.manifest).resolve().relative_to(ROOT)) if Path(args.manifest).resolve()
           .is_relative_to(ROOT) else args.manifest, "limit_per_kind": args.limit or None, "n": len(rows),
           "errors": sum(bool(r.get("error")) for r in rows), "seconds": round(time.time() - t_all, 1),
           "summary": summarise(rows), "items": rows}
    out = Path(args.out)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(res), encoding="utf-8")
    print(f"\n{res['n']} samples, {res['errors']} errors, {res['seconds']:.0f}s -> {out}")
    for kind, s in res["summary"].items():
        print(f"  {kind:18} {s}")
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
