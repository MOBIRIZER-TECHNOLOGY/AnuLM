"""Turn a sweep into the samples the demo page shows.

    python tools_sweep.py --device cuda        # writes sweep/<segment>.jsonl
    python tools_samples.py                    # writes docs/samples.json + docs/SAMPLES.md

The demo page used to show nothing until you had downloaded 0.8 GB and
pressed Generate. These are real replies from the sweep, so a visitor sees
what the models actually produce before deciding whether to wait.

**Selected, and the page says so.** Picking only the good ones and calling it
a demo is the oldest trick there is, so two things guard against it: the
aggregate numbers from the whole sweep travel with the samples and are shown
beside them, and the selection rule is written here rather than done by eye.
A reply is eligible when it stopped on its own, is long enough to read, and
is not mostly repetition; among those, the ones nearest the segment's median
repetition are taken, so the page shows typical output rather than the best
in the file.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
SWEEP = HERE / "sweep"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

TITLE = {"code": "Writing Python", "translate": "Translating",
         "continue": "Continuing text"}
BLURB = {
    "code": "Asked in prose, from MBPP and HumanEval. The model writes the whole function.",
    "translate": "FLORES-200 devtest, both directions. The direction is picked from the script.",
    "continue": "The opening line of a held-out document, in Hindi, English or Python.",
}


def load(segment: str) -> list[dict]:
    f = SWEEP / f"{segment}.jsonl"
    if not f.exists():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass                      # a half-written last line while a sweep runs
    return out


def eligible(r: dict, segment: str) -> bool:
    if r.get("error") or not (r.get("completion") or "").strip():
        return False
    if r.get("rep4", 1.0) > 0.35:
        return False
    body = r["completion"].strip()
    if len(body) < (40 if segment != "translate" else 8):
        return False
    if segment != "continue" and not r.get("stopped"):
        return False                  # ran to the cap: the reader sees it cut off
    return True


def pick(rows: list[dict], segment: str, n: int) -> list[dict]:
    """Typical rather than best: nearest the median repetition, spread across
    sources so one dataset cannot fill the list."""
    ok = [r for r in rows if eligible(r, segment)]
    if not ok:
        return []
    med = statistics.median(r.get("rep4", 0.0) for r in ok)
    ok.sort(key=lambda r: abs(r.get("rep4", 0.0) - med))
    out, seen = [], {}
    for r in ok:
        src = r.get("source", "?")
        if seen.get(src, 0) >= max(1, n // max(len({x.get("source") for x in ok}), 1)):
            continue
        seen[src] = seen.get(src, 0) + 1
        out.append(r)
        if len(out) >= n:
            break
    for r in ok:                       # top up if one source was thin
        if len(out) >= n:
            break
        if r not in out:
            out.append(r)
    return out[:n]


def stats_for(rows: list[dict]) -> dict:
    done = [r for r in rows if not r.get("error")]
    n = max(len(done), 1)
    return {
        "prompts": len(rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "empty": sum(1 for r in done if not (r.get("completion") or "").strip()),
        "stopped_pct": round(100 * sum(1 for r in done if r.get("stopped")) / n, 1),
        "rep4_mean_pct": round(100 * sum(r.get("rep4", 0.0) for r in done) / n, 1),
        "ms_mean": round(1000 * sum(r.get("seconds", 0.0) for r in done) / n),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-segment", type=int, default=8)
    p.add_argument("--max-chars", type=int, default=700)
    args = p.parse_args()

    payload, md = {}, ["# Sample outputs", "",
                       "Real replies from `tools_sweep.py`, which runs every released "
                       "checkpoint over thousands of held-out prompts. Selected by the "
                       "rule in `tools_samples.py` -- nearest the median repetition among "
                       "replies that stopped on their own -- so these are typical, not "
                       "the best in the file. The aggregate numbers for the whole sweep "
                       "are in each section and in `docs/RESULTS.md`.", ""]
    for segment in ("code", "translate", "continue"):
        rows = load(segment)
        if not rows:
            print(f"{segment}: no sweep file, skipped")
            continue
        chosen = pick(rows, segment, args.per_segment)
        st = stats_for(rows)
        payload[segment] = {
            "title": TITLE[segment], "blurb": BLURB[segment], "stats": st,
            "samples": [{"prompt": r["prompt"][:args.max_chars],
                         "completion": r["completion"][:args.max_chars],
                         "source": r.get("source", "?"),
                         "tokens": r.get("tokens"), "seconds": r.get("seconds")}
                        for r in chosen],
        }
        print(f"{segment}: {len(rows)} replies, {len(chosen)} samples chosen")
        md += [f"## {TITLE[segment]}", "", BLURB[segment], "",
               f"*{st['prompts']} prompts, {st['errors']} errors, {st['empty']} empty, "
               f"{st['stopped_pct']}% stopped on their own, {st['rep4_mean_pct']}% "
               f"repeated 4-grams on average, {st['ms_mean']} ms per reply.*", ""]
        for r in chosen:
            md += [f"**{r.get('source','?')}** — prompt:", "",
                   "```", r["prompt"].strip()[:args.max_chars], "```", "", "reply:", "",
                   "```", r["completion"].strip()[:args.max_chars], "```", ""]

    (HERE / "docs" / "samples.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "docs" / "SAMPLES.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print("wrote docs/samples.json and docs/SAMPLES.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
