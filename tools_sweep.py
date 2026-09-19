"""Run every released checkpoint over a thousand-plus real prompts and report
what actually comes back.

    python tools_sweep.py --device cuda                  # all three segments
    python tools_sweep.py --segment translate --limit 50 # one, quickly

`eval_code.py`, `eval_translate.py` and `eval_bench.py` each answer "how good
is it" on their own metric. This asks a different question: **what does the
serving path do over thousands of calls?** It drives `serve.Engine` -- the
same object `serve.py` and `app.py` use, with the same prompt templates,
stop rules and decoding -- and records every reply, so the failures that a
benchmark score hides show up:

  * replies that come back empty, which a pass@1 of 0 looks identical to;
  * replies that never stop, i.e. run to the token cap instead of EOS;
  * degeneration, measured as the share of repeated 4-grams;
  * crashes, encoding errors and anything else thrown along the way;
  * latency, so the demo's "a few tokens per second" is a measurement.

Prompts are real, not synthetic:

  code       MBPP (974: full test/train/validation/prompt) + HumanEval (164)
  translate  FLORES-200 devtest, 1,012 sentences in each direction
  continue   the held-out bench files mix_corpus.py sets aside, one prompt
             per document, across Hindi, English and Python

It writes `sweep/<segment>.jsonl` (every prompt and reply, so samples can be
picked later without re-running) and prints a summary per segment.

**Resumable, because it has to be.** Four thousand generations is hours, and
on Windows a process started from a terminal does not outlive the session
that started it -- `TASKS.md` has the story and the scheduled-task pattern
that does. So a rerun reads what is already in the jsonl, skips those
prompts and appends; `run_sweep.cmd` is the wrapper a task fires every 30
minutes, and a firing while the sweep is alive is a no-op.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "sweep"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


# ---------------------------------------------------------------- prompts

def code_prompts(limit: int | None) -> list[dict]:
    import pyarrow.parquet as pq
    rows = []
    for split in ("test", "train", "validation", "prompt"):
        f = HERE / "data" / "raw" / "google-research-datasets__mbpp" / "full" / \
            f"{split}-00000-of-00001.parquet"
        if f.exists():
            for r in pq.read_table(str(f)).to_pylist():
                rows.append({"id": f"mbpp/{split}/{r['task_id']}", "prompt": r["text"],
                             "mode": "question", "source": "mbpp"})
    he = HERE / "data" / "raw" / "openai__openai_humaneval" / "openai_humaneval" / \
        "test-00000-of-00001.parquet"
    if he.exists():
        for r in pq.read_table(str(he)).to_pylist():
            rows.append({"id": f"humaneval/{r['task_id']}", "prompt": r["prompt"],
                         "mode": "continue", "source": "humaneval"})
    return rows[:limit] if limit else rows


def translate_prompts(limit: int | None) -> list[dict]:
    f = HERE / "data" / "flores_devtest.jsonl"
    rows = []
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
        d = json.loads(line)
        rows.append({"id": f"flores/{d.get('lang', '?')}/{i}", "prompt": d["question"],
                     "mode": "translate", "source": "flores",
                     "reference": d.get("answer"), "lang": d.get("lang")})
    return rows[:limit] if limit else rows


def continue_prompts(limit: int | None) -> list[dict]:
    """One prompt per held-out document: its opening, which is what someone
    types into the box. Hindi and English get the title line, Python the
    signature."""
    from bpe import DOC_SEP
    rows, rng = [], random.Random(7)
    for name in ("hindi", "english", "python"):
        f = HERE / "data" / f"bench_{name}.txt"
        if not f.exists():
            continue
        docs = [d for d in f.read_text(encoding="utf-8").split(DOC_SEP) if d.strip()]
        rng.shuffle(docs)
        for i, doc in enumerate(docs):
            head = doc.strip().split("\n")[0][:120]
            if len(head) < 8:
                continue
            rows.append({"id": f"{name}/{i}", "prompt": head + "\n\n",
                         "mode": "continue", "source": name})
    rng.shuffle(rows)
    return rows[:limit] if limit else rows


SEGMENTS = {
    "code":      ("toonist/AnuLM-Coder-400M",   code_prompts,      "release/AnuLM-Coder-400M"),
    "translate": ("toonist/AnuLM-Translate-400M", translate_prompts, "release/AnuLM-Translate-400M"),
    "continue":  ("toonist/AnuLM-Base-400M",    continue_prompts,  "release/AnuLM-Base-400M"),
}


# ---------------------------------------------------------------- measures

def repeated_4grams(text: str) -> float:
    toks = text.split()
    if len(toks) < 8:
        return 0.0
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    return 1 - len(set(grams)) / len(grams)


def done_ids(path: Path) -> set:
    """Ids already answered, so a restart picks up where it stopped."""
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.add(json.loads(line)["id"])
        except (json.JSONDecodeError, KeyError):
            pass                      # a half-written last line after a kill
    return out


def read_stats(path: Path) -> dict:
    """Summarise a whole jsonl, however many runs wrote it."""
    st = {"n": 0, "empty": 0, "stopped": 0, "errors": 0, "tokens": 0, "seconds": 0.0,
          "degenerate": 0, "rep4": 0.0, "capped": 0, "wall": 0.0}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error"):
            st["errors"] += 1
            continue
        st["n"] += 1
        st["tokens"] += r.get("tokens", 0)
        st["seconds"] += r.get("seconds", 0.0)
        st["stopped"] += bool(r.get("stopped"))
        st["empty"] += not (r.get("completion") or "").strip()
        st["rep4"] += r.get("rep4", 0.0)
        st["degenerate"] += r.get("rep4", 0.0) > 0.5
    st["wall"] = st["seconds"]
    return st


def run_segment(name: str, args) -> dict:
    repo, builder, local = SEGMENTS[name]
    prompts = builder(args.limit)
    if not prompts:
        print(f"{name}: no prompts found -- see the docstring for what it needs")
        return {}
    src = local if (HERE / local).is_dir() else repo
    print(f"\n=== {name}: {len(prompts)} prompts through {src} ===")

    from serve import Engine
    if args.device.startswith("cuda"):
        import torch
        free, total = torch.cuda.mem_get_info()
        if free / total < 0.55:
            print(f"  WARNING: only {free/1e9:.1f} of {total/1e9:.1f} GB of VRAM is free. "
                  f"Windows does not raise OutOfMemoryError -- the driver spills to "
                  f"system RAM and this gets ~12x slower (docs/DEVELOPING.md, 'Traps'). "
                  f"Close the demo server and anything else holding a model first.")
    engine = Engine(str(HERE / local) if (HERE / local).is_dir() else repo, args.device)

    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}{args.tag}.jsonl"
    done = done_ids(path)
    if done:
        prompts = [p for p in prompts if p["id"] not in done]
        print(f"  resuming: {len(done)} already answered, {len(prompts)} to go")
        if not prompts:
            return read_stats(path)
    stats = {"n": 0, "empty": 0, "stopped": 0, "errors": 0, "tokens": 0, "seconds": 0.0,
             "degenerate": 0, "rep4": 0.0, "capped": 0}
    t0 = time.time()
    with path.open("a", encoding="utf-8") as fh:
        for i, item in enumerate(prompts):
            rec = dict(item)
            try:
                pen = 1.3 if (name == "continue" and args.penalty) else 1.0
                r = engine.generate(item["prompt"], args.max_tokens, args.temperature,
                                    1, 1337, mode=item["mode"], repetition_penalty=pen)
                rec.update(completion=r["completion"], tokens=r["tokens"],
                           seconds=r["seconds"], stopped=r["stopped_at_eos"],
                           mode_used=r["mode"])
                stats["n"] += 1
                stats["tokens"] += r["tokens"]
                stats["seconds"] += r["seconds"]
                stats["stopped"] += bool(r["stopped_at_eos"])
                stats["capped"] += r["tokens"] >= args.max_tokens
                if not r["completion"].strip():
                    stats["empty"] += 1
                rep = repeated_4grams(r["completion"])
                rec["rep4"] = round(rep, 4)
                stats["rep4"] += rep
                stats["degenerate"] += rep > 0.5
            except Exception as e:                       # never lose the sweep to one prompt
                stats["errors"] += 1
                rec.update(error=f"{type(e).__name__}: {e}",
                           traceback=traceback.format_exc()[-400:])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()      # a sweep you cannot watch is a sweep you cannot trust
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {i+1:5d}/{len(prompts)}  {el:6.0f}s  "
                      f"{stats['errors']} errors, {stats['empty']} empty", flush=True)
    stats["wall"] = time.time() - t0
    return read_stats(path)          # count the whole file, not just this run


def summarise(name: str, s: dict) -> None:
    if not s:
        return
    n = max(s["n"], 1)
    print(f"\n  {name}")
    print(f"    prompts            {s['n'] + s['errors']}   ({s['errors']} errors)")
    print(f"    empty replies      {s['empty']}  ({100*s['empty']/n:.1f}%)")
    print(f"    stopped at EOS     {s['stopped']}  ({100*s['stopped']/n:.1f}%)")
    print(f"    hit the token cap  {s['capped']}  ({100*s['capped']/n:.1f}%)")
    print(f"    repeated 4-grams   {s['rep4']/n:.1%} mean, "
          f"{s['degenerate']} replies over 50% ({100*s['degenerate']/n:.1f}%)")
    print(f"    throughput         {s['tokens']/max(s['seconds'],1e-9):.1f} tok/s, "
          f"{s['seconds']/n*1000:.0f} ms per reply, {s['wall']/60:.1f} min total")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment", choices=list(SEGMENTS) + ["all"], default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, help="prompts per segment; default is all of them")
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--tag", default="",
                   help="suffix for the output file, so a second run with different "
                        "decoding settings does not resume into the first one's results")
    p.add_argument("--penalty", action="store_true",
                   help="apply the 1.3 repetition penalty to the continue segment")
    args = p.parse_args()

    names = list(SEGMENTS) if args.segment == "all" else [args.segment]
    out = {}
    for name in names:
        out[name] = run_segment(name, args)
    print("\n" + "=" * 62)
    for name in names:
        summarise(name, out[name])
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nreplies in {OUT}/<segment>.jsonl, summary in {OUT}/summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
