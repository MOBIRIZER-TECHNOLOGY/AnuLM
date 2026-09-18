"""
Turn downloaded Hugging Face parquet / json files into this project's formats.

    python convert_parquet.py fineweb   --in data/raw/HuggingFaceFW__fineweb-edu/sample/10BT/*.parquet --out data/english_edu.txt --mb 4000
    python convert_parquet.py exercises --in data/raw/jinaai__code_exercises/data/*.parquet --out data/exercises.txt --jsonl data/sft_exercises.jsonl
    python convert_parquet.py pairs     --in <parquet or json> --q-col question --a-col answer --jsonl data/sft_x.jsonl

Text outputs use the corpus convention (title line, blank line, body, DOC_SEP
after, runs of 3+ newlines collapsed). Pair outputs are finetune.py's jsonl
(question, answer, title, lang). Parquet is streamed batch by batch, so a
2 GB shard never sits in memory as a table.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

from bpe import DOC_SEP

_BLANKS = re.compile(r"\n{3,}")
_FENCE = re.compile(r"```(?:python|py)?\n(.*?)```", re.DOTALL)


def clean(text: str) -> str:
    return _BLANKS.sub("\n\n", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def rows(paths):
    """Yield dict rows from parquet files (streamed) or json/jsonl files."""
    import pyarrow.parquet as pq
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            if path.endswith(".parquet"):
                pf = pq.ParquetFile(path)
                for batch in pf.iter_batches(batch_size=2048):
                    yield from batch.to_pylist()
            elif path.endswith(".jsonl"):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            else:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                yield from (data if isinstance(data, list) else data.get("data", []))


def write_docs(out, docs, mb_cap: float | None):
    """Write docs to `out`, returns (count, MB); stops at mb_cap MB."""
    cap = (mb_cap or 1e12) * 1024 * 1024
    n, total = 0, 0
    with Path(out).open("w", encoding="utf-8", newline="\n") as f:
        for title, body in docs:
            body = clean(body)
            if not body:
                continue
            doc = f"{title.strip()}\n\n{body}{DOC_SEP}"
            f.write(doc)
            total += len(doc.encode("utf-8"))
            n += 1
            if n % 5000 == 0:
                print(f"\r  {n} docs, {total/1e6:.0f} MB", end="", flush=True)
            if total >= cap:
                break
    print(f"\r  {n} docs, {total/1e6:.0f} MB -> {out}")
    return n, total


def write_pairs(path, pairs):
    n = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for it in pairs:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
    print(f"  {n} pairs -> {path}")
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["fineweb", "exercises", "pairs"])
    p.add_argument("--in", dest="inputs", nargs="+", required=True)
    p.add_argument("--out", help="text corpus output (fineweb, exercises)")
    p.add_argument("--jsonl", help="pair output (exercises, pairs)")
    p.add_argument("--mb", type=float, default=None, help="cap the text output at this many MB")
    p.add_argument("--q-col", default="question")
    p.add_argument("--a-col", default="answer")
    p.add_argument("--min-score", type=float, default=0.0, help="fineweb: keep rows with score >= this")
    p.add_argument("--max-answer-chars", type=int, default=2000)
    p.add_argument("--python-only", action="store_true",
                   help="pairs: keep only answers that contain a ```python block, and reduce the answer to it")
    p.add_argument("--score-col", default=None, help="pairs: numeric column to filter on (e.g. average_test_score)")
    p.add_argument("--min-pair-score", type=float, default=0.0)
    args = p.parse_args()

    if args.kind == "fineweb":
        def docs():
            for r in rows(args.inputs):
                if r.get("score", 99) < args.min_score:
                    continue
                text = r.get("text", "")
                if len(text) < 500:
                    continue
                yield r.get("url") or r.get("id") or "document", text
        write_docs(args.out, docs(), args.mb)

    elif args.kind == "exercises":
        # `problem` is the signature + docstring, `solution` the indented body
        # that continues it -- the same shape as a HumanEval prompt and its
        # completion. Joined with one newline they form the whole function.
        # A share of the set are fill-in-the-blank exercises whose "solution"
        # still has the blank ("# Complete the missing code ..."); those would
        # teach the model to leave gaps, so they are dropped.
        blank = re.compile(r"complete the (missing|code)|your code here|todo|\.\.\.\s*$|^\s*pass\s*$", re.I | re.M)
        items = []
        for r in rows(args.inputs):
            q, a = (r.get("problem") or "").rstrip(), (r.get("solution") or "").strip("\n").rstrip()
            if not q or not a or len(a) > args.max_answer_chars or not q.lstrip().startswith("def "):
                continue
            if blank.search(a) or "return" not in a:
                continue
            name = q.lstrip()[4:].split("(")[0].strip()
            items.append({"title": name, "question": q, "answer": a, "lang": "py"})
        if args.out:
            write_docs(args.out, (("# exercise: " + it["title"], it["question"] + "\n" + it["answer"]) for it in items), args.mb)
        if args.jsonl:
            write_pairs(args.jsonl, items)

    else:  # pairs
        def pairs():
            for r in rows(args.inputs):
                if args.score_col:
                    try:
                        if float(r.get(args.score_col) or 0) < args.min_pair_score:
                            continue
                    except (TypeError, ValueError):
                        continue
                q, a = str(r.get(args.q_col) or "").strip(), str(r.get(args.a_col) or "").strip()
                if args.python_only:
                    m = _FENCE.search(a)
                    if not m:
                        continue
                    a = m.group(1).strip()
                if not q or not a or len(a) > args.max_answer_chars or len(q) > args.max_answer_chars:
                    continue
                yield {"title": q.split("\n")[0][:60], "question": q, "answer": a, "lang": "py"}
        write_pairs(args.jsonl, pairs())


if __name__ == "__main__":
    main()
