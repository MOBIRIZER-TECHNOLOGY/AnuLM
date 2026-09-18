"""
Build a Python question-answer set from the code corpus.

    python make_qa_py.py --data data/python.txt --out data/qa_python.jsonl

Every documented top-level function becomes one pair, of two kinds in equal
number:

  write     Question: Write a Python function `add`: return the sum of two numbers.
            Answer:   the function's source, docstring included
  explain   Question: What does this Python function do?
                      <the function's source, docstring removed>
            Answer:   the docstring's first sentence

Files are parsed with `ast` (Python 2 files fail to parse and are skipped),
functions are de-duplicated by name and description, and only functions of
80-600 characters are used so a pair fits a 512-token row with room to spare.
The split is held out by file, so a held-out function's file never
contributed a training pair.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import warnings
from pathlib import Path

from bpe import DOC_SEP

warnings.filterwarnings("ignore", category=SyntaxWarning)
_FIRST_SENTENCE = re.compile(r"^(.+?[.!?])(?:\s|$)", re.DOTALL)
PROMPT_WRITE = "Write a Python function `{name}`: {desc}"
PROMPT_EXPLAIN = "What does this Python function do?\n{code}"


def first_sentence(doc: str) -> str | None:
    doc = " ".join(doc.strip().split())
    m = _FIRST_SENTENCE.match(doc)
    s = (m.group(1) if m else doc).strip()
    if not (15 <= len(s) <= 200) or s.startswith((":", "@", ">>>")):
        return None
    return s


def functions(body: str):
    """(node, source, source without its docstring) for each documented top-level function."""
    try:
        tree = ast.parse(body)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return
    for n in tree.body:
        if not isinstance(n, ast.FunctionDef) or n.name.startswith("__"):
            continue
        doc = ast.get_docstring(n)
        if not doc or len(n.body) < 2:
            continue
        src = ast.get_source_segment(body, n)
        if not src or not (80 <= len(src) <= 600):
            continue
        lines = src.split("\n")
        d = n.body[0]                                   # the docstring statement
        del lines[d.lineno - n.lineno: d.end_lineno - n.lineno + 1]
        yield n, src, "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/python.txt")
    p.add_argument("--out", default="data/qa_python.jsonl")
    p.add_argument("--heldout", default="data/qa_python_heldout.jsonl")
    p.add_argument("--heldout-frac", type=float, default=0.02)
    p.add_argument("--max", type=int, default=20000, help="cap on pairs, after de-duplication")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = random.Random(args.seed)

    docs = [d for d in Path(args.data).read_text(encoding="utf-8").split(DOC_SEP) if d.strip()]
    rng.shuffle(docs)
    n_held_files = int(len(docs) * args.heldout_frac)
    seen: set[tuple[str, str]] = set()
    train, held = [], []
    kind = 0
    for i, d in enumerate(docs):
        body = d.split("\n", 2)[2] if d.count("\n") >= 2 else ""
        for n, src, stripped in functions(body):
            desc = first_sentence(ast.get_docstring(n))
            if desc is None or (n.name, desc) in seen:
                continue
            seen.add((n.name, desc))
            if kind % 2 == 0:
                d0 = desc[0].lower() + desc[1:] if desc[:2].isupper() is False else desc
                item = {"title": n.name, "kind": "write",
                        "question": PROMPT_WRITE.format(name=n.name, desc=d0), "answer": src}
            else:
                item = {"title": n.name, "kind": "explain",
                        "question": PROMPT_EXPLAIN.format(code=stripped), "answer": desc}
            item["lang"] = "py"
            kind += 1
            (held if i < n_held_files else train).append(item)
        if len(train) + len(held) >= args.max:
            break

    rng.shuffle(train)
    for path, rows in ((args.out, train), (args.heldout, held)):
        with Path(path).open("w", encoding="utf-8", newline="\n") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_write = sum(1 for it in train if it["kind"] == "write")
    print(f"{i + 1} files scanned -> {len(train)} train pairs ({n_write} write, {len(train) - n_write} explain), "
          f"{len(held)} held out from {n_held_files} files")
    for it in train[:2]:
        print(f"\n  Question: {it['question'][:300]}\n  Answer: {it['answer'][:300]}")


if __name__ == "__main__":
    main()
