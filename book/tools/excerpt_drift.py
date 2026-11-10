"""
Which of the book's pinned excerpts still match the code at another commit?

    python book/tools/excerpt_drift.py                       # against HEAD
    python book/tools/excerpt_drift.py --ref origin/main
    python book/tools/excerpt_drift.py --ref 2e80b3e --strict   # exit 1 unless every excerpt is "same"
    python book/tools/excerpt_drift.py --ref origin/main --write  # also technical-review/excerpt-drift.md

Every excerpt box in the chapters names a file, a symbol, a line range and the commit the
lines were copied from (the pin). This tool reads the same file at `--ref` and classes the
excerpt:

- same:    every quoted line is still inside the cited range;
- moved:   the quoted lines are all in the file, together, at other line numbers (reported);
- changed: at least one quoted line is no longer in the file;
- absent:  the file does not exist at that ref.

It changes nothing. A reader working from a newer checkout runs it to learn which pages to
read at the pin instead; the book's own re-pinning is `repin.py`, which uses the diff between
two pins and rewrites the chapters. Blank lines and `# ...` ellipses in a quote are ignored,
as in `verify_chapters.py`; comparison is line by line, whitespace stripped.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
ROOT = BOOK.parent
CHAPTERS = BOOK / "chapters"
REPORT = BOOK / "technical-review" / "excerpt-drift.md"
EXCERPT = re.compile(r"\*\*In the repo\*\* · `([\w./]+)` · `([\w.]+)` · lines? (\d+)(?:–(\d+))? · @(\w+)\n\n```python\n(.*?)```", re.S)

_cache: dict[str, list[str] | None] = {}


def at(ref: str, path: str) -> list[str] | None:
    """The file's lines at `ref`, stripped; None when it is not there."""
    if path not in _cache:
        r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        _cache[path] = [l.strip() for l in r.stdout.splitlines()] if r.returncode == 0 else None
    return _cache[path]


def classify(ref: str, path: str, a: int, b: int, code: str) -> tuple[str, str]:
    """(status, note) for one excerpt."""
    lines = [l.strip() for l in code.splitlines() if l.strip() and l.strip() != "# ..."]
    file = at(ref, path)
    if file is None:
        return "absent", f"`{path}` is not at {ref}"
    window = file[a - 1:b]
    if all(l in window for l in lines):
        return "same", ""
    missing = [l for l in lines if l not in file]
    if missing:
        return "changed", f"`{missing[0][:60]}`" + (f" and {len(missing) - 1} more" if len(missing) > 1 else "")
    starts = [i for i, l in enumerate(file) if l == lines[0]]
    for i in starts:                                  # the quote, together, somewhere else
        span = file[i:i + 2 * max(b - a + 1, len(lines))]
        if all(l in span for l in lines):
            last = i + max(span.index(l) for l in lines)
            return "moved", f"now lines {i + 1}–{last + 1}"
    return "changed", "the quoted lines are in the file but no longer together"


def scan(ref: str) -> tuple[list[tuple], dict]:
    rows, pins = [], Counter()
    files = sorted(CHAPTERS.glob("[0-9][0-9]-*.md")) + sorted(CHAPTERS.glob("A[0-9]*-*.md"),
                                                                 key=lambda p: int(p.name[1:p.name.index("-")]))
    for ch in files:
        text = re.sub(r"(?m)^[ \t]*> ?", "", ch.read_text(encoding="utf-8"))
        for m in EXCERPT.finditer(text):
            path, sym, a, b, pin, code = m.groups()
            a, b = int(a), int(b or a)
            pins[pin] += 1
            status, note = classify(ref, path, a, b, code)
            rows.append((ch.name.split("-")[0].lstrip("0"), path, sym, f"{a}–{b}" if b != a else str(a), status, note))
    return rows, pins


def report(ref: str, rows: list[tuple], pins: Counter) -> str:
    n = Counter(r[4] for r in rows)
    pin = ", ".join(f"@{p} ({c})" for p, c in pins.most_common())
    L = [f"# Excerpt drift against `{ref}`", "",
         f"{len(rows)} excerpts pinned to {pin}: {n['same']} same, {n['moved']} moved, "
         f"{n['changed']} changed, {n['absent']} absent. Written by `book/tools/excerpt_drift.py`.", ""]
    bad = [r for r in rows if r[4] != "same"]
    if bad:
        L += ["| Ch | File | Symbol | Pinned lines | Status | Note |", "|---|---|---|---|---|---|"]
        L += [f"| {c} | `{p}` | `{s}` | {ln} | {st} | {note} |" for c, p, s, ln, st, note in bad]
    else:
        L.append("Every excerpt reads the same at this ref as at its pin.")
    return "\n".join(L) + "\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="HEAD", help="commit, branch or tag to compare with (default HEAD)")
    ap.add_argument("--strict", action="store_true", help="exit 1 unless every excerpt is 'same'")
    ap.add_argument("--write", action="store_true", help=f"also write {REPORT.relative_to(BOOK)}")
    a = ap.parse_args()
    rows, pins = scan(a.ref)
    n = Counter(r[4] for r in rows)
    print(f"{len(rows)} excerpts against {a.ref}: {n['same']} same, {n['moved']} moved, "
          f"{n['changed']} changed, {n['absent']} absent")
    per = {}
    for c, *_, st, _n in rows:
        per.setdefault(c, Counter())[st] += 1
    for c, cnt in per.items():
        if cnt["same"] != sum(cnt.values()):
            print(f"  ch {c:>3}: " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
    if a.write:
        REPORT.write_text(report(a.ref, rows, pins), encoding="utf-8", newline="\n")
        print(f"wrote {REPORT.relative_to(BOOK)}")
    return 1 if a.strict and n["same"] != len(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
