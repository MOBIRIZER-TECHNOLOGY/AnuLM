"""
Check every chapter in book/chapters/ against the repository at the pinned commit.

    python book/tools/verify_chapters.py      # writes technical-review/chapter-verification.md

For each chapter:

- every **In the repo** excerpt: each quoted line (other than `# ...`) must occur
  inside the cited line range of the cited file at the pin;
- every "`name` (line N)" or "`name` (`file` line N)" reference to a function or
  test must name a definition that starts on line N (the `def` line, as the
  source map cites it);
- the page furniture of STYLE.md: front matter, numbered sections, six key
  points, an "In one breath" box, a references section that lists every
  [Rnn]/[Snn] the text cites, and at least one badge.

It never edits a chapter.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
ROOT = BOOK.parent
CHAPTERS = BOOK / "chapters"
OUT = BOOK / "technical-review" / "chapter-verification.md"
PIN = re.search(r"`@([0-9a-f]{7,40})`", (BOOK / "REPOSITORY_SOURCE_MAP.md").read_text(encoding="utf-8")).group(1)
BADGES = ("ANALOGY", "RULE OF THUMB", "IN THE REPO", "NOT IN THE REPO", "EXTENSION", "IMPLEMENTATION-DEPENDENT", "MEASURED")
EXCERPT = re.compile(r"\*\*In the repo\*\* · `([\w./]+)` · `([\w.]+)` · lines? (\d+)(?:–(\d+))? · @(\w+)\n\n```python\n(.*?)```", re.S)
LINE_REF = re.compile(r"`(\w+)` \((?:`([\w./]+)` )?line (\d+)\)")
_cache: dict[str, list[str]] = {}


def at(path: str) -> list[str]:
    if path not in _cache:
        r = subprocess.run(["git", "show", f"{PIN}:{path}"], cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        _cache[path] = r.stdout.splitlines() if r.returncode == 0 else []
    return _cache[path]


def def_lines(path: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    src = "\n".join(at(path))
    if not src:
        return out
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.setdefault(node.name, []).append(node.lineno)
    return out


def check(path: Path) -> tuple[list[str], list[str], dict]:
    text = path.read_text(encoding="utf-8")
    problems, notes = [], []
    stats = {"excerpts": 0, "excerpt_lines": 0, "line_refs": 0}

    for m in EXCERPT.finditer(text):
        f, sym, a, b, pin, code = m.groups()
        a, b = int(a), int(b or a)
        stats["excerpts"] += 1
        if pin != PIN:
            problems.append(f"excerpt `{sym}` is pinned to @{pin}, the map to @{PIN}")
        window = [l.strip() for l in at(f)[a - 1:b]]
        if not window:
            problems.append(f"excerpt `{sym}`: `{f}` lines {a}–{b} not found at @{PIN}")
            continue
        for line in code.splitlines():
            s = line.strip()
            if not s or s == "# ...":
                continue
            stats["excerpt_lines"] += 1
            if s not in window:
                problems.append(f"excerpt `{sym}` ({f} {a}–{b}): line not in range: `{s[:70]}`")

    defs = {f: def_lines(f) for f in ("model.py", "test_model.py", "train.py")}
    for m in LINE_REF.finditer(text):
        name, f, ln = m.group(1), m.group(2), int(m.group(3))
        files = [f] if f else [k for k in defs if name in defs[k]]
        if f and f not in defs:
            defs[f] = def_lines(f)
        hits = [x for k in files for x in defs.get(k, {}).get(name, [])]
        if not hits:
            continue                                   # not a function name (e.g. a variable); the excerpt check covers code
        stats["line_refs"] += 1
        if ln not in hits:
            problems.append(f"`{name}` cited at line {ln}, defined at {hits}")

    if not text.startswith("---\n") or "\ntitle:" not in text.split("\n---", 2)[0] + "\n":
        problems.append("no front matter with a title")
    appendix = path.name[:1] == "A"
    if not appendix and not re.search(r"^## \d+\.\d+ ", text, re.M):
        problems.append("no numbered sections (## N.M)")
    kp = re.search(r"^## Key points\n(.*?)(?=^>|^## )", text, re.M | re.S)
    n_kp = len(re.findall(r"^- ", kp.group(1), re.M)) if kp else 0
    if n_kp != 6 and not appendix:
        problems.append(f"key points: {n_kp} bullets, STYLE.md asks for six")
    if "**In one breath.**" not in text and not appendix:
        problems.append("no In one breath box")
    refs = text[text.find("## References"):] if "## References" in text else ""
    body = text[: text.find("## References")] if refs else text
    cited = sorted(set(re.findall(r"\[([RS]\d{2})\]", body)))
    listed = set(re.findall(r"\[([RS]\d{2})\]", refs))
    if not refs and not appendix:
        problems.append("no References section")
    missing = [c for c in cited if c not in listed]
    if missing and not appendix:                # an appendix may be the reference list itself
        problems.append(f"cited but not listed in References: {', '.join(missing)}")
    for src in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if not (path.parent / src).resolve().is_file():
            problems.append(f"figure not found: {src}")
    # the code standard (BOOK_PLAN.md, "Page budget and code share")
    main = text[: text.find("## Key points")] if "## Key points" in text else body
    code_lines = sum(len(b.strip("\n").splitlines()) for b in re.findall(r"```python\n(.*?)```", main, re.S))
    other = re.sub(r"```.*?```", "", main, flags=re.S)
    prose_lines = sum(1 for l in other.splitlines() if l.strip() and not l.lstrip().startswith("|"))
    share = code_lines / max(code_lines + prose_lines, 1)
    stats.update(code_lines=code_lines, code_share=share)
    if share < 0.45 and not appendix:
        notes.append(f"code share {share:.0%} ({code_lines} Python lines), below the 45% standard")
    if stats["excerpts"] < 4 and not appendix:
        notes.append(f"{stats['excerpts']} repository excerpts, below the standard of four")
    badges = sorted({b for b in BADGES if f"`{b}`" in text})
    if not badges and not appendix:
        problems.append("no badge")
    notes.insert(0, f"badges: {', '.join(badges) or 'none'}; references cited: {', '.join(cited) or 'none'}")
    return problems, notes, stats


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    files = sorted(CHAPTERS.glob("[0-9][0-9]-*.md")) + sorted(CHAPTERS.glob("A[0-9]-*.md"))
    wanted = {a.upper().zfill(2) for a in sys.argv[1:]}
    if wanted:                                  # a writer checking its own chapters: no shared report
        files = [f for f in files if f.name[:2].upper() in wanted]
    L = ["# Chapter verification", "", f"Written by `book/tools/verify_chapters.py` against `@{PIN}`.", ""]
    total = 0
    for p in files:
        problems, notes, st = check(p)
        total += len(problems)
        L += [f"## {p.name}", "",
              f"Excerpts: {st['excerpts']} ({st['excerpt_lines']} quoted lines). Line references to definitions: "
              f"{st['line_refs']}. Problems: **{len(problems)}**.", ""]
        L += [f"- {x}" for x in problems] or ["- none"]
        L += ["", f"_{notes[0]}_", ""]
        L += [f"**Code standard:** {st['code_lines']} Python lines, {st['code_share']:.0%} of the body."]
        L += [f"- {n}" for n in notes[1:]] + [""]
        print(f"{p.name}: {st['excerpts']} excerpts, {st['excerpt_lines']} lines, {st['line_refs']} line refs, "
              f"{len(problems)} problems; code {st['code_lines']} lines = {st['code_share']:.0%}")
        for n in notes[1:]:
            print("    standard:", n)
        for x in problems:
            print("   ", x)
    L.insert(4, f"Chapters: {len(files)}. Problems: **{total}**.")
    L.insert(5, "")
    if not wanted:
        OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}" if not wanted else "(chapters given: shared report not written)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
