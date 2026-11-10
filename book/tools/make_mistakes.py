"""
Appendix I, "Common mistakes, collected": generated from the chapters, never written by hand.

    python book/tools/make_mistakes.py            # rewrite chapters/A9-common-mistakes.md
    python book/tools/make_mistakes.py --check    # exit 1 if the file differs from what the chapters give

Every chapter's "Common mistakes" section has one of three shapes, and each becomes rows
of one table (section, mistake, what happens silently, how to catch it):

- bullets: `- **Mistake.** What happens. *Catch it:* how.`; a bullet may name several
  mistakes in bold before one explanation (Chapter 30), or hold a code block (Chapter 17);
- a table, `| Mistake | Catch it |` (Chapter 26) or with a middle column (Chapter 8);
- prose (Chapters 21, 22, 34): "Mistakes: a; b; c." is split on its semicolons; the two
  paragraphs that name mistakes in running text are transcribed in PROSE below, and every
  phrase the transcription quotes must still be in the section, or the tool refuses.

The wording is the chapters' own, shortened only by dropping the bold and the "Catch it"
marker; a dash means the chapter does not state that cell. `ch41_audit.py` regenerates the
table and compares, so Appendix I cannot drift from the chapters it summarises.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parent.parent
CHAPTERS = BOOK / "chapters"
OUT = CHAPTERS / "A9-common-mistakes.md"
HEADING = re.compile(r"^## (\d+)\.(\d+)\s+(.*[Cc]ommon mistakes.*)$", re.M)
PIN = re.compile(r'pinned: "(@\w+)"')
BOLD_RUN = re.compile(r"((?:\*\*[^*]+?\*\*[,:.]?\s*)+)(.*)", re.S)
CATCH = re.compile(r"\*[Cc]atch it:?\*:?\s*")

# Sections whose mistakes are named in running prose: (mistake, what happens, catch it, phrases
# that must appear in the section). The first three columns paraphrase only where a cell says so.
PROSE = {
    "21.10": [
        ("a tail split of an ordered file", "no loss curve shows it; the held-out split tests one source",
         "count the sources in the held-out split",
         ["a tail split of an ordered file", "Count the sources in the held-out split"]),
        ("`y = x`", "trains a copier", "assert `torch.equal(x[:, 1:], y[:, :-1])`",
         ["`y = x`", "trains a copier", "torch.equal(x[:, 1:], y[:, :-1])"]),
    ],
    "22.9": [
        ("reading progress from the train loss",
         "its standard deviation over the last 20 log lines was 0.271", "—",
         ["reading progress from the train loss", "0.271"]),
        ("accumulating without dividing", "the loss will not show it",
         "a gradient comparison: `gradient_accumulation_equals_large_batch`",
         ["accumulating without dividing", "only a gradient comparison catches", "gradient_accumulation_equals_large_batch"]),
    ],
}


def section(text: str, m: re.Match) -> str:
    """The section's body: from its heading to the next `## `."""
    start = m.end()
    nxt = re.search(r"^## ", text[start:], re.M)
    return text[start: start + nxt.start()] if nxt else text[start:]


def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().rstrip(".").strip()
    return s.replace("|", "\\|") or "—"


def items_of(body: str) -> list[str]:
    """Bullet items of a section, each as one string; code fences inside an item are dropped."""
    items, cur, fence = [], None, False
    for line in body.splitlines():
        if line.startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if line.startswith("- "):
            if cur is not None:
                items.append(" ".join(cur))
            cur = [line[2:].strip()]
        elif cur is not None and (line.startswith((" ", "\t")) or not line.strip()):
            if line.strip():
                cur.append(line.strip())
        elif cur is not None:
            items.append(" ".join(cur))
            cur = None
    if cur is not None:
        items.append(" ".join(cur))
    return items


def rows_from_bullets(body: str, sec: str) -> list[tuple]:
    rows = []
    for item in items_of(body):
        if item.startswith("`") and " (line " in item:          # a proof list, not a mistake
            continue
        m = BOLD_RUN.match(item)
        if m:
            mistakes = [x.rstrip(".") for x in re.findall(r"\*\*(.+?)\*\*", m.group(1))]
            rest = m.group(2)
        else:
            mistakes, rest = [item], ""
        parts = CATCH.split(rest, 1)
        what, catch = parts[0], parts[1] if len(parts) > 1 else ""
        rows += [(sec, clean(mk), clean(what), clean(catch)) for mk in mistakes]
    return rows


def rows_from_table(body: str, sec: str) -> list[tuple]:
    rows = []
    for line in body.splitlines():
        if not line.startswith("|") or line.startswith("|---") or "Mistake" in line.split("|")[1]:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 2:
            rows.append((sec, clean(cells[0]), "—", clean(cells[1])))
        elif len(cells) >= 3:
            rows.append((sec, clean(cells[0]), clean(cells[1]), clean(cells[2])))
    return rows


def rows_from_prose(body: str, sec: str) -> list[tuple]:
    flat = re.sub(r"\s+", " ", body)
    if sec in PROSE:
        rows = []
        for mistake, what, catch, phrases in PROSE[sec]:
            for ph in phrases:
                if ph not in flat:
                    sys.exit(f"make_mistakes: §{sec} no longer says {ph!r}; update PROSE in this tool")
            rows.append((sec, mistake, what, catch))
        return rows
    m = re.search(r"Mistakes:\s*(.+?)\.\s*(?:\*\*|$)", flat)
    return [(sec, clean(c), "—", "—") for c in m.group(1).split(";")] if m else []


def collect() -> tuple[list[tuple], str]:
    rows, pin = [], "@?"
    for f in sorted(CHAPTERS.glob("[0-9][0-9]-*.md"), key=lambda p: int(p.name[:2])):
        text = f.read_text(encoding="utf-8")
        pin = PIN.search(text).group(1) if PIN.search(text) else pin
        for m in HEADING.finditer(text):
            sec = f"{m.group(1)}.{m.group(2)}"
            body = section(text, m)
            if re.search(r"^\| Mistake \|", body, re.M):
                got = rows_from_table(body, sec)
            else:
                got = rows_from_bullets(body, sec) or rows_from_prose(body, sec)
            if not got:
                sys.exit(f"make_mistakes: §{sec} has a Common-mistakes heading but no row was parsed")
            rows += got
    return rows, pin


def render(rows: list[tuple], pin: str) -> str:
    chapters = sorted({int(r[0].split(".")[0]) for r in rows})
    stated = sum(r[3] != "—" for r in rows)
    head = f"""---
title: "Appendix I · Common mistakes, collected"
part: "Appendices"
accent: slate
pinned: "{pin}"
---

# Appendix I  Common mistakes, collected

The "Common mistakes" sections of the {len(chapters)} chapters that have one, in one
table: {len(rows)} rows, {stated} of them with the check that catches the mistake. The
wording is the chapters' own, shortened; the section in the first column has the example
that shows the mistake happening and the test that pins it. A dash means the chapter does
not state that cell. Nearly every row shares the property that made the book collect them:
the code still runs, and the loss still falls.

This table is **generated** by `book/tools/make_mistakes.py` from the chapter files, and
`book/code-examples/ch41_audit.py` regenerates it on every run and compares. To change a
row, change the chapter.

| § | Mistake | What happens, silently | Catch it |
|---|---|---|---|
"""
    body = "\n".join(f"| {sec} | {mk} | {what} | {catch} |" for sec, mk, what, catch in rows)
    tail = f"""

## I.1  Reading the table

Three kinds of row recur. **Right size, wrong meaning**: a `view` where a `transpose` was
meant (4.7), `repeat` where `repeat_interleave` was meant (13.7), a mask applied after the
softmax (12.7); the tensor has the right shape and the loss falls anyway. **The wrong thing
saved or loaded**: EOS cast to a byte (8.7), weights saved without the optimiser and then
resumed (25.9), a checkpoint written in place (25.9). **The wrong instrument**: progress read
from the train loss (22.9), a move judged by the printed ± (24.8), a test that has never
been seen to fail (26.11). The catch column is almost always the same move: an equivalence
or invariance check, stated as an assertion, that a falling loss cannot satisfy by accident
(Chapter 26).

`VERIFIED` · generated from the chapters at {pin}.
"""
    return head + body + tail


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare only; exit 1 on a difference")
    a = ap.parse_args()
    rows, pin = collect()
    text = render(rows, pin)
    n_ch = len({r[0].split(".")[0] for r in rows})
    if a.check:
        same = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print(f"Appendix I: {len(rows)} rows regenerated from {n_ch} chapters: "
              + ("ok" if same else "MISMATCH, rerun make_mistakes.py"))
        return 0 if same else 1
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(BOOK)}: {len(rows)} rows from {n_ch} chapters at {pin}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
