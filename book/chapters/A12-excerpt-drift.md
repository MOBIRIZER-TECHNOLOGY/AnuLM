---
title: "Appendix L · Keeping the excerpts honest"
part: "Appendices"
accent: slate
pinned: "@2e80b3e"
---

# Appendix L  Keeping the excerpts honest

Every code box in this book names a file, a symbol, a line range and the commit the lines
were copied from: `@2e80b3e`, the pin. The repository keeps moving. A reader with a newer
checkout will find some of those lines at other numbers and some of them rewritten, and
has no way to tell which from the page. This appendix is the tool that tells them.

## L.1  What the tool does

`book/tools/excerpt_drift.py` reads every excerpt box in the chapters (347 at this pin),
fetches the cited file at any commit you name with `git show`, and classes each excerpt:

| Status | Meaning |
|---|---|
| same | every quoted line is still inside the cited range |
| moved | the quoted lines are all in the file, together, at other line numbers (it reports them) |
| changed | at least one quoted line is no longer in the file (it reports the first) |
| absent | the file does not exist at that commit |

Blank lines and `# ...` ellipses in a quote are ignored, and lines are compared with their
indentation stripped, exactly as `verify_chapters.py` compares them at the pin. The tool
never edits a chapter. Re-pinning the book is `repin.py`'s job, which works from the diff
between two pins and rewrites line numbers; this is the reader's check.

```
python book/tools/excerpt_drift.py                        # against HEAD
python book/tools/excerpt_drift.py --ref origin/main      # against a branch
python book/tools/excerpt_drift.py --ref 2e80b3e --strict # exit 1 unless every excerpt is "same"
python book/tools/excerpt_drift.py --ref v0.3 --write     # also technical-review/excerpt-drift.md
```

## L.2  Three runs

Run on 2026-11-10 from the `restructure` branch (*verified*; the three commands and
their first lines, unedited).

At the pin itself, and at the commit before it, nothing has moved. This is the audit's
self-check: `ch41_audit.py` runs the first command and requires 347 of 347.

```
$ python book/tools/excerpt_drift.py --ref 2e80b3e
347 excerpts against 2e80b3e: 347 same, 0 moved, 0 changed, 0 absent
```

Against the pin before this one, `b42998d` (the 7 November code rewrite lies between
them), most excerpts have moved and a few dozen have changed. These are the numbers
`repin.py` worked through when the book was re-pinned (`technical-review/repin-report.md`):

```
$ python book/tools/excerpt_drift.py --ref b42998d
347 excerpts against b42998d: 132 same, 186 moved, 29 changed, 0 absent
```

Against the repository's default branch, `main` at `ae59238`, almost everything is
absent, and that is the right answer: `main` still has the old flat layout, with
`model.py` at the root instead of `anulm/model.py`, so the files the book cites are not
there under those names. The six that are not absent are in `experiments/`, which both
layouts keep at the same path:

```
$ python book/tools/excerpt_drift.py --ref origin/main --write
347 excerpts against origin/main: 0 same, 4 moved, 2 changed, 341 absent
```

| Ch | File | Symbol | Pinned lines | Status | Note |
|---|---|---|---|---|---|
| 21 | `experiments/build_v2.py` | `SOURCES` | 76–83 | moved | now lines 74–81 |
| 21 | `experiments/build_v2.py` | `main` | 103–114 | moved | now lines 101–112 |
| 21 | `experiments/build_v2.py` | `main` | 136–140 | moved | now lines 134–138 |
| 37 | `experiments/omni_one.py` | `gates` | 403–411 | moved | now lines 230–238 |
| 37 | `experiments/omni_one.py` | `ACCUM` | 65–100 | changed | the accumulation block was rewritten after `main`'s copy |
| 37 | `experiments/omni_one.py` | `main` | 493–502 | changed | the launch arguments changed |

The full table is `book/technical-review/excerpt-drift.md`, written by the third command.

## L.3  What to do with the answer

If your checkout reports an excerpt *moved*, read the chapter as printed and open the file
at the new lines the tool gives. If it reports *changed*, read the code at the pin
(`git show 2e80b3e:anulm/model.py`) beside your version: the chapter explains the pinned
version, and the diff tells you what has been done since. If it reports *absent*, you are
on a branch that does not have the `anulm/` package; the front matter says which branch
does. Chapter 1.7 explains the pin and the labels; Appendix F reproduces the numbers.

`VERIFIED` · the three runs above, and `ch41_audit.py`'s self-check at the pin.
