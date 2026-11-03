---
title: "Appendix F · Reproducing every number"
part: "Appendices"
accent: slate
pinned: "@fdbb09a"
---

# Appendix F  Reproducing every number

Every number in this book carries one of four labels (`STYLE.md` §7):

- ***verified***: checked against the repository at @fdbb09a, or recomputed by
  a book script;
- ***measured***: from a run recorded in `docs/RESULTS.md` or `logs/`, or from
  a book script run on the released checkpoint, whose output is saved in
  `book/measurements/`. Chapter 41 also quotes a run recorded in `README.md`,
  a source `STYLE.md` §7 does not list;
- ***cited***: from a source in Appendix E whose status is CHECKED;
- ***illustrative***: made up for teaching and labelled so. The chapter's own
  example is meant to recompute it, but not every page names that example
  beside the number (F.4).

This appendix lists the scripts that produce or check those numbers, says what
each one actually checks, and gives the order to run them in. None trains a
model. No single script recomputes every number: each chapter's examples
recompute that chapter's, and the tools below check excerpts, citations,
figures and four appendix tables.

## F.1  The repository at the pin

The line numbers refer to commit `fdbb09a`, the `anulm/` package layout
after a code-quality pass, on the GitHub branch `port-bundle-fixes`; the
default branch, `main` (`ae59238`), has the old flat layout, whose line
numbers differ (front matter).
To work on exactly the cited code, export the pin rather than checking it out:

```bash
mkdir pin && git archive fdbb09a | tar -x -C pin
cd pin && CUDA_VISIBLE_DEVICES=-1 python -m tests.test_model   # 76 passed, 0 failed, 76 total (Chapter 41.5)
CUDA_VISIBLE_DEVICES=-1 python -m unittest discover -s tests -p "test_quality_*.py" -t .   # Ran 63 tests ... OK
```

The first suite runs in under a minute on a CPU (46 s on the book's
machine), the second in about 5 s. `-1` hides the GPU; on the book's machine
an empty `CUDA_VISIBLE_DEVICES` did not. A test that cannot run for want of a
package prints SKIP, and the first suite's summary then reads
"N passed, M skipped, F failed, T total" (Appendix C).

The export has no untracked data. The tests need none: they build `tiny()`
models and temporary files. The book's measurements need two things git does
not hold:

- the released checkpoint `ckpt_base_v2.pt`. The Hugging Face repository holds
  an export whose large tensors are stored in bf16, which moves some
  measurements in the last digits (Chapter 41.5);
- for statistics on held-out text, the split `data/v2/val.u16`, which
  `experiments/build_v2.py` rebuilds (`docs/DATASETS.md`).

## F.2  The book's tools

All are in `book/tools/`. They read cited code at the pin with `git show`. The
two that import `anulm/model.py`, `verify_claims.py` and `make_figures.py`, use the
working tree's copy, which `verify_source_map.py` checks is unchanged since the
pin.

| Command | What it checks | Writes |
|---|---|---|
| `python book/tools/verify_source_map.py` | every row of `REPOSITORY_SOURCE_MAP.md`: the symbol's span, from Python's parser, falls inside the cited lines; the test map; cited files unchanged since the pin | `technical-review/source-verification.md` |
| `python book/tools/verify_claims.py` | the plan's numeric and narrative claims, recomputed from the code and the released checkpoint | appends to the same file |
| `python book/tools/verify_chapters.py [NN or AN ...]` | chapters (`NN-*.md`) and appendices (`A[0-9]-*.md`): each "In the repo" excerpt, line for line, against the cited range; "`name` (line N)" references against the `def` line; and, for chapters only, the page furniture and the code standard (45% Python, four excerpts) | `technical-review/chapter-verification.md` (when run on everything) |
| `python book/tools/research_audit.py` | references by status, citations of anything not CHECKED, references no chapter cites | `technical-review/research-audit.md` |
| `python book/tools/make_figures.py [part]` | draws every figure from its source, asserting each key number against the value the chapter prints | `book/figures/*.svg` |
| `python book/tools/build_preview.py --print` | renders chapters and appendices; before linking a citation to GitHub it checks that the cited file is identical at the public commit | `book/preview/` |

## F.3  The book's examples

Each chapter has at least one example in `book/code-examples/`, named
`chNN_*.py` (Chapters 11 and 20 use `attention_by_hand.py`, `trace_forward.py`
and `measure_attention.py`) and labelled **Book example, not repository code**.
It imports the repository's code unchanged and checks itself against it, and
the chapter prints what it prints. They fall into three kinds:

- **Self-contained examples** run on a CPU in seconds, on `tiny()`, `nano_30b`
  or hand-sized tensors, and write nothing into the book.
- **Examples that need the released model or data**: every `chNN_measure_*.py`,
  `measure_attention.py`, `ch41_checkpoints.py released`, and others whose
  source names `ckpt_base_v2.pt`. They use a GPU when at least 4 GB is free.
- **Examples that write the book's measurement files**, `book/measurements/<part>_*.json`.
  Every one does so only when given `--save`; without it, the JSON goes to the
  current directory. Two examples, `ch07_optim.py` and `ch24_error_bars.py`,
  only read a measurement file. Each saves statistics, never text, and names
  its data, sample size, seed and precision, as does the page that quotes it.

`trace_forward.py` (Chapter 20) and `ch41_shapes.py` (Appendix A) print every
shape; `ch41_arithmetic.py` prints Appendix B's counts.

Four sources the book measures from are not in git: they exist only on the
author's machine. They are the base run's log `v2_train_run.log` (Chapters 18
and 22), the joint run's `logs/omni_one/6_train.log` and `results.json`
(Chapter 37), and the local export `release/AnuLM-Base-v2-400M/` (Chapter 39).
The numbers the chapters quote from them are kept in `book/measurements/`:
`part_iv_b_trained.json`, `part_v_a_run.json`, `part_viii_ix_omni_log.json`
and `part_viii_ix_export.json`. `local_sources.json` lists each source, the
file that holds its numbers and the script that parsed it. Without the source,
check a quoted number against that file.

## F.4  The audit script

`book/code-examples/ch41_audit.py` (**Book example, not repository code**) does
three things.

**It checks four appendix tables against their sources:**

```
Appendix C: 76 rows against 76 tests at @fdbb09a: ok
Appendix B.1: 42 configuration rows recomputed: ok
Appendix E: 66 rows against 66 register entries: ok
Appendix A: 47 distinct concrete shapes, 0 not in ch41_shapes.py's output
```

- Appendix C: test names and lines, exactly.
- Appendix B.1: every configuration row, exactly.
- Appendix E: every bibliography row, exactly.
- Appendix A: membership only. Each shape must occur somewhere in the shape
  script's output; nothing checks that it sits in the right row.

**It takes an inventory of evidence.** For each chapter it counts how many
passages carry each label and how many references the chapter cites. It also
lists every paragraph labelled *illustrative*, with the script that paragraph
names. At the time of writing 32 of 39 such paragraphs named no script in the
paragraph itself. A reviewer can then check each of those against its
chapter's example.

**With `--run`, it runs the self-contained examples.** Each runs in a temporary
directory. The audit skips the examples in its `SKIP` list, each with its
reason: they need `ckpt_base_v2.pt`, the v2 corpus, the base run's log or a
local export in `release/`. It prints what it skipped and why, and fails if
the list names a file that does not exist. A new example runs unless someone adds it to the list.

It exits with status 1 if an appendix check fails or an example run fails.

*verified*: the four lines above are the script's output at the time of
writing. At @fdbb09a the lines for Appendices C, E and A were rerun and are
unchanged; the B.1 line needs `ckpt_base_v2.pt` and was not rerun, but its
Line column was recomputed from `anulm/model.py` at the pin and still
matches.

## F.5  Everything, in order

```bash
python book/tools/verify_source_map.py        # the map against the pin
python book/tools/verify_claims.py            # the plan's numbers
python book/tools/verify_chapters.py          # every chapter's and appendix's excerpts
python book/tools/research_audit.py           # every citation
python book/tools/make_figures.py             # every figure, with its asserts
python book/code-examples/ch41_audit.py --run # four appendix tables, the evidence inventory, the self-contained examples
```

Then, with the checkpoint and the held-out split in place, run the remaining
examples one at a time. Where a script supports it, write to a path of your
own rather than to `book/measurements/`, and compare its output with the
book's JSON. Results in bf16 can differ in the last digits from results in
fp32. This includes the released model's experts on its own `grouped` path,
which runs them in bf16 even without autocast. The pages that quote a
measurement name its precision. A number that moves by more than its stated
sampling error is a finding: report it with the script, the machine and the
PyTorch version.
