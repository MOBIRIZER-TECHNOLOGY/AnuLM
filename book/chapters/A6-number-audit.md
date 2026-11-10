---
title: "Appendix F · Reproducing every number"
part: "Appendices"
accent: slate
pinned: "@2e80b3e"
---

# Appendix F  Reproducing every number

Every number in this book carries one of four labels (`STYLE.md` §7):

- ***verified***: checked against the repository at @2e80b3e, or recomputed by
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
figures and four appendix tables. F.6 states once the settings the
measurements share, and F.7 shows the code behind Chapter 1's repository
counts and citation check.

## F.1  The repository at the pin

The line numbers refer to commit `2e80b3e`, the `anulm/` package layout
after a code-quality pass, on the GitHub branch `port-bundle-fixes`; the
default branch, `main` (`ae59238`), has the old flat layout, whose line
numbers differ (front matter).
To work on exactly the cited code, export the pin rather than checking it out:

```bash
mkdir pin && git archive 2e80b3e | tar -x -C pin
cd pin && CUDA_VISIBLE_DEVICES=-1 python -m tests.test_model   # 80 passed, 0 failed, 80 total (Chapter 41.5)
CUDA_VISIBLE_DEVICES=-1 python -m unittest discover -s tests -p "test_quality_*.py" -t .   # Ran 63 tests ... OK
```

The first suite runs in under a minute on a CPU (46 s on the book's
machine, at @fdbb09a, when it held 76 tests), the second in about 5 s. `-1`
hides the GPU; on the book's machine an empty `CUDA_VISIBLE_DEVICES` did not. A test that cannot run for want of a
package prints SKIP, and the first suite's summary then reads
"N passed, M skipped, F failed, T total" (Appendix C).

The export has no untracked data. The tests need none: they build `tiny()`
models and temporary files. The book's measurements need two things git does
not hold:

- the released checkpoint `ckpt_base_v2.pt`. The Hugging Face repository holds
  an export whose large tensors are stored in bf16, which moves some
  measurements in the last digits (Chapter 41, 41.5);
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

- **Self-contained examples** run on a CPU in seconds, on `tiny()`, `tiny_gqa`
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
Appendix C: 82 rows against 82 tests at @2e80b3e: ok
Appendix B.1: 42 configuration rows recomputed: ok
Appendix E: 79 rows against 79 register entries: ok
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
names. At the last run 2 of 39 such paragraphs named no script in the
paragraph itself. A reviewer can then check each of those against its
chapter's example.

**With `--run`, it runs the self-contained examples.** Each runs in a temporary
directory. The audit skips the examples in its `SKIP` list, each with its
reason: they need `ckpt_base_v2.pt`, the v2 corpus, the base run's log or a
local export in `release/`. It prints what it skipped and why, and fails if
the list names a file that does not exist. A new example runs unless someone adds it to the list.

It exits with status 1 if an appendix check fails or an example run fails.

*verified*: the four lines above are the script's output at @2e80b3e, rerun
on a CPU with `ckpt_base_v2.pt` present (the B.1 line needs it). The run
printed `0 problems`.

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
which runs them in bf16 even without autocast (F.6). The pages that quote a
measurement name its precision. A number that moves by more than its stated
sampling error is a finding: report it with the script, the machine and the
PyTorch version.

## F.6  Measurement settings

The book's measurements on the released model share one set of settings. A
page that quotes a measurement points here and names only what differs.

| Setting | Standard |
|---|---|
| model | `ckpt_base_v2.pt` (step 1,899,999), loaded through `anulm/model.py` unchanged, with the configuration stored in the checkpoint |
| data | the base run's held-out split `data/v2/val.u16`; statistics only, no held-out text printed or saved |
| windows | start positions drawn with `numpy.random.default_rng(0)`; 64 windows of 512 tokens, in batches of 4, for the main sample; 16 or 32 windows where a page says so |
| grading | where a script computes a loss, the target after an end-of-document id (32,767) is not graded |
| seeds | 0, for the window draw and for `torch.manual_seed` |
| precision | fp32 weights and activations, no autocast, except the experts (below) |
| machine | the book's GPU (an NVIDIA GeForce RTX 5070 Ti, 16 GB) when at least 4 GB is free, otherwise the CPU; PyTorch 2.11.0+cu128 |

**The one exception to fp32.** The checkpoint's configuration sets
`moe_impl="grouped"`, and that path runs the experts in bf16 even with
autocast off: it casts the stacked expert weights to bf16 (`anulm/model.py`
line 766) and the expert-sorted token vectors (line 794) before its grouped
matrix multiplies. Everything else, the router included, stays in fp32. A
measurement that needs the experts in fp32 as well sets `moe_impl="sparse"`,
which picks the same experts and differs only by rounding.

*verified*: the settings are the docstrings and constants of
`measure_part_i_ii.py`, `measure_attention.py`, `ch12_measure_masks.py`,
`ch13_measure_groups.py`, `ch14_measure_shift.py`, `ch15_measure_stream.py` and
`ch16_measure_trained.py`, and the `precision`, `windows`, `seed` and `device`
fields of their files in `book/measurements/`; lines 766 and 794 read at
@2e80b3e.

What differs, by measurement, as each page says:

| Differs in | Where |
|---|---|
| sample size: 64 × 512 (Chapters 3, 5, 11, 16–18, 20); 32 × 1,024 (12); 16 × 512 (13–15) | each chapter's measurement script |
| bf16 autocast, as `sample.py`, `serve.py` and `api.py` run the model | 1.3's sample; Chapter 28 (`part_vi_ch28_sampling.json`); Chapter 29 (`part_vi_ch29_context.json`); Chapter 27's bf16 half (`part_vi_ch27_cache.json`) |
| bf16 autocast, as training ran: the router measured under it | Chapter 23 (`ch23_measure.py`) and Chapter 26 (`ch26_measure_router_bf16.py`), 32 windows of 1,024 |
| bf16 autocast over the base run's own validation windows, 400 strided windows of 1,024 | Chapter 24 (`part_v_b_val_noise.json`) |
| `moe_impl="sparse"`, so fp32 throughout | Chapter 27's fp32 half; Chapter 30's pair loss (CPU); Chapter 31 (`part_vii_lora.json`); Chapter 35 (`part_viii_ix_grow.json`); Chapter 39 (`part_viii_ix_export.json`); Chapter 41 (`part_x_released.json`) |
| a training-shaped batch: 4 windows of 1,024, numpy seed 1, bf16 autocast, train mode | Chapter 7's gradient norms (`measure_part_i_ii.py`) |

## F.7  Chapter 1's counts and the citation check

`book/code-examples/ch01_four_steps.py` (**Book example, not repository
code**) counts the repository by role for Chapter 1.5 and runs the check
behind every code citation. Both read the tree at the pin with git, never the
working tree. The roles are the book's grouping, by package folder, with each
folder's `__init__.py` counted in it; the first test that matches wins:

```python
ROLES = [  # (role, test on the path at the pin); the first match wins
    ("the model", lambda f: f == "anulm/model.py"),
    ("tests", lambda f: f.startswith("tests/")),
    ("training and tokenizer", lambda f: f.startswith("anulm/training/")
     or f in ("anulm/bpe.py", "anulm/muon.py", "anulm/bigram_teacher.py")),
    ("data", lambda f: f.startswith("anulm/data/")),
    ("evaluation", lambda f: f.startswith("anulm/eval/")),
    ("post-training", lambda f: f.startswith("anulm/tasks/") or f in ("anulm/lora.py", "anulm/bidirectional.py")),
    ("speech and vision", lambda f: f.startswith("anulm/multimodal/")),
    ("serving and apps", lambda f: f.startswith(("anulm/inference/", "anulm/apps/")) or f == "anulm/quickstart.py"),
    ("export", lambda f: f.startswith("anulm/hf/") or f == "anulm/hf_tok.py"),
    ("package entry", lambda f: f in ("anulm/__init__.py", "anulm/__main__.py", "model.py")),
    ("tools", lambda f: f.startswith("tools/")),
]
```

The tree is listed and counted by these functions; `at_pin` reads one file at
the pin:

```python
def at_pin(path: str) -> list[str]:
    out = subprocess.run(["git", "show", f"{PIN}:{path}"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", check=True).stdout
    return out.splitlines()


def tree_at_pin() -> list[str]:
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", PIN], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    # A folder of third-party reference code was at the pin and has since been
    # removed from the repository; count only what a clone still holds.
    return [f for f in names if f.endswith(".py") and (ROOT / f.split("/")[0]).exists()]


def repo_map() -> dict:
    """The package, its tests and tools (everything outside the four folders below), by role."""
    files = [f for f in tree_at_pin() if not f.startswith(tuple(d + "/" for d in FOLDERS))]
    groups: dict[str, list] = {}
    for f in files:
        role = next(r for r, test in ROLES if test(f))
        groups.setdefault(role, []).append((f, len(at_pin(f))))
    print(f"\nPython files in anulm/, tests/, tools/ and the root @{PIN}: {len(files)}")
    for role, _ in ROLES:
        fs = groups.get(role, [])
        print(f"  {role:24s} {len(fs):3d} files {sum(n for _, n in fs):6,d} lines")
    return groups


def subdirectories() -> None:
    """The rest of the Python: the lab book, run scripts, examples, the book."""
    py = tree_at_pin()
    print(f"\nPython files in the whole tree @{PIN}: {len(py)}")
    for d in FOLDERS:
        fs = [f for f in py if f.startswith(d + "/")]
        print(f"  {d + '/':24s} {len(fs):3d} files {sum(len(at_pin(f)) for f in fs):6,d} lines")
```

`FOLDERS` is `course`, `experiments`, `examples` and `book`, the folders
Chapter 1.5's table counts on rows of their own.

The citation check is the core of what `verify_chapters.py` does for every
excerpt (F.2):

```python
def citation_holds(path: str, first: int, last: int, quoted: list[str]) -> bool:
    """What the book's checker does for every excerpt: each quoted line, stripped,
    must occur inside the cited range of the file at the pin."""
    window = [s.strip() for s in at_pin(path)[first - 1:last]]
    return all(q.strip() in window for q in quoted if q.strip() and q.strip() != "# ...")
```

Three lines of 1.2's excerpt pass at 1164–1169 and fail at 1064–1069:

```text
citation anulm/model.py 1164-1169 holds: True
citation anulm/model.py 1064-1069 holds: False
```

*verified*: printed by `ch01_four_steps.py`. The check tests membership, not
order; a stricter check, that the lines appear in order and are contiguous
except at `# ...`, also passes for every excerpt in Chapters 1–7.
