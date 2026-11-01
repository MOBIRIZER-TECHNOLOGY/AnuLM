# Brief for chapter writers

You are writing chapters of *Inside AnuLM*, a reference book that teaches a
complete language model by reading its own source code. Several writers work
at the same time, one per part. This brief is the contract you all share.
Read it fully before writing.

## Read first

1. `book/docs/authoring/style.md`: the voice, page furniture, badges, evidence levels,
   citation forms, notation (§3) and figures (§9). Follow it exactly.
2. `book/docs/authoring/book-plan.md`:
   - your chapters' entries, which list each chapter's topics, repository
     code, tests and RESULTS sections;
   - "Page budget and code share", the length and the code standard;
   - "Chapter dependencies".
3. `book/chapters/11-attention.md` and `book/chapters/20-the-whole-model.md`:
   the reviewed exemplars. Match their furniture: front matter, numbered
   sections, In the repo strips, *verified* / *measured* / *illustrative*
   labels, badges at section ends, Details boxes, Key points (exactly six),
   In one breath (about fifty words), exercises with answers, References
   table, repository evidence list. Do not copy their length or their
   prose-to-code balance; see "The code standard" below.
4. `book/docs/reference/repository-source-map.md`: where each component lives, at the pin.
5. `book/docs/reference/research-references.md`: the only sources you may cite.
6. The lab book, `course/chNN-*/chapter.md`, for "Build it" cross-links.

## The code at the pin

Every line number refers to commit `773b6a8`. Read code with
`git show 773b6a8:model.py` (or `train.py`, `test_model.py`, …) and find lines
with `grep -n` on that output. Never cite line numbers from memory, from the
working tree, or from `.claude/worktrees/`. The working tree's `omni_train.py`
has uncommitted edits, so cite it only at the pin. Quote code verbatim; mark
elided lines `# ...`. Never elide a branch that changes the meaning of what
you show. Docstrings can be stale, so quote code, not docstrings, as evidence.

## The code standard (checked by the tools)

- **At least 45% of each chapter's body lines are Python.** That is repository
  excerpts plus book examples; printed output and tables do not count.
- **At least four repository excerpts** per chapter. Each is followed by a
  walkthrough that goes through the lines in order and says what each part
  does, with the tensor shapes.
- **One runnable book example per chapter**, in
  `book/code-examples/chNN_<name>.py`. Its first docstring line is
  "Book example, not repository code." It imports the repository's code
  unchanged and checks itself against it. Quote it in the chapter where it
  teaches, and print what it prints.
- **Short prose.** State each idea once. Put what a first reading can skip in
  a `> **Details.**` box.
- **Pages.** Your chapters' budget in `docs/authoring/book-plan.md` is a guide, not a cap (revised 2026-11-01). Never drop a
  figure, test, excerpt or walkthrough to fit it; cut only repetition. As a rough
  measure, 18 Markdown body lines of prose or 40 lines of code make one printed
  page.

## Evidence rules

- **Repository claims** need an In the repo citation or a line number at the
  pin, checked.
- **External claims** cite a `[Rnn]`/`[Snn]` whose status is CHECKED, and only
  for what its register row's "claims" cell says. If you need a source that is
  not in the register, do not cite it. Write the claim without the citation,
  and add a proposal (title, authors, year, arXiv or DOI, the exact claim, how
  you verified it) to `book/docs/reviews/register-additions/<part>.md`.
  The integrator decides.
- **Quality claims** (what helps, what hurts) must cite a `docs/RESULTS.md`
  section or say *not measured here*.
- **Numbers** must be printed by a script you ran: a book example, a figure
  script, or a measurement script. Label each one *verified*, *measured* or
  *illustrative*.
- **Absent features** (top-p, distributed training, quantisation, DPO in the
  main code, …) are taught and badged `NOT IN THE REPO`. Never imply the
  repository does something it does not.

## Figures

Two to four per chapter, drawn as SVG by your part's module
`book/tools/figures/<part>.py`. Every function `fig_*` there returns the file
name it wrote. Import the helpers and palette from `make_figures.py` (see the
Ch 11 and 20 functions there for examples). Name the files
`figNN-k-name.svg`, and assert key numbers against the chapter. Run
`python book/tools/make_figures.py <part>` and look at each SVG rendered with
headless Chrome:

```bash
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu --screenshot=<out.png> --window-size=780,600 file:///C:/workspace/AnuLM/book/figures/<file>.svg
```

Then read the PNG and fix overlaps and clipping. Follow docs/authoring/style.md §9 for
colours: one gold ramp for magnitudes, coral and teal for two series, text
never in a data colour.

## Measurements on the trained model

Where a chapter explains a behaviour of the model, measure it on
`ckpt_base_v2.pt`. Use the held-out split `data/v2/val.u16` for statistics
only, and never print or save held-out text. Never use
`data/bench_english.txt`, which is web spam. Save results to
`book/measurements/<your-name>.json`.

The machine has one 16 GB GPU, shared by ten writers. Before using it, run
`nvidia-smi`. Use it only if at least 4 GB is free, keep runs to about a minute,
and fall back to the CPU with a smaller sample otherwise.

## What you may and may not edit

- **You may create and edit:**
  - your chapters, `book/chapters/NN-<slug>.md`;
  - your examples, `book/code-examples/chNN_*.py`;
  - your figure module, `book/tools/figures/<part>.py`;
  - your figures, `book/figures/figNN-*.svg`;
  - your measurements, `book/measurements/<part>_*.json`;
  - your chapter records, `book/docs/reviews/claims/NN.md`;
  - your register proposals, `book/docs/reviews/register-additions/<part>.md`.
- **You may not edit** anything else: no repository code; no other part's
  files; not `README.md`, `docs/authoring/style.md`, `docs/authoring/book-plan.md`, `docs/reference/research-references.md`,
  `docs/reference/repository-source-map.md` or the shared tools; and not Chapters 11 or 20.
  If one of those needs a change, say so in your report.
- **No git.** No commits, branches or stashes.

## Checking your own work before you report

For each chapter:

1. `python book/tools/verify_chapters.py NN` must show 0 problems and meet the
   code standard. It checks excerpts against the pin, `def` lines, figures,
   furniture, the code share and the excerpt count.
2. Run every book example. Every number in the chapter must appear in the
   output of a script.
3. Re-read the chapter as the target reader: an engineer who knows Python and
   a little PyTorch but has never studied transformers. Every term must be
   defined before it is used.
4. Write `book/docs/reviews/claims/NN.md`. It opens
   `## Chapter NN · <title>` and has the line `**Status:** drafted`. It lists
   the sources of every number, the excerpts and their line ranges, any
   register proposals, and anything you could not verify. Do not write
   "reviewed": an independent review comes after you.

## Practical notes

- The scratchpad is shared by every writer: keep your scratch files in your own subfolder.
- Bash heredocs on this machine can eat backslashes. Write any script that
  contains backslashes to a file and run that.
- Write one chapter at a time and save it before starting the next, so a long
  session never loses work.
- Your final report is your last message: per chapter, the file, its sections,
  its printed page estimate, code share, excerpt count, figures, examples,
  measurements, register proposals, and every open question. Report anything
  that failed. Do not claim a check you did not run.
