# Reviewer guide

Every chapter so far has been reviewed by AI reviewers, each given a narrow
brief: an independent technical reviewer and a simulated target reader
(`claims-review.md`). Those reviews found real errors, but they are not a
substitute for people. Before publication each part needs two human reviews:
one by an **expert**, one by a **target reader**. This guide is what to hand
them.

A chapter takes an expert about an hour and a reader about two.

---

## What to send

- The chapter files from `book/chapters/`, or the rendered preview
  (`python book/tools/build_preview.py` → `book/preview/index.html`).
- This guide, and the feedback table at the end of it.
- For the expert: access to the repository at the commit the chapter pins
  (`@773b6a8`), and `book/RESEARCH_REFERENCES.md`.

Ask each reviewer not to read the other reviews first.

## The expert

Someone who has trained or implemented transformer language models, ideally a
mixture-of-experts model.

**What to check, in order of importance:**

1. **Is anything wrong?** Every formula, shape, number and statement about
   what the code does. The chapter's numbers come from scripts in
   `book/code-examples/`; rerun any you doubt:
   ```bash
   python book/code-examples/trace_forward.py
   ```
   ```bash
   python book/code-examples/attention_by_hand.py
   ```
2. **Is anything misleading though true?** A simplification that will trip a
   reader up later, a measurement on a sample presented as general, an
   initialisation property presented as permanent.
3. **Is anything important missing?** A failure mode practitioners hit, a
   standard explanation the chapter skips, a better reference.
4. **Do the citations support their claims?** A chapter may cite a source only
   for what its row in `RESEARCH_REFERENCES.md` says. Flag any stretch.

Checks that are already automated, so the expert need not repeat them:
- code excerpts and line numbers (`book/tools/verify_chapters.py`);
- the reference register (`book/tools/research_audit.py`);
- the source map (`book/tools/verify_source_map.py`).

## The target reader

A software engineer who writes Python and has used PyTorch a little, who knows
what a matrix multiply and a gradient are, and who has **not** studied
transformers. They should read the chapters in order, alone, as they would a
book.

**Ask them to:**

1. Mark every place they had to re-read, and say what they expected.
2. Mark every term they met before it was explained.
3. Do the exercises before looking at the answers, and say which needed
   something the chapter did not teach.
4. Run one companion script, following only the chapter's instructions.
5. At the end, write three sentences: what they now understand, what they
   still do not, and the one change that would have helped most.

## Feedback table

One row per finding. Quote the sentence; a line number alone goes stale.

| # | Chapter and section | Quote | Kind: error / misleading / missing / unclear / other | What you expected or suggest | Severity: blocks understanding / slows / minor |
|---|---|---|---|---|---|
| 1 | | | | | |

## What happens to the findings

- Each finding is reproduced against the repository before anything changes.
- Each one is recorded in `claims-review.md`, under the chapter, with what was
  done about it. Findings that are declined are recorded with the reason.
- Reviewers are thanked by name in the book's acknowledgements, if they wish.

## Status

| Part | Expert review | Target-reader review |
|---|---|---|
| IV (Ch 11, 20 so far) | not yet | not yet (AI-simulated reader review done, 2026-10-31) |
