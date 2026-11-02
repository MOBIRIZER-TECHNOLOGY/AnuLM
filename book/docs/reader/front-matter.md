---
title: "Inside AnuLM"
subtitle: "A language model you can read, line by line"
author: "The AnuLM authors"
edition: "Complete technical preview, 2026-11-02"
---

# Inside AnuLM

*A language model you can read, line by line*

**By the AnuLM authors**

This book explains a complete language model, AnuLM, by reading its own source
code: the theory, the mathematics, the exact lines that implement each idea,
and measurements of the trained model. The public companion code supports
the introductory examples. Some historical measurements require local
artifacts, and the optional course labs remain unpublished; their availability
is stated below and in the reader guide.

Start with [Start here](start-here.md) for installation and a first CPU model,
then [Learning paths](learning-paths.md). This is a technical preview with
41 chapters and six appendices. It explains a working model and demonstration
server; it does not certify a production service. The
[Launch checklist](launch-checklist.md) states the remaining acceptance gates.

## Current companion update

The [current workflow](current-workflow.md), [language coverage](language-coverage.md)
and [course route](course-route.md) document the revised package. Historical
chapter citations and measurements retain their recorded revision. The new run
is ongoing; its quality is not approved. Revised changes are still local.

## Where everything is

| What | Where |
|---|---|
| Source code, MIT licence | [github.com/MOBIRIZER-TECHNOLOGY/AnuLM](https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM) |
| The model this book traces, AnuLM-Base-v2-400M | [huggingface.co/toonist/AnuLM-Base-v2-400M](https://huggingface.co/toonist/AnuLM-Base-v2-400M) |
| Every released AnuLM checkpoint | [huggingface.co/toonist](https://huggingface.co/toonist) |
| Current practical study route | [Course route](course-route.md); historical `course/` stub labs remain unavailable |
| This book's source, scripts and review records | `book/` in the same repository |

## The code this book cites

**Current layout.** The organized source is under `anulm/`, grouped by purpose.
Historical citations below retain their original filenames and revisions.
Use the companion repository's `docs/STRUCTURE.md` and `docs/source-paths.json`
to find current modules. The native model and standalone HF templates retain
their implementation; training imports, resource locations and command entry
points have changed. Run current commands through `python -m anulm`.

Every code citation names a file, a symbol, line numbers and a commit:

> **In the repo** · `model.py` · `_sdpa` · lines 393–397 · @773b6a8

The line numbers are checked against commit `773b6a8` by the book's own tools.
That commit is not on GitHub. The public commit
[`ae59238`](https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM/tree/ae592387dc373cc6f579abce444c7f08c9e373a3),
the public revision used for the core examples, was checked against the
historical core files during authoring. The preview builder adds a code link
only when it can verify equivalence from Git. In a standalone book folder
without those Git objects it leaves the citation unlinked; this is not a new
verification of the historical pin. The two commits differ in the lab book: the `course/`
chapters that the "Build it" boxes point to are on the repository's
`course-book` branch, which is not yet published.

## Licences

- **Code:** MIT (`LICENSE` in the repository).
- **AnuLM-Base-v2-400M weights:** CC BY-SA 4.0, inherited from its training
  data (the repository's `NOTICE` §2 and `docs/DATASETS.md`).
- **The redistributed Sarvam files** in `sarvam/` keep their own Apache 2.0
  licence and headers.
- **This book's text:** not yet chosen; to be settled before publication.

AnuLM is an independent project. It is not affiliated with or endorsed by
Sarvam AI, BharatGen, AI4Bharat or the Government of India.

## How to cite

The repository's `CITATION.cff` gives the citation for AnuLM itself:

> The AnuLM authors (2026). *AnuLM: a small, trainable mixture-of-experts
> language model in the shape of Sarvam 30B, trained from scratch on one GPU
> for Hindi, English and Python.* Version 0.1.0.
> https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM


## Text license

The author selected CC BY 4.0 for original book prose and reader/course guides on 2 November 2026. Companion code remains MIT. Separately identified third-party materials retain their own terms. See the book root LICENSE-TEXT.md and LICENSE-CC-BY-4.0.txt.
