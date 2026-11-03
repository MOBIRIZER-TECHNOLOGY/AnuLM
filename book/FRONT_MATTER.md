---
title: "Inside AnuLM"
subtitle: "A language model you can read, line by line"
author: "The AnuLM authors"
edition: "Preview, 2026-10-31"
---

# Inside AnuLM

*A language model you can read, line by line*

**By the AnuLM authors**

This book explains a complete language model, AnuLM, by reading its own source
code: the theory, the mathematics, the exact lines that implement each idea,
and measurements of the trained model. Everything it describes can be
downloaded, run and checked.

## Where everything is

| What | Where |
|---|---|
| Source code, MIT licence | [github.com/MOBIRIZER-TECHNOLOGY/AnuLM](https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM) |
| The model this book traces, AnuLM-Base-v2-400M | [huggingface.co/toonist/AnuLM-Base-v2-400M](https://huggingface.co/toonist/AnuLM-Base-v2-400M) |
| Every released AnuLM checkpoint | [huggingface.co/toonist](https://huggingface.co/toonist) |
| The lab book that builds the model from stubs | `course/` in the same repository |
| This book's source, scripts and review records | `book/` in the same repository |

## The code this book cites

Every code citation names a file, a symbol, line numbers and a commit:

> **In the repo** · `anulm/model.py` · `_sdpa` · lines 407–411 · @084cbdd

The line numbers are checked against commit `084cbdd` by the book's own tools.
That commit is the tip of the repository's `restructure` branch, which moved
the code into the `anulm/` package, and it is not on GitHub yet. The public
`main`,
[`ae59238`](https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM/tree/ae592387dc373cc6f579abce444c7f08c9e373a3),
still has the old flat layout (`model.py` at the root, not `anulm/model.py`),
so its line numbers differ and the HTML edition does not link citations yet.
Once the branch is published, `tools/build_preview.py` links every citation
whose file is identical at the published commit. The lab book under `course/`,
which the "Build it" boxes point to, is in the pin.

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
