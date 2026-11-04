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

## Preface

**Who this book is for.** You write Python comfortably. You know what a vector
and a matrix are and can multiply two small ones by hand. You may have used
PyTorch little or not at all, and you have never studied how a transformer
works. You want to know what happens inside a language model, not as a box in
a diagram but as code that runs. Each term is defined when it first appears,
or the page says which chapter defines it, and Part II teaches the PyTorch and
the mathematics you need, on the model's own lines.

**What you will be able to do.** By the end you can read, trace, test and
change a real language model. The model is AnuLM: a mixture of experts with
397.7M parameters, trained from scratch on one consumer GPU, whose
architecture follows Sarvam AI's released 30B and 105B models, scaled down.
You will be able to:

- read `anulm/model.py`, the 1,364 lines that define it, and say what each
  part does and why it is there;
- trace one forward pass with the shape of every tensor, and recompute its
  parameter count by hand;
- follow its training run, its decoding loop and its server, and measure what
  the trained model actually does;
- make a change, run the test suite, and show that the change does what you
  meant and breaks nothing else.

The road starts with counting which byte follows which (Chapter 2) and ends
with the same model taking in speech and images (Part VIII).

**The ten parts.** Part I sets out the loop every language model runs and the
one objective it is trained on. Part II is just enough mathematics: shapes,
softmax, gradients and optimisers. Part III turns text into vectors. Part IV,
the core, opens the transformer one piece at a time: attention, masks, heads,
positions, normalisation, experts, then the whole model end to end. Part V
trains it; Part VI generates text with it; Part VII adapts it after
pre-training; Part VIII gives it inputs that are not text; Part IX takes it to
production; Part X, the capstone, has you train it, trace it and change it.
Chapter 1.6 has the same map as a table, with a route for each kind of reader.

**How each chapter works.** Most chapters take one idea through four steps. First the idea, in plain words. Then by hand, with numbers small
enough to check on paper: two-number vectors, three tokens. Then the real
code: the exact lines from the repository, each followed by a walkthrough with
the tensor shapes. Then a check: a short script, written for the book, that
runs the repository's code unchanged and asserts that it computes what the
page says. Each chapter ends with exercises and their answers, its key points,
a summary of about fifty words, and its references.

**How to read the evidence.** Every number carries a label in italics.
*verified* means checked against the code or recomputed by a book script;
*measured* means it comes from a recorded run or from a book script run on the
released model; *cited* means it comes from a checked paper; *illustrative*
means it was made up for teaching. A strip like this one opens every excerpt
of repository code:

> **In the repo** · `anulm/model.py` · `_sdpa` · lines 414–418 · @b42998d

It names the file, the function, the lines and the commit they are read at, so
you can open the same lines yourself (`git show b42998d:anulm/model.py`).
Boxes marked **Details** hold what a first reading can skip; boxes marked
**Build it** point to the lab book chapter that builds the same piece. Chapter
1.7 lists every label and badge, and Appendix F says how to reproduce every
number.

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

> **In the repo** · `anulm/model.py` · `_sdpa` · lines 414–418 · @b42998d

The line numbers are checked against commit `b42998d` by the book's own tools.
That commit is on the repository's
[`port-bundle-fixes`](https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM/tree/port-bundle-fixes) branch on GitHub, which holds the code in the
`anulm/` package. The default branch, `main`
([`ae59238`](https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM/tree/ae592387dc373cc6f579abce444c7f08c9e373a3)),
still has the old flat layout (`anulm/model.py` at the root), so its line numbers
differ: read the code at the pin, not at `main`. The lab book under `course/`,
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
