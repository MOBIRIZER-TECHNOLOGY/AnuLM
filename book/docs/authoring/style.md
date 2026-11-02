# Style and design guide

The look and the editorial habits of this book follow the reference deck
*The Illustrated Transformer in Practice* (the PDF the author supplied). Only its
**style and design** are borrowed. Its content, examples and numbers are not:
every fact in this book comes from this repository, from `docs/RESULTS.md`, or
from a cited paper.

This guide is separate from `course/docs/authoring/style.md`. The course is a hands-on lab
book with its own palette. This book is the reference companion, and it uses
the palette below.

---

## 1. Palette

Taken from the reference deck, and used for the same jobs.

| name | hex (approx.) | used for |
|---|---|---|
| **night** | `#13232A` | part-divider pages, "In one breath" boxes, dark callouts, code blocks |
| **ink** | `#17262E` | body headings and table header rows |
| **paper** | `#FFFFFF` | page background |
| **card** | `#F0F2F1` | light callout cards, "New words" cards |
| **slate** | `#56676C` | Part 0, secondary text, frozen or inactive parts in diagrams |
| **coral** | `#E5704F` | Part 1, the thing that goes wrong, the highlighted token |
| **gold** | `#D9A13B` (text on white: `#8F6A1E`) | Part 2, attention, the cell being computed |
| **teal** | `#2B8C82` | Part 3, data, inputs, "verified" marks |
| **plum** | `#7B5E8E` | Parts 4–6, advanced and extension material |

Each part keeps one accent colour, shown on its divider page, its section
chips and its diagrams. A diagram highlights **one** thing in the accent; the
rest stays slate or grey.

## 2. Type

- **Headings:** a bold serif (Lora, or Source Serif in print).
- **Body:** a humanist sans (Lato, or Source Sans).
- **Code and tensor shapes:** a monospace (JetBrains Mono or Consolas).
- Section numbers come first and stay in the heading: `2.4  Three jobs per token: Q, K and V`.

## 3. Page furniture

Every part, and every chapter inside it, uses the same furniture in the
same order.

1. **Part divider.** Dark page: `PART 2 · INSIDE THE MODEL`, a title, and one
   sentence on what the part opens up.
2. **New words for Part N.** Four to eight cards, each one term and a
   one-sentence definition. A term is defined the first time it is used, here
   or in the sentence that uses it.
3. **Numbered sections** (`2.1`, `2.2`, ...). Each one is built around a figure, a
   table or a worked example.
4. **By hand.** Worked examples with numbers small enough to check on paper:
   two-number vectors, three tokens. One running example is carried through a
   whole part (see §5).
5. **In the code.** The same idea in this repository: file, symbol, lines, a
   short excerpt, then a walkthrough (§6).
6. **Part N: key points.** Six bullets, plus an **In one breath** box (a dark
   card holding one paragraph that retells the part, in at most about fifty
   words).

Two asides may appear inside any section:

- **Details.** A blockquote that opens `> **Details.**` holds implementation
  detail a first reading can skip: a second call site, a rarely used branch,
  test tolerances, provenance. It is as rigorous as the main text and cited the
  same way, but the main text must read correctly without it. The HTML
  preview draws it as a collapsed box.
- **Build it.** A blockquote that opens `> **Build it.**` points to the lab
  book chapter that builds the same piece.

Notation is fixed for a whole part when the part opens (Part IV's is in
Chapter 11). A symbol never means two things in one part: matrices of vectors
are bold (**Q**, **K**, **V**), sizes are plain capitals (V is the vocabulary
size), and the experts active per token are A.

## 4. Badges

These are small pill labels at the bottom left of a section, as in the
reference deck.

| badge | meaning |
|---|---|
| `ANALOGY` | A teaching shortcut. The section says where the analogy stops working. |
| `RULE OF THUMB` | A heuristic. Test it on your own workload before relying on it. |
| `IN THE REPO` | The mechanism is implemented here. The section cites it. |
| `NOT IN THE REPO` | Background or an extension. Nothing in this repository implements it. |
| `EXTENSION` | A proposed change to this repository, drawn dashed in diagrams. |
| `IMPLEMENTATION-DEPENDENT` | The behaviour differs by engine, hardware or PyTorch version. |
| `MEASURED` | The number comes from a run recorded in `docs/RESULTS.md` or `logs/`, or from a book script run on the released checkpoint whose output is saved in `book/measurements/`. |

## 5. Running examples

The reference deck threads one real model (Qwen3-14B) and one tiny sentence
through every part. This book does the same with its own objects:

- **The real model:** `AnuLM-Base-v2-400M` (`ckpt_base_v2.pt`, model card
  `docs/MODEL_CARD_BASE_V2.md`). Every "at real scale" number is computed
  from its config.
- **The tiny model:** `tiny()` from `test_model.py`, which has small
  dimensions and runs in milliseconds. Shapes printed in the text come from
  running it, and the text says so.
- **By-hand numbers:** two-dimensional vectors chosen for the page. They are
  labelled *illustrative*, and the companion script recomputes them.

## 6. Code citations

Every implementation claim carries a reference in one fixed form:

> **In the repo** · `model.py` · `GQAttention.forward` · lines 426–465 · @773b6a8

- The path is relative to the repository root.
- The symbol is written `Class.method` or `function`.
- Lines are checked against the commit named after the `@`. If code moves,
  the reference is checked again before publication
  (`docs/reviews/source-verification.md`).
- Excerpts are the smallest span that shows the idea, usually 5 to 25 lines,
  and are copied verbatim. Any elided lines are marked `# ...`.
- Code written for the book, rather than taken from the repository, sits in
  `book/code-examples/` and is labelled **Book example, not repository code**.

## 6a. Research citations

External sources follow `docs/reference/research-references.md`, and only entries whose
status is **CHECKED** may appear in a chapter.

- **Form.** Cite by register ID right after the claim it supports:
  "Grouped-query attention shares one key/value head across a group of
  query heads [R12]." Cite the sentence, not the paragraph, unless every
  sentence in it rests on the same source.
- **Support, not keywords.** Cite a source only for what it was checked to
  support (its *Claims to support* cell). A paper whose title matches the
  topic is not evidence for a claim it doesn't make.
- **Never for the repository.** A paper never shows that a feature exists
  here. Pair the two explicitly: the mechanism from [R12], the
  implementation from the source map with an **In the repo** citation (§6),
  and any difference between them in words (the matrix in
  `docs/reference/research-references.md` §7).
- **Original and later work.** Name which is which: "introduced in [R15];
  the sigmoid router AnuLM uses comes from later work [R20]".
- **Every chapter ends with its references**: a list of the register IDs it
  cites, each with its status, which must be CHECKED at publication. The
  audit (`tools/research_audit.py`) reads the `[R..]` and `[S..]` markers
  and fails the book on any that is not CHECKED.

## 7. Evidence levels

The reference deck puts an evidence level on every source line. This book
uses four levels.

| level | means |
|---|---|
| **verified** | Checked against the repository at the commit given, or recomputed by the book's audit script. |
| **measured** | A number from a recorded run in `docs/RESULTS.md` or `logs/`, with the section given; or from a book script in `book/code-examples/` run on the released checkpoint, with its output saved in `book/measurements/` (the script, data, sample size and seed named on the page). A measurement on a sample is labelled as one. |
| **cited** | Taken from a named paper or official documentation whose register status is **CHECKED** (`docs/reference/research-references.md` §2). A reference still at register status CITED gives no evidence level at all. |
| **illustrative** | Made up for teaching, and labelled as such on the page. |

The deck also separates two kinds of claim, and this book keeps that split:

- **Arithmetic claims** are parameter counts, bytes and cache sizes. You can
  recompute every one, and the audit script does.
- **Quality claims** are about what helps or what hurts. Each one is either
  *measured here*, with its RESULTS.md section, or *not measured here*.

## 8. Voice

- Second person, present tense, plain words, and one idea per paragraph.
- Every number is real, measured or labelled illustrative. An estimate
  says *about* and shows its arithmetic.
- Failures are content. Where this repository found a bug the hard way,
  the book tells that story, and the code comments usually hold it.
- No hype words, no "as we all know", no filler.

## 9. Figures and diagrams

- **Designed figures are the default.** Every chapter carries figures drawn as
  SVG by `tools/make_figures.py`: diagrams of the mechanism and charts of its
  numbers. A chapter references one as
  `![Figure N.M · title](../figures/name.svg)`, followed by a caption paragraph
  that opens `*Figure N.M · title.*` and gives the figure's evidence level.
- **A figure's numbers come from the same source as the text**, and the
  generator asserts each against the value the chapter prints, so a figure
  cannot drift from its chapter.
- **Colour.** Magnitudes use one gold ramp, from faint to deep. Two series use
  coral and teal, validated for colour-blind readers: #C4532F and #0E9488 on
  light pages, #D9714F and #139C8F on dark. Text is never in a data colour.
  Every mark has a hover title, and every chart has a table or prose that
  carries the same numbers.
- **Mermaid** remains for quick flow sketches that need no numbers; a
  chapter's main diagram is a designed figure.
- A diagram shows only components that exist in the code. Anything else is
  drawn dashed and labelled `EXTENSION`.
- Tensor shapes go on the arrows as `(B, S, D)`.
