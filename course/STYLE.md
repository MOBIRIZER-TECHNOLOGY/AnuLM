# Style guide: *Build Your Own LLM, from Scratch to Omni*

The book's promise is that every idea is **seen, run and measured**: a figure
you understand at a glance, code you can run on your own machine, and the
real number we got when we ran it. This file is how we keep that promise
consistent across twelve chapters and several authors.

## Voice

- **Second person, present tense, plain words.** "You train a model that
  only counts" -- not "In this section, a count-based model will be trained."
- **One idea per paragraph.** If a paragraph needs two figures, it is two
  paragraphs.
- **Define every term the first time, in the sentence that uses it.** No
  glossary-hunting. *Loss* is "how surprised the model is by the real next
  byte, on average".
- **Numbers are real or labelled.** Every number in the text comes from a lab
  run (`results.json`) or from the AnuLM project's records (`docs/RESULTS.md`),
  and says which. Estimates say *about* and show the arithmetic.
- **Failures are content.** Each chapter has a *From the lab notebook* box
  with something that went wrong in the real project and how it was found.
- **Respect the reader's time.** No "as we all know", no hype words
  (*revolutionary, powerful, magic*), no apologies.

## The chapter template

Every chapter has these parts, in this order:

1. **Opening** (one short paragraph): what you will be able to do at the end,
   and why it matters.
2. **The idea**, in numbered sections, each built around a figure.
3. **Lab**: the command, how long it takes on a laptop CPU and on a GPU,
   and the output you should see, with our numbers to compare against.
4. **From the lab notebook**: one real story from the AnuLM project.
5. **Try it on your own data**: 3-5 exercises, the last one open-ended.
6. **Key takeaways**: 4-6 one-line bullets.

Callouts, used sparingly:

| callout | for |
|---|---|
| `> **Note**` | a detail worth knowing, safe to skip |
| `> **Watch out**` | a mistake that costs hours |
| `> **From the lab notebook**` | the real-project story (one per chapter) |

## Code

- **Every lab is one file**, runs with `python lab.py`, needs only PyTorch,
  and finishes on a laptop CPU in under ten minutes by default. Flags
  (`--data`, `--steps`) scale it up; the defaults never need a GPU.
- **Your own data first**: every lab takes `--data` and works on any text
  file, in any language. Defaults use public-domain or CC BY data, credited.
- **Fixed seeds**, so readers can match our numbers to the second decimal.
- **Comments explain why, not what**, in the same plain voice as the text.
- In the book, listings show the *important* 10-30 lines with the file and
  function named; the full file is in the repo.

## Visual language

The figures carry the book. Three kinds, and only three:

| kind | what it shows | example |
|---|---|---|
| **Concept** | an idea, drawn as objects and arrows | text flowing through a model to a probability for each next byte |
| **Anatomy** | how something is built, as stacked labelled blocks | a transformer layer; LoRA's frozen W beside its thin A and B |
| **Evidence** | a real measurement, drawn from a lab or project log | the loss curve; full fine-tune vs LoRA vs QLoRA |

### Palette

| name | hex | used for |
|---|---|---|
| ink | `#1F2433` | text, axes, outlines |
| paper | `#FBF8F3` | page and figure background |
| saffron | `#E8833A` | **the thing being trained or changed** -- the accent, one per figure |
| indigo | `#3B4CCA` | the model and its parts |
| teal | `#1F9E89` | data and inputs |
| rose | `#D1495B` | errors, loss, warnings |
| stone | `#8A8F98` | frozen, inactive, or background elements |
| mist | `#ECE7DF` | fills, gridlines, table stripes |

One rule does most of the work: **saffron always means "this is what
learns".** A frozen weight is stone; the LoRA adapter beside it is saffron.

### Type

- Body: **Source Serif 4**, 11 pt print / 18 px web.
- Figures, captions, tables: **Inter**.
- Code: **JetBrains Mono**.
- Hindi: **Noto Sans Devanagari** (figures) and **Noto Serif Devanagari** (body).

### Figure rules

- **Drawn by code**, from `course/figures/fig.py`, so all share one look and
  every evidence figure is regenerated from the lab's own output.
- SVG, 720 px wide, white space generous. **Maximum one accent colour.**
- **Every figure has a numbered caption that is a sentence with a verb**:
  "Figure 1.2: A table of counts already knows that *q* is followed by *u*."
- Label things directly; avoid legends where a label next to the line will do.
- No 3-D, no shadows, no gradients, no clip-art.

## Files

```
course/
  STYLE.md                 this file
  README.md                the course map
  figures/fig.py           the shared SVG figure library
  chNN-short-name/
    chapter.md             the chapter (Quarto-compatible Markdown)
    lab.py                 the lab
    make_figures.py        draws figures/ from the lab's results.json
    figures/*.svg
  build_preview.py         one HTML page per chapter, for review
```
