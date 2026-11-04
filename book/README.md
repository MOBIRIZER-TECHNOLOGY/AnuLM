# The book: a repository-driven reference on building an LLM

The title, fixed in Stage 2, is
***Inside AnuLM: A Language Model You Can Read, Line by Line***.

This folder holds a technical reference book built on this repository's
own from-scratch code. Its visual and editorial style follows the supplied
deck *The Illustrated Transformer in Practice*; see `STYLE.md`. Nothing
outside `book/` is modified.

**Pin.** Every line number refers to commit `b42998d` (the `anulm/` package
layout). The book was moved there from `fdbb09a` on 2026-11-04 by
`tools/repin.py` (and to `fdbb09a` from `773b6a8` on 2026-11-03);
`technical-review/repin-report.md` lists every citation the last re-pin moved,
and `technical-review/repin-overrides.tsv` the decisions a person made. To re-pin after the code changes again:

```bash
python book/tools/repin.py b42998d NEWCOMMIT            # dry run: the report only
python book/tools/repin.py b42998d NEWCOMMIT --write    # then fix every "Needs a person" row
python book/tools/verify_chapters.py && python book/tools/verify_source_map.py && python book/tools/verify_claims.py
```

`preview/` is build output (`tools/build_preview.py`) and is not committed.

## How it relates to `course/`

The repository already contains a book, `course/` (*Build Your Own LLM, from
Scratch to Omni*), with 20 chapters drafted. That one is a **lab book**: the
reader fills in stubs, and checks hold the stubs to `anulm/model.py`.

This one is the **reference companion**. It covers the theory, the
mathematics and line-cited walkthroughs of the real code, in a depth that a lab
chapter has no room for. The two books should cross-link rather than repeat
each other. The decision, option A, is recorded at the end of this file.

## Deliverables so far

| File | Contents |
|---|---|
| `REPOSITORY_ARCHITECTURE.md` | What the model is, the released configuration, the repository layout, the training and inference execution paths (with Mermaid), engineering features, and what is **not** implemented |
| `REPOSITORY_SOURCE_MAP.md` | Component → file → symbol → lines → status → verification, for about 120 components; the tests it cites mapped to concepts |
| `BOOK_PLAN.md` | Stage 2 blueprint: audience, prerequisites, 41-chapter TOC with repo, test and RESULTS mapping, dependencies, diagram plan, capstone, writing order |
| `RESEARCH_REFERENCES.md` | The external-source register (requirements 16–22): two sources of truth, the CITED / CHECKED / CORRECTED / REJECTED lifecycle, the verification procedure, 53 inventory and 13 supplementary references, a corrections log, and the research-and-implementation matrix |
| `tools/research_audit.py` | Counts references by status, unresolved identifiers, chapter citations of unchecked references, and which written chapters have a Stage 5 review; writes `technical-review/research-audit.md` |
| `chapters/` | The chapters, one Markdown file each, written in Stage 4 (so far: Ch 11, Ch 20) |
| `code-examples/` | Book examples, not repository code: scripts that run `anulm/model.py` unchanged and print what a chapter shows (so far: `trace_forward.py`, `attention_by_hand.py`, `measure_attention.py`) |
| `tools/verify_chapters.py` | Checks every chapter against the pinned code: excerpts inside their cited ranges, `def` lines, and STYLE.md's page furniture; writes `technical-review/chapter-verification.md` |
| `tools/build_preview.py` | Renders chapters as one styled HTML page in the book's design (`preview/index.html`) |
| `measurements/` | Output of book scripts run on the released checkpoint (attention statistics, held-out loss, expert loads); the source for *measured* numbers that are not in `docs/RESULTS.md` |
| `technical-review/REVIEWER_GUIDE.md` | What to send human reviewers (an expert and a target reader), what to ask, and a feedback table |
| `FRONT_MATTER.md` | Title page: author, the GitHub repository, the model on Hugging Face, licences, how to cite, and how the pinned commit relates to the public one |
| `tools/make_figures.py` | Draws the chapters' figures as SVG (`figures/`) from the same data the text quotes, asserting each number against the chapter |
| `figures/` | The figures, one SVG each; the preview inlines them |
| `STYLE.md` | Palette, typography, page furniture, badges, code-citation format and evidence levels, adapted from the reference deck |

## Initial technical assessment

**What the repository is.**

- A decoder-only sparse-MoE transformer written from scratch in PyTorch:
  `anulm/model.py`, 1,224 lines.
- It follows the design of Sarvam 30B and 105B, which descend from DeepSeek.
- Released models: 400M total / about 173M active parameters, a 32k byte-level
  BPE, RoPE at θ = 1e6, and GQA 16:4 with QK-norm.
- Sliding-window attention on half the layers, SwiGLU, and 24 experts with
  top-4 routing behind a sigmoid router with aux-loss-free balancing.
- The training path needs only `torch`.

**Why it suits a book.**

1. **Small enough to read whole, real enough to matter.** Every component a
   modern open model has is here and runs on one consumer GPU.
2. **The code explains itself.** Comments record *why* and *what broke*. Two
   examples: the router is computed in fp32 because bf16 flipped 3.2% of
   expert choices (`anulm/model.py` lines 615–621), and atomic saves exist because
   a kill at step 265,000 destroyed a checkpoint (`anulm/training/train.py` lines 32–46). These
   comments give the book its "failures are content" material.
3. **Claims are testable.** 66 named tests pin the mechanisms: causal mask,
   RoPE relativity, MoE equivalence, cache equivalence, LoRA merge, and more.
   The book can quote a test as the proof of each claim.
4. **A measured record.** `docs/RESULTS.md` has 39+ sections of A/B
   results, so quality claims can be marked *measured here* rather than
   asserted.

**Where the book must be careful.**

- **The textbook path is missing.** The prompt's outline assumes a GPT-2
  model: absolute positional embeddings, LayerNorm, plain MHA and a dense
  FFN. None of these is implemented. Each one is taught as history or as a
  special case. MHA is GQA with `n_kv_head == n_head`, a dense FFN is the
  layer-0 `SwiGLU`, and sinusoidal and learned positions get history
  sections marked `NOT IN THE REPO`.
- **MoE and MLA make the model harder than the textbook one.** Part IV needs
  a dense-path walkthrough first (layer 0, GQA) before routing and the latent
  cache.
- **Features to mark absent:**
  - top-p sampling (`anulm/inference/api.py` accepts the argument and ignores it)
  - beam search and speculative decoding
  - paged KV cache and continuous batching
  - weight quantisation
  - distributed training
  - DPO, RLHF and GRPO (DPO and GRPO are written from scratch in `course/`
    Chapters 11 and 19, not in the repository root)
  - LayerNorm
  - dropout: wired in but off at 0.0
- **Two copies of the code.** `.claude/worktrees/api-demo/` has a second
  copy with different line numbers. Cite only the root copy.
- **Code is still moving.** The working tree has uncommitted edits to 8 files,
  none of them in the core four (`anulm/model.py`, `anulm/training/train.py`, `anulm/bpe.py`,
  `anulm/inference/sample.py`). Line references are pinned to `@b42998d` and must be
  checked again before publication.
- **Stale docstrings.** Some docstrings contradict the code (see the source
  map, §5). The book quotes code, not docstrings.

## Identified gaps (implementation vs. the requested outline)

| Requested topic | In the repository | How the book handles it |
|---|---|---|
| Learned or sinusoidal positions | No (RoPE only) | History sections; a book example in `code-examples/` |
| LayerNorm | No (RMSNorm) | Compare it with RMSNorm, then show the repo's `RMSNorm` |
| Plain MHA / MQA | As GQA config cases | Teach GQA as the general case |
| GELU 4× FFN | No (SwiGLU) | History, then SwiGLU |
| Weight tying | Flag only, off | Explain the flag and why both Sarvam and AnuLM untie |
| Top-p | No | Teach it, mark it absent, and give a 6-line `EXTENSION` patch |
| KV caching | **Yes** | Full chapter |
| Mixed precision | bf16 autocast | Full section; fp16 + GradScaler marked absent |
| Gradient accumulation | **Yes**, with a test | Full section |
| Distributed training | No | Part IX, marked `NOT IN THE REPO` |
| Quantisation | No | Part IX, marked `NOT IN THE REPO` |
| LoRA | **Yes** | Full chapter |
| Instruction tuning | **Yes** (SFT with prompt masking) | Full chapter |
| Preference optimisation | No (course stub only) | Extension chapter, cross-linked to `course/ch11` |
| Evaluation | **Yes** (val SE, golden sets, benchmarks) | Full chapter |
| Multimodal | **Yes** (frozen encoders + projectors) | Advanced part, beyond the requested outline |

## Progress tracker

| Stage | Status |
|---|---|
| 1 · Repository discovery | **done**: core files read in full; 25 peripheral files inventoried; 14 of 14 spot-checks matched |
| 2 · Book blueprint | **done**: `BOOK_PLAN.md` (41 chapters in 10 parts, file and test map, 17 diagrams, capstone, writing order); `RESEARCH_REFERENCES.md` (49 references, all at *cited*) |
| 3 · Technical validation of the blueprint | **in progress.** Research half **done** (2026-10-31): all 66 references opened at their sources (arXiv API and full texts, Crossref, the originals) and CHECKED; 3 metadata corrections (R36's volume, R37 and S05's DOIs), 6 claims narrowed, R15's unconfirmed venue dropped; PyTorch docs pinned to 2.11; Sarvam provenance traced to the upstream revisions (the local 105B modelling file predates Sarvam's 2026-09-21 routing change); three historical orders corrected in `BOOK_PLAN.md`. Audit: references verified, research-complete **no** until the Stage 5 claim review. Repository half **done** (2026-10-31): `tools/verify_source_map.py` checked all 94 source-map rows and 61 test-map entries at `@b42998d` with Python's parser; `tools/verify_claims.py` recomputed 23 claims from the code and `ckpt_base_v2.pt` (all hold), including that every file and test the plan names exists. Fixed: two `anulm/multimodal/omni_train.py` rows that cited uncommitted code, one range 3 lines long, two rows naming a method for a class range, and a shared-expert statement (the released model has none). **Stage 3 is complete**; Stage 4 may start |
| 4 · Chapter authoring | **in progress**: 2 of 41 written (Ch 20; Ch 11, "Attention from first principles", 2026-10-31, with `code-examples/attention_by_hand.py`). Next in the writing order: Ch 12–13 |
| 5 · Independent review | **in progress**, chapter by chapter: 2 of 2 written chapters reviewed (Ch 11, Ch 20), each by an independent technical review, then revised after a simulated target-reader review and re-checked (record: `technical-review/claims-review.md`). The trained model is now measured where a chapter explains a behaviour (`measurements/`). Styled preview: `tools/build_preview.py`, published as an artifact. **Not yet done: human reviews** by an expert and a target reader (`technical-review/REVIEWER_GUIDE.md`) |
| 6 · Integration | pending |
| 7 · Publication preparation | pending |

**Not yet inspected:**

- `experiments/` (about 40 scripts)
- `eval_*.py`
- `app*.py`
- `anulm/multimodal/speech_data.py`, `anulm/inference/voice.py`
- `web/`, `space/`
- `sarvam/` annotated references
- `docs/RESULTS.md`, beyond its index

These are needed only for the chapters that cite them, and will be read
then.

## Decision (2026-10-31)

**A: reference companion.** This is a separate book. It cross-links the
course labs ("Build it" boxes) and does not repeat them.
