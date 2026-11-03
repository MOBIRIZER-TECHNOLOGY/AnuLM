## Appendices A–F

**Status:** reviewed (2026-11-01; independent technical review; all fixes applied by the writer and re-verified by the integrator)

The appendices follow `BOOK_PLAN.md`'s list (A shapes, B the 400M arithmetic,
C test index, D glossary, E bibliography, F reproducing every number). The
configuration reference the brief asked for is Appendix B.1. Files:

| Appendix | File | Title |
|---|---|---|
| A | `book/chapters/A1-shape-trace.md` | Tensor shapes for every module |
| B | `book/chapters/A2-parameter-arithmetic.md` | The 400M arithmetic (B.1 is the configuration reference) |
| C | `book/chapters/A3-test-index.md` | Test index |
| D | `book/chapters/A4-glossary.md` | Glossary |
| E | `book/chapters/A5-bibliography.md` | Bibliography |
| F | `book/chapters/A6-number-audit.md` | Reproducing every number |

`verify_chapters.py` globs `[0-9][0-9]-*.md` and does not read these files.
They contain no "In the repo" excerpts. The quoted-code checker used for
Chapter 41 found 0 problems on them. `book/code-examples/ch41_audit.py`
checks them against their sources and exits 0:

```
Appendix C: 66 rows against 66 tests at @773b6a8: ok
Appendix B.1: 42 configuration rows recomputed: ok
Appendix E: 66 rows against 66 register entries: ok
Appendix A: 47 distinct concrete shapes, 0 not in ch41_shapes.py's output
```

### Sources

- **A:** `code-examples/ch41_shapes.py`: forward hooks plus wrappers on
  `apply_rope` and `_sdpa`; `tiny()`, `tiny()` with MLA at
  `cache_holds_what_it_should`'s sizes, and `released()` with random weights on
  the CPU. The script asserts the cache shapes. The symbolic column was written
  by hand from the printed shapes and the code's `view`/`split` calls. The
  expert row's n_e depends on random routing (6 for `tiny()`'s expert 0,
  4 for the released config's expert 1). `model.py` line 841 for
  `F.embedding`.
- **B:** `code-examples/ch41_arithmetic.py` (reuses `trace_forward.arithmetic`)
  writes `book/measurements/part_x_arithmetic.json`. B.1 comes from `--config`:
  the declared defaults (`dataclasses.fields`, before `__post_init__`), field
  lines from `git show 773b6a8:model.py`, the `nano_30b()` and `nano_350m()`
  presets, and the `cfg` pickled in `ckpt_base_v2.pt`. The meaning column was
  written by hand from the code. File sizes come from `ckpt_base_v2.pt` (1,591,281,456 B)
  and `.last` (4,862,324,329 B), opened with `mmap=True`. The script asserts
  AdamW state = 8·397,659,648 + 4·1,513 B, and model bytes = 4 × state-dict
  numbers. 1,513 parameter tensors and 912 buffer numbers (19 routers ×
  `expert_bias` and `load_counts`, 24 each) were checked separately in Python.
  The `nano_350m` peak memory is *measured*, `docs/RESULTS.md` §6 (6.34 GB with
  checkpointing, 15.75 GB without), quoted in RESULTS's own units.
- **C:** names, `def` lines and order checked by `ch41_audit.py` against the
  AST of `test_model.py` at the pin. The "what it checks" column was written
  by reading each test's assertions (all 66 read), not docstrings. "*optional*"
  marks tests that return early when a package or `F.grouped_mm` is missing. The
  chapter numbers follow `BOOK_PLAN.md` and `REPOSITORY_SOURCE_MAP.md` §6, with
  the author's judgement for tests the map does not list.
- **D:** definitions reuse the "New words" cards of the chapters that exist
  (01, 04, 08, 11, 16, 17, 27, 28, 30–39 as written by other parts on
  2026-11-01) and Chapter 11/20's notation. Chapter numbers for terms whose
  chapter is not yet written follow `BOOK_PLAN.md`. "imbalance" follows
  `update_expert_biases` (max/mean, worst layer).
- **E:** generated from `RESEARCH_REFERENCES.md` (66 entries, all CHECKED on
  2026-11-01) by `ch41_audit.py --bibliography`; the audit fails if they diverge.
- **F:** tool descriptions from each tool's docstring and code (`build_preview.py`
  has `--print`; `verify_claims.py` and `make_figures.py` import the working
  tree's `model.py`). The `git archive` command was run here; the test run is
  in `book/measurements/part_x_tests.json`.

### Open questions for the integrator

1. **Appendix files are not rendered.** `build_preview.py` and
   `verify_chapters.py` read only `[0-9][0-9]-*.md`. Rendering the appendices
   needs a change to the shared tools, which this part may not edit.
2. **Naming.** The brief suggested A1–A6 file names with A3 as a configuration
   reference. `BOOK_PLAN.md` lists A–F differently (E is the bibliography), so
   the plan was followed and the configuration went into B.1. If the integrator
   prefers a separate configuration appendix, B.1 can move as it is.
3. **Chapter 20 is inaccurate on one point.** It says `released()` "match[es]
   every field of the configuration stored in `ckpt_base_v2.pt`". Six fields
   differ (`tokenizer_path`, `bias_update_rate`, `moe_impl`,
   `seq_balance_alpha`, `router_z_alpha`, `yarn_original_context`), none of
   which changes a shape or a count. The `trace_forward.py` docstring says the
   same. The fix is a wording change to Ch 20 or a change to `released()`;
   both are outside this part.
4. **The planned `by_hand.py`** ("recomputes every illustrative number",
   `BOOK_PLAN.md`) does not exist. Appendix F describes what does exist, and
   `ch41_audit.py` lists every *illustrative* paragraph with the script it names.
   On 2026-11-01 (chapters still being written), 29 of 37 such paragraphs named
   no script in the same paragraph (Chapters 01, 05, 09, 10, 11, 12, 14, 17, 18,
   25, 28, 32, 34, 40). Many may be recomputed by their chapter's example without
   naming it there. `ch41_audit.py --run` ran all 43 teaching examples then
   present: every one exited 0 (measurement scripts were not run).
5. `README.md` and `docs/TUTORIAL.md` say 55 tests; the pin has 66. And
   `docs/TUTORIAL.md` §3 says the bare `--preset 30b` command trains 1,000 steps,
   though the default is 2,000. Repository docs, not book files.

### Review fixes (2026-11-01)

All applied; none disputed. `verify_chapters.py` now reads `A[0-9]-*.md`, and
`verify_chapters.py 41 A1 A2 A3 A4 A5 A6` reports 0 problems. Open question 1
above (appendices not read by the tools) no longer applies.

- **A:**
  - The chunked-loss note now says logits are `None` only in training mode
    (`model.py` line 864).
  - A closing note says the audit's shape check is membership only, and that
    the rows were transcribed by hand.
- **B:**
  - The 15.75 GB is now attributed to RESULTS §6's table row for the 443M AdamW
    run without checkpointing. The conflict with §6's prose and with
    `nano_350m`'s docstring (`model.py` 175–176) is stated.
  - Units unified: exact bytes, then GB (10⁹, RESULTS' unit) with GiB beside.
  - `ch41_arithmetic.py` no longer needs `ckpt_base_v2.pt.last`: it prints
    B.2–B.4 and B.6 first, skips the B.5 file check when the files are absent,
    and writes `book/measurements` only with `--save`.
  - B.6 now gives the paged block for all 20 layers (327,680 B in bf16,
    491,520 B with fp32 keys), and the INT8 total includes the 456-number
    router bias.
- **C:**
  - Wording fixed for `causality_holds_under_gradients_too` (positions 0–5
    include position 5), `moe_with_all_experts_selected_is_dense` (normalised
    sigmoid weights, no shared expert) and the WSD test (flat over steps
    10–79).
  - The LoRA test's router-bias check is now described as conditional and never
    run. Confirmed: the test never calls `update_expert_biases`.
  - New C.4, "What the suite does not catch". I re-checked each gap on a pinned
    export, with `scratchpad/part_x/gaps.py`:
    - `initial_loss_is_ln_vocab` passes with embeddings at std 1.0;
    - `sliding_window_agrees_with_cache` passes with `_attn_mask` widened by one
      key;
    - `native_gqa_*` uses `n_kv_head=1` (line 1590);
    - the causal-mask test runs with `mask None` (Appendix A's trace);
    - early returns at 688, 709, 727, 888–891, 1282–1286 and 1342–1346.
- **D:**
  - YaRN now sharpens attention (mscale² > 1, a lower temperature).
  - fp32 now distinguishes `.pt` checkpoints from exported folders (bf16 large
    tensors, `export_hf.py` 271–279).
  - The buffer entry separates persistent buffers from non-persistent ones
    (the rotary tables).
  - D.2 adds d, γ, ε, w<sub>i</sub>, n<sub>e</sub>, n<sub>shared</sub>, r and
    d<sub>n</sub>/d<sub>r</sub>/d<sub>v</sub>, and a table of letters reused
    across parts: B and A (LoRA's bold **B** and **A**, Chapter 31), r (MLA
    latent width and LoRA rank), M (micro-batches in Chapters 21–22 against
    Chapter 11's bold mask **M**), V/**V**, and p.
- **F:**
  - `ae59238` now holds only the cited code files; the lab book is on
    `course-book` only.
  - F.2 updated: both tools now read appendices.
  - F.3 now has three kinds of example, and admits that some other parts'
    scripts write `book/measurements` on every run.
  - F.4 says exactly what the audit checks (three exact tables, and membership
    for A).
  - The illustrative definition no longer claims that every number is
    recomputed by a named script: 31 of 39 such paragraphs name none today.
  - The README-as-measured-source extension of STYLE §7 is flagged.
- **`ch41_audit.py --run`** now runs each example in a temporary directory,
  skips (and lists) any whose source names `ckpt_base_v2` or `measurements`,
  and exits 1 if an example fails.
  - Its run on 2026-11-01 found one failure outside Part X:
    `ch22_one_step.py` exits 1 from any directory other than the repository
    root. It loads `data/v2/meta.json` by a relative path (`train.py`
    `load_memmap`, line 157), so it also needs the v2 corpus. Reported to the
    integrator; Part X may not edit it.
  - The skip test is textual, so `trace_forward.py` and `attention_by_hand.py`
    are skipped because they mention the checkpoint.

Re-pinned to @fdbb09a on 2026-11-03: A3's Line column regenerated from the `@test` defs at the pin (all 76 had been at 084cbdd), `tiny()` line 44, PASS/SKIP/FAIL and `ModelTests` described, *optional* now means SKIP, `block_size_boundary` and `config_rejects_unreachable_topk` raise `ValueError`; C.5's three early-return rows replaced by a SKIP row (699, 720, 738, 1299, 1358) and the demo page's partial skip (901); new C.6 lists the three `tests/test_quality_*.py` files (19, 26, 18 tests). A2: config Line column recomputed, unchanged; `__post_init__` refuses with `ValueError`. A4: the pin's definition. A6: `CUDA_VISIBLE_DEVICES=-1`, the quality-suite command, timings, and a note that the B.1 audit line was not rerun (needs `ckpt_base_v2.pt`). Audit (C, E, A) rerun: ok; 32 of 39 illustrative paragraphs name no script, unchanged.

Revised 2026-11-03 (integration items #8, #18, #29, #30, #58): A6 F.3 gains one paragraph naming the four local-only sources (v2_train_run.log, logs/omni_one/6_train.log and results.json, release/AnuLM-Base-v2-400M/) and the book/measurements/ files that hold their numbers, with the new local_sources.json as the index; F.4 now describes ch41_audit.py --run's explicit SKIP list (with reasons) instead of the textual test, so trace_forward.py and attention_by_hand.py run.

Revised 2026-11-03 (citations R54–R63, S14–S17): A5's rows for R54–R63 and S14–S17 (added from the register) are now cited by chapters 2, 7, 8, 10, 23 and 40; no A5 text changed in this pass.
