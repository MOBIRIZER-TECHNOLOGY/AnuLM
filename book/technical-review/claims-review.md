# Claims review (Stage 5)

One section per chapter. A chapter counts as reviewed only when its section
says **Status:** reviewed. `tools/research_audit.py` reads this file and
reports which written chapters are still unreviewed.

Each review covers the five checks of the master prompt's quality control:
technical accuracy, source-code verification, educational, engineering and
editorial. It also covers the three claim checks `research-audit.md` lists:
- repository claims without a line citation;
- external claims without a CHECKED reference;
- quality claims without a RESULTS section or the words *not measured here*.

---

## Chapter 20 · The whole model, traced end to end

**Status:** reviewed (2026-10-31)

**File:** `chapters/20-the-whole-model.md`, against `@773b6a8`, `ckpt_base_v2.pt`
and `code-examples/trace_forward.py`.

### How it was reviewed

**Independent technical review.** A separate reviewer was given only:
- the chapter;
- the repository at the pin;
- the checkpoint;
- the register;
- the plan.

It was not given the author's notes. It reran the trace, recomputed every
number by hand, and diffed every excerpt against `git show 773b6a8:<file>`. It
read every cited test, and checked every chapter and course cross-reference.

**Author checks.**
- The chapter against `STYLE.md` and the master prompt's chapter requirements.
- A scripted check that every quoted code line lies inside its cited range at
  the pin: 9 excerpts, 0 mismatches.
- A scripted check that every printed-output line matches the trace output.
- The exercise answers, recomputed with `model.py`.

### Findings and what was done

| # | Kind | Finding | Resolution |
|---|---|---|---|
| 1 | error | "the assembled model in [course] Chapter 8": the course assembles and loads it in Ch 7 §7.4; Ch 8 is pre-training | fixed: Ch 4 skeleton, 5 positions, 6 attention, 7 experts and assembly (§7.4), 8 pre-training |
| 2 | error | "43.6% do any work" and "weight arithmetic of a 173M dense model": `num_params()` counts the whole input embedding table (33,554,432) as active | fixed: "active by `num_params()`"; about 140M (139,972,096) multiplied per token; the convention stated in 20.8 |
| 3 | error | "3.2% of expert choices": the comment says expert **sets**; cited range 580–586 included code | fixed: "the chosen expert set for 3.2% of tokens", lines 580–584, marked *not measured here* |
| 4 | error | "they compute the same thing", one test cited: grouped runs in bf16 and has its own test | fixed: both tests cited (463, 683) with their tolerances (10⁻⁵; 5% of the output's largest value) |
| 5 | excerpt | `GQAttention.forward` elided the `NATIVE_GQA` branch, so `repeat_interleave` read as unconditional; T undefined | fixed: branch shown, default named (line 366), T defined in the notation table and the text |
| 6 | excerpt | first `AnuLM.forward` excerpt cut the `embeds` branch without a marker | fixed: the full `if embeds … else` shown (836–844) |
| 7 | excerpt | the chunked-loss branch and the layer loop were hidden | fixed: the loop shown (845–856), the full tail shown (857–873), chunked loss explained |
| 8 | excerpt | `Router._gate` hid group-limited top-k without saying so | fixed: one sentence; `n_group = 1` in both models |
| 9 | improvement | the model never shifts targets | fixed: `make_batch` (`train.py` 199–208) cited; a common mistake added |
| 10 | improvement | "from the configuration" and "equals the state dict" were loose; internal process references | fixed: `released()` matches every checkpoint field; total = checkpoint tensors less 912 buffer numbers |
| 11 | improvement | "a matrix anywhere fails the suite" overstated the coverage; the exact sum is not in CI | fixed: limited to what `tiny_gqa`/`tiny_mla` build; the book script named as such |
| 12 | improvement | the windowed layers still cache every position, so 20 MiB needed a note | fixed: the mask limits reach, the cache never trims (441–447); Ch 27 and 29 |
| 13 | improvement | cache layout cross-referenced to Ch 13, where the plan covers it in Ch 27 | fixed |
| 14 | improvement | notation table missed I, I<sub>e</sub>, T; `k` meant two things; `Dh` vs D<sub>h</sub>; a parameter count in a shape diagram | fixed: table extended; K for experts per token, `k` for the key tensor; `Dh` named as the diagram spelling; diagram carries only shapes |
| 15 | improvement | "one batched matrix multiply" for a grouped GEMM with groups of different sizes | fixed |
| 16 | improvement | the shortcut skips B·(S − 1) rows, not S − 1, and matters for prefill and uncached decode | fixed |
| 17 | improvement | rotary tables are cached across calls (`Rotary._grow`), not computed once per call | fixed |
| 18 | editorial | stray badge; Stage references; "Chapters 11 to 19"; Ch 14, 16, 17 never cross-referenced | fixed |
| 19 | style (author) | no equations; the trace stopped at the loss; no common mistakes, debugging or exercises; badges not at section ends | fixed: 20.3 equations mapped to lines; 20.7 logits to a token; 20.11 common mistakes; 20.12 five exercises with answers; badges added |
| 20 | error (author) | the first draft of exercise 1 gave `tiny_gqa` H = 16, H<sub>kv</sub> = 4 from memory | fixed: H = 6, H<sub>kv</sub> = 2, read from the preset |

### Claim inventory after the fixes

- **Repository claims.** Each one carries an **In the repo** citation or a line
  number at `@773b6a8`. Every cited line and test name was re-read at the pin.
  No claim rests on a docstring alone.
- **External claims.** There are 7: [R04], [R06], [R08], [R11], [R12], [R18]
  and [R20]. All are CHECKED, and each is used within its register row.
- **Arithmetic claims.**
  - Every parameter count, share, byte count and exercise answer was
    recomputed, by `trace_forward.py` and by the reviewer by hand.
  - The trace's printed values: loss 4.1550 against ln 64 = 4.1589, on random
    ids with seed 0.
- **Quality claims.** There is one: bf16 routing changed 3.2% of expert sets.
  It comes from a code comment and is labelled *not measured here*. The chapter
  makes no other claim about what helps or hurts.

### Open items

- None for this chapter. Line numbers must be re-verified if `model.py`,
  `train.py` or `test_model.py` change before publication (Stage 7).

---

## Chapter 11 · Attention from first principles

**Status:** reviewed (2026-10-31)

**File:** `chapters/11-attention.md`, against `@773b6a8`, `ckpt_base_v2.pt`,
the installed PyTorch 2.11.0 and `code-examples/attention_by_hand.py`.

### How it was reviewed

**Independent technical review.** As for Chapter 20, the reviewer was given
only the chapter, the repository at the pin, the checkpoint, the register, the
plan, the style guide and the course. It also reviewed the companion script.

- It reran the script and recomputed by hand every number, derivation and
  exercise answer: the variance argument, the softmax derivative, the bound,
  FLOPs and bytes.
- It probed PyTorch's kernel choice and read the trained gains from the
  checkpoint.
- It ran the cited tests with `_attn_mask`'s output inverted.

**Author checks.**

- Every finding below was reproduced before it was fixed:
  - the kernel choice, by `torch._fused_sdp_choice` and the profiler;
  - the per-layer gains, from the checkpoint;
  - the three tests with the mask inverted: the causal test passes, the other
    two fail.
- The chapter was checked against `STYLE.md` and against the master prompt's
  attention mathematics (Phase 4) and progression (§10).
- All excerpts were re-checked against the pin, and every quoted number was
  checked against the script's output.

### Findings and what was done

| # | Kind | Finding | Resolution |
|---|---|---|---|
| 1 | error | "the optimised kernels are used only on CUDA" and "on a CPU every run goes through the reference implementation": PyTorch 2.11 picks a fused flash kernel on CPU unless dropout is on; a fourth backend, cuDNN, exists | fixed: the documentation's claim attributed to it [R50]; the observed behaviour stated separately and printed by the script (section 5); R50's notes record the discrepancy |
| 2 | error | `causal_mask_does_not_leak_future` offered as the catch for an inverted boolean mask: at its settings `_attn_mask` returns `None`, so it passes with the mask inverted | fixed: `incremental_forward_matches_full_forward` (550) and `sliding_window_limits_reach` (642) cited, both of which fail with it inverted |
| 3 | error | "every query and key head is passed through its own RMSNorm": one norm for queries and one for keys per layer, each gain shared by all heads of its kind | fixed (418–420 cited) |
| 4 | error | "the gains, which training changes slowly", and "QK-norm keeping the scores bounded": in the released checkpoint the per-layer limit runs from 7.1 to 225.6 | fixed: the trained gains and limits added (script section 3b); key point and In one breath reworded |
| 5 | error | "applying the scale twice … sharper": twice makes attention nearly uniform; the YaRN bug dropped one factor, not the scale | fixed |
| 6 | error | "256 random keys per query": the script uses one set of 256 keys for all queries | fixed wording; the conclusions do not change |
| 7 | error | the released-size layer used PyTorch's default init, not `AnuLM._init_weights` | fixed in the script; the length before QK-norm becomes 5.08, the largest score 4.338 |
| 8 | error | "largest scaled score" was the largest absolute value over all pairs, unmasked | fixed: label states it |
| 9 | citation | "kept only the attention [R01]" claimed more than R01's register row | fixed: "built on attention in the scaled dot-product form [R01]" |
| 10 | improvement | the released model's sliding window: layers 0–9 always take an explicit mask, and the window saves no compute here | added to 11.5 and 11.7 |
| 11 | improvement | `bidir_mask`'s (B, 1, 1, T) shape omitted | added |
| 12 | improvement | "softmax nearly one-hot and its gradients vanish" overstated | fixed: often nearly one-hot; the gradient through the scores vanishes, the dominant value still gets its gradient |
| 13 | improvement | "a width-64 head" read as a statement about the model | fixed: "a random width-64 head"; table headers say "mean" |
| 14 | improvement | notation: D, D<sub>h</sub> versus d versus d<sub>k</sub>, B and H used before they are defined; diagram T = S | fixed |
| 15 | improvement | "a layer holds H·S·T scores" contradicted "if stored" | fixed: "computes"; the causal half that a kernel may skip noted |
| 16 | improvement | a comparison with an outside model's code uncited; RMSNorm's own chapter not referenced | fixed: the line-436 comment quoted; Chapter 15 added |
| 17 | improvement | "kernel", "bf16" used without definition | defined |
| 18 | improvement | the `NATIVE_GQA` call site omitted | one sentence added (366) |
| 19 | improvement | script printed an unused line; unmasked scores not labelled | fixed |
| 20 | style (author) | ANALOGY badge mid-section; no mapping of W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> and the output projection to code; no pointer to the real-shape trace | fixed: badge at the section's end; `query_key_value` (429) and `dense` (465) mapped; Chapter 20 §20.5 referenced |

### Claim inventory after the fixes

- **Repository claims.** Each one carries an **In the repo** citation or a line
  number at `@773b6a8`, re-read at the pin.
- **External claims.** There are 5: [R01], [R02], [R14], [R29] and [R50]. All
  are CHECKED and used within their rows. R01's and R50's rows were extended
  on 2026-10-31 for this chapter, from the paper's full text and the installed
  docstring.
- **Arithmetic and numbers.**
  - The toy, the √d table, the released-size layer, the trained limits, the
    costs and the kernel choices all come from the companion script.
  - The toy and the random-vector results are labelled *illustrative*.
  - The trained gains and the kernel choice are labelled *verified*, from the
    checkpoint and from this machine's PyTorch.
- **Quality claims.** There are none about what helps or hurts training. The
  two such sentences in the draft were removed: "less stable attention" and
  "training changes the gains slowly". The random-vector table explicitly says
  nothing about trained attention.

### Open items

- The kernel behaviour is specific to PyTorch 2.11.0 on this machine. Re-run
  `attention_by_hand.py` section 5 before publication (Stage 7).

---

## Revision of Chapters 11 and 20 after the reader review (2026-10-31)

Both chapters were revised after their reviews above. A chapter counts as
reviewed again only after its revision has been checked, so this revision is
recorded here with its own checks.

### What prompted it

- **Simulated target-reader review.** An AI reviewer role-played a software
  engineer new to transformers. It read Ch 11 then Ch 20 in order, ran both
  companion scripts as instructed and did every exercise. It reported where it
  got lost, terms used before they were explained, padding, and exercises that
  needed what the text did not teach. It is a simulation;
  `REVIEWER_GUIDE.md` sets out the human reviews still needed.
- **Measurements of the trained model.** The new script
  `code-examples/measure_attention.py` measures, on 64 windows of 512 held-out
  tokens:
  - what every head attends to;
  - the loss;
  - how evenly the experts are used.

  Ch 11 gained a section on it (11.7), and Ch 20 gained the real-text loss and
  the expert loads.

### What changed

**Notation, settled book-wide** (STYLE.md):
- bold **Q**, **K**, **V** for matrices; V is the vocabulary size;
- A for the experts active per token (was K);
- γ for RMSNorm gains, w for router weights, P for the graded positions.

The notation table moved to the start of Part IV, in Ch 11; Ch 20 repeats it
with both models' values.

**Ch 11:**
- why d = 64 when D = 1,024 (16 heads);
- the output is added to the residual stream, not swapped in;
- S and T defined before the formula;
- the geometry of the toy's largest score;
- the √d table explained before it is shown, with a √d column;
- δ<sub>ij</sub> defined;
- an unsupported sentence removed;
- the `is_causal` paragraph split into the common case and three departures;
- side material moved into Details boxes;
- the RMS-to-length step written out;
- a "why this matters" paragraph linking QK-norm back to 11.4;
- the measured section 11.7;
- Exercise 2 replaced (its answer was in the text);
- Exercise 6 shows the √2 step and its side effect;
- the script's sections are labelled with chapter sections, and the profiler
  warning is silenced.

**Ch 20:**
- the 20.3 equations get a "where" list, with u and w<sub>i</sub> defined;
- three code idioms explained: `h, aux = …`, the `transpose`/`view` that
  reassembles heads, and `v[:, [-1]]`;
- the regularisers named;
- autocast, buffers, presets, `block_size` and forward hooks defined;
- reference material moved into Details boxes (merge tree, `embeds`, dispatch
  paths and their tests);
- the test-coverage analysis cut to two sentences;
- "no biases" and the SwiGLU shapes stated before the parameter table;
- the trace script's label fixed to (N, A);
- Exercise 1 gives the command that prints every term;
- the real-text loss and the expert loads added.

### Technical check of the revision

An independent reviewer checked only the new and rewritten text, and the new
script. It reran the script with an extra check: the recomputed weights times
the values match `_sdpa`'s output to 3.1·10⁻⁵ on every call, and the call
counter matched the real layer index on all 320 calls.

It found 5 errors and 10 improvements. All were reproduced, then fixed:

| # | Kind | Finding | Resolution |
|---|---|---|---|
| 1 | error | "§39 records 3.3523 on the whole held-out split": that figure is from 400 evenly spaced 1,024-token windows, in bf16, ± 0.0491 (`experiments/v2_train.sh`, the model card) | fixed in both chapters, with the settings, and a note that the two losses are not comparable |
| 2 | error | "the windowed layers show almost none of this": on the queries that can still see position 0, layers 8 and 9 put 0.263 and 0.158 there | the script now records the first-position weight where visible; the text says the pattern starts at layer 8 |
| 3 | error | "20 KiB per token in bf16" presented as what the code stores: under the repository's bf16 autocast, the cache holds fp32 keys (QK-norm's fp32 gain) and bf16 values | fixed: 20 KiB if all bf16, 30 KiB as served, 40 KiB in fp32; dtypes checked; Exercise 3 says "if stored in bf16"; the trace script's label says the same |
| 4 | error | "`_attn_mask` builds an explicit mask in these cases" listed the single query, where it returns `None`, and missed several tokens against a cache | fixed: three departures, each with what happens |
| 5 | error | "the first elided stretch … builds the mask": the mask is built in the second, after T is known | fixed |
| 6–15 | improvement | layers 10–18 "a handful of keys, mostly the first position" overstated; autocast described too broadly; "16 × 64 = 1,024" true for queries only; layer 0's small gains worth a sentence; script robustness (assert the recompute, assert the call count, remove the router hooks before the example sentence, state fp32); why 32,732 tokens; "the next layer can see two tokens" was interpretation; "look back at most 256" imprecise; the grouped test passes silently without `F.grouped_mm`; the router weights sum to 2.5 | all fixed; the script reran, and every number in both chapters was re-checked against its output |

### Checks after the revision

- `tools/verify_chapters.py`: 0 problems in either chapter.
- Every Ch 20 output block matches `trace_forward.py`; every quoted number in
  both chapters matches `attention_by_hand.py` or `measurements/`.
- `tools/research_audit.py`: all citations resolve to CHECKED references.

**Status:** Chapters 11 and 20 remain reviewed, including this revision.

### Code-first rewrite (recorded 2026-11-03, integration item #12)

After the reviews and the revision above, Chapters 11 and 20 were rewritten
code-first, like the rest of the book: each idea is now taught from a
line-cited excerpt pinned to the code. The Stage 5 reviews in this file
checked the earlier text. The rewritten sections have not been re-reviewed;
this record does not re-review them. **Open:** a Stage 5 check of the
rewritten sections of Chapters 11 and 20.

Revised 2026-11-03 (integration items #4/#47): in Ch 11, the caption of
Figure 11.2, the √d paragraph and the caption of Figure 11.4 now name
`attention_by_hand.py`, which recomputes their *illustrative* numbers. (Item
#6: Ch 11 at the pin no longer quotes the bf16 router flip, so nothing there
needed citing; Ch 20 already cites Ch 23's 3.17% and was not changed.)

---

## Figures and front matter (2026-10-31)

**Added:** seven figures, generated by `tools/make_figures.py`; the front
matter (`FRONT_MATTER.md`), which gives the author, the GitHub repository and
the model on Hugging Face; and GitHub links on every code citation in the
preview.

**Ch 11 figures:**
- 11.1, one attention row by hand;
- 11.2, the toy's weight grid;
- 11.3, keys in play with and without √d;
- 11.4, the trained heads' previous-token and first-position weights.

**Ch 20 figures:** 20.1, the whole model (it replaces the Mermaid sketch); 20.2,
where the parameters are; 20.3, the expert loads.

**How the figures were checked:**
- Every value a figure shows is recomputed from the chapter's own source:
  - the toy's matrices;
  - the seeded √d draw;
  - `measurements/attention.json`;
  - the eight-term sum, which is checked against `num_params()`.
- The generator asserts the key numbers against what the chapter prints:
  0.620, 1.546, 2.5, 157.0, 0.881, 0.860, 397,659,648, 173,526,528, 0.084 and
  0.095.
- The two-series colours passed the colour-blind validator in both themes;
  the book's original teal failed its chroma floor and was replaced for charts.
- Each figure was rendered with headless Chrome and inspected. Four defects
  were fixed: an overlap in 11.1, a washed-out ramp in 11.4, a clipped label
  in 11.2, and an arrow crossing a box in 20.1.

**How the links were checked:**
- All are live (HTTP 200): the GitHub repository, the public commit's file
  URLs, `huggingface.co/toonist` and `huggingface.co/toonist/AnuLM-Base-v2-400M`.
- The pinned commit 773b6a8 is not on GitHub. The public commit ae59238 has
  every cited code file byte-identical (`git diff` is empty outside `course/`
  and an editor setting). The preview re-checks this per file before linking.
- The author line is "The AnuLM authors", as in `LICENSE` and
  `CITATION.cff`; no personal name is given anywhere in the repository.

**Open:** the book's own text licence is not chosen.

## Re-review of the code-first rewrites (2026-11-03)

An independent technical reviewer read Chapters 11 and 20 in full at @fdbb09a
(integration item 12), re-ran their CPU examples, and recomputed the worked
numbers, tables and parameter counts; all matched. It found 16 issues (4 wrong,
4 misleading, 8 unclear or style), none changing what the chapters teach. All
were fixed in the proofreading follow-up: Ch 11 items 9, 11, 16, 17, 22-24, 27
(including Cauchy-Schwarz in the score bound, "almost no gradient", "checks"
not "proves", the 215/288 visible-key means, `# ...` elision marks); Ch 20
items 10, 14, 15, 18, 25, 26, 28 (the untied head at 841-843 and 117, the
gain-vector widths per preset, the ln V noise note, hook return value, tracing
of plain functions). Verdict after fixes: both chapters ready.
