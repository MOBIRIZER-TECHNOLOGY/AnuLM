# Inside AnuLM: publication and production-readiness review

> Historical review of the supplied draft before the usability revision. See [the usability update](USABILITY_UPDATE_2026-11-01.md) for fixes and remaining launch gates.

Reviewed 1 November 2026. Scope: the manuscript supplied in this folder.

## Verdict

**Not ready for a final publication release, and not sufficient to promise that readers will build a production-ready LLM service.** It is a substantial technical reference draft with useful explanations, traceable implementation excerpts, numerical examples, exercises, and unusually candid discussion of limitations. Its current title, *Inside AnuLM: A language model you can read, line by line*, fits the material better than a promise of production readiness.

There are two different release decisions:

| Decision | Assessment |
|---|---|
| Share the Markdown as a clearly labelled technical preview | Reasonable, with prerequisites and known limitations stated |
| Publish the supplied PDF/HTML as the complete book | Blocked: the editions contain different, incomplete chapter selections |
| Publish a finished technical reference | Blocked by reproducibility, review, and integration issues below |
| Present the implementation as a production-ready service | Blocked by demonstrated serving bugs and missing operational validation |

Single-GPU training and the absence of quantisation, speculative decoding, or distributed training do **not**, by themselves, prevent production use. The decisive problems are correctness, a reproducible supported release, and evidence that the service meets its intended users' requirements.

## What I checked

- Inventoried all 41 chapters, six appendices, 66 example scripts, and 98 SVG figures. The chapter and appendix sources contain approximately 146,000 whitespace-separated words, including code and tables.
- Parsed all 66 example scripts with Python's syntax parser: no syntax errors.
- Applied the supplied chapter verifier to all 47 chapter/appendix files, substituting downloaded files at public revision `ae592387dc373cc6f579abce444c7f08c9e373a3` for unavailable local Git objects. Of 347 repository excerpts, 344 had available public source and no quoted-line mismatches were reported; three depend on an unavailable course file. The verifier checked 3,637 quoted source lines. This checks excerpts and specified structure, not every explanatory claim or every formula.
- Focused manual review on the introduction, data, training/evaluation/resume path, sampling, SFT, serving, export, production gap, capstone, and reproduction appendix; inspected the review and measurement records. This was not an exhaustive sentence-by-sentence mathematical review of all 146,000 words.
- Extracted the complete 45-page book PDF's text and visually sampled five pages. The PDF contains Chapters 11 and 20. Inspected the chapter headings in both HTML editions.
- Ran the forward-trace and top-p examples successfully against the public source in an isolated folder. The trace reproduced 397,659,648 total and 173,526,528 active parameters. The top-p example's correct implementation passed its three tests and its deliberately incorrect variants failed as intended.
- Ran the public CPU suite on Python 3.14 and installed PyTorch 2.14.0+cu130. It reported 65 passed and one missing-file failure in my initially partial source download; two of the reported passes were explicit Gradio skips. After retrieving the remaining supporting files, the failed export-contract test passed when rerun directly. Thus 64 tests executed successfully across these runs, with two UI checks skipped; this is not a claim of a fresh, complete 66-test run in the book's PyTorch 2.11 environment.
- Independently reproduced both streaming faults described in Chapter 38 and tested the literal repair suggested in its prose using the public `Generation` class and a scripted byte emitter.

I did not retrain the released model, download its large native checkpoint or validation corpus, reproduce all recorded GPU measurements, or perform a live HTTP load/security test. Existing measurement files are evidence supplied with the manuscript, not results independently regenerated in this review.

## Findings, in priority order

### 1. Release blocker: the distributed editions are incomplete and inconsistent

**Location:** `preview/inside-anulm.pdf`, `preview/print.html`, `preview/index.html`, `README.md`.

The PDF's cover explicitly says “Chapters 11, 20 of 41.” Its 45 pages are not the full 41-chapter manuscript. The print HTML likewise contains those two chapters. The other HTML preview contains Chapter 41 and Appendices A and B. The README still describes authoring progress as two chapters, although all 41 are present.

**Impact:** a reader receiving the PDF or opening the preview cannot read the book that the source folder promises. Review status and reading order are ambiguous.

**Required change:** build a complete edition from a declared chapter manifest; include the front matter and all six appendices; generate the contents and navigation from that same manifest; refresh the README and edition date. Verify the complete rendered book before release. The sampled older PDF pages are generally orderly; they cannot establish layout quality for the missing chapters.

### 2. Release blocker: the cited revision and course material are not reproducibly available

**Location:** `docs/reader/front-matter.md`, Appendix F.1, Chapter 34's three course excerpts, and the “Build it” boxes.

The front matter explicitly says pin `773b6a8` and branch `course-book` are unpublished. Appendix F nevertheless instructs readers to run `git archive 773b6a8`. The public replacement revision supplies the main model source but not `course/build/solutions/ch11.py`; retrieving that path at the stated public revision returned HTTP 404. Those are the three excerpts unavailable to this review's source check.

**Impact:** public readers cannot follow the exact reproduction command or several lab links. The public-code equivalence workaround does not supply missing course material.

**Required change:** release an immutable, publicly accessible version containing every required lab and supporting file, or remove/replace unavailable dependencies. Make clone/download instructions and every verifier use that release. Test from a fresh download.

### 3. High priority: the “production-ready” promise exceeds the implemented result

**Location:** Chapters 38.7 and 40.7, and the capstone.

The manuscript explicitly documents a single-request device lock, ignored `top_p`, ignored earlier conversation turns, optional authentication controlled by an environment variable, and absent rate limits and metrics. The capstone ends in training, tracing, a sampling extension, and model tests. It does not deliver or validate an operational service.

**Impact:** readers may confuse a working model/server with a service ready for their production workload.

**Required change:** retain the narrower reference-book positioning, or add an implemented deployment capstone with a defined use case, request/queue limits, timeouts and cancellation, authentication configuration, metrics, load tests, acceptance thresholds, and a rollback procedure. Features such as batching should be justified by workload measurements rather than treated as universal production requirements.

### 4. High priority: Chapter 38's proposed streaming repair is incomplete

**Location:** Chapter 38.5, paragraph beginning “One fix is to call `self._release(final=True)`”.

Independent reproduction with bytes for `Hello`, then EOS, and stop list `["###"]`:

| Version | Text sent to the streaming client | Returned text |
|---|---|---|
| Public implementation | `Hel` | `Hel` |
| Literal suggested change: call `_release(final=True)` on EOS | `Hel` | `Hello` |

`_release` updates internal state and returns a delta; it does not emit that delta. Calling it alone still loses the final streaming characters. I also reproduced `Hi �` when a token limit cuts a UTF-8 character.

**Required change:** explain and demonstrate a repair that emits the final delta, distinguishes EOS from an actual matched stop sequence, and handles an incomplete final UTF-8 sequence. Add assertions for both streamed and non-streamed results. The book correctly identifies the original bugs; the problem is the completeness of its repair advice.

### 5. High priority: the API silently changes the meaning of accepted requests

**Location:** Chapter 38.4 and 38.7; public `api.py`.

The manuscript documents that `top_p` is accepted but never reaches generation, earlier turns are ignored, and temperature is clamped above zero despite a greedy-mode description.

**Impact:** apparently valid client requests can produce behaviour different from what the caller requested. A compatibility label needs a precise supported subset.

**Required change:** implement supported options or reject unsupported ones with an explicit error. Add request-to-generation contract tests, including a multi-turn request, nucleus sampling, and zero-temperature behaviour. If preserving the historical code for teaching, keep its limitations prominent and supply a separately tested corrected serving path.

### 6. High priority: the chat held-out set is contaminated

**Location:** Chapter 30.5, `make_chat.py`, `measurements/part_vii_chat_split.json`.

The supplied measurement record reports four of 300 held-out questions also appearing in training, including two with the same answer. I verified that the public split code selects held-out objects without first deduplicating or grouping equivalent questions. I did not regenerate the four-question count because the underlying dataset is absent here.

**Impact:** chat validation results are not fully independent. This finding does not establish contamination of the separate pretraining validation corpus.

**Required change:** deduplicate/group before splitting, regenerate the affected chat evaluation, and update affected claims. Keep the faulty split as a teaching example if useful, paired with a verified corrected procedure.

### 7. High priority: nine chapters have not completed the recorded review process

**Location:** `docs/reviews/research-audit.md` and `docs/reviews/claims/`.

The recorded reviewed count is 32 of 41. Chapters 27–29 and 35–40 remain marked `drafted`, including serving, export, and the production gap. Separate chapter review files support this count; the shorter consolidated `claims-review.md` is not the complete record. The reviewer guide also records human expert and target-reader review as outstanding.

**Required change:** complete the nine chapter reviews, resolve resulting findings, and arrange an expert review plus a new reader's end-to-end walkthrough. Automated excerpt matching does not validate that a deployment recommendation works.

### 8. Medium priority: the beginner reproduction path needs a reliable starting point

**Location:** Chapter 1, Chapter 41.1, Appendix F, `attention_by_hand.py`.

The supplied folder is a book subtree; examples import source files from its parent. Running them directly here initially fails with missing `model` or `api`. After supplying the public source, `trace_forward.py` and `ch41_top_p.py` pass, but the simple `attention_by_hand.py` command still stops at a mandatory `ckpt_base_v2.pt` read after successfully running its toy sections. The native checkpoint is not supplied, and the book distinguishes it from the public bf16 export.

**Required change:** add an early setup section with the exact folder layout, tested dependency versions, expected output, and artifact download/rebuild commands. Separate CPU-only teaching examples from optional released-model measurements, for example with `--released`. Avoid making a first attention demonstration require a large native training artifact.

### 9. Medium priority: calculated memory limits are presented too confidently as serving capacity

**Location:** Chapter 40.2–40.3 and its key points.

The chapter calculates that 478 full contexts fit beside weights and the expert stack on 16 GiB, then concludes memory does not limit concurrency and batching is the first gap. The arithmetic is useful, but it excludes runtime workspaces, prefill activations, allocator effects, process overhead, and the latency target. It is not a measured concurrency limit.

**Required change:** call this a weights-and-cache upper bound, state exclusions beside the result, and measure the actual throughput/memory/latency trade-off before making a deployment recommendation.

### 10. Medium priority: integration notes and chapter text disagree with current work

**Location:** `docs/reviews/integration-todo.md`, Appendix F.3, Chapters 17, 20, and 23.

Several earlier findings are partly resolved in code or later chapters but remain open or misleading elsewhere. For example, multiple scripts now protect book measurement files behind `--save`, while Appendix F still says some other parts write them every time. Chapter 20 calls the router's 3.2% figure “not measured here,” while Chapter 23 supplies a measured 3.17% result and Chapter 17 discusses the distinction.

**Required change:** reconcile each integration item against the current source; mark fixes with their evidence; cross-reference measurements with their precise method and scope. Do not turn the full historical to-do list into a list of current defects without rechecking it.

### 11. Medium priority: the audit cannot establish complete executable coverage

**Location:** `ch41_audit.py`, Appendix F.4, `test_model.py`.

The example runner skips scripts based on whether their source text contains `ckpt_base_v2` or `measurements`, which can exclude a script just because of a comment or documentation string. The model suite also counts optional early returns as passes; this review observed two explicit Gradio skips.

**Required change:** maintain an explicit example/dependency manifest and report passed, failed, skipped, and unavailable checks separately. Keep assertion-based arithmetic checks, but describe their limits. Syntax parsing, source matching, and a passing tiny-model test each establish different things.

### 12. Publication administration remains unfinished

**Location:** `docs/reader/front-matter.md`.

The edition is labelled “Preview, 2026-10-31,” and the book text's licence is explicitly undecided.

**Required change:** settle the publication identity, edition/version, text permissions, and distribution package before a final release. This review does not assess the legal sufficiency of training-data or weight licensing.

## What is working well

- The core total/active parameter arithmetic reproduced exactly in the trace example.
- Publicly available code excerpts closely track their stated source ranges.
- The top-p capstone uses deliberately incorrect implementations to show that its tests detect real mistakes.
- The manuscript separates illustrative numbers, measured results, cited claims, and absent features more carefully than many tutorials.
- Evaluation discusses paired comparisons and tokenizer-dependent loss units; SFT discloses its own contaminated split rather than hiding it.
- The rendered sample has a consistent visual style. Complete-book visual QA is still needed.

## Recommended release order

1. Publish the complete, immutable companion source and fix the setup/reproduction path.
2. Correct the streaming repair and accepted-but-ignored API behaviour; repair the chat split and refresh affected evaluation.
3. Complete the nine outstanding chapter reviews and reconcile the integration records.
4. Build the full PDF/HTML, inspect its complete layout, and run a new reader through the documented setup and capstone.
5. Release as a technical reference. Add a measured operational capstone before making a production-readiness promise.

No manuscript, model, or serving code was changed during this review. Review-only downloads, reproductions, and logs are under `tmp/review/`.
