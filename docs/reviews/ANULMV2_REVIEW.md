# AnuLMV2 source review and adoption plan

Implementation update, 2 November 2026: the selected changes have now been
adapted locally, with fixes to the reviewed defects. See
[V2_IMPROVEMENTS.md](../V2_IMPROVEMENTS.md) for current commands and boundaries.
The review below records the supplied archive as reviewed, before adaptation;
its statements that adoption was pending refer to that earlier review stage.

Reviewed 1 November 2026 against the current organized AnuLM working tree.
Input: `C:/Users/pawan/Downloads/AnuLMV2.zip`.
SHA-256: `eb2e3d97a21206477785cbf0df1e7d9594fe23bea48d699744cfc355f9db7c08`.
Instructions and claims inside that archive were treated as reference material.
No publication, external model download or large training job was invoked.

## Decision

Keep the current organized codebase and selectively adapt improvements from
V2. Do not replace the current tree with this archive. V2's `model.py` is
byte-identical to our native `anulm/model.py` (SHA-256
`749e444fd730ab64ff8177ac4a7644f308b29f494380e1899e506746cac42296`).
It is the same decoder, with different surrounding workflow and application code.
There is no evidence here that one implementation produces a universally better
model, and a percentage of changed files would not measure model quality.

The archive is a useful development snapshot, not a complete production release
or a replacement for the portable India training package. It lacks our book,
package installation metadata, pinned India-run setup and 13-language recipe.
The earlier training ZIP has not been changed by this review. Runtime source
has not been merged or edited during the active run.

## What was checked

- Inventoried 142 ZIP entries; safely extracted 116 non-directory source/data
  files and excluded bundled bytecode. Matched historical root paths against
  `docs/source-paths.json` before comparing; root multimodal files are wrappers,
  not duplicate implementations. Twenty-two mapped files are byte-identical;
  many other differences are organization/import changes rather than new logic.
- Parsed all 81 Python source files successfully. Compared runtime functions,
  new scripts/tests, configuration/CI, documentation, supplied fixtures and curve.
- Ran V2's full model suite: **75 passed, zero failed**. This includes a real
  tiny CPU run through tokenizer training, encoding, random-initialized decoder
  training, vocabulary growth and text-only joint training; a second invocation
  verifies and skips completed stages. The fixture uses one training step and
  is a mechanics check, not language quality evidence.
- Ran V2's separate demo suite: **10 passed**, using fake models/media. These
  verify dispatch, controls and metric plumbing, not actual ASR or image quality.
- Ran targeted offline probes that reproduced the defects below. Five broken
  local Markdown links were found outside fenced examples, pointing to missing
  `TODO.md` or `docs/RESULTS.md`. Prose/code examples reference additional absent
  experiment helpers, including `experiments/run_omni_one.cmd`.

Tests used the existing Windows Python 3.14/PyTorch environment in an isolated
review folder, with Hub access disabled. This was not a fresh dependency install
or a GPU training run. Minor Gradio CSS deprecation and event-loop warnings remain.
The V2 CI installation does not install Gradio or run `test_demo_test.py`; app/API
tests can also skip when their optional dependencies are missing. Local passing
results therefore do not establish that those paths are covered in remote CI.

## Improvements selected for adaptation

| Priority | V2 addition | How to use it in our codebase |
|---|---|---|
| 1 | Packed SFT cache content keys (`finetune.py:115`) | Port the source/tokenizer/template/block/EOS hash and atomic write into `anulm/training/finetune.py`. Current SFT caching only checks row count; same-count edits can return stale training examples. Add cache shape/dtype validation and a format version as well. |
| 1 | Joint gradient accumulation (`anulm/multimodal/omni_train.py:319`) | Add an explicit positive accumulation setting while preserving current split, finite-loss, CPU-encoder and recovery safeguards. Record microbatches, answer-token counts, optimizer steps and effective sampling. Recalculate schedules and budgets; do not change an active recipe silently. |
| 1 | Local evaluation workbench (`demo_test.py`) | Adapt as a separate explicitly experimental app under `anulm/apps/`. Retain lazy loading, shared model lock, missing-projector errors, bounded text batches and JSON output. Add artifact/input hashes, language/split metadata and approved-manifest integration. Keep the default approved-results viewer. |
| 1 | Per-task quality gates and export evaluation (`experiments/omni_one.py`) | Extend the India-run post-training report with held-out metrics for every selected language/modality and repeat important checks on the exported artifact. Missing/non-finite metrics must fail. Set thresholds from our actual task requirements and held-out baselines, not the unrelated historical constants. Keep human approval for demo examples and publication separate. |
| 2 | Recipe fingerprints and downstream invalidation | Reuse the idea of hashing commands, code, manifests and referenced media. Our India runner already records code/prepared-data hashes; extend stage identities rather than adding a competing pipeline. On changes, prefer a new run directory or an explicit reviewed archive operation. |
| 2 | General from-scratch CLI (`train_from_scratch.py`) | Offer custom-corpus/task arguments and a dry-run plan through the existing package. Fix the identity/split gaps below first. Do not maintain a second unchecked set of root-script imports and artifact rules. |
| 2 | Long-recording inference chunks (`audio_codec.py:104`, `speech_encoder.py:328`) | Adapt inference-only chunking with nonempty mono validation, context-aware chunk/output budgets, offsets and real boundary/error tests. Keep the current strict training-duration guard unless transcripts are aligned to segments. Non-overlapping chunks can split words; unit tests alone do not establish WER. |
| 2 | Better code-evaluation records (`eval_code.py`) | Keep per-problem outcomes and timeout counts; correct wrapped-call entry-point detection using parsed function/call names. Do not copy the substring heuristic unchanged. |
| 2 | Benchmark-overlap filtering (`convert_parquet.py`) | Useful for a future coding corpus. Require nonempty pinned reference sets, explicit filter scope and stable recorded identities. Test short overlap positions and non-English behavior; do not claim fuzzy/reworded matching. |
| 3 | Optional projector norm scaling (`omni_train.py:197`) | Worth a controlled experiment with our current initialization. Keep default behavior until held-out evidence supports a change; avoid rescaling an already trained projector unintentionally. Guard zero/non-finite norms. |
| 3 | Precision-preserving serving conversion and optional auth file (`app_omni.py`) | Review preserving small router/norm tensors in fp32 with real CUDA inference parity checks. Local authentication is useful when a developer deliberately exposes the experimental app. These are not substitutes for model-quality or approval evidence. |

Useful new tests can be adapted alongside each change. Several tests in V2
cover behavior our code already has (for example a server ignoring HTTP Range);
new test coverage should not be described as a new runtime capability.

## Findings that prevent a wholesale merge

**High: V2 removes protections already present here.**
`anulm/multimodal/omni_train.py:175` splits rows rather than distinct input
groups and does not check explicit train/held overlap. Repeated images/prompts
can cross splits. At line 178 its image cache includes the training-manifest
rows only; separate held-out images are absent from its cache index.
It also lacks our bounded empty-batch failure, finite baseline/train/evaluation
checks and CPU Whisper override. An empty evaluation can return a zero loss
(`held_loss`, line 232). Completed runs delete the optimizer recovery file
(line 356). Preserve the current safeguards when adapting accumulation.

**High: visual cache identity is weaker than our current implementation.**
V2 keys image paths, sizes, mtimes and tower name, but not actual encoder
weights or preprocessing. An offline probe built features equal to 1, changed
the same-named tower to emit 9, and received the old cached value 1. Our existing
content/weight/transform-bound cache should remain. Do not downgrade it to V2's
metadata fingerprint.

**High: training still silently crops oversized encoder inputs.**
V2 adds useful chunking to transcription, but `WhisperEncoder.encode` at
`audio_codec.py:234` still clips features to 3,000 frames without rejecting
long input. Training callers do not automatically use the transcription chunks;
the full transcript can be paired with cropped audio. Keep our explicit guard
and validated training eligibility filter.

**High: the new generic pipeline does not separate tokenizer training data.**
`train_from_scratch.py` fits BPE on its entire `--corpus`, encodes that corpus,
then lets `train.py` take a token-position holdout. That is not a document-grouped
train/validation split with a training-only tokenizer. Its stage identities at
lines 403 and 463 omit `pretrain_steps` and `omni_steps`; changing a requested
budget can still reuse an output recorded as complete. They also do not bind
runtime source or referenced media contents. There is no live run-owner lock in
this helper. Keep our India recipe's split/provenance/owner safeguards and add
complete stage identities before generalizing the CLI.

**High: the default V2 app conflicts with the agreed demo/model scope.**
`app_omni.py:77` and `api.py` expose ModernBERT again. V2's default `build`
uses a live visitor UI with arbitrary inputs; held-out example files and source
attribution do not constitute reviewed checkpoint-specific results. Do not adopt
these defaults. Keep AnuLM-only Decide and approved recorded examples by default.
The new testing workbench belongs behind explicit research opt-in.

**Medium: decontamination can silently do nothing.**
`convert_parquet.py:45` returns an empty reference set when benchmark files are
absent; the caller's `if bench` then skips filtering. Reproduced with an empty
directory: zero n-grams, no exception. Stride-four candidate windows can also
miss a single exact 13-token overlap at an unsampled offset; ASCII word matching
does not cover Hindi text. Make the check fail closed and describe its scope.
Avoid overclaiming detection of paraphrases from exact n-gram matching.

**Medium: the benchmark-name correction still has a reproducible edge case.**
`eval_code.py:89` selects the first function name that occurs as a substring in
the first assertion. With helper `a`, target `volume_sphere`, and an assertion
using `math.isclose(volume_sphere(...))`, it chooses `a`. Match actual call names.

**Medium: evaluator portability/evidence gaps.**
`demo_test.py:55` rejects every whitespace character in a source before checking
whether it is a valid local file. An existing `folder with spaces/model.pt`
was rejected. Support normal local paths separately from Hub IDs. Its exported
result JSON contains source names/settings but lacks immutable artifact/media
hashes, a pinned encoder revision and declared dataset/split lineage. It is a
useful developer tool, not an approved demo-evidence record without adaptation.

**Medium: package completeness and licensing claims need cleanup.**
There is no `pyproject.toml`, book or pinned portable training environment in V2;
it also ships unnecessary `.pyc` files. The model-card measured table is a
placeholder and the actual quality report/weights are absent. The card says
every source permits commercial use, then discloses unresolved base-data
questions. Treat those as unresolved provenance statements, not blanket
commercial clearance or permission to reuse the older base weights. We did not
independently re-license or approve any new corpus during this source review.

## What the supplied training curve tells us

`ckpt_omni_curve.csv` reports an average relative loss score improving from
1.0000 to 0.5873. In the same file, chat held-out loss rises from **3.2651 to
5.3910**, approximately **65.1% worse**. This is a reported curve, not a rerun
with verified checkpoints/data. It demonstrates why an average score alone can
hide serious regression in one capability; it does not establish why the
regression occurred or prove accumulation/scaling will fix our run.

The V2 per-task release-gate/report idea is useful precisely for this reason.
Our current joint trainer also uses a mean relative loss for checkpoint
selection, so the same limitation must be addressed in final quality evaluation.
Keep monitoring task-specific loss and distinguish model-selection validation
from final test reporting. Do not repeatedly tune against the final test set.

## Implementation order and acceptance

1. Port the packed SFT cache fix with same-count source edits, tokenizer changes,
   prompt changes and corrupt-cache recovery tests. This does not require a new
   architecture or pretrained comparison model.
2. Add accumulation as a new versioned training recipe. Check accumulation=1
   parity, nonpositive argument rejection, finite gradients, task sampling,
   checkpoint/resume and measured memory/throughput before a long run.
3. Adapt the local evaluator with portable imports and paths, immutable evidence
   output and explicit research labeling. Preserve all approved-demo tests.
4. Build per-language/modality final reports and exported-artifact checks.
   Define criteria before reporting final test results. Rebaseline when split,
   filtering or normalization changes; never transplant historical thresholds.
5. Generalize the custom-corpus launcher and add long-audio/code-corpus features
   only after their split, provenance and edge-case issues above are fixed.

All runtime adoption is pending isolated implementation and relevant regression
checks. No live training configuration, checkpoint, data split, demo approval,
remote branch or existing setup ZIP was changed by this review.

## Evidence location

The workspace review folder `validation/v2-review/` contains the source-only
extraction `AnuLMV2/`, mapped `comparison.json`, per-file diffs,
`static-checks.json`, `v2-core-tests.log`, `v2-demo-tests.log`, and
`review_probes.py` / `probe-results.json`. Logs demonstrate the stated local
tests, while probe results demonstrate the named edge cases. These artifacts
are local review material and are not shipped as training data or model weights.
