# Selective V2 implementation verification

Completed locally on 2 November 2026. Usage is documented in
[V2_IMPROVEMENTS.md](../V2_IMPROVEMENTS.md). This adapts selected workflows from
the reviewed archive into the existing organized repository; it does not replace
the decoder, enable comparison models, approve demo samples or publish anything.

Implemented:

- Content-bound, shape-checked, atomic SFT caches with corrupt-cache recovery.
- Joint microbatch accumulation, resume setting checks, answer-token counters,
  optional per-task regression gates and optional fresh-projector norm scaling.
  Existing group splits, finite-loss checks, CPU Whisper support and recovery
  checkpoint retention remain in place. Defaults preserve accumulation=1.
- A separate local research evaluation app with lazy model loading, ordinary
  paths containing spaces, missing-projector errors, text/ASR/image evaluation,
  declared prompt-language support and unreviewed JSON evidence records.
- Per-task/language held-out reports with explicit source/license/split plans,
  file identities, sample counts and fail-closed numeric criteria. The same
  command can be repeated against exported artifacts. India training generates
  a validation plan; final test sets are not automatically used by the trainer.
- A custom-corpus package CLI that deduplicates/groups documents before splitting,
  fits a training-only tokenizer, binds source/data/media/encoder/step identities,
  verifies completed artifacts, owns a run lock and bounds subprocess execution.
- Long-audio inference windows while retaining strict aligned-duration training
  checks; exact Unicode-token decontamination that refuses missing references;
  parsed MBPP entry points and per-problem/timeout evaluation records.
- Opt-in precision preservation in the legacy serving helper and optional local
  auth-file support in the existing demo. The default approved-results viewer
  remains unchanged in scope.
- Shared tokenizer loading in serving/sampling, discovered by real checkpoint
  integration tests. Joint checkpoints now save the prompt templates actually
  used for their text tasks. Both native BPE and the pipeline's HF tokenizer
  format are supported through the existing tokenizer adapter.
- Updated usage/setup documentation and CI commands for the new tests. CI has
  not been run remotely and the destination 16 GB GPU has not been tested here.

Verification:

- Original model suite: **66/66 passed**, including export/wrapper and legacy
  tokenizer/checkpoint regression checks.
- **62 distinct supporting tests passed**: layout, multimodal/demo contracts,
  dataset setup, India-run controls, workflow fixes, evaluator and the custom
  pipeline. The 60-test aggregate passed, followed by all 12 workflow tests
  after adding two optional-experiment checks (10 overlap with the aggregate).
- The custom pipeline test performs actual one-step CPU decoder and joint
  training, then loads that checkpoint through the reporter, creates an
  unreviewed held-out report, verifies reuse, and rejects a changed step budget.
- Real local GPU preflight passed text, caption and English/Hindi ASR
  forward/backward optimizer updates with locally pinned SigLIP/Whisper encoders.
  Peak allocated CUDA memory: **1,721,712,640 bytes**, about 1.6 GiB, on the
  local 8,546,484,224-byte GPU. This is not a worst-case memory estimate or
  evidence of trained quality, and it does not validate all 13 speech languages.
- Native decoder SHA-256 remains
  `749e444fd730ab64ff8177ac4a7644f308b29f494380e1899e506746cac42296`.

Local evidence is in `validation/v2-review/implementation-*-tests.log` and
`implementation-gpu-preflight.log`, plus
`runs/india-v1/v2-adoption-preflight/RESULT.json`. These runtime artifacts are
excluded from the portable source ZIP. Gradio event-loop cleanup warnings were
observed without test failures.

The old downloader/orchestrator was no longer running when implementation began;
its status file was stale. Partial downloaded files were preserved. No long
training job was restarted during this work. The new ZIP provides a fresh setup
on the target machine; it does not migrate prepared manifests/checkpoints with
absolute paths. Full training, representative multilingual/multimodal quality
evaluation, destination GPU checks and human demo approval remain separate steps.
