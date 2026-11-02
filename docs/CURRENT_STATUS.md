# Current project status

Snapshot: 2 November 2026, after completed training and release checks.

## Training and assessment

All 102 pinned source files were downloaded and hashed. Native text pretraining
completed 10,000 optimizer updates; joint text/image/speech-input training
completed 60,000. No training job is currently required to finish this run.
This computer has an approximately 8 GB RTX 5050 Laptop GPU; the separately
reported 16 GB machine was not used. The existing native model architecture was
preserved. Whisper/SigLIP are frozen pretrained encoders; decoder/projectors were
initialized randomly. Speech generation is outside this run.

**The trained model is not ready for production.** Final-step validation WER was
94.9% for Hindi and 90.7% for English; most other languages remained near 100%.
The loss-selected step-1,750 checkpoint performed poorly on held-out tests too.
Do not present successful training mechanics as useful language proficiency.
See the [complete assessment](reviews/INDIA_MODEL_ASSESSMENT.md) and
[research model card](MODEL_CARD_INDIA_RESEARCH.md).

## Completed checks

- 66 core tests and 75 supporting tests passed locally. These are correctness
  checks, not independent model-quality measurements.
- Native selected/final checkpoints were evaluated on 530 validation cases each;
  the preselected checkpoint also received 920 held-out test cases. Both FP32
  exports passed exact state parity and 66-case validation exercises each.
  Repeated native/export evaluations overlap; these are not all unique examples.
- Evaluation records retain per-language/task scope, hashes, source conditions,
  eligibility filters, failures and the documented EOS scoring correction.
  Quality criteria were empty, so every report correctly withholds approval.
- Fixed streaming EOS/stop-buffer/UTF-8 behavior, true greedy decoding, tokenizer
  detection and explicit rejection of unsupported API options/history. Real
  checkpoint API smoke checks passed. Operational load/recovery remain untested.
- Chat preparation now groups equivalent prompts before splitting. Old
  contaminated chat scores remain invalid; that corpus was not retrained here.
- Local demo and research workbench are provided. Recorded cases require explicit
  reviewer approval; no general live multimodal quality claim is enabled.
- Original book prose and current reader guides use CC BY 4.0, selected by the
  author on 2 November 2026. Code remains MIT; third-party terms are preserved.

## Remaining release gates

1. Improve task quality through deliberate data/tokenizer/context/task-balance
   experiments and validation-only selection. Keep an untouched final test set
   for the next candidate; do not tune on the test results reported here.
2. Define use-case acceptance targets before evaluation. Add representative
   Indian speech, code-switching, real visual tasks and regional-script OCR.
3. Measure resource use, concurrent load, disconnects, reliability and recovery
   for the actual intended service. A CPU API smoke check is not a capacity test.
4. Complete human technical/editorial review and reader trials. Nine historical
   chapter review records remain drafted; the private historical pin/labs remain
   unavailable. An immutable public revision/fresh-clone trial is a separate gate.
5. Human approval of specific recorded demo cases remains separate from code
   tests, book licensing and general model approval.

## Reproduce and inspect

Read [START_TRAINING.md](../START_TRAINING.md), [language coverage](LANGUAGE_SUPPORT.md)
and [the course route](../book/docs/reader/course-route.md). Local detailed outputs
are under `runs/india-v1/post-training/`; current run status is
`logs/india-v1/status.json`. Corpora, checkpoints and live logs are excluded from
source contributions and portable setup ZIPs. The existing draft contribution
uses the [verified account and branch](reviews/CONTRIBUTION_TARGET.md).
