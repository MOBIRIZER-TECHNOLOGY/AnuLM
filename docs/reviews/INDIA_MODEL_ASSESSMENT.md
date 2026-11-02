# India-run trained-model assessment

Completed 2 November 2026. **Not approved for production or general task quality.**
The native implementation trained successfully, but output quality is inadequate.
Passing code tests and a completed training schedule do not establish useful ability.

## Artifacts and selection

Text pretraining completed 10,000 optimizer updates; joint training completed
60,000. The existing mean-relative-validation-loss selector saved `omni.pt` at
**step 1,750**, not the final step. The final 60,000-step artifact is separately
preserved as `final-step-60000.pt`. The loss-selected artifact was fixed before
held-out testing; no test score selected a different checkpoint.

Both artifacts now have local FP32 exports, with pinned Whisper/SigLIP snapshots
bundled through relative paths. Exact decoder/projector state equality passed;
the loss-selected export also passed explicit tokenizer identity checks. Each
export was exercised on 66 validation cases. These checks establish local
serialization/inference mechanics, not quality or cross-hardware certification.

## Sampled speech results

WER is total word edits divided by total reference words. Values above 100%
are possible due to insertions. Validation uses 20 clips per language, while the
loss-selected checkpoint's test sample uses 50. These are first eligible rows
in fixed manifests, not full official benchmark results or speaker-independent
claims. The 512-token transcript eligibility filter remains part of the result.

| Language | Step 1,750 validation WER | Step 60,000 validation WER | Selected checkpoint test WER | Test clips |
|---|---:|---:|---:|---:|
| Hindi | 98.0% | 94.9% | 99.5% | 50 |
| English | 160.1% | 90.7% | 169.9% | 50 |
| Bengali | 99.7% | 99.4% | 99.9% | 50 |
| Gujarati | 99.2% | 96.9% | 99.5% | 50 |
| Kannada | 100.0% | 103.6% | 100.0% | 50 |
| Malayalam | 100.0% | 101.5% | 100.0% | 50 |
| Marathi | 100.0% | 100.0% | 100.0% | 50 |
| Odia | 99.7% | 100.0% | 99.0% | 50 |
| Punjabi | 98.9% | 103.7% | 98.3% | 50 |
| Tamil | 100.0% | 101.6% | 100.2% | 50 |
| Telugu | 99.6% | 98.9% | 99.9% | 50 |
| Assamese | 100.9% | 99.7% | 99.8% | 50 |
| Urdu | 94.2% | 101.4% | 94.4% | 50 |

## Other tasks

Reading is a **derived passage-title identification task**, not broad question
answering. It was evaluated separately for all 13 text configurations. Final
text test documents were absent from both prepared text splits, and their input
prompts were disjoint from prepared reading prompts. This does not establish
conversational, factual, code or reasoning quality.

The reading and synthetic arithmetic results are poor. General image captions
have only lexical exact-match/F1 diagnostics; these are not semantic caption
quality judgments. Synthetic price-card VQA has some correct results, but they
must be reported separately from general photos, regional-script OCR and real
Indian visual coverage. Full outputs and failures are retained locally.

## Evaluation scope and correction

The two native validation reports contain 530 cases each. The selected artifact's
held-out report contains 920 cases. Each export report contains 66 validation
cases, overlapping the native validation inputs; these are repeated artifact
checks, not additional independent benchmark examples.

A serving bug displayed the HF EOS control token as text. It was fixed and
**all affected text tasks were regenerated and rescored**. Audio/image outputs
were unaffected and retain their original code provenance. Canonical reports
record component sources, and the original pre-correction reports remain beside
them. No strings were silently edited to inflate a score.

Quality thresholds were not predeclared for this exploratory assessment; empty
criteria do not pass the reporter's gates. The poor observed outputs are not
suitable evidence for launch. Do not tune repeatedly against the sampled test
cases; reserve untouched examples for a later finalized recipe.

## Completed engineering checks and remaining work

- Core model regressions: 66 passed. Supporting checks cover multimodal contracts,
  grouped splits, setup, recovery, evaluation and the API fixes.
- Known EOS/UTF-8/stop-buffer faults, exact greedy behavior, token-limit validation,
  unsupported history/sampling settings and grouped chat splitting are corrected.
- An in-process real-checkpoint API smoke test verified configured-key rejection
  and streamed/non-streamed parity. Its small CPU timings include empty/short
  outputs and are **not a deployment throughput benchmark**.
- The default demo still requires approved recorded evidence. Pending synthetic
  successes are prepared for reviewer inspection; they do not certify the model.
- Remaining research: strengthen the text foundation, investigate task interference
  and language/context imbalance, test accumulation/projector curricula as isolated
  recipes, and select checkpoints using declared task requirements. None of those
  experiments is asserted to improve quality before it is measured.
- Remaining release decisions: intended workload/quality targets, independent
  technical/reader review, and any model-weight distribution license. Original book
  text is now CC BY 4.0 by the author's choice; code remains MIT and third-party
  material keeps its own terms.

## Reproduce and inspect

Use `scripts/post_training.py` after the India run to prepare the fixed cohorts,
run validation, export locally and evaluate the training-selected artifact. Do
not rerun training against changed input identities or silently choose by test
scores. Full reports, plans, original/corrected text evidence and export parity
records are under `runs/india-v1/post-training/` (ignored by Git). The summary
`docs/reviews/india-assessment-2026-11-02.json` records report hashes and counts.
No model weights or datasets were uploaded by these checks.
