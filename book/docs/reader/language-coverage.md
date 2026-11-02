# Language coverage

Coverage snapshot: 2 November 2026. These are training-data and mechanics records,
not claims that a trained model has passed useful quality thresholds.

## Sources and modality scope

The India recipe includes Hindi, English, Bengali, Gujarati, Kannada, Malayalam,
Marathi, Odia, Punjabi, Tamil, Telugu, Assamese and Urdu. Text uses pinned Wikipedia
snapshots, with Simple English for English. The first text stage caps each language
at approximately 32 MiB of training text, while retaining the complete selected raw
shards. One shared 16,384-token tokenizer is fitted on training text only.

Speech uses FLEURS official train/dev/test splits. The Urdu configuration is
`ur_pk`; its inclusion means Urdu language coverage, not Indian geographic origin.
English uses `en_us`. India is not a single language or uniform domain.

General image captions use DOCCI and English descriptions. Controlled Indian-context
price cards, image questions and arithmetic are explicitly synthetic Hindi/English
examples. They do not demonstrate regional-language vision, OCR across Indian
scripts, conversational multilingual ability, noisy speech or speech generation.
The encoders are pretrained; only the decoder/projectors start from random weights.

## Prepared speech examples

These counts are taken from `data/india-v1/TOKEN_FILTERS.json` after upstream
audio validity/duration and training-transcript overlap checks. The final column
counts additional **training rows** removed by the token-context filter. It does
not count all exclusions from the original archive. Validation and test have
their own token exclusions, retained in that file.

| Language | FLEURS ID | Train | Validation | Test | Extra train exclusions |
|---|---|---:|---:|---:|---:|
| Hindi | hi_in | 1,984 | 230 | 388 | 107 |
| English | en_us | 2,503 | 392 | 643 | 0 |
| Bengali | bn_in | 2,661 | 362 | 814 | 263 |
| Gujarati | gu_in | 2,879 | 406 | 892 | 245 |
| Kannada | kn_in | 1,755 | 307 | 691 | 375 |
| Malayalam | ml_in | 2,262 | 291 | 668 | 722 |
| Marathi | mr_in | 2,869 | 389 | 846 | 309 |
| Odia | or_in | 931 | 339 | 719 | 144 |
| Punjabi | pa_in | 1,523 | 246 | 541 | 65 |
| Tamil | ta_in | 1,732 | 272 | 415 | 576 |
| Telugu | te_in | 2,026 | 271 | 398 | 245 |
| Assamese | as_in | 2,503 | 384 | 880 | 195 |
| Urdu | ur_pk | 1,869 | 267 | 297 | 0 |

The 512-token decoder context reserves 375 projected Whisper positions plus
control space, leaving a bounded transcript budget. Eligible audio must be mono,
16 kHz, nonempty and at most 24 seconds. This affects language coverage unevenly:
for example, Malayalam and Tamil lose many rows at the token filter. Raw data is
retained for later tokenizer/context experiments. Do not change the filter during
a live run or report filtered results as the full official benchmark.

## What is verified and what remains unknown

The run's real preflight performed finite forward/backward updates for all 13
speech configurations plus reading, captions, synthetic QA and VQA. That checks
the input/encoder/projector/decoder path. It is not an ASR accuracy assessment.
Only the explicitly named modalities and samples were checked.

The completed run received per-language output evaluation. The following WER
values are word edits divided by reference words; insertions can exceed 100%.
Validation uses 20 clips per language, test 50, drawn as first eligible rows from
fixed manifests. These are filtered samples, not full official benchmarks.

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

The mean relative-loss selector chose step 1,750 before testing. The final model
is recorded separately; it was not substituted based on test performance. These
results are insufficient for useful ASR. Text/title and arithmetic results are
also poor. Synthetic price-card VQA has some correct answers; it does not prove
real-world Indian visual competence. Each native validation report contains 530
cases; the selected checkpoint test report contains 920. Each export additionally
ran 66 overlapping validation cases with exact parameter parity. Lexical caption
scores do not replace semantic human assessment. The companion assessment retains
failures, filters, hashes and an explicit correction to text EOS handling.

Code-switching, dialects, proper names, borrowed words, noisy microphones and
regional-script visual questions need separate representative tests. They remain
unverified by this recipe. No language should be advertised as production-ready
merely because it appears in this table.

## Reproducibility and source conditions

The companion `config/india-training.json`, `config/dataset-sources.json` and
source metadata pin the selected revisions. FLEURS/DOCCI carry CC BY 4.0 source
conditions; the pinned Wikipedia card records CC BY-SA 3.0/GFDL conditions.
Retain attribution and transformation records and review distribution obligations.
This run does not clear older checkpoints' data lineage or settle all commercial
rights. Follow the companion dataset guides and the [Launch checklist](launch-checklist.md).
