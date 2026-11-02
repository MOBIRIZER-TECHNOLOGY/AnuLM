# Selective V2 workflow improvements


Completed-run update (2 November 2026): text and joint training, native evaluations and both FP32 export checks are finished. Quality is inadequate for production. See [current status](CURRENT_STATUS.md) and the detailed assessment linked there. This guide describes reproducible setup, not an instruction to restart the completed run.

Current companion update (2 November 2026): see [current status](CURRENT_STATUS.md) and [language coverage](LANGUAGE_SUPPORT.md). Historical model cards, experiment numbers and commands below must not be presented as measurements of the completed India research run.

These tools extend the same AnuLM implementation. The native decoder has not
been replaced. ModernBERT is not enabled. The default demo still displays only
approved recorded cases; the evaluation workbench below is a research tool.

## Custom-corpus training from scratch

Install the environment described in [START_TRAINING.md](../START_TRAINING.md).
For the existing 13-language corpus, continue using the
[India training pipeline](INDIA_TRAINING.md). For your own local corpus:

```bash
python -m anulm.training.from_scratch --corpus data/my-corpus.txt --work-dir runs/my-run --source "Describe your source" --license "Record its terms" --device cuda --grad-ckpt --dry-run
```

Remove `--dry-run` to execute. Input documents use the project's `DOC_SEP` or
blank-line separators. Exact duplicates are grouped before a deterministic 10%
validation split. The tokenizer is fitted on training documents only. Each split
needs enough tokens for at least two context windows. The default context is 256,
vocabulary target 16,384, native preset `30b`, batch size one and 2,000 text steps.
These are initial research settings, not a model quality guarantee. The preset
name refers to the reference architecture shape, not 30 billion actual parameters.

An optional joint stage uses explicit separate training/validation manifests:

```bash
python -m anulm.training.from_scratch --corpus data/my-corpus.txt --work-dir runs/my-joint-run --source "Describe your source" --license "Record its terms" --device cuda --grad-ckpt --steps 2000 --joint-steps 2000 --accum 4 --task chat=text:data/chat_train.jsonl:1:1 --held chat=data/chat_validation.jsonl
```

Use `kind=image` or `kind=asr` for media tasks. Supply `--tower LOCAL_PINNED_DIR`
or `--whisper LOCAL_PINNED_DIR` respectively; these components remain frozen
pretrained encoders. Media paths resolve beside their manifest; use absolute
paths for older manifests with a different convention. Every task requires an
explicit held-out manifest. The current trainer checks distinct input groups
and disjoint splits. License/source arguments describe your own corpus and do
not grant rights to third-party data.

`pipeline.json` binds arguments (including both step budgets), runtime source,
corpus/manifests, referenced media and encoders to the run. Completed stage hashes
are verified before reuse. A live owner lock blocks duplicate jobs. Use the same
command with `--resume` after interruption; `.last` recovery files remain available.
Changed recipes require a new run directory. `--max-hours` bounds training stages
within each invocation; retries do not promise a global lifetime budget. Preparation
is separate. Keep the machine powered and prevent sleep yourself for this foreground
custom launcher. It does not install a permanent service or schedule training.

## Joint gradient accumulation and optional experiments

The native joint trainer accepts `--accum N` (default one). Each optimizer step
averages N independently sampled task microbatches. This is a mean of microbatch
losses, not an answer-token-weighted mean. `seen` counts microbatches and the
recovery checkpoint records graded `answer_tokens` per task. Step counts, warmup
and evaluation intervals remain **optimizer steps**. Raising accumulation without
adjusting budgets increases total work; do not assume the same runtime.

The India orchestrator reads optional `joint_accum` from its configuration and
defaults to one. Change it before launching a new recipe, then measure memory,
throughput and each task's validation behavior. No learning-rate or batch-size
improvement is claimed without that experiment.

`--max-task-loss-ratio R` optionally restricts best-checkpoint selection so every
task's validation loss must be no more than R times its initial value. Failed
gates still write recovery checkpoints. An aggregate score alone can hide a
task regression. Choose requirements before selection; a ratio gate is not a
substitute for final held-out quality measurements.

`--proj-scale embed` is an opt-in experiment for fresh projectors only. It
initializes their final normalization output near text-embedding norms; `ln`
preserves existing behavior. It rejects explicit projector warm starts. There is
no claim that this improves the India run. Resume checks accumulation, scale mode
and the optional gate. The legacy app's `_half(..., preserve_small_fp32=True)`
is also opt-in; its default precision path remains unchanged.

## Local evaluation webapp

If you did not install the pinned portable training environment, install the
evaluation extras with `python -m pip install -e ".[evaluation]"`; image/audio
inference also needs the pinned encoder dependencies in the training setup.

```bash
python -m anulm.apps.evaluate --ckpt runs/my-joint-run/omni.pt
```

It binds to localhost by default and loads a model only when requested. It
supports single/batch text evaluation, ASR transcription and image caption/VQA
testing, with missing-projector errors and JSON downloads. Normal local paths
containing spaces are supported. Use `examples/evaluation/text_task.jsonl` only
to learn the schema; its two example rows are not a model-quality benchmark.

JSON output is always marked **unreviewed**. It records model file hashes, input
identities, media hashes and runtime source. Local tokenizer/encoder identities
are recorded when available; symbolic Hub/encoder names are marked unresolved.
For reproducible evidence, download pinned local snapshots first. The standalone
workbench does not assert language/source/license/split provenance for arbitrary
uploads; the report plan below declares those fields. Workbench output does not
automatically satisfy the separate approved-demo schema.

Text exact match and token F1 are lexical measures. They are not sufficient
judgments of chat or caption quality. Speech reports edit count and reference
word count; aggregate WER is total edits divided by total reference words.

The normal demo also accepts `--auth-file PATH` with one local `user:password`
line for deliberate sharing. Keep that file outside version control and source
bundles. Authentication does not approve arbitrary research results.

## Per-language/modality reports and export verification

Use an explicit JSON plan with `required_tasks` listing every task name. Example:

```json
{
  "required_tasks": ["asr_hi"],
  "tasks": [{
    "name": "asr_hi", "kind": "asr", "language": "hi",
    "split": "validation", "path": "../data/hi_validation.jsonl",
    "source": "Pinned source URL and revision", "license": "Source terms",
    "limit": 100,
    "criteria": [{"metric": "wer", "op": "<=", "threshold": 0.2}]
  }]
}
```

The threshold is an **illustration**, not an AnuLM acceptance target. Declare
your own requirements before reporting. Supported kinds are `text`, `asr`,
`caption`, `vqa`; metrics are WER for ASR and lexical exact match/F1 otherwise.
Paths resolve beside the plan. Sampling is the first N manifest rows in their
declared order (at most 1,000 per task); construct a representative manifest
before evaluation and report the sample counts and resource/length filters.

```bash
python -m anulm.evaluation.report --checkpoint runs/my-joint-run/omni.pt --plan config/my-evaluation.json --out runs/my-joint-run/validation-report.json --device cuda
python -m anulm.evaluation.report --checkpoint release/MyExport --plan config/my-evaluation.json --out runs/my-joint-run/export-validation-report.json --device cuda
```

Review both reports and compare every declared metric before distributing an
export. Missing/non-finite metrics and empty criteria fail the declared gates;
a passing numeric gate does not grant demo approval or production certification.
The reporter never uploads models. After India training, the orchestrator writes
`runs/india-v1/validation-plan.json` covering all 13 ASR configurations and the
other prepared task groups, with **no invented quality thresholds**. Run the
report command separately. Final test reporting needs its own predeclared plan;
the trainer does not automatically consume final test sets or tune against them.

## Long-audio inference

Continuous transcription now splits recordings into non-overlapping windows of
at most 30 seconds and joins their outputs in order. Inputs must be finite,
nonempty mono samples at 16 kHz; file inputs are read/resampled as before.
The per-window output budget respects remaining decoder context. This avoids
silently ignoring the tail of long recordings, but word boundaries can still
degrade accuracy. No long-recording WER improvement is claimed. Training retains
its strict 30-second encoder guard and needs transcripts aligned to any segments.

## Exact benchmark overlap filtering

```bash
python -m anulm.data.decontaminate --corpus data/code.txt --reference data/reference.jsonl --out data/code.filtered.txt
```

References are explicit local JSONL or Parquet files; fields default to prompt,
text, code, canonical_solution, test and test_list. Missing files or no usable
reference n-grams fail instead of silently disabling filtering. Every Unicode-token
window is scanned (default 13, stride one); hashes and filtering counts are
written beside the output. Existing input/output files are not overwritten.
This detects exact normalized token overlap, not paraphrases, short overlaps or
all contamination. Retain reference/source terms and do not use final test scores
to tune filtering thresholds. No dataset is added or downloaded by this command.

The code evaluator also records per-problem outcomes and timeout counts, and
identifies MBPP target functions through parsed calls instead of substring matches.
Generated-code execution remains an explicit separate evaluation action.
