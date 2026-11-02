# Current companion workflow

Updated 2 November 2026. This guide describes the current local companion code.
Chapter citations and historical numerical results retain their recorded revision;
they have not been silently replaced by the new India-focused experiment.
Use the supplied source bundle or the matching draft contribution revision.
A draft contribution is not an approved production release.

## What you can build

You can initialize and train the native AnuLM decoder and its modality projectors
from random weights. The image and audio encoders in the current recipe are
frozen pretrained SigLIP and Whisper components. Speech means transcription of
audio input; the India run does not train speech generation. A successful small
run establishes mechanics, not a general assistant or production readiness.

Read [Language coverage](language-coverage.md) for supported training sources,
eligibility filters and what has not been demonstrated. The [course route](course-route.md)
provides exercises that use available files rather than the unpublished historical labs.

## Set up the current package

Extract the portable source ZIP and read the companion root `START_TRAINING.md`.
The pinned GPU recipe uses Python 3.14 and the tested CUDA PyTorch pair; the book's
introductory CPU examples retain their broader Python 3.10+ requirement. From the
companion repository on Windows:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\windows\setup_training.ps1
```

The script creates `.venv`, installs the pinned dependencies, checks dependency
consistency, runs GPU arithmetic and checks browser rendering. Linux instructions
are in `START_TRAINING.md`. The source ZIP excludes datasets and checkpoints;
download scripts retain source URLs, versions, hashes and attribution.

## Start with a small custom corpus

Use a UTF-8 corpus containing distinct documents separated by blank lines.
From the companion root:

```bash
python -m anulm.training.from_scratch --corpus data/my-corpus.txt --work-dir runs/my-run --source "Describe the source" --license "Record its terms" --device cpu --steps 10 --context 64 --vocab 1024 --dry-run
```

Remove `--dry-run` to execute. Each split must contain enough tokens; a few words
are insufficient. The launcher groups exact duplicate documents before splitting
and fits its tokenizer only on training documents. Its default is a text-only
learning run. To train text/image/ASR tasks jointly, supply explicit task manifests
and separate held-out manifests; local pinned encoders are required for media.
The companion `docs/V2_IMPROVEMENTS.md` describes these options.

The launcher binds step budgets, code and inputs to its run directory. Use the
same command with `--resume` after interruption. Changed inputs or budgets need
a new directory. Native and HF-format tokenizers use the shared loader.

## Run the full India recipe

After reviewing the source conditions and resource settings:

```powershell
.\scripts\windows\start_india_training.ps1 -PythonExecutable (Resolve-Path '.\.venv\Scripts\python.exe').Path
Get-Content .\logs\india-v1\status.json
```

The stages are: verified downloads, preparation, synthetic examples, training-only
tokenizer fit, real-data preflight, decoder pretraining, vocabulary growth and joint
training. Keep the machine powered, awake and online. The original not-before
time was 1 November at 20:00 IST; when that time has passed there is no additional
date wait after preparation/preflight. The initial training budget is 40 hours,
excluding downloading/preparation. Runtime code/configuration must remain stable
during the run. Do not start duplicate jobs or remove a live owner's lock.

Joint `--accum N` averages N independently sampled microbatch losses per optimizer
step. It is not answer-token weighting; changing N changes total work at the same
step count. Default accumulation remains one. Optional projector scaling and
per-task loss-ratio gates require a new deliberate recipe, not an unrecorded change
to a running experiment. Optimizer recovery files are retained.

## Evaluate and decide whether to release

```bash
python -m anulm.apps.evaluate --ckpt runs/india-v1/omni.pt --port 7864
python -m anulm.evaluation.report --checkpoint runs/india-v1/omni.pt --plan runs/india-v1/validation-plan.json --out runs/india-v1/validation-report.json --device cuda
```

The generated validation plan becomes available after joint training completes.
Declare meaningful per-task criteria before a quality decision. It starts with
empty criteria, which do not pass the quality gate. Use a separate predeclared
test plan for final reporting and repeat relevant checks on the exported artifact.
The workbench records unreviewed outputs; it does not approve public demo cases.
The default demo remains an approved recorded-results viewer and can be empty.

Do not run extra GPU evaluations concurrently with the active training job on
the small laptop GPU. Plan evaluation after training or on separate resources.
Passing unit tests, declining loss and a preflight update do not establish WER,
caption quality, factual reliability or production service correctness.

## Completed run and remaining work

The November run completed 10,000 text and 60,000 joint optimizer updates.
The loss-selected checkpoint is step 1,750; the final step-60,000 checkpoint is
separate. Both were evaluated and exported locally with exact parameter parity.
The selected checkpoint received a held-out test evaluation. The companion
`docs/reviews/INDIA_MODEL_ASSESSMENT.md` records the samples, failures and hashes.
These results are poor: final validation WER is 94.9% for Hindi and 90.7% for
English, with most regional languages near 100%. Language coverage is not
language competence. No quality gate is approved.

Streaming EOS/UTF-8/stop handling and option validation were repaired and tested
in the current source. Conversation history is explicitly rejected; it is not
silently discarded. Chat preparation groups prompts before splitting, but the
old contaminated chat scores remain invalid. Production load, recovery and
representative use-case acceptance remain untested.

Original book prose and these reader guides are CC BY 4.0; code remains MIT.
Third-party assets/data/weights retain their own terms. Human technical/editorial
review, reader trials and an immutable public version are still required.
Historical numerical claims remain tied to their original revision.

The historical `course/` stub labs and private revision remain unavailable.
Follow [Course route](course-route.md); this update does not complete every
chapter's expert review or approve a production service.
