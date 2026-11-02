# Initial India-focused training run


Completed-run update (2 November 2026): text and joint training, native evaluations and both FP32 export checks are finished. Quality is inadequate for production. See [current status](CURRENT_STATUS.md) and the detailed assessment linked there. This guide describes reproducible setup, not an instruction to restart the completed run.

Current companion update (2 November 2026): see [current status](CURRENT_STATUS.md) and [language coverage](LANGUAGE_SUPPORT.md). Historical model cards, experiment numbers and commands below must not be presented as measurements of the completed India research run.

**Running on another computer:** follow [START_TRAINING.md](../START_TRAINING.md)
for the portable ZIP, tested dependency pins, Windows setup helper, Linux recipe,
GPU/disk/browser checks and launch commands. The ZIP is a fresh source setup;
it excludes the current machine's data, checkpoints and process records. The
destination regenerates paths and downloads the pinned sources itself. Its
16 GB GPU still needs its own real-data preflight; the measurements below belong
to the source laptop. Use `scripts/requirements-india.txt` for application
dependency pins and install the matching CUDA torch/torchvision pair first.

This is a local research run using the existing AnuLM decoder. It is not a
claim of world-class quality or a production release. The author authorized
full selected data downloads and overnight training on 1 November 2026.

## Scope

- Languages: Hindi, English, Bengali, Gujarati, Kannada, Malayalam, Marathi,
  Odia, Punjabi, Tamil, Telugu, Assamese and Urdu. FLEURS Urdu uses the `ur_pk`
  configuration: language coverage does not imply Indian geographic provenance.
- Text: pinned 2023 Wikimedia Wikipedia snapshots for those languages (Simple
  English for English). Complete selected raw shards are downloaded; initial
  text preparation bounds each language's training text to 32 MiB and uses a
  content-grouped held-out split. Source URLs/titles remain in the JSONL records.
- Audio: full pinned FLEURS train/dev/test archives for all 13 configurations.
  Training uses only official train; dev selects checkpoints; test stays separate.
  Clips longer than 24 seconds and exact transcripts overlapping held-out data
  are excluded from training. This is not a speaker-independent guarantee.
  A predeclared token-length filter reserves the full Whisper frame budget and
  removes transcripts that would exceed the decoder's 512-token context from
  each prepared split; accepted/dropped counts are recorded in `TOKEN_FILTERS.json`.
  Raw official files remain intact. Evaluation must name this eligibility filter.
- Images: complete DOCCI images/descriptions. General photos are not presented
  as an India-specific photo corpus. The official test set stays separate;
  validation is selected from official train using deterministic groups.
- Indian-context controlled examples: procedural Hindi/English price cards,
  image questions and integer arithmetic. These are explicitly synthetic,
  with computed ground truth and groups assigned before rendering. They do not
  represent India's real-world visual or conversational diversity.

The tokenizer is fitted on training text only. The decoder starts from random
weights through the existing native trainer; its projectors also start fresh.
Whisper-small and SigLIP image encoders are frozen pretrained components, not
trained from scratch. Decide comparison models are not used or downloaded.

## Resources and timing

The local probe reported an RTX 5050 Laptop GPU with 8,151 MiB total memory;
the author's separately reported original 16 GB setup remains a distinct record.
This run uses the smaller GQA AnuLM preset, eight experts, 512-token context,
gradient checkpointing, and single-example joint batches. Whisper uses CPU
encoding to avoid consuming decoder GPU memory and CUDA runtime conflicts.

`config/india-training.json` records seeds, sources, versions, step budgets and
a **not-before time of 8 p.m. India time on 1 November 2026**. Downloads and CPU
preparation can start earlier. Training starts later if data/preflight are not
ready. Actual completion depends on measured throughput; the training budget
is bounded to 40 hours. This is not a promise that all stages finish tonight.

Keep the laptop powered and running. The active runner prevents ordinary idle
sleep temporarily and releases that request on exit; it cannot prevent power
loss, manual sleep, closing the lid or reboot. No permanent power settings change.

## Layout and recovery

Install a CUDA PyTorch build appropriate to the machine first, then inspect a
dependency dry run before installation. The tested versions are recorded in
`config/india-training-environment.json`; installing torchvision blindly can
otherwise replace a working CUDA build.

```bash
python -m pip install -e ".[data]"
python -m pip install --dry-run torchvision==0.29.0 timm==1.0.30 faster-whisper==1.2.1 pyarrow==26.0.0
python -m pip install torchvision==0.29.0 timm==1.0.30 faster-whisper==1.2.1 pyarrow==26.0.0
python -m pip install playwright==1.64.1
```

Those torchvision/PyTorch versions are a tested pair for this run, not a universal
CUDA installation command. Native GPU training uses eager PyTorch here; CUDA
Inductor/Triton compilation is not assumed available on this Windows environment.
Synthetic Hindi rendering uses an installed Chrome/Edge or a Playwright Chromium
installation (`python -m playwright install chromium`). The pinned Noto font and
OFL notice are downloaded by the generator. Incorrect unshaped Hindi is not used.

```text
config/india-training.json       committed run configuration
anulm/data/india_corpus.py       full pinned downloads and preparation
anulm/data/india_examples.py     controlled synthetic Indian-context tasks
anulm/training/india_run.py      logged orchestration using existing trainers
data/india-v1/raw/               complete selected source files
data/india-v1/media/             extracted images/audio
data/india-v1/text/              document groups, source attribution and splits
data/india-v1/tokens/            new tokenizer and memory-mapped token streams
runs/india-v1/                   preflight artifacts and text/joint checkpoints
logs/india-v1/                   status, per-stage logs and process records
.cache/huggingface/              pinned pretrained encoder cache
```

All runtime data, weights, logs and caches are ignored by Git. A 40 GiB free
disk reserve is enforced during downloads. Downloads resume only after the
HTTP server confirms the requested byte offset; complete files are hashed.
Tar members are copied by selected basename; links and arbitrary extraction
paths are never followed.

Resume source downloads with the same environment:

```bash
python -m anulm.data.india_corpus fetch
```

After those complete, rerun the orchestrator:

```bash
python -m anulm.training.india_run
```

For the hidden Windows launcher (reuses an existing download helper):

```powershell
.\scripts\windows\start_india_training.ps1 -PythonExecutable C:\path\to\environment\Scripts\python.exe
```

The initial real-data preflight passed text/decision-label, caption, English ASR
and Hindi ASR forward/backward updates on the local GPU. Peak allocated CUDA
memory was 1,721,712,640 bytes (about 1.6 GiB), not a worst-case full-training
memory measurement. It used the existing project tokenizer solely for a
mechanics check; actual training fits a new tokenizer. A second real preflight
checks all selected languages with that tokenizer before long training begins.
The native decoder source remains unchanged. Original core tests passed 66/66;
the supporting suites passed, including data, multimodal and process-budget checks.

The orchestrator refuses a second live owner of the same run. If a download helper is already
running, the launcher can pass its recorded process ID so orchestration waits
for completion. The launcher must first verify no other instance owns this run.
Per-stage logs explain failures; `status.json` records the current stage.
`source-provenance.json` records hashes of the actual locally modified runtime
source, not just its base Git commit. The runner refuses to begin training if
that source changes while it is preparing/waiting.
The existing trainers' atomic `.last` files carry optimizer state for recovery.
Process-tree termination enforces stage timeouts even with Windows virtualenv
launcher children; the current owned child stage is recorded in `active-child.json`.
Resumed joint runs reshuffle tasks, so they are not bit-exact replays.

The pipeline stops on download errors, insufficient data, failed preflight,
non-finite loss, timeout or resource errors. It does not silently substitute a
different model or pretend training started. No model/data uploads or publication
are part of this run. The existing draft contribution account/branch was verified; see docs/reviews/CONTRIBUTION_TARGET.md.

## License and quality boundaries

The [V2 workflow guide](V2_IMPROVEMENTS.md) documents optional accumulation and
the per-language/modality reporter. Training now generates a validation plan;
it does not automatically use final test sets. Default accumulation remains one.
Validation/report gates do not grant demo approval or publication permission.

FLEURS and DOCCI publish CC BY 4.0 terms. The pinned Wikipedia card states
CC BY-SA 3.0/GFDL conditions, including attribution/share-alike obligations;
those need to be retained and reviewed before distributing a commercial artifact.
This is not an exclusively CC-BY-only run. Whisper weights are MIT; the pinned
timm SigLIP weights are Apache 2.0; the Devanagari font is OFL 1.1. Keep source
cards, licenses, author attribution and transformation records with the run.
New data does not clear existing model lineage; this run does not warm-start
the decoder from the earlier checkpoints.

Forward/backward preflight confirms mechanics, not capability. Final evaluation
must report language-specific ASR errors, text loss, caption/VQA results, failures,
latency/memory and the provenance of every split. Synthetic tasks must be reported
separately. Demo entries remain unapproved until those results are reviewed.
