# AnuLM development and training rules

- Build on this repository's existing AnuLM implementation. Do not replace
  working architecture or reorganize public imports without a concrete need,
  compatibility plan and regression checks. Preserve old checkpoints.
- Organize runtime code in `anulm/`, tests in `tests/`, guides in `docs/`,
  versioned configuration in `config/`, manuscript material in `book/`.
  Put generated data in `data/`, checkpoints in `runs/`, logs in `logs/`.
  Do not commit corpora, weights, local environments, secrets or temporary files.
- Keep Decide on AnuLM's own backbone/adapters. Do not add comparison-model
  downloads or choices to the active demo, API or training run.
- Treat India-focused language coverage explicitly. Record each language,
  source, amount, split and task; include Hindi, English and selected regional
  languages in evaluation. Do not call general photos India-specific data.
  Label procedural/synthetic Indian-context examples as synthetic.
- Pin dataset/model versions and preserve source URLs, hashes, license text,
  attribution and transformation records. Exclude non-commercial or unclear
  sources from a commercial-intended run until reviewed. Commercial permission
  is conditional on source terms; new data cannot clear old checkpoint lineage.
- Split by document/media/input groups before training. Deduplicate across
  training and evaluation; never fit tokenizers or train on held-out data.
  Keep validation for model selection and test for final reporting.
- Distinguish a randomly initialized decoder/projectors from pretrained image
  and audio encoders. Do not claim all encoders were trained from scratch.
  Synthetic tests, declining loss and dataset downloads do not prove capability.
- Measure actual GPU memory. Choose model/batch/context settings that fit;
  preserve free disk space. Long jobs need logs, status, atomic checkpoints,
  recovery instructions, finite-loss checks and a bounded run duration.
- Run relevant CPU correctness tests and a real-data forward/backward preflight
  before an unattended training stage. Report OOMs, missing data and failed
  stages honestly. Do not restart repeatedly with unrecorded configuration changes.
- Default demos expose approved recorded cases only, with matching artifacts
  and reviewer evidence. Live research controls require explicit opt-in.
- Keep production and quality claims tied to repeatable held-out measurements,
  including per-language and per-modality results. Do not promise world-class
  quality merely because a run or a test suite passed.
- Keep contributions local until the intended publishing account and branch
  are confirmed. Do not publish models or upload user data without authorization.
