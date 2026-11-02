# Language and modality support

The maintained [book language guide](../book/docs/reader/language-coverage.md)
records the 13 language configurations, eligible train/validation/test counts,
context exclusions and unsupported claims. It is included in the generated book.
Read [INDIA_TRAINING.md](INDIA_TRAINING.md) for pinned sources and execution,
and [V2_IMPROVEMENTS.md](V2_IMPROVEMENTS.md) for evaluation commands.

The completed run's output assessment is recorded in
[INDIA_MODEL_ASSESSMENT.md](reviews/INDIA_MODEL_ASSESSMENT.md). Final-step
validation WER was Hindi 94.9%, English 90.7%, with most regional languages near
100%; the loss-selected artifact's held-out tests were also poor. Native and
export checks finished, but no quality gate is approved. Synthetic Hindi/English
visual tasks are separate from general DOCCI English captions. Decoder/projectors
started randomly; Whisper/SigLIP remain frozen pretrained components. No speech
generation or comprehensive Indian-script vision claim is made.
