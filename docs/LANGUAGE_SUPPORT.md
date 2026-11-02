# Language and modality support

The maintained [book language guide](../book/docs/reader/language-coverage.md)
records the 13 language configurations, eligible train/validation/test counts,
context exclusions and unsupported claims. It is included in the generated book.
Read [INDIA_TRAINING.md](INDIA_TRAINING.md) for pinned sources and execution,
and [V2_IMPROVEMENTS.md](V2_IMPROVEMENTS.md) for evaluation commands.

Prepared data and a passing preflight do not establish trained quality. Regional
speech validation losses currently need regression review; final WER and caption/
VQA/text quality results remain pending. Synthetic Hindi/English visual tasks
are separate from general DOCCI English captions. The decoder/projectors start
randomly; Whisper/SigLIP remain frozen pretrained components. No speech-generation
or comprehensive Indian-script vision claim is made.
