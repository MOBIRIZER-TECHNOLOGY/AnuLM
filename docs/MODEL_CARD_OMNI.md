# Model card: AnuLM-Omni-400M

One model for everything the AnuLM demo does: chat, answering from passages,
routing a message to an intent, English and Hindi speech recognition,
describing a photo and answering a question about it. One 456M decoder
(232M active per token), trained from scratch for Hindi, English and Python,
then fine-tuned on all seven tasks at once. Two small projectors sit beside it
and bring in two frozen, borrowed encoders: Whisper-small for audio and
SigLIP for images.

Part of AnuLM: code, data scripts and the demo app (`app_omni.py`) at
https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM.
Not affiliated with Sarvam AI, AI4Bharat, BharatGen or the Government of
India.

## What it is

| part | what | trained here? |
|---|---|---|
| decoder | AnuLM mixture-of-experts, 20 layers, hidden 1024, 61,445 ids (32,768 text + audio and image control ids) | yes: from scratch, then on the seven tasks |
| audio projector | 12.6M parameters: 4 Whisper frames -> one decoder position | yes |
| image projector | SigLIP patches pooled 2x2 -> decoder positions | yes |
| audio encoder | Whisper-small encoder (MIT), frozen, loaded by faster-whisper | no |
| image encoder | SigLIP `vit_base_patch16_siglip_224` (Apache 2.0), frozen, loaded by timm | no |

Until this model each skill was its own full copy of the decoder (five of
them); this is one.

## Measured

Numbers from `experiments/omni_one.py`'s report, beside the separate model
each task used before. WER in %, lower is better; F1 and accuracy higher is
better; losses lower is better.

<!-- MEASURED -->

## Training data

Every source allows commercial use; most require attribution, some
share-alike (docs/DATASETS.md, NOTICE).

| task | data | licence |
|---|---|---|
| pretraining (the base) | fineweb-edu, fineweb-2 Hindi (ODC-BY; Common Crawl terms), Hindi Wikipedia (CC BY-SA), Python from codeparrot-clean (mixed, see below), OpenCodeInstruct (CC BY 4.0), opc-sft-stage1 (see below) | |
| chat | databricks-dolly-15k (CC BY-SA 3.0), OpenAssistant oasst2 (Apache 2.0), Aya Hindi (Apache 2.0), Hindi Wikipedia QA pairs (CC BY-SA) | commercial OK |
| reading | SQuAD v1.1 (CC BY-SA 4.0), MLQA translate-train hi (CC BY-SA 3.0), Wikipedia passages | commercial OK, share-alike |
| intents | BANKING77 (CC BY 4.0), MASSIVE hi-IN (CC BY 4.0) | commercial OK |
| English speech | LibriSpeech train-clean-100 (CC BY 4.0) | commercial OK |
| Hindi speech | Kathbath (AI4Bharat, CC BY 4.0), MUCS 2021 Hindi-English (OpenSLR 104, CC BY-SA 4.0), FLEURS hi (CC BY 4.0) | commercial OK |
| captions | Localized Narratives (CC BY 4.0) on Open Images photos licensed CC BY 2.0 (authors in `ATTRIBUTION_open_images.csv`), DOCCI (CC BY 4.0) | commercial OK, attribution |
| image questions | TextVQA (CC BY 4.0) on Open Images photos | commercial OK, attribution |

Left out on purpose, because they do not allow commercial use or their terms
are unclear: Flickr30k, COCO photos, IndicTTS, and a synthetic Indian-banking
set written with an AI assistant.

## Licence: CC BY-SA 4.0

You may use it commercially, sell a product or service built on it, and
modify it, with attribution. If you distribute the weights (or a modified
version), you must do so under CC BY-SA 4.0 too, and you may not add terms
that take those rights away. Serving the model behind an API distributes no
weights. This is the licence the share-alike sources (Wikipedia, SQuAD, MLQA,
Dolly, MUCS) require; whether model weights are an adaptation of training data
is legally unsettled, and this release takes the cautious reading.

**Two open questions in the base model**, disclosed so you can judge them:
its Python data includes GitHub code under copyleft licences (GPL/AGPL)
through codeparrot-clean, and opc-sft-stage1, which mixes model-generated
answers, including some that come from ChatGPT conversations. Neither is
known to forbid this use; neither is settled. A base pretrained without them
would remove both.

## Intended use

A small, fully open model to study, fine-tune, and build demos and products
on, for Hindi and English. It runs on one consumer GPU, and on a CPU.

## Out of scope and limitations

- **Facts.** It is a 400M model: fluent and often wrong. Use reading with
  retrieved passages for facts, not chat from memory.
- **Speech.** Read and conversational speech in English and Hindi only;
  noisy audio, other languages and long recordings degrade it.
- **Images.** Captions follow Localized Narratives' and DOCCI's style; VQA is
  trained on text-in-image questions (TextVQA) and is weak on everything else.
- **No safety tuning** beyond what its chat data carries. Do not deploy it
  where a wrong answer causes harm without a human in the loop.

## Reproduce it

```bash
git clone https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM && cd AnuLM
python omni_data.py fetch all        # text and speech, from each primary source
python omni_images.py fetch          # photos, captions, VQA (~26 GB)
python experiments/omni_one.py       # chat model, joint training, evaluation, export
```

docs/OMNI.md has the hardware, disk and time each step needs.

## Citation

```bibtex
@software{anulm2026,
  title  = {AnuLM: a small, trainable mixture-of-experts language model in the shape of Sarvam 30B, trained from scratch on one GPU for Hindi, English and Python},
  author = {{The AnuLM authors}},
  year   = {2026},
  url    = {https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM}
}
```
