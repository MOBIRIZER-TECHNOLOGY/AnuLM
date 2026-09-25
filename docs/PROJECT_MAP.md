# Project map: what this is, how it fits together, and where the data comes from

`docs/ARCHITECTURE.md` explains how Sarvam 30B/105B are built and which of
those choices survived at nano scale. This file is the layer above: the whole
project as a system — every stage from a public dataset to a served model, which
file owns each stage, and what has actually been measured.

For the corpora themselves — links, licences, fetch commands, disk sizes — see
`docs/DATASETS.md`. For the experiment log with numbers, `docs/RESULTS.md`.

---

## 1. What the thing is

One 353M-parameter sparse mixture-of-experts transformer, carrying Sarvam's
architecture, trained from scratch on one consumer GPU. The `350m` preset:

| | |
| --- | --- |
| layers | 20 |
| hidden size | 1024 |
| attention | GQA, 16 heads (MLA is implemented and switchable) |
| experts | 12 routed, top-3, plus 1 shared; first layer dense |
| expert width | 384 (against 2816 for the dense FFN) |
| parameters | **353.3M total, 151.5M active (43%)** |
| context | 512 default, 2,048 in the `ctx2k` line |

Everything downstream — coder, translator, question answerer, speech, vision —
is this same backbone with a different corpus or a different head bolted on.
That is the project's one structural idea: hold the model fixed and change only
the data, so a difference in the number is attributable to the data.

## 2. The pipeline, stage by stage

```
  PUBLIC SOURCE          BUILD                     TRAIN                EVAL / SERVE
  ─────────────          ─────                     ─────                ────────────
  Wikimedia dumps   ─┐
  C4, fineweb-edu    ├─ fetch_*.py ─┐
  codeparrot         │              │
  code_exercises    ─┘              ├─ bpe.py ──── train.py ──┬─ eval_bench.py
                                    │  (tokenizer) (pretrain) │  eval_qa.py
  jsonl pair sets   ─── make_qa.py ─┘                         │  eval_code.py
                                                              │  eval_context.py
                                       finetune.py ───────────┤  eval_translate.py
                                       (SFT / LoRA)           │  eval_golden.py
                                                              │
  LibriSpeech        ─┐                                       ├─ eval_speech.py
  IndicTTS-Hindi     ├─ speech_data.py ── audio_codec.py ─────┤   (Whisper as judge)
                     │                    (SNAC / Whisper)    │
  Flickr8k          ─── vision_encoder.py prep ───────────────┤
                                                              │
                                                              └─ serve.py, app.py,
                                                                 voice.py, app_voice.py
```

**Fetch.** `fetch_hindi.py` streams Wikimedia bz2 dumps and stops at a byte
budget; `fetch_web.py` takes C4; `fetch_code.py` filters codeparrot for
hand-written-looking Python; `fetch_hf.py` pulls arbitrary Hugging Face parquet
shards. Nothing downloads a whole dump.

**Tokenize.** `bpe.py` trains the byte-pair vocabularies kept in `data/*.json` —
`hi16k` for Hindi, `multi32k` for the three-language base, `code32k` for the
coder line. These are the only data files committed, because the checkpoints
cannot be run without them.

**Pretrain.** `train.py`. Windows sampled without replacement in epochs, a
restartable sampler in the `.last` checkpoint, `--stop-at` so a multi-day run
proceeds in phases, grouped-GEMM MoE on CUDA. The scheduled-task wrappers
(`run_*_phase.cmd`, `_*_run.cmd`) resume it every 30 minutes so a crash costs at
most one `--eval-every` interval.

**Fine-tune.** `finetune.py` takes jsonl question/answer pairs, packs several
into each `block_size` row and masks the loss to the answer. `--speech` swaps in
audio pairs; `lora.py` provides adapters when full fine-tuning is too big.

**Evaluate.** One evaluator per claim: `eval_code.py` executes generated Python
for pass@1, `eval_translate.py` computes chrF on FLORES-200, `eval_qa.py` and
`eval_golden.py` score question answering, `eval_context.py` tests long context,
`eval_speech.py` scores speech with Whisper as judge.

**Serve.** `serve.py` is a stdlib-only HTTP server plus `web/index.html`;
`app.py` is the Gradio demo used by `demo_colab.ipynb` and `space/`;
`voice.py` and `app_voice.py` add the speech cascade; `export_hf.py` writes a
`transformers`-loadable folder.

## 3. The multimodal layer

Three ways a non-text modality reaches the backbone, and they differ in whether
the model has to *learn* the representation.

| path | file | representation | new parameters | status |
| --- | --- | --- | --- | --- |
| audio out (and in) | `audio_codec.py`, `speech_vocab.py` | SNAC codes as 28,672 new vocabulary ids | **58.7M** embeddings, from scratch | acoustics work, alignment does not |
| audio in | `speech_encoder.py` | frozen Whisper encoder, 50 Hz × 768, stacked 4→12.1 Hz | 12.6M projector | shape-checked only |
| image in | `vision_encoder.py` | frozen SigLIP ViT-B/16, 196 patches → 49 after 2×2 pooling | 12.6M projector | **works** |

`AnuLM.forward(..., embeds=)` is the single hook that makes the continuous paths
possible: twelve lines that bypass the embedding table so a projector's output
can enter the backbone directly, since a continuous feature has no id to look up.

The vocabulary layout, grown once by `speech_vocab.py` so no id ever shifts:

```
[0, V)                text tokens the checkpoint was trained on
[V, V + 28672)        SNAC codes, slot-major: V + slot*4096 + code
V + 28672 + 0..4      <|speech|> <|/speech|> <|text|> <|image|> <|/image|>
```

**Why vision works and speech does not** is the most useful thing this layer
taught. Captioning needs a 12.6M projector and emits ~12 tokens in a vocabulary
the model already speaks. Speech needs 58.7M embedding rows learned from nothing
and must emit ~600 tokens in a language it has never spoken, aligned to ~30 text
tokens with no alignment mechanism. Same backbone, same effort, opposite results.

## 4. Where the data comes from

Full table in `docs/DATASETS.md`. In brief, by stage:

- **Hindi/English text** — Wikimedia dumps (CC BY-SA), C4 and fineweb-edu (ODC-BY
  over Common Crawl).
- **Python** — codeparrot-clean (mixed source licences, includes GPL/AGPL),
  `code_exercises` (CC BY-NC-SA, ChatGPT-generated), OpenCodeInstruct (CC BY 4.0),
  glaive-code-assistant (Apache 2.0).
- **Translation pairs** — Samanantar and IIT Bombay en-hi, both non-commercial.
- **Speech** — LibriSpeech (CC BY 4.0) for English, IndicTTS-Hindi (CC BY 4.0)
  for Hindi. IndicVoices and Shrutilipi are the obvious next step and are gated.
- **Images** — Flickr8k.
- **Evaluation only, never trained on** — HumanEval (MIT), MBPP (CC BY 4.0),
  FLORES-200 (CC BY-SA).

Two consequences worth stating plainly. First, **no corpus is redistributed**:
`data/` is gitignored and every set is rebuilt from its source, which is what
makes each checkpoint reproducible. Second, **the corpus licences constrain the
checkpoints, not the code** — the code is Apache 2.0, while the non-commercial
terms on Samanantar and `code_exercises` are why the released coder is
CC BY-NC-SA and the translator CC BY-NC. `NOTICE` carries that mapping and
`docs/PUBLISHING.md` §1 explains how each release licence was chosen.

Pretrained components used but not trained here: Whisper-small (ASR and judge),
SNAC 24 kHz (codec), Piper voices (TTS), SigLIP ViT-B/16 (vision tower). The
ear, the mouth and the eye are other people's models; only the sentence in the
middle is this project's.

## 5. What is measured

| claim | number | where |
| --- | --- | --- |
| Python | MBPP 12.5%, HumanEval 4.9% | `docs/RESULTS.md` §25 |
| en↔hi translation | chrF 41.5 / 43.4 on FLORES-200 | §24 |
| question answering | three languages | §23 |
| long context | 2,048 tokens | §28 |
| textbook corpus A/B | **unresolved** — 0.0% in continue mode is a format confound | §31, `docs/SPEECH.md` |
| image captioning | held-out loss 2.434; ~5/10 held-out captions correct | `docs/SPEECH.md` |
| speech generation | WER 100–131%; audible and speaker-matched, words wrong | `docs/SPEECH.md` |
| the speech cascade | 3–6 s per spoken turn, fully local | `docs/SPEECH.md` |

Negative results are kept deliberately. The codec floor was measured (Whisper
3.78% on real audio, 7.45% after a SNAC round trip) before any model was judged
against it; the epoch-4.7 ceiling was found twice on different corpora; and
cross-entropy was shown to move *opposite* to audio quality, which is why
`eval_speech.py` now reports loudness alongside word error rate.

## 6. Reading order

1. `README.md` — what exists and how to load a checkpoint.
2. `docs/TUTORIAL.md` — a fresh machine to a trained model, in order, with times.
3. `docs/ARCHITECTURE.md` — the model internals, and §10 for what held up.
4. This file — how the pieces connect.
5. `docs/DATASETS.md` — every source and its licence.
6. `docs/RESULTS.md` — every experiment with its number.
7. `docs/DEVELOPING.md` — which script needs which dependency.
8. `TODO.md` and `CONTRIBUTING.md` — the open work.
