# AnuLM-Omni: train the one model yourself

AnuLM-Omni is one 456M decoder that chats, answers from passages, routes a
message to an intent, transcribes English and Hindi speech, describes a photo
and answers questions about it. This page takes you from a fresh clone to
that model on your own GPU, with every dataset fetched from its original
source. Every source allows commercial use (docs/DATASETS.md, NOTICE); the
model is CC BY-SA 4.0 (docs/MODEL_CARD_OMNI.md says what that means).

## What you need

| | minimum | what this was built on |
|---|---|---|
| GPU | 16 GB, CUDA (bf16) | RTX 5070 Ti, 16 GB |
| RAM | 32 GB | 64 GB |
| disk | ~150 GB free | |
| network | the downloads below, once | 2-10 MB/s |
| OS | Windows or Linux, Python 3.11+ | Windows 11, Python 3.14 |

Disk, roughly: Hindi speech 45 GB (Kathbath 16, MUCS 28, FLEURS 2), English
speech 6.5 GB, images 26 GB plus a 28 GB feature cache built on the first
training run, checkpoints 10 GB.

## 1. Install

```bash
git clone https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM && cd AnuLM
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install safetensors pyarrow "huggingface_hub>=1.5,<2" faster-whisper av soundfile numpy
pip install --no-deps torchvision==0.26.0 timm pillow   # --no-deps, or pip swaps in a newer, CPU torch
pip install gradio piper-tts            # only for the demo app
```

Two traps measured on the machine this was built on: a later `pip install`
can replace the CUDA build of torch with a CPU one (run `pip install --dry-run`
first and check it does not touch torch), and `huggingface_hub` 2.x breaks
`transformers` 5.x (keep it below 2).

## 2. Data

```bash
python -m anulm omni_data list            # every source, its licence, size and status
python -m anulm omni_data fetch all       # text and speech; skips what is already there
python -m anulm omni_images fetch         # photos, captions, image questions (~26 GB)
python -m anulm omni_data check && python -m anulm omni_images check
```

**One dataset needs you first.** Kathbath (AI4Bharat's Hindi read speech) is
gated: open https://huggingface.co/datasets/ai4bharat/Kathbath, accept its
terms, then `hf auth login`. Everything else downloads without an account.

Both scripts resume after an interruption: run the same command again.

## 3. Train, evaluate, export -- one command

```bash
python experiments/omni_one.py
```

On Windows, `experiments\run_omni_one.cmd` starts it detached. It runs, in order:

1. the data steps above (skipped if done)
2. the chat model: downloads the released base (toonist/AnuLM-Base-v2-400M)
   and fine-tunes it on the chat data, about 20 minutes. The base itself was
   pretrained for 1.9M steps; `experiments/v2_train.sh` and
   `experiments/build_v2.py` reproduce it, but that takes weeks on one GPU.
3. grows its vocabulary with the audio and image ids (`speech_vocab.py`)
4. waits for the GPU to be free
5. measures the chat model it starts from, as the baseline
6. **trains the one model**: 3,125 optimizer steps, each summing 8 batches
   drawn at random from the seven tasks (25,000 batches), about 6 hours on
   the GPU above; the first run also caches the image features (~15 minutes).
   The 2026-11-03 run took 12,500 steps (24.2 hours) and failed only the chat
   gate: chat held-out loss rose from the first evaluation while the other
   tasks trained. So the run is shorter, and a checkpoint is kept only if its
   chat loss is within `--max-ratio chat=1.049` of the first evaluation.
   One batch per step was tried first and drifted the shared decoder: an
   intent or image-question batch is ~30 answer tokens, too few to steer a
   full optimizer step, and chat held-out loss went 3.27 -> 5.39
7. evaluates every task (about 30 minutes)
8. writes `logs/omni_one/REPORT.md`: every number beside the model it
   replaced, and whether each release gate passed
9. exports `release/AnuLM-Omni-400M` (safetensors, config, tokenizer, card)
10. repeats the full evaluation on that exported bf16 folder; publishing is
    allowed only when these exported-model results pass every release gate

**If anything stops it** -- a reboot, a crash, a full disk -- fix the cause
and run the same command. Finished steps are skipped and training resumes
from its last save (every 625 steps, about 75 minutes of work at most).
If the training recipe (code, command or data) changes during a run, previous
outputs are preserved under `logs/omni_one/archive/` and the changed recipe
starts fresh. Once training has *finished*, a changed recipe stops the
pipeline instead of discarding the model: run with `--retrain` to archive it
and train again.
`logs/omni_one/pipeline.log` says what happened; each step has its own log
beside it.

## 4. Use it

```bash
python -m anulm app_omni                  # every tab, one model
```

```python
from anulm.inference.serve import Engine
chat = Engine("ckpt_omni.pt", "cuda")
print(chat.generate("What is the capital of India?", 40, 0.0, 1, 0, mode="question")["completion"])
```

## What is in the training mix, and why

| task | data | share of steps | batch |
|---|---|---|---|
| chat | dolly-15k, oasst2, Aya Hindi, Hindi Wikipedia QA | 21% | 8 |
| reading | SQuAD, MLQA hi, Wikipedia passages | 12% | 4 |
| intents | BANKING77, MASSIVE hi-IN | 15% | 8 |
| English speech | LibriSpeech train-clean-100 | 10% | 4 |
| Hindi speech | Kathbath, MUCS 2021, FLEURS | 22% | 4 |
| captions | Localized Narratives on Open Images, DOCCI | 11% | 8 |
| image questions | TextVQA | 10% | 8 |

Hindi speech gets the most steps because it has the most data and is the
weakest skill. Chat started at 7%: the model starts from the chat model and
only has to keep it, but at 7% the other tasks pulled chat off course, so it
was raised (with half the learning rate). Intents went from 5% to 15% on
2026-11-05 (below). The shares are `TASKS` in `experiments/omni_one.py`;
change them there.

### Runs so far, and what each changed

| run | recipe | kept | chat loss (<= 3.30) | intents (>= 80) | failed gates |
|---|---|---|---|---|---|
| 2026-11-03 | 12,500 steps, 24.2 h, intents 5% | step 9,375 | 3.475 | 87.7 | chat |
| 2026-11-05 a | 3,125 steps, `--max-ratio chat=1.049`, 3.9 h | step 2,500 | 3.298 | 69.7 | intents |
| 2026-11-05 b | as a, intents 15% (from Hindi speech and captions), 4.3 h | step 2,500 | 3.309 | 85.0 | chat, by 0.009 |
| run b, 5% blend | `experiments/blend_weights.py --alpha 0.05` | - | 3.299 | 85.7 | none, on the exported copy too: chat 3.2985, English WER 8.3%, FLEURS 41.5%, Hindi mean 42.2% |
| 2026-11-06, run 1 | as b, chat on `chat_train_v6.jsonl` (+13,163 human-written pairs, `experiments/make_chat_v6.py`), ~7 h | running | training curve 3.306 at step 2,500 (run b 3.376) | - | gates measured after training |
| 2026-11-06, run 2 | run 1 + `--data v2` (LibriSpeech 360 h, VizWiz answerable, 60k more captions, MLQA Hindi; `experiments/make_omni_v2_data.py`) | queued after run 1 | - | - | - |

- **Chat rises in the first 625 steps** (3.27 -> 3.37 on the training curve)
  and is then nearly flat, whatever the shares: the other tasks move the
  shared decoder. Distilling chat from its starting model did not slow it
  (`experiments/probe_distill.py`: 3.358 vs 3.351 at step 625).
- **The gate's chat loss is about 0.067 below the training curve's** (a
  different held-out set), so a curve above ~3.366 at the kept step fails.
- **Blending** mixes a fraction of the starting chat model's decoder
  weights back in (audio and image rows and the projectors stay as
  trained). On run b, 5% / 10% / 15% / 20% gave chat 3.299 / 3.290 / 3.282 /
  3.274, but 10% already put English speech at 9.2% (gate <= 9) and FLEURS
  at 42.7% (<= 43): speech uses the same weights. Results: `logs/blend/`.

**Release gates** (`GATES` in the same file) say how much a task may lose
against the separate model it replaces before the run counts as a failure.
`--publish` uploads only if every gate passes.
