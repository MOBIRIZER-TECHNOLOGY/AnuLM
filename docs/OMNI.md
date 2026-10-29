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
pip install safetensors pyarrow "huggingface_hub>=1.5,<2" faster-whisper timm pillow av soundfile numpy
pip install gradio piper-tts            # only for the demo app
```

Two traps measured on the machine this was built on: a later `pip install`
can replace the CUDA build of torch with a CPU one (run `pip install --dry-run`
first and check it does not touch torch), and `huggingface_hub` 2.x breaks
`transformers` 5.x (keep it below 2).

## 2. Data

```bash
python omni_data.py list            # every source, its licence, size and status
python omni_data.py fetch all       # text and speech; skips what is already there
python omni_images.py fetch         # photos, captions, image questions (~26 GB)
python omni_data.py check && python omni_images.py check
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
6. **trains the one model**: 100,000 steps of seven tasks drawn at random,
   about 9-10 hours on the GPU above; the first run also caches the image
   features (~15 minutes)
7. evaluates every task (about 1.5 hours)
8. writes `logs/omni_one/REPORT.md`: every number beside the model it
   replaced, and whether each release gate passed
9. exports `release/AnuLM-Omni-400M` (safetensors, config, tokenizer, card)

**If anything stops it** -- a reboot, a crash, a full disk -- fix the cause
and run the same command. Finished steps are skipped and training resumes
from its last save (every 10,000 steps, about 50 minutes of work at most).
`logs/omni_one/pipeline.log` says what happened; each step has its own log
beside it.

## 4. Use it

```bash
python app_omni.py                  # every tab, one model
```

```python
from serve import Engine
chat = Engine("ckpt_omni.pt", "cuda")
print(chat.generate("What is the capital of India?", 40, 0.0, 1, 0, mode="question")["completion"])
```

## What is in the training mix, and why

| task | data | share of steps | batch |
|---|---|---|---|
| chat | dolly-15k, oasst2, Aya Hindi, Hindi Wikipedia QA | 7% | 8 |
| reading | SQuAD, MLQA hi, Wikipedia passages | 14% | 4 |
| intents | BANKING77, MASSIVE hi-IN | 6% | 8 |
| English speech | LibriSpeech train-clean-100 | 12% | 4 |
| Hindi speech | Kathbath, MUCS 2021, FLEURS | 32% | 4 |
| captions | Localized Narratives on Open Images, DOCCI | 17% | 8 |
| image questions | TextVQA | 12% | 8 |

Hindi speech gets the most steps because it has the most data and is the
weakest skill; chat the fewest because the model starts from the chat model
and only has to keep it. The shares are `TASKS` in `experiments/omni_one.py`;
change them there.

**Release gates** (`GATES` in the same file) say how much a task may lose
against the separate model it replaces before the run counts as a failure.
`--publish` uploads only if every gate passes.
