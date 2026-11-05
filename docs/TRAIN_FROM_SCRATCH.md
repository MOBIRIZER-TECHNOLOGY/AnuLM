# Train AnuLM-Omni-400M yourself, from nothing

This guide is for you if you have cloned the repository and want to train
the model yourself, not just download it. AnuLM-Omni-400M is one decoder
that chats, answers from a passage, routes a message to an intent,
transcribes English and Hindi speech, and describes and answers questions
about photos.

You do not have to do everything at once. There are three routes. Pick the
one your machine and your patience allow, and move up later.

Every command below exists in this repository with the flags shown. Every
number comes from a log or a document named next to it. Where something was
not recorded, the guide says so.

## The three routes

| | A. Learn the loop | B. Omni from the released base | C. Everything from nothing |
|---|---|---|---|
| what you train | a 17M toy model, or a small Hindi model | the chat model and the Omni model, on top of the released pretrained base | your own tokenizer files, your own 400M base, then everything in B |
| GPU | none needed | NVIDIA, 16 GB VRAM, CUDA, bf16 | the same |
| RAM | 8 GB | 32 GB (64 GB used here) | the same |
| disk | under 1 GB | about 150 GB | about 150 GB + about 85 GB |
| downloads | small: 1 MB of Shakespeare, part of a Wikipedia dump | about 60 GB, one gated set | about 60 GB + about 50 GB |
| time | 30 s (GPU) to 30 min (CPU) | download time + about 6 h of GPU | B + about 7 days of GPU for the base |
| what you get | a model that writes Shakespeare-shaped or Hindi-shaped text | `ckpt_omni.pt`, an exported folder, and a report card | the same, with a base you trained |

Sources for the table: route A from `README.md` and `docs/TUTORIAL.md` §0,
§3 and §4. Route B from `docs/OMNI.md` "What you need" and the logs of the
latest run in `logs/omni_one/`. Route C's extra disk was measured on the
training machine (raw parquet 51 GB, Python text 3.8 GB, token files 22 GB,
checkpoints 6.5 GB). The base's GPU time comes from `v2_train_run.log`:
1.9M steps at 300 to 340 ms each.

Every 400M run here used one RTX 5070 Ti (16 GB) on Windows 11 with Python
3.14.7 and torch 2.11.0+cu128. Smaller cards were not tested for routes B
and C.

---

## Setup (all routes)

### Python and torch

Use Python 3.11 or newer. `pyproject.toml` allows 3.10, but `docs/OMNI.md`
asks for 3.11+ for the Omni route.

```bash
git clone https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM && cd AnuLM
python -m venv .venv
# Windows: .venv\Scripts\activate        Linux: source .venv/bin/activate
pip install --upgrade pip

# NVIDIA GPU (match your driver at https://pytorch.org/get-started/locally/):
pip install torch --index-url https://download.pytorch.org/whl/cu128
# CPU only (route A):
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

With an NVIDIA card, the last line must print `True`. If it prints
`False`, you have the CPU wheel: reinstall from the CUDA index. The fast MoE
path (`--moe-impl grouped`) needs torch 2.11 or newer.

### Packages for routes B and C

```bash
pip install --dry-run safetensors pyarrow pandas numpy tokenizers "huggingface_hub>=1.5,<2" faster-whisper av soundfile
# read the dry-run output: it must not list torch. Then run it without --dry-run.

pip install --no-deps torchvision==0.26.0 timm pillow     # images; --no-deps on purpose
pip install --dry-run gradio piper-tts                    # only for the demo app; check, then install
```

**Warning: a later `pip install` can replace your CUDA torch.** On the
training machine, installing `torchvision` (or `open_clip_torch`) pulled in a
newer CPU-only torch and training silently stopped seeing the GPU. So:

- always run `pip install --dry-run ...` first and check it does not touch
  torch;
- install `torchvision`, `timm` and `pillow` with `--no-deps`, as above
  (`requirements.txt` explains the pin: torchvision 0.26 pairs with torch
  2.11);
- keep `huggingface_hub` below 2 (`docs/OMNI.md`).

After any install, run the `torch.cuda.is_available()` check again.

### Keeping a job off the GPU

Sometimes you want to run something on the CPU while the GPU trains, for
example a `--help` or a test. Hide the GPU with `-1`, not an empty string:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m tests.test_model
```

On Windows Git Bash, `CUDA_VISIBLE_DEVICES=""` does **not** hide the card.
A second job on the same card can halve the speed of both.

### Windows notes

- The `.sh` scripts need Git Bash. The `bash.exe` Windows puts on PATH is
  the WSL stub (`docs/TUTORIAL.md` §1).
- If Smart App Control is on, it may block some DLLs. The project avoids
  `transformers` for this reason (`requirements.txt`, "Deliberately
  absent").

---

## Route A: learn the training loop (minutes, no GPU needed)

Start here even if you plan route C. It checks your install and shows you
what a training log looks like.

```bash
python -m anulm quickstart            # checks Python, torch, CUDA and packages; prints fixes
python -m anulm quickstart --train    # trains a 17M model for 200 steps and samples it
```

`--train` takes about 30 seconds on a GPU and 3 to 6 minutes on a CPU
(`README.md`). It downloads tinyshakespeare (1 MB). You get text that looks
like Shakespeare but means nothing. That is the point: the whole loop works.

Next, a Hindi model from scratch, `docs/TUTORIAL.md` §4 (about 30 minutes,
CPU or GPU):

```bash
python -m anulm fetch_hindi --mb 8 --out data/hindi.txt
python -m anulm train --preset 30b --data data/hindi.txt --steps 1000
python -m anulm sample --ckpt ckpt.pt --prompt "भारत"
```

`docs/TUTORIAL.md` §4 then adds your own BPE tokenizer. Do that too: route C
uses the same `bpe.py`, only bigger.

---

## Route B: Omni from the released base (about 6 hours of GPU)

This skips pretraining. You start from `toonist/AnuLM-Base-v2-400M`, which
the pipeline downloads for you. Everything else is trained on your machine.

### B1. Get the data (bandwidth-bound)

**One dataset needs you first.** Kathbath (Hindi read speech) is gated.
Open https://huggingface.co/datasets/ai4bharat/Kathbath, accept its terms,
then log in:

```bash
hf auth login
```

Then fetch everything. Each command skips what is already there, and a
rerun resumes after a dropped connection.

```bash
python -m anulm omni_data list          # every source, its licence, size, and present/missing
python -m anulm omni_data fetch all     # chat, reading, intents, English + Hindi speech
python -m anulm omni_images fetch       # photos, captions, image questions (~26 GB)
python -m anulm omni_data check         # must end with "all outputs look right"
python -m anulm omni_images check
```

Sizes (from `omni_data.py` and `docs/OMNI.md`): Kathbath 16.6 GB of parquet
plus 16 GB of clips; MUCS 7.8 GB of tarballs that grow to 28 GB; LibriSpeech
6.4 GB; FLEURS 1.6 GB plus 1.9 GB; images about 26 GB; text under 1 GB.
Training later adds a feature cache of about 28 GB. How long the downloads
take depends on your connection; a single end-to-end download time was not
recorded here.

**Run `check` before you train. Do not skip it.** The evaluation and the
demo read some files from `data/` directly. If they are missing, nothing
crashes at training time: the reading score is silently "not measured",
and the demo's intent tab crashes when you open it. This happened in the
latest run (see "What to expect" below).

Every file the pipeline, the evaluation and the demo read, and the command
that makes it:

| file(s) | made by | used for |
|---|---|---|
| `data/chat_train.jsonl`, `data/chat_heldout.jsonl` (+ `dolly_*`, `qa_lora.jsonl`) | `omni_data fetch chat` | chat training; the chat gate |
| `data/rc_train.jsonl`, `data/rc_heldout.jsonl` | `omni_data fetch read` | reading training |
| `data/rc_eval_squad_en.jsonl`, `data/rc_eval_mlqa_hi.jsonl` (+ `rc_eval_xquad_hi`, `rc_eval_indicqa_hi`) | `omni_data fetch read` | **reading F1 in the evaluation** |
| `data/rag/wiki/20231101.{hi,simple}/*.parquet`, `data/rag/meta.json` and the BM25 index | `omni_data fetch read` (it runs `rag build`) | reading passages and hard negatives |
| `data/banking77/{train,test}.csv`, `data/banking77/categories.json` | `omni_data fetch banking77` | intent data; **the demo's intent labels** |
| `data/decide_general/massive_{hi-IN,en-US}/{train,validation}.jsonl` | `omni_data fetch massive_hi` | intent data; **the demo's Hindi intent labels** |
| `data/omni/intent_train.jsonl`, `data/omni/intent_test.jsonl` | `omni_data fetch intent` | intent training; intent accuracy |
| `data/speech_ls100.jsonl` | `omni_data fetch librispeech` | English speech; English WER |
| `data/fleurs_hi/{train,dev,test,devtest}.jsonl` | `omni_data fetch fleurs_hi` | Hindi speech; FLEURS WER |
| `data/kathbath_hi/{train,test}.jsonl` | `omni_data fetch kathbath_hi` (gated) | Hindi speech; Kathbath WER |
| `data/mucs_hi_en/{train,test}.jsonl` | `omni_data fetch mucs_hi` | Hindi speech; MUCS WER |
| `data/omni/hindi_train.jsonl` | `omni_data fetch hindi_mix` | the joined Hindi speech set (146,350 clips) |
| `data/omni/caption_{train,test}.jsonl`, `data/omni/vqa_{train,test}.jsonl`, `data/omni_img/` | `omni_images fetch` | captions, image questions; VQA accuracy |
| `data/multi32k.json` | already in the repository | the tokenizer |

`fetch all` runs every row above except the images. The speech manifests
store absolute paths to the audio, so if you move the checkout, run
`omni_data check`: it samples the paths and reports missing audio.

### B2. Train, evaluate, export: one command

```bash
python experiments/omni_one.py
```

On Windows, `experiments\run_omni_one.cmd` starts the same thing in its own
minimised window and logs to `logs\omni_one\stdout.log`. It does **not**
publish anything. Only `--publish` uploads, and only if every release gate
passes on the exported model.

What it does, in order, with the times measured in the latest run
(`logs/omni_one/pipeline.log`, 5 November 2026):

| step | what it does | time |
|---|---|---|
| 1 | the data steps from B1 (skipped if done) | your bandwidth |
| 2 | no `ckpt_chat_v3.pt`? Downloads the released base and fine-tunes it on `data/chat_train.jsonl` into `ckpt_chat_v3.pt` | about 13 min (747 s in `logs/post_v2/2_chat_sft.log`) |
| 3 | grows the vocabulary with audio and image ids: `ckpt_chat_mm.pt` | 5 s |
| 4 | waits until no other process holds more than 1 GB of the GPU | 6 min here |
| 5 | scores the chat model it starts from, as the baseline | about 1.5 min |
| 6 | **trains**: 3,125 optimizer steps, each summing 8 batches drawn from seven tasks, into `ckpt_omni.pt` | **about 4.3 h** (15,540 s), plus about 11 min to fingerprint the data first |
| 7 | evaluates: WER on four speech sets, reading F1, intent accuracy, chat loss, VQA, captions | about 31 min |
| 8 | writes `logs/omni_one/REPORT.md` with every number and each gate, pass or fail | seconds |
| 9 | exports `release/AnuLM-Omni-400M` (safetensors, config, tokenizer, card) | seconds |
| 10 | runs the whole evaluation again on the exported folder | about 32 min |

The 3 November run trained four times as long (12,500 steps, 24.2 hours)
and failed the chat gate; `docs/OMNI.md` ("Runs so far") says why the run
is now shorter and what each run measured.

**If it stops** (reboot, crash, full disk, power cut): fix the cause, then
run the same command again. That is the whole recovery. Finished steps are
recorded in `logs/omni_one/state.json` and skipped. Training resumes from
`ckpt_omni.pt.last`, saved at every eval (every 625 steps, about 73 minutes
of work in the latest run). If you change the training code, the command or
the data, the pipeline notices (a content fingerprint), moves the old
outputs to `logs/omni_one/archive/` and starts training fresh.

Where to look while it runs: `logs/omni_one/pipeline.log` for the steps,
`logs/omni_one/6_train.log` for training. Each step has its own log beside
them.

**One difference from the official run.** That run started the audio
projector from `ckpt_asr_en_v3.pt`, an earlier English speech model trained
on LibriSpeech. A fresh clone does not have that file, so the projector
starts fresh (`omni_one.py` checks for it). Your speech numbers may differ
somewhat from the table below; how much was not measured.

### B3. What to expect

From `logs/omni_one/results.json` (the latest run, 4 November 2026). The
gates are `GATES` in `experiments/omni_one.py`.

| measure | result | gate | pass? |
|---|---|---|---|
| English WER, LibriSpeech held-out (100 clips) | 5.5% | ≤ 9.0 | yes |
| Hindi WER, FLEURS | 29.6% | ≤ 43.0 | yes |
| Hindi WER, Kathbath | 20.3% | | |
| Hindi WER, MUCS | 46.1% | | |
| Hindi WER, mean of the three | 32.0% | ≤ 50.9 | yes |
| intent accuracy, BANKING77 | 87.67% | ≥ 80 | yes |
| intent accuracy, MASSIVE hi-IN | 80.33% | | |
| reading F1, SQuAD (English) | 48.4 | ≥ 30 | yes |
| reading F1, MLQA (Hindi) | 11.7 | | |
| VQA accuracy | 17.3% | ≥ 10 | yes |
| caption held-out loss | 2.7249 | | |
| **chat held-out loss** | **3.4754** | **≤ 3.30** | **no** |

WER and loss: lower is better. F1 and accuracy: higher is better.

**The reading numbers are not in `REPORT.md`.** When the run evaluated,
`data/rc_eval_squad_en.jsonl` and `data/rc_eval_mlqa_hi.jsonl` were missing,
so the report shows "-" and counts the reading gate as failed. They were
measured afterwards (`logs/omni_one/eval_reading.json`) and added to
`results.json`. This is why B1 tells you to run `omni_data check` first.

**The chat gate failed, and the model was not published.** The chat model it
started from scored 3.146 on the same measure. Why it got worse: the Omni run
re-reads `data/chat_train.jsonl`, the same 22,873 examples the starting chat
model had already been tuned on for two epochs. Over 12,500 steps, chat got
21,132 batches of 8, about seven more passes over that data. The model
memorised it, and the loss on held-out chat rose steadily while every other
task improved: in `6_train.log`, chat held-out goes from 3.27 at step 0 to
3.58 at step 12,500. The share of chat steps and the learning rate are the
two levers already tried (comments above `LR` and `TASKS` in
`omni_one.py`). This is an open problem, not a bug in your setup. If your
run fails the same gate, you reproduced it.

Read the five caption samples at the end of `REPORT.md` as well. Numbers do
not tell you whether the model describes the right animal.

---

## Route C: everything from nothing (about a week of GPU, then route B)

Route C replaces the released base with one you pretrain yourself. After
stage C5 you have your own `ckpt_chat_v3.pt`, and route B's pipeline uses it
instead of downloading anything.

Be honest with yourself about the cost. The base read about 7.8 billion
tokens. That is about 7 days of uninterrupted GPU time on an RTX 5070 Ti.
The real run took 19 calendar days (28 September to 17 October 2026),
because of a shutdown, a reboot schedule and the GPU being needed for other
work (`docs/MODEL_CARD_BASE_V2.md`, `experiments/v2_train.sh`).

One more honest note: the base corpus was not built by one script. Stages C2
and C3 put together the downloads that `experiments/build_v2.py` expects.
Where the exact command used on the training machine was not recorded, the
guide says so.

### C1. Tokenizer

**What it does.** Turns text into token ids. AnuLM uses its own byte-level
BPE (`anulm/bpe.py`) with 32,768 entries, `multi32k`, for Hindi, English and
Python.

**Recommended: use the committed one.** `data/multi32k.json` is in the
repository. It is the exact tokenizer of every released 400M model. If you
train a new one, nothing released will work with it, and you must redo every
stage below.

`build_v2.py` reads the tokenizer in the Rust `tokenizers` format. Convert
it (this checks the two agree token for token before it writes anything):

```bash
python -m anulm.hf.hf_tokenizer data/multi32k.json release/AnuLM-Base-400M
```

- **Input:** `data/multi32k.json`. **Output:** `release/AnuLM-Base-400M/tokenizer.json`
  and `tokenizer_config.json`.
- **Check:** it prints `verified token-for-token on N probes`. If it
  disagrees, it raises an error and writes no config.
- **Time:** seconds.

**If you want to train it yourself anyway** (to learn how): `experiments/build_multi.sh`
step 3 did it with `python -m anulm bpe train --data data/multi_toksample.txt --vocab 32768 --out data/multi32k.json`
on a 550 MB sample (300 MB Hindi, 150 MB English, 100 MB Python). That took
588 s (`docs/RESULTS.md` §22). Building that sample needs the earlier
corpora of `experiments/build_mixed.sh` and `build_multi.sh`
(`docs/TUTORIAL.md` §5 and §6). Write it to a different file name so you do
not overwrite the committed one.

### C2. Download the pretraining sources

**What it does.** Fetches the raw text: English and Hindi web pages, Hindi
Wikipedia, and Python. `fetch_hf` skips files already downloaded at the right
size, so a rerun resumes.

```bash
# English: fineweb-edu sample/10BT, all 14 shards (27 GB on disk)
python -m anulm fetch_hf HuggingFaceFW/fineweb-edu --prefix sample/10BT/000_ sample/10BT/001_ sample/10BT/002_ sample/10BT/003_ sample/10BT/004_ sample/10BT/005_ sample/10BT/006_ sample/10BT/007_ sample/10BT/008_ sample/10BT/009_ sample/10BT/010_ sample/10BT/011_ sample/10BT/012_ sample/10BT/013_

# Hindi web: fineweb-2 hin_Deva, shards 000-003 (19 GB)
python -m anulm fetch_hf HuggingFaceFW/fineweb-2 --prefix data/hin_Deva/train/000_ data/hin_Deva/train/001_ data/hin_Deva/train/002_ data/hin_Deva/train/003_

# Hindi Wikipedia (20231101.hi parquet): fetched by the reading data of route B
python -m anulm omni_data fetch read

# Benchmarks, needed only so the Python sources can be decontaminated against them
python -m anulm fetch_hf openai/openai_humaneval --prefix openai_humaneval/
python -m anulm fetch_hf google-research-datasets/mbpp --prefix sanitized/

# Python, model-written exercises: OpenCodeInstruct (20 shards on the training machine)
python -m anulm fetch_hf nvidia/OpenCodeInstruct --prefix data/train-0000 data/train-0001

# Python, OpenCoder opc-sft-stage1 (the 8 files on the training machine)
python -m anulm fetch_hf OpenCoder-LLM/opc-sft-stage1 --prefix data/filtered_infinity_instruct- data/largescale_diverse_instruct-00000 data/largescale_diverse_instruct-00001 data/largescale_diverse_instruct-00002 data/largescale_diverse_instruct-00003

# Python from GitHub: codeparrot-clean, about 730 MB of hand-written-looking files
python -m anulm fetch_code --mb 730 --out data/gh_python.txt
```

Not recorded: the exact `--prefix` lists for fineweb-2, OpenCodeInstruct and
opc-sft-stage1, and the exact `fetch_code` command behind
`data/gh_python.txt` (734 MB on the training machine). The prefixes above
match the files that are on the training machine's disk. `fetch_code` reads
shard 1 by default (about 1 GB of Python per shard, before filtering); if it
stops short of 730 MB, add `--shards 1 2`.

Do **not** fetch `jinaai/code_exercises` for this. Its CC BY-NC-SA terms
would make the whole base non-commercial (`docs/DATASETS.md`).

Then turn the Python sources into decontaminated text. These are the
commands from `experiments/textbook_build.sh`. Run them one by one; do not
run that whole script, because it also builds a non-commercial file and an
unrelated mix.

```bash
python -m anulm convert_parquet textbook --in "data/raw/nvidia__OpenCodeInstruct/data/*.parquet" --q-col input --a-col output --python-only --decontaminate --score-col average_test_score --min-pair-score 1.0 --out data/tb_opencode.txt
python -m anulm convert_parquet textbook --in "data/raw/OpenCoder-LLM__opc-sft-stage1/data/*.parquet" --q-col instruction --a-col output --python-only --decontaminate --out data/tb_opc.txt
python -m anulm convert_parquet clean --in data/gh_python.txt --out data/gh_python.clean.txt
```

- **Outputs:** `data/raw/...` parquet, `data/rag/wiki/20231101.hi/`,
  `data/tb_opencode.txt` (1.1 GB here), `data/tb_opc.txt` (1.2 GB),
  `data/gh_python.clean.txt` (0.73 GB).
- **Check:** each `convert_parquet` run prints how many benchmark documents
  it dropped. `textbook_build.sh` records 13,562 for OpenCodeInstruct and
  14,378 for opc-sft-stage1.
- **Time:** downloads depend on your connection; the conversions were not
  timed separately (`textbook_build.sh` says "CPU, ~1 h after the downloads"
  for its whole build).

### C3. Build the token corpus

**What it does.** Reads every source above, encodes it with `multi32k`, and
writes one long file of 16-bit token ids that training memory-maps. One
document in a hundred (chosen by a hash) goes to validation.

```bash
python experiments/build_v2.py
```

- **Inputs:** C1's `tokenizer.json` and C2's files.
- **Outputs:** `data/v2/train.u16` (22 GB), `data/v2/val.u16` (216 MB),
  `data/v2/meta.json`.
- **Check:** open `meta.json`. The training machine's says 11,312,144,198
  train tokens: English 6.00B, Hindi web 4.50B, Hindi Wikipedia 0.066B,
  Python 0.746B. Your Python number will differ a little if your Python
  files differ.
- **Time:** not recorded. The encoder runs at 20 to 36 MB/s
  (`build_v2.py` docstring) over about 72 GB of text, so expect at least an
  hour.
- **Resume:** none. It rewrites the files from the start; rerun it.

About `--scale`: a smoke build with `--scale 0.001` is not small. It scales
the web budgets to 10.5M tokens, but the Wikipedia and Python budgets mean
"all" (1e12), so even at 0.001 they are read in full, about 0.8B tokens. Use
`--out data/v2_smoke` if you try it, so it does not overwrite `data/v2`.

### C4. Pretrain the base (about 7 days of GPU)

**What it does.** Trains the 400M mixture-of-experts model (397.7M
parameters, 173.5M active per token) from random weights on the corpus, one
1,024-token window at a time, 4 windows per step, for 1,900,000 steps.

Run it from Git Bash on Windows, or any shell on Linux:

```bash
bash experiments/v2_train.sh 2000     # a first test: stop at step 2,000, eval and save (about 12 min)
bash experiments/v2_train.sh          # the real run; resumes from the test
```

- **Inputs:** `data/v2/meta.json` and the two `.u16` files.
- **Outputs:** `ckpt_base_v2.pt` (best validation, 1.6 GB) and
  `ckpt_base_v2.pt.last` (the resume point with optimizer state, 4.9 GB),
  both written at every eval, every 10,000 steps.
- **Check early:** the log should show about `params: 397.66M total, 173.53M active/token`.
  On the training machine the loss was 10.56 at step 0, 6.48 at step 1,000
  and 4.78 at step 6,000, at about 12,700 tokens/s and 300 to 340 ms per
  step (`v2_train_run.log`). If your steps are much slower than that on a
  similar card, see the warning below.
- **Check at the end:** `val loss 3.3523 +/- 0.0491 | bits/byte 0.756` at
  step 1,900,000 (`docs/MODEL_CARD_BASE_V2.md`). Within about ±0.05 is a
  reproduction.
- **Resume:** run `bash experiments/v2_train.sh` again. It adds `--resume`
  by itself when `ckpt_base_v2.pt.last` exists. At most 10,000 steps (about
  50 minutes) are lost.

Things that went wrong on the training machine, so they need not go wrong
for you:

- **The GPU must be free.** Training uses about 15.7 GB of the 16 GB card
  with no gradient checkpointing. When other apps took 5.5 GB, the overflow
  went to system RAM and a step took 13 s instead of 0.3 s. Freeing the GPU
  did not help until training was restarted. Close other GPU programs
  before you start.
- **A smaller card:** the script has no `--grad-ckpt`. `train.py` accepts
  it; add it to `COMMON` in a copy of the script. It costs about 30% of the
  speed (`v2_train.sh` comments). Not tested for this run.
- **Back up `.last`** every few hours if your machine is unreliable. Copy it
  only while training is stopped or well after a save: on Windows, a copy
  that overlaps the save can crash training.
- **The schedule is fixed by the flags.** `--steps 1900000 --decay-frac 0.1631579`
  means the learning rate stays at 6e-4 until step 1,590,000 and then falls
  linearly to 6e-5. The run was planned for 2.75M steps and shortened; this
  is the shortened recipe that produced the released base. If you change
  `--steps`, the decay moves with it.

### C5. Chat fine-tune (about 13 minutes)

**What it does.** Teaches the base to answer instead of only continuing
text, on 23,022 chat examples (Dolly-15k, oasst2, Aya Hindi, Hindi
Wikipedia QA). This is exactly what `experiments/archive/post_v2.py` ran to
make chat v3.

```bash
python -m anulm omni_data fetch chat      # if B1 has not already made it
python -m anulm finetune --ckpt ckpt_base_v2.pt --qa data/chat_train.jsonl --heldout data/chat_heldout.jsonl --out ckpt_chat_v3.pt --epochs 2 --batch-size 4 --lr 5e-5 --grad-ckpt --eval-every 500 --log-every 100
```

- **Output:** `ckpt_chat_v3.pt` (1.6 GB).
- **Check:** `logs/post_v2/2_chat_sft.log` on the training machine reads
  `held-out answer loss 3.4415 -> 3.1445` after 1,844 steps.
- **Time:** 747 s.
- **Resume:** `finetune` has `--resume`; at 13 minutes, rerunning is simpler.

### C6. Omni

Now do all of route B. Because `ckpt_chat_v3.pt` exists, `omni_one.py`
skips its step 2 (no download, no second fine-tune) and builds on your base.

```bash
python experiments/omni_one.py
```

The checks and expected numbers are in B3. Your numbers will move with your
base; if your base reached about 3.35, they should land near B3's.

---

## Licensing: what the Omni recipe may and may not learn from

AnuLM-Omni-400M is meant for commercial use, so its own training mix (route
B) uses only data whose terms allow that. `docs/DATASETS.md` lists every
source with its link and licence; `NOTICE` §2 is the authoritative table.
`python -m anulm omni_data list` prints each source's licence.

- **Chat:** Dolly-15k (CC BY-SA 3.0), oasst2 (Apache 2.0), Aya Hindi rows
  (Apache 2.0), Hindi Wikipedia QA (CC BY-SA 4.0).
- **Reading:** SQuAD (CC BY-SA 4.0), MLQA hi (CC BY-SA 3.0), Wikipedia
  passages (CC BY-SA 4.0).
- **Intents:** BANKING77 and MASSIVE (CC BY 4.0).
- **Speech:** LibriSpeech (CC BY 4.0), Kathbath (CC BY 4.0, gated), MUCS
  2021 (CC BY-SA 4.0), FLEURS (CC BY 4.0).
- **Images:** Localized Narratives annotations (CC BY 4.0) on Open Images
  photos kept only if CC BY 2.0, DOCCI (CC BY 4.0), TextVQA (CC BY 4.0).
  CC BY needs attribution: keep `data/omni/ATTRIBUTION_open_images.csv`
  and `ATTRIBUTION_textvqa.csv` with anything you ship.

Left out on purpose (`NOTICE` §2, `omni_one.py` docstring): Flickr30k
(non-commercial only); A-OKVQA and its COCO photos (some photos
non-commercial); IndicTTS-Hindi (commercial terms unclear,
`INCLUDE_INDICTTS = False`); the synthetic Indian-banking intent set (written
with an AI assistant whose terms forbid training models on its output
without authorization); the vision projector learned on Flickr30k (the image
projector starts fresh); `jinaai/code_exercises` in the base (CC BY-NC-SA).

Two open questions stay in the base (route C's corpus and the released
base) and so in Omni. codeparrot-clean is GitHub code, including GPL and
AGPL files. opc-sft-stage1's card says MIT, but it includes data derived
from Infinity-Instruct (CC BY-SA 4.0) and from ShareGPT/WildChat
conversations. Neither is settled (`docs/DATASETS.md`). The model you train
is CC BY-SA 4.0 because of its share-alike sources. Read `NOTICE` before you
use it for anything commercial.

---

## Test the result

The test suites need no GPU and no training data. Run them with the GPU
hidden if a job is training:

```bash
python -m tests.test_model
python -m unittest tests.test_training_pipeline tests.test_multimodal_contracts tests.test_release_fixes
python -m unittest discover -s tests -t . -p "test_quality_*.py"
```

On the development machine these gave 80 passed and 1 skipped for the
second line, and 69 passed for the third (`docs/VERIFICATION.md`).

Score a checkpoint again by hand (GPU):

```bash
python -m anulm omni_eval --ckpt ckpt_omni.pt --out logs/my_eval.json --vision
python -m anulm speech_encoder eval --ckpt ckpt_omni.pt --prefix data/fleurs_hi/devtest --limit 1000
```

Then try it yourself in the demo:

```bash
pip install --dry-run gradio faster-whisper piper-tts     # check torch is untouched, then install
python -m anulm.apps.app_omni                             # http://127.0.0.1:7863
```

With `ckpt_omni.pt` in the repository root, every tab uses your model. The
intent tab needs `data/banking77/` and `data/decide_general/massive_hi-IN/`
(B1). The Wikipedia lookup in the Chat tab (the one model is the reader)
also needs the retrieval index in `data/rag/`: `python -m anulm omni_data
fetch read` downloads Simple English and Hindi Wikipedia (`wikimedia/wikipedia`
20231101, CC BY-SA) and runs `rag build` (chunks.jsonl and BM25, ~1.5 GB),
then `python -m anulm rag build-dense` embeds the 932,648 chunks with
multilingual-e5-small into `dense_e5s.npy` (716 MB; ~5 minutes on a GPU,
most of a day on a CPU; `--limit 1000 --out /tmp/x.npy` is a quick trial).
Without it, Chat answers in the model's own words. Add `--device cpu` to run
without a GPU (slowly).

---

## When something goes wrong

- **`torch.cuda.is_available()` is False:** you have the CPU wheel, often
  after a later `pip install` replaced it. Reinstall from the CUDA index.
- **`omni_one.py` sits at "4_wait_gpu":** another program holds more than
  1 GB of the card. Close it. The pipeline checks every minute.
- **Kathbath fails:** accept its terms on the Hub and run `hf auth login`.
- **Reading shows "-" in `REPORT.md`:** the `rc_eval_*.jsonl` files were
  missing. Run `python -m anulm omni_data fetch read`, then
  `python -m anulm omni_eval --ckpt ckpt_omni.pt --out logs/omni_one/eval_reading.json`.
- **Training is 10x slower than the numbers here:** the GPU memory spilled
  into system RAM. Free the card and restart the run (it resumes).
- **Anything else stopped:** rerun the same command. Every long step here
  resumes.

Next, read `docs/OMNI.md` for why the training mix looks the way it does,
and `docs/MODEL_CARD_OMNI.md` for what the model can and cannot do.
