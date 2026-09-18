# Tutorial: from a fresh machine to your own AnuLM

This walks through the whole pipeline on a computer that has nothing
installed: set up, a first model in twenty minutes on a CPU, then each of
the real runs in the order they were done here, with the time, disk and
GPU each one needs and the number you should see at the end. Every step
is one command that already exists in the repository; nothing here is
new code.

`docs/DEVELOPING.md` has the reference detail behind each step (flags,
layout, tests, gotchas). `docs/DATASETS.md` lists every data source with
its licence. `docs/RESULTS.md` is the log of every run, which is where
the expected numbers come from.

## 0. What you need

| tier | hardware | what you can do |
| --- | --- | --- |
| CPU only | any laptop, 8 GB RAM | install, tests, the 17M and 47M presets on tinyshakespeare or a small Hindi corpus (§2–3), run the released checkpoints slowly |
| small GPU | NVIDIA, 8 GB VRAM | the 400M preset with `--grad-ckpt`: Hindi runs, the three-language base, the fine-tunes |
| the runs here | RTX 5070 Ti, 16 GB | everything, including the 700k-step coder (3.3 GPU-days) |

Disk: 5 GB for the Hindi work, 30 GB for the translation and base runs,
60 GB to rebuild the coder. Python 3.10 or newer (3.14 was used here).
Git. On Windows, Git for Windows provides the `bash` the scripts need; on
Linux and macOS the system bash is fine.

## 1. Install

```bash
git clone https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM.git
cd AnuLM
python -m venv .venv
# Linux/macOS: source .venv/bin/activate     Windows: .venv\Scripts\activate
pip install --upgrade pip

# CPU only:
pip install --index-url https://download.pytorch.org/whl/cpu torch
# NVIDIA GPU: pick the wheel for your driver at https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt       # safetensors; pyarrow for the parquet converters
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`--moe-impl grouped`, the fast MoE path used by every GPU run here, needs
torch 2.11 or newer. Older torch still works with `--moe-impl sparse`.

Windows only: the `bash.exe` that Windows puts on PATH is the WSL stub.
Run the scripts with Git Bash (`"C:\Program Files\Git\usr\bin\bash.exe"
experiments/<script>.sh`) or from a Git Bash terminal.

## 2. Verify the code (2 minutes, CPU)

```bash
python quickstart.py        # checks Python, torch, CUDA, the optional packages and the tokenizers
python test_model.py        # 47 tests: routing math, attention, tokenizer, data loaders, loss masks
python model.py             # builds both reference presets, forward + backward, prints parameter tables
```

`quickstart.py` is the one to run if anything below misbehaves: it prints a
line per dependency and the exact command that fixes each problem it finds,
which is also what a bug report here should include. If the tests pass, the
implementation is correct on your machine. Nothing so far touched data.

Every push runs steps 1-3 of this tutorial on Ubuntu and Windows with the
CPU wheel (`.github/workflows/tests.yml`), so the path you are on is the
path CI walks.

## 3. Your first model (20 minutes, CPU)

```bash
python train.py --preset 30b            # downloads tinyshakespeare (1 MB), trains 1,000 steps
python sample.py --ckpt ckpt.pt --prompt "KING RICHARD II:"
```

Expected: val loss around **1.52**, and Shakespeare-shaped text. This is
the 17M-parameter preset in the shape of Sarvam 30B (GQA, 8 layers, MoE).
`--preset 105b` is the MLA-shaped one (47M, ~30 min). `README.md`
"Results" has the exact numbers to compare with.

## 4. A Hindi model from scratch (30 minutes, CPU or GPU)

```bash
python fetch_hindi.py --mb 8 --out data/hindi.txt          # streams the Hindi Wikipedia dump, keeps 8 MB of clean prose
python train.py --preset 30b --data data/hindi.txt --steps 1000
python sample.py --ckpt ckpt.pt --prompt "भारत"
```

This is the byte-level path (no tokenizer to train) that `docs/RESULTS.md`
§1–5 used. To add a BPE tokenizer, as §6 onward does:

```bash
python bpe.py train  --data data/hindi.txt --vocab 16384 --out data/hi16k.json
python bpe.py encode --tokenizer data/hi16k.json --data data/hindi.txt --out data/hindi.hi16k.bin
python train.py --preset 30b --data data/hindi.hi16k.bin --steps 1000
```

## 5. The recipe the released models use (GPU, hours)

The ablations in `docs/RESULTS.md` §9–16 converged on one configuration
(24 experts of 192, top-4, no shared expert, a 256-token window on the
lower layers). Every released checkpoint uses it. The Hindi version, 18k
steps, about 3 h on an 8 GB card:

```bash
bash experiments/build_mixed.sh      # Hindi Wikipedia + Wikisource corpus, hi16k tokenizer, .bin  (§17)
bash experiments/run_long.sh         # ckpt_hindi_mixed18k                                          (§18)
python eval_golden.py ckpt_hindi_mixed18k.pt --device cuda     # expected: ~11% exact / ~86% multiple choice
```

## 6. The three-language base: Hindi + English + Python (GPU, ~5 h)

```bash
bash experiments/build_multi.sh      # fetches enwiki, C4 and codeparrot slices; multi32k tokenizer; .bin  (§22)
bash experiments/run_multi.sh        # ckpt_multi36k: 36,000 steps at batch 8 x 512
```

Expected: val loss **4.44** on the mixed held-out split and sensible
continuations in all three languages. This checkpoint is released as
`AnuLM-Base-400M` and is the starting point for §7 and §8.

## 7. Fine-tune it to answer questions (GPU, ~1 h)

```bash
bash experiments/run_qa_multi.sh     # builds question pairs from the corpora, tunes, scores per language  (§23)
python ask.py --ckpt ckpt_multi_qa.pt
```

Released as `AnuLM-Hindi-QA-400M`. It answers "X क्या है?" in the
encyclopaedia register; it is not a chatbot (see the model card).

## 8. Fine-tune it to translate (GPU, ~6.5 h)

```bash
bash experiments/translate_build.sh          # 2M Samanantar + IIT Bombay pairs both ways, FLORES devtest  (§24)
bash experiments/translate_train.sh 25000    # to step 25k, save, exit  (repeat with a larger step or no argument)
bash experiments/translate_train.sh          # resume to the end, 61,672 steps
LIMIT= bash experiments/translate_eval.sh    # chrF on all 1,012 FLORES sentences per direction, ~2.5 h
```

Expected: **chrF 41.5 en→hi, 43.4 hi→en**. Released as
`AnuLM-Translate-400M`. Non-commercial licence, from the data.

## 9. The Python coder (GPU, 3.3 days + 8 h)

```bash
bash experiments/coder_fetch.sh              # ~15 GB: codeparrot-clean, fineweb-edu, code_exercises, instruction sets, tests
bash experiments/coder_build.sh              # convert, mix 65/30/5, code32k tokenizer, encode: 2.78B tokens, ~2 h CPU
bash experiments/coder_train.sh 100000       # pretrain to step 100k and exit (~11 h); rerun with a larger step to continue
bash experiments/coder_train.sh              # no argument: run to 700,000 (47 h more)
EX_CAP=0 bash experiments/coder_sft.sh       # instruction-tune on 1.38M pairs (6.6 h), then pass@1 on both benchmarks (1.2 h)
```

Expected at the end: val loss **2.72**; **MBPP 12.5%** pass@1 asked as a
question; **HumanEval 4.9%** for the base in continue mode. Intermediate
checkpoints are usable and scorable at any phase boundary:

```bash
python eval_code.py ckpt_coder.pt --bench humaneval --device cuda
```

`docs/MODEL_CARD.md` describes the finished checkpoint; `docs/CODER_PLAN.md`
is the day-by-day log of this exact run, including what went wrong.

### Running for days without a terminal

A multi-day run must not depend on a shell staying open. On Linux use
`tmux` or `nohup bash experiments/coder_train.sh > coder.log 2>&1 &`, or
a systemd user service. On Windows a Task Scheduler task is the only
thing that survives a closed session; `run_coder_phase.cmd` and
`TASKS.md` document the pattern (a 30-minute task with a live-process
guard and a done-marker) and the mistakes that cost idle GPU hours here.
Every training script resumes from `<ckpt>.last`, so a killed run loses
at most one save interval.

## 10. Run the demo

```bash
python serve.py --ckpt ckpt_coder_sft.pt         # or ckpt_translate.pt, ckpt_multi_qa.pt, or an exported folder
# open http://127.0.0.1:8000
```

The page adapts to the checkpoint: the coder offers *write a function*
and *continue the code*, the translator *translate*, the Q&A model
*answer a question*. Stdlib only, nothing to install.

`serve.py` picks the MoE dispatch from the device it is given — the
grouped GEMM on CUDA, the portable `sparse` path on CPU — so the same
checkpoint serves on a laptop and on a GPU box.

For a Gradio interface:

```bash
pip install gradio huggingface_hub
python app.py --ckpt ckpt_coder_sft.pt                     # local Gradio UI on http://127.0.0.1:7860
ANULM_REPO=toonist/AnuLM-Coder-400M python app.py       # downloads the weights from the Hub first
```

**Without installing anything**, `demo_colab.ipynb` does the same three
steps in Colab — clone, download a checkpoint, launch `app.py` with a
public link — and the free tier is enough:
[open it in Colab](https://colab.research.google.com/github/MOBIRIZER-TECHNOLOGY/AnuLM/blob/main/demo_colab.ipynb).

To publish as a Hugging Face Space: create a Gradio Space, add `app.py`,
`serve.py`, `model.py`, `bpe.py`, `make_qa.py`, `requirements.txt` plus
`gradio` and `huggingface_hub`, and set the Space variable `ANULM_REPO`
to the model repo; `space/` holds the front matter and the requirements
file to copy. The free CPU tier runs the 174M-active model at a few tokens
per second — but as of 2026-09 Hugging Face requires a PRO subscription to
*host* a Gradio Space at all (the API returns 402 on free `cpu-basic`),
which is why no Space is up for this project and Colab is the hosted
route. `docs/PUBLISHING.md` §6 has the detail.

## 11. Use the released weights instead of training

```bash
pip install huggingface_hub
hf download toonist/AnuLM-Coder-400M --local-dir AnuLM-Coder-400M   # `hf` is installed by huggingface_hub
python serve.py  --ckpt AnuLM-Coder-400M
python sample.py --ckpt AnuLM-Coder-400M --prompt "def is_prime(n):"
python eval_code.py AnuLM-Coder-400M --bench mbpp --device cuda      # reproduces 12.5%
```

Every script accepts an exported folder (safetensors + config) wherever
it accepts a `.pt`. `export_hf.py` makes such a folder from your own
checkpoint; `docs/PUBLISHING.md` covers uploading it.

## 12. Fine-tune a released model on your own data

`finetune.py` takes JSONL with one object per line, fields `question` and
`answer`, plus an optional `lang` of `hi`, `en` or `py` that picks the
prompt template (Hindi is the default):

```json
{"question": "What does len() return?", "answer": "The number of items in a container.", "lang": "en"}
```

```bash
python finetune.py --ckpt AnuLM-Base-400M --qa my_pairs.jsonl --heldout my_heldout.jsonl \
    --out ckpt_mine.pt --epochs 1 --lr 5e-5 --grad-ckpt
python serve.py --ckpt ckpt_mine.pt
```

Loss is taken on the answer tokens only; `docs/DEVELOPING.md` "Extending"
describes the packing and how to add a new template or language.

## 13. Expected numbers, in one table

| run | script | time on a 16 GB card | number to expect |
| --- | --- | --- | --- |
| tinyshakespeare, `30b` preset | `train.py --preset 30b` | 22 min CPU | val 1.52 |
| Hindi mixed, 18k steps | `run_long.sh` | ~3 h | golden 11.2% / 85.8% |
| three-language base | `run_multi.sh` | ~5 h | val 4.44 |
| question answering | `run_qa_multi.sh` | ~1 h | held-out answer loss 3.32 |
| translation | `translate_train.sh` | 6 h | chrF 41.5 / 43.4 |
| coder pretraining | `coder_train.sh` | 75 h | val 2.72 |
| coder instruction tune | `coder_sft.sh` | 6.6 h | MBPP 12.5%, HumanEval 3.0% (4.9% base) |

Val losses are read on 20 held-out windows and carry about ±0.03 of
noise; the benchmark numbers score whole files or whole test sets and are
stable. If yours land within that of the table, you have reproduced the
run.

**How much does the machine matter?** Less than the noise. The 50-step
`--preset 30b` run that CI does on every push lands at val **2.7868** on
Ubuntu (python 3.12, torch 2.14.0+cpu), **2.7867** on Windows with the same
wheel, and **2.7865** on the RTX 5070 Ti this project was built on (torch
2.11.0+cu128, bf16 autocast, `--moe-impl grouped`). Three operating
systems, two torch versions, CPU against GPU, fp32 against bf16: a spread
of 3e-4, two orders of magnitude under the ±0.03 an eval carries. Parameter
counts are identical everywhere (17,440,384 total, 6,602,368 active). So if
your number is off by more than the noise, suspect the data or the flags,
not the hardware.

## Troubleshooting

- **`torch.cuda.is_available()` is False** with an NVIDIA card: you have
  the CPU wheel. Reinstall from the CUDA index in §1.
- **`grouped_mm` not found**: torch older than 2.11. Add `--moe-impl sparse`.
- **Out of memory** on the 400M preset: keep `--grad-ckpt`, lower
  `--batch-size` to 4 and double `--accum` if the script exposes it, or
  shorten `--block-size`.
- **A script exits immediately on Windows**: it ran under the WSL stub.
  Use Git Bash (§1).
- **Fetch stops early**: the fetchers cut at a byte budget by design; pass
  a larger `--mb`, or more `--shards` to `fetch_code.py`.
- **Resuming after a crash**: run the same script again; it finds
  `<ckpt>.last` and continues. `python update_tasks.py` prints the step
  of the latest save.
