# Model card: the AnuLM Python coder (`ckpt_coder_sft.pt`)

One page on what the finished coder checkpoint is: the data it saw and where
that data came from, the architecture, how it was trained, what it can and
cannot do, and how to try it. Written 2026-09-18 from the files on disk;
`docs/CODER_PLAN.md` is the plan and the day-by-day log, `docs/RESULTS.md`
§25 the full curves and samples, `TASKS.md` the operations.

## In one paragraph

A 398M-parameter decoder-only mixture-of-experts model in the shape of
Sarvam 30B (GQA attention, 24 routed experts, aux-loss-free balancing),
174M parameters active per token, pretrained from scratch on one pass over
2.78B tokens that are two-thirds Python, then instruction-tuned for one
epoch on 1.38M exercise-style question/answer pairs. On the two standard
Python benchmarks it scores **MBPP 12.5% pass@1** (32/257, asked as a
question) and **HumanEval 4.9%** (8/164, base model completing a
signature; 3.0% after tuning). It writes short, common functions that often
run; it is not a general assistant. Trained on one RTX 5070 Ti in about 82
GPU-hours across 2026-09-10 to 09-15.

## Data

### Pretraining corpus: 13.7 GB of text, 2.78B tokens, one pass

All sources are public, ungated Hugging Face datasets, downloaded by
`experiments/coder_fetch.sh` and converted, mixed and encoded by
`experiments/coder_build.sh` (2 h of CPU; `coder_build.log` is the record).

| part | source | text | documents | share of tokens (approx.) |
| --- | --- | --- | --- | --- |
| Python source | [codeparrot/codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean), shards 1–10, de-duplicated GitHub Python, filtered to files that look hand-written (`fetch_code.py`) | 7,297 MB | 948,416 | ~51% |
| Python exercises | [jinaai/code_exercises](https://huggingface.co/datasets/jinaai/code_exercises), phi-1-style problems with complete solutions; fill-in-the-blank items dropped; 25,789 held out as `bench_exercises.txt` | 992 MB | 1,289,468 | ~7% |
| English | [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), two shards of `sample/10BT` | 4,090 MB | 833,253 | ~35% |
| Hindi | Hindi Wikipedia + Wikisource, the §17 corpus already on disk (`data/mixed.txt`) | 1,324 MB | 224,490 | ~6% |
| **total** | interleaved by `mix_corpus.py` into `data/coder.txt` | **13,702 MB** | 3.3M | 2,778,171,683 tokens |

Token shares are the byte shares divided by the measured bytes-per-token of
each language below; the plan's target was 65% Python / 30% English / 5%
Hindi and the build landed close to it. The encoded corpus
(`data/coder.code32k.bin`, uint16, 5.56 GB) is 2.78B tokens, and 700,000
steps at batch 8 × 512 is 2.87B, so the run is one pass plus a 3% wrap:
epoch 1.00 at the last step, every token seen once.

**Tokenizer.** `data/code32k.json`: a byte-level BPE with 32,768 entries
(32,511 merges), trained by `bpe.py` in 998 s on a 500 MB sample of the mix
(250 MB Python, 50 MB exercises, 100 MB English, 100 MB Hindi). Bytes per
token on the held-out benchmark files: Python 4.83, English 4.20, Hindi
8.44. Against the three-language `multi32k` tokenizer used for the
translation model, Python got 10–17% cheaper per byte and Hindi 12% dearer,
which is the point of training a tokenizer per mix.

### Instruction-tuning pairs: 1,383,159, one epoch

| source | pairs used | what |
| --- | --- | --- |
| jinaai/code_exercises | 1,289,468 (all) | problem statement → complete Python solution |
| [nvidia/OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct), 1 shard | 43,728 | Python-only pairs whose `average_test_score` ≥ 0.9, answer reduced to its Python block |
| [glaiveai/glaive-code-assistant](https://huggingface.co/datasets/glaiveai/glaive-code-assistant) | 40,000 of 86,259 | Python-only pairs, answer reduced to its Python block |
| docstring pairs from the pretraining Python (`make_qa_py.py`) | 10,763 | "what does this function do" from real code |
| **total** | **1,383,959**, of which 800 held out | packed into 475,807 rows of 512; 87.4M answer tokens |

Each pair is rendered as `Question: {q}` newline `Answer:` and the loss is
taken on the answer tokens only (34% of positions). Held-out answer loss
went 1.6442 → 0.9179 over the epoch and was still falling at the end.

### Evaluation sets (never trained on)

[openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)
(164 problems) and the sanitized split of
[google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp)
(257 problems). `eval_code.py` generates greedily and *executes* the
model's code against each problem's tests; a problem passes only if every
test does.

## Architecture

`train.py --preset 350m` with the "combo" overrides that the Hindi
ablations converged on (`docs/RESULTS.md` §12, §16), a 32k vocabulary, and
the code32k tokenizer. Read directly from the checkpoint's `cfg`:

| | |
| --- | --- |
| type | decoder-only transformer, pre-norm (RMSNorm, eps 1e-6), no biases |
| layers | 20 |
| hidden size | 1,024 |
| attention | grouped-query: 16 query heads, 4 key/value heads, head dim 64, QK-norm, RoPE θ = 1,000,000 |
| attention window | sliding window of 256 tokens on layers 0–9, full attention on layers 10–19 |
| context | 512 tokens (block size; no YaRN) |
| MLP, layer 0 | dense SwiGLU, intermediate 2,816 |
| MLP, layers 1–19 | mixture of experts: 24 routed experts of intermediate 192, top-4 per token, no shared expert, sigmoid router scores scaled by 2.5, `n_group = 1` |
| load balancing | aux-loss-free bias correction (`bias_update_rate` 3e-3) plus a light sequence-balance loss (1e-4) and router z-loss (1e-3) |
| vocabulary | 32,768, embeddings not tied to the output head |
| parameters | **397.7M total, 173.5M active per token (44%)** |

Where the parameters are:

| component | parameters |
| --- | --- |
| routed experts (19 layers × 24 × 3 matrices) | 269.0M |
| attention (20 layers) | 52.5M |
| token embedding | 33.6M |
| output head | 33.6M |
| dense MLP (layer 0) | 9.1M |

What this shares with Sarvam 30B and what it deliberately changes is in
`README.md` ("What is faithful" / "What is deliberately different") and
`docs/ARCHITECTURE.md` §10.

## Training

| | pretraining | instruction tuning |
| --- | --- | --- |
| script | `experiments/coder_train.sh` → `train.py` | `experiments/coder_sft.sh` → `finetune.py` |
| data | `data/coder.code32k.bin`, 2.78B tokens | `data/sft_code.jsonl`, 1.38M pairs |
| steps | 700,000 at batch 8 × 512 (2.87B tokens, epoch 1.00) | 59,476 at batch 8 (one epoch) |
| learning rate | 6e-4, cosine to 6e-5, 2,000 warmup steps | 5e-5 |
| precision | bf16 autocast, gradient checkpointing, grouped-GEMM MoE | same |
| hardware | one RTX 5070 Ti 16 GB, ~0.39 s/step | same |
| wall clock | ~75 h of GPU (2026-09-10 15:29 → 09-14 18:38, with restarts) | 6 h 38 min (2026-09-14 20:41 → 09-15 03:20) |
| result | val loss 4.586 at step 5k → **2.7222** best (step 685k), 2.7234 final | held-out answer loss 1.6442 → **0.9179** |

Validation loss over the run (`coder_curve.csv` has all 138 points;
bits/byte in the last column is against the code32k byte count):

| step | 50k | 100k | 150k | 200k | 250k | 300k | 350k | 400k | 450k | 500k | 550k | 600k | 650k | 700k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val loss | 3.555 | 3.440 | 3.368 | 3.335 | 3.276 | 3.217 | 3.161 | 3.104 | 3.050 | 2.961 | 2.869 | 2.796 | 2.734 | 2.723 |
| bits/byte | 1.041 | 1.007 | 0.986 | 0.976 | 0.959 | 0.942 | 0.925 | 0.909 | 0.893 | 0.867 | 0.840 | 0.819 | 0.800 | 0.797 |

The val loss is measured on 20 windows of the held-out split at each eval,
so single readings carry about ±0.03 of noise; the trend is real, the
last-digit comparisons are not (`docs/RESULTS.md` §25, "A review of the
training loop"). The run was owned by a Windows scheduled task that
restarted it after every interruption; the operational lessons are in
`TASKS.md`.

## Capability

pass@1, greedy decoding, code executed against the benchmark tests:

| | base `ckpt_coder.pt` | tuned `ckpt_coder_sft.pt` |
| --- | --- | --- |
| MBPP (257), asked as a question | 0.0% | **12.5%** (32/257) |
| HumanEval (164), continuing a signature + docstring | **4.9%** (8/164) | 3.0% (5/164) |
| HumanEval, asked as a question | — | 0.0% |

For scale: CodeParrot-110M reaches ~4% on HumanEval after 25B tokens;
SantaCoder-1.1B ~18%; phi-1-small (350M, 7B tokens of textbook-quality
Python) ~45%. This model reaches 4.9% after 2.87B tokens, about a third of
compute-optimal for its size, with phi-1-style data making up 7% of its
corpus rather than most of it.

**How the two prompt modes differ, and why it matters.** MBPP items are
prose tasks, so the tuned model is asked `Question: … Answer:` and writes a
fresh function. HumanEval items *are* a function to continue, and asked as
a question the tuned model writes a different function and scores 0; asked
to continue, the same weights score 3.0%. Always report which mode produced
a number. The tune bought 0 → 12.5% on MBPP and cost about two HumanEval
points of raw continuation ability, a real specialisation trade.

**What it does well.** Short, common functions from a one-sentence
description: primality, Fibonacci, reversing, counting, filtering a list,
simple string manipulation. The output is syntactically valid Python
nearly always and idiomatic more often than not.

**What it does badly.** Anything needing more than one idea, careful edge
cases, or an API it has not seen often. It calls functions and methods that
do not exist, gets off-by-one and boundary conditions wrong, and can
continue past the end of a function into unrelated code. It knows no
library beyond what appears often in GitHub Python. The context is 512
tokens, so it cannot read a long file. With 6% Hindi and 35% English in
pretraining it will also continue prose in both, but it was not tuned to
answer questions or translate; the earlier checkpoints
(`ckpt_multi_qa.pt`, `ckpt_translate.pt`) do those.

**Safety.** It executes nothing itself; `eval_code.py` and anyone using its
output do. Treat generated code as untrusted: read it, run it in a sandbox,
test it. It has had no alignment or refusal training of any kind.

## Try it

```bash
python serve.py --ckpt ckpt_coder_sft.pt        # then open http://127.0.0.1:8000
```

The page offers *write a function* (the MBPP setting: describe a function
in a sentence) and *continue the code* (the HumanEval setting: paste a
signature and docstring). *Greedy (as evaluated)* reproduces the benchmark
decoding; untick it to sample at the temperature shown. On this machine the
server is kept alive by the scheduled task `nanosarvam_serve`
(`run_serve.cmd`); `TASKS.md` says how to stop it.

```bash
python eval_code.py ckpt_coder_sft.pt --bench mbpp --device cuda            # ~10 min
python eval_code.py ckpt_coder.pt ckpt_coder_sft.pt --bench humaneval --mode continue --device cuda
python sample.py --ckpt ckpt_coder.pt --prompt "def is_prime(n):"          # the base model, raw
```

## Files

| file | size | what |
| --- | --- | --- |
| `ckpt_coder_sft.pt` | 1.59 GB | the tuned model (fp32 weights + config + prompt templates) |
| `ckpt_coder.pt` | 1.59 GB | the pretrained base at step 700,000 |
| `ckpt_coder_sft.pt.last`, `ckpt_coder.pt.last` | 4.8 GB each | the same plus optimizer state, for resuming |
| `ckpt_coder_700k_backup.pt` | 1.59 GB | a copy of the base taken before tuning |
| `data/code32k.json` | 0.4 MB | the tokenizer |
| `data/coder.code32k.bin` | 5.56 GB | the encoded pretraining corpus |
| `data/sft_code.jsonl` | 1.21 GB | the instruction pairs as trained |
| `coder_curve.csv`, `coder_train_phase1.log`, `coder_sft.log`, `eval_*_sft.log` | | the training and evaluation records |

## Licence and provenance

The code is MIT (`LICENSE`). The training data keeps its own licences:
codeparrot-clean and code_exercises redistribute permissively licensed
GitHub code and synthetic exercises, fineweb-edu is ODC-By, OpenCodeInstruct
is CC BY 4.0, glaive-code-assistant is Apache 2.0, and the Hindi corpus is
CC BY-SA from Wikipedia and Wikisource. A model trained on GitHub code can
reproduce licensed snippets verbatim; check anything you ship.
