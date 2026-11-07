# AnuLM 1B: plan

Written 2026-11-06. **Decision (the user, 2026-11-06): train from scratch on clean
data**, not grow base v2. Status: data and tools built and tested on the CPU;
nothing has run on the GPU yet. Every time below is an estimate until the probe
(step 3) measures it.

## Goal

A ~1B-parameter base model with clean data provenance, then AnuLM-Omni on top
of it, on the same single RTX 5070 Ti (16 GB).

## Why from scratch

Base v2 learned from two code sources that `NOTICE` lists as open questions,
and every model built on it inherits them:

- **codeparrot-clean**: scraped GitHub, GPL and AGPL files mixed in;
- **opc-sft-stage1**: card says MIT, but it mixes CC BY-SA and ChatGPT-derived data.

Base v2 also read OpenCodeInstruct (CC BY 4.0, model-generated answers). A
model trained from scratch on a corpus without them has clean provenance from
the first token, which matters for commercial use. Growing base v2
(`anulm/training/grow.py`, kept and tested as the fast fallback) would have taken
~5-7 days of GPU instead of ~2-3 weeks, but would have kept those questions.

## Data: `data/v3`, 11.33B tokens (built 2026-11-06)

| source | licence | tokens | how |
|---|---|---|---|
| English: FineWeb-Edu sample/10BT | ODC-BY | 6.00B | copied from `data/v2` |
| Hindi web: FineWeb-2 `hin_Deva` | ODC-BY | 4.50B | copied from `data/v2` |
| Hindi Wikipedia 20231101 | CC BY-SA 4.0 | 0.07B | copied from `data/v2` |
| Python: codeparrot/github-code-clean, MIT / Apache-2.0 / BSD-2 / BSD-3 / ISC / CC0 / Unlicense files only | permissive | 0.77B (528,850 files) | `experiments/build_code_clean.py` |

- **The English and Hindi tokens are copied, not re-encoded.**
  `experiments/build_v2.py` wrote its sources one after another, so they are
  the first 10,566,602,342 training tokens and 106,272,785 validation tokens of
  `data/v2`. Checked 2026-11-06: each source ends on an end-of-document token,
  and the next token starts the next source.
- **The code** keeps only files whose repository licence is on the list above.
  `build_code_clean.py` applies StarCoder's quality filters (size, line
  length, alphanumeric share, no "auto-generated" marker), removes exact
  duplicates, and stores the repository, path and licence of every file, so
  attribution travels with the data.
- **No overlap with the code benchmarks.** `experiments/build_v3.py` drops any
  file that shares a 13-word window with an MBPP or HumanEval prompt, solution
  or test: 1,819 of 530,669 files (0.34%).
- **Licences of the kept files:** MIT 239,067; Apache-2.0 163,267; BSD-3-Clause
  93,796; BSD-2-Clause 20,922; Unlicense 6,248; CC0 2,816; ISC 2,734
  (`data/v3/meta.json`).
- **The tokenizer** stays `multi32k`. It was trained on a 550 MB sample that
  included 100 MB of Python, which was probably from the old code set
  (`docs/RESULTS.md` §22). A BPE merge table records which byte sequences are
  frequent, not the code itself, so the risk is low. Retraining it is about an
  hour of CPU plus re-encoding everything, if a reviewer asks for zero
  inheritance.

## The model

The same shape as the grown model, from a random init: base v2's design
(mixture of experts, grouped-query attention, sliding windows on the first 10
layers), larger.

| | base v2 | 1B |
|---|---|---|
| layers | 20 | 24 |
| hidden size, heads | 1024, 16 (4 key/value) | same |
| experts per layer, used per token | 24, 4 | same |
| expert hidden width | 192 | 512 |
| parameters, total | 398M | 1,008M |
| parameters used per token | 174M | 284M |

## Memory: why a new optimizer

AdamW keeps fp32 weights, fp32 gradients and two fp32 moments: 16 bytes per
parameter, 16 GB for 1B, the whole card before any activations. The repo's
Muon keeps one moment (12 bytes) but measured 2.6x slower per step on this
GPU (`anulm/muon.py`).

`anulm/training/adamw_lowmem.py` is AdamW with the two moments stored in
bf16: 12 bytes per parameter, AdamW's arithmetic in fp32.
`tests/test_adamw_lowmem.py` checks it against `torch.optim.AdamW`:

- with fp32 moments it matches AdamW to within 1e-3 over 60 steps;
- with bf16 moments, the mean of the last 20 losses over 150 steps was 0.719,
  against 0.709 for AdamW and 0.708 for AdamW started from weights nudged by
  1e-4, so the gap is within ordinary run-to-run noise;
- a resumed run keeps bf16 moments and continues identically. torch's loader
  casts them back to fp32 by default; the test caught that.

Budget: 12.1 GB of weights, gradients and optimizer state, plus activations
with gradient checkpointing (about 1-2 GB at batch 4 x 1,024) and the CUDA
context, so about 14 GB of 16. The probe measures the real peak. On Windows,
going over silently pages to system RAM and runs ~20x slower
(`anulm/model.py: fit_memory`).

## Steps

| # | step | time | GPU |
|---|---|---|---|
| 0 | finish Omni runs 1 and 2 (running now) | until ~21:00, 2026-11-06 | busy |
| 1 | data: collect permissive Python, assemble `data/v3` | **done** (1 h 15 min) | none |
| 2 | wire `--optimizer adamw-lowmem` into `train.py` | ~1 h | none |
| 3 | **probe**: 12 h from scratch, measuring speed, memory and the loss curve against base v2's at equal tokens | 0.5 day | yes |
| 4 | pretraining | **~15-22 days** (below) | yes |
| 5 | chat model on the new base (as chat v3, `chat_train_v6.jsonl`) | ~0.5 day | yes |
| 6 | AnuLM-Omni on the new base (`experiments/omni_one.py --data v2`) | ~0.5-1 day | yes |
| 7 | gates, export, demo check, docs, book | ~0.5 day | partly |
| | **total** | **~3-3.5 weeks**, mostly unattended | |

Step 2 waits for run 2: `train.py` is part of the Omni recipe's fingerprint,
and editing it while run 2 trains would make any relaunch of run 2 start over.

**How many tokens.** One pass over `data/v3` is 11.3B tokens, 2.77M steps of
4 x 1,024. At the central estimate of ~6k tokens/s that is ~22 days; 8B tokens
(~15 days) is the shorter option. For comparison, base v2 read 7.8B tokens. The
probe's measured speed sets the number. Like base v2's, the schedule is
warmup-stable-decay, so the decay can start early if the run must end sooner.

### The probe (step 3): go / no-go

```
python -I experiments/build_v3.py --v2 data/v2 --code data/code_clean --out data/v3
python -m anulm.training.train --preset 350m --device cuda --moe-impl grouped --data data/v3/meta.json ^
  --out ckpt_base_1b.pt --block-size 1024 --batch-size 4 --steps 2770000 ^
  --optimizer adamw-lowmem --grad-ckpt --lr 6e-4 --min-lr 6e-5 --warmup 2000 --schedule wsd --decay-frac 0.16 ^
  --eval-every 10000 --eval-windows 400 --log-every 1000 --seed 2026 ^
  --seq-balance-alpha 1e-4 --router-z-alpha 1e-3 ^
  --cfg n_layer=24 sliding_window=256 max_window_layers=10 num_experts=24 num_experts_per_tok=4 ^
        num_shared_experts=0 moe_intermediate_size=512 bias_update_rate=0.003
```

The flags are base v2's (`_v2_run.cmd`) with three changes: the 1B shape, the
low-memory optimizer and gradient checkpointing. Validation reads 400 strided
windows of `data/v3/val.u16`. Its English and Hindi part is base v2's own
validation data, so the curves compare directly; the code part is new.

Go on to step 4 only if, after 12 h:

1. the peak GPU memory is under 15 GB and the speed is at least 5,000 tokens/s;
2. at equal tokens, the training loss is at or below base v2's curve
   (`v2_train_run.log`); a bigger model should learn faster per token;
3. no expert collapse: the router imbalance stays near base v2's (median ~1.4-1.6x).

## Risks

- **Time.** 2-3 weeks of one GPU, during which the GPU can't run anything
  else. Windows Update restarts are covered: training resumes from its last
  save every 10,000 steps.
- **Speed.** 284M per token is 1.6x base v2's compute, plus about 1.3x for
  gradient checkpointing: ~6k tokens/s is the central estimate, 5-8k the range.
- **About the same amount of Python as base v2.** 0.77B tokens of human-written,
  permissively licensed code replaces 0.75B that included model-written
  solutions. Code benchmarks may move either way; MBPP and HumanEval stay
  out of the training data.
- **Inference** costs ~1.6x base v2 per token. A CPU demo stays usable, slower.

## What carries over

The per-task adapters (to stop chat drifting) and a higher-resolution vision
encoder (for text in photos) are separate fixes and apply in step 6. The Omni
pipeline, gates, demo, site and book need only the new base path. The chat
data (`chat_train_v6.jsonl`) and the run-2 data (`data/next`) are already clean
and ready.
