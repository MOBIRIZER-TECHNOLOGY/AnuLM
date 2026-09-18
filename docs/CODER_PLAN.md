# One week to a Python coding demo

Goal: a checkpoint of this architecture that writes small, working Python
functions live, with a pass rate on the standard tests (HumanEval, MBPP) that
can be quoted to an investor. Hindi and English stay in the mix so the
multilingual story survives; the compute goes to Python.

**Status 2026-09-11:** fetch and build are done (`coder_fetch.sh`,
`coder_build.sh`; numbers at the end of this file). **Pretraining is at step
100,000 of 700,000**, the first of the five training phases, finished
2026-09-11 04:52. Best val 3.4361 at step 85,000. The curve and what to
watch are at the end of this file; `coder_train.sh 250000` is next.

The plan is phase by phase. Every phase is a single script that can be rerun
after an interruption and picks up where it stopped, and the training phases
stop at a chosen step and save, so the machine can sleep between them. What
cannot happen is sleeping *during* a training phase without losing up to 35
minutes of work; that is the only constraint.

## Sizing

| | |
| --- | --- |
| GPU | RTX 5070 Ti 16 GB, ~2.5 steps/s at batch 8 x 512 = ~10k tokens/s |
| Model | the combo preset (24 x 192 experts, top-4, no shared, window 256), ~398M params with a 32k vocab, ~150M active |
| Budget | 700,000 steps = 2.87B tokens, one pass over the 2.78B-token corpus, ~3.3 GPU-days |
| Mix | ~65% Python, ~30% English, ~5% Hindi, single pass (no repetition) |
| Data on disk | ~15 GB text, 6 GB encoded, 1 TB free |
| Expected result | 10-25% pass@1 on HumanEval; phi-1-small reached 45% at this size with far cleaner data |

Reference points for the "how much data" question: CodeParrot-110M saw 25B
tokens for ~4%; phi-1 (1.3B) and phi-1-small (350M) saw 7B tokens of
textbook-quality Python for 50% / 45%. Data quality moves the number more
than volume at this scale, which is why the mix includes the code_exercises
set (a public phi-1-style dataset) and why the instruction stage uses
exercise-style pairs rather than only docstrings.

A multi-language coder (StarCoder2-3B: 3T tokens, DeepSeek-Coder-1.3B: 2T)
is not a one-GPU project; at 1B tokens/day it is years. That needs cloud
compute or a continued-pretraining start from an open checkpoint.

## Sources (all public, ungated, fetched by `experiments/coder_fetch.sh`)

| Source | What | Size used |
| --- | --- | --- |
| codeparrot/codeparrot-clean, shards 1-10 | de-duplicated GitHub Python | ~9 GB, ~2.0B tokens |
| jinaai/code_exercises | 1.29M phi-1-style exercises with complete solutions (fill-in-the-blank ones dropped) | 1.0 GB text + SFT pairs |
| HuggingFaceFW/fineweb-edu, 2 shards | educational English web text | 3.9 GB, ~0.9B tokens |
| data/mixed.txt (already built) | Hindi Wikipedia + Wikisource | 1.3 GB, ~0.14B tokens |
| nvidia/OpenCodeInstruct, 1 shard; glaiveai/glaive-code-assistant | instruction pairs, reduced to their Python block | SFT only |
| openai/openai_humaneval; google-research-datasets/mbpp (sanitized) | the tests | eval only |

Not used: `python-edu` from the SmolLM corpus, the closest open match to
phi-1's data, ships only file ids and needs a Software Heritage S3 download
per file (boto3). Worth adding if the first result disappoints.

## The week

| Day | Phase | Script | Time | Sleep OK? |
| --- | --- | --- | --- | --- |
| 1 | Fetch | `coder_fetch.sh` | 1-2 h, network | yes, rerun to finish |
| 1 | Build: convert, mix, 32k tokenizer, encode | `coder_build.sh` | ~2 h, CPU | yes, rerun to finish |
| 2 | Train to 100k | `coder_train.sh 100000` | 11 h | between calls |
| 3 | Train to 250k | `coder_train.sh 250000` | 17 h | between calls |
| 4 | Train to 400k | `coder_train.sh 400000` | 17 h | between calls |
| 5 | Train to 550k | `coder_train.sh 550000` | 17 h | between calls |
| 6 | Train to 700k (end) | `coder_train.sh` | 17 h | between calls |
| 7 | Instruction tune + pass@1 + demo | `coder_sft.sh` | 1 h tune, 1.5 h eval | yes |

The step targets are suggestions. Any call of `coder_train.sh N` trains from
the last save to step N and exits; call it with whatever fits the day. If
the machine sleeps mid-phase and the process dies, call it again: it
resumes from `ckpt_coder.pt.last`, written every 5,000 steps (~35 min).

Checkpoints along the way are usable: `eval_code.py ckpt_coder.pt` at any
point gives the current pass@1 (about 40 minutes for HumanEval), and
`serve.py --ckpt ckpt_coder.pt` serves it. A mid-week number is worth
having before the demo, in case the end of the week slips.

## Day 7: the demo

1. `coder_sft.sh` produces `ckpt_coder_sft.pt` and prints pass@1 for the base
   and the tuned model on both tests.
2. `serve.py --ckpt ckpt_coder_sft.pt` gives the web UI a question mode; ask
   "Write a Python function `is_prime`: return True if n is prime." and the
   answer is a complete function.
3. Show the pass rate, then live prompts. Prompts the model is most likely to
   get right: primes, fibonacci, string reversal, palindromes, list sum / max,
   FizzBuzz, factorial, counting vowels. Keep temperature low (0.2) and the
   repetition penalty on.
4. Say what it is: a ~400M-parameter model of this architecture, trained on
   one consumer GPU in a week, on 3B tokens. The number to compare against
   is CodeParrot's 4% at 25B tokens.

## What would move the number afterwards

- python-edu (quality) and more codeparrot shards (volume): the corpus is the
  ceiling.
- A second week of training: the loss curve will not have flattened.
- Batch 16 if memory allows (the 32k-vocab logits are what grew the
  footprint); better GPU utilisation, same tokens per step count.
- An execution-filtered SFT set: keep only instruction answers whose code
  passes its own tests, the OpenCodeInstruct columns carry that signal.

## The final result (2026-09-15)

`ckpt_coder_sft.pt`: the step-700,000 base instruction-tuned on all
1,383,159 pairs (475,807 packed rows, 87.4M answer tokens), one epoch,
59,476 steps, 6 h 38 min. Held-out answer loss **1.6442 -> 0.9179**, a new
best at the final eval, so even the uncapped epoch had not saturated.

| | base, `continue` | tuned, `question` | tuned, `continue` |
| --- | --- | --- | --- |
| HumanEval | **4.9%** (8/164) | 0.0% (0/164) | 3.0% (5/164) |
| MBPP | 0.0% (0/257) | **12.5%** (32/257) | — |
| *CodeParrot-110M, 25B tokens* | *~4%* | | |
| *SantaCoder-1.1B / phi-1-small-350M* | *~18% / ~45%* | | |

**The headline is MBPP 12.5%**, up from 5.1% at the step-250,000 probe:
2.8x the pretraining and 4.7x the instruction data roughly doubled it.
**HumanEval reached 4.9%**, its first non-zero, on the *base* model.

Both figures land where §the plan predicted at step 250,000 ("MBPP in the
low teens, HumanEval in low single digits"), and below the original plan's
10-25% on HumanEval, for the reason given there: 2.87B tokens is about a
third of compute-optimal for 398M parameters, and the phi-1-style exercise
data that justified the higher target is 7% of this corpus rather than most
of it. For scale, CodeParrot-110M needs 25B tokens to reach ~4% on
HumanEval; this reaches 4.9% on 2.87B.

### The 0.0% was a prompting artifact, and finding it changed the number

`eval_code.py` chose `question` mode for any checkpoint carrying
`qa_templates` and `continue` for any that did not, with no override. That
is right for MBPP, whose items are prose task descriptions, and wrong for
HumanEval, whose native form *is* a continuation: a signature and docstring
to complete. Asked as a question, the tuned model writes a fresh function
instead of continuing the given one, and scores 0.0%. Asked to continue,
the same weights score 3.0%.

`--mode {continue,question}` now forces it. Two things follow.

- **Report which mode produced a number.** A 0.0% that is really 3.0% is
  the kind of error that survives into a paper.
- **Instruction tuning genuinely cost a little raw continuation ability**:
  4.9% base against 3.0% tuned, both in `continue` mode, a fair comparison.
  The tune bought 0 -> 12.5% on MBPP and paid ~2 points of HumanEval for
  it. That is a real specialisation trade, not an artifact, and at this
  scale it is clearly worth it.

## Pretraining is finished (2026-09-14 18:38)

700,000 steps, 2.87B tokens, one pass, epoch 1.00. **Best val 2.7222** at
step 685,000; 2.7234 at the final step. 47.8 h of GPU for the 250k→700k
stretch, unattended, owned by the `nanosarvam_coder` scheduled task.

| step | 250k | 350k | 450k | 550k | 650k | **700k** |
| --- | --- | --- | --- | --- | --- | --- |
| val | 3.276 | 3.161 | 3.049 | 2.869 | 2.734 | **2.723** |

The full 138-point curve is in `coder_curve.csv` (steps 255,000 and
260,000 are permanently lost; see TASKS.md). What remains is the final
instruction tune and its pass@1 -- run it **uncapped** (`EX_CAP=0`), since
the capped probe's answer loss was still falling when its epoch ended.

## The probe: instruction tuning at step 250,000 (2026-09-12)

The decision this was run to make is in "the plan from here" below. **It
came back in the best row: the pipeline produces a working coder.**

`ckpt_coder.pt` (step 250,000) tuned for one epoch on 293,691 pairs —
200,000 code_exercises, 43,728 OpenCodeInstruct, 40,000 glaive, 10,763
docstring pairs — 14,727 steps, 1 h, lr 5e-5, loss on answer tokens only.

| | base, step 250k | **instruction-tuned** |
| --- | --- | --- |
| held-out answer loss | 2.2808 | **1.1476** |
| HumanEval pass@1 | 0.0% (0/164) | 0.0% (0/164) |
| MBPP pass@1 (sanitized test) | 0.0% (0/257) | **5.1% (13/257)** |

**Thirteen MBPP problems execute correctly against their hidden tests.**
That is the first non-zero in this log, and it arrives at 36% of the
pretraining schedule, on 1.02B of 2.87B tokens. CodeParrot-110M reaches
~4% on HumanEval after 25B tokens; this is 5.1% on MBPP after 1.02B.

Three readings.

- **Instruction tuning is what converts the model's idiom into programs.**
  The base checkpoint scores zero on the same problems with the same
  decoder. Everything §25 observed — docstrings, `for`/`range`, correct
  `isinstance` — was latent capability that the base model could not aim
  at a task. 293k pairs aimed it.
- **The two benchmarks disagree, and MBPP is the right one to watch here.**
  MBPP's problems are short, self-contained and close in shape to the
  exercise pairs; HumanEval's are longer and need more compositional
  reasoning. A model that solves 5% of MBPP and 0% of HumanEval is exactly
  what a 400M model at a third of Chinchilla-optimal should look like.
  Expect HumanEval to stay at or near zero until much later, if it moves
  at all.
- **The answer loss was still falling when the epoch ended** (1.1476 at
  the last of 30 evals, a new best every time). The 200k cap on exercise
  pairs is leaving value unspent: the final tune at step 700,000 should
  run on more of the 1.29M, and `EX_CAP=0` is the knob.

### What it means for the remaining budget

Proceed. The 450,000 steps left are now buying a known-good pipeline more
compute rather than testing whether there is one. Revised expectation for
the tuned step-700,000 checkpoint: **MBPP in the low teens, HumanEval in
low single digits**, on the reasoning that MBPP went 0 → 5.1% for the
first 36% of pretraining plus a capped tune, and the remaining 64% plus an
uncapped tune is a larger increment than that but not a multiple of it.
That is below the plan's original 10-25% on HumanEval, and the honest
number to quote is MBPP's.

## The plan from here (written 2026-09-12, at step 250,000)

### The recommendation: run the instruction tune now, at 250k, before spending 50 more GPU hours

The remaining pretraining is 450,000 steps, 49.8 h, just over two days of
GPU. The question that budget is buying an answer to is "does this
architecture on this corpus produce a model that writes working Python",
and **nothing measured so far answers it**, because pass@1 on a base model
is the wrong instrument (see the two zeros above) and the plan's headline
number was always the *instruction-tuned* one.

`coder_sft.sh` on the current checkpoint costs **3 to 3.5 h**: 1.5 h to
tune with `sft_exercises.jsonl` capped to ~200k pairs, then 1.2-1.9 h of
pass@1 (two checkpoints on two benchmarks; 1.2 h if the two `eval_code.py`
calls are run concurrently, 1.9 h sequentially as the script is written).
Three hours to convert a two-day bet into an informed one:

| what the 250k tune scores | what it means | what to do |
| --- | --- | --- |
| 2-5% pass@1 | the pipeline works; the rest is compute | run phases 3-5 as planned, re-tune at 700k |
| 0-1%, but the answers are whole functions with correct signatures | format learned, semantics not; on the phi-1 curve but earlier | run phases 3-5, and consider the mix change below |
| 0%, answers still loop or ignore the question | something is wrong that more steps will not fix | stop and diagnose before spending the 50 h |

The evidence favours the middle row. The samples at 250k show Python
*idiom* learned — `isinstance`, `raise ValueError`, comprehensions with a
filter, docstring-then-body — and semantics absent. That is precisely what
instruction data is for, and the exercise set is the part of the corpus
that carries it.

### The number this run is likely to reach, stated honestly

The plan's 10-25% pass@1 deserves a second look against what the corpus
actually is:

| | params | tokens seen | pass@1 |
| --- | --- | --- | --- |
| CodeParrot-110M | 110M | 25B | ~4% |
| phi-1-small | 350M | 7B, textbook-quality | ~45% |
| **this run at step 700,000** | 398M | **2.87B**, 65% GitHub Python | ? |

2.87B tokens for a 398M model is roughly a third of Chinchilla-optimal
(~8B), and a ninth of what CodeParrot-110M consumed to reach 4%. The
plan's 10-25% rests entirely on the *quality* argument — that
`code_exercises` is phi-1-style data — and that set is 1.0 GB of the 13.7
GB mix, about 7% by bytes. phi-1's result came from a corpus that was
mostly such data. **A realistic expectation for the tuned 700k checkpoint
is low single digits, with 10% a good outcome and 25% unlikely.** Better
to write that down now than to discover it on day 7.

### If the mix is the ceiling, the cheapest fix

Do not re-tokenize or re-mix the 13.7 GB. Two cheaper levers, in order:

1. **Weight the exercises up in the instruction tune**, which
   `coder_sft.sh` already draws from — it is the one place the good data
   is not diluted. Its 1.29M pairs are currently capped only by wall
   clock, and that cap is the knob.
2. **A second pass over the exercises alone** as a short pretraining
   finish (say 20k steps on `data/exercises.txt` re-encoded with
   `code32k`) before the instruction tune, i.e. anneal on the clean
   subset. This is the "textbook finish" that phi-1 argues for and costs
   ~2 h, not a rebuild.

Both are decidable after the 250k tune above.

### Sequence, with the GPU free

The GPU cannot hold training and a fine-tune at once (~10 GB each against
16.3 GB), so these are serial:

| # | step | command | time |
| --- | --- | --- | --- |
| 1 | cap the exercise pairs, then instruction-tune the 250k checkpoint | edit `coder_sft.sh`'s slice, `bash experiments/coder_sft.sh` | 1 h 30 |
| 2 | pass@1, base and tuned, both benchmarks | inside `coder_sft.sh` | 1 h 15 - 1 h 55 |
| 3 | decide from the table above | | |
| 4 | phase 3 | `schtasks /change /tn nanosarvam_coder /tr "...\run_coder_phase.cmd 400000"` | 16.5 h |
| 5 | phase 4 | same, `550000` | 16 h |
| 6 | phase 5 | same, `run_coder_phase.cmd` with no argument | 16 h |
| 7 | final instruction tune + pass@1 from step 700,000 | `coder_sft.sh` | 3 h capped / 8 h on all 1.38M pairs |
| 8 | demo | `python serve.py --ckpt ckpt_coder_sft.pt` | minutes |

### The phase boundaries are now vestigial

Steps 4-6 exist because the original plan had one constraint: "what cannot
happen is sleeping *during* a training phase without losing up to 35
minutes of work". A scheduled task owns the run now, so that constraint is
gone, and the three gates are pure manual overhead. Three facts settle it:

- **The unit of safety is already small.** `--eval-every 5000` writes both
  checkpoints every 5,000 steps, about 33 minutes. That is what a crash
  costs, with or without phase boundaries.
- **The unit of recovery is already automatic.** The task fires every 30
  minutes and restarts a dead run from `ckpt_coder.pt.last`.
- **Phases do not change the training.** `--steps 700000` is in `COMMON`
  regardless, so the cosine schedule spans the full run either way;
  `--stop-at` only decides when the process exits. Collapsing the phases
  produces bit-identical training to running them separately.

So the choice is only about how often you want to be asked for a command:

| granularity | gates left | each | when it makes sense |
| --- | --- | --- | --- |
| **one target of 700000** | 0 | 49.8 h unattended | the default now; nothing to do but read `TASKS.md` |
| every 50,000 | 9 | 5.5 h | you want to eyeball pass@1 on a ladder |
| as written, 3 phases | 3 | ~16 h | no reason left |

To collapse them: `schtasks /change /tn nanosarvam_coder /tr
"C:\workspace\AnuLM\run_coder_phase.cmd 700000"`. Evaluation does not
need a gate either — training and both benchmarks coexist at 12.7 GB of
16.3 GB, so `eval_code.py` can be run against `ckpt_coder.pt` at any
moment without stopping anything.

One bug this exposed, fixed 2026-09-12: the completion alert matched only
`phase done at step N`, which `train.py` prints when `--stop-at` lands
*below* `--steps`. On the final phase it prints `done in ... | ckpt
ckpt_coder.pt` instead, so a run taken straight to 700,000 would have
finished silently. `run_coder_phase.cmd` now matches either line.

**Total remaining GPU: about 57 h, two and a half days**, as 3.5 h for the
probe, 49.8 h of pretraining, and 3.5 h for the final tune and its
benchmarks. Run back to back that is roughly 2026-09-14 mid-afternoon.
Skipping the probe saves 3.5 h and buys nothing back; the pretraining is
the same either way.

Measured inputs behind those numbers, so they can be recomputed when
anything changes: 0.398 s/step at batch 8 x 512; the instruction set is
1.38M pairs / 237M tokens uncapped (57,744 steps, 6.4 h) or 56M tokens
with exercises capped at 200k (13,612 steps, 1.5 h); HumanEval takes ~24
min per base checkpoint and ~20 min per tuned one, MBPP ~42 and ~30.

### What is already finished and needs nothing further

The Hindi ladder (§1-21), the three-language model (§22), multilingual
question answering (§23) and English-Hindi translation (§24, chrF 41.5 /
43.4) are all complete and documented. `serve.py` demonstrates the last
two today. Only the coder is open.

## What the build produced (2026-09-10, `coder_build.log`)

| file | size | docs | note |
| --- | --- | --- | --- |
| `data/python_big_raw.txt` | 7.3 GB | 948,416 | codeparrot-clean shards 1-10; ran out 253 MB short of the 7.2 GB budget |
| `data/exercises.txt` | 1.0 GB | 1,289,468 | code_exercises, fill-in-the-blank items dropped; 25,789 docs to `bench_exercises.txt` |
| `data/english_edu.txt` | 4.2 GB | 854,719 | fineweb-edu, two shards |
| `data/mixed.txt` (Hindi) | 1.3 GB | 224,490 | the §17 corpus, all of it |
| `data/coder.txt` | 13.7 GB | | the interleaved mix |
| `data/code32k.json` | | | trained on a 500 MB sample in 998 s |
| `data/coder.code32k.bin` | 5.6 GB | | ~2.78B tokens (uint16), i.e. one pass = 700k steps at 8 x 512 |
| `data/sft_exercises.jsonl` / `sft_opencode.jsonl` / `sft_glaive.jsonl` | 1.1 GB / 74 MB / 61 MB | | for `coder_sft.sh` |

The code-weighted tokenizer against the three-language one from §22:

| benchmark | `code32k` bytes/token | `multi32k` bytes/token |
| --- | --- | --- |
| `bench_python_clean` / `bench_python` | 5.18 / 4.83 | 4.42 |
| `bench_english` | 4.20 | 4.33 |
| `bench_hindi` | 8.44 | 9.54 |

Python got 10-17% cheaper per byte, Hindi 12% dearer -- the vocabulary
went where the sample did, which is the point of training it per mix.

## Phase 1: steps 0 to 100,000 (2026-09-10 15:29 to 2026-09-11 04:52)

The combo preset with the 32k code tokenizer, batch 8 x 512, lr 6e-4 ->
6e-5 cosine over the full 700k, warmup 2,000, grouped dispatch, seq-balance
and z-loss on. 0.396 s/step; 11 h of GPU for the phase, of which 35,607 s
was the final uninterrupted stretch. Two interruptions, neither costing a
step: the run was killed with its launching session at ~16:36 (resumed
18:58 from the step-10,000 save, ~7 min of work repeated) and that is why
the work now belongs to a scheduled task instead.

Held-out val loss, every 5,000 steps:

| step | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k | 45k | 50k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | 4.586 | 4.048 | 3.872 | 3.810 | 3.690 | 3.664 | 3.613 | 3.597 | 3.582 | 3.555 |

| step | 55k | 60k | 65k | 70k | 75k | 80k | **85k** | 90k | 95k | 100k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | 3.520 | 3.531 | 3.490 | 3.485 | 3.475 | 3.459 | **3.436** | 3.439 | 3.451 | 3.440 |

**The one thing to watch: the last 20,000 steps are flat.** From step
80,000 the curve sits between 3.436 and 3.451 with no trend, and the best
checkpoint is step 85,000 rather than the newest. That is early for a
plateau — the cosine has barely decayed at 100k of 700k (lr still ~5.9e-4),
and the model has seen 410M of the corpus's 2.78B tokens, so nothing has
been repeated yet. Two readings, and phase 2 distinguishes them:

- **Noise around a still-falling curve.** The eval draws the same windows
  every time, so this is not sampling noise, but a 0.015 band over four
  evals is small. If val is materially below 3.43 by step 150,000, this
  was nothing.
- **The learning rate is too high for this mix.** 6e-4 was carried over
  from the 36k-step Hindi runs, where the whole cosine fitted inside the
  budget. If val is still ~3.44 at step 150,000, the fix is a lower peak
  (3e-4) or a shorter cosine, restarted from `ckpt_coder.pt` rather than
  continued.

Nothing to change yet. Note it, run phase 2, and look again at 150,000.

### pass@1 at step 100,000: zero, as expected

`eval_code.py ckpt_coder.pt` on both tests, greedy, one sample per problem,
the base checkpoint prompted in `continue` mode:

| benchmark | problems | pass@1 |
| --- | --- | --- |
| HumanEval | 164 | **0.0%** (0/164) |
| MBPP (sanitized test) | 257 | **0.0%** (0/257) |
| *reference: CodeParrot-110M ~4% (25B tokens), SantaCoder-1.1B ~18%, phi-1-small-350M ~45%* | | |

The two ran concurrently in about an hour: decoding is launch-bound, so a
second process raised GPU utilisation from 8% to 61% and cost almost
nothing. Samples at temperature 0.2 say why the score is zero — the model
writes Python-shaped text with no working logic, and falls into the
repeated-token loop every base model in this log shows:

```
def is_prime(n):        ->  if n == 'default': return False ... then a
                            `def check(self, check)` repeated four times
def fibonacci(n):       ->  # n = n + 1 + n + n + n + n + n ...
# Compute the factorial ->  the factorial of n-neutrality in the number of
                            n-neutralities in the number of ...
```

This is the expected trajectory, not a problem: 410M of the corpus's 2.78B
tokens are in, 14% of the schedule, and the plan's 10-25% is the figure
for step 700,000 *after* instruction tuning. What the run buys is a
zero-point. Re-run both at every phase boundary; the first non-zero is the
signal worth acting on, and the gap between base and instruction-tuned at
the end is what `coder_sft.sh` is for.

## Phase 2: steps 100,000 to 250,000 (done 2026-09-12 02:24)

Same command with `--stop-at 250000`, launched by the `nanosarvam_coder`
scheduled task rather than a terminal. 150,000 steps in 16 h at 0.398
s/step; running it alongside the MBPP eval peaked at 12.7 GB of the 16.3
GB card, so the two coexisted without WDDM spillover.

| step | 100k | 125k | 150k | 175k | 200k | 225k | **250k** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val | 3.440 | 3.417 | 3.368 | 3.367 | 3.335 | 3.312 | **3.276** |

**−0.160, ending on its own best**, and the phase-1 plateau is explained:
the step-150,000 test ("materially below 3.43 or change the schedule")
came in at 3.368 and the bar had already been passed at 125,000. The
learning rate stays at 6e-4. A second flat stretch at 195k–225k resolved
the same way, so treat flat runs of five evals as the noise floor.

### pass@1 at step 250,000: still zero, but the samples moved

| | step 100,000 | step 250,000 |
| --- | --- | --- |
| HumanEval | 0.0% (0/164) | 0.0% (0/164) |
| MBPP, sanitized test | 0.0% (0/257) | 0.0% (0/257) |
| val loss | 3.4361 | 3.2758 |

The score is the same and the model is visibly not. Same three prompts,
same seed and temperature 0.2, at the two checkpoints:

```
def is_prime(n):
  100k   if n == 'default': return False   ... then `def check(self, check)` x4
  250k   if n == 1: return n / else: return n / return 0
         then __add__, __radd__, __rsub__ with isinstance(other, self.__class__)

def fibonacci(n):
  100k   # n = n + 1 + n + n + n + n + n + n + n ...
  250k   \"\"\"Find the index of a fibonacci number from the number of n\"\"\"
         for i in range(0, n + 1, -1):
             if n % n == 0: return i
         return -1

# Compute the factorial of n
  100k   the factorial of n-neutrality in the number of n-neutralities in ...
  250k   n-way of the gyro. / - Ayurvedic: Ayurvedic is the term used to ...
```

At 100,000 steps the completions were Python-shaped noise. At 250,000 they
are *wrong programs* — a docstring that states a plausible task, a `for`
over `range(...)`, a modulo test, a `return -1` sentinel, and a run of
dunder methods using `isinstance` and `self.__class__` correctly as
idiom. The logic is still broken in every case (`range(0, n+1, -1)` is
empty, `n % n` is always 0), the repetition loop is still there, and the
third prompt still escapes into English prose, which is the 30% of the
corpus that is fineweb-edu showing through a comment.

**pass@1 is the wrong instrument at this end of the curve.** It is
thresholded on executing correctly against hidden tests, so it reports
zero across an interval in which the model learned function structure,
docstrings, control flow and class protocol. Val loss moved 0.160 over the
same span and says more. Keep running both benchmarks at each phase
boundary for the series, but read the samples and the loss until the first
non-zero appears.

### What the step-250,000 model actually writes

`sample_many.py` (new: loads the checkpoint once and walks a prompt list,
where `sample.py` reloads 1.6 GB per prompt) over nine prompts spanning
the corpus mix — six Python, two English, one Hindi.

**Every completion is a repetition loop.** Not one of the nine escaped it
inside 90 tokens; the only one that terminated was an English prompt that
hit EOS at 50. What differs is how long the model stays coherent before
the loop closes, and there the structure is real:

```
class Stack:
    def __init__(self):
        self.stack = []          <- correct, unprompted
    def get_frame(self, frame):
        if isinstance(frame, Packet):
            return frame
        else:
            raise ValueError("Invalid Packet")
    def get_frame_size(self, frame):
        # TODO: This is a hack to get the size of the frame.
```

`isinstance`, `raise ValueError`, a TODO comment, and a class body that
reads like the scapy-ish code in codeparrot. Elsewhere: correct list
comprehensions (`[file.strip() for file in files if os.path.isfile(file)]`),
`glob.glob`, docstring-then-body layout, `try/except StopIteration`. The
idiom is learned; the semantics are not (`def add(a, b): return (a, b)`).

**A repetition penalty delays the loop but does not prevent it.** At
`--rep 1.15` the completions stay varied for perhaps twice as long and
then close anyway, usually on a longer cycle. That is worth knowing for
two reasons. The demo should not decode this model greedily —
`serve.py` already applies 1.3 in question mode. And `eval_code.py` scores
pass@1 with greedy decoding, which is the *worst* case for a model whose
dominant failure is looping, so the zeros are partly a decoding artifact
on top of a genuine capability gap. Do not "fix" that by adding a penalty
to the benchmark: greedy pass@1 is the number every coding paper reports,
and changing it would make these results incomparable. Note it, and read
the samples alongside.

English and Hindi both still work — the English prompt produced fluent
if vacuous prose about learning and stopped cleanly at EOS, the Hindi one
produced fluent Devanagari with invented distances — so the 30/5 split of
the mix has not been forgotten while Python was learned.

**One interruption, and the last one of its kind.** The phase was killed
at step 105,000 by a Ctrl+C reaching the training process: a scheduled
task registered `/it` runs under an interactive token and therefore shares
the interactive console. It shows as `Last Result: -1073741510`
(`STATUS_CONTROL_C_EXIT`) on the task and a bare `^C` in the stderr log,
and it was also what killed phase 1 at step 10,000. `run_coder_phase.cmd`
now launches `_coder_run.cmd` through `start`, giving the trainer its own
console; the 16 h that followed were uninterrupted.
