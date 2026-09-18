# Translation demo first, coder second

**Status: finished 2026-09-10.** `ckpt_translate.pt` scores chrF **41.5
en->hi / 43.4 hi->en** on all 1,012 FLORES-200 devtest sentences per
direction, is released as
[toonist/AnuLM-Translate-400M](https://huggingface.co/toonist/AnuLM-Translate-400M)
(card: `docs/MODEL_CARD_TRANSLATE.md`), and `serve.py` serves it. The log
is `docs/RESULTS.md` §24. The rest of this file is the plan as written, with
what actually happened recorded against it.

Decided 2026-09-10: the English <-> Hindi translation demo ran before the
Python coder (docs/CODER_PLAN.md), because it is a day of GPU instead of a
week. The coder's data and corpus were built and waiting; its training
started when the translation run released the GPU (it finished 2026-09-15).

Both runs stop and resume the same way: every training script takes an
optional step number, trains from the last save to that step, evaluates,
saves, and exits. Call it again to continue. If the machine sleeps or the
process dies mid-phase, call it again: it resumes from `<checkpoint>.last`,
losing at most one eval interval.

## To do

| # | Step | Command | Time | Status |
| --- | --- | --- | --- | --- |
| 1 | Fetch Samanantar (hi), IITB en-hi, FLORES-200 | `fetch_hf.py` x2, curl | 10 min | done 2026-09-10 |
| 2 | Build 2M pairs both directions + FLORES devtest | `experiments/translate_build.sh` | 5 min | done 2026-09-10: 4M items, 493,370 packed rows |
| 3 | Fine-tune from ckpt_multi36k, 1 pass, 61,672 steps | `experiments/translate_train.sh [stop]` | 6 h GPU | done 2026-09-10 14:12; held-out loss 4.05 -> 2.30 |
| 4 | Score chrF, 300 sentences/direction | `experiments/translate_eval.sh` | 40 min | skipped for the full run; 50-sentence preview at step 48k was 44.5 / 43.0 |
| 5 | Score chrF on all 1012 (the number to quote) | `python eval_translate.py ckpt_translate.pt --device cuda` | 55 min | **done 2026-09-10 15:00: 41.5 en->hi, 43.4 hi->en** (docs/RESULTS.md §24) |
| 6 | Demo: web UI translate mode | `serve.py --ckpt ckpt_translate.pt` | minutes | done: the page shows a *translate* mode whenever the checkpoint carries the direction templates |
| 7 | Hand the GPU to the coder | `experiments/coder_train.sh 100000` ... | days 3-7 | done: the coder took the GPU 2026-09-10 15:29 and finished 2026-09-15 (docs/CODER_PLAN.md) |

Suggested phasing for step 3: `translate_train.sh 25000` (3 h), sleep,
`translate_train.sh 50000` (3 h), sleep, `translate_train.sh` (to the end).
Or one call overnight.

What actually happened: the first call ran to 25,000 (2 h 44 min, held-out
2.606); the second was started with no stop step and died at some point
after its step-48,000 save (the log shows nothing past the phase header
because of `grep`'s block buffering, but `ckpt_translate.pt.last` says
47,999, loss 2.348). The third call resumed and finished the remaining
13,672 steps in 1 h 28 min (done 14:12, final held-out loss 2.297). The
loss curve is in docs/RESULTS.md §24. Found on the way: every experiment
script exported the MSYS PATH *after* the `cd` that needs `dirname`, so a
detached Git Bash landed in the wrong directory, saw no `.last`, and
announced a fresh start; it failed before overwriting anything, and the
export now comes first in all nineteen scripts.

## What is being trained

- Base: `ckpt_multi36k.pt`, the 398M three-language model (already fluent
  in Hindi and English; translation is the alignment it lacks).
- Data: 2M sentence pairs from Samanantar and the IIT Bombay corpus,
  filtered for length, script and duplicates, each used in both directions
  (`English: ... \nHindi: ...` and the reverse), ~280M tokens.
- Loss on the target sentence only; EOS ends it, so the UI gets one clean
  sentence.
- Held out: 1,000 pairs from the training distribution (for the loss
  curve) and FLORES-200 devtest (for the number), which no training data
  touches.

## What to expect

chrF on FLORES devtest, English -> Hindi: NLLB-600M ~55, IndicTrans2 and
Google ~60. A 400M decoder after one pass over 2M pairs should land in the
40s; a second pass and the remaining 8M Samanantar pairs would add a few
points for another day of GPU. Hindi -> English scores higher for every
system. The demo sentences that work best are everyday statements and
questions, 8 to 25 words; rare proper names and numbers are where it slips.

## The demo

`serve.py --ckpt ckpt_translate.pt` shows a "translate" mode when the
checkpoint carries the translation templates. Type English, get Hindi; type
Hindi, get English -- the direction is picked from the script. Keep
temperature at 0.2 and turn the repetition penalty off for translation.
