# Task list

> **For readers who are not the operator of this machine:** this file is
> the operations log of the Windows PC with one RTX 5070 Ti on which every
> run in `docs/RESULTS.md` was done: which scheduled task owned which run,
> how a killed run was resumed, and the mistakes that cost idle GPU hours.
> It is kept because the *process* is part of what is being published.
> Nothing in it is needed to use the repository; `docs/TUTORIAL.md` is the
> place to start, and the "Running for days without a terminal" section
> there gives the Linux equivalents of the Task Scheduler pattern below.

> **Closed again 2026-09-19 06:29.** The long-context run finished its
> 10,000 steps unattended overnight and its tasks are disabled; the section
> below is the record of it. `docs/RESULTS.md` §28 has the result.

> **Closed 2026-09-18.** Every run below is finished. The training rig it
> describes — the working directory it ran in before the rename, holding the
> checkpoints, corpora and logs — has since been deleted, and the
> five scheduled tasks that drove it (`anulm_coder`, `_updater`, `_refresh`,
> `_sft`, `_serve`) have been removed from the machine, so nothing here is
> running any more and the *Live status* block at the bottom is the last
> reading rather than a current one. Two of them, `anulm_coder` and
> `anulm_updater`, have since been re-registered **disabled** against this
> checkout, ready for a retrain — see "Retraining on more data" below. `C:\workspace\AnuLM` is now the only
> checkout, and the `.cmd` wrappers are kept, renamed to the project's
> current name, as the recipe for anyone re-registering the pattern. What
> survives the rig is what was published: the four exported checkpoints on
> Hugging Face, `coder_curve.csv`, and the numbers in `docs/RESULTS.md`.

The to-do list for the two demos as it stood, with the command that resumed
each step. It was kept current by `update_tasks.py`, which rewrote the
*Live status* block every hour while a run was in flight (`python
update_tasks.py --loop 3600`), and could be run once by hand.

## If the run stopped (power cut, closed session, crash)

1. Check where it stopped: `python update_tasks.py` prints the step of
   `ckpt_coder.pt.last` and refreshes this file.
2. Resume with the phase target you were in: `bash experiments/coder_train.sh <STOP>`.
   It sees `ckpt_coder.pt.last`, adds `--resume`, and continues; at most
   5,000 steps (~33 min) of work are lost.
3. Restart the hourly updater: `python update_tasks.py --loop 3600`.

The translation checkpoint is finished and safe; nothing there needs
restarting.

### How to launch so it survives

Two traps, both learned the hard way on 2026-09-10:

- **The `bash.exe` on PATH is the WSL stub**, not Git Bash. Use
  `C:\Program Files\Git\usr\bin\bash.exe`, or `run_coder_phase.cmd`, which
  hard-codes it.
- **Ctrl+C in the interactive console kills the training, wherever it was
  launched from.** This is the one that cost real time, twice. A scheduled
  task registered with `/it` runs under an *interactive token* and shares
  the interactive session's console, so a Ctrl+C delivered there — by a
  shell exiting, a tool call being interrupted, a session ending — reaches
  the trainer too. The signature is the task's `Last Result:
  -1073741510`, which is `STATUS_CONTROL_C_EXIT` (0xC000013A), and a bare
  `^C` at the end of `coder_train_phase1.err`. It killed phase 1 at step
  10,000 on 2026-09-10 (2 h 20 min idle) and phase 2 at step 105,000 on
  2026-09-11 (31 min idle).

  The fix, in place since 2026-09-11 10:25: `run_coder_phase.cmd` no
  longer runs the trainer itself. It calls `_coder_run.cmd` through
  `start`, which gives that process **its own console**, out of reach of
  any Ctrl+C in the interactive one. The wrapper then returns immediately,
  so the task instance ends in seconds and can never block the next firing
  either.

- **A new console is not enough on its own: only the Task Scheduler can
  start something that outlives a session.** `start` defeats Ctrl+C, but
  the new process is still a *descendant* of whatever launched it, and
  Windows kills the whole tree when the launching session's job object
  closes. The instruction-tune probe was launched with `start` from a
  terminal on 2026-09-12 and died at 06:42 with the session anyway, 2 h 20
  min of idle GPU, while the task-launched training beside it had survived
  16 h. The rule, finally: **if it must outlive the session, register a
  scheduled task — never `start`, never a bare command.** `anulm_sft`
  exists for exactly that reason.

- **A self-healing task can loop on a fatal error and look healthy.** A
  truncated `ckpt_coder.pt.last` made every restart die in 11 seconds;
  the task kept firing, the guard log kept filling, and nothing said
  anything was wrong for ~3 h. `update_tasks.py`'s hourly line is the
  check that catches this -- it reports the checkpoint's step, and a step
  that has not moved in two entries means the loop is spinning. Read the
  step, not the fact that a task is running.

- **Every 30-minute task needs a guard and a done-marker, or it will
  stampede.** `anulm_sft` was registered without either for a few
  minutes, and its next firing started a *second* fine-tune against the
  same checkpoint; worse, the probe's first run had been launched by the
  marker-less wrapper, so when it finished cleanly no marker was written
  and the task restarted the whole thing, truncating both `eval_*_sft.log`
  files. The numbers survived only because `coder_sft.sh` echoes the
  tables into `coder_sft.log` as well. Both wrappers now check for a live
  process *and* a completion marker before doing anything.

### The self-healing setup (registered 2026-09-10 19:08)

Two scheduled tasks now own the work, so nothing depends on a terminal
staying open:

| task | runs | every | purpose |
| --- | --- | --- | --- |
| `anulm_coder` | `run_coder_phase.cmd 700000` | 30 min | keep training alive |
| `anulm_updater` | `run_updater.cmd` | 30 min | start the `--loop 3600` refresher if none is alive |
| `anulm_refresh` | `run_refresh.cmd` | 30 min | one refresh and exit, no guard |
| `anulm_sft` | `_sft_run.cmd` | disabled | re-enable for the final instruction tune |

**Why both `_updater` and `_refresh`.** The looping refresher survives
`schtasks /end` and keeps whatever code it started with, so a fix to
`update_tasks.py` would not take effect until a reboot. `anulm_refresh`
runs the script fresh every 30 minutes and exits, so the newest code always
runs and a stale loop cannot block it. They write the same two files through
`os.replace`, so overlapping is harmless. Each has its own log, because
`cmd`'s `>>` takes an exclusive handle and sharing one makes the second
wrapper die with "cannot access the file".

### What is saved, and how often

| what | where | every | survives a power cut? |
| --- | --- | --- | --- |
| weights + optimizer + sampler position | `ckpt_coder.pt.last` | 5,000 steps (~33 min) | yes, and the write is atomic |
| best-val weights | `ckpt_coder.pt` | when val improves | yes, atomic |
| the val-loss curve | `coder_curve.csv` | 30 min | yes |
| live status for you | `TASKS.md` live block | 30 min | yes |
| one status line, appended | `tasks_history.log` | 30 min | yes |
| live state for a fresh assistant session | `…/memory/live-training-state.md` | 30 min | yes |

**Worst case from a power cut: 33 minutes of training**, plus up to 30
minutes before the task notices and restarts. Nothing else is lost.

`coder_curve.csv` exists because the curve is the most valuable artefact of
a 50-hour run and it otherwise lived in exactly one place,
`coder_train_phase1.log` — a single file, written by a wrapper whose append
handle a crash can leave in any state, and this project has already lost
two eval logs to a task that restarted and truncated them. The CSV is
rebuilt from every `coder_train_phase*.log` on each refresh, *keeps rows
whose log line has since disappeared*, and is written with `os.replace`.

Six points had already been lost before it existed. Four were recovered
from values recorded elsewhere (steps 5,000 and 10,000 from the phase-1
table in `docs/CODER_PLAN.md`; 105,000 and 265,000 from checkpoints read at
the time) by `tools_backfill_curve.py`, which is kept so the provenance is
auditable. **Steps 255,000 and 260,000 are permanently gone** — they were
written only to the log of the run that was killed at 265,000.

**What a refresh writes.** The *Live status* block at the bottom of this
file, a line in `tasks_history.log`, the curve CSV, and
`~/.claude/projects/C--workspace-AnuLM/memory/live-training-state.md`
— the last one so a fresh assistant session after a power cut inherits the
current step, the resume command and the open decisions instead of
re-deriving them.

`run_coder_phase.cmd` does three things on each firing: alert once if the
phase has just completed, exit if training is already alive, otherwise
launch `_coder_run.cmd` detached through `start`. The guard looks for a
live `python.exe` whose command line contains `train.py` (or
`update_tasks.py` for the updater) and does nothing if it finds one. So a firing while the run is healthy is a
no-op, logged to `coder_train_guard.log`, and a firing after a crash,
closed session, reboot or power cut restarts the run from
`ckpt_coder.pt.last`. **Worst case after any interruption: 30 minutes of
idle plus at most 33 minutes of unsaved steps.**

Managing them, from any shell:

```
schtasks /query /tn anulm_coder /fo list        status and next run
schtasks /run   /tn anulm_coder                 start now, if not already running
schtasks /end   /tn anulm_coder                 stop the task's process
schtasks /change /tn anulm_coder /tr "C:\workspace\AnuLM\run_coder_phase.cmd 250000"
schtasks /delete /tn anulm_coder /f             when the coder is finished
```

**When a phase completes** the wrapper beeps six times, writes
`anulm_phase_<step>_done.txt` to the Desktop, logs `PHASE <step>
COMPLETE` to `coder_train_guard.log`, and drops a
`phase_<step>_done.marker` so it never alerts twice. The alert is
deliberately not a dialog box: a modal window would hold the task
instance open and the scheduler would skip the next firing, leaving
training dead.

**Advancing a phase is that `/change` line.** When phase 1 reaches step
100,000 the trainer exits immediately on every later firing, since
`--stop-at` is already met, and it will keep doing that harmlessly until
the step in the task is raised. Point it at 250000, then 400000, 550000,
and finally `run_coder_phase.cmd` with no argument for the last stretch
to 700,000.

One thing the tasks do not cover: the training process running *right
now* was started by an assistant session at 18:58 and will still die when
that session ends. That is fine, because `anulm_coder` will pick it
back up at the next half-hour boundary. From the first task-started run
onward, sessions are irrelevant.

## Translation demo (docs/TRANSLATE_PLAN.md) — done

| # | step | status |
| --- | --- | --- |
| 1 | fetch Samanantar, IITB, FLORES-200 | done 2026-09-10 |
| 2 | build 2M pairs both ways + FLORES devtest | done 2026-09-10 |
| 3 | fine-tune `ckpt_multi36k` → `ckpt_translate.pt`, 61,672 steps | done 2026-09-10 14:12, held-out loss 4.05 → 2.297 |
| 4–5 | chrF on all 1,012 FLORES sentences per direction | done 2026-09-10 15:00: **41.5 en→hi, 43.4 hi→en** |
| 6 | demo: `python serve.py --ckpt ckpt_translate.pt` | ready |

## Coder demo (docs/CODER_PLAN.md) — done

Each phase call trains from the last save to the step given and exits.
0.39 s/step measured; 5,000 steps between saves.

| phase | command | steps | time | status |
| --- | --- | --- | --- | --- |
| 1 fetch | `bash experiments/coder_fetch.sh` | | | done 2026-09-10 |
| 2 build | `bash experiments/coder_build.sh` | | | done 2026-09-10 05:45 |
| 3 | `bash experiments/coder_train.sh 100000` | 0 → 100k | 11 h | **done 2026-09-11 04:52**, best val 3.4361 at step 85k; curve in docs/CODER_PLAN.md |
| 4 | `bash experiments/coder_train.sh 250000` | → 250k | 16 h | **done 2026-09-12 02:24**, best val 3.2758 (final eval); curve in docs/CODER_PLAN.md |
| 4b | instruction-tune the step-250k checkpoint for an early read | `EX_CAP=200000 bash experiments/coder_sft.sh` | 3 h | **done 2026-09-12 11:03: MBPP 5.1% (13/257), HumanEval 0.0%**, answer loss 2.28 → 1.15 |
| 5-7 | pretrain to the end, one unattended run (phases collapsed) | task `anulm_coder` → `run_coder_phase.cmd 700000` | 47.8 h | **done 2026-09-14 18:38**, best val 2.7222 at step 685k; task now disabled |
| 8 | `bash experiments/coder_sft.sh` | instruction tune + pass@1 | 7.7 h uncapped | **done 2026-09-15 04:01**: MBPP **12.5%** (32/257) tuned, HumanEval **4.9%** (8/164) base; answer loss 1.644 -> 0.918 |
| 9 | demo: `python serve.py --ckpt ckpt_coder_sft.pt` | | minutes | **done**: served at http://127.0.0.1:8000 from 2026-09-18, owned by the task `anulm_serve` (`run_serve.cmd`, every 30 min, restarting the server if it was down) until the rig was deleted and the task with it. To serve it again from this checkout, point `serve.py` at an exported folder: `python serve.py --ckpt release/AnuLM-Coder-400M`, or `hf download toonist/AnuLM-Coder-400M --local-dir AnuLM-Coder-400M` first. |

A checkpoint at any phase boundary is usable: `python eval_code.py
ckpt_coder.pt --bench humaneval --device cuda` gives the current pass@1,
and `serve.py` serves it. Both benchmarks run concurrently in about an
hour, and can share the GPU with training (12.7 GB of 16.3 GB together).
**The base checkpoint is 0.0% on both at steps 100,000 and 250,000**, and
that is the expected zero-point rather than a warning: over the same span
val loss moved 3.4361 -> 3.2758 and the samples went from Python-shaped
noise to wrong-but-real programs. **Instruction-tuned at step 250,000 it
scores MBPP 5.1% (13/257), HumanEval 0.0%** -- the first non-zero, and the
evidence that the remaining pretraining is buying a known-good pipeline
more compute. Watch MBPP, not HumanEval, at this size. See
docs/CODER_PLAN.md.

## Retraining on more data

The pattern is registered and waiting. Nothing fires until it is enabled,
because there is no corpus and no `.pt` on this machine yet — the training
rig went with the old directory, and what survived it is the exported
checkpoints, the tokenizers under `data/`, `coder_curve.csv` and the
numbers in `docs/RESULTS.md`.

| task | runs | every | state |
| --- | --- | --- | --- |
| `anulm_coder` | `run_coder_phase.cmd 700000` | 30 min | **disabled**, registered 2026-09-18 |
| `anulm_updater` | `run_updater.cmd` | 30 min | **disabled**, registered 2026-09-18 |

To retrain, in order:

1. **Get the data back.** `bash experiments/coder_fetch.sh` then
   `bash experiments/coder_build.sh` rebuild the coder corpus from public
   sources (~15 GB down, ~25 GB on disk, 1 h + 2 h). For a different mix,
   `docs/DATASETS.md` lists every source with the script that fetches it.
   Nothing is redistributed, so this is always a fresh fetch.
2. **Decide where it starts.** From scratch, or continued from a released
   checkpoint: `hf download toonist/AnuLM-Base-400M --local-dir
   AnuLM-Base-400M`, which every script takes in place of a `.pt`. No `.pt`
   with optimizer state survives, so a continuation starts from weights
   alone and puts the usual dent in the loss curve (`docs/DEVELOPING.md`,
   "Checkpoints").
3. **Point the task at the target step** and enable it:

```
schtasks /change /tn anulm_coder /tr "C:\workspace\AnuLM\run_coder_phase.cmd <STEP>"
schtasks /change /tn anulm_coder /enable
schtasks /change /tn anulm_updater /enable      # hourly Live status refresh
schtasks /run    /tn anulm_coder                # or wait for the next half hour
```

4. **Watch it.** `python update_tasks.py` prints the step of the newest
   `.last` and rewrites the *Live status* block below; `--loop 3600` keeps
   it current, which is what `anulm_updater` starts. To recreate the serve
   task once there is something to serve:
   `schtasks /create /tn anulm_serve /tr "C:\workspace\AnuLM\run_serve.cmd" /sc minute /mo 30 /it`.
5. **Stop.** `schtasks /change /tn <name> /disable` keeps the registration
   for next time; `/delete /tn <name> /f` removes it. Disable both when the
   run is done — leaving them firing is the mistake this file exists to
   record.

**The alarm.** `run_coder_phase.cmd` watches for the phase it was given to
complete, and on the firing after it does: **six 880 Hz beeps**, an
`anulm_phase_<step>_done.txt` on the Desktop, a `PHASE <step> COMPLETE`
line in `coder_train_guard.log`, and a `phase_<step>_done.marker` so it
never alerts twice. Deliberately not a dialog box: a modal window holds the
task instance open and the scheduler skips the next firing, which is how a
crashed run would go unnoticed for half an hour. A firing while training is
healthy is a no-op — the wrapper checks for a live `train.py` first.

## Long context at 2,048 — done (01:37 to 06:29, 2026-09-19)

The training half of the "Longer context" item. Every released checkpoint is
trained at 512 tokens; this continues the released base at **2,048 with YaRN**
and then re-measures with `eval_context.py`, to see whether the design's
long-context claim survives at 400M rather than at the 17M/47M of
`docs/RESULTS.md` §3.

The zero-shot half is already measured and needed no training (§27).

| | |
| --- | --- |
| task | `anulm_ctx` -> `run_ctx_phase.cmd 10000`, every 30 min, **enabled** |
| launcher | `_ctx_run.cmd` (its own console, out of reach of Ctrl+C) |
| from | `release/AnuLM-Base-400M`, upcast to fp32, written as `ckpt_ctx2k.pt.last` at step -1 |
| data | `data/ctx_mix.multi32k.bin` -- 46.0M tokens, 189 MB Hindi + 79 MB English + 38 MB Python, the base's own proportions and its `multi32k` tokenizer |
| shape | block 2,048, batch 2 x grad-accum 4 = 8,192 tokens/step, `--yarn` with `yarn_original_context=512 yarn_factor=4.0` |
| budget | 10,000 steps = 82M tokens, about 2 epochs, ~5 h at the measured 1.8 s/step |
| **outcome** | **finished 06:29 unattended**, val 4.3031 at step 250 to **4.0501** best. Ten task firings, nine of them no-ops while training was healthy, one launch. No crash, no resume needed. `ctx2k_curve.csv` has all 40 points |
| logs | `ctx_train.log`, `ctx_train.err`, `ctx_train_guard.log` |

**Why `yarn_original_context=512` is set by hand.** `train.py` defaults it to
`--block-size`, which is right for a fresh run and wrong for an extension: the
NTK-by-parts correction range has to be measured against the length the model
was *trained* at, not the one it is being extended to. Left at the default it
would be the trap `docs/ARCHITECTURE.md` §4 calls the most-often-mis-set field
when copying a `rope_scaling` block.

### Checking on it

```
type ctx_train.log                     the curve; eval every 250 steps
type ctx_train_guard.log               one line per task firing
schtasks /query /tn anulm_ctx /fo list status and next run
python update_tasks.py                 step of the newest .last, and refreshes this file
```

A firing while training is healthy is a no-op; a firing after a crash resumes
from `ckpt_ctx2k.pt.last`, losing at most 250 steps. When step 10,000 lands,
the wrapper beeps six times and drops `anulm_ctx_10000_done.txt` on the
Desktop.

### What it found, and one thing it nearly got wrong

Both tasks are disabled. `docs/RESULTS.md` §28 is the write-up; the short
version is that nothing regressed in any language (Hindi 0.6001 -> 0.5323
bits/byte, English 1.5208 -> 1.3897, Python 1.0433 -> 0.8102) but almost none
of that is about context -- it is 55% more tokens on a model that had seen
148M. The context question is the *slope* of loss across a 2,048-token
window, and there zero-shot YaRN had already done most of the work: the base
goes from +0.073 (worse the further out) to -0.068 with YaRN alone, and five
GPU-hours of training at 2,048 moved it to -0.085.

**The first evaluation was contaminated and its numbers were discarded.**
`mix_corpus.py` was given the whole of `data/hindi_ctx.txt`, and
`eval_context.py` evaluates on the last 10% of the file handed to it -- so the
eval text was in the training corpus, and every length showed a suspiciously
uniform -0.72. Nine probe passages out of nine from that tail are in the mix;
zero of the same nine are in `data/bench_hindi.txt`, which `mix_corpus.py`
holds out and which everything published uses. A held-out split of a file is
not held out if the file went into the mix whole.

## Housekeeping

- Both stale copies are gone: the duplicate under
  `C:\workspace\custommodel\Transformers\` and the training rig the
  project ran in before the rename. `C:\workspace\AnuLM` is the only
  checkout, and no path in this repository points anywhere else.
- The val-loss curve of the whole run is in `docs/MODEL_CARD.md` (every
  50k steps) and `coder_curve.csv` (all 138 points); `docs/RESULTS.md` §25
  has it phase by phase. Done 2026-09-18.
- `docs/MODEL_CARD.md` (2026-09-18) is the one-page summary of the finished
  coder: data sources and sizes, architecture from the checkpoint, training,
  benchmark numbers by prompt mode, how to run it.
- All five scheduled tasks were deleted on 2026-09-18 once the rig went
  (`schtasks /delete /tn <name> /f`). They had gone on firing every 30
  minutes at a directory that no longer existed, failing silently — a
  scheduled task outlives the thing it was written for, which is the last
  operational lesson in this file. `anulm_coder` and `anulm_updater` were
  then re-registered against `C:\workspace\AnuLM` and left **disabled**, so
  the pattern is one command away without anything firing at an empty
  directory meanwhile. `anulm_serve`, `_refresh` and `_sft` were not
  re-created; the section above has the line that recreates one.

<!-- live-status -->
## Live status

Written 2026-09-19 09:13 by `update_tasks.py`. **No training process running**; GPU: 0 MiB, 16303 MiB, 0 %.

| checkpoint (`.last`) | step | val loss | best val | saved |
| --- | --- | --- | --- | --- |
| `ckpt_coder.pt` | not on disk | | | |
| `ckpt_ctx2k.pt` | 10,000 | 4.0505 | 4.0501 | 2026-09-19 06:29 |

Tail of `ctx_train.log`:

```
optimizer: AdamW (fused), 397.7M params
resuming from ckpt_ctx2k.pt.last at step 10000 (best val so far 4.0501, epoch 3.96)
done in 0s | best val 4.0501 | ckpt ckpt_ctx2k.pt
sample with:  python sample.py --ckpt ckpt_ctx2k.pt
```
<!-- /live-status -->
