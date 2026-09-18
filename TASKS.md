# Task list

> **For readers who are not the operator of this machine:** this file is
> the operations log of the Windows PC with one RTX 5070 Ti on which every
> run in `docs/RESULTS.md` was done: which scheduled task owned which run,
> how a killed run was resumed, and the mistakes that cost idle GPU hours.
> It is kept because the *process* is part of what is being published.
> Nothing in it is needed to use the repository; `docs/TUTORIAL.md` is the
> place to start, and the "Running for days without a terminal" section
> there gives the Linux equivalents of the Task Scheduler pattern below.

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

Written 2026-09-18 22:07 by `update_tasks.py` and **frozen there**: the last
reading taken before the rig was deleted, not a live one. The coder run had
finished at step 700,000 four days earlier, and the process this line called
running was the final phase wrapper on its way out. GPU at the time: 9499 MiB
of 16303 MiB, 0 %.

| checkpoint (`.last`) | step | val loss | best val | saved |
| --- | --- | --- | --- | --- |
| `ckpt_coder.pt` | 700,000 | 2.7234 | 2.7222 | 2026-09-14 18:38 |

Tail of `coder_train_phase1.log`:

```
resuming from ckpt_coder.pt.last at step 700000 (best val so far 2.7222, epoch 0.71)
done in 0s | best val 2.7222 | ckpt ckpt_coder.pt
sample with:  python sample.py --ckpt ckpt_coder.pt
=== PHASE DONE (18:42) ===
```
<!-- /live-status -->
