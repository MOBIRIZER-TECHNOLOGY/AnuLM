---
title: "25 · Checkpoints, resuming and failure"
part: "V · Training"
accent: teal
pinned: "@084cbdd"
---

# 25  Checkpoints, resuming and failure

The released model trained for 1.9 million steps on one GPU, in phases, with
restarts. This chapter reads how `anulm/training/train.py` saves, resumes, and survives a
broken resume point; each mechanism came from a failure, told with it.
Prerequisites: Chapters 22 and 24. A bare "§n" is a section of `docs/RESULTS.md`.

| Term | Meaning |
|---|---|
| checkpoint | a file holding a model's state, written with `torch.save` |
| best-val file | `--out`: the weights at the lowest val loss so far, for sampling |
| resume point | `<out>.last`: the newest step, with everything needed to continue it |
| optimizer moments | AdamW's running averages of each gradient and its square (Chapter 7) |
| atomic write | here: a write that, if the process is killed, either completes or leaves the old file untouched |

## 25.1  Two files

At every evaluation `anulm/training/train.py` writes up to two checkpoints:

> **In the repo** · `anulm/training/train.py` · `main` · lines 888–902 · @084cbdd

```python
        if (step + 1) % args.eval_every == 0 or step == end - 1:
            val, val_sem = evaluate(raw_model, val_data, args)
            flag = ""
            if val < best_val:
                best_val = val
                atomic_save({"model": raw_model.state_dict(), "cfg": cfg,
                             "step": step, "val_loss": val, "val_sem": val_sem,
                             "eval_spec": eval_spec}, args.out)
                flag = "  <- saved"
            # Always write the resume point, best or not: it exists to recover
            # from a kill, and the newest step is what we want to continue from.
            atomic_save({"model": raw_model.state_dict(), "opt": opt.state_dict(),
                         "cfg": cfg, "step": step, "val_loss": val, "val_sem": val_sem,
                         "eval_spec": eval_spec, "best_val": best_val,
                         "sampler": sampler.state()}, last_path)
```

- **When:** every `--eval-every` steps, and at this invocation's last step.
- **The best-val file**, only on improvement: weights, configuration, step and
  `eval_spec`, all a sampler needs.
- **The resume point**, every time: adds the optimizer state, `best_val` and
  the sampler's position.
- **`state_dict()`** holds parameters *and buffers*, such as each router's
  `expert_bias`, which steers routing without being trained (Chapter 18).

The released run's pair, read memory-mapped:

![Figure 25.1 · What a resume point holds](../figures/fig25-1-sizes.svg)

*Figure 25.1 · What a resume point holds.* *measured*:
`book/code-examples/ch25_measure_resume_point.py` (with `--save`, it writes
`book/measurements/part_v_b_resume_point.json`). Sizes are decimal GB, as in
the code's comments; 1.59 GB is 1.48 GiB.

| | `ckpt_base_v2.pt` | `ckpt_base_v2.pt.last` |
|---|---|---|
| on disk | 1.59 GB | 4.86 GB |
| weights and buffers | 397,660,560 fp32 numbers, 1.59 GB | the same |
| AdamW `exp_avg`, `exp_avg_sq` | | 1.59 GB each |
| sampler | | epoch 0, 7,600,000 of 11,047,015 windows used |
| `step` (0-based), `val_loss`, `best_val` | 1,899,999, 3.3523 | 1,899,999, 3.3523, 3.3523 |

The optimizer state is twice the model. The run ended on its best
evaluation, so both files hold the same weights: all 1,551 tensors are equal
(*verified* by the same script).

`IN THE REPO` · `MEASURED`

## 25.2  A write that a kill cannot destroy

> **In the repo** · `anulm/training/train.py` · `atomic_save` · lines 32–46 · @084cbdd

```python
def atomic_save(obj, path) -> None:
    """torch.save, but a kill mid-write cannot destroy the existing file.

    `<out>.last` is 4.8 GB on the 400M coder and takes seconds to write.
    Killed in the middle -- and on Windows a scheduled task shares the
    interactive console, so Ctrl+C reaches it -- `torch.save` leaves a
    truncated zip that `torch.load` refuses with "failed finding central
    directory", i.e. the resume point is gone. That happened at step
    265,000 of the coder run on 2026-09-12 and cost the optimizer state.
    Write beside it and rename: `os.replace` is atomic within a volume, so
    the old checkpoint survives until the new one is complete.
    """
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
```

`torch.save` writes to `<path>.tmp`, never to the file a resume reads; only a
finished write is swapped into place by `os.replace`, in one step. A kill
leaves a broken `.tmp` and the old checkpoint intact. `anulm/training/finetune.py` uses it too.
The sizes in the story are different runs: 4.8 GB is the 400M coder's resume
point, 4.6 GB the Hindi 350M run's (§8), 4.86 GB the released run's (25.1).

> **Details.** What it does not cover. There is no `fsync` before the rename,
> so it guards against a killed process, not necessarily against power loss or
> an operating-system crash; and what happens when another process holds the
> target open is left to the platform. Neither is handled or tested here. A
> kill *between* the two saves of 25.1 is possible too: the best-val file is
> new, `.last` is old, and the resumed run's older `best_val` can let a worse
> model overwrite the best-val file at its next evaluation.

![Figure 25.2 · In-place and atomic checkpoint writes](../figures/fig25-2-atomic.svg)

*Figure 25.2 · In-place and atomic checkpoint writes.* *illustrative*; the
behaviour of each lane is verified by the book example below.

After the kill at step 265,000, a scheduled task restarted training every 30
minutes; each restart died on the unreadable file in 11 seconds, and for about
three hours it looked like healthy self-restarting (§25). Earlier, a kill had
truncated the best-val file and `.last` restored it (§8).

`book/code-examples/ch25_resume.py` (**Book example, not repository code**)
swaps `torch.save` for a writer that stops halfway, then saves in place and
through `atomic_save`:

```python
    def killed_save(obj, path):                     # writes half the file, then the process "dies"
        real_save(obj, str(path) + ".full")
        data = Path(str(path) + ".full").read_bytes()
        os.remove(str(path) + ".full")
        Path(path).write_bytes(data[: len(data) // 2])
        raise KeyboardInterrupt("killed mid-write")

    # ...
    for how in ("torch.save in place", "train.atomic_save"):
        path = tmp / f"{how.split('.')[1].split()[0]}.pt"
        real_save(old, path)
        torch.save = killed_save
        try:
            (T.atomic_save if how.startswith("train") else torch.save)(new, path)
        except KeyboardInterrupt:
            pass
        finally:
            torch.save = real_save
```

```
Part 2. a kill in the middle of a checkpoint write
  torch.save in place  -> save.pt RuntimeError: PytorchStreamReader failed reading zip archive: failed findi
  train.atomic_save    -> atomic_save.pt loads; holds the old checkpoint
```

*verified*: in place, the error the docstring quotes; through `atomic_save`,
the old checkpoint survives.

`IN THE REPO` · `IMPLEMENTATION-DEPENDENT`

## 25.3  What a resume point must hold

To continue exactly, a resume point needs everything that decides the next
step: weights and buffers, both AdamW moments, the step (it sets the learning
rate), the data order, kept by `WindowSampler` (Chapter 21), and any other
random state. `anulm/training/train.py` saves all but the last: the global RNG is reseeded at
line 649 and not saved, so dropout above 0 or `--sample-with-replacement`
(line 856) would not resume exactly. The sampler's state:

> **In the repo** · `anulm/training/train.py` · `WindowSampler.state` · lines 152–160 · @084cbdd

```python
    def state(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "order": self.order, "g": self.g.get_state()}

    def load_state(self, s: dict):
        self.epoch, self.pos, self.order = s["epoch"], s["pos"], s["order"]
        self.g.set_state(s["g"])
```

`order` is this epoch's shuffled starts, shape (W,) for W windows per epoch;
`pos` counts those used; `g`'s state makes the *next* epoch's shuffle match too:

> **In the repo** · `tests/test_model.py` · `window_sampler_resumes_exactly` · lines 812–822 · @084cbdd

```python
def window_sampler_resumes_exactly():
    from anulm.training.train import WindowSampler
    a = WindowSampler(50_000, 128, 4, seed=11)
    for _ in range(137):
        a.next()
    state = a.state()
    b = WindowSampler(50_000, 128, 4, seed=999)     # different seed: state must win
    b.load_state(state)
    for _ in range(300):                             # crosses an epoch boundary
        assert torch.equal(a.next(), b.next()), "resumed sampler diverged from the original"
    assert a.epoch == b.epoch and a.pos == b.pos
```

`b` has another seed, so only the state can make it agree, and 300 batches
cross an epoch boundary, where `g` matters. `checkpoint_roundtrip_is_exact`
(line 352) requires a reloaded model's logits to match exactly.

`IN THE REPO`

## 25.4  The resume path

With `--resume`, `main` reads the resume point before the first step:

> **In the repo** · `anulm/training/train.py` · `main` · lines 790–837 · @084cbdd

```python
    start_step, best_val = 0, float("inf")
    last_path = Path(str(args.out) + ".last")
    if args.resume and last_path.exists():
        # ...
        try:
            ck = torch.load(last_path, map_location="cpu", weights_only=False)
        except Exception as e:
            # ...
            print(f"WARNING: {last_path} is unreadable ({type(e).__name__}: "
                  f"{str(e)[:80]}).")
            if not Path(args.out).exists():
                raise
            ck = torch.load(args.out, map_location="cpu", weights_only=False)
            ck = {"model": ck["model"], "step": ck["step"],
                  "best_val": ck.get("val_loss", float("inf"))}
            # ...
        raw_model.load_state_dict(ck["model"])
        if "opt" in ck:
            assert len(ck["opt"]) == len(opt.opts), (
                f"{last_path} holds state for {len(ck['opt'])} optimizer(s) but --optimizer "
                f"{args.optimizer} builds {len(opt.opts)}; resume with the optimizer the run used")
            opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        best_val = ck.get("best_val", float("inf"))
        if "sampler" in ck:
            sampler.load_state(ck["sampler"])
        prev = ck.get("eval_spec")
        if prev and prev != eval_spec:
            print(f"  NOTE: this run measures val loss differently from the one it "
                  f"resumes ({prev} -> {eval_spec}). The curve's earlier points are "
                  f"not comparable with the ones from here.")
        # ...
        del ck
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    elif args.resume:
        print(f"--resume given but {last_path} does not exist; starting fresh")
```

1. **To the CPU**, then `del ck`. Loading the 4.6 GB blob onto an 8 GB GPU and
   keeping it spilled to system memory: 210 tokens per second, not about 4,400 (§8).
2. **Fall back, loudly.** An unreadable resume point is replaced by the
   best-val file's weights and step, with two warnings; if that is missing, the
   error is raised.
3. **Restore** the weights and buffers, the optimizer (checking the count:
   Muon builds two, Chapter 7), the step, `best_val` and the sampler.
4. **Compare `eval_spec`** (Chapter 24), and say so if the method changed.

`IN THE REPO`

## 25.5  Stopping, resuming and falling back

`--stop-at` ends an invocation early, after an evaluation and a resume point;
the next invocation passes `--resume`:

> **In the repo** · `anulm/training/train.py` · `main` · lines 846–847 · @084cbdd

```python
    end = min(args.steps, args.stop_at) if args.stop_at else args.steps
    for step in range(start_step, end):
```

`--steps` still sets the schedule. The book example runs `anulm/training/train.py`
unchanged on a byte corpus: A straight through 60 steps; B stopped at 30 and
resumed; C as B, with its resume point cut in half first.

```python
def train(folder: Path, *extra: str) -> str:
    r = subprocess.run([sys.executable, str(ROOT / "train.py"), *CMD, "--data", str(folder.parent / "corpus.txt"),
                        "--out", str(folder / "ckpt.pt"), *extra],
                       cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout
```

```python
    log_a = train(A)
    log_b1 = train(B, "--stop-at", "30")
    log_b2 = train(B, "--resume")
    log_c1 = train(C, "--stop-at", "30")
    last = C / "ckpt.pt.last"
    last.write_bytes(last.read_bytes()[: last.stat().st_size // 2])      # a resume point cut in half
    log_c2 = train(C, "--resume")
```

```
Part 1. train.py, 60 steps, stopped at step 30 and resumed
  B> phase done at step 30 of 60 in 5s | best val 1.8364 | resume point <tmp>\B\ckpt.pt.last
  B> resuming from <tmp>\B\ckpt.pt.last at step 30 (best val so far 1.8364, epoch 0.26)
  C> WARNING: <tmp>\C\ckpt.pt.last is unreadable (RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory).
  C> WARNING: falling back to <tmp>\C\ckpt.pt at step 30 WITHOUT optimizer state or sampler position.
  A and B agree on model   : True
  A and B agree on opt     : True
  A and B agree on sampler : True
  A and B agree on step    : True
  A and B agree on best_val: True
  C, after the fallback, agrees with A on model: False
  logged loss, steps 30-59 (the resumed half): A 1.5110  B 1.5110  C 1.6710
  step 30: A 1.7326  B 1.7326  C 2.1005
  final val loss: A 1.2510  C 1.2905
```

*verified* (CPU, default flags, deterministic). B ends bit for bit equal to A.
That holds for this setting; on a GPU, non-deterministic kernels make even two
uninterrupted runs differ. C continues with fresh moments and a sampler that
restarts its epoch.

![Figure 25.3 · Exact resume, and the fallback's dent](../figures/fig25-3-curves.svg)

*Figure 25.3 · Exact resume, and the fallback's dent.* Logged training loss
per step: the loss of each step's batch *before* its update. *verified*: from
the same run (`--save` writes `book/measurements/part_v_b_resume_curves.json`).

Read C carefully. Its best-val file was written at step 29, at the same
evaluation as `.last`, so C starts step 30 with A's exact weights. Its logged
2.1005 against A's 1.7326 is a different *batch*: the restarted sampler serves
step 0's windows again (the loss is computed before the update, lines 694–702).
The fresh moments first act on step 31. The means over steps 30–59 also
compare different batches; only the final val loss, 1.2905 against 1.2510, is
read on the same windows. That is the "visible dent" the fallback's comment
promises, and still better than starting over.

`IN THE REPO`

## 25.6  Extending a finished run

Resuming a cosine run with a larger `--steps` lands mid-cosine, at a much
higher rate; `--rewarmup N` ramps back to it:

> **In the repo** · `anulm/training/train.py` · `lr_at` · lines 398–403 · @084cbdd

```python
    # Extending a finished run (--resume with a larger --steps) lands mid-cosine,
    # several times the lr the weights last saw. --rewarmup ramps back up to
    # the schedule over N steps instead of jumping.
    rewarmup = getattr(args, "rewarmup", 0)
    if rewarmup and start_step and step < start_step + rewarmup:
        lr *= (step - start_step + 1) / rewarmup
```

For N steps after `start_step` the rate is scaled by (step − `start_step` +
1)/N. It applies to the cosine schedule only: `lr_at` returns earlier, before
these lines, during the initial warmup and on every step of `--schedule wsd`
(lines 384–395). A WSD run needs none in its flat phase.

> **Details.** The coder run's owner was a Windows scheduled task, firing
> every 30 minutes and resuming from `.last` if no trainer was alive. Sharing
> the interactive console, it died to Ctrl+C at steps 10,000 and 105,000, until
> the trainer was launched with its own console (§25).

`IN THE REPO` · `IMPLEMENTATION-DEPENDENT`

## 25.7  Loading a checkpoint anywhere

Scripts that only *read* a checkpoint call `load_checkpoint`, which accepts a
`.pt`, a folder written by `anulm/hf/export_hf.py`, or a Hub id:

> **In the repo** · `anulm/model.py` · `load_checkpoint` · lines 1205–1238 · @084cbdd

```python
def load_checkpoint(path, map_location="cpu", base: bool = True) -> dict:
    # ...
    p = _P(path)
    if not p.exists() and re.fullmatch(r"[\w.-]+/[\w.-]+", str(path).replace("\\", "/")):
        from huggingface_hub import snapshot_download
        p = _P(snapshot_download(str(path).replace("\\", "/")))
    if not p.is_dir():
        return torch.load(str(p), map_location=map_location, weights_only=False)
    # ...
    if (p / "model.safetensors").exists():
        ck["model"] = load_file(str(p / "model.safetensors"), device=dev)
```

A path that does not exist and looks like `owner/name` goes to the Hub, so a
mistyped local name is downloaded, or fails there. A file is a `.pt`, loaded
with `weights_only=False` because the configuration is a pickled `AnuLMConfig`.
In a folder, the elided lines start `ck = {}`, rebuild the configuration from
`config.json` keeping only known fields (an unknown one is dropped silently),
then load `model.safetensors`. A folder holds no optimizer or sampler state, so
it can be sampled from or fine-tuned, not resumed. Five evaluation scripts once crashed on exported
folders by calling `torch.load` (§27); `an_exported_folder_loads_everywhere_a_pt_does`
(line 914) round-trips a folder and scans eleven scripts for `torch.load(ckpt`.

`IN THE REPO`

> **Build it.** The lab book's Chapter 8 writes the same atomic save (§8.5),
> and its checks `resume_is_exact` and
> `a_crash_mid_save_keeps_the_old_checkpoint` catch the two bugs this chapter
> describes (§8.6).

## 25.8  Common mistakes, and how to catch them

| Mistake | Catch it |
|---|---|
| saving weights only, then resuming | compare a resumed run with an uninterrupted one, as in 25.5 |
| writing the checkpoint in place | kill a save halfway (25.2) and load what is left |
| loading a resume point onto the GPU and keeping it | `map_location="cpu"`, then `del` it; watch tokens per second after a resume |
| a restart loop that dies on the same file | log each restart's exit; the coder run looped for three hours |
| saving `parameters()` instead of `state_dict()` | buffers such as `expert_bias` vanish; compare a resumed run with an uninterrupted one (25.5). `expert_bias_is_not_trainable` checks the model's state dict, not what a writer saves |
| trusting `atomic_save` and the fallback to keep working | no test in `tests/test_model.py` covers either; the book example's part 2 and run C are the checks |

## 25.9  Exercises

1. The released model has 397,659,648 parameters (Chapter 20). How many
   bytes do AdamW's two fp32 moments add?
2. Run C resumed from the best-val file at step 30. Which training windows does
   it see again, and why?
3. Why must `atomic_save` write `<path>.tmp` beside the target, rather than in
   a temporary directory elsewhere?
4. You resume a WSD run with `--steps` doubled. Do you need `--rewarmup`?

**Answers.** (1) 2 × 4 × 397,659,648 bytes = 3.18 GB, as measured. (2) Its sampler
starts epoch 0's permutation from the beginning, so the windows of steps 0–29.
(3) `os.replace` is atomic only within a volume; another directory may be on
another one. (4) Not if the run is still in its flat phase, and `--rewarmup`
would do nothing under WSD anyway. If the run had already begun its decay,
doubling `--steps` moves the decay's start later, the rate jumps back to the
peak, and `--rewarmup` cannot soften it. *verified*: (1) is printed by `ch25_measure_resume_point.py`.

---

## Key points

- `anulm/training/train.py` writes two files: the best-val weights for sampling, and a resume
  point with weights, buffers, optimizer moments, step, sampler and `eval_spec`.
- On the released run the resume point is 4.86 GB, three times the 1.59 GB
  model, because AdamW keeps two moments per weight.
- `atomic_save` writes beside the target and renames, so a process killed
  mid-write leaves the previous checkpoint intact (no `fsync`: not a guard
  against power loss).
- On the CPU with default flags, a resumed run matches an uninterrupted one bit
  for bit; one without moments or sampler position re-reads data and ends worse.
- The resume path loads to the CPU, falls back loudly to the best-val file,
  checks the optimizer count and warns when the evaluation method changed.
- `load_checkpoint` reads a `.pt`, an exported folder or a Hub id into one
  dictionary, and a test keeps every script using it.

> **In one breath.** Save the newest step with everything that decides the
> next one, write it beside the old file and rename, keep a smaller best-val
> file as a second copy, and resume on the CPU with a loud fallback, so a kill
> costs a dent rather than the run.

## References

This chapter cites no external sources. Quality claims are measured here or in
`docs/RESULTS.md` §8, §25 and §27.

Repository evidence, all at @084cbdd:
- `anulm/training/train.py`: `atomic_save` 32–46, `WindowSampler.state` and `load_state`
  152–160, `lr_at` 379–404, `main` 790–837 (resume), 846–847 (`--stop-at`),
  888–902 (saves).
- `anulm/model.py`: `load_checkpoint` 1205–1254.
- `tests/test_model.py`: `checkpoint_roundtrip_is_exact` 352–369,
  `window_sampler_resumes_exactly` 812–822, `expert_bias_is_not_trainable`
  310–317, `an_exported_folder_loads_everywhere_a_pt_does` 914–967.
- `anulm/training/finetune.py` line 52 imports `atomic_save`.
- Measurements: `book/measurements/part_v_b_resume_point.json` from
  `code-examples/ch25_measure_resume_point.py --save`, and
  `book/measurements/part_v_b_resume_curves.json` from the book example
  `code-examples/ch25_resume.py --save`.
