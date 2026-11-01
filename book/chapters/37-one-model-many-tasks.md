---
title: "37 · One model, many tasks"
part: "VIII · Beyond text"
accent: plum
pinned: "@773b6a8"
---

# 37  One model, many tasks

Before `omni_train.py`, every skill was its own copy of the backbone.
`omni_train.py` trains one decoder on chat, reading, intents, two speech tasks,
captions and image questions at once. It can, because Chapters 35 and 36
reduced every task to one pair, (embeds, targets), with −1 wherever the loss
must not look. This chapter reads the trainer, then the first full run, whose
score improved at every evaluation while the skill it was meant to keep got
worse.

`omni_train.py` has uncommitted edits in the working tree. Every line below is
quoted at @773b6a8, and the book example loads the file from that commit.

Prerequisites: Chapters 30 (instruction tuning), 35 and 36.

---

## 37.1  Every task becomes (embeds, targets)

Speech and image batches come from `speech_encoder.batches` and
`vision_encoder.batches` (Chapter 36). Text needs its own builder, because a
text row has no projector but must arrive in the same form:

> **In the repo** · `omni_train.py` · `text_batches` · lines 101–124 · @773b6a8

```python
def pad(built, device):
    S = max(e.shape[0] for e, _ in built)
    eb = torch.zeros(len(built), S, built[0][0].shape[1], device=device, dtype=built[0][0].dtype)
    tb = torch.full((len(built), S), -1, dtype=torch.long, device=device)
    for k, (e, t) in enumerate(built):
        eb[k, : e.shape[0]] = e
        tb[k, : t.shape[0]] = t
    return eb, tb


def text_batches(rows, model, device, batch_size, shuffle=True, seed=0):
    order = np.arange(len(rows))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    emb = model.embed_tokens
    for i in range(0, len(order), batch_size):
        built = []
        for j in order[i: i + batch_size]:
            p, a = rows[j]["p"], rows[j]["a"]
            ids = torch.tensor(p + a, device=device)
            t = torch.full((len(ids),), -1, dtype=torch.long, device=device)
            t[len(p) - 1: len(p) - 1 + len(a)] = ids[len(p):]
            built.append((emb(ids), t))
        yield pad(built, device)
```

Each row is a prompt `p` and an answer `a`, tokenised with the chat template
(`load_rows`, lines 82–93). The ids go through `embed_tokens`, (len, D), and
the targets are −1 except one position before each answer token (Chapter 35).
`pad` right-pads to the longest example, zeros and −1; causal attention keeps
padding from touching earlier positions. Unlike `finetune.py`, nothing is
packed.

The book example checks that this path is the plain id path in disguise:

```
2. text_batches: embeds (2, 6, 32), targets [[-1, -1, 9, 10, 1, -1], [-1, 11, 1, -1, -1, -1]]
   loss through embeds= 4.138337, through ids 4.138337
```

*verified*: `ch37_task_mix.py` on `tiny()`.

`IN THE REPO`

## 37.2  Describing tasks, and holding some back

A task is one command-line string, `name=kind:path[:weight[:batch]]`:

> **In the repo** · `omni_train.py` · `Task.__init__` · lines 66–79 · @773b6a8

```python
    def __init__(self, spec: str):
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        # a Windows drive letter ("C:\...") splits too; glue it back on
        if len(parts) > 2 and len(parts[1]) == 1 and parts[1].isalpha() and parts[2].startswith(("\\", "/")):
            parts[1:3] = [parts[1] + ":" + parts[2]]
        self.name, self.kind, self.path = name, parts[0], parts[1]
        assert self.kind in KINDS, f"{spec}: kind must be one of {KINDS}"
        self.weight = float(parts[2]) if len(parts) > 2 and parts[2] else None
        self.batch = int(parts[3]) if len(parts) > 3 and parts[3] else DEFAULT_BATCH[self.kind]
        self.rows: list[dict] = []
        self.held: list[dict] = []
        self.epoch = 0
        self._it = None
```

The kind is `text`, `asr` or `image` (line 61). A missing batch size falls back
to `DEFAULT_BATCH`, 8 for text and images and 4 for speech (line 62). The
drive-letter repair matters on Windows, where `C:\data` would otherwise split
into a path `C` and a weight `\data`:

```
   asr_hi=asr:C:\data\hindi:0.32            -> name asr_hi, kind asr, path C:\data\hindi, weight 0.32, batch 4
   vqa=image:data/vqa.jsonl                 -> name vqa, kind image, path data/vqa.jsonl, weight None, batch 8
```

`Omni.__init__` then loads each task and splits off its held-out set:

> **In the repo** · `omni_train.py` · `Omni.__init__` · lines 163–178 · @773b6a8

```python
        held = dict(h.split("=", 1) for h in (args.held or []))
        for t in self.tasks:
            rows = load_rows(t, self.tok, self.cfg.block_size)
            if t.name in held:
                t.rows, t.held = rows, load_rows(Task(f"{t.name}={t.kind}:{held[t.name]}"), self.tok,
                                                  self.cfg.block_size)
            else:
                cut = max(1, int(len(rows) * 0.02))
                t.held, t.rows = rows[:cut], rows[cut:]
            if t.kind == "image":
                from vision_encoder import build_cache
                self.caches[t.name] = build_cache(rows, self.tower, Path(t.path).with_suffix(".feats.f16"))
            if t.weight is None:
                t.weight = float(len(t.rows))
            print(f"task {t.name:8s} {t.kind:5s} {len(t.rows):7d} train {len(t.held):5d} held  "
                  f"batch {t.batch}  weight {t.weight:g}  ({t.path})", flush=True)
```

A task holds out its first 2%, as the single-task trainers do, unless `--held`
names a separate file. Image tasks get Chapter 36's feature cache. A task with
no weight is weighted by its size. The released recipe sets the weights
explicitly instead, in `experiments/omni_one.py` (lines 61–69):

| task | kind | weight | batch |
|---|---|---|---|
| chat | text | 0.07 | 8 |
| read | text | 0.14 | 4 |
| intent | text | 0.06 | 8 |
| asr_en | asr | 0.12 | 4 |
| asr_hi | asr | 0.32 | 4 |
| caption | image | 0.17 | 8 |
| vqa | image | 0.12 | 8 |

*verified*: printed by `ch37_task_mix.py` from the file at @773b6a8.

> **Details.** Those weights share out *steps*, not examples: a speech batch
> holds 4 clips, a chat batch 8 rows. What the data itself does to a skill is
> separate: adding 28.5k mostly synthetic Hindi chat pairs (machine-translated,
> or answered by Llama-2-70B) raised the
> single-task chat model's held-out loss from 3.1445 to 3.5662
> (`docs/RESULTS.md` §40, *measured*).

`IN THE REPO`

## 37.3  One step: draw a task, take its batch

> **In the repo** · `omni_train.py` · `train` · lines 282–305 · @773b6a8

```python
    w = np.array([t.weight for t in om.tasks], dtype=float)
    w /= w.sum()
    rng = np.random.default_rng(args.seed + start)
    run_loss = {t.name: [] for t in om.tasks}
    model.train()
    for step in range(start, args.steps + 1):
        # linear warmup, then cosine to 10%: the same shape finetune.py uses
        frac = min(1.0, step / max(1, args.warmup))
        if step > args.warmup:
            prog = (step - args.warmup) / max(1, args.steps - args.warmup)
            frac = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
        for g, lr in zip(opt.param_groups, base_lrs):
            g["lr"] = lr * frac
        t = om.tasks[rng.choice(len(om.tasks), p=w)]
        eb, tb = om.next_batch(t)
        with autocast:
            _, loss = model(None, tb, embeds=eb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        imbalance = model.update_expert_biases()
```

In order:

1. **Weights to probabilities.** `w /= w.sum()`.
2. **Schedule.** Linear warmup, then a cosine down to 10%, applied as one factor
   to every parameter group (Chapter 7).
3. **Draw.** One task per step, with probability w. `next_batch` restarts a
   task's iterator with a new shuffle when it runs out (lines 202–209).
4. **One batch, one step.** Forward under bf16 autocast with `embeds=`,
   backward, clip at 1.0, step, clear the gradients, and nudge the router
   biases (Chapter 18). Each optimizer step sees exactly one task's batch.

The optimizer has the backbone in one group and each projector in another,
at `--lr` 5e-5 and `--proj-lr` 1e-3 (lines 241–245, 345–346), as in Chapter 36.

The draw is seeded by `args.seed + start`, so it can be replayed. The book
example replays it and compares with the real run's log:

```python
def replay_draw(names, w, resumed_at, last_step) -> dict:
    """train()'s draw: rng = default_rng(args.seed + start), one task per step; seed 0."""
    seen = {n: 0 for n in names}
    starts = [1] + [r + 1 for r in resumed_at] + [last_step + 1]
    for a, b in zip(starts[:-1], starts[1:]):
        rng = np.random.default_rng(0 + a)
        for _ in range(a, b):
            seen[names[rng.choice(len(names), p=w)]] += 1
    return seen
```

```
   replayed draw to step 100000 (resumed at [90000]): {'chat': 7017, 'read': 13921, 'intent': 6013, 'asr_en': 12131, 'asr_hi': 31708, 'caption': 17046, 'vqa': 12164}
   the log's batches seen:                         {'chat': 7017, 'read': 13921, 'intent': 6013, 'asr_en': 12131, 'asr_hi': 31708, 'caption': 17046, 'vqa': 12164}
```

*verified*: `ch37_task_mix.py` against `logs/omni_one/6_train.log`. The run was
interrupted and resumed at step 90,000, and a resume reseeds the draw with the
new `start` (line 284). The match confirms that this run executed the code at
the pin: one batch per optimizer step. Chat got 7,017 of 100,000 steps.

![Figure 37.1 · One training step](../figures/fig37-1-one-step.svg)

*Figure 37.1 · One training step.* One task is drawn, its batch built, and the
one backbone stepped. *verified*: the shares are the pinned `TASKS`.

`IN THE REPO`

## 37.4  One number for seven tasks

The trainer keeps the best checkpoint, so it needs one number. Each task's
held-out loss is divided by its value before training, and the ratios are
averaged:

> **In the repo** · `omni_train.py` · `train` · lines 311–318 · @773b6a8

```python
        if step % args.eval_every == 0 or step == args.steps:
            losses = evaluate()
            score = float(np.mean([losses[k] / first[k] for k in losses]))
            flag = ""
            if score < best:
                best = score
                atomic_save(om.state(step, score, losses), args.out)
                flag = "  <- saved"
```

`first` holds the held-out losses at step 0 (line 278). Dividing by them puts
every task on the same footing: a caption loss of 5.5 and a speech loss of 6.6
both start at 1.0. Lower is better, and a checkpoint is saved whenever the
mean improves.

The first full run, from the pinned recipe, recomputed from its log:

```
   step    score  recomputed      chat     read   intent   asr_en   asr_hi  caption      vqa
      0   1.0000      1.0000     1.000    1.000    1.000    1.000    1.000    1.000    1.000
    10000   0.6690      0.6690     1.699    1.028    0.075    0.239    0.423    0.524    0.696
    50000   0.6259      0.6259     1.713    0.955    0.034    0.194    0.362    0.481    0.641
    90000   0.5902      0.5902     1.651    0.898    0.027    0.163    0.336    0.442    0.615
   100000   0.5873      0.5873     1.651    0.878    0.033    0.159    0.331    0.444    0.615
   chat: 3.2651 at step 0 -> 5.391 at step 100000 (x1.651) while the score fell to 0.5873
```

*measured*: `logs/omni_one/6_train.log`, every row recomputed by
`ch37_task_mix.py` (rows for 20,000–80,000 omitted here; all are in
`book/measurements/part_viii_ix_omni_log.json`).

The score fell at every evaluation, so every evaluation saved a checkpoint.
Six tasks improved, the new skills enormously: intent classification went to
3% of its starting loss. Chat, the skill the model started with, got about
70% worse within the first 10,000 steps (ratio 1.699) and ended 65% worse. A
mean of seven ratios cannot show that: six large gains outweigh one large loss.

The run started from `ckpt_chat_mm.pt`: the chat model `ckpt_chat_v3.pt` grown
by `speech_vocab.py` (`experiments/omni_one.py` line 314;
`logs/omni_one/3_grow.log`). So step 0 is already the grown model, and its
chat loss already includes the full-softmax cost of growing, about 0.065 nats
on base v2's text (Chapter 35).

![Figure 37.2 · The score improved; chat did not](../figures/fig37-2-task-ratios.svg)

*Figure 37.2 · The score improved; chat did not.* Held-out loss ÷ step-0 loss
per task, and their mean. *measured*: `book/measurements/part_viii_ix_omni_log.json`.

`IN THE REPO` · `MEASURED`

## 37.5  One checkpoint, every skill

What the trainer saves is the backbone plus the small parts that differ per
input:

> **In the repo** · `omni_train.py` · `Omni.state` · lines 223–233 · @773b6a8

```python
    def state(self, step, score, losses) -> dict:
        a = self.args
        st = {"model": self.model.state_dict(), "cfg": replace(self.cfg, moe_impl=self.ck["cfg"].moe_impl),
              "speech_text_vocab": self.ck["speech_text_vocab"], "step": step, "val_loss": score,
              "task_losses": losses, "tasks": [t.name for t in self.tasks], "base_ckpt": a.ckpt,
              "qa_templates": self.ck.get("qa_templates")}
        if self.sm is not None:
            st.update(proj=self.sm.proj.state_dict(), whisper=a.whisper, stack=a.stack)
        if self.vl is not None:
            st.update(vision_proj=self.vl.proj.state_dict(), tower=a.tower, pool=a.pool)
        return st
```

The audio projector is saved as `proj`, the key `speech_encoder.py` uses, and
the image projector as `vision_proj`. The encoders are not saved, only their
names (`whisper`, `tower`), because they are frozen and downloaded on use.
`val_loss` holds the score, not a loss. `omni.py` then builds every skill on
one loaded copy:

> **In the repo** · `omni.py` · `Omni.ear` · lines 47–57 · @773b6a8

```python
    def ear(self):
        """voice.OwnASR's interface over the shared decoder."""
        if self._ear is None:
            from audio_codec import WhisperEncoder
            # ...
            enc = WhisperEncoder(self.meta.get("whisper", "small"), self.device)
            sm = ContinuousSpeech(self.model, vocab_of(self.meta), enc.dim, self.meta.get("stack", 4)).to(self.device)
            sm.proj.load_state_dict(self.proj)
```

`self.model` is the chat engine's model (line 39), and the serving engine has
already set its `text_vocab` (Chapter 35). The ear wraps that same module with
the saved projector; `eye()` does the same with `vision_proj`. One decoder in
memory serves chat, reading, intents, speech and images.

`IN THE REPO`

## 37.6  A score picks checkpoints; gates decide releases

The pipeline that ran this training, `experiments/omni_one.py`, does not trust
the score with the release. Each task gets a gate: how much it may lose against
the separate model it would replace.

> **In the repo** · `experiments/omni_one.py` · `gates` · lines 230–238 · @773b6a8

```python
def gates(res) -> dict:
    hi = [res.get(k) for k in ("wer_hi_fleurs", "wer_hi_kathbath", "wer_hi_mucs")]
    if all(v is not None for v in hi):
        res["wer_hi_mean"] = round(sum(hi) / 3, 1)
    out = {}
    for k, (op, thr) in GATES.items():
        v = res.get(k)
        out[k] = v is not None and (v <= thr if op == "<=" else v >= thr)
    return out
```

A missing measurement fails its gate (`v is not None and ...`). Hindi speech
is gated on the mean of three test sets, because each set alone pulled a
different way (comment, lines 76–79). The pinned rule, applied by the book
example to the run's evaluation results:

```
5. release gates (experiments/omni_one.py GATES at the pin) on logs/omni_one/results.json
   wer_en                  12.2  <= 9.0    FAIL
   wer_hi_mean             46.2  <= 50.9   pass
   wer_hi_fleurs           48.8  <= 43.0   FAIL
   chat_heldout_loss     5.3181  <= 3.3    FAIL
   read_squad_en           24.6  >= 30.0   FAIL
   intent_banking77        90.0  >= 80.0   pass
   vqa_acc                 17.6  >= 10.0   pass
   all pass: False
```

*measured*: `logs/omni_one/results.json`, written by the pipeline's
evaluation step. Four gates failed, so the model was not published.

Two different chat numbers appear in this chapter. The trainer's held-out loss
(5.391 at step 100,000, 37.4) comes from `Omni.held_loss`:

> **In the repo** · `omni_train.py` · `Omni.held_loss` · lines 211–221 · @773b6a8

```python
    @torch.no_grad()
    def held_loss(self, t: Task, autocast) -> float:
        self.model.eval()
        tot = n = 0.0
        for eb, tb in self.batches(t, t.held[: self.args.eval_n], 1, shuffle=False):
            with autocast:
                _, loss = self.model(None, tb, embeds=eb)
            k = int((tb != -1).sum())
            tot, n = tot + loss.item() * k, n + k
        self.model.train()
        return tot / max(n, 1)
```

It scores embeddings under the trainer's autocast, token-weighted over at most
`--eval-n` examples. The gate's 5.3181 comes from a separate scorer:

> **In the repo** · `omni_eval.py` · `chat_loss` · lines 115–131 · @773b6a8

```python
def chat_loss(m, tok, path, device):
    from finetune import _pair_ids
    tot = n = 0.0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        p, a = _pair_ids(tok, r)
        ids = (p + a)[: m.cfg.block_size]
        x = torch.tensor([ids], device=device)
        t = torch.full_like(x, -1)
        t[0, len(p) - 1: len(ids) - 1] = x[0, len(p):]
        logits, _ = m(x, t)
        lp = F.log_softmax(logits.float()[0, :-1], -1)
        tgt = x[0, 1:]
        mask = t[0, :-1] != -1
        tot += float(-lp[mask].gather(1, tgt[mask][:, None]).sum())
        n += int(mask.sum())
    return round(tot / max(n, 1), 4)
```

It feeds ids, one example at a time, and takes its own log-softmax. Its
threshold, 3.3, was set from `ckpt_chat_v3.pt`'s
3.146 under that same scorer (comment, `omni_one.py` line 82), measured before
the vocabulary was grown. They measure the same held-out file in different
ways, so compare each only with its own kind. For
comparison, the separate English speech model scored 7.1% WER on the same
100-clip measure (`docs/RESULTS.md` §39). `docs/RESULTS.md` at @773b6a8
records no result for the joint model.

> **Details.** After the pin, short probes of 1,000 to 3,000 batches pointed
> to a likely cause, too few batches per optimizer step, and an option to sum
> several exists only as an uncommitted edit to `omni_train.py`; it is not
> cited here. By the lab book's account, no probe over the full mix brought
> chat back to its step-0 loss, and the combined fix was never tested as a
> set. The lab book's Chapter 14
> (`course/ch14-one-model`) tells the story.

> **Build it.** The lab book's Chapter 14 builds the task draw, the score and
> the gates on a small model, with checks that catch each silent bug.

`IN THE REPO` · `MEASURED`

## 37.7  Exercises

1. A task spec reads `cap=image:D:/imgs/train.jsonl::16`. What are its path,
   weight and batch?
2. Three tasks end at ratios 0.2, 0.3 and 1.6. What is the score, and is it
   better than a run that ends at 0.6, 0.6, 0.6?
3. Why must a resumed run not reuse the seed it started with?

**Answers.** 1. `D:/imgs/train.jsonl`; weight `None`, so its size; batch 16.
2. 0.7, which is worse than 0.6, although one task in the first run is 60%
worse than at step 0 and none in the second is. 3. It would replay
the same task sequence from step 1, not continue it; the run would see the
early order twice.

*verified*: exercise 1 by `omni_train.Task` at the pin; exercise 2 by arithmetic.

---

## Key points

- `omni_train.py` turns every task into (embeds, targets) with −1 off the
  answer, so one optimizer and one backbone serve text, speech and images.
- A task is `name=kind:path[:weight[:batch]]`; it holds out its first 2%
  unless given a held-out file, and image tasks reuse cached features.
- Each optimizer step draws one task by weight and takes one batch of it; the
  draw is seeded, and the book example replays it exactly from the run's log.
- The checkpoint score is the mean of held-out losses divided by their step-0
  values, so a task with a large loss cannot drown out one with a small loss.
- That mean improved at every evaluation of the first run while chat's
  held-out loss rose 65%: one regression hides behind six gains.
- Gates, one per task against the model it replaces, decided the release; four
  failed, so nothing was published.

> **In one breath.** One decoder learns seven tasks because each arrives as
> embeddings with answer-only targets; a seeded draw picks one task's batch
> per step, a mean of step-0-normalised losses picks checkpoints, and per-task
> gates, not that mean, decide whether the model ships.

## References

This chapter cites no external research. Its evidence is the repository, the
run's logs and `docs/RESULTS.md`.

Repository evidence, all at @773b6a8:
- `omni_train.py`: `KINDS`, `DEFAULT_BATCH` 61–62, `Task` 65–79, `load_rows`
  82–98, `pad` 101–108, `text_batches` 111–124, `Omni.__init__` 131–178,
  `Omni.next_batch` 202–209, `Omni.state` 223–233, `train` 236–330, `main`
  333–361.
- `omni.py`: `Omni` 33–86.
- `experiments/omni_one.py`: `TASKS` 61–69, `GATES` 74–86, `gates` 230–238.
- `omni_eval.py`: `chat_loss` 115–131.
- `docs/RESULTS.md` §39, §40.
- The run: `logs/omni_one/6_train.log` and `logs/omni_one/results.json`
  (outside the repository's history; numbers copied into
  `book/measurements/part_viii_ix_omni_log.json`).
- Book example: `book/code-examples/ch37_task_mix.py`.
