"""
One model for every task: a single decoder trained on chat, reading, speech
and images at once.

    python speech_vocab.py --ckpt ckpt_chat_v3.pt --out ckpt_chat_mm.pt
    python omni_train.py --ckpt ckpt_chat_mm.pt --out ckpt_omni.pt --steps 90000 \
        --task chat=text:data/chat_train.jsonl --held chat=data/chat_heldout.jsonl \
        --task read=text:data/rc_train.jsonl::4 --held read=data/rc_heldout.jsonl \
        --task asr_en=asr:data/speech_ls100 \
        --task asr_hi=asr:data/hindi_mix5/train \
        --task caption=image:data/caption_train.jsonl \
        --task vqa=image:data/vqa_train.jsonl \
        --audio-proj-from ckpt_asr_hi_v3.pt --vision-proj-from ckpt_caption_v3.pt

Until now every skill was its own full copy of the backbone (RESULTS 39):
chat, reader, two speech models, captions and image questions, five 456M
decoders that cannot share a GPU and cannot be released as one thing. Here
they are one. Each step draws one task, in proportion to its weight (default:
its size), and takes a batch from it; every task's batch is the same thing by
the time it reaches the model -- (embeds, targets), loss on the answer only --
so one optimizer and one backbone serve all of them. The pieces that differ
per input are small and live beside the backbone in the same checkpoint: the
Whisper projector (`proj`, as speech_encoder.py writes it) and the SigLIP
projector (`vision_proj`). The encoders themselves stay frozen and outside.

**Task spec.** `name=kind:path[:weight[:batch]]`, kind one of
  text   a jsonl of question/answer/lang rows, formatted with make_qa.PROMPTS
         exactly as finetune.py does (chat, and reading as answer-from-passages)
  asr    a speech manifest prefix (rows with `audio` and `text`), Whisper states
  image  an image manifest (captions; VQA rows add `question`), SigLIP patches,
         cached once per image in `<manifest>.feats.f16`

**Held out.** The first 2% of each task, as the single-task trainers do, so
the numbers line up with theirs; at most `--eval-n` of them are scored. The
best checkpoint is the one with the lowest mean of each task's held-out loss
divided by its value at step 0, so a task with a large loss (captions) cannot
drown out one with a small one (speech).

**Licences.** One set of weights inherits every task's data terms. Check
docs/DATASETS.md before adding a task to a model meant for commercial use.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

KINDS = ("text", "asr", "image")
DEFAULT_BATCH = {"text": 8, "asr": 4, "image": 8}


class Task:
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


def load_rows(task: Task, tok, max_len: int) -> list[dict]:
    if task.kind == "text":
        from finetune import _pair_ids
        rows = []
        for line in Path(task.path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            p, a = _pair_ids(tok, r)
            if len(p) + len(a) <= max_len:
                rows.append({"p": p, "a": a})
        return rows
    if task.kind == "asr":
        from speech_encoder import load_manifest
        return load_manifest(task.path)
    from vision_encoder import load_manifest
    return load_manifest(task.path)


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


class Omni:
    """The backbone, both projectors and both frozen encoders, and a batch
    iterator per task."""

    def __init__(self, args):
        from hf_tok import load_tokenizer
        from model import AnuLM, fit_memory, load_checkpoint
        from speech_vocab import vocab_of

        self.dev = dev = args.device
        ck = load_checkpoint(args.ckpt, dev)
        self.ck = ck
        self.cfg = replace(ck["cfg"], moe_impl="grouped" if dev.startswith("cuda") else "sparse")
        self.vocab = vocab_of(ck)
        self.tok = load_tokenizer(self.cfg.tokenizer_path)
        self.model = AnuLM(self.cfg).to(dev)
        self.model.load_state_dict(ck["model"])
        fit_memory(self.model)
        if args.grad_ckpt:          # reading rows run to 1,024 tokens; chat v3 needed this at batch 4
            self.model.enable_gradient_checkpointing()
        self.args = args
        self.tasks = [Task(s) for s in args.task]
        kinds = {t.kind for t in self.tasks}
        self.sm = self.vl = self.enc = self.tower = None
        if "asr" in kinds:
            from audio_codec import WhisperEncoder
            from speech_encoder import ContinuousSpeech
            self.enc = WhisperEncoder(args.whisper, dev)
            self.sm = ContinuousSpeech(self.model, self.vocab, self.enc.dim, args.stack).to(dev)
            self._warm(self.sm.proj, args.audio_proj_from, "proj", "audio")
        if "image" in kinds:
            from vision_encoder import VisionLanguage, VisionTower
            self.tower = VisionTower(args.tower, dev)
            self.vl = VisionLanguage(self.model, self.vocab, self.tower.dim, args.pool).to(dev)
            self._warm(self.vl.proj, args.vision_proj_from, "proj", "vision")
        self.caches = {}
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

    def _warm(self, proj, path, key, what):
        """A projector trained by the single-task trainer already knows how to
        read its encoder; starting from it saves the first thousands of steps
        (speech_encoder.py's English run spent 1,500 before carrying audio)."""
        if not path:
            print(f"{what} projector: fresh")
            return
        from model import load_checkpoint
        src = load_checkpoint(path, "cpu")
        proj.load_state_dict(src[key])
        print(f"{what} projector warm-started from {path}")

    def batches(self, t: Task, rows, batch_size, shuffle=True, seed=0):
        if t.kind == "text":
            return text_batches(rows, self.model, self.dev, batch_size, shuffle, seed)
        if t.kind == "asr":
            from speech_encoder import batches
            return batches(rows, self.enc, self.tok, self.sm, self.dev, batch_size, shuffle, seed)
        from vision_encoder import batches
        return batches(rows, self.tower, self.tok, self.vl, self.dev, batch_size, shuffle, seed,
                       cache=self.caches[t.name])

    def next_batch(self, t: Task):
        while True:
            if t._it is None:
                t._it = self.batches(t, t.rows, t.batch, seed=self.args.seed + 1000 * t.epoch)
            try:
                return next(t._it)
            except StopIteration:
                t._it, t.epoch = None, t.epoch + 1

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


def train(args) -> None:
    from train import atomic_save

    om = Omni(args)
    model, dev = om.model, om.dev
    groups = [{"params": model.parameters(), "lr": args.lr}]
    for m in (om.sm, om.vl):
        if m is not None:
            groups.append({"params": m.proj.parameters(), "lr": args.proj_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.1, betas=(0.9, 0.95), fused=dev.startswith("cuda"))
    base_lrs = [g["lr"] for g in opt.param_groups]
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if dev.startswith("cuda") else torch.autocast("cpu", enabled=False))
    params = [p for g in opt.param_groups for p in g["params"]]

    curve = Path(args.out).with_name(Path(args.out).stem + "_curve.csv")
    names = [t.name for t in om.tasks]
    if not curve.exists():
        curve.write_text("step,score," + ",".join(names) + "\n", encoding="utf-8")

    def evaluate():
        return {t.name: round(om.held_loss(t, autocast), 4) for t in om.tasks}

    # `<out>.last` carries the optimizer and the step, written at every eval,
    # so a reboot ten hours in costs at most one eval interval, not the run.
    last = Path(str(args.out) + ".last")
    t0 = time.time()
    start, best, seen = 1, float("inf"), {t.name: 0 for t in om.tasks}
    if args.resume and last.exists():
        st = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        if om.sm is not None:
            om.sm.proj.load_state_dict(st["proj"])
        if om.vl is not None:
            om.vl.proj.load_state_dict(st["vision_proj"])
        opt.load_state_dict(st["opt"])
        first, best, seen = st["first"], st["best"], st["seen"]
        start = st["step"] + 1
        for t in om.tasks:      # a fresh shuffle per task: the exact batch order is not worth saving
            t.epoch = 100 + st["step"]
        print(f"resumed from {last} at step {st['step']} (best score {best:.4f})", flush=True)
    else:
        first = evaluate()
        with open(curve, "a", encoding="utf-8") as f:
            f.write("0,1.0," + ",".join(str(first[n]) for n in names) + "\n")
    print("step  start | held-out at step 0: " + "  ".join(f"{k} {v:.4f}" for k, v in first.items()), flush=True)
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
        seen[t.name] += 1
        run_loss[t.name].append(loss.item())
        if step % args.log_every == 0:
            msg = "  ".join(f"{k} {np.mean(v):.3f}" for k, v in run_loss.items() if v)
            print(f"step {step:6d} | {msg} | imbalance {imbalance:4.2f}x | "
                  f"lr x{frac:.2f} | {time.time()-t0:6.0f}s", flush=True)
            run_loss = {k: [] for k in run_loss}
        if step % args.eval_every == 0 or step == args.steps:
            losses = evaluate()
            score = float(np.mean([losses[k] / first[k] for k in losses]))
            flag = ""
            if score < best:
                best = score
                atomic_save(om.state(step, score, losses), args.out)
                flag = "  <- saved"
            with open(curve, "a", encoding="utf-8") as f:
                f.write(f"{step},{score:.4f}," + ",".join(str(losses[n]) for n in names) + "\n")
            print(f"  eval @ {step:6d} | score {score:.4f}{flag} | "
                  + "  ".join(f"{k} {v:.4f}" for k, v in losses.items())
                  + f" | batches seen {seen}", flush=True)
            if step < args.steps:
                st = om.state(step, score, losses)
                st.update(opt=opt.state_dict(), first=first, best=best, seen=seen)
                atomic_save(st, last)
    if last.exists():
        last.unlink()            # finished: the optimizer state is 3.7 GB nobody needs
    print(f"\ndone in {time.time()-t0:.0f}s | best score {best:.4f} | {args.out}")


def main() -> None:
    from speech_encoder import STACK
    from vision_encoder import DEFAULT_TOWER, POOL

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="a checkpoint grown by speech_vocab.py (the chat model)")
    p.add_argument("--out", required=True)
    p.add_argument("--task", action="append", required=True, help="name=kind:path[:weight[:batch]]")
    p.add_argument("--held", action="append",
                   help="name=path: a task's own held-out file instead of its first 2%% "
                        "(chat's first rows are the hand-written identity pairs)")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--proj-lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-n", type=int, default=300, help="held-out examples scored per task")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--audio-proj-from", help="a speech_encoder.py checkpoint whose projector to start from")
    p.add_argument("--vision-proj-from", help="a vision_encoder.py checkpoint whose projector to start from")
    p.add_argument("--whisper", default="small")
    p.add_argument("--stack", type=int, default=STACK)
    p.add_argument("--tower", default=DEFAULT_TOWER)
    p.add_argument("--pool", type=int, default=POOL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from <out>.last if it exists")
    p.add_argument("--grad-ckpt", action="store_true", help="recompute activations: less memory, ~30%% slower")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(p.parse_args())


if __name__ == "__main__":
    main()
