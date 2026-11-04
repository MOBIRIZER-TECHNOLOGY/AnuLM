"""
One model for every task: a single decoder trained on chat, reading, speech
and images at once.

    python -m anulm speech_vocab --ckpt ckpt_chat_v3.pt --out ckpt_chat_mm.pt
    python -m anulm omni_train --ckpt ckpt_chat_mm.pt --out ckpt_omni.pt --steps 90000 \
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

**Held out.** The first 2% of distinct input groups (normalized media paths
or text prompts); repeated captions/recordings stay in one partition. Explicit
partitions must be nonempty and disjoint. Copied media under different paths
still requires corpus deduplication. These splits differ from older row-based
results, which must be rebaselined; at most `--eval-n` examples are scored. The
best checkpoint is the one with the lowest mean of each task's held-out loss
divided by its value at step 0, so a task with a large loss (captions) cannot
drown out one with a small one (speech).

**Distillation.** `--distill chat` keeps a task where it started instead of
re-learning it: on that task's batches the answer-token loss blends the
cross-entropy with the KL divergence to a frozen copy of the starting
checkpoint (`--distill-alpha`, 1.0 = KL only). The 2026-11-03 run re-read the
22,873 chat pairs about seven more times after chat v3 had learned them, and
chat held-out loss rose 3.27 -> 3.58 while every other task improved; the
teacher's distribution on those prompts is where chat should stay.

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
        from anulm.training.finetune import _pair_ids
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
        from .speech_encoder import load_manifest, whisper_window_rows
        return whisper_window_rows(load_manifest(task.path), task.name)
    from .vision_encoder import load_manifest
    return load_manifest(task.path)


def pad(built, device):
    S = max(e.shape[0] for e, _ in built)
    eb = torch.zeros(len(built), S, built[0][0].shape[1], device=device, dtype=built[0][0].dtype)
    tb = torch.full((len(built), S), -1, dtype=torch.long, device=device)
    for k, (e, t) in enumerate(built):
        eb[k, : e.shape[0]] = e
        tb[k, : t.shape[0]] = t
    return eb, tb


def _row_group(row, kind):
    """Keep all captions of one image, recordings, or identical prompts together."""
    if kind == "image":
        return str(Path(row["image"]).resolve())
    if kind == "asr":
        return str(Path(row["audio"]).resolve())
    return tuple(row["p"])


def split_rows(rows, kind, held_fraction=0.02):
    """Deterministic group split; repeated views of an input cannot cross it.

    This groups by normalized media path or tokenized prompt. Corpus-level
    deduplication is still needed to detect copied media at different paths.
    """
    groups = list(dict.fromkeys(_row_group(row, kind) for row in rows))
    if len(groups) < 2:
        raise ValueError(f"{kind}: need at least two independent input groups for train/held-out splits.")
    cut = min(len(groups) - 1, max(1, int(len(groups) * held_fraction)))
    selected = set(groups[:cut])
    training = [row for row in rows if _row_group(row, kind) not in selected]
    held = [row for row in rows if _row_group(row, kind) in selected]
    return training, held


def check_split(training, held, kind, name):
    if not training or not held:
        raise ValueError(f"{name}: training and held-out partitions must both be nonempty.")
    overlap = {_row_group(row, kind) for row in training} & {_row_group(row, kind) for row in held}
    if overlap:
        raise ValueError(f"{name}: {len(overlap)} input groups overlap between training and held-out data.")



def disjoint_held(training, held, kind, name):
    """An explicit held-out file minus every input that training also sees.

    The release data repeats a few prompts across its train and held-out files
    ("What is the best way to cook a steak?" is in both chat files), and a
    held-out score over prompts seen in training is optimistic. Those rows are
    dropped from the held-out side, with a count, rather than refusing to train."""
    seen = {_row_group(row, kind) for row in training}
    kept = [row for row in held if _row_group(row, kind) not in seen]
    if len(kept) < len(held):
        print(f"{name}: dropped {len(held) - len(kept)} held-out rows whose input is also in training", flush=True)
    return kept

def text_batches(rows, model, device, batch_size, shuffle=True, seed=0, with_ids=False):
    """(embeds, targets) per batch; with_ids adds the padded token ids (0 in
    padding), which a distillation teacher embeds with its own table."""
    order = np.arange(len(rows))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    emb = model.embed_tokens
    for i in range(0, len(order), batch_size):
        built, all_ids = [], []
        for j in order[i: i + batch_size]:
            p, a = rows[j]["p"], rows[j]["a"]
            ids = torch.tensor(p + a, device=device)
            t = torch.full((len(ids),), -1, dtype=torch.long, device=device)
            t[len(p) - 1: len(p) - 1 + len(a)] = ids[len(p):]
            built.append((emb(ids), t))
            all_ids.append(ids)
        eb, tb = pad(built, device)
        if not with_ids:
            yield eb, tb
            continue
        ib = torch.zeros(tb.shape, dtype=torch.long, device=device)
        for k, ids in enumerate(all_ids):
            ib[k, : len(ids)] = ids
        yield eb, tb, ib


def _ce_kl_sums(s, t, y):
    import torch.nn.functional as F
    ls, lt = F.log_softmax(s.float(), -1), F.log_softmax(t.float(), -1)
    return F.nll_loss(ls, y, reduction="sum"), (lt.exp() * (lt - ls)).sum()


def distill_loss(logits, teacher_logits, targets, loss, alpha: float, chunk: int = 512):
    """The model's `loss` (answer cross-entropy + router regulariser) with
    `alpha` of the cross-entropy replaced by KL(teacher || student) on the same
    answer positions. Returns (loss, cross-entropy, KL), the last two detached.

    `logits` is the student's (B, S, V); `teacher_logits` is (N, V), the
    teacher's rows at the N answer positions only (targets != -1), in order.
    The router term is kept whole: only the language-modelling part moves
    toward the teacher. With alpha 0 this is `loss` unchanged.

    Float log-softmaxes of every answer row at the 61,445-id vocabulary did
    not fit beside training on the 16 GB card (out of memory at the first
    step), so the sums run `chunk` rows at a time under activation
    checkpointing: one chunk's float tensors exist at once. Same value."""
    m = targets != -1
    s, y = logits[m], targets[m]
    n = y.numel()
    ce_sum = kl_sum = 0.0
    for i in range(0, n, chunk):
        args = (s[i:i + chunk], teacher_logits[i:i + chunk], y[i:i + chunk])
        if s.requires_grad:
            c, k = torch.utils.checkpoint.checkpoint(_ce_kl_sums, *args, use_reentrant=False)
        else:
            c, k = _ce_kl_sums(*args)
        ce_sum, kl_sum = ce_sum + c, kl_sum + k
    ce, kl = ce_sum / n, kl_sum / n
    return loss + alpha * (kl - ce), ce.detach(), kl.detach()


def parse_limits(specs) -> dict[str, float]:
    """`--max-ratio chat=1.049` -> {"chat": 1.049}."""
    out = {}
    for s in specs or []:
        name, _, value = s.partition("=")
        if not name or not value:
            raise SystemExit(f"--max-ratio takes task=ratio, got {s!r}")
        out[name] = float(value)
    return out


def within_limits(losses: dict, first: dict, limits: dict) -> bool:
    """True if every limited task's held-out loss is at most `ratio` times its
    value at step 0."""
    return all(losses[k] / first[k] <= r for k, r in limits.items())


def better_checkpoint(score: float, ok: bool, best: float, best_ok: bool) -> bool:
    """Which evaluation to keep. The combined score alone kept the November
    run's step 9,375, whose chat had got 8.7% worse than at the start and
    failed its release gate, over earlier checkpoints that passed it. Here a
    checkpoint within --max-ratio always beats one outside it; among equals,
    the lower score wins (ties keep the first, as before)."""
    if ok != best_ok:
        return ok
    return score < best


def answer_logits(model, embeds, targets):
    """The model's logits at the answer positions only, (N, V): the final
    hidden state is caught on its way to the head, so the full (B, S, V)
    logits are never built."""
    caught = {}
    hook = model.norm.register_forward_hook(lambda _m, _i, out: caught.setdefault("h", out))
    try:
        model(None, None, embeds=embeds)            # no targets: the head runs on the last position only
    finally:
        hook.remove()
    return model._head(caught["h"][targets != -1])


def half_teacher(model, device):
    """A frozen teacher in bf16, as the demo keeps its models (Models._half):
    the routers stay fp32 (bf16 cannot hold their biases) and RoPE's inverse
    frequencies are rebuilt in fp32 (in bf16 the angles drift)."""
    from anulm.model import Rotary, Router
    routers = {m: {k: v.detach().clone() for k, v in m.state_dict().items()}
               for m in model.modules() if isinstance(m, Router)}
    model.to(torch.bfloat16)
    for m, state in routers.items():
        m.float()
        m.load_state_dict(state)
    for m in model.modules():
        if isinstance(m, Rotary):
            m.inv_freq = Rotary._build_inv_freq(m.cfg, m.dim).to(device)
            m._cached_len, m._cached_key = 0, None
    return model.eval().requires_grad_(False)


class Omni:
    """The backbone, both projectors and both frozen encoders, and a batch
    iterator per task."""

    def __init__(self, args):
        from anulm.hf_tok import load_tokenizer
        from anulm.model import AnuLM, fit_memory, load_checkpoint
        from .speech_vocab import vocab_of

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
        self.distill = set(getattr(args, "distill", None) or [])
        unknown = self.distill - {t.name for t in self.tasks if t.kind == "text"}
        if unknown:
            raise SystemExit(f"--distill names no text task: {', '.join(sorted(unknown))}")
        self.teacher = None
        if self.distill:            # the starting checkpoint, frozen: where the distilled tasks should stay
            self.teacher = AnuLM(self.cfg).to(dev)
            self.teacher.load_state_dict(ck["model"])
            self.teacher = half_teacher(self.teacher, dev) if dev.startswith("cuda") else \
                self.teacher.eval().requires_grad_(False)
        kinds = {t.kind for t in self.tasks}
        self.sm = self.vl = self.enc = self.tower = None
        if "asr" in kinds:
            from .audio_codec import WhisperEncoder
            from .speech_encoder import ContinuousSpeech
            self.enc = WhisperEncoder(args.whisper, dev)
            self.sm = ContinuousSpeech(self.model, self.vocab, self.enc.dim, args.stack).to(dev)
            self._warm(self.sm.proj, args.audio_proj_from, "proj", "audio")
        if "image" in kinds:
            from .vision_encoder import VisionLanguage, VisionTower
            self.tower = VisionTower(args.tower, dev)
            self.vl = VisionLanguage(self.model, self.vocab, self.tower.dim, args.pool).to(dev)
            self._warm(self.vl.proj, args.vision_proj_from, "proj", "vision")
        if args.proj_scale == "embed":
            for m, what in ((self.sm, "audio"), (self.vl, "vision")):
                if m is not None:
                    self._match_scale(m.proj, what)
        self.caches = {}
        held = dict(h.split("=", 1) for h in (args.held or []))
        for t in self.tasks:
            rows = load_rows(t, self.tok, self.cfg.block_size)
            if t.name in held:
                t.rows, t.held = rows, load_rows(Task(f"{t.name}={t.kind}:{held[t.name]}"), self.tok,
                                                  self.cfg.block_size)
                t.held = disjoint_held(t.rows, t.held, t.kind, t.name)
            else:
                t.rows, t.held = split_rows(rows, t.kind)
            check_split(t.rows, t.held, t.kind, t.name)
            self._validate_context(t)
            if t.kind == "image":
                from .vision_encoder import build_cache
                self.caches[t.name] = build_cache(t.rows + t.held, self.tower,
                                                  Path(t.path).with_suffix(".feats.f16"))
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
        from anulm.model import load_checkpoint
        src = load_checkpoint(path, "cpu")
        proj.load_state_dict(src[key])
        print(f"{what} projector warm-started from {path}")

    @torch.no_grad()
    def _match_scale(self, proj, what):
        """Shrink the projector's final LayerNorm so its vectors arrive at the
        size of a text token's embedding. A unit-gain LayerNorm emits norm
        sqrt(hidden) = 32, about 80x the 0.39 of the chat model's embeddings,
        and the shared blocks then retune to the loud inputs at the expense of
        text. For a normalised input z, E||g*z + b||^2 = sum(g^2) + sum(b^2),
        so one factor on g and b sets the expected norm; it stays learnable
        and lives in the checkpoint, so inference needs nothing new."""
        target = self.model.embed_tokens.weight[: self.vocab.text].float().norm(dim=-1).mean()
        ln = proj.net[-1]
        now = (ln.weight.float().pow(2).sum() + ln.bias.float().pow(2).sum()).sqrt()
        ln.weight.mul_(target / now)
        ln.bias.mul_(target / now)
        print(f"{what} projector output scaled {now:.2f} -> {target:.3f} (text embedding norm)")

    @torch.no_grad()
    def _validate_context(self, t: Task):
        """Fail at startup, not hours in, on an example longer than block_size.

        Running the frozen encoder over every clip and image here cost a full
        pass of Whisper or SigLIP on every start and resume. The media part is
        bounded instead: an image always yields the same number of patches, and
        a clip at most the encoder's 30 s window. Only rows whose text could
        overflow under that bound are built for real, which raises the
        descriptive error for the ones that do."""
        if t.kind == "text":
            return
        rows = t.rows + t.held
        if not rows:
            return
        if t.kind == "asr":
            from .audio_codec import ASR_RATE
            states = torch.from_numpy(self.enc.encode(np.zeros(30 * ASR_RATE, dtype=np.float32))).to(self.dev)
            media = self.sm.proj(states[None])[0].shape[0]

            def text(r):
                return len(self.tok.encode(" " + r["text"].strip())) + 1
        else:
            from .vision_encoder import qa_ids
            media = self.vl.proj(self.tower.encode(rows[0]["image"]).to(self.dev)).shape[0]

            def text(r):
                return sum(map(len, qa_ids(self.tok, r)))
        limit = self.model.cfg.block_size
        risky = [r for r in rows if 2 + media + text(r) > limit]
        for _ in self.batches(t, risky, t.batch, shuffle=False):
            pass

    def batches(self, t: Task, rows, batch_size, shuffle=True, seed=0):
        if t.kind == "text":
            return text_batches(rows, self.model, self.dev, batch_size, shuffle, seed,
                                with_ids=t.name in self.distill)
        if t.kind == "asr":
            from .speech_encoder import batches
            return batches(rows, self.enc, self.tok, self.sm, self.dev,
                           batch_size, shuffle, seed, task=t.name)
        from .vision_encoder import batches
        return batches(rows, self.tower, self.tok, self.vl, self.dev, batch_size, shuffle, seed,
                       cache=self.caches.get(t.name), task=t.name)

    def next_batch(self, t: Task):
        exhausted = 0
        while True:
            if t._it is None:
                t._it = self.batches(t, t.rows, t.batch, seed=self.args.seed + 1000 * t.epoch)
            try:
                return next(t._it)
            except StopIteration:
                t._it, t.epoch = None, t.epoch + 1
                exhausted += 1
                if exhausted >= 2:
                    raise ValueError(f"{t.name}: no usable training batches; check lengths and media.")

    @torch.no_grad()
    def held_loss(self, t: Task, autocast) -> float:
        self.model.eval()
        tot = n = 0.0
        for eb, tb, *_ in self.batches(t, t.held[: self.args.eval_n], 1, shuffle=False):
            with autocast:
                _, loss = self.model(None, tb, embeds=eb)
            k = int((tb != -1).sum())
            tot, n = tot + loss.item() * k, n + k
        self.model.train()
        if n <= 0 or not math.isfinite(tot):
            raise ValueError(f"{t.name}: held-out set has no finite graded tokens.")
        return tot / n

    def state(self, step, score, losses) -> dict:
        a = self.args
        qa_templates = self.ck.get("qa_templates")
        if qa_templates is None and any(t.kind == "text" for t in self.tasks):
            from anulm.data.make_qa import PROMPTS
            qa_templates = PROMPTS
        st = {"model": self.model.state_dict(), "cfg": replace(self.cfg, moe_impl=self.ck["cfg"].moe_impl),
              "speech_text_vocab": self.ck["speech_text_vocab"], "step": step, "val_loss": score,
              "task_losses": losses, "tasks": [t.name for t in self.tasks], "base_ckpt": a.ckpt,
              "qa_templates": qa_templates}
        if self.sm is not None:
            st.update(proj=self.sm.proj.state_dict(), whisper=a.whisper, stack=a.stack)
        if self.vl is not None:
            st.update(vision_proj=self.vl.proj.state_dict(), tower=a.tower, pool=a.pool)
        return st


def recipe_of(args, tasks) -> dict:
    """The settings a resumed run must share with the run that wrote `.last`:
    resuming under a different accumulation, schedule, scale or task mix would
    silently splice two recipes into one checkpoint."""
    return {"accum": args.accum, "proj_scale": args.proj_scale, "lr": args.lr, "proj_lr": args.proj_lr,
            "steps": args.steps, "warmup": args.warmup,
            "tasks": [[t.name, float(t.weight), int(t.batch)] for t in tasks],
            **({"distill": sorted(args.distill), "distill_alpha": args.distill_alpha}
               if getattr(args, "distill", None) else {}),
            **({"max_ratio": sorted(args.max_ratio)} if getattr(args, "max_ratio", None) else {})}


def train(args) -> None:
    from anulm.training.train import atomic_save

    # Fresh projectors (vision always, audio without --audio-proj-from) are
    # initialised from torch's global generator: seed it so a rerun, or an A/B
    # probe against it, starts from the same weights.
    torch.manual_seed(args.seed)
    om = Omni(args)
    recipe = recipe_of(args, om.tasks)
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
    limits = parse_limits(args.max_ratio)
    unknown = set(limits) - {t.name for t in om.tasks}
    if unknown:
        raise SystemExit(f"--max-ratio names no task: {', '.join(sorted(unknown))}")
    best_ok = False             # nothing kept yet: the first evaluation is saved, within limits or not
    if args.resume and last.exists():
        from anulm.model import safe_load
        st = safe_load(last, map_location=dev)       # weights-only: a resume file runs no code
        if "recipe" not in st:
            print(f"warning: {last} predates recorded recipes; cannot check it matches this run", flush=True)
        elif st["recipe"] != recipe:
            changed = sorted(k for k in recipe if st["recipe"].get(k) != recipe[k])
            raise SystemExit(f"{last} was written with a different recipe ({', '.join(changed)}): "
                             "resume with the original settings, or train to a new --out")
        model.load_state_dict(st["model"])
        if om.sm is not None:
            om.sm.proj.load_state_dict(st["proj"])
        if om.vl is not None:
            om.vl.proj.load_state_dict(st["vision_proj"])
        opt.load_state_dict(st["opt"])
        first, best, seen = st["first"], st["best"], st["seen"]
        best_ok = st.get("best_ok", True)       # files from before --max-ratio had no limits to fail
        start = st["step"] + 1
        for t in om.tasks:      # a fresh shuffle per task: the exact batch order is not worth saving
            t.epoch = 100 + st["step"]
        print(f"resumed from {last} at step {st['step']} (best score {best:.4f})", flush=True)
    else:
        first = evaluate()
        if any(not math.isfinite(value) or value <= 0 for value in first.values()):
            raise ValueError("Baseline task losses must be finite and positive.")
        with open(curve, "a", encoding="utf-8") as f:
            f.write("0,1.0," + ",".join(str(first[n]) for n in names) + "\n")
    print("step  start | held-out at step 0: " + "  ".join(f"{k} {v:.4f}" for k, v in first.items()), flush=True)
    print(f"each optimizer step sums {args.accum} batch(es)", flush=True)     # app_omni.py reads this
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
        # Every AdamW step moves the weights by about lr whatever the batch
        # held, and an intent or VQA batch is ~30 answer tokens: --accum sums
        # several batches (any tasks) into one step so each step averages more.
        for _ in range(args.accum):
            t = om.tasks[rng.choice(len(om.tasks), p=w)]
            batch = om.next_batch(t)
            eb, tb = batch[0], batch[1]
            with autocast:
                logits, loss = model(None, tb, embeds=eb)
                if t.name in om.distill:
                    with torch.no_grad():
                        t_logits = answer_logits(om.teacher, om.teacher.embed_tokens(batch[2]), tb)
                    loss, ce, kl = distill_loss(logits, t_logits, tb, loss, args.distill_alpha)
                    run_loss.setdefault(t.name + "_kl", []).append(kl.item())
            if not torch.isfinite(loss):
                raise ValueError(f"{t.name}: non-finite training loss at step {step}.")
            (loss / args.accum).backward()
            seen[t.name] += 1
            # logged as cross-entropy for every task, so a distilled run reads like any other
            run_loss[t.name].append(ce.item() if t.name in om.distill else loss.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        imbalance = model.update_expert_biases()
        if step % args.log_every == 0:
            msg = "  ".join(f"{k} {np.mean(v):.3f}" for k, v in run_loss.items() if v)
            print(f"step {step:6d} | {msg} | imbalance {imbalance:4.2f}x | "
                  f"lr x{frac:.2f} | {time.time()-t0:6.0f}s", flush=True)
            run_loss = {k: [] for k in run_loss}
        # --stop-after evaluates where it stops, even off the --eval-every grid,
        # so an A/B probe trains exactly the steps it asked for
        if step % args.eval_every == 0 or step == args.steps or step == args.stop_after:
            losses = evaluate()
            score = float(np.mean([losses[k] / first[k] for k in losses]))
            flag = ""
            ok = within_limits(losses, first, limits)
            if better_checkpoint(score, ok, best, best_ok):
                best, best_ok = score, ok
                atomic_save(om.state(step, score, losses), args.out)
                flag = "  <- saved" + ("" if ok else " (outside --max-ratio: kept until one within it)")
            elif not ok:
                flag = "  (outside --max-ratio)"
            with open(curve, "a", encoding="utf-8") as f:
                f.write(f"{step},{score:.4f}," + ",".join(str(losses[n]) for n in names) + "\n")
            print(f"  eval @ {step:6d} | score {score:.4f}{flag} | "
                  + "  ".join(f"{k} {v:.4f}" for k, v in losses.items())
                  + f" | batches seen {seen}", flush=True)
            if args.stop_after and step >= args.stop_after:
                break            # an A/B probe: the full run's schedule, cut short, no resume file
            if step < args.steps:
                st = om.state(step, score, losses)
                st.update(opt=opt.state_dict(), first=first, best=best, best_ok=best_ok, seen=seen,
                          recipe=recipe)
                atomic_save(st, last)
    if last.exists():
        last.unlink()            # finished: the optimizer state is 3.7 GB nobody needs
    print(f"\ndone in {time.time()-t0:.0f}s | best score {best:.4f} | {args.out}")


def main() -> None:
    from .speech_encoder import STACK
    from .vision_encoder import DEFAULT_TOWER, POOL

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
    p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln",
                   help="ln: projectors emit their LayerNorm's own scale; embed: start them at the "
                        "text embeddings' norm")
    p.add_argument("--accum", type=int, default=1, help="batches summed into each optimizer step")
    p.add_argument("--stop-after", type=int, default=0,
                   help="stop at this step (keeping --steps' LR schedule), for short A/B probes")
    p.add_argument("--distill", action="append",
                   help="a text task to keep near the starting checkpoint (KL to a frozen copy of --ckpt)")
    p.add_argument("--distill-alpha", type=float, default=1.0,
                   help="share of a distilled task's cross-entropy replaced by the KL (1.0 = KL only)")
    p.add_argument("--max-ratio", action="append",
                   help="task=R: prefer checkpoints whose held-out loss on task is at most R x its step-0 "
                        "value (chat=1.049 mirrors the chat release gate, 3.30 / 3.146)")
    p.add_argument("--whisper", default="small")
    p.add_argument("--stack", type=int, default=STACK)
    p.add_argument("--tower", default=DEFAULT_TOWER)
    p.add_argument("--pool", type=int, default=POOL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from <out>.last if it exists")
    p.add_argument("--grad-ckpt", action="store_true", help="recompute activations: less memory, ~30%% slower")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.accum < 1:
        p.error("--accum must be at least 1 (0 would train nothing and still save a 'best' checkpoint)")
    train(args)


if __name__ == "__main__":
    main()
