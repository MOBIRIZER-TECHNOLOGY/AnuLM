"""
AnuLM-Decide: typed, calibrated decisions instead of text (a "System One" model).

    python decide.py train --ckpt ckpt_demo_v2.pt --out ckpt_decide.pt
    python decide.py eval  --ckpt ckpt_decide.pt
    python decide.py ask   "I still haven't received my card"

The idea, from TypeSafe's Jev (docs/RESULTS.md section 34): software often
needs a decision -- which queue, which intent, is this phishing -- not prose.
So the model never writes text. It is given a message and a declared list of
options, reads both in ONE forward pass, and returns a probability for every
option; the answer can only be one of the options, and the confidence is
calibrated so that "0.9" means right about 90% of the time.

How, on this project's own from-scratch backbone:

  * the prompt is the message, then every option's name: "Message: ... Options:
    - card arrival - card linking - ...". The backbone is causal, so the
    hidden state at the LAST token of each option's name has read the message
    and that option; a linear head scores it. Softmax over the options gives
    the distribution. One pass, whatever the number of options (77 here).
  * options are shuffled on every training example, so the head learns to
    read option names, not positions -- the same model takes any option list.
  * trained with cross-entropy (a proper scoring rule), then calibrated with a
    single temperature fitted on held-out data (Guo et al. 2017). Reported:
    accuracy, expected calibration error (ECE), Brier score, and accuracy when
    it answers only above a confidence threshold, handing the rest to a human.

First use case: BANKING77 (CC BY 4.0), 77 customer-support intents, 10,003
training / 3,080 test queries, from PolyAI's repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import AnuLM, fit_memory, load_checkpoint

HERE = Path(__file__).parent
DATA = HERE / "data" / "banking77"


# --------------------------------------------------------------------- data
def load_split(name: str, task: str = "banking77") -> list[tuple[str, str]]:
    """(text, label) rows. MASSIVE (Amazon, CC BY 4.0) intents in Hindi or
    English, 60 labels; only train and validation are on disk, so its
    validation split serves as the test set (training holds out its own
    validation rows from train)."""
    if task.startswith("massive-"):
        lang = {"massive-hi": "hi-IN", "massive-en": "en-US"}[task]
        split = {"train": "train", "test": "validation"}[name]
        rows = [json.loads(l) for l in open(HERE / "data" / "decide_general" / f"massive_{lang}" / f"{split}.jsonl",
                                            encoding="utf-8")]
        return [(r["text"], r["label_text"]) for r in rows]
    return [(r["text"], r["category"]) for r in csv.DictReader(open(DATA / f"{name}.csv", encoding="utf-8"))]


def labels(task: str = "banking77") -> list[str]:
    if task.startswith("massive-"):
        return sorted({l for _, l in load_split("train", task)})
    return json.load(open(DATA / "categories.json"))


def readable(label: str) -> str:
    """'card_arrival' -> 'card arrival'; 'EducationalInstitution' -> 'educational institution'."""
    import re
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", label).replace("_", " ").lower()


def encode(tok, text: str, options: list[str], max_len: int = 1024):
    """-> (ids, positions) where positions[k] is the index of the last token of
    option k's name. Pieces are encoded separately so the positions are exact."""
    ids = [tok.eos_id] + tok.encode(f"Message: {text.strip()}\nWhich of these does it ask for?\nOptions:")
    pos = []
    for o in options:
        ids += tok.encode(f"\n- {readable(o)}")
        pos.append(len(ids) - 1)
    assert len(ids) <= max_len, f"{len(ids)} tokens; too many options for the context"
    return ids, pos


# --------------------------------------------------------------------- model
class Decider(nn.Module):
    """A decision head on one of three backbones:

    kind "causal"  AnuLM as trained: an option sees the message and the
                   options before it, never those after (the first version).
    kind "bidir"   AnuLM converted to an encoder (bidirectional.py): every
                   token sees every real token.
    kind "hf"      an open pretrained encoder (ModernBERT) -- a reference
                   point, not this project's own model.
    """

    def __init__(self, model, kind: str = "causal", pad_id: int = 0):
        super().__init__()
        self.model, self.kind, self.pad_id = model, kind, pad_id
        hid = model.cfg.hidden_size if kind != "hf" else model.config.hidden_size
        self.head = nn.Linear(hid, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("temperature", torch.ones(()))
        # "full": every weight trains. "head" / "lora": the backbone stays
        # exactly as loaded, so one checkpoint can still chat (section 37).
        self.mode, self.lora, self.lora_rank, self._lora_on = "full", None, 0, False
        self.lora_targets, self.lora_alpha = "dense", 32.0

    LORA_TARGETS = ("qv", "attn", "dense", "all")

    def freeze(self, mode: str, rank: int = 16, targets: str = "dense", alpha: float | None = None):
        """Keep the backbone's weights fixed. "lora" adds low-rank adapters
        through forward hooks, so the backbone's own modules and state dict are
        untouched and the adapters act only inside hidden(): generation
        through the same model is unchanged. Which linears get one:

          qv     Q and V only (Hu et al. 2021). AnuLM fuses Q, K and V into one
                 query_key_value matrix, so the adapter writes into its Q and V
                 output slices and leaves K's alone (LoRAQV).
          attn   query_key_value (Q, K, V) and the output projection, every layer.
          dense  every linear except lm_head and the routed experts: attention,
                 layer 0's MLP, any shared expert (the default, sections 37-38).
          all    dense plus every routed expert. The grouped MoE kernel never
                 calls the expert modules, so this switches the MoE layers to
                 the sparse path (slower; an experiment, not for serving).
        """
        assert targets in self.LORA_TARGETS, targets
        self.mode = mode
        for p in self.model.parameters():
            p.requires_grad_(False)
        if mode != "lora":
            return
        alpha = 2.0 * rank if alpha is None else alpha
        self.lora_rank, self.lora_targets, self.lora_alpha = rank, targets, alpha
        cfg, mods = self.model.cfg, {}
        for name, mod in self.model.named_modules():
            if not isinstance(mod, nn.Linear) or name == "lm_head":
                continue
            attn = name.endswith(("attn.query_key_value", "attn.dense"))
            if (targets == "qv" and name.endswith("attn.query_key_value")) or (targets == "attn" and attn) \
                    or (targets == "dense" and ".experts." not in name) or targets == "all":
                mods[name.replace(".", "__")] = mod
        assert mods, f"no linears matched lora targets {targets!r}"
        if targets == "all":
            for m in self.model.modules():
                if hasattr(m, "impl") and hasattr(m, "experts"):
                    m.impl, m._stacked = "sparse", None
        q, kv = cfg.n_head * cfg.head_dim, getattr(cfg, "n_kv_head", cfg.n_head) * cfg.head_dim
        self.lora = nn.ModuleDict({
            k: LoRAQV(m.in_features, q, kv, rank, alpha) if targets == "qv"
            else LoRA(m.in_features, m.out_features, rank, alpha) for k, m in mods.items()})
        for k, mod in mods.items():
            mod.register_forward_hook(self._lora_hook(k))

    def _lora_hook(self, key):
        def hook(mod, inp, out):
            return out + self.lora[key](inp[0]).to(out.dtype) if self._lora_on else out
        return hook

    def train(self, mode: bool = True):
        super().train(mode)
        if self.mode != "full":
            self.model.eval()                  # frozen: no router statistics, no bias updates
        return self

    def hidden(self, idx):
        if self.mode == "head":
            with torch.no_grad():
                return self._hidden(idx)
        self._lora_on = self.lora is not None
        try:
            return self._hidden(idx)
        finally:
            self._lora_on = False

    def _hidden(self, idx):
        # Real tokens: everything but padding (position 0 is the start token,
        # which for AnuLM shares its id with the padding).
        attn = idx != self.pad_id
        attn[:, 0] = True
        if self.kind == "hf":
            return self.model(input_ids=idx, attention_mask=attn.long()).last_hidden_state
        if self.kind == "bidir":
            from bidirectional import bidir_hidden
            return bidir_hidden(self.model, idx, attn)
        m = self.model                         # causal: right padding never reaches real tokens
        x = F.embedding(idx, m._table("in"))
        cos, sin = m.rotary(idx.shape[1], x.device, x.dtype)
        for layer in m.layers:
            if m.grad_ckpt and self.training:
                x, _ = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x, _ = layer(x, cos, sin)
        return m.norm(x)

    def balance(self):
        if self.mode == "full" and hasattr(self.model, "update_expert_biases"):
            self.model.update_expert_biases()

    def save(self, path, **extra):
        state = {"decide_head": self.head.state_dict(), "kind": self.kind, "mode": self.mode, **extra}
        if self.lora is not None:
            state.update(lora=self.lora.state_dict(), lora_rank=self.lora_rank,
                         lora_targets=self.lora_targets, lora_alpha=self.lora_alpha)
        if self.kind == "hf":
            state.update(hf_backbone=self.hf_path, hf_state=self.model.state_dict())
        else:
            state.update(model=self.model.state_dict(), cfg=self.model.cfg, bidirectional=self.kind == "bidir")
        torch.save(state, path)

    def forward(self, idx, pos, mask=None):
        """idx (B, S); pos (B, K) option positions; mask (B, K) real options
        when the examples in a batch have different numbers of them -> (B, K)."""
        h = self.hidden(idx)
        g = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))
        logits = self.head(g).squeeze(-1).float()
        return logits if mask is None else logits.masked_fill(~mask, float("-inf"))


class LoRA(nn.Module):
    """x -> B(A(x)) * alpha / r, B starting at zero so training starts from
    the frozen model (Hu et al. 2021)."""

    def __init__(self, n_in: int, n_out: int, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.A = nn.Linear(n_in, r, bias=False)
        self.B = nn.Linear(r, n_out, bias=False)
        nn.init.normal_(self.A.weight, std=1 / r)
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


class LoRAQV(nn.Module):
    """LoRA on a fused query_key_value projection that adapts Q and V only:
    one shared down-projection, an up-projection into each of the Q and V
    output slices, and nothing written into K's slice. The fused output is
    laid out [Q | K | V] (GQAttention.forward splits it that way)."""

    def __init__(self, n_in: int, q: int, kv: int, r: int = 16, alpha: float = 32.0):
        super().__init__()
        self.A = nn.Linear(n_in, r, bias=False)
        self.Bq = nn.Linear(r, q, bias=False)
        self.Bv = nn.Linear(r, kv, bias=False)
        nn.init.normal_(self.A.weight, std=1 / r)
        nn.init.zeros_(self.Bq.weight)
        nn.init.zeros_(self.Bv.weight)
        self.scale, self.kv = alpha / r, kv

    def forward(self, x):
        h = self.A(x)
        k = h.new_zeros(*h.shape[:-1], self.kv)
        return torch.cat([self.Bq(h), k, self.Bv(h)], -1) * self.scale


def batchify(examples, tok, pad, with_mask=False):
    L = max(len(e[0]) for e in examples)
    K = max(len(e[1]) for e in examples)
    x = torch.full((len(examples), L), pad, dtype=torch.long)
    pos = torch.zeros(len(examples), K, dtype=torch.long)
    mask = torch.zeros(len(examples), K, dtype=torch.bool)
    for b, e in enumerate(examples):
        x[b, :len(e[0])] = torch.tensor(e[0])
        pos[b, :len(e[1])] = torch.tensor(e[1])
        mask[b, :len(e[1])] = True
    y = torch.tensor([e[2] for e in examples])
    return (x, pos, y, mask) if with_mask else (x, pos, y)


def make_examples(rows, tok, labs, rng=None):
    out = []
    for text, cat in rows:
        opts = labs[:]
        if rng:
            rng.shuffle(opts)
        ids, pos = encode(tok, text, opts)
        out.append((ids, pos, opts.index(cat), opts))
    return out


def _hf_encoder(path: str):
    from transformers import AutoModel

    from hf_tok import EncoderTokenizer
    if not Path(path).is_absolute() and not Path(path).exists():
        path = str(HERE / path)                           # saved relative to the repo
    return AutoModel.from_pretrained(path), EncoderTokenizer(Path(path) / "tokenizer.json")


def load(ckpt: str, device: str):
    """A decide checkpoint, an AnuLM base (causal or bidirectional.py's), or
    the folder of an open encoder (config.json + tokenizer.json)."""
    from hf_tok import load_tokenizer
    p = Path(ckpt)
    if p.is_dir() and "config" not in json.load(open(p / "config.json", encoding="utf-8")):
        enc, tok = _hf_encoder(ckpt)                          # a fresh open encoder (export_hf.py
        d = Decider(enc, "hf", tok.pad_id)                    # folders nest AnuLM's under "config")
        d.hf_path = ckpt
        return d.to(device), tok, {}
    ck = load_checkpoint(ckpt, "cpu")
    if "hf_backbone" in ck:                                   # a trained decider on an open encoder
        enc, tok = _hf_encoder(ck["hf_backbone"])
        enc.load_state_dict(ck["hf_state"])
        d = Decider(enc, "hf", tok.pad_id)
        d.hf_path = ck["hf_backbone"]
    else:
        cfg = replace(ck["cfg"], moe_impl="grouped" if device.startswith("cuda") else "sparse")
        m = AnuLM(cfg)
        m.load_state_dict(ck["model"])
        tp = Path(cfg.tokenizer_path)
        tok = load_tokenizer(tp if tp.is_absolute() or tp.exists() else HERE / tp)
        d = Decider(m, "bidir" if ck.get("bidirectional") else "causal", tok.eos_id)
    if ck.get("mode", "full") != "full":
        d.freeze(ck["mode"], ck.get("lora_rank", 16), ck.get("lora_targets", "dense"), ck.get("lora_alpha", 32.0))
        if "lora" in ck:
            d.lora.load_state_dict(ck["lora"])
    if "decide_head" in ck:
        d.head.load_state_dict(ck["decide_head"])
        d.temperature.fill_(ck.get("temperature", 1.0))
    return d.to(device), tok, ck


# --------------------------------------------------------------------- calibration and metrics
def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """The single T minimising held-out NLL of softmax(logits / T)."""
    best = (float("inf"), 1.0)
    for t in [x / 100 for x in range(30, 501, 5)]:
        nll = F.cross_entropy(logits / t, y).item()
        best = min(best, (nll, t))
    return best[1]


def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> dict:
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean().item() * abs(conf[m].mean().item() - correct[m].mean().item())
    onehot = F.one_hot(y, probs.shape[1]).float()
    brier = ((probs - onehot) ** 2).sum(1).mean().item()
    out = {"accuracy": correct.mean().item(), "ece": ece, "brier": brier,
           "mean_confidence": conf.mean().item(), "n": len(y)}
    for tau in (0.5, 0.7, 0.9):
        keep = conf >= tau
        out[f"answer_if_conf>={tau}"] = {"coverage": keep.float().mean().item(),
                                         "accuracy": correct[keep].mean().item() if keep.any() else float("nan")}
    return out


@torch.no_grad()
def predict_logits(d: Decider, examples, tok, device, bs=16):
    """Logits per example; examples in one call must share an option count."""
    d.eval()
    out = []
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))
    for i in range(0, len(examples), bs):
        x, pos, _ = batchify(examples[i:i + bs], tok, d.pad_id)
        with ac:
            out.append(d(x.to(device), pos.to(device)).cpu())
    return torch.cat(out)


# --------------------------------------------------------------------- the general version
GEN = HERE / "data" / "decide_general"
# Kept out of training so BANKING77 is a true zero-shot test: CLINC's banking
# and credit-card domains, its exchange_rate intent, MASSIVE's qa_currency,
# and CLINC's out-of-scope class.
CLINC_EXCLUDE = {
    "transfer", "transactions", "balance", "freeze_account", "pay_bill", "bill_balance", "bill_due",
    "interest_rate", "routing", "min_payment", "order_checks", "pin_change", "report_fraud",
    "account_blocked", "spending_history",
    "credit_score", "report_lost_card", "credit_limit", "rewards_balance", "new_card",
    "application_status", "card_declined", "international_fees", "apr", "redeem_rewards",
    "credit_limit_change", "damaged_card", "replacement_card_duration", "expiration_date",
    "improve_credit_score", "exchange_rate", "oos"}
MASSIVE_EXCLUDE = {"qa_currency"}


def general_tasks(seed: int = 0) -> dict:
    """{task: (labels, train_rows, val_rows)} -- five label sets, none of them
    about banking. Rows are (text, label)."""
    import pandas as pd
    rng = random.Random(seed)
    tasks = {}
    names = json.load(open(GEN / "clinc" / "intent_names.json"))
    def clinc(split):
        df = pd.read_parquet(GEN / "clinc" / "plus" / f"{split}-00000-of-00001.parquet")
        return [(t, names[i]) for t, i in zip(df.text, df.intent) if names[i] not in CLINC_EXCLUDE]
    tr, va = clinc("train"), clinc("validation")
    tasks["clinc intent"] = (sorted({l for _, l in tr}), tr, va)
    for lang in ("en-US", "hi-IN"):
        def massive(split):
            rows = [json.loads(l) for l in open(GEN / f"massive_{lang}" / f"{split}.jsonl", encoding="utf-8")]
            return [(r["text"], r["label_text"]) for r in rows if r["label_text"] not in MASSIVE_EXCLUDE]
        tr, va = massive("train"), massive("validation")
        tasks[f"massive intent {lang}"] = (sorted({l for _, l in tr}), tr, va)
        sc_tr = [(t, l.split("_")[0]) for t, l in rng.sample(tr, 4000)]
        sc_va = [(t, l.split("_")[0]) for t, l in va]
        tasks[f"massive scenario {lang}"] = (sorted({l for _, l in sc_tr}), sc_tr, sc_va)
    df = pd.read_parquet(GEN / "dbpedia" / "dbpedia_14" / "train-00000-of-00001.parquet").sample(7000, random_state=seed)
    dbl = json.load(open(GEN / "dbpedia" / "label_names.json"))
    rows = [(f"{t}. {c[:300]}", dbl[l]) for t, c, l in zip(df.title, df.content, df.label)]
    tasks["dbpedia topic"] = (sorted(set(dbl)), rows[:6000], rows[6000:])
    return tasks


def sample_options(labels, gold, rng, k_min=5, k_max=77):
    """The gold label plus a random subset of the others, shuffled: the model
    must learn to pick from any list, of any length, in any order."""
    k = rng.randint(min(k_min, len(labels)), min(k_max, len(labels)))
    opts = [gold] + rng.sample([l for l in labels if l != gold], k - 1)
    rng.shuffle(opts)
    return opts


def train_general(a):
    torch.manual_seed(0)
    rng = random.Random(0)
    d, tok, ck = load(a.ckpt, a.device)
    if d.kind != "hf":
        fit_memory(d.model)
    tasks = general_tasks()
    pool = [(name, t, l) for name, (labs, tr, _) in tasks.items() for t, l in tr]
    rng.shuffle(pool)
    val = []
    for name, (labs, _, va) in tasks.items():                     # temperature: training tasks only
        for t, l in rng.sample(va, min(a.val_per_task, len(va))):
            opts = sample_options(labs, l, rng)
            ids, pos = encode(tok, t, opts)
            val.append((ids, pos, opts.index(l), opts))
    for name, (labs, tr, va) in tasks.items():
        print(f"  {name:24s} {len(labs):3d} labels  {len(tr):6d} train  {len(va):5d} val")
    opt = torch.optim.AdamW([{"params": d.model.parameters(), "lr": a.lr},
                             {"params": d.head.parameters(), "lr": 1e-3}], betas=(0.9, 0.95),
                            weight_decay=0.0, fused=a.device.startswith("cuda"))
    steps = int(len(pool) * a.epochs) // a.batch_size
    base = [g["lr"] for g in opt.param_groups]
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda"))
    print(f"general training: {len(pool)} examples over {len(tasks)} tasks, {steps} steps; "
          f"BANKING77 and banking-like intents excluded", flush=True)
    t0, best = time.time(), float("inf")
    d.train()
    for step in range(steps):
        batch = []
        for name, t, l in pool[(step * a.batch_size) % len(pool):][:a.batch_size]:
            opts = sample_options(tasks[name][0], l, rng)
            ids, pos = encode(tok, t, opts)
            batch.append((ids, pos, opts.index(l), opts))
        x, pos, y, mask = batchify(batch, tok, d.pad_id, with_mask=True)
        scale = min(1.0, (step + 1) / 200) * max(0.0, 1 - step / steps)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * scale
        with ac:
            logits = d(x.to(a.device), pos.to(a.device), mask.to(a.device))
        loss = F.cross_entropy(logits, y.to(a.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        d.balance()
        if step % 200 == 0:
            print(f"step {step:5d}/{steps} | loss {loss.item():.4f} | {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % a.eval_every == 0 or step == steps - 1:
            d.eval()
            lv, yv = [], []
            with torch.no_grad():
                for i in range(0, len(val), 16):
                    xb, pb, ybt, mb = batchify(val[i:i + 16], tok, d.pad_id, with_mask=True)
                    with ac:
                        lv.append(d(xb.to(a.device), pb.to(a.device), mb.to(a.device)).cpu())
                    yv.append(ybt)
            K = max(t.shape[1] for t in lv)
            lv = torch.cat([F.pad(t, (0, K - t.shape[1]), value=float("-inf")) for t in lv])
            yv = torch.cat(yv)
            nll = F.cross_entropy(lv, yv).item()
            acc = (lv.argmax(1) == yv).float().mean().item()
            flag = ""
            if nll < best:
                best = nll
                tmp = fit_temperature(lv, yv)
                d.save(a.out, temperature=tmp, step=step, val_accuracy=acc, labels=None,
                       general=True, tasks=list(tasks), base_ckpt=a.ckpt)
                flag = f"  <- saved (temperature {tmp:.2f})"
            print(f"  eval @ {step:5d} | held-out (training tasks) accuracy {100 * acc:.2f}% nll {nll:.4f}{flag}",
                  flush=True)
            d.train()
    print(f"done in {time.time() - t0:.0f}s | best held-out nll {best:.4f} | {a.out}")


# --------------------------------------------------------------------- train / eval / ask
def train(a):
    torch.manual_seed(0)
    rng = random.Random(0)
    d, tok, ck = load(a.ckpt, a.device)
    if d.kind != "hf":
        fit_memory(d.model)
    if a.grad_ckpt and d.kind != "hf":
        d.model.enable_gradient_checkpointing()
    labs = labels(a.task)
    rows = load_split("train", a.task)
    rng.shuffle(rows)
    if a.limit:
        rows = rows[:a.limit]
    n_val = min(1000, len(rows) // 10)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    val = make_examples(val_rows, tok, labs)                       # canonical order, as at test
    if a.mode != "full":
        d.freeze(a.mode, a.lora_rank, a.lora_targets, a.lora_alpha)
        d.to(a.device)
    body = {"full": d.model.parameters(), "head": [], "lora": d.lora.parameters() if d.lora else []}[a.mode]
    groups = [{"params": list(body), "lr": a.lr}, {"params": d.head.parameters(), "lr": a.head_lr}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.95),
                            weight_decay=0.0, fused=a.device.startswith("cuda"))
    n_train = sum(p.numel() for g in opt.param_groups for p in g["params"])
    print(f"mode {a.mode}: training {n_train:,} parameters", flush=True)
    steps = int(len(train_rows) * a.epochs) // a.batch_size
    base = [g["lr"] for g in opt.param_groups]
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device.startswith("cuda"))
    print(f"train {len(train_rows)} / val {len(val_rows)} queries, {len(labs)} intents, {steps} steps", flush=True)
    t0, best, order = time.time(), -1.0, []
    d.train()
    for step in range(steps):
        if len(order) < a.batch_size:
            order = list(range(len(train_rows)))
            rng.shuffle(order)
        batch = make_examples([train_rows[order.pop()] for _ in range(a.batch_size)], tok, labs, rng)
        x, pos, y = batchify(batch, tok, d.pad_id)
        scale = min(1.0, (step + 1) / 100) * max(0.0, 1 - step / steps)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * scale
        with ac:
            logits = d(x.to(a.device), pos.to(a.device))
        loss = F.cross_entropy(logits, y.to(a.device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(d.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        d.balance()
        if step % 100 == 0:
            print(f"step {step:5d}/{steps} | loss {loss.item():.4f} | {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % a.eval_every == 0 or step == steps - 1:
            lv = predict_logits(d, val, tok, a.device)
            yv = torch.tensor([e[2] for e in val])
            acc = (lv.argmax(1) == yv).float().mean().item()
            flag = ""
            if acc > best:
                best = acc
                t = fit_temperature(lv, yv)
                d.save(a.out, temperature=t, step=step, val_accuracy=acc, labels=labs, base_ckpt=a.ckpt, task=a.task)
                flag = f"  <- saved (temperature {t:.2f})"
            print(f"  eval @ {step:5d} | val accuracy {100 * acc:.2f}%{flag}", flush=True)
            d.train()
    print(f"done in {time.time() - t0:.0f}s | best val accuracy {100 * best:.2f}% | {a.out}")


def evaluate(a):
    d, tok, ck = load(a.ckpt, a.device)
    task = ck.get("task", "banking77")
    labs = ck.get("labels") or labels(task)
    test = make_examples(load_split("test", task)[:a.limit or None], tok, labs)
    y = torch.tensor([e[2] for e in test])
    logits = predict_logits(d, test, tok, a.device)
    t = float(d.temperature)
    for name, temp in (("raw (T=1)", 1.0), (f"calibrated (T={t:.2f}, fitted on val)", t)):
        m = metrics(F.softmax(logits / temp, 1), y)
        print(f"\n{name}: accuracy {100 * m['accuracy']:.2f}% on {m['n']} {task} test queries | "
              f"ECE {100 * m['ece']:.2f}% | Brier {m['brier']:.4f} | mean confidence {100 * m['mean_confidence']:.1f}%")
        for k, v in m.items():
            if k.startswith("answer_if"):
                print(f"  {k}: answers {100 * v['coverage']:.1f}% of queries at {100 * v['accuracy']:.2f}% accuracy, "
                      f"hands {100 * (1 - v['coverage']):.1f}% to a human")
    # latency: one query, one forward pass
    ex = test[:50]
    with torch.no_grad():
        for e in ex[:3]:
            predict_logits(d, [e], tok, a.device, bs=1)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for e in ex:
            predict_logits(d, [e], tok, a.device, bs=1)
        if a.device.startswith("cuda"):
            torch.cuda.synchronize()
    print(f"\nlatency: {1000 * (time.time() - t0) / len(ex):.1f} ms per query on {a.device} ({len(labs)} options, one pass)")


class Decide:
    """The typed interface: decide(message, options) -> choice, probabilities, confidence, action."""

    def __init__(self, ckpt: str = "ckpt_decide.pt", device: str = "cuda", threshold: float = 0.7):
        self.d, self.tok, ck = load(ckpt, device)
        self.d.eval()
        self.device, self.threshold = device, threshold
        self.labels = ck.get("labels") or labels()

    @classmethod
    def on(cls, model, tok, ckpt: str, device: str, threshold: float = 0.7):
        """Decide on a model that is already loaded (the chat model), from a
        --mode head / lora checkpoint: only the head and adapters are added,
        so one copy of the weights serves chat and decisions. Refuses if the
        loaded model is not the backbone the adapter was trained on."""
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        assert ck.get("mode") in ("head", "lora"), f"{ckpt} is a full fine-tune; it has its own weights"
        if "model" in ck:
            live = model.state_dict()
            diff = [k for k, v in ck["model"].items() if not torch.equal(live[k].cpu().to(v.dtype), v)]
            assert not diff, f"the loaded model differs from {ckpt}'s backbone ({len(diff)} tensors)"
        d = Decider(model, "causal", tok.eos_id)
        d.freeze(ck["mode"], ck.get("lora_rank", 16), ck.get("lora_targets", "dense"), ck.get("lora_alpha", 32.0))
        if "lora" in ck:
            d.lora.load_state_dict(ck["lora"])
        d.head.load_state_dict(ck["decide_head"])
        d.temperature.fill_(ck.get("temperature", 1.0))
        d.head.to(device)
        if d.lora is not None:
            d.lora.to(device)
        d.eval()
        self = cls.__new__(cls)
        self.d, self.tok, self.device, self.threshold = d, tok, device, threshold
        self.labels = ck.get("labels") or labels()
        return self

    @torch.no_grad()
    def __call__(self, message: str, options: list[str] | None = None, top: int = 3) -> dict:
        options = options or self.labels
        ids, pos = encode(self.tok, message, options)
        logits = predict_logits(self.d, [(ids, pos, 0, options)], self.tok, self.device, bs=1)[0]
        p = F.softmax(logits / float(self.d.temperature), 0)
        k = p.argsort(descending=True)[:top]
        conf = float(p[k[0]])
        return {"choice": options[int(k[0])],
                "probabilities": {options[int(i)]: round(float(p[i]), 4) for i in k},
                "confidence": round(conf, 4),
                "action": "auto-route" if conf >= self.threshold else "send to human"}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--ckpt", default="ckpt_demo_v2.pt")
    t.add_argument("--out", default="ckpt_decide.pt")
    t.add_argument("--epochs", type=float, default=3.0)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--grad-ckpt", action="store_true")
    t.add_argument("--mode", choices=("full", "head", "lora"), default="full",
                   help="head / lora keep the backbone unchanged, so the same checkpoint can still chat")
    t.add_argument("--lora-rank", type=int, default=16)
    t.add_argument("--lora-alpha", type=float, default=None, help="default 2 x rank")
    t.add_argument("--lora-targets", choices=Decider.LORA_TARGETS, default="dense",
                   help="qv: Q and V only; attn: Q, K, V, output; dense: + layer 0 MLP; all: + experts")
    t.add_argument("--head-lr", type=float, default=1e-3)
    t.add_argument("--task", choices=("banking77", "massive-hi", "massive-en"), default="banking77")
    t.add_argument("--limit", type=int, default=0, help="use only this many training queries (smoke tests)")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g = sub.add_parser("train-general", help="many tasks, BANKING77 excluded, for a zero-shot test")
    g.add_argument("--ckpt", default="ckpt_demo_v2.pt")
    g.add_argument("--out", default="ckpt_decide_general.pt")
    g.add_argument("--epochs", type=float, default=1.0)
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--lr", type=float, default=3e-5)
    g.add_argument("--eval-every", type=int, default=1000)
    g.add_argument("--val-per-task", type=int, default=300)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", default="ckpt_decide.pt")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    q = sub.add_parser("ask")
    q.add_argument("message")
    q.add_argument("--ckpt", default="ckpt_decide.pt")
    q.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
    elif a.cmd == "train-general":
        train_general(a)
    elif a.cmd == "eval":
        evaluate(a)
    else:
        print(json.dumps(Decide(a.ckpt, a.device)(a.message), indent=2))


if __name__ == "__main__":
    main()
