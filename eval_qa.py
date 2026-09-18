"""
Does the model answer questions? Score checkpoints on the held-out QA set.

    python eval_qa.py ckpt_hindi_mixed18k.pt ckpt_hindi_qa.pt --device cuda

Three numbers per checkpoint, on questions whose answer the model never saw
in this form (make_qa.py holds articles out by title):

  MC     4-way multiple choice: the gold answer against three other held-out
         answers, ranked by mean log-probability per token given the prompt.
         Chance 25%. Measures whether the model knows *which* answer belongs
         to the question -- recall of the subject, not surface fluency.
  F1     token-level F1 between the greedy-decoded answer and the gold one
         (SQuAD's metric, on whitespace tokens with punctuation stripped).
         Rewards getting the right words, tolerates paraphrase.
  subj   fraction of decoded answers that mention the question's subject
         (the title, or its first word) -- the cheapest "did it answer the
         question that was asked" check.

Also prints a few decoded answers. The base model is scored with the same
prompt template so the before/after is on one axis.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from bpe import BPE
from make_qa import PROMPT, PROMPTS, template_for
from model import AnuLM

_PUNCT = re.compile(r"[\"'()\[\]“”‘’,;:।.!?\-]")


def tokens(s: str) -> list[str]:
    return _PUNCT.sub(" ", s).split()


def f1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def score(ckpt: str, items: list[dict], device: str, n_gen: int, seed: int = 0) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if device.startswith("cuda"):
        cfg = replace(cfg, moe_impl="grouped")
    m = AnuLM(cfg).to(device).eval()
    m.load_state_dict(ck["model"])
    tok = BPE.load(cfg.tokenizer_path)
    # Per-language templates; a base or Hindi-only checkpoint gets the defaults
    # so every model is asked the same way.
    templates = {**PROMPTS, **(ck.get("qa_templates") or {"hi": ck.get("qa_template", PROMPT)})}
    bs = cfg.block_size
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))
    rng = random.Random(seed)
    t0 = time.time()

    # --- multiple choice ----------------------------------------------------
    answers = [it["answer"] for it in items]
    mc_hits = 0
    for i, it in enumerate(items):
        prompt = tok.encode(template_for(it["question"], templates).format(q=it["question"]))
        others = rng.sample([a for j, a in enumerate(answers) if j != i], 3)
        cands = [it["answer"]] + others
        enc = [tok.encode(" " + c) + [tok.eos_id] for c in cands]
        longest = max(len(e) for e in enc)
        pre = prompt[-(bs - longest):]
        rows = [pre + e + [0] * (longest - len(e)) for e in enc]
        x = torch.tensor(rows, dtype=torch.long, device=device)
        with ac:
            logits, _ = m(x[:, :-1], x[:, 1:])
        logp = F.log_softmax(logits.float(), dim=-1)
        means = []
        for r, e in enumerate(enc):
            pos = torch.arange(len(pre) - 1, len(pre) - 1 + len(e), device=device)
            tgt = x[r, len(pre): len(pre) + len(e)]
            means.append(logp[r, pos, tgt].mean().item())
        mc_hits += int(max(range(4), key=means.__getitem__) == 0)

    # --- generation: F1 and subject mention --------------------------------
    gen_items = items[:n_gen]
    f1s, subj_hits, samples = [], 0, []
    for it in gen_items:
        prompt = tok.encode(template_for(it["question"], templates).format(q=it["question"]))[-(bs - 96):]
        x = torch.tensor([prompt], dtype=torch.long, device=device)
        with ac:
            out = m.generate(x, 96, temperature=1.0, top_k=1, eos_id=tok.eos_id)
        text = tok.decode(out[0, len(prompt):].tolist()).strip()
        text = text.split("\n")[0].strip()          # the base model runs on into a new line
        f1s.append(f1(text, it["answer"]))
        subject = it["title"].split()[0]
        subj_hits += int(subject in text)
        if len(samples) < 6:
            samples.append((it["question"], text, it["answer"]))

    return {"ckpt": ckpt, "mc": mc_hits / len(items), "f1": sum(f1s) / max(len(f1s), 1),
            "subj": subj_hits / max(len(gen_items), 1), "n_mc": len(items), "n_gen": len(gen_items),
            "samples": samples, "seconds": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--heldout", default="data/qa_hindi_heldout.jsonl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=400, help="questions for MC")
    p.add_argument("--gen", type=int, default=100, help="questions to decode for F1 / subj")
    args = p.parse_args()

    # split("\n"): splitlines() also breaks on U+2028 and friends, which
    # json.dumps(ensure_ascii=False) leaves raw inside strings (see finetune.py).
    items = [json.loads(l) for l in Path(args.heldout).read_text(encoding="utf-8").split("\n") if l.strip()]
    items = items[: args.limit]
    print(f"held-out QA {args.heldout}: {len(items)} questions for MC, {args.gen} decoded\n")
    print(f"{'checkpoint':26s} {'MC%':>6s} {'F1':>6s} {'subj%':>6s}")
    print("-" * 48)
    results = []
    for ckpt in args.ckpts:
        r = score(ckpt, items, args.device, args.gen)
        results.append(r)
        print(f"{Path(ckpt).name:26s} {100 * r['mc']:6.1f} {100 * r['f1']:6.1f} {100 * r['subj']:6.1f}   ({r['seconds']:.0f}s)")
    print("\nchance: MC 25%\n")
    for r in results:
        print(f"=== {Path(r['ckpt']).name}")
        for q, pred, gold in r["samples"]:
            print(f"  Q: {q}\n  A: {pred}\n  gold: {gold}\n")


if __name__ == "__main__":
    main()
