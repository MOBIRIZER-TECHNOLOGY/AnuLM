"""
The text half of the joint model's report card: reading, intents, chat.

    python omni_eval.py --ckpt ckpt_omni.pt --out logs/omni_one/eval_text.json
    python omni_eval.py --ckpt ckpt_chat_v3.pt ...      # the same numbers for a baseline

Speech and images have their own evaluators (`speech_encoder.py eval`,
`vision_encoder.py eval / vqa-eval`); this covers what is left, with the
model alone and greedy decoding, so a run is a number and not a sample:

  read    token F1 against the gold answers (eval_qa.f1, best over golds) on
          the make_rc.py eval files, answer-from-the-passages prompt, the
          passages given -- reading measured without retrieval in the way
  intent  accuracy on BANKING77 and MASSIVE hi-IN, the label chosen by the
          model's own likelihood of each label string (length-normalised), so
          the answer is always a valid intent and a confidence comes with it
  chat    held-out answer loss on data/chat_heldout.jsonl, the number chat v3
          is known by (3.1445)
  vision  (--vision) VQA accuracy against ten human answers on
          data/omni/vqa_test.jsonl, caption held-out loss and five samples

Generation never emits an audio id: a grown checkpoint has 28,677 rows past
the text vocabulary, and they are masked out here rather than trusted to be
unlikely.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).parent


def load(ckpt: str, device: str):
    from hf_tok import load_tokenizer
    from model import AnuLM, load_checkpoint
    ck = load_checkpoint(ckpt, device)
    cfg = replace(ck["cfg"], moe_impl="grouped" if device.startswith("cuda") else "sparse")
    m = AnuLM(cfg).to(device).eval()
    m.load_state_dict(ck["model"])
    text_vocab = ck.get("speech_text_vocab") or cfg.vocab_size
    return m, load_tokenizer(cfg.tokenizer_path), text_vocab


@torch.no_grad()
def greedy(m, tok, prompt_ids, text_vocab, max_new=32, device="cuda"):
    ids = torch.tensor([prompt_ids[-(m.cfg.block_size - max_new):]], device=device)
    out = []
    for _ in range(max_new):
        logits, _ = m(ids)
        logits = logits[0, -1, :text_vocab]
        t = int(logits.argmax())
        if t == tok.eos_id:
            break
        out.append(t)
        ids = torch.cat([ids, torch.tensor([[t]], device=device)], 1)
    return tok.decode(out).strip()


def read_f1(m, tok, text_vocab, path, limit, device):
    from eval_qa import f1
    from make_qa import PROMPTS
    rows = [json.loads(l) for l in open(path, encoding="utf-8")][:limit]
    tot = 0.0
    for r in rows:
        golds = r.get("answers") or [r["answer"]]
        if isinstance(golds, str):
            golds = ast.literal_eval(golds)
        p = tok.encode(PROMPTS[r.get("lang", "rc-en")].format(q=r["question"]))
        pred = greedy(m, tok, p, text_vocab, device=device)
        tot += max(f1(pred, g) for g in golds) if golds else float(pred == "")
    return round(100 * tot / max(len(rows), 1), 1)


@torch.no_grad()
def intent_acc(m, tok, rows, labels, device, limit):
    """Pick argmax over labels of mean log p(label | prompt)."""
    from make_qa import PROMPTS
    lab_ids = [tok.encode(" " + l) + [tok.eos_id] for l in labels]
    right = 0
    rows = rows[:limit]
    for r in rows:
        p = tok.encode(PROMPTS[r["lang"]].format(q=r["question"]))
        seqs = [p + l for l in lab_ids]
        S = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), S, dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s, device=device)
        logits, _ = m(x, x)        # targets in: full-sequence logits
        lp = F.log_softmax(logits.float(), -1)
        scores = []
        for i, l in enumerate(lab_ids):
            pos = torch.arange(len(p) - 1, len(p) - 1 + len(l), device=device)
            scores.append(float(lp[i, pos, torch.tensor(l, device=device)].mean()))
        right += labels[max(range(len(labels)), key=scores.__getitem__)] == r["answer"]
    return round(100 * right / max(len(rows), 1), 2)


@torch.no_grad()
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


def vqa_norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def vision(ckpt: str, device: str, limit: int) -> dict:
    """VQA accuracy the standard way -- min(humans who gave this answer / 3, 1)
    -- on data/omni/vqa_test.jsonl, and the captioner's held-out loss on
    caption_test.jsonl plus five captions to read."""
    from vision_encoder import answer, caption, evaluate, load_manifest, load_trained
    vl, tower, tok = load_trained(ckpt, device)
    res = {}
    base = HERE / "data" / "omni"
    vt = base / "vqa_test.jsonl"
    if vt.exists():
        rows = [json.loads(l) for l in open(vt, encoding="utf-8")][:limit]
        acc = 0.0
        for r in rows:
            pred = vqa_norm(answer(vl, tower, tok, str(base / r["image"]), r["question"], device=device))
            acc += min(sum(vqa_norm(h) == pred for h in r["answers"]) / 3, 1.0)
        res["vqa_acc"] = round(100 * acc / max(len(rows), 1), 1)
        print(f"vqa              acc {res['vqa_acc']}  ({len(rows)} questions)", flush=True)
    ct = base / "caption_test.jsonl"
    if ct.exists():
        rows = load_manifest(str(ct))[:limit]
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda")
                    else torch.autocast("cpu", enabled=False))
        res["caption_heldout_loss"] = round(evaluate(vl, rows, tower, tok, device, autocast), 4)
        res["caption_samples"] = [{"reference": r["text"], "model": caption(vl, tower, tok, r["image"], device=device)}
                                  for r in rows[:5]]
        print(f"caption held-out loss {res['caption_heldout_loss']}", flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=300, help="examples per reading / intent / vqa set")
    p.add_argument("--vision", action="store_true", help="also VQA accuracy and caption loss")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    t0 = time.time()
    m, tok, tv = load(a.ckpt, a.device)
    res = {"ckpt": a.ckpt}
    for name, f in (("read_squad_en", "rc_eval_squad_en"), ("read_mlqa_hi", "rc_eval_mlqa_hi")):
        path = HERE / "data" / f"{f}.jsonl"
        if path.exists():
            res[name] = read_f1(m, tok, tv, path, a.limit, a.device)
            print(f"{name:16s} F1 {res[name]}", flush=True)
    test = HERE / "data" / "omni" / "intent_test.jsonl"
    if test.exists():
        rows = [json.loads(l) for l in open(test, encoding="utf-8")]
        for lang, name in (("en", "intent_banking77"), ("hi", "intent_massive_hi")):
            sub = [r for r in rows if r["lang"] == lang]
            labels = sorted({r["answer"] for r in sub})
            if sub:
                res[name] = intent_acc(m, tok, sub, labels, a.device, a.limit)
                print(f"{name:16s} acc {res[name]}%  ({len(labels)} labels)", flush=True)
    held = HERE / "data" / "chat_heldout.jsonl"
    if held.exists():
        res["chat_heldout_loss"] = chat_loss(m, tok, held, a.device)
        print(f"chat held-out loss {res['chat_heldout_loss']}", flush=True)
    if a.vision:
        del m
        res.update(vision(a.ckpt, a.device, a.limit))
    res["seconds"] = round(time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
