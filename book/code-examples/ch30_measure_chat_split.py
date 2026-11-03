"""
Book example, not repository code.

Chapter 30: measure the chat data that make_chat.py wrote and finetune.py
packed, without printing a single held-out row.

    python book/code-examples/ch30_measure_chat_split.py           # writes part_vii_chat_split.json here
    python book/code-examples/ch30_measure_chat_split.py --save    # writes book/measurements/part_vii_chat_split.json

1. How many held-out questions (data/chat_heldout.jsonl) also appear, as the
   exact same question string, in data/chat_train.jsonl; how many of those
   also have the same answer; and how many training questions occur more than
   once (the copies that make the leak possible).
   Then finetune.py's rule at @084cbdd: how many held-out rows it drops
   because their make_qa.question_key is also a training question.
2. The mechanism, on made-up rows: make_chat.main's old split (@773b6a8,
   copied verbatim), which built the files on disk, holds out one object and
   keeps its identical twin in training; make_chat.split_chat_rows (@084cbdd)
   puts every copy of a question on one side.
3. The packed training cache finetune.py wrote beside chat_train.jsonl:
   rows, block length, and the share of positions that carry a target.

4. The released model (ckpt_base_v2.pt, CPU, fp32 with moe_impl "sparse"; the
   stored "grouped" path would run the experts in bf16): the loss on one made-up
   pair in eval mode and in train mode, where the router regulariser is added.

Writes part_vii_chat_split.json (into book/measurements/ only with --save).
Counts only; no held-out text.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "book" / "measurements" / "part_vii_chat_split.json"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def make_chat_split(rows_in: list[dict], rng: random.Random):
    """make_chat.py main() at @773b6a8, lines 109-112, unchanged: the split that built data/chat_*.jsonl."""
    rows = list(rows_in)
    rng.shuffle(rows)
    held = [r for r in rows if r["source"] != "identity"][:300]
    held_ids = {id(r) for r in held}
    train = [r for r in rows if id(r) not in held_ids]
    return train, held


def main() -> None:
    res: dict = {}
    train = rows(ROOT / "data" / "chat_train.jsonl")
    held = rows(ROOT / "data" / "chat_heldout.jsonl")
    tq = Counter(r["question"] for r in train)
    tqa = {(r["question"], r["answer"]) for r in train}
    leaked = [r for r in held if r["question"] in tq]
    same = [r for r in leaked if (r["question"], r["answer"]) in tqa]
    dup_q = sum(1 for q, c in tq.items() if c > 1)
    res["train_rows"], res["heldout_rows"] = len(train), len(held)
    res["heldout_questions_in_train"] = len(leaked)
    res["of_those_same_answer"] = len(same)
    res["train_questions_seen_more_than_once"] = dup_q
    print(f"train {len(train):,} rows, held-out {len(held)} rows")
    print(f"held-out questions also in train: {len(leaked)} of {len(held)} "
          f"({100 * len(leaked) / len(held):.1f}%), {len(same)} with the same answer")
    ident = sum(1 for q, c in tq.items() if c >= 20)        # make_chat.py repeats each identity row 20 times
    res["of_those_repeated_20_or_more_times_identity"] = ident
    print(f"training questions that occur more than once: {dup_q:,} ({ident} of them 20 or more times: "
          f"the identity rows, which are never held out)")
    dolly_path = ROOT / "data" / "dolly_train.jsonl"
    if dolly_path.exists():
        dq = Counter(r.get("question", r.get("instruction", "")) for r in rows(dolly_path))
        twice = sum(1 for r in leaked if dq[r["question"]] >= 2)
        res["leaked_questions_present_twice_or_more_in_dolly_train"] = twice
        print(f"leaked questions that occur at least twice in data/dolly_train.jsonl: {twice} of {len(leaked)}")

    # finetune.py's rule: drop held-out rows whose question_key is a training question
    from anulm.data.make_qa import question_key
    trained_keys = {question_key(r["question"]) for r in train}
    dropped = sum(question_key(r["question"]) in trained_keys for r in held)
    res["finetune_drops_heldout_rows"] = dropped
    print(f"finetune's rule (question_key: NFKC, case-folded, whitespace collapsed): "
          f"drops {dropped} of {len(held)} held-out rows")

    # 2. the mechanism on made-up rows: two objects with equal content
    a = {"question": "What is 2 + 2?", "answer": "4", "source": "dolly"}
    toy = [a, dict(a)] + [{"question": f"q{i}", "answer": "x", "source": "dolly"} for i in range(400)]
    tr, he = make_chat_split(toy, random.Random(0))
    twin_split = sum(r["question"] == a["question"] for r in he) == 1 and \
        sum(r["question"] == a["question"] for r in tr) == 1
    res["toy_twin_split_by_seed_0"] = twin_split
    print(f"made-up rows, one question present twice: with seed 0 one copy is held out "
          f"and the other trained on: {twin_split}")
    from anulm.data.make_chat import split_chat_rows
    from anulm.data.make_qa import question_key as qk
    variant = {"question": "  what is 2 + 2? ", "answer": "Four.", "source": "oasst2"}
    toy2 = [a, dict(a), variant] + [{"question": f"q{i}", "answer": "x", "source": "dolly"} for i in range(400)]
    sides = {}
    for seed in range(20):
        tr2, he2 = split_chat_rows(toy2, seed=seed)
        n_tr = sum(qk(r["question"]) == qk(a["question"]) for r in tr2)
        n_he = sum(qk(r["question"]) == qk(a["question"]) for r in he2)
        sides[seed] = (n_tr, n_he)
    both = sum(t > 0 and h > 0 for t, h in sides.values())
    kept = {t + h for t, h in sides.values()}
    held_seeds = sum(h > 0 for t, h in sides.values())
    res["split_chat_rows_toy"] = {"seeds": 20, "both_sides": both, "copies_kept": sorted(kept),
                                  "seeds_holding_it_out": held_seeds}
    print(f"split_chat_rows on the same rows plus a re-spaced, re-cased copy with another answer, "
          f"20 seeds: question on both sides {both} times; copies kept {sorted(kept)} of 3 "
          f"(the exact duplicate dropped); held out with {held_seeds} seeds")

    # 3. the packed cache finetune.load_packed wrote
    cache = ROOT / "data" / "chat_train.jsonl.multi32k.b1024.pt"
    if cache.exists():
        c = torch.load(cache, weights_only=False)
        x, y = c["x"], c["y"]
        share = float((y != -1).float().mean())
        res["packed"] = {"rows": int(x.shape[0]), "row_length": int(x.shape[1]),
                         "pairs": int(c["n_items"]), "answer_share": round(share, 4),
                         "answer_tokens": int((y != -1).sum())}
        print(f"packed cache: {c['n_items']:,} pairs -> {x.shape[0]:,} rows of {x.shape[1]}; "
              f"{100 * share:.1f}% of positions carry a target ({int((y != -1).sum()):,} tokens)")

    # 4. the released model: the loss on one made-up pair in eval mode and in train mode
    from dataclasses import replace
    from anulm.bpe import BPE
    from anulm.training.finetune import _pair_ids, pack
    from anulm.model import AnuLM, load_checkpoint
    ck = load_checkpoint(str(ROOT / "ckpt_base_v2.pt"), "cpu")
    cfg = replace(ck["cfg"], moe_impl="sparse")
    tok = BPE.load(str(ROOT / cfg.tokenizer_path))
    model = AnuLM(cfg)
    model.load_state_dict(ck["model"])
    p, a = _pair_ids(tok, {"question": "What is the capital of India?", "answer": "New Delhi.", "lang": "en"})
    xb, yb = pack([(p, a)], len(p) + len(a), tok.eos_id)
    with torch.no_grad():
        _, ev = model.eval()(xb.long(), yb.long())
        _, tr = model.train()(xb.long(), yb.long())
        aux = float(model.aux_loss)
    res["released_pair_loss"] = {"precision": "fp32 on the CPU (moe_impl sparse)", "eval": round(ev.item(), 4), "train": round(tr.item(), 4),
                                 "train_minus_aux": round(tr.item() - aux, 4), "aux": round(aux, 4),
                                 "seq_balance_alpha": cfg.seq_balance_alpha, "router_z_alpha": cfg.router_z_alpha}
    print(f"released model, one made-up pair: loss {ev.item():.4f} in eval mode, {tr.item():.4f} in train mode "
          f"(router regulariser {aux:.4f}; without it {tr.item() - aux:.4f})")
    out = OUT if "--save" in sys.argv else Path.cwd() / OUT.name     # book/measurements/ only with --save
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
