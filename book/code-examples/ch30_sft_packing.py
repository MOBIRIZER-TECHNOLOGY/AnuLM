"""
Book example, not repository code.

Chapter 30: instruction tuning's data path, written out plainly and checked
against finetune.py's own functions.

    python book/code-examples/ch30_sft_packing.py

1. Two question-answer pairs become (prompt ids, answer ids + EOS) through the
   English template, and joined they tokenise like the whole string.
2. `pack_by_hand` packs them into rows with the prompt masked to -1; its rows
   equal finetune.pack's exactly.
3. The loss AnuLM.forward returns on a packed row is the mean cross-entropy
   over the answer positions only.
4. Packing has no wall between examples: the second answer's loss changes
   when the first example sits before it in the row.

A tiny BPE and the tiny test model; everything is made up for teaching.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.bpe import BPE  # noqa: E402
from anulm.training.finetune import _pair_ids, pack  # noqa: E402  the repository's code, unchanged
from anulm.data.make_qa import PROMPTS  # noqa: E402
from anulm.model import AnuLM, AnuLMConfig  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def pack_by_hand(pairs, block_size, eos):
    """Greedy packing: whole examples only, prompt targets -1, pad with EOS / -1."""
    L = block_size + 1                       # one extra id so x = row[:-1], y = row[1:]
    rows, tgts, ids, lab = [], [], [], []
    for p, a in pairs:
        if len(p) + len(a) > block_size:     # too long for any row: dropped, never cut
            continue
        if len(ids) + len(p) + len(a) > L:   # does not fit: close this row first
            rows.append(ids + [eos] * (L - len(ids)))
            tgts.append(lab + [-1] * (L - len(lab)))
            ids, lab = [], []
        ids += p + a
        lab += [-1] * len(p) + a             # the prompt is never graded
    if ids:
        rows.append(ids + [eos] * (L - len(ids)))
        tgts.append(lab + [-1] * (L - len(lab)))
    x = torch.tensor(rows)[:, :-1]
    y = torch.tensor(tgts)[:, 1:]            # target at position t is the id at t + 1
    return x, y


def main() -> None:
    text = ("Question: What is the capital of India?\nAnswer: New Delhi is the capital of India.\n"
            "Question: What is two plus two?\nAnswer: Two plus two is four.\n") * 30
    tok = BPE.train(text, 300, verbose=False)
    items = [{"question": "What is the capital of India?", "answer": "New Delhi.", "lang": "en"},
             {"question": "What is two plus two?", "answer": "Four.", "lang": "en"}]

    # 1. one pair as ids
    pairs = [_pair_ids(tok, it) for it in items]
    for it, (p, a) in zip(items, pairs):
        joined = tok.encode(PROMPTS["en"].format(q=it["question"]) + " " + it["answer"])
        assert p + a == joined + [tok.eos_id]
        print(f"pair: {len(p)} prompt ids + {len(a)} answer ids (EOS last: {a[-1] == tok.eos_id}); "
              f"joined string tokenises the same: True")

    # 2. packing, by hand and by finetune.pack
    n = sum(len(p) + len(a) for p, a in pairs)
    x, y = pack_by_hand(pairs, n, tok.eos_id)
    xr, yr = pack(pairs, n, tok.eos_id)
    assert torch.equal(x, xr.long()) and torch.equal(y, yr.long())
    print(f"packed into a block of {n}: x {tuple(x.shape)}, y {tuple(y.shape)}; equal to finetune.pack: True")
    print("targets, one mark per position (P prompt or padding, A answer, E answer EOS):")
    marks = "".join("P" if t == -1 else ("E" if t == tok.eos_id else "A") for t in y[0].tolist())
    print("  ", marks)
    print(f"graded positions: {int((y != -1).sum())} of {y.numel()} "
          f"(= answer ids {sum(len(a) for _, a in pairs)})")
    for b in (n - 1, n - 2):
        x2, _ = pack(pairs, b, tok.eos_id)
        print(f"block of {b}: {x2.shape[0]} row(s) of {x2.shape[1]}")

    # 3. the loss is the mean over graded positions only
    cfg = AnuLMConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=3, hidden_size=32, n_head=4,
                      head_dim=8, n_kv_head=2, num_experts=8, num_experts_per_tok=2,
                      moe_intermediate_size=16, intermediate_size=64, n_group=1, topk_group=1)
    torch.manual_seed(0)
    model = AnuLM(cfg).eval()
    with torch.no_grad():
        logits, loss = model(x, y)
        keep = y != -1
        mine = F.cross_entropy(logits[keep].float(), y[keep])
    print(f"model loss {loss.item():.6f} = mean CE over the {int(keep.sum())} graded positions "
          f"{mine.item():.6f}")
    assert abs(loss.item() - mine.item()) < 1e-6

    # 4. no wall between examples in a row
    p2, a2 = pairs[1]
    with torch.no_grad():
        xa, ya = pack([pairs[1]], len(p2) + len(a2), tok.eos_id)
        _, alone = model(xa.long(), ya.long())
        packed_logits, _ = model(x, y)
        second = slice(len(pairs[0][0]) + len(pairs[0][1]), n)
        after = F.cross_entropy(packed_logits[0, second][y[0, second] != -1].float(),
                                y[0, second][y[0, second] != -1])
    print(f"second answer's loss alone {alone.item():.4f}, after the first example {after.item():.4f}: "
          f"differs by {abs(alone.item() - after.item()):.4f}")


if __name__ == "__main__":
    main()
