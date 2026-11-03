"""
Book example, not repository code.

Chapter 24: what a val loss's "+/-" is worth, on a model small enough to train
on a CPU in seconds.

    python book/code-examples/ch24_error_bars.py

It builds an ordered two-register corpus (an easy repeating pattern, then
uniform random ids, as a multi-source corpus is built one source after
another), trains test_model.py's tiny() on it with train.py's own optimizer,
sampler and batches, and then measures the held-out loss three ways with
train.py's evaluate(), eval_windows() and strided_windows(), unchanged:

1. evaluate()'s mean and standard error, recomputed by hand from its batches;
2. 25 random window sets against one strided set, both scored against the
   mean over every window of the split;
3. two nearby checkpoints compared on the same windows (a paired difference)
   and on independent ones.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.training import train as T  # noqa: E402  the repository's train.py, unchanged
from anulm.model import AnuLM, AnuLMConfig  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def tiny() -> AnuLMConfig:
    """test_model.py's tiny(), copied so this script needs no test harness."""
    return AnuLMConfig(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8,
                       n_kv_head=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
                       intermediate_size=64, n_group=1, topk_group=1)


def corpus(n: int, easy_frac: float, g: torch.Generator) -> torch.Tensor:
    """An easy register (0, 1, ..., 15 repeated) followed by a hard one (uniform ids)."""
    n_easy = int(n * easy_frac)
    easy = torch.arange(n_easy) % 16
    hard = torch.randint(16, 64, (n - n_easy,), generator=g)
    return torch.cat([easy, hard])


def batch_losses(model, data, batches, args):
    """What evaluate() computes inside its loop: one loss per batch of windows."""
    model.eval()
    out = []
    with torch.no_grad():
        for ix in batches:
            x, y = T.make_batch(data, ix, args.block_size, "cpu")
            out.append(model(x, y)[1].item())
    model.train()
    return torch.tensor(out)


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    train_data, val = corpus(60_000, 0.6, g), corpus(20_000, 0.6, g)
    cfg = tiny()
    model = AnuLM(cfg)
    opt_args = SimpleNamespace(device="cpu", optimizer="adamw", lr=3e-3, weight_decay=0.1)
    opt = T.make_optimizer(model, opt_args)
    sampler = T.WindowSampler(len(train_data), cfg.block_size, 16, seed=0)
    snapshots = {}
    for step in range(150):
        x, y = T.make_batch(train_data, sampler.next(), cfg.block_size, "cpu")
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.update_expert_biases()
        if step + 1 in (140, 150):
            snapshots[step + 1] = copy.deepcopy(model)
    print(f"trained 150 steps; last training loss {loss.item():.3f}\n")

    # 1. evaluate(): the mean, and the standard error of the mean across batches
    args = argparse.Namespace(batch_size=8, block_size=cfg.block_size, device="cpu",
                              autocast=contextlib.nullcontext(), eval_iters=20, eval_windows=0)
    mean, sem = T.evaluate(model, val, args)
    b = batch_losses(model, val, T.eval_windows(val, 8, cfg.block_size, 20), args)
    by_hand = (((b - b.mean()) ** 2).mean() / len(b)).sqrt().item()
    print(f"1. evaluate(): {mean:.4f} +/- {sem:.4f}")
    print(f"   by hand from its 20 batch losses: mean {b.mean():.4f}, "
          f"sqrt(var / 20) = {by_hand:.4f}")
    assert abs(b.mean().item() - mean) < 1e-6 and abs(by_hand - sem) < 1e-6

    # 2. a reference (every non-overlapping window) against random and strided sets of 160
    n_all = (len(val) - 1) // cfg.block_size
    every = list((torch.arange(n_all) * cfg.block_size).split(8))
    truth = batch_losses(model, val, every, args)
    truth = ((truth * torch.tensor([len(i) for i in every])).sum() / n_all).item()
    rand = torch.tensor([batch_losses(model, val, T.eval_windows(val, 8, cfg.block_size, 20, seed=s), args).mean().item()
                         for s in range(25)])
    args.eval_windows = 160
    strided, strided_sem = T.evaluate(model, val, args)
    print(f"\n2. reference, all {n_all} non-overlapping windows: {truth:.4f}")
    print(f"   random, 160 windows, 25 seeds: mean {rand.mean():.4f}, sd {rand.std():.4f}, "
          f"worst error {(rand - truth).abs().max():.4f}")
    print(f"   strided, 160 windows: {strided:.4f} (+/- {strided_sem:.4f} as evaluate() reports it), "
          f"error {abs(strided - truth):.4f}")

    # 3. two checkpoints ten steps apart, on the same windows and on different ones
    args.eval_windows = 0
    w = T.eval_windows(val, 8, cfg.block_size, 20)
    a, c = batch_losses(snapshots[140], val, w, args), batch_losses(snapshots[150], val, w, args)
    d = c - a
    se = lambda v: (((v - v.mean()) ** 2).mean() / len(v)).sqrt().item()
    print(f"\n3. step 140: {a.mean():.4f} +/- {se(a):.4f}   step 150: {c.mean():.4f} +/- {se(c):.4f}")
    print(f"   difference on the same windows: {d.mean():+.4f} +/- {se(d):.4f} (paired)")
    print(f"   if the windows had been drawn independently: +/- {math.hypot(se(a), se(c)):.4f}")
    print(f"   the paired error is {math.hypot(se(a), se(c)) / se(d):.0f}x smaller")

    # bits per byte, as train.py prints it: nats -> bits, then per byte of text
    print(f"\nbits/byte of the step-150 loss at 1 byte per token: {mean / math.log(2) / 1.0:.4f}")
    answers()


def answers():
    """The chapter's exercise answers, from docs/RESULTS.md section 7 and the saved measurement."""
    print("\nexercise answers")
    for name, nats, bpt in (("byte model", 0.5975, 1.0), ("hi16k BPE model", 4.6795, 9.64)):
        print(f"  {name}: {nats} / ln 2 / {bpt} = {nats / math.log(2) / bpt:.3f} bits/byte")
    meas = ROOT / "book" / "measurements" / "part_v_b_val_noise.json"
    if meas.exists():
        import json
        r = json.loads(meas.read_text(encoding="utf-8"))
        se80 = r["random80"]["mean_sem"]
        print(f"  random windows for +/- 0.01, from +/- {se80:.4f} at 80: 80 x ({se80:.4f} / 0.01)^2 = "
              f"{80 * (se80 / 0.01) ** 2:,.0f}")
        en, wiki = r["per_source"]["english"], r["per_source"]["hindi_wiki"]
        d, se = wiki["loss"] - en["loss"], math.hypot(wiki["sem"], en["sem"])
        print(f"  Hindi Wikipedia minus English: {d:.3f} +/- {se:.3f}, {d / se:.1f} standard errors")
        mix_check(r)
    print(f"  the floor for the leaky toy of Chapter 26: ln 64 / 32 = {math.log(64) / 32:.3f}")


def mix_check(r):
    """Does the source mix of each random 80-window set explain its val loss on the released split?
    Recomputes each set's windows with eval_windows (only the split's length is needed) and predicts
    its mean from the per-source means measured with strided windows."""
    n, edges, per = r["val_tokens"], r["positions"]["edges"], r["per_source"]

    class Length:                                      # eval_windows reads only len(data)
        def __len__(self):
            return n

    obs, pred = [], []
    print("\nreleased split: source mix of each default random set, and the mean it predicts")
    for s in r["random80"]["sets"]:
        ix = [int(i) for b in T.eval_windows(Length(), 4, r["block"], 20, seed=s["seed"]) for i in b]
        cnt = {k: sum(a <= p < z for p in ix) for k, (a, z) in edges.items()}
        pred.append(sum(cnt[k] * per[k]["loss"] for k in cnt) / len(ix))
        obs.append(s["mean"])
        print(f"  seed {s['seed']}: windows {list(cnt.values())}  predicted {pred[-1]:.4f}  observed {obs[-1]:.4f}")
    o, p = torch.tensor(obs), torch.tensor(pred)
    print(f"  correlation {torch.corrcoef(torch.stack([o, p]))[0, 1]:.3f}; sd predicted from the mix "
          f"{p.std():.3f}, sd observed {o.std():.3f}")


if __name__ == "__main__":
    main()
