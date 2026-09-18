"""
Supervised fine-tuning on question-answer pairs (and translation pairs).

    python finetune.py --ckpt ckpt_hindi_mixed18k.pt --qa data/qa_hindi.jsonl \
                       --out ckpt_hindi_qa.pt --epochs 2
    python finetune.py --ckpt ckpt_multi36k.pt --qa data/tr_pairs.jsonl \
                       --out ckpt_translate.pt --stop-at 20000      # then --resume

Takes a pretrained checkpoint and its tokenizer, formats every pair with the
template for its `lang` (make_qa.PROMPTS: Hindi and English questions,
Python, and the two translation directions), for example

    प्रश्न: {question}\\nउत्तर: {answer}<EOS>
    English: {sentence}\\nHindi: {translation}<EOS>

and trains with the loss on the *answer tokens only* -- the prompt is
masked to -1, which `AnuLM.forward` already ignores. Examples are
packed into block_size rows (an example never straddles two rows), so a
step sees several pairs and no padding is wasted. Everything else is
`train.py`'s machinery: the same optimizer split, cosine schedule, bf16
autocast, grouped-GEMM dispatch, gradient checkpointing, and the
aux-loss-free bias update after every step so routing keeps adapting.

Long runs (millions of pairs) are the reason for three more things:
  * encoding runs in a process pool and the packed rows are cached next to
    the jsonl, so a rerun starts training in seconds;
  * `--stop-at N` ends the invocation at step N after an eval and a save of
    `<out>.last` (weights, optimizer, data order), and `--resume` continues
    from it -- so the machine may sleep between phases;
  * `<out>` (the best held-out checkpoint) carries `qa_templates`, which
    serve.py, eval_qa.py and eval_translate.py read to know how to ask.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from array import array
from dataclasses import replace
from pathlib import Path

import torch

import bpe
from bpe import BPE, _encode_init
from make_qa import PROMPT, PROMPTS
from model import AnuLM
from train import OptimizerSet, atomic_save, lr_at, make_optimizer


def _pair_ids(tok: BPE, it: dict) -> tuple[list[int], list[int]]:
    """(prompt ids, answer ids + EOS). The answer is encoded with its leading
    space attached, the way the BPE chunks text, so prompt + answer tokenises
    identically to the joined string."""
    template = PROMPTS.get(it.get("lang", "hi"), PROMPT)
    return tok.encode(template.format(q=it["question"])), tok.encode(" " + it["answer"]) + [tok.eos_id]


def encode_pairs(tok: BPE, items: list[dict]) -> list[tuple[list[int], list[int]]]:
    return [_pair_ids(tok, it) for it in items]


def _encode_chunk(items):
    return [_pair_ids(bpe._worker_tok, it) for it in items]


def encode_pairs_parallel(tokenizer_path: str, items: list[dict], workers: int):
    if workers <= 1 or len(items) < 20000:
        return encode_pairs(BPE.load(tokenizer_path), items)
    from multiprocessing import Pool
    chunks = [items[i: i + 5000] for i in range(0, len(items), 5000)]
    out = []
    with Pool(workers, initializer=_encode_init, initargs=(tokenizer_path,)) as pool:
        for i, part in enumerate(pool.imap(_encode_chunk, chunks)):
            out.extend(part)
            if i % 20 == 0:
                print(f"\r  encoded {len(out)}/{len(items)} pairs ({workers} workers)", end="", flush=True)
    print()
    return out


def pack(pairs, block_size: int, eos: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of block_size + 1 ids and matching targets with the prompt masked.
    Position t's target is id t+1, so a row of L ids yields L-1 predictions;
    the extra id makes the standard x = row[:-1], y = row[1:] split work.
    Built in flat int32 arrays: a few million pairs as Python lists of ints
    would be tens of GB."""
    L = block_size + 1
    rows, tgts = array("i"), array("i")
    ids, lab = [], []

    def flush():
        rows.extend(ids + [eos] * (L - len(ids)))
        tgts.extend(lab + [-1] * (L - len(lab)))

    for p, a in pairs:
        ex = p + a
        if len(ex) > block_size:
            continue
        if len(ids) + len(ex) > L:
            flush()
            ids, lab = [], []
        ids += ex
        lab += [-1] * len(p) + a
    if ids:
        flush()
    x = torch.frombuffer(rows, dtype=torch.int32).clone().view(-1, L) if len(rows) else torch.zeros(0, L, dtype=torch.int32)
    y = torch.frombuffer(tgts, dtype=torch.int32).clone().view(-1, L) if len(tgts) else torch.zeros(0, L, dtype=torch.int32)
    return x[:, :-1].contiguous(), y[:, 1:].contiguous()


def load_packed(path: str, tokenizer_path: str, block_size: int, eos: int, workers: int):
    """Packed rows for a jsonl, cached beside it as <jsonl>.<tokenizer>.b<block>.pt."""
    # split("\n"), not splitlines(): Python also splits on \x0b \x0c \x1c-\x1e
    # \x85 \u2028 \u2029, and json.dumps(ensure_ascii=False) leaves those raw
    # inside strings. One U+2028 in 1.38M code pairs split a valid JSON line in
    # two and failed the entire load with "Unterminated string".
    items = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]
    cache = Path(f"{path}.{Path(tokenizer_path).stem}.b{block_size}.pt")
    if cache.exists():
        c = torch.load(cache, weights_only=False)
        if c.get("n_items") == len(items):
            return c["x"], c["y"]
    t0 = time.time()
    x, y = pack(encode_pairs_parallel(tokenizer_path, items, workers), block_size, eos)
    torch.save({"x": x, "y": y, "n_items": len(items)}, cache)
    print(f"  packed {len(items)} pairs -> {len(x)} rows in {time.time() - t0:.0f}s (cached at {cache.name})")
    return x, y


@torch.no_grad()
def evaluate(model, x, y, batch_size, device, autocast) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i: i + batch_size].long().to(device), y[i: i + batch_size].long().to(device)
        with autocast:
            _, loss = model(xb, yb)
        k = int((yb != -1).sum())
        tot += loss.item() * k
        n += k
    model.train()
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt_hindi_mixed18k.pt")
    p.add_argument("--qa", default="data/qa_hindi.jsonl")
    p.add_argument("--heldout", default="data/qa_hindi_heldout.jsonl")
    p.add_argument("--out", default="ckpt_hindi_qa.pt")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=5e-6)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--moe-impl", default=None, help="default: grouped on cuda")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    p.add_argument("--muon-lr", type=float, default=0.005)
    p.add_argument("--workers", type=int, default=12, help="encoder processes for large pair sets")
    p.add_argument("--resume", action="store_true", help="continue from <out>.last")
    p.add_argument("--stop-at", type=int, default=None,
                   help="end this invocation at this step after an eval and a <out>.last save")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    assert getattr(cfg, "tokenizer_path", None), "finetune.py needs a BPE checkpoint"
    if args.moe_impl or args.device.startswith("cuda"):
        cfg = replace(cfg, moe_impl=args.moe_impl or "grouped")
    tok = BPE.load(cfg.tokenizer_path)
    model = AnuLM(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    if args.grad_ckpt:
        model.enable_gradient_checkpointing()

    x, y = load_packed(args.qa, cfg.tokenizer_path, cfg.block_size, tok.eos_id, args.workers)
    hx, hy = load_packed(args.heldout, cfg.tokenizer_path, cfg.block_size, tok.eos_id, args.workers)
    n_ans = int((y != -1).sum())
    print(f"train: {len(x)} rows of {cfg.block_size}, {n_ans/1e6:.2f}M answer tokens "
          f"({100 * n_ans / max(y.numel(), 1):.0f}% of positions)  |  held-out: {len(hx)} rows")

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    args.autocast = (torch.autocast("cuda", dtype=dtype) if args.device.startswith("cuda")
                     else torch.autocast("cpu", enabled=False))
    args.steps = int(math.ceil(len(x) / args.batch_size * args.epochs))
    opt: OptimizerSet = make_optimizer(model, args)
    print(f"{args.steps} steps = {args.epochs} epochs at batch {args.batch_size}, lr {args.lr}")

    # --- resume -------------------------------------------------------------
    last_path = Path(str(args.out) + ".last")
    g = torch.Generator().manual_seed(args.seed)
    order, pos = torch.randperm(len(x), generator=g), 0
    start_step, best, base = 0, float("inf"), None
    if args.resume and last_path.exists():
        r = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(r["model"])
        opt.load_state_dict(r["opt"])
        start_step, best, base = r["step"] + 1, r["best"], r.get("base")
        order, pos = r["order"], r["pos"]
        g.set_state(r["g"])
        print(f"resuming from {last_path} at step {start_step} (best held-out {best:.4f})")
        del r
    else:
        if args.resume:
            print(f"--resume given but {last_path} does not exist; starting fresh")
        base = evaluate(model, hx, hy, args.batch_size, args.device, args.autocast)
        print(f"held-out answer loss before fine-tuning: {base:.4f}")

    def save_last(step):
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "base": base, "order": order, "pos": pos, "g": g.get_state(),
                    "base_ckpt": args.ckpt}, last_path)

    end = min(args.steps, args.stop_at) if args.stop_at else args.steps
    t0 = time.time()
    model.train()
    for step in range(start_step, end):
        if pos + args.batch_size > len(order):
            order, pos = torch.randperm(len(x), generator=g), 0
        ix = order[pos: pos + args.batch_size]
        pos += args.batch_size
        lr = lr_at(step, args)
        opt.set_lr_scale(lr / args.lr)
        xb, yb = x[ix].long().to(args.device), y[ix].long().to(args.device)
        with args.autocast:
            _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        imbalance = model.update_expert_biases()
        if step % args.log_every == 0 or step == end - 1:
            print(f"step {step:6d} | loss {loss.item():6.4f} | lr {lr:.2e} | "
                  f"imbalance {imbalance:4.2f}x | {time.time() - t0:5.0f}s")
        if (step + 1) % args.eval_every == 0 or step == end - 1:
            val = evaluate(model, hx, hy, args.batch_size, args.device, args.autocast)
            flag = ""
            if val < best:
                best = val
                atomic_save({"model": model.state_dict(), "cfg": replace(cfg, moe_impl=ck["cfg"].moe_impl),
                            "step": step, "val_loss": val, "qa_template": PROMPT,
                            "qa_templates": PROMPTS, "base_ckpt": args.ckpt}, args.out)
                flag = "  <- saved"
            save_last(step)
            print(f"  eval @ {step:6d} | held-out answer loss {val:.4f}{flag}")
    if end < args.steps:
        print(f"\nphase done at step {end} of {args.steps} in {time.time() - t0:.0f}s | best held-out {best:.4f}"
              f" | resume point {last_path}")
        return
    print(f"\ndone in {time.time() - t0:.0f}s | held-out answer loss "
          f"{base if base is not None else float('nan'):.4f} -> {best:.4f} | {args.out}")


if __name__ == "__main__":
    main()
