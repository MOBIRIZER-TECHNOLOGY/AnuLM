"""
Book example, not repository code.

Chapters 3, 5 and 7: statistics of the released model's next-token
predictions on its own held-out split.

    python book/code-examples/measure_part_i_ii.py        # GPU if free, else CPU (slow)

Runs `ckpt_base_v2.pt` through `model.py` unchanged over the same 64 windows of
512 tokens that Chapter 20 measures (`data/v2/val.u16`, numpy seed 0, the
target after an end-of-document id not graded), and records:

- Chapter 3: the distribution of per-token surprise, -log p(true next token);
  bits per token and bits per byte;
- Chapter 5: for each temperature T, the loss of softmax(z / T), the mean
  entropy of that distribution and the mean probability of its top token;
- Chapter 7: the total gradient norm on four batches shaped like the released
  run's (4 windows of 1,024 tokens, numpy seed 1, bf16 autocast, train mode),
  the quantity `clip_grad_norm_` compares with `--grad-clip`.

No held-out text is printed or saved: only statistics. Output:
part_i_ii_released.json in the current directory, or in book/measurements/ with --save.

Precision: fp32 with no autocast, except the experts. The checkpoint's
configuration has moe_impl='grouped', whose kernel runs the experts in bf16
(model.py lines 756 and 784), as in Chapter 20's measurement.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = ROOT / "book" / "measurements" / "part_i_ii_released.json"
WINDOWS, SEQ, BATCH, SEED, EOS = 64, 512, 4, 0, 32767
TEMPS = [0.5, 0.8, 0.9, 0.93, 0.95, 0.96, 0.97, 0.98, 1.0, 1.02, 1.05, 1.1, 1.25, 1.5, 2.0]


def main():
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = M.AnuLM(cfg)
    m.load_state_dict(ck["model"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev).eval()
    meta = json.loads((ROOT / "data" / "v2" / "meta.json").read_text())
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(val) - SEQ - 1, size=WINDOWS)       # Chapter 20's windows

    surprise, top1 = [], []
    ce = {t: 0.0 for t in TEMPS}
    ent = {t: 0.0 for t in TEMPS}
    pmax = {t: 0.0 for t in TEMPS}
    n_tok = 0
    batches = []
    with torch.no_grad():
        for b in range(0, WINDOWS, BATCH):
            rows = np.stack([val[s: s + SEQ + 1] for s in starts[b: b + BATCH]]).astype(np.int64)
            x = torch.from_numpy(rows[:, :-1]).to(dev)
            y = torch.from_numpy(rows[:, 1:]).to(dev).masked_fill(torch.from_numpy(rows[:, :-1]).to(dev) == EOS, -1)
            batches.append((x, y))
            logits, loss = m(x, y)
            keep = y != -1
            z = logits[keep].float()                                    # (n, V) graded positions
            t = y[keep]
            logp = F.log_softmax(z, -1)
            s = -logp.gather(1, t[:, None])[:, 0]
            surprise.append(s.cpu())
            top1.append((z.argmax(-1) == t).cpu())
            for T in TEMPS:
                lp = F.log_softmax(z / T, -1)
                ce[T] += -lp.gather(1, t[:, None]).sum().item()
                ent[T] += -(lp.exp() * lp).sum(-1).sum().item()
                pmax[T] += lp.max(-1).values.exp().sum().item()
            n_tok += int(keep.sum())
    s = torch.cat(surprise).double()
    mean = s.mean().item()
    q = torch.quantile(s.float(), torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])).tolist()
    lnV = math.log(cfg.vocab_size)
    res = {"checkpoint": "ckpt_base_v2.pt", "data": "data/v2/val.u16", "windows": WINDOWS, "seq": SEQ,
           "seed": SEED, "graded_tokens": n_tok, "precision": "fp32, no autocast, except the experts: the checkpoint's moe_impl='grouped' runs them in bf16 (model.py 756, 784), as in Chapter 20",
           "loss": mean, "ln_vocab": lnV, "perplexity": math.exp(mean),
           "quantiles": dict(zip(["p10", "p25", "p50", "p75", "p90"], q)),
           "frac_below_0.1": (s < 0.1).double().mean().item(),
           "frac_below_1": (s < 1.0).double().mean().item(),
           "frac_above_lnV": (s > lnV).double().mean().item(),
           "tail": {str(c): {"share_of_tokens": (s >= c).double().mean().item(),
                             "share_of_loss": (s[s >= c].sum() / s.sum()).item()} for c in (4, 6, 8)},
           "max_surprise": s.max().item(),
           "histogram_half_nat": torch.histc(s.float(), bins=36, min=0.0, max=18.0).tolist(),
           "top1_accuracy": torch.cat(top1).double().mean().item(),
           "bits_per_token": mean / math.log(2),
           "bytes_per_token_corpus": meta["bytes_per_token"],
           "bits_per_byte_corpus_avg": mean / math.log(2) / meta["bytes_per_token"],
           "temperature": {str(T): {"loss": ce[T] / n_tok, "entropy": ent[T] / n_tok, "top_prob": pmax[T] / n_tok}
                           for T in TEMPS}}

    # Chapter 7: the gradient norm clip_grad_norm_ would see at the end of training, on batches shaped like
    # the released run's (4 x 1,024 tokens, bf16 autocast, train mode so the router regularisers are on)
    m.train()
    rng2 = np.random.default_rng(SEED + 1)
    norms = []
    for b in range(4):
        st = rng2.integers(0, len(val) - 1024 - 1, size=4)
        rows = np.stack([val[s: s + 1025] for s in st]).astype(np.int64)
        x, y = torch.from_numpy(rows[:, :-1]).to(dev), torch.from_numpy(rows[:, 1:]).to(dev)
        m.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            _, loss = m(x, y)
        loss.backward()
        total = torch.nn.utils.clip_grad_norm_(m.parameters(), float("inf"))   # inf: measure, never scale
        norms.append(total.item())
    m.zero_grad(set_to_none=True)
    res["grad_norm_batches_4x1024_bf16"] = norms

    out = OUT if "--save" in sys.argv else Path(OUT.name)       # book/measurements/ only with --save
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    print(f"device {dev}; {n_tok:,} graded tokens")
    print(f"loss {mean:.4f} nats (ln V = {lnV:.4f}); perplexity {math.exp(mean):.1f}; top-1 accuracy "
          f"{res['top1_accuracy']:.3f}")
    print("surprise quantiles  " + "  ".join(f"{k} {v:.3f}" for k, v in res["quantiles"].items()))
    print(f"share of tokens with surprise < 0.1: {res['frac_below_0.1']:.3f}; < 1: {res['frac_below_1']:.3f}; "
          f"> ln V: {res['frac_above_lnV']:.4f}; max {res['max_surprise']:.2f}")
    print(f"bits/token {res['bits_per_token']:.3f}; bits/byte at the corpus average "
          f"{meta['bytes_per_token']} bytes/token: {res['bits_per_byte_corpus_avg']:.4f}")
    for c, v in res["tail"].items():
        print(f"tokens with surprise >= {c} nats: {v['share_of_tokens']:.3f} of tokens, {v['share_of_loss']:.3f} of the loss")
    print("T      loss    entropy  top-prob")
    for T in TEMPS:
        r = res["temperature"][str(T)]
        print(f"{T:<5}  {r['loss']:.4f}  {r['entropy']:.4f}   {r['top_prob']:.4f}")
    print("gradient norm per batch of 4 x 1024 (train mode, bf16 autocast):", " ".join(f"{v:.3f}" for v in norms))
    assert abs(mean - 3.3790) < 2e-3, "should reproduce Chapter 20's 3.379 on the same windows"


if __name__ == "__main__":
    main()
