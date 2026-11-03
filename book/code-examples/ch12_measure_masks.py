"""
Book example, not repository code.

Chapter 12: what the masks do to the released model, measured.

    python book/code-examples/ch12_measure_masks.py     # GPU if free, else CPU (slow); add --save to write book/measurements/

Runs `ckpt_base_v2.pt` through `model.py` unchanged over 32 windows of 1,024
tokens from the base run's held-out split (`data/v2/val.u16`), drawn with seed
0, three ways:

- as trained: causal everywhere, a 256-token window on layers 0–9;
- with the window removed (every layer's `window` set to None);
- with no mask at all (`_attn_mask` swapped for an all-True mask), so every
  position sees the future.

It records the loss over all graded positions and over positions 0–255 and
256–1,023 separately. Precision: fp32 with no autocast, except the experts, which the released
configuration runs in bf16 on its grouped path (`model.py` 756, 784).

Only statistics are saved; no held-out text is printed.
With --save, results go to book/measurements/part_iv_a_masks.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Writes to book/measurements/ only with --save; otherwise to the current directory.
OUT = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_iv_a_masks.json"
WINDOWS, SEQ, BATCH, SEED, EOS = 32, 1024, 4, 0, 32767


def load():
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False)
    m = M.AnuLM(ck["cfg"])
    m.load_state_dict(ck["model"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), ck["cfg"], dev


def batches(dev):
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    starts = np.random.default_rng(SEED).integers(0, len(val) - SEQ - 1, size=WINDOWS)
    for b in range(0, WINDOWS, BATCH):
        rows = np.stack([val[s: s + SEQ + 1] for s in starts[b: b + BATCH]]).astype(np.int64)
        x = torch.from_numpy(rows[:, :-1]).to(dev)
        y = torch.from_numpy(rows[:, 1:]).to(dev).masked_fill(x == EOS, -1)   # not graded after a document end
        yield x, y


def losses(m, dev):
    """Summed loss and graded-token count for positions 0–255 and 256–1023."""
    tot = torch.zeros(2, dtype=torch.float64)
    n = torch.zeros(2, dtype=torch.float64)
    with torch.no_grad():
        for x, y in batches(dev):
            logits, _ = m(x, y)                                                  # (B, S, V)
            nll = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-1, reduction="none")
            ok = (y != -1)
            for i, sl in enumerate((slice(0, 256), slice(256, SEQ))):
                tot[i] += nll[:, sl][ok[:, sl]].double().sum().cpu()
                n[i] += ok[:, sl].sum().cpu()
    return {"all": (tot.sum() / n.sum()).item(), "pos_0_255": (tot[0] / n[0]).item(),
            "pos_256_1023": (tot[1] / n[1]).item(), "graded": int(n.sum())}


def main() -> None:
    m, cfg, dev = load()
    res = {"checkpoint": "ckpt_base_v2.pt", "data": "data/v2/val.u16", "windows": WINDOWS, "seq": SEQ,
           "seed": SEED, "precision": "fp32, no autocast, except the experts: bf16 on the released grouped path (model.py 756, 784)", "device": dev}
    res["as_trained"] = losses(m, dev)
    windows = [layer.window for layer in m.layers]
    for layer in m.layers:
        layer.window = None
    res["no_window"] = losses(m, dev)
    for layer, w in zip(m.layers, windows):
        layer.window = w
    real = M._attn_mask
    M._attn_mask = lambda S, T, start, window, device: torch.ones(S, T, dtype=torch.bool, device=device)
    try:
        res["no_mask"] = losses(m, dev)
    finally:
        M._attn_mask = real
    print(f"device {dev}; {WINDOWS} windows of {SEQ} tokens, seed {SEED}; "
          f"{res['as_trained']['graded']:,} graded targets")
    for k in ("as_trained", "no_window", "no_mask"):
        r = res[k]
        print(f"  {k:10s}: loss {r['all']:.3f}  (positions 0–255 {r['pos_0_255']:.3f}, "
              f"256–1023 {r['pos_256_1023']:.3f})")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
