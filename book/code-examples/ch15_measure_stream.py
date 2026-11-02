"""
Book example, not repository code.

Chapter 15: the released model's residual stream, measured.

    python book/code-examples/ch15_measure_stream.py     # GPU if free, else CPU (slow); add --save to write book/measurements/

Runs `ckpt_base_v2.pt` through `model.py` unchanged over 16 windows of 512
tokens from the held-out split (`data/v2/val.u16`), seed 0, and records with
forward hooks, for every layer:

- the size of the residual stream entering the layer: each token's RMS over
  the 1,024 dimensions, averaged over tokens (position 0 reported apart);
- the size of what the attention half and the feed-forward half add to it;
- the mean absolute value of each RMSNorm's learned gain.

Precision: fp32 with no autocast, except the experts, which the released
configuration runs in bf16 on its grouped path (`model.py` 716, 743).

Only statistics are saved; no held-out text is printed.
With --save, results go to book/measurements/part_iv_a_stream.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Writes to book/measurements/ only with --save; otherwise to the current directory.
OUT = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_iv_a_stream.json"
WINDOWS, SEQ, BATCH, SEED = 16, 512, 4, 0


def rms(t):
    """Per-token RMS over the last dimension: (B, S, D) -> (B, S)."""
    return t.float().pow(2).mean(-1).sqrt()


def main() -> None:
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = M.AnuLM(cfg)
    m.load_state_dict(ck["model"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev).eval()
    L = cfg.n_layer
    names = ["stream", "attn", "mlp"]
    acc = {k: torch.zeros(L + 1, 2, dtype=torch.float64) for k in names}  # [rest of positions, position 0]

    pos0 = [[] for _ in range(L + 1)]                                     # stream at position 0, per window

    def add(key, layer, t):
        r = rms(t)
        acc[key][layer, 0] += r[:, 1:].mean().item()
        acc[key][layer, 1] += r[:, 0].mean().item()
        if key == "stream":
            pos0[layer] += r[:, 0].tolist()

    hooks = []
    for i, block in enumerate(m.layers):
        hooks.append(block.register_forward_pre_hook(lambda mod, args, i=i: add("stream", i, args[0])))
        hooks.append(block.attn.register_forward_hook(lambda mod, args, out, i=i: add("attn", i, out)))
        hooks.append(block.mlp.register_forward_hook(
            lambda mod, args, out, i=i: add("mlp", i, out[0] if isinstance(out, tuple) else out)))
    hooks.append(m.norm.register_forward_pre_hook(lambda mod, args: add("stream", L, args[0])))
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    starts = np.random.default_rng(SEED).integers(0, len(val) - SEQ - 1, size=WINDOWS)
    with torch.no_grad():
        for b in range(0, WINDOWS, BATCH):
            rows = np.stack([val[s: s + SEQ] for s in starts[b: b + BATCH]]).astype(np.int64)
            m(torch.from_numpy(rows).to(dev))
    for h in hooks:
        h.remove()
    n = WINDOWS // BATCH
    res = {"checkpoint": "ckpt_base_v2.pt", "data": "data/v2/val.u16", "windows": WINDOWS, "seq": SEQ, "seed": SEED,
           "precision": "fp32, no autocast, except the experts: bf16 on the released grouped path (model.py 716, 743)", "device": dev,
           "note": "index L (=20) of 'stream' is the input to the final norm"}
    for k in names:
        res[k] = (acc[k][:, 0] / n).tolist()
        res[k + "_pos0"] = (acc[k][:, 1] / n).tolist()
    res["stream_pos0_min"] = [min(v) for v in pos0]
    res["stream_pos0_max"] = [max(v) for v in pos0]
    res["gain_input"] = [b.input_layernorm.weight.abs().mean().item() for b in m.layers]
    res["gain_post_attn"] = [b.post_attention_layernorm.weight.abs().mean().item() for b in m.layers]
    res["gain_final"] = m.norm.weight.abs().mean().item()
    print(f"device {dev}; {WINDOWS} windows of {SEQ}, seed {SEED}; RMS per token, positions 1–511 [position 0]")
    for i in range(L):
        print(f"  layer {i:2d}: stream in {res['stream'][i]:7.3f} [{res['stream_pos0'][i]:8.2f}]  "
              f"attn adds {res['attn'][i]:6.3f} [{res['attn_pos0'][i]:6.2f}]  "
              f"mlp adds {res['mlp'][i]:6.3f} [{res['mlp_pos0'][i]:6.2f}]  "
              f"mean |gain| {res['gain_input'][i]:.4f}, {res['gain_post_attn'][i]:.3f}")
    print("  position 0, stream entering layers 8–19, smallest and largest window: "
          + "; ".join(f"L{i} {res['stream_pos0_min'][i]:.1f}–{res['stream_pos0_max'][i]:.1f}" for i in range(8, L)))
    print(f"  final norm input {res['stream'][L]:.3f} [{res['stream_pos0'][L]:.2f}]; final mean |gain| {res['gain_final']:.3f}")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
