"""
Book example, not repository code.

Chapter 29: the released model, `ckpt_base_v2.pt`, past the 1,024 tokens it
was trained on, with plain RoPE and with YaRN, and with its sliding window
switched off.

    python book/code-examples/ch29_measure_context.py            # writes ./part_vi_ch29_context.json
    python book/code-examples/ch29_measure_context.py --save     # the book's copy, book/measurements/part_vi_ch29_context.json

The same weights are loaded into configurations that differ only in
`block_size`, `yarn`, `yarn_factor` and `sliding_window`, the way
`eval_context.py` does it: no retraining. Windows come from the held-out split
`data/v2/val.u16`, the same starting offsets at every length, so the settings
are compared on the same tokens (a paired comparison). Windows cross document
boundaries, so this measures whether the model stays sound at positions it was
never trained on, not how well it uses a long document. Targets that follow an
end-of-document token are not graded. Only statistics are saved.
Results go to ./part_vi_ch29_context.json, or with --save to book/measurements/part_vi_ch29_context.json.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# The book's copy is written only with --save; otherwise the result goes to the current directory,
# so a reader or reviewer re-running the script never overwrites the numbers the chapter quotes.
SAVE = "--save" in sys.argv[1:]
OUT = (ROOT / "book" / "measurements" if SAVE else Path.cwd()) / "part_vi_ch29_context.json"
SEED, WINDOWS, EOS = 0, 16, 32767


@torch.no_grad()
def per_window(state, cfg, rows, dev):
    """Mean loss per window and per quarter of the window: (n, 4)."""
    m = M.AnuLM(cfg)
    m.load_state_dict(state, strict=True)
    m = m.to(dev).eval()
    for layer in m.layers:
        if layer.is_moe:
            layer.mlp.impl = "grouped" if dev == "cuda" else "sparse"
    out = []
    for r in rows:
        x = r[None, :-1].to(dev)
        y = r[None, 1:].to(dev).masked_fill(x == EOS, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            logits, _ = m(x, y)
        nll = F.cross_entropy(logits[0].float(), y[0], ignore_index=-1, reduction="none")
        ok = (y[0] != -1).float()
        n, q = len(nll), []
        for j in range(4):
            sl = slice(j * n // 4, (j + 1) * n // 4)
            q.append((nll[sl] * ok[sl]).sum() / ok[sl].sum())
        out.append(torch.stack(q).cpu())
    del m
    if dev == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(out).double()                              # (n, 4)


def main():
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False)
    base, state = ck["cfg"], ck["model"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    trained = base.block_size
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(val) - 4 * trained - 1, size=WINDOWS)
    rows = lambda L: [torch.from_numpy(val[s: s + L + 1].astype(np.int64)) for s in starts]
    print(f"device {dev}; trained at {trained}; window {base.sliding_window} on layers 0-{base.max_window_layers - 1}")

    settings = {
        "1024 trained": replace(base),
        "1024 window off": replace(base, sliding_window=None),
        "2048 naive": replace(base, block_size=2 * trained, yarn=False),
        "2048 yarn": replace(base, block_size=2 * trained, yarn=True,
                             yarn_factor=2.0, yarn_original_context=trained),
        "4096 naive": replace(base, block_size=4 * trained, yarn=False),
        "4096 yarn": replace(base, block_size=4 * trained, yarn=True,
                             yarn_factor=4.0, yarn_original_context=trained),
    }
    res = {"checkpoint": "ckpt_base_v2.pt", "data": "data/v2/val.u16", "seed": SEED, "windows": WINDOWS,
           "precision": "bf16 autocast" if dev == "cuda" else "fp32", "settings": {}}
    per = {}
    for name, cfg in settings.items():
        L = cfg.block_size
        w = per_window(state, cfg, rows(L), dev)
        per[name] = w
        mean = w.mean(1)
        res["settings"][name] = {"length": L, "yarn": cfg.yarn, "yarn_factor": cfg.yarn_factor if cfg.yarn else None,
                                 "window": cfg.sliding_window, "mean": float(mean.mean()),
                                 "se": float(mean.std(unbiased=True) / math.sqrt(len(mean))),
                                 "quarters": [float(v) for v in w.mean(0)],
                                 "last_minus_first": float((w[:, 3] - w[:, 0]).mean())}
        s = res["settings"][name]
        print(f"{name:16s} mean {s['mean']:.4f} ± {s['se']:.4f}  quarters "
              + "  ".join(f"{v:.4f}" for v in s["quarters"]) + f"  last-first {s['last_minus_first']:+.4f}")
    for L in (2048, 4096):
        d = per[f"{L} yarn"].mean(1) - per[f"{L} naive"].mean(1)
        res[f"yarn_minus_naive_{L}"] = {"mean": float(d.mean()), "se": float(d.std(unbiased=True) / math.sqrt(len(d)))}
        print(f"YaRN - naive at {L}: {d.mean():+.4f} ± {d.std(unbiased=True) / math.sqrt(len(d)):.4f} (paired)")
        q = per[f"{L} yarn"][:, 0] - per[f"{L} naive"][:, 0]     # first quarter: inside the trained range
        res[f"yarn_minus_naive_first_quarter_{L}"] = {"mean": float(q.mean()),
                                                      "se": float(q.std(unbiased=True) / math.sqrt(len(q)))}
        print(f"YaRN - naive at {L}, first quarter: {q.mean():+.4f} ± "
              f"{q.std(unbiased=True) / math.sqrt(len(q)):.4f} (paired)")
    d = per["1024 window off"].mean(1) - per["1024 trained"].mean(1)
    res["window_off_minus_on_1024"] = {"mean": float(d.mean()), "se": float(d.std(unbiased=True) / math.sqrt(len(d)))}
    print(f"window off - on at 1024: {d.mean():+.4f} ± {d.std(unbiased=True) / math.sqrt(len(d)):.4f} (paired)")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
