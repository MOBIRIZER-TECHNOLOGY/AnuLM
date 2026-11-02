"""
Book example, not repository code.

Chapter 28: what the released model's next-token distributions look like to
the sampling code in `AnuLM.generate`.

    python book/code-examples/ch28_measure_sampling.py            # writes ./part_vi_ch28_sampling.json
    python book/code-examples/ch28_measure_sampling.py --save     # the book's copy, book/measurements/part_vi_ch28_sampling.json

Runs `ckpt_base_v2.pt` through `model.py` unchanged over windows of the held-out
split `data/v2/val.u16`, under bf16 autocast as `serve.py` and `api.py` run it,
and at every position records:

- whether the two largest logits are exactly equal (bf16 has 8 significant bits);
- how many tokens survive `generate`'s top-k cut at k = 40 (ties can add more);
- the probability mass those 40 hold at temperature 1;
- how many tokens a top-p nucleus at p = 0.9 would hold (top-p is not in the repo);
- the mass off the most likely token at temperature 0.001, the floor `api.py`
  applies when a client asks for temperature 0.

The tie count is repeated in fp32 for contrast. Only statistics are saved; no
held-out text is printed. Results go to ./part_vi_ch28_sampling.json, or with
--save to book/measurements/part_vi_ch28_sampling.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# The book's copy is written only with --save; otherwise the result goes to the current directory,
# so a reader or reviewer re-running the script never overwrites the numbers the chapter quotes.
SAVE = "--save" in sys.argv[1:]
OUT = (ROOT / "book" / "measurements" if SAVE else Path.cwd()) / "part_vi_ch28_sampling.json"
SEED, WINDOWS, SEQ, BATCH, EOS = 0, 16, 512, 4, 32767


def load():
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False)
    m = M.AnuLM(ck["cfg"])
    m.load_state_dict(ck["model"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), dev


def set_impl(m, impl):
    for layer in m.layers:
        if layer.is_moe:
            layer.mlp.impl = impl


def nucleus_size(logits, p):
    """Tokens Chapter 28's top_p_filter keeps: the smallest set whose probability
    reaches p, plus ties with the last one kept, at least the top token."""
    z = logits.sort(-1, descending=True).values
    probs = z.softmax(-1)
    keep = probs.cumsum(-1) - probs < p
    keep[..., 0] = True
    cut = z.masked_fill(~keep, math.inf).min(-1, keepdim=True).values
    return (logits >= cut).sum(-1)


@torch.no_grad()
def collect(m, dev, bf16, windows):
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(val) - SEQ - 1, size=windows)
    out = {k: [] for k in ("tie", "survivors", "mass40", "nucleus90", "off_argmax")}
    for b in range(0, windows, BATCH):
        rows = np.stack([val[s: s + SEQ + 1] for s in starts[b: b + BATCH]]).astype(np.int64)
        x = torch.from_numpy(rows[:, :-1]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and dev == "cuda"):
            logits, _ = m(x, x)                                   # targets given: logits at every position
        z = logits.reshape(-1, logits.size(-1)).float()           # as generate does: .float() after the head
        top = torch.topk(z, 40, dim=-1).values
        out["tie"].append(top[:, 0] == top[:, 1])
        out["survivors"].append((z >= top[:, [-1]]).sum(-1))      # generate keeps every logit not below the 40th
        p = z.softmax(-1)
        out["mass40"].append(torch.where(z >= top[:, [-1]], p, 0).sum(-1))
        out["nucleus90"].append(nucleus_size(z, 0.9))
        out["off_argmax"].append(1 - (z / 1e-3).softmax(-1).max(-1).values)
    return {k: torch.cat(v).double().cpu().numpy() for k, v in out.items()}


def main():
    torch.manual_seed(SEED)
    m, dev = load()
    set_impl(m, "grouped" if dev == "cuda" else "sparse")
    r = collect(m, dev, True, WINDOWS)
    n = len(r["tie"])
    pct = lambda a, q: float(np.percentile(a, q))
    res = {"checkpoint": "ckpt_base_v2.pt", "data": "data/v2/val.u16", "seed": SEED, "windows": WINDOWS, "seq": SEQ,
           "positions": n, "precision": "bf16 autocast, moe_impl grouped" if dev == "cuda" else "fp32 cpu",
           "top2_tie_share": float(r["tie"].mean()),
           "topk40_more_than_40_share": float((r["survivors"] > 40).mean()),
           "topk40_max_survivors": int(r["survivors"].max()),
           "topk40_mass_T1": {"p10": pct(r["mass40"], 10), "median": pct(r["mass40"], 50), "p90": pct(r["mass40"], 90)},
           "nucleus90_size_T1": {"p10": pct(r["nucleus90"], 10), "median": pct(r["nucleus90"], 50),
                                 "p90": pct(r["nucleus90"], 90), "share_le_40": float((r["nucleus90"] <= 40).mean())},
           "T0001_off_argmax": {"mean": float(r["off_argmax"].mean()),
                                "share_above_1pct": float((r["off_argmax"] > 0.01).mean())}}
    histo, edges = np.histogram(np.log2(r["nucleus90"]), bins=np.arange(0, 16))
    res["nucleus90_log2_histogram"] = {"edges_log2": edges.tolist(), "counts": histo.tolist()}
    set_impl(m, "sparse")
    r32 = collect(m, dev, False, 8)
    res["fp32_precision"] = "fp32, no autocast, moe_impl sparse (the checkpoint's grouped path would run experts in bf16)"
    res["fp32_positions"] = len(r32["tie"])
    res["fp32_top2_tie_share"] = float(r32["tie"].mean())
    for k, v in res.items():
        print(f"{k}: {v}")
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
