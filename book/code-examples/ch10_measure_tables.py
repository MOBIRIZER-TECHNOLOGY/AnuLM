"""
Book example, not repository code.

Chapter 10: what training did to the released model's two token tables.
Reads embed_tokens.weight and lm_head.weight from ckpt_base_v2.pt and counts
how often each id occurs in the held-out split data/v2/val.u16 (counts only:
no held-out text is decoded, printed or saved). Prints vocabulary entries,
which are part of the released tokenizer, never text from the split.

    python book/code-examples/ch10_measure_tables.py   # -> ./part_iii_tables.json
    python book/code-examples/ch10_measure_tables.py --save   # -> book/measurements/part_iii_tables.json
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
from anulm.bpe import BPE  # noqa: E402  the repository's code, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# --save writes the book's copy; otherwise the result goes to the current directory.
OUT = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_iii_tables.json"
BUCKETS = [(0, 0, "0"), (1, 9, "1-9"), (10, 999, "10-999"), (1000, 99999, "1k-100k"), (100000, 10**12, "100k+")]
PROBES = [" dog", " India", " भारत", "def", " the", " 1"]


def main() -> None:
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["cfg"]
    E = ck["model"]["embed_tokens.weight"].float()             # (V, D)
    H = ck["model"]["lm_head.weight"].float()                  # (V, D)
    V, D = E.shape
    print(f"ckpt_base_v2.pt step {ck['step']:,}: V={V}, D={D}, tie_word_embeddings={cfg.tie_word_embeddings}, "
          f"merge_tree={cfg.merge_tree!r}, initializer_range={cfg.initializer_range}")
    tok = BPE.load(ROOT / cfg.tokenizer_path)

    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    counts = np.bincount(np.asarray(val), minlength=V)
    print(f"id counts over all {len(val):,} held-out tokens; {int((counts == 0).sum()):,} ids never occur")

    init = cfg.initializer_range * math.sqrt(D)
    nE, nH = E.norm(dim=1), H.norm(dim=1)
    cos_same = F.cosine_similarity(E, H, dim=1)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(V, generator=g)
    cos_rand = F.cosine_similarity(E, H[perm], dim=1)
    res = {"checkpoint": "ckpt_base_v2.pt", "step": ck["step"], "V": V, "D": D,
           "held_out_tokens": int(len(val)), "ids_never_in_held_out": int((counts == 0).sum()),
           "init_row_norm": round(init, 4), "buckets": [],
           "cos_in_out_same_token_mean": round(cos_same.mean().item(), 4),
           "cos_in_out_random_pair_mean": round(cos_rand.mean().item(), 4)}
    print(f"row norm at initialisation: 0.02 x sqrt({D}) = {init:.3f}")
    print("held-out count   ids     input norm  output norm  cos(in, out) same token")
    c = torch.from_numpy(counts.astype(np.int64))
    for lo, hi, name in BUCKETS:
        m = (c >= lo) & (c <= hi)
        row = {"bucket": name, "ids": int(m.sum()), "in_norm": round(nE[m].mean().item(), 3),
               "out_norm": round(nH[m].mean().item(), 3), "cos_same": round(cos_same[m].mean().item(), 3)}
        res["buckets"].append(row)
        print(f"{name:12} {row['ids']:8,}   {row['in_norm']:9.3f}  {row['out_norm']:11.3f}  {row['cos_same']:10.3f}")
    print(f"cos(in, out): same token {res['cos_in_out_same_token_mean']:.3f}, "
          f"random pairs {res['cos_in_out_random_pair_mean']:.3f}")
    # Same token, two tables: are the neighbourhoods alike even if the directions are not?
    common = torch.nonzero(c >= 1000).flatten()
    pick = common[torch.randperm(len(common), generator=g)[:2000]]
    Hn = F.normalize(H, dim=1)
    En_ = F.normalize(E, dim=1)
    overlap = 0
    for i in pick.tolist():
        a_ = En_ @ En_[i]
        b_ = Hn @ Hn[i]
        a_[i] = b_[i] = -2
        overlap += len(set(a_.topk(10).indices.tolist()) & set(b_.topk(10).indices.tolist()))
    res["top10_neighbour_overlap"] = round(overlap / (10 * len(pick)), 4)
    res["top10_overlap_chance"] = round(10 / (V - 1), 6)
    print(f"top-10 neighbours, input table vs output table, {len(pick)} ids seen >= 1,000 times: "
          f"overlap {100 * overlap / (10 * len(pick)):.1f}% (chance {100 * 10 / (V - 1):.3f}%)")
    eos = tok.eos_id
    res["eos"] = {"id": eos, "count": int(counts[eos]), "in_norm": round(nE[eos].item(), 3),
                  "out_norm": round(nH[eos].item(), 3)}
    print(f"EOS {eos}: count {counts[eos]:,}, input norm {nE[eos]:.3f}, output norm {nH[eos]:.3f}")

    En = F.normalize(E, dim=1)
    res["neighbours"] = {}
    for s in PROBES:
        ids = tok.encode(s)
        assert len(ids) == 1, (s, ids)
        sims = En @ En[ids[0]]
        sims[ids[0]] = -2
        top = sims.topk(5)
        nb = [(tok.decode([i]), round(v, 3)) for v, i in zip(top.values.tolist(), top.indices.tolist())]
        res["neighbours"][s] = nb
        print(f"{s!r:9} id {ids[0]:5}: " + ", ".join(f"{t!r} {v:.2f}" for t, v in nb))
    import warnings
    from anulm import model as M                                          # the repository's model.py, unchanged
    warnings.filterwarnings("ignore", message="Sparse invariant checks")
    P = M.merge_tree_matrix(tok.merges, V)
    res["merge_tree_nnz_multi32k"] = P._nnz()
    print(f"merge_tree_matrix for multi32k: {P._nnz():,} nonzeros, {P._nnz() / V:.1f} per token")
    # Weight decay in the released run: what the stored optimizer says, and what decay alone would do.
    last = torch.load(ROOT / "ckpt_base_v2.pt.last", map_location="cpu", weights_only=False, mmap=True)["opt"]
    last = last[0] if isinstance(last, list) else last
    groups = [{"weight_decay": g["weight_decay"], "tensors": len(g["params"]), "fused": g.get("fused")}
              for g in last["param_groups"]]
    res["optimizer_groups"] = groups
    print("ckpt_base_v2.pt.last optimizer groups:", groups)
    from types import SimpleNamespace
    from anulm.training.train import lr_at
    run = SimpleNamespace(lr=6e-4, min_lr=6e-5, warmup=2000, steps=1900000, schedule="wsd",
                          decay_frac=0.1631579)               # the flags in _v2_run.cmd / experiments/v2_train.sh
    wd = groups[0]["weight_decay"]
    log_keep = sum(math.log1p(-lr_at(s, run) * wd) for s in range(run.steps))
    res["decay_only_log10_factor"] = round(log_keep / math.log(10), 1)
    print(f"decay alone over {run.steps:,} steps multiplies a row by 10^{log_keep / math.log(10):.1f}: "
          f"0.640 -> {init * math.exp(log_keep):.1e}")

    # The first loss of the released configuration at random initialisation.
    sys.path.insert(0, str(ROOT / "book" / "code-examples"))
    from trace_forward import released
    torch.manual_seed(0)
    m = M.AnuLM(released()).eval()
    x = torch.randint(0, V, (4, 256))
    with torch.no_grad():
        logits, loss = m(x, x)
    res["released_init_loss"] = round(loss.item(), 3)
    res["released_init_logit_std"] = round(logits.std().item(), 3)
    print(f"released config at random init (seed 0, 4 x 256 random ids): loss {loss.item():.3f}, "
          f"logit std {logits.std().item():.3f}, ln V {math.log(V):.3f}")
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
