"""
Book example, not repository code.

Chapter 25 measurement: what the released run's two checkpoint files hold.

    python book/code-examples/ch25_measure_resume_point.py      # CPU, a few seconds

ckpt_base_v2.pt is the best-val file train.py writes to --out, and
ckpt_base_v2.pt.last is the resume point it writes beside it at every
evaluation. Both are opened memory-mapped, so no tensor is read into RAM;
the script only adds up sizes and reads the small metadata fields.
With --save the results go to book/measurements/part_v_b_resume_point.json;
without it, to the current directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))          # the pickled AnuLMConfig needs model.py
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Results go to book/measurements/ only with --save, so a reader's rerun never overwrites them.
SAVE = "--save" in sys.argv[1:]
OUT = (ROOT / "book" / "measurements" if SAVE else Path.cwd()) / "part_v_b_resume_point.json"


def nbytes(obj) -> int:
    """Bytes held by every tensor inside a nested dict/list."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(v) for v in obj)
    return 0


def main():
    R = {}
    for name in ("ckpt_base_v2.pt", "ckpt_base_v2.pt.last"):
        path = ROOT / name
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        parts = {k: nbytes(v) for k, v in ck.items()}
        meta = {k: (v if isinstance(v, (int, float, str, dict)) and k not in ("model", "opt", "sampler") else None)
                for k, v in ck.items()}
        row = {"file_bytes": path.stat().st_size, "keys": list(ck.keys()), "tensor_bytes": parts,
               "step": ck["step"], "val_loss": ck.get("val_loss"), "best_val": ck.get("best_val"),
               "eval_spec": ck.get("eval_spec")}
        if "opt" in ck:
            opts = ck["opt"]
            st = opts[0]["state"]
            kinds = sorted({k for s in st.values() for k in s})
            row["optimizers"] = len(opts)
            row["opt_state_kinds"] = kinds
            row["opt_param_groups"] = [{k: v for k, v in g.items() if k != "params"} for g in opts[0]["param_groups"]]
            row["opt_bytes_by_kind"] = {k: sum(nbytes(s[k]) for s in st.values() if k in s) for k in kinds}
        if "sampler" in ck:
            s = ck["sampler"]
            row["sampler"] = {"epoch": s["epoch"], "pos": s["pos"], "windows_per_epoch": len(s["order"]),
                              "order_bytes": nbytes(s["order"]), "generator_state_bytes": nbytes(s["g"])}
        model_numel = sum(v.numel() for v in ck["model"].values())
        row["model_numel"] = model_numel
        row["model_dtypes"] = sorted({str(v.dtype) for v in ck["model"].values()})
        R[name] = row
        print(f"{name}: {path.stat().st_size / 1e9:.2f} GB on disk; keys {list(ck.keys())}")
        for k, b in parts.items():
            if b:
                print(f"   {k:8s} {b / 1e9:6.2f} GB of tensors")
        print(f"   step {ck['step']}, val_loss {ck.get('val_loss')}, best_val {ck.get('best_val')}")
        if "opt" in ck:
            print(f"   optimizer: {row['optimizers']} AdamW, state per parameter {row['opt_state_kinds']}, "
                  f"bytes by kind " + ", ".join(f"{k} {v / 1e9:.2f} GB" for k, v in row["opt_bytes_by_kind"].items()))
            for g in row["opt_param_groups"]:
                print(f"   param group: lr {g['lr']:.3g}, weight_decay {g['weight_decay']}, betas {g['betas']}")
        if "sampler" in ck:
            print(f"   sampler: epoch {row['sampler']['epoch']}, position {row['sampler']['pos']:,} of "
                  f"{row['sampler']['windows_per_epoch']:,} windows; order tensor "
                  f"{row['sampler']['order_bytes'] / 2**20:.1f} MiB")
        print(f"   model: {model_numel:,} numbers ({model_numel * 4 / 1e9:.2f} GB = "
              f"{model_numel * 4 / 2**30:.2f} GiB), dtypes {row['model_dtypes']}")
        R[name]["_model"] = ck["model"]
    # Do the two files hold the same weights? Compare every tensor.
    a, b = R["ckpt_base_v2.pt"].pop("_model"), R["ckpt_base_v2.pt.last"].pop("_model")
    equal = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    R["same_weights"] = {"tensors": len(a), "all_equal": equal}
    print(f"best-val file and resume point: {len(a):,} tensors, all equal = {equal}")
    OUT.write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
