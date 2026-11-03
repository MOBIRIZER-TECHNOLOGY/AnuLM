"""
Book example, not repository code.

Appendix B: the 400M arithmetic. Parameters term by term, the active count,
the KV cache, and the memory a training run needs, each computed from the
configuration and checked against something independent: `num_params()`, the
tensors of the released checkpoint, and the byte sizes of the checkpoint files
that run actually wrote.

    python book/code-examples/ch41_arithmetic.py             # B.2-B.6; no files needed
    python book/code-examples/ch41_arithmetic.py --config    # Appendix B.1's table; needs ckpt_base_v2.pt
    python book/code-examples/ch41_arithmetic.py --save      # the book's own run, into book/measurements

The checks against the released run's files run only when those files are
present: `ckpt_base_v2.pt`, and the resume point `ckpt_base_v2.pt.last`
(4.9 GB, not distributed). Both are read memory-mapped, so nothing large is
loaded. Results go to ./ch41_arithmetic.json, or with --save to
book/measurements/part_x_arithmetic.json. Bytes are printed exactly, and
sizes in GB (10^9 bytes, as docs/RESULTS.md prints them) beside GiB (2^30).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from anulm import model as M  # noqa: E402  the repository's anulm/model.py, unchanged
from trace_forward import arithmetic, released, tiny  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SAVED = ROOT / "book" / "measurements" / "part_x_arithmetic.json"


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB ({n / 2**30:.2f} GiB)"


def tensor_bytes(o) -> int:
    if torch.is_tensor(o):
        return o.numel() * o.element_size()
    if isinstance(o, dict):
        return sum(tensor_bytes(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(tensor_bytes(v) for v in o)
    return 0


def kv_cache(cfg) -> dict:
    """Bytes per token of context, for the three precisions Chapter 20 names."""
    n = cfg.n_layer * cfg.n_kv_head * cfg.head_dim            # numbers per token, for K or for V
    return {"numbers_per_token": 2 * n, "bf16": 4 * n, "fp32_keys_bf16_values": 6 * n, "fp32": 8 * n}


def training_state(total: int) -> dict:
    """AdamW in fp32: weights, gradients, and two moments, 4 bytes each."""
    return {"weights": 4 * total, "gradients": 4 * total, "adam_moments": 8 * total, "sum": 16 * total}


def extensions(rel: dict) -> dict:
    """Chapter 41's three larger extensions, as arithmetic on the released model (none is in the repo)."""
    cfg = released()
    m = M.AnuLM(cfg)
    tables = [mod for mod in m.modules() if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding))]
    big = sum(t.weight.numel() for t in tables)                      # every matrix and table
    rows = sum(t.weight.shape[0] for t in tables)                    # one fp32 scale per row for int8
    small = rel["total"] - big                                       # routers and norm gains
    bias = sum(b.numel() for n, b in m.named_buffers() if n.endswith("expert_bias"))   # decides routing
    kv, kv_mixed = rel["kv"]["bf16"], rel["kv"]["fp32_keys_bf16_values"]
    int8 = big + 4 * rows + 4 * small + 4 * bias
    out = {"paged_block_16_tokens_bytes": 16 * kv, "paged_block_16_tokens_fp32_keys_bytes": 16 * kv_mixed,
           "full_context_bytes": cfg.block_size * kv,
           "int8_matrix_bytes": big, "int8_scale_bytes": 4 * rows, "fp32_small_bytes": 4 * small,
           "fp32_router_bias_bytes": 4 * bias, "int8_total_bytes": int8,
           "fp32_all_bytes": 4 * rel["total"], "bf16_all_bytes": 2 * rel["total"],
           "dpo_reference_fp32_bytes": 4 * rel["total"],
           "dpo_total_state_bytes": rel["training_state"]["sum"] + 4 * rel["total"]}
    print("\nextensions, on the released model:")
    print(f"  paged KV cache: a 16-token block, all {cfg.n_layer} layers, holds {16 * kv:,} B "
          f"({16 * kv / 2**10:g} KiB) in bf16, {16 * kv_mixed:,} B ({16 * kv_mixed / 2**10:g} KiB) with the "
          f"fp32 keys autocast leaves; a full {cfg.block_size}-token context {cfg.block_size * kv / 2**20:g} MiB in bf16")
    print(f"  int8 weights: {big:,} matrix numbers at 1 B + {rows:,} row scales at 4 B "
          f"+ {small:,} router and norm numbers and {bias:,} router-bias numbers kept fp32 = {int8:,} B "
          f"({int8 / 2**20:.0f} MiB), against fp32 {4 * rel['total']:,} B "
          f"({4 * rel['total'] / 2**20:.0f} MiB) and bf16 {2 * rel['total'] / 2**20:.0f} MiB")
    print(f"  DPO: a frozen fp32 reference copy adds {gb(4 * rel['total'])} to the "
          f"{gb(rel['training_state']['sum'])} of AdamW state: "
          f"{gb(out['dpo_total_state_bytes'])} before activations")
    return out


def config_table() -> None:
    """Every AnuLMConfig field: its line and default at the pin, two presets, and the released value."""
    import dataclasses
    import re
    import subprocess
    src = subprocess.run(["git", "show", "084cbdd:anulm/model.py"], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8").stdout.splitlines()
    end = next(i for i, l in enumerate(src) if l.startswith("    def __post_init__"))   # the fields come first
    line = {m.group(1): i + 1 for i, l in enumerate(src[:end]) if (m := re.match(r"    (\w+): ", l))}
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False, mmap=True)["cfg"]
    declared = {f.name: f.default for f in dataclasses.fields(M.AnuLMConfig)}   # before __post_init__
    cols = [declared] + [dataclasses.asdict(c) for c in (M.AnuLMConfig.nano_30b(), M.AnuLMConfig.nano_350m(), ck)]
    print("| Field | Line | Default | `nano_30b` | `nano_350m` | Released |")
    print("|---|---|---|---|---|---|")
    for f in dataclasses.fields(M.AnuLMConfig):
        vals = [c[f.name] for c in cols]
        cells = [f"`{v!r}`" for v in vals]
        for i in (1, 2, 3):                       # show only what differs from the default
            if vals[i] == vals[0]:
                cells[i] = "="
        print(f"| `{f.name}` | {line[f.name]} | " + " | ".join(cells) + " |")
    print(f"{len(dataclasses.fields(M.AnuLMConfig))} fields")


def main() -> None:
    res = {}
    for name, cfg in (("tiny()", tiny()), ("nano_30b", M.AnuLMConfig.nano_30b()),
                      ("nano_350m", M.AnuLMConfig.nano_350m()), ("released", released())):
        arithmetic(name, cfg)
        total, active = M.AnuLM(cfg).num_params()
        kv, st = kv_cache(cfg), training_state(total)
        print(f"  KV cache per token: {kv['numbers_per_token']:,} numbers; bf16 {kv['bf16']:,} B, "
              f"fp32 keys + bf16 values {kv['fp32_keys_bf16_values']:,} B, fp32 {kv['fp32']:,} B; "
              f"x block_size {cfg.block_size}: {kv['bf16'] * cfg.block_size / 2**20:.2f} MiB in bf16")
        print(f"  AdamW training state, fp32: {st['sum']:,} B = {gb(st['sum'])}: weights "
              f"{gb(st['weights'])}, gradients {gb(st['gradients'])}, moments {gb(st['adam_moments'])}; "
              f"before activations")
        res[name] = {"total": total, "active": active, "kv": kv, "training_state": st,
                     "block_size": cfg.block_size}

    res["extensions"] = extensions(res["released"])
    out = SAVED if "--save" in sys.argv else Path("ch41_arithmetic.json")
    if (ROOT / "ckpt_base_v2.pt.last").exists():
        res["released_files"] = released_files(res["released"]["total"])
    else:
        print("\n(ckpt_base_v2.pt.last not found: the check against the released run's files is skipped)")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {out}")


def released_files(total: int) -> dict:
    """The released run's own files, read without loading them."""
    best = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False, mmap=True)
    last = torch.load(ROOT / "ckpt_base_v2.pt.last", map_location="cpu", weights_only=False, mmap=True)
    sd_numbers = sum(v.numel() for v in best["model"].values())
    files = {"ckpt_base_v2.pt": (ROOT / "ckpt_base_v2.pt").stat().st_size,
             "ckpt_base_v2.pt.last": (ROOT / "ckpt_base_v2.pt.last").stat().st_size}
    parts = {k: tensor_bytes(last[k]) for k in ("model", "opt", "sampler")}
    n_tensors = sum(1 for _ in last["opt"][0]["state"])
    print(f"\nreleased checkpoint: state-dict numbers {sd_numbers:,} = parameters {total:,} "
          f"+ buffers {sd_numbers - total:,}")
    for f, size in files.items():
        print(f"  {f:<22} {size:>15,} bytes on disk")
    print(f"  .last holds: model {parts['model']:,} B, AdamW state {parts['opt']:,} B "
          f"(= 8 x {total:,} + 4 x {n_tensors} step counters: {8 * total + 4 * n_tensors:,}), "
          f"sampler {parts['sampler']:,} B")
    assert parts["opt"] == 8 * total + 4 * n_tensors, "the moments are not 8 bytes per parameter"
    assert parts["model"] == 4 * sd_numbers
    print(f"  sampler order: {last['sampler']['order'].numel():,} window starts, int64")
    return {"bytes_on_disk": files, "last_parts": parts, "state_dict_numbers": sd_numbers,
            "opt_tensors": n_tensors, "sampler_windows": int(last["sampler"]["order"].numel())}


if __name__ == "__main__":
    config_table() if "--config" in sys.argv else main()
