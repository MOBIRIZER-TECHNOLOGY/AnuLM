"""
Reference solution for Chapter 15: shipping.

to_bf16 and tensor_groups match export_hf.py's; export_folder and load_folder
write and read the same layout (model.safetensors, extras.safetensors,
config.json) in a few lines; release_report turns measurements and gates into
the table a model card carries. course/checks/ch15.py proves them.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

CORE_KEYS = {"model", "cfg", "step", "val_loss", "qa_templates", "base_ckpt"}


def to_bf16(sd: dict) -> dict:
    """Big floating tensors to bfloat16; floating tensors under 100,000 numbers (norm gains, the
    experts' biases) stay as they are, because they are cheap and precision-sensitive; integer
    tensors are never touched."""
    out = {}
    for k, v in sd.items():
        v = v.detach().cpu().contiguous()
        if v.is_floating_point() and v.numel() >= 100_000:
            v = v.to(torch.bfloat16)
        out[k] = v
    return out


def tensor_groups(ck: dict) -> dict:
    """Every state dict in the checkpoint other than the backbone's: a projector, a head, adapters."""
    return {k: v for k, v in ck.items() if k not in CORE_KEYS and isinstance(v, dict) and v
            and all(torch.is_tensor(t) for t in v.values())}


def export_folder(ck: dict, folder) -> Path:
    """model.safetensors (the backbone, through to_bf16), extras.safetensors (every tensor group,
    keys "group.name", through to_bf16), config.json (cfg, step, val_loss, and which groups)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_file(to_bf16(ck["model"]), str(folder / "model.safetensors"))
    groups = tensor_groups(ck)
    if groups:
        save_file(to_bf16({f"{g}.{k}": t for g, sd in groups.items() for k, t in sd.items()}),
                  str(folder / "extras.safetensors"))
    config = {"cfg": ck["cfg"], "step": ck.get("step"), "val_loss": ck.get("val_loss"), "extras": sorted(groups)}
    (folder / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    return folder


def load_folder(folder) -> dict:
    """The checkpoint dict back: model and every group as float32 (what was stored in bfloat16 is
    widened again), plus cfg, step and val_loss."""
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    widen = lambda sd: {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    ck = {"model": widen(load_file(str(folder / "model.safetensors"))), "cfg": config["cfg"],
          "step": config["step"], "val_loss": config["val_loss"]}
    if config["extras"]:
        flat = widen(load_file(str(folder / "extras.safetensors")))
        for g in config["extras"]:
            ck[g] = {k[len(g) + 1:]: v for k, v in flat.items() if k.startswith(g + ".")}
    return ck


def release_report(results: dict, rules: dict, gate_results: dict) -> tuple[str, bool]:
    """A markdown table, one row per gated measure: | measure | value | gate | pass |, with a
    missing value shown as "-" and a failure as **no**; then one line that says whether every gate
    passed. Returns (text, all passed)."""
    lines = ["| measure | value | gate | pass |", "|---|---|---|---|"]
    for k, (op, thr) in rules.items():
        v = results.get(k, "-")
        lines.append(f"| {k} | {v} | {op} {thr} | {'yes' if gate_results.get(k) else '**no**'} |")
    ok = all(gate_results.get(k) for k in rules)
    lines += ["", f"Release gates: **{'all pass' if ok else 'NOT all pass -- not published'}**"]
    return "\n".join(lines) + "\n", ok
