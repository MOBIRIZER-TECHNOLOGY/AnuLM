"""
Export a training checkpoint to a Hugging Face-style folder, and load it back.

    python export_hf.py ckpt_coder_sft.pt release/AnuLM-Coder-400M \
        --license cc-by-nc-sa-4.0 --card docs/MODEL_CARD.md

The folder holds `model.safetensors` (weights in bfloat16, no pickle),
`config.json` (the architecture config plus step, val loss, prompt
templates and provenance), the tokenizer the checkpoint was trained with,
and a `README.md` model card with the YAML front matter the Hub reads.
It is about half the size of the `.pt` and contains no executable code.

The architecture is not in `transformers`; load the folder with this
repository's code instead, which every script here accepts in place of a
`.pt` path:

    python serve.py  --ckpt release/AnuLM-Coder-400M
    python sample.py --ckpt release/AnuLM-Coder-400M --prompt "def is_prime(n):"

or directly:

    from model import load_checkpoint, AnuLM
    ck = load_checkpoint("release/AnuLM-Coder-400M")
    m = AnuLM(ck["cfg"]); m.load_state_dict(ck["model"])
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from model import load_checkpoint

TOKENIZER_NOTE = ("Byte-level BPE in this repository's own JSON format (bpe.py), "
                  "not the `tokenizers` library format. Load it with `bpe.BPE.load(path)`.")


def to_bf16(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Weights to bfloat16; small floating tensors (norm gains, router
    biases) stay in float32 because they are cheap and precision-sensitive."""
    out = {}
    for k, v in sd.items():
        v = v.detach().cpu().contiguous()
        if v.is_floating_point() and v.numel() >= 100_000:
            v = v.to(torch.bfloat16)
        out[k] = v
    return out


def front_matter(args, cfg, ck) -> str:
    langs = ["hi", "en"]
    tags = ["mixture-of-experts", "moe", "hindi", "indic", "small-language-model",
            "pytorch", "sarvam-architecture"]
    if "code" in Path(cfg.tokenizer_path or "").stem:
        tags += ["code", "python"]
    fm = {
        "license": args.license,
        "language": langs,
        "pipeline_tag": "text-generation",
        "tags": tags,
        "datasets": args.datasets or [],
        "base_model": args.base_model,
        "model-index": [{"name": args.name or Path(args.out).name, "results": []}],
    }
    fm = {k: v for k, v in fm.items() if v not in (None, [], "")}
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            lines.append(f"{k}:")
            for d in v:
                lines.append(f"  - name: {d['name']}")
                lines.append("    results: []")
        elif isinstance(v, list):
            lines.append(f"{k}:")
            lines += [f"  - {x}" for x in v]
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--name", help="display name for the card; default: the output folder name")
    p.add_argument("--license", default="cc-by-nc-sa-4.0",
                   help="SPDX-style id the Hub understands (mit, apache-2.0, cc-by-sa-4.0, cc-by-nc-4.0, ...)")
    p.add_argument("--card", help="markdown file to use as the model card body (e.g. docs/MODEL_CARD.md)")
    p.add_argument("--datasets", nargs="*", help="Hub dataset ids for the card's front matter")
    p.add_argument("--base-model", help="Hub id of the base checkpoint, for fine-tunes")
    p.add_argument("--repo", help="Hub repo id this will be uploaded to, for the usage snippet")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = load_checkpoint(args.ckpt, "cpu")
    cfg = ck["cfg"]

    # 1. weights
    sd = to_bf16(ck["model"])
    save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt", "exported_from": Path(args.ckpt).name})
    n_params = sum(v.numel() for v in sd.values())

    # 2. tokenizer
    tok_file = None
    if getattr(cfg, "tokenizer_path", None):
        src = Path(cfg.tokenizer_path)
        tok_file = f"tokenizer.{src.stem}.json"
        shutil.copyfile(src, out / tok_file)

    # 3. config
    cfg_d = dataclasses.asdict(cfg)
    cfg_d["tokenizer_path"] = tok_file  # resolved relative to the folder by load_checkpoint
    meta = {
        "model_type": "anulm",
        "architecture": "decoder-only MoE in the shape of Sarvam 30B; load with model.py from the AnuLM repository",
        "config": cfg_d,
        "tokenizer_file": tok_file,
        "tokenizer_format": TOKENIZER_NOTE if tok_file else "bytes",
        "weights_dtype": "bfloat16 (tensors under 100k elements kept in float32)",
        "num_parameters": n_params,
        "step": ck.get("step"),
        "val_loss": ck.get("val_loss"),
        "qa_templates": ck.get("qa_templates"),
        "base_ckpt": ck.get("base_ckpt"),
        "exported_from": Path(args.ckpt).name,
        "license": args.license,
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")

    # 4. card
    body = Path(args.card).read_text(encoding="utf-8") if args.card else f"# {args.name or out.name}\n\nSee the AnuLM repository for details.\n"
    repo = args.repo or f"toonist/{out.name}"
    usage = (
        "\n\n## How to load\n\n"
        "The architecture is not in `transformers`. Clone the AnuLM repository and point its scripts at this folder:\n\n"
        "```bash\n"
        f"hf download {repo} --local-dir {out.name}\n"
        f"python serve.py  --ckpt {out.name}          # web page at http://127.0.0.1:8000\n"
        f"python sample.py --ckpt {out.name} --prompt \"def is_prime(n):\"\n"
        "```\n\n"
        "```python\n"
        "from model import load_checkpoint, AnuLM\n"
        f"ck = load_checkpoint(\"{out.name}\")\n"
        "m = AnuLM(ck[\"cfg\"]).eval(); m.load_state_dict(ck[\"model\"])\n"
        "```\n\n"
        f"Weights: `model.safetensors`, bfloat16, {n_params/1e6:.1f}M parameters. "
        f"Tokenizer: `{tok_file}`. {TOKENIZER_NOTE if tok_file else ''}\n"
    )
    (out / "README.md").write_text(front_matter(args, cfg, ck) + body + usage, encoding="utf-8", newline="\n")

    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"exported {args.ckpt} -> {out}/  ({total/1e9:.2f} GB, {n_params/1e6:.1f}M params, step {ck.get('step')})")
    for f in sorted(out.iterdir()):
        print(f"  {f.stat().st_size/1e6:9.1f} MB  {f.name}")


if __name__ == "__main__":
    main()
