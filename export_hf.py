"""
Export a training checkpoint to a Hugging Face-style folder, and load it back.

    python export_hf.py ckpt_coder_sft.pt release/AnuLM-Coder-400M \
        --license cc-by-nc-sa-4.0 --card docs/MODEL_CARD.md

The folder holds `model.safetensors` (weights in bfloat16, no pickle),
`config.json` (the architecture config plus step, val loss, prompt
templates and provenance), the tokenizer the checkpoint was trained with,
and a `README.md` model card with the YAML front matter the Hub reads.
It is about half the size of the `.pt` and contains no executable code.

Checkpoints that are more than a language model -- speech and vision
(`--kind asr|caption|vqa`), the Wikipedia reader (`reader`) and Decide
classifiers (`decide`) -- keep their extra parts (projector, heads, LoRA,
labels) in `extras.safetensors` and `config.json["extras"]`, and an adapter
can be exported without its backbone (`--adapter-of`). docs/PUBLISHING.md §3c.

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


PIPELINE = {"lm": "text-generation", "asr": "automatic-speech-recognition", "caption": "image-to-text",
            "vqa": "visual-question-answering", "decide": "text-classification",
            "reader": "question-answering", "omni": "any-to-any"}

# Keys every exporter writes into config.json itself; anything else that is
# not a tensor group travels in config.json["extras"].
CORE_KEYS = {"model", "cfg", "step", "val_loss", "qa_templates", "base_ckpt"}


def front_matter(args, cfg, ck) -> str:
    langs = ["hi", "en"]
    tags = ["mixture-of-experts", "moe", "hindi", "indic", "small-language-model",
            "pytorch", "sarvam-architecture"]
    if cfg is not None and "code" in Path(cfg.tokenizer_path or "").stem:
        tags += ["code", "python"]
    if cfg is None:                                    # an open encoder (ModernBERT) fine-tuned here
        tags = ["modernbert", "hindi", "indic", "pytorch"]
    fm = {
        "license": args.license,
        "language": langs,
        "pipeline_tag": PIPELINE[args.kind],
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


def tensor_groups(ck: dict) -> dict[str, dict[str, torch.Tensor]]:
    """The checkpoint's state dicts other than the backbone's: the speech or
    vision projector, LoRA adapters, a classifier or pointer head."""
    return {k: v for k, v in ck.items() if k not in CORE_KEYS and isinstance(v, dict) and v
            and all(torch.is_tensor(t) for t in v.values())}


def usage_text(kind: str, repo: str, name: str) -> str:
    if kind == "asr":
        return ("\n\n## How to load\n\nWith the AnuLM repository. The speech encoder is Whisper-small's, "
                "frozen; faster-whisper downloads it on first use.\n\n```bash\n"
                f"hf download {repo} --local-dir {name}\n"
                f"python speech_encoder.py eval --ckpt {name} --prefix data/fleurs_hi/devtest\n```\n\n"
                "```python\nfrom speech_encoder import load_trained\n"
                f"asr = load_trained(\"{repo}\", \"cuda\")   # a Hub id or the downloaded folder\n```\n")
    if kind in ("caption", "vqa"):
        return ("\n\n## How to load\n\nWith the AnuLM repository. The image encoder is SigLIP "
                "(`vit_base_patch16_siglip_224`), frozen; timm downloads it on first use.\n\n```python\n"
                "from vision_encoder import load_trained\n"
                f"vl, tower, tok = load_trained(\"{repo}\", \"cuda\")   # a Hub id or the downloaded folder\n```\n")
    if kind == "decide":
        return ("\n\n## How to load\n\nWith the AnuLM repository:\n\n```python\nfrom decide import Decide\n"
                f"d = Decide(\"{repo}\", device=\"cuda\")   # fetches the model it was trained on, if it is an adapter\n"
                "print(d(\"my card has not arrived yet\"))\n```\n")
    if kind == "omni":
        return ("\n\n## How to load\n\nWith the AnuLM repository; every loader takes the same Hub id. "
                "The speech and image encoders (Whisper-small, SigLIP) are frozen and download on first use.\n\n"
                "```python\n"
                f"from serve import Engine;  chat = Engine(\"{repo}\", \"cuda\")          # chat, reading, intents\n"
                f"from speech_encoder import load_trained;  asr = load_trained(\"{repo}\", \"cuda\")\n"
                f"from vision_encoder import load_trained;  vl, tower, tok = load_trained(\"{repo}\", \"cuda\")\n"
                "```\n\n"
                "```bash\ngit clone https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM && cd AnuLM\n"
                "python app_omni.py          # every tab on this one model\n```\n")
    if kind == "reader":
        return ("\n\n## How to load\n\nWith the AnuLM repository:\n\n```python\nfrom rc_pointer import PointerQA\n"
                f"qa = PointerQA(\"{repo}\", \"cuda\")\n```\n")
    return (
        "\n\n## How to load\n\n"
        "With `transformers` (the modelling code travels with the weights, so "
        "`trust_remote_code=True` is required and there is nothing to clone):\n\n"
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        f'model = AutoModelForCausalLM.from_pretrained("{repo}", trust_remote_code=True)\n'
        f'tok = AutoTokenizer.from_pretrained("{repo}")\n'
        'ids = tok("def is_prime(n):", return_tensors="pt")\n'
        "print(tok.decode(model.generate(**ids, max_new_tokens=60)[0]))\n"
        "```\n\n"
        "Greedy output is identical to this repository's own `generate`, cached "
        "or not; `test_model.py` holds the two paths together. Batches must be "
        "unpadded, and beam search is not supported.\n\n"
        "Or with the AnuLM repository, which is what every script here expects:\n\n"
        "```bash\n"
        f"hf download {repo} --local-dir {name}\n"
        f"python serve.py  --ckpt {name}          # web page at http://127.0.0.1:8000\n"
        f"python sample.py --ckpt {name} --prompt \"def is_prime(n):\"\n"
        "```\n\n"
        "```python\n"
        "from model import load_checkpoint, AnuLM\n"
        f"ck = load_checkpoint(\"{name}\")\n"
        "m = AnuLM(ck[\"cfg\"]).eval(); m.load_state_dict(ck[\"model\"])\n"
        "```\n")


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
    p.add_argument("--kind", choices=sorted(PIPELINE), default="lm",
                   help="what the checkpoint is; sets the card's pipeline tag and usage snippet")
    p.add_argument("--adapter-of", metavar="CKPT",
                   help="the backbone checkpoint a LoRA/head adapter was trained on: the export keeps "
                        "only the adapter, after checking its backbone copy equals CKPT, and records "
                        "--base-model as where to fetch the backbone from")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = load_checkpoint(args.ckpt, "cpu")
    cfg = ck.get("cfg")
    if cfg is not None and getattr(cfg, "merge_tree", ""):
        # A merge-tree checkpoint stores the raw tables; the exported model
        # (and the transformers wrapper) must see the composed ones.
        from model import AnuLM
        m = AnuLM(cfg)
        m.load_state_dict(ck["model"])
        m.bake_merge_tree()
        ck["model"], cfg = m.state_dict(), m.cfg
        print("baked the merge-tree tables into plain weights")

    groups = tensor_groups(ck)
    extras = {k: v for k, v in ck.items() if k not in CORE_KEYS and k not in groups}
    if args.adapter_of:
        assert args.base_model, "--adapter-of needs --base-model (the Hub id the backbone is fetched from)"
        live = load_checkpoint(args.adapter_of, "cpu")["model"]
        diff = [k for k, v in ck["model"].items() if not torch.equal(live[k], v)]
        assert not diff, f"{args.ckpt}'s backbone differs from {args.adapter_of} ({len(diff)} tensors)"
        del ck["model"]
        extras["base_model"] = args.base_model
        print(f"  adapter only: backbone equals {args.adapter_of}, fetched from {args.base_model} on load")
    hf_state = groups.pop("hf_state", None)

    # 1. weights. An open encoder's (ModernBERT) go into backbone/ in its own
    # format, beside its config and tokenizer, so transformers loads it as-is.
    n_params = 0
    if "model" in ck:
        sd = to_bf16(ck["model"])
        save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt", "exported_from": Path(args.ckpt).name})
        n_params = sum(v.numel() for v in sd.values())
    if hf_state is not None:
        src, bb = Path(ck["hf_backbone"]), out / "backbone"
        bb.mkdir(exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.suffix in (".json", ".txt"):
                shutil.copyfile(f, bb / f.name)
        save_file({k: v.detach().cpu().contiguous() for k, v in hf_state.items()}, str(bb / "model.safetensors"),
                  metadata={"format": "pt"})
        n_params = sum(v.numel() for v in hf_state.values())
        extras["hf_backbone"] = "backbone"
    if groups:
        flat = {f"{g}/{k}": v.detach().cpu().contiguous() for g, sd_ in groups.items() for k, v in sd_.items()}
        save_file(flat, str(out / "extras.safetensors"), metadata={"format": "pt"})
        n_extra = sum(v.numel() for v in flat.values())
        print(f"  extras: {', '.join(groups)} ({n_extra/1e6:.2f}M params, float32)")
        n_params += n_extra

    # 2. tokenizer -- the project's own JSON, and the tokenizers-library
    # equivalent beside it so AutoTokenizer works with no custom code. The
    # converter verifies the two agree token-for-token before writing.
    tok_file = None
    if cfg is not None and getattr(cfg, "tokenizer_path", None):
        src = Path(cfg.tokenizer_path)
        tok_file = f"tokenizer.{src.stem}.json"
        shutil.copyfile(src, out / tok_file)
        try:
            import tools_hf_tokenizer as hf_tok
            hf_tok.main_for(src, out)
            print("  tokenizer.json verified against bpe.py and written")
        except Exception as e:                       # never block a weight export
            print(f"  WARNING: tokenizer.json not written ({type(e).__name__}: {e});"
                  f" AutoTokenizer will not work for this folder")

    # 3. config
    meta = {"model_type": "anulm"}
    if cfg is not None:
        cfg_d = dataclasses.asdict(cfg)
        cfg_d["tokenizer_path"] = tok_file  # resolved relative to the folder by load_checkpoint
        meta.update({
            "architectures": ["AnuLMForCausalLM"],
            # trust_remote_code: transformers fetches these two modules from the
            # repo, and they pull model.py down with them.
            "auto_map": {
                "AutoConfig": "configuration_anulm.AnuLMConfig",
                "AutoModelForCausalLM": "modeling_anulm.AnuLMForCausalLM",
            },
            "architecture": "decoder-only MoE in the shape of Sarvam 30B; loads with transformers (trust_remote_code) or with model.py from the AnuLM repository",
            "config": cfg_d,
            "tokenizer_file": tok_file,
            "tokenizer_format": TOKENIZER_NOTE if tok_file else "bytes",
            # Small tensors stay float32 on disk AND must be loaded that way: the
            # router's per-expert bias decides which experts fire, and rounding it
            # to bfloat16 changes the routing enough to collapse the output. So
            # the export declares float32 and transformers honours it.
            "dtype": "float32",
            "torch_dtype": "float32",
            "weights_dtype": ("bfloat16 on disk for the big tensors, float32 for the small "
                              "ones (router bias, norms); load in float32 -- 16-bit weights "
                              "change the expert routing and break the model"),
        })
    if args.kind != "lm":
        # Not a plain language model: transformers' AutoModelForCausalLM
        # would load the backbone and silently drop what makes it this model.
        for k in ("architectures", "auto_map"):
            meta.pop(k, None)
        meta["kind"] = args.kind
    meta.update({
        "num_parameters": n_params,
        "step": ck.get("step"),
        "val_loss": ck.get("val_loss"),
        "qa_templates": ck.get("qa_templates"),
        "base_ckpt": ck.get("base_ckpt"),
        "exported_from": Path(args.ckpt).name,
        "license": args.license,
    })
    if extras:
        meta["extras"] = extras
    (out / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")

    # 4. the code trust_remote_code loads, copied verbatim from this checkout
    if args.kind == "lm":
        for mod in ("configuration_anulm.py", "modeling_anulm.py", "model.py"):
            shutil.copyfile(Path(__file__).parent / mod, out / mod)

    # 5. card
    body = Path(args.card).read_text(encoding="utf-8") if args.card else f"# {args.name or out.name}\n\nSee the AnuLM repository for details.\n"
    repo = args.repo or f"toonist/{out.name}"
    usage = usage_text(args.kind, repo, out.name)
    if "model" in ck:
        usage += (f"\nWeights: `model.safetensors`, bfloat16, {sum(v.numel() for v in ck['model'].values())/1e6:.1f}M "
                  f"parameters. Tokenizer: `{tok_file}`. {TOKENIZER_NOTE if tok_file else ''}\n")
    elif args.adapter_of:
        usage += (f"\nAdapter only: `extras.safetensors` holds the {', '.join(groups)} weights "
                  f"({n_params/1e6:.2f}M parameters); the backbone is [{args.base_model}]"
                  f"(https://huggingface.co/{args.base_model}).\n")
    (out / "README.md").write_text(front_matter(args, cfg, ck) + body + usage, encoding="utf-8", newline="\n")

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"exported {args.ckpt} -> {out}/  ({total/1e9:.2f} GB, {n_params/1e6:.1f}M params, step {ck.get('step')})")
    for f in sorted(x for x in out.rglob("*") if x.is_file()):
        print(f"  {f.stat().st_size/1e6:9.1f} MB  {f.relative_to(out)}")


if __name__ == "__main__":
    main()
