"""Add the `transformers` layer to an export that already exists, without
touching model.safetensors.

    python tools_hf_upgrade.py release/AnuLM-Coder-400M

`export_hf.py` now writes everything below as part of a normal export, so this
is for folders exported before it did -- which includes the four already on the
Hub. Their weights are byte-for-byte fine and re-uploading 0.8 GB apiece to
change three small files would be silly, so this adds only what is missing:

  * `tokenizer.json` + `tokenizer_config.json`, converted from the project's
    own tokenizer and verified token-for-token against `bpe.py` first, so
    `AutoTokenizer.from_pretrained` works;
  * `configuration_anulm.py`, `modeling_anulm.py`, `model.py`, copied from this
    checkout, which is the code `trust_remote_code=True` runs;
  * `architectures` and `auto_map` in `config.json`, leaving every existing key
    exactly as it was, so `load_checkpoint()` keeps reading the same file.

It then loads the folder back through `AutoModelForCausalLM` and checks the
logits against this repository's own model before reporting success.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
CODE = ("configuration_anulm.py", "modeling_anulm.py", "model.py")
AUTO_MAP = {
    "AutoConfig": "configuration_anulm.AnuLMConfig",
    "AutoModelForCausalLM": "modeling_anulm.AnuLMForCausalLM",
}


def upgrade(folder: Path) -> list[str]:
    cfg_path = folder / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"{folder} has no config.json -- is it an export?")
    meta = json.loads(cfg_path.read_text(encoding="utf-8"))
    written: list[str] = []

    # 1. the tokenizer, if this checkpoint has one (byte-level models do not)
    tok_file = meta.get("tokenizer_file")
    if tok_file and (folder / tok_file).exists():
        import tools_hf_tokenizer as conv
        conv.main_for(folder / tok_file, folder)      # raises if it disagrees
        written += ["tokenizer.json", "tokenizer_config.json"]

    # 2. the code trust_remote_code will run
    for mod in CODE:
        shutil.copyfile(HERE / mod, folder / mod)
        written.append(mod)

    # 3. the two keys transformers reads, added beside the existing ones
    meta["architectures"] = ["AnuLMForCausalLM"]
    # float32, deliberately: see the note in export_hf.py and the model card.
    meta["dtype"] = "float32"
    meta["torch_dtype"] = "float32"
    meta["auto_map"] = AUTO_MAP
    meta["architecture"] = ("decoder-only MoE in the shape of Sarvam 30B; loads with "
                            "transformers (trust_remote_code) or with model.py from "
                            "the AnuLM repository")
    cfg_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8", newline="\n")
    written.append("config.json")
    return written


def check(folder: Path) -> None:
    """Load it the way a stranger would, and hold it to the native numbers."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from model import AnuLM, load_checkpoint

    ck = load_checkpoint(str(folder), "cpu")
    native = AnuLM(ck["cfg"]).eval()
    native.load_state_dict(ck["model"])

    cfg_json = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert cfg_json.get("dtype") == "float32", (
        "config.json must declare float32: bfloat16 weights change the router's "
        "expert selection and collapse the output")
    AutoConfig.from_pretrained(str(folder), trust_remote_code=True)
    # float32 explicitly: the export keeps big tensors in bfloat16, so
    # transformers infers bfloat16 and the comparison below would be measuring
    # precision rather than correctness. Users get bfloat16 by default, which
    # is the right default for running it.
    hf = AutoModelForCausalLM.from_pretrained(str(folder), trust_remote_code=True,
                                              dtype=torch.float32).eval()
    assert torch.equal(native.rotary.inv_freq, hf.rotary.inv_freq), (
        "rotary tables differ: the non-persistent buffers were not rebuilt "
        "after load, so positions would be noise while every weight looked fine")

    ids = torch.tensor([[7, 11, 42, 13, 5, 99]], dtype=torch.long)
    with torch.no_grad():
        nat, _ = native(ids)
        out = hf(input_ids=ids)
        gen_n = native.generate(ids, 16, temperature=1.0, top_k=1)[0].tolist()
        gen_h = hf.generate(input_ids=ids, max_new_tokens=16, do_sample=False,
                            pad_token_id=0)[0].tolist()
    delta = (nat[:, -1] - out.logits[:, -1]).abs().max().item()
    assert delta < 1e-4, f"logits differ by {delta}"
    assert gen_n == gen_h, "greedy continuations differ"
    print(f"  loads through AutoModelForCausalLM; logits agree to {delta:.1e}, "
          f"greedy identical over 16 tokens")

    if (folder / "tokenizer.json").exists():
        from transformers import AutoTokenizer

        from bpe import BPE
        t = AutoTokenizer.from_pretrained(str(folder))
        ours = BPE.load(str(folder / json.loads((folder / "config.json")
                                                .read_text(encoding="utf-8"))["tokenizer_file"]))
        probe = "def is_prime(n):\n    return n > 1  # भारत"
        assert t(probe)["input_ids"] == ours.encode(probe), "tokenizers disagree"
        print("  AutoTokenizer matches bpe.py token-for-token")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for name in sys.argv[1:]:
        folder = Path(name)
        print(f"{folder}:")
        for f in upgrade(folder):
            print(f"  + {f}")
        check(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
