"""
Book example, not repository code.

Chapter 31: LoRA's promises measured on the released model, ckpt_base_v2.pt.

    python book/code-examples/ch31_measure_lora.py           # writes part_vii_lora.json here
    python book/code-examples/ch31_measure_lora.py --save    # writes book/measurements/part_vii_lora.json

1. apply_lora with the defaults (r 16, alpha 32, attention projections): how
   far the logits move at step 0, on one made-up sentence.
2. With A initialised under seed 0, give every B a random value (std 1e-3,
   seed 0), then merge_lora: how far the merged model's logits are from the
   adapted model's.
3. The released model's router biases: their range per layer, the scale a
   fine-tune's bias updates are measured against.

fp32 throughout: moe_impl is set to "sparse", because the stored "grouped" path
would run the experts in bf16. On the GPU if one has 4 GB free, else the CPU.
Writes part_vii_lora.json (into book/measurements/ only with --save). No
held-out text is read.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.tokenization.bpe import BPE  # noqa: E402
from anulm.training.lora import apply_lora, lora_parameters, merge_lora  # noqa: E402
from anulm.model import AnuLM, load_checkpoint  # noqa: E402

OUT = ROOT / "book" / "measurements" / "part_vii_lora.json"
TEXT = "The capital of India is New Delhi, and the river Ganga flows past many old cities."
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def free_gb() -> float:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        return float(r.stdout.split()[0]) / 1024
    except Exception:
        return 0.0


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() and free_gb() >= 4 else "cpu"
    ck = load_checkpoint(str(ROOT / "ckpt_base_v2.pt"), "cpu")
    cfg = replace(ck["cfg"], moe_impl="sparse")
    tok = BPE.load(str(ROOT / cfg.tokenizer_path) if not Path(cfg.tokenizer_path).is_absolute() else cfg.tokenizer_path)
    model = AnuLM(cfg)
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    x = torch.tensor([tok.encode(TEXT)], device=dev)
    res = {"device": dev, "precision": "fp32 (moe_impl sparse; no bf16 experts)", "tokens": x.shape[1],
           "bias_update_rate": cfg.bias_update_rate}
    with torch.no_grad():
        base = model(x, x)[0]                            # targets given: logits for every position
    bias = {n: b.detach().cpu() for n, b in model.named_buffers() if n.endswith("expert_bias")}
    spans = [float(b.max() - b.min()) for b in bias.values()]
    res["router_bias_min"] = min(float(b.min()) for b in bias.values())
    res["router_bias_max"] = max(float(b.max()) for b in bias.values())
    res["router_bias_span_per_layer_min"], res["router_bias_span_per_layer_max"] = min(spans), max(spans)

    torch.manual_seed(0)                                 # A's random init: fixed, so the run repeats
    n = apply_lora(model, r=16, alpha=32.0)
    model.to(dev).eval()
    with torch.no_grad():
        step0 = model(x, x)[0]
    res["adapters"], res["trainable"] = n, sum(p.numel() for p in lora_parameters(model))
    res["step0_max_logit_diff"] = float((step0 - base).abs().max())
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("lora_b"):
                p.copy_(torch.randn(p.shape, generator=g) * 1e-3)
        adapted = model(x, x)[0]
        merge_lora(model)
        merged = model(x, x)[0]
    res["adapted_vs_base_max_logit_diff"] = float((adapted - base).abs().max())
    res["merged_vs_adapted_max_logit_diff"] = float((merged - adapted).abs().max())
    res["lora_left_after_merge"] = sum("lora" in k for k in model.state_dict())
    print(f"device {dev}; {x.shape[1]} tokens; {n} adapters, {res['trainable']:,} trainable")
    print(f"step 0: adapted vs base, largest logit difference {res['step0_max_logit_diff']:.1e}")
    print(f"random B (std 1e-3): adapted vs base {res['adapted_vs_base_max_logit_diff']:.3f}; "
          f"merged vs adapted {res['merged_vs_adapted_max_logit_diff']:.1e}; "
          f"lora tensors left after merging: {res['lora_left_after_merge']}")
    print(f"router biases in ckpt_base_v2.pt: {res['router_bias_min']:.3f} to {res['router_bias_max']:.3f}; "
          f"span per layer {min(spans):.3f} to {max(spans):.3f}; bias_update_rate {cfg.bias_update_rate}")
    out = OUT if "--save" in sys.argv else Path.cwd() / OUT.name     # book/measurements/ only with --save
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
