"""
Load Qwen3's dense checkpoints into AnuLM's own code.

    python tools_qwen.py convert data/qwen3-0.6b-base ckpt_qwen3_0.6b.pt
    python tools_qwen.py check   data/qwen3-0.6b-base ckpt_qwen3_0.6b.pt

Qwen3-0.6B is, in model.py's terms, a dense AnuLM: GQA with QK-norm before
RoPE, split-halves RoPE at theta 1e6, pre-norm RMSNorm blocks, SwiGLU, tied
embeddings. Only the numbers differ -- 28 layers, 16 query / 8 KV heads of
width 128, FFN 3072, vocab 151,936 -- and `first_k_dense_replace = n_layer`
makes every layer dense. The one layout change is that model.py fuses q, k,
v into `query_key_value`; the converter concatenates them in that order.

`check` holds the conversion to the reference implementation: identical
logits from transformers' Qwen3ForCausalLM on the same text.

Why here at all (docs/RESULTS.md sections 32-33): the retrieval reader built
on this project's own 400M base reads at SQuAD F1 37 and answers 0 of 20
demo questions, and the limit is how little that base has read (~230M
tokens). Qwen3-0.6B read 36T, is Apache 2.0, and covers Hindi. Everything
above the backbone -- rag.py, make_rc.py, rc_pointer.py -- is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from model import AnuLM, AnuLMConfig


def config_from(hf: dict, tokenizer_path: str, block_size: int = 2048) -> AnuLMConfig:
    assert hf.get("attention_bias") in (False, None) and hf.get("rope_scaling") is None
    L = hf["num_hidden_layers"]
    return AnuLMConfig(
        vocab_size=hf["vocab_size"], block_size=block_size, tokenizer_path=tokenizer_path,
        n_layer=L, hidden_size=hf["hidden_size"], n_head=hf["num_attention_heads"],
        attn="gqa", head_dim=hf["head_dim"], n_kv_head=hf["num_key_value_heads"],
        use_qk_norm=True, intermediate_size=hf["intermediate_size"], first_k_dense_replace=L,
        rope_theta=float(hf["rope_theta"]), rms_norm_eps=hf["rms_norm_eps"],
        tie_word_embeddings=hf["tie_word_embeddings"], moe_impl="sparse")


def convert(src: str, out: str) -> AnuLM:
    from safetensors.torch import load_file
    src = Path(src)
    hf = json.load(open(src / "config.json"))
    cfg = config_from(hf, str(src / "tokenizer.json"))
    sd = load_file(str(src / "model.safetensors"))
    sd = {k.removeprefix("model."): v.float() for k, v in sd.items()}
    new = {"embed_tokens.weight": sd["embed_tokens.weight"], "norm.weight": sd["norm.weight"]}
    for i in range(cfg.n_layer):
        p = f"layers.{i}."
        new[p + "input_layernorm.weight"] = sd[p + "input_layernorm.weight"]
        new[p + "post_attention_layernorm.weight"] = sd[p + "post_attention_layernorm.weight"]
        new[p + "attn.query_key_value.weight"] = torch.cat(
            [sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"],
             sd[p + "self_attn.v_proj.weight"]], dim=0)
        new[p + "attn.dense.weight"] = sd[p + "self_attn.o_proj.weight"]
        new[p + "attn.query_layernorm.weight"] = sd[p + "self_attn.q_norm.weight"]
        new[p + "attn.key_layernorm.weight"] = sd[p + "self_attn.k_norm.weight"]
        for w in ("gate_proj", "up_proj", "down_proj"):
            new[p + f"mlp.{w}.weight"] = sd[p + f"mlp.{w}.weight"]
    new["lm_head.weight"] = new["embed_tokens.weight"]
    m = AnuLM(cfg)
    missing, unexpected = m.load_state_dict(new, strict=False)
    assert not unexpected, unexpected
    assert not [k for k in missing if not k.startswith("lm_head")], missing
    total, active = m.num_params()
    torch.save({"model": m.state_dict(), "cfg": cfg, "step": 0, "val_loss": float("nan"),
                "base_ckpt": f"Qwen/Qwen3-0.6B-Base ({src})"}, out)
    print(f"converted {src} -> {out}: {total / 1e6:.1f}M params, {cfg.n_layer} dense layers")
    return m


@torch.no_grad()
def check(src: str, ck: str) -> None:
    from transformers import AutoModelForCausalLM

    from hf_tok import HFTokenizer
    tok = HFTokenizer(Path(src) / "tokenizer.json")
    text = "The capital of India is New Delhi. भारत की राजधानी नई दिल्ली है। def add(a, b): return"
    ids = torch.tensor([tok.encode(text)])
    ref = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32).eval()
    want = ref(ids).logits
    c = torch.load(ck, map_location="cpu", weights_only=False)
    m = AnuLM(c["cfg"]).eval()
    m.load_state_dict(c["model"])
    got, _ = m(ids, ids)
    d = (got - want).abs().max().item()
    agree = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    print(f"{ids.shape[1]} tokens: max |logit diff| {d:.2e}, next-token argmax agreement {agree:.1%}")
    nxt = tok.decode(m.generate(tok_ids := torch.tensor([tok.encode("The capital of France is")]),
                                8, top_k=1)[0, tok_ids.shape[1]:].tolist())
    print(f"AnuLM(Qwen3) greedy: 'The capital of France is' -> {nxt!r}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd, src, out = sys.argv[1:4]
    convert(src, out) if cmd == "convert" else check(src, out)
