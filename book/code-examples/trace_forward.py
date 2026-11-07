"""
Book example, not repository code.

Chapter 20: one forward pass of AnuLM, from token ids to the loss, with every
tensor shape printed in the order the model produces it. Then the parameter
arithmetic for the same model and for the released 400M model.

    python book/code-examples/trace_forward.py

It runs `model.py` unchanged. It only watches: forward hooks on modules, and
thin wrappers around two module-level functions (`apply_rope`, `_sdpa`) that
print their arguments' shapes and call the original. Every shape the chapter
prints comes from this script's output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def tiny() -> M.AnuLMConfig:
    """test_model.py's tiny(), copied so this script needs no test harness."""
    return M.AnuLMConfig(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8,
                         n_kv_head=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
                         intermediate_size=64, n_group=1, topk_group=1)


def released() -> M.AnuLMConfig:
    """AnuLM-Base-v2-400M's configuration: base_350m with the settings the released run changed.
    Every field that sets a shape or a parameter count matches ckpt_base_v2.pt's own config (397,659,648
    parameters). Six training and serving settings differ and change no shape: tokenizer_path,
    bias_update_rate, moe_impl, seq_balance_alpha, router_z_alpha, yarn_original_context."""
    return M.AnuLMConfig.base_350m(vocab_size=32768, block_size=1024, sliding_window=256, max_window_layers=10,
                                   num_experts=24, num_experts_per_tok=4, num_shared_experts=0,
                                   moe_intermediate_size=192)


def shape(t):
    return "(" + ", ".join(str(d) for d in t.shape) + ")"


def trace() -> None:
    torch.manual_seed(0)
    cfg = tiny()
    model = M.AnuLM(cfg).eval()
    B, S = 2, 5
    idx = torch.randint(0, cfg.vocab_size, (B, S))
    targets = torch.randint(0, cfg.vocab_size, (B, S))
    print(f"tiny(): D={cfg.hidden_size}, H={cfg.n_head}, Hkv={cfg.n_kv_head}, Dh={cfg.head_dim}, "
          f"L={cfg.n_layer}, E={cfg.num_experts}, k={cfg.num_experts_per_tok}, V={cfg.vocab_size}; "
          f"shared experts {cfg.num_shared_experts}")
    print(f"input ids                      idx {shape(idx)}  (B, S)")

    log = []
    hooks = []

    def watch(module, name, show_input=False):
        def hook(mod, inp, out):
            if show_input:
                log.append(f"{name:<33}in  {shape(inp[0])}")
            o = out[0] if isinstance(out, tuple) else out
            log.append(f"{name:<33}out {shape(o)}" + (f"  aux={out[1]}" if isinstance(out, tuple) and len(out) == 2
                                                       and not torch.is_tensor(out[1]) else ""))
        hooks.append(module.register_forward_hook(hook))

    rot = model.rotary
    hooks.append(rot.register_forward_hook(
        lambda m, i, o: log.append(f"{'rotary cos, sin':<31}out {shape(o[0])} each  (S, Dh)")))
    for li, layer in enumerate(model.layers):
        p = f"layer {li}"
        hooks.append(layer.register_forward_pre_hook(
            lambda m, i, p=p: log.append(f"{p + " input x":<33}    {shape(i[0])}  (B, S, D)")))
        watch(layer.input_layernorm, f"{p} input_layernorm")
        watch(layer.attn.query_key_value, f"{p} attn.query_key_value")
        watch(layer.attn.dense, f"{p} attn.dense", show_input=True)
        watch(layer.post_attention_layernorm, f"{p} post_attention_layernorm")
        if layer.is_moe:
            r = layer.mlp.router
            hooks.append(r.register_forward_hook(lambda m, i, o, p=p: log.append(
                f"{p + ' mlp.router':<31}in  {shape(i[0])}  -> expert ids {shape(o[0])}, weights {shape(o[1])}  (N, A)")))
            if layer.mlp.shared_expert is not None:
                watch(layer.mlp.shared_expert, f"{p} mlp.shared_expert")
            watch(layer.mlp, f"{p} mlp (MoE)")
        else:
            watch(layer.mlp, f"{p} mlp (dense SwiGLU)")
        hooks.append(layer.register_forward_hook(
            lambda m, i, o, p=p: log.append(f"{p + ' output x':<31}    {shape(o[0])}")))
    watch(model.norm, "final norm")
    watch(model.lm_head, "lm_head (logits)")

    orig_rope, orig_sdpa = M.apply_rope, M._sdpa

    def rope(q, k, cos, sin):
        log.append(f"{'  apply_rope q, k':<31}    q {shape(q)} (B, H, S, Dh), k {shape(k)} (B, Hkv, S, Dh)")
        return orig_rope(q, k, cos, sin)

    def sdpa(q, k, v, mask, S_, scale, drop):
        y = orig_sdpa(q, k, v, mask, S_, scale, drop)
        log.append(f"{'  _sdpa q, k, v -> y':<31}    k,v after repeat_interleave {shape(k)}; y {shape(y)}")
        return y

    M.apply_rope, M._sdpa = rope, sdpa
    try:
        with torch.no_grad():
            logits, loss = model(idx, targets)
    finally:
        M.apply_rope, M._sdpa = orig_rope, orig_sdpa
        for h in hooks:
            h.remove()
    for line in log:
        print(line)
    print(f"{'loss':<31}    {loss.item():.4f}  (ln V = {torch.log(torch.tensor(float(cfg.vocab_size))).item():.4f})")
    with torch.no_grad():
        last, _ = model(idx)
    print(f"{'without targets: logits':<31}    {shape(last)}  (B, 1, V): the last position only")


def arithmetic(name, cfg) -> None:
    m = M.AnuLM(cfg)
    total, active = m.num_params()
    D, V, L = cfg.hidden_size, cfg.vocab_size, cfg.n_layer
    H, Hkv, Dh, E, k = cfg.n_head, cfg.n_kv_head, cfg.head_dim, cfg.num_experts, cfg.num_experts_per_tok
    emb = 2 * V * D
    att = L * ((H + 2 * Hkv) * Dh * D + H * Dh * D)
    qk_norm = L * 2 * Dh if cfg.use_qk_norm else 0
    norms = L * 2 * D + D
    dense = cfg.first_k_dense_replace * 3 * D * cfg.intermediate_size
    moe_layers = L - cfg.first_k_dense_replace
    experts = moe_layers * E * 3 * D * cfg.moe_intermediate_size
    shared = moe_layers * 3 * D * cfg.moe_intermediate_size * cfg.num_shared_experts
    routers = moe_layers * E * D
    parts = [("embedding + output head (2·V·D)", emb), ("attention (L·((H+2Hkv)·Dh·D + H·Dh·D))", att),
             ("QK-norm gains (L·2·Dh)", qk_norm), ("RMSNorm gains (2·L·D + D)", norms),
             ("dense layer 0 (3·D·I)", dense), (f"routed experts ({moe_layers}·E·3·D·I_e)", experts),
             (f"shared experts ({moe_layers}·3·D·I_e·n_shared)", shared), (f"routers ({moe_layers}·E·D)", routers)]
    s = sum(v for _, v in parts)
    print(f"\n{name}: num_params() total {total:,}, active {active:,} ({active / total:.1%})")
    for label, v in parts:
        print(f"  {label:<44} {v:>14,}")
    print(f"  {'sum':<44} {s:>14,}   {'matches' if s == total else 'DOES NOT MATCH'} num_params()")
    idle = moe_layers * (E - k) * 3 * D * cfg.moe_intermediate_size
    print(f"  inactive per token: {moe_layers} layers × {E - k} idle experts × 3·D·I_e = {idle:,}; total − that = {total - idle:,}")
    kv = L * 2 * Hkv * Dh * 2
    print(f"  KV cache per token if stored in bf16: L·2·Hkv·Dh·2 bytes = {kv:,} bytes ({kv / 1024:g} KiB)")


if __name__ == "__main__":
    trace()
    arithmetic("tiny()", tiny())
    arithmetic("AnuLM-Base-v2-400M", released())
