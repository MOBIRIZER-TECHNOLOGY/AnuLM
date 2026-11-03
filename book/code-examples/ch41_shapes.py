"""
Book example, not repository code.

Chapter 41, checkpoint 2, and Appendix A: the shape of every tensor that enters
and leaves every module of AnuLM, for a prefill call and for one cached decode
step, printed by running `model.py` unchanged.

    python book/code-examples/ch41_shapes.py            # tiny() GQA, tiny MLA, and the released 400M
    python book/code-examples/ch41_shapes.py --tiny     # only the two tiny models (a second on a CPU)

Three models:

- `tiny()` from test_model.py (GQA), the model every printed shape in the book
  comes from;
- the same with `attn="mla"` and the small MLA sizes test_model.py's
  `cache_holds_what_it_should` uses;
- AnuLM-Base-v2-400M's configuration (`trace_forward.released()`), with
  random weights on the CPU (about 1.6 GB of memory, a few seconds).

Only forward hooks are used: they see each module's inputs and outputs and
change nothing. The script checks a few shapes against the formulas Appendix A
prints and stops if one differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged
from trace_forward import released, tiny  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def tiny_mla() -> M.AnuLMConfig:
    """tiny() with MLA, at the sizes test_model.py's cache test uses."""
    c = tiny()
    return M.AnuLMConfig(**{**c.__dict__, "attn": "mla", "head_dim": None, "qk_nope_head_dim": 8,
                            "qk_rope_head_dim": 8, "v_head_dim": 8, "kv_lora_rank": 16})


def shp(t) -> str:
    if torch.is_tensor(t):
        return "(" + ", ".join(str(d) for d in t.shape) + ")"
    if isinstance(t, (tuple, list)):
        return " ".join(shp(x) for x in t if x is not None)
    return ""


def record(model, layers=(0, 1)):
    """Forward hooks on the top-level modules and on every module inside the chosen layers."""
    rows, hooks = [], []
    wanted = {"embed_tokens", "rotary", "norm", "lm_head"}
    for name, mod in model.named_modules():
        parts = name.split(".")
        inside = parts[0] == "layers" and len(parts) > 1 and int(parts[1]) in layers
        if not (name in wanted or inside):
            continue
        if ".experts." in name and name.count(".") > 4:
            continue                              # an expert's own Linears: same as layer 0's SwiGLU

        def hook(m, inp, out, name=name):
            rows.append((name, type(m).__name__, shp(inp[:1]), shp(out)))
        hooks.append(mod.register_forward_hook(hook))
    return rows, hooks


def run(cfg, label, B=2, S=5):
    torch.manual_seed(0)
    model = M.AnuLM(cfg).eval()
    idx = torch.randint(0, cfg.vocab_size, (B, S))
    print(f"\n=== {label}: B={B}, S={S}, then one cached decode step (S=1, T={S + 1})")
    print(f"    D={cfg.hidden_size} H={cfg.n_head} attn={cfg.attn} L={cfg.n_layer} V={cfg.vocab_size} "
          f"E={cfg.num_experts} A={cfg.num_experts_per_tok} I={cfg.intermediate_size} "
          f"I_e={cfg.moe_intermediate_size} shared={cfg.num_shared_experts}")
    cache = model.new_cache()
    for phase, ids, start in (("prefill", idx, 0), ("decode", idx[:, -1:], S)):
        rows, hooks = record(model)
        calls = []                                # the two module-level functions, layer 0's call only
        orig_rope, orig_sdpa = M.apply_rope, M._sdpa

        def rope(q, k, cos, sin):
            calls.append(f"apply_rope: q {shp(q)}, k {shp(k)}, cos {shp(cos)}")
            return orig_rope(q, k, cos, sin)

        def sdpa(q, k, v, mask, S_, scale, drop):
            calls.append(f"_sdpa: q {shp(q)}, k {shp(k)}, v {shp(v)}, mask {shp(mask) or None}")
            return orig_sdpa(q, k, v, mask, S_, scale, drop)

        M.apply_rope, M._sdpa = rope, sdpa
        try:
            with torch.no_grad():
                logits, _ = model(ids, cache=cache, start=start)
        finally:
            M.apply_rope, M._sdpa = orig_rope, orig_sdpa
            for h in hooks:
                h.remove()
        print(f"--- {phase}: ids {shp(ids)} -> logits {shp(logits)}")
        for c in calls[:2]:
            print(f"    layer 0 {c}")
        seen = set()
        for name, kind, i, o in rows:
            # experts run only when tokens reach them: show the first one that ran, per layer
            key = name.rsplit(".", 1)[0] + ".*" if ".experts." in name else name
            if key in seen:
                continue
            seen.add(key)
            print(f"    {name:<38} {kind:<12} in {i:<16} out {o}")
        c0 = {k: shp(v) for k, v in cache[0].items()}
        print(f"    cache[0] after {phase}: {c0}")
    return model, cache


def check(cfg, cache, S):
    """The formulas Appendix A prints, against what the run produced."""
    B = 2
    if cfg.attn == "gqa":
        assert tuple(cache[0]["k"].shape) == (B, cfg.n_kv_head, S + 1, cfg.head_dim)
        assert tuple(cache[0]["v"].shape) == (B, cfg.n_kv_head, S + 1, cfg.head_dim)
    else:
        assert tuple(cache[0]["c"].shape) == (B, S + 1, cfg.kv_lora_rank)
        assert tuple(cache[0]["k_pe"].shape) == (B, 1, S + 1, cfg.qk_rope_head_dim)


if __name__ == "__main__":
    for cfg, label in ((tiny(), "tiny() GQA"), (tiny_mla(), "tiny MLA")):
        _, cache = run(cfg, label)
        check(cfg, cache, 5)
    if "--tiny" not in sys.argv:
        cfg = released()
        _, cache = run(cfg, "AnuLM-Base-v2-400M (random weights)")
        check(cfg, cache, 5)
    print("\nall shape checks passed")
