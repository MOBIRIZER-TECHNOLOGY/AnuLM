"""
Book example, not repository code.

Chapter 39: exporting a checkpoint, and loading it back two ways.

    python book/code-examples/ch39_export_roundtrip.py            # small models, CPU, seconds
    python book/code-examples/ch39_export_roundtrip.py --measure [--save]  # + the released model

It runs `export_hf.py`, `model.load_checkpoint`, `configuration_anulm.py` and
`modeling_anulm.py` unchanged (the last two need `transformers`) and checks:

1. which tensors `to_bf16` stores in bfloat16 and which it keeps in float32;
2. that `export_hf.main()` writes a folder that `load_checkpoint` reads back;
3. that the `transformers` wrapper loads that folder and matches the native
   model: same logits, same greedy tokens, and a loss that shifts its labels
   itself;
4. what the wrapper refuses.

With --measure it reads the released folder's safetensors header (no weights),
and runs the released checkpoint and its exported folder over the same 64
held-out windows of 512 tokens (seed 0, as measure_attention.py) in float32,
to measure what bfloat16 storage costs. No held-out text is printed or saved.
Results: ./part_viii_ix_export.json, or book/measurements/ with --save.
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.inference import export_hf as export_hf                                            # noqa: E402  repository code, unchanged
from anulm import model as M                                           # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=FutureWarning)
RELEASE = ROOT / "release" / "AnuLM-Base-v2-400M"
OUT = ROOT / "book" / "measurements" / "part_viii_ix_export.json"


def small(**kw) -> M.AnuLMConfig:
    """tiny() with a vocabulary large enough that two tables pass to_bf16's 100,000 threshold."""
    base = dict(vocab_size=2048, block_size=64, n_layer=3, hidden_size=64, n_head=4, head_dim=16,
                n_kv_head=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
                intermediate_size=128, n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


def dtype_rule() -> None:
    torch.manual_seed(0)
    sd = M.AnuLM(small()).state_dict()
    out = export_hf.to_bf16(sd)
    bf = sorted(k for k, v in out.items() if v.dtype == torch.bfloat16)
    print(f"1. to_bf16 on a 2,048-word model: {len(bf)} of {len(out)} tensors to bfloat16: {bf}")
    print(f"   kept float32 include: layers.1.mlp.router.weight {tuple(out['layers.1.mlp.router.weight'].shape)}, "
          f"layers.1.mlp.router.expert_bias {out['layers.1.mlp.router.expert_bias'].dtype}")
    assert bf == ["embed_tokens.weight", "lm_head.weight"]


def export_and_load(d: Path):
    torch.manual_seed(0)
    cfg = small()
    native = M.AnuLM(cfg).eval()
    torch.save({"model": native.state_dict(), "cfg": cfg, "step": 7, "val_loss": 1.25}, d / "small.pt")
    argv = sys.argv
    try:                                                   # the exporter's own command line
        sys.argv = ["export_hf.py", str(d / "small.pt"), str(d / "Small"), "--license", "mit"]
        export_hf.main()
    finally:
        sys.argv = argv
    files = sorted(f.name for f in (d / "Small").iterdir())
    ck = M.load_checkpoint(str(d / "Small"))
    again = M.AnuLM(ck["cfg"]).eval()
    again.load_state_dict(ck["model"])
    x = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        a, _ = native(x)
        b, _ = again(x)
    print(f"\n2. export_hf.main() wrote {files}")
    print(f"   load_checkpoint(folder): step {ck['step']}, val_loss {ck['val_loss']}; "
          f"last-position logits differ by at most {(a - b).abs().max().item():.2e} (bf16 tables)")
    return native, d / "Small"


def hf_wrapper(native: M.AnuLM, folder: Path) -> None:
    from anulm.hf.modeling_anulm import AnuLMForCausalLM
    hf = AnuLMForCausalLM.from_pretrained(str(folder), dtype=torch.float32).eval()
    ck = M.load_checkpoint(str(folder))
    nat = M.AnuLM(ck["cfg"]).eval()
    nat.load_state_dict(ck["model"])                      # the same bf16-rounded weights as hf
    x = torch.randint(0, native.cfg.vocab_size, (1, 24))
    with torch.no_grad():
        full = hf(input_ids=x).logits
        last, _ = nat(x)
        hf_loss = hf(input_ids=x, labels=x).loss
        _, nat_loss = nat(x[:, :-1], x[:, 1:])
        want = nat.generate(x, 12, top_k=1)[0, 24:].tolist()
        got = hf.generate(input_ids=x, max_new_tokens=12, do_sample=False, pad_token_id=0)[0, 24:].tolist()
    print(f"\n3. AnuLMForCausalLM.from_pretrained(folder): logits {tuple(full.shape)} (every position), "
          f"native {tuple(last.shape)}")
    print(f"   last-position difference {(full[:, -1] - last[:, -1]).abs().max().item():.1e}; "
          f"loss with labels=x {hf_loss.item():.6f}, native with shifted targets {nat_loss.item():.6f}")
    print(f"   greedy 12 tokens equal: {got == want}")
    assert got == want and abs(hf_loss.item() - nat_loss.item()) < 1e-5
    refusals = []
    try:
        hf(inputs_embeds=torch.zeros(1, 4, native.cfg.hidden_size))
    except NotImplementedError as e:
        refusals.append(f"inputs_embeds: {e}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        AnuLMForCausalLM.from_pretrained(str(folder), dtype=torch.bfloat16)
        refusals += [f"bfloat16 load: RuntimeWarning '{str(x.message)[:58]}...'" for x in w
                     if issubclass(x.category, RuntimeWarning)]
    print("4. refusals: " + "\n             ".join(refusals))


def header(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def measure() -> dict:
    import subprocess
    res = {"release_files": {}}
    print(f"\n5. {RELEASE.relative_to(ROOT)} (local copy; release/ is git-ignored)")
    for f in sorted(RELEASE.iterdir()):
        same = ""
        if f.suffix == ".py":
            pin = subprocess.run(["git", "show", f"773b6a8:{f.name}"], cwd=ROOT, capture_output=True).stdout
            same = "  identical to @773b6a8" if pin == f.read_bytes() else "  differs from @773b6a8"
        res["release_files"][f.name] = f.stat().st_size
        print(f"   {f.stat().st_size:>13,}  {f.name}{same}")
    n, h = header(RELEASE / "model.safetensors")
    meta = h.pop("__metadata__", {})
    count, nbytes = Counter(), Counter()
    kinds = Counter()
    for k, v in h.items():
        count[v["dtype"]] += 1
        nbytes[v["dtype"]] += v["data_offsets"][1] - v["data_offsets"][0]
        numel = int(np.prod(v["shape"]))
        assert (v["dtype"] == "BF16") == (numel >= 100_000), k     # export_hf.to_bf16's rule, on the release
        if v["dtype"] == "F32":
            kinds["router " + k.split(".")[-1] if "router" in k else "norm gains"] += 1
    size = (RELEASE / "model.safetensors").stat().st_size
    res["release_header"] = {"metadata": meta, "header_bytes": n, "tensors": dict(count),
                             "bytes": dict(nbytes), "file_bytes": size, "float32_kinds": dict(kinds)}
    print(f"\n6. model.safetensors: {size:,} bytes = 8 + {n:,} header + data")
    for dt in count:
        print(f"   {dt}: {count[dt]:5d} tensors, {nbytes[dt]:,} bytes = {nbytes[dt] // (2 if dt == 'BF16' else 4):,} numbers")
    print(f"   float32 tensors by kind: {dict(kinds)}; rule numel >= 100,000 <=> BF16 holds for every tensor")
    pt = (ROOT / "ckpt_base_v2.pt").stat().st_size
    print(f"   numbers in all: {nbytes['BF16'] // 2 + nbytes['F32'] // 4:,}; ckpt_base_v2.pt is {pt:,} bytes, "
          f"{pt / size:.2f}x the safetensors file")
    # What 16 bits cost, on held-out windows. moe_impl="sparse" so that, except
    # where a variant says otherwise, every matrix multiply runs in float32
    # (the checkpoint's own "grouped" path would run the experts in bf16).
    from dataclasses import replace
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    starts = np.random.default_rng(0).integers(0, len(val) - 513, size=64)
    rows = np.stack([val[s: s + 513] for s in starts]).astype(np.int64)

    def bias(sd, f):
        return {k: (f(v) if k.endswith("expert_bias") else v) for k, v in sd.items()}
    bf = lambda v: v.to(torch.bfloat16).float()
    variants = [  # name, source, state-dict edit, run the whole model in bfloat16
        ("native .pt, fp32", ROOT / "ckpt_base_v2.pt", None, False),
        ("exported folder (bf16 matrices), fp32 compute", RELEASE, None, False),
        ("  + router biases rounded to bf16", RELEASE, lambda sd: bias(sd, bf), False),
        ("  + router biases set to zero", RELEASE, lambda sd: bias(sd, torch.zeros_like), False),
        ("  + biases centred per layer, then bf16", RELEASE, lambda sd: bias(sd, lambda v: bf(v - v.mean())), False),
        ("native .pt, whole model .to(bfloat16)", ROOT / "ckpt_base_v2.pt", None, True),
    ]
    sd0 = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", mmap=True, weights_only=False)["model"]
    bs = [v.float() for k, v in sd0.items() if k.endswith("expert_bias")]
    means, spans = [float(b.mean()) for b in bs], [float(b.max() - b.min()) for b in bs]
    distinct = [int(bf(b).unique().numel()) for b in bs]
    print(f"   router biases, 19 layers: mean {min(means):.1f} to {max(means):.1f}; spread (max - min) "
          f"{min(spans):.3f} to {max(spans):.3f}; distinct values per layer after bf16 rounding: "
          f"{min(distinct)} to {max(distinct)} (of 24)")
    res["router_bias"] = {"mean_range": [min(means), max(means)], "span_range": [min(spans), max(spans)],
                          "distinct_after_bf16": [min(distinct), max(distinct)]}
    del sd0
    out, sets, ref = {}, {}, None
    for name, src, edit, whole in variants:
        ck = M.load_checkpoint(str(src), "cpu")
        sd = edit(ck["model"]) if edit else ck["model"]
        m = M.AnuLM(replace(ck["cfg"], moe_impl="sparse")).to(dev).eval()
        m.load_state_dict(sd)
        del ck, sd
        if whole:
            m = m.to(torch.bfloat16)                      # parameters and buffers, compute in bf16
        chosen = {}

        def hook(i):
            def f(mod, inp, o):
                chosen.setdefault(i, []).append(o[0].sort(-1)[0].cpu())
            return f
        hooks = [l.mlp.router.register_forward_hook(hook(i)) for i, l in enumerate(m.layers) if l.is_moe]
        tot = n_tok = 0
        with torch.no_grad():
            for b in range(0, 64, 8):
                x = torch.from_numpy(rows[b: b + 8, :-1]).to(dev)
                y = torch.from_numpy(rows[b: b + 8, 1:]).to(dev).masked_fill(x == 32767, -1)
                _, loss = m(x, y)
                k = int((y != -1).sum())
                tot, n_tok = tot + loss.float().item() * k, n_tok + k
        for h_ in hooks:
            h_.remove()
        chosen = {i: torch.cat(v) for i, v in chosen.items()}
        if ref is None:
            ref = chosen
        changed = sorted(float((chosen[i] != ref[i]).any(-1).float().mean()) for i in ref)
        sets[name] = {"median": round(changed[len(changed) // 2], 4), "max": round(changed[-1], 4)}
        out[name] = tot / n_tok
        print(f"   {name:46s} loss {out[name]:.5f} ({out[name] - out[variants[0][0]]:+.5f}); expert sets "
              f"changed: median layer {100 * sets[name]['median']:.1f}%, max {100 * sets[name]['max']:.1f}%")
        if whole or name.startswith("native .pt, fp32"):    # greedy continuations: share of distinct tokens
            with torch.no_grad():
                g = m.generate(torch.from_numpy(rows[:8, :64]).to(dev), 128, top_k=1)[:, 64:]
            share = float(np.mean([len(set(r.tolist())) / 128 for r in g]))
            res.setdefault("greedy_distinct_share", {})[name] = round(share, 4)
            print(f"   {'':46s} greedy, 8 prompts x 128 tokens: distinct-token share {share:.3f}")
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    res["heldout_loss"] = {k: round(v, 8) for k, v in out.items()}
    res["expert_sets_changed_vs_native"] = sets
    res["sample"] = ("64 windows of 512 from data/v2/val.u16, numpy default_rng(0), targets after EOS 32767 "
                     "ungraded; moe_impl sparse; prompts for greedy = first 64 tokens of the first 8 windows")
    print(f"   graded tokens per variant: {n_tok:,}")
    return res


if __name__ == "__main__":
    dtype_rule()
    with tempfile.TemporaryDirectory() as d:
        native, folder = export_and_load(Path(d))
        hf_wrapper(native, folder)
    if "--measure" in sys.argv:
        r = measure()
        out = OUT if "--save" in sys.argv else Path.cwd() / OUT.name   # book/measurements only with --save
        out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        print(f"   -> {out}")
    print("\nall checks passed")
