"""
Book example, not repository code.

Chapters 16, 17 and 18: what the released model's feed-forward layers and
routers do on held-out text.

    python book/code-examples/ch16_measure_trained.py          # GPU if free; JSON to the current directory
    python book/code-examples/ch16_measure_trained.py --save   # rewrite book/measurements/ (the book's copy)

Runs `ckpt_base_v2.pt` through `model.py` unchanged, with no autocast: fp32
except the experts, which the released configuration's grouped path
(`moe_impl="grouped"`, model.py 716 and 743) runs in bf16 even so,
over 64 windows of 512 tokens drawn with seed 0 from the base run's held-out
split (`data/v2/val.u16`), the same draw as `measure_attention.py`. Forward
hooks only watch; nothing in the model is changed. It records:

- layer 0's SwiGLU: the sign of the gate's inputs, how many of the 2,816 inner
  channels carry a token, and the size of the layer's output against the
  residual stream it is added to (Ch 16);
- every router: the stored balancing biases, the sigmoid scores and the
  combine weights, the regularisers' values, and how often the chosen experts
  change if the bias is dropped, rounded to bf16, re-centred, computed in
  float64, or computed under bf16 autocast as the code did before its fix
  (Ch 17, 18);
- the three dispatch paths on one real layer's inputs: their agreement, and
  their time on this machine (Ch 18).

It also reads the training log `v2_train_run.log` for the logged imbalance and
regulariser. No held-out text is printed or saved: only statistics. Results go
to part_iv_b_trained.json in the current directory, or with --save to
book/measurements/part_iv_b_trained.json.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Only `--save` writes the book's copy; otherwise the JSON goes to the current directory.
OUT = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_iv_b_trained.json"
WINDOWS, SEQ, BATCH, SEED = 64, 512, 4, 0
EOS = 32767


def load():
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = M.AnuLM(cfg)
    m.load_state_dict(ck["model"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), cfg, dev


def topk_set(scores, k):
    return scores.topk(k, dim=-1)[1].sort(-1)[0]


def imbalance(idx, E):
    load = torch.bincount(idx.flatten(), minlength=E).double()
    return (load.max() / load.mean()).item()


class RouterStats:
    """Accumulates one MoE layer's routing statistics over all windows."""

    def __init__(self, router, E, k):
        self.r, self.E, self.k = router, E, k
        self.n = 0
        self.sum = {}
        self.idx = {name: [] for name in ("stored", "no_bias", "bf16_bias", "centred", "fp64", "bf16_autocast")}

    def add(self, key, v):
        self.sum[key] = self.sum.get(key, 0.0) + float(v)

    def __call__(self, mod, inp, out):
        x = inp[0]                                                     # (N, D) fp32 (the experts before it ran in bf16)
        idx_model = out[0]
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            W, b = mod.weight.float(), mod.expert_bias
            logits = F.linear(x.float(), W)                            # (N, E)
            scores = torch.sigmoid(logits)
            stored = topk_set(scores + b, self.k)
            assert torch.equal(stored, idx_model.sort(-1)[0]), "recomputed choice differs from the router's"
            variants = {
                "stored": stored,
                "no_bias": topk_set(scores, self.k),
                "bf16_bias": topk_set(scores + b.bfloat16().float(), self.k),
                "centred": topk_set(scores + (b - b.mean()), self.k),
                "fp64": topk_set(torch.sigmoid(logits.double()) + b.double(), self.k),
            }
        with torch.no_grad(), torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            variants["bf16_autocast"] = mod._gate(x.float(), SEQ)[0].sort(-1)[0]   # the gate as it ran before the fix
        for name, v in variants.items():
            self.idx[name].append(v.cpu())
        N = x.shape[0]
        self.n += N
        s_sorted = scores.sort(-1, descending=True)[0]
        c_sorted = (scores + b).sort(-1, descending=True)[0]
        w = out[1].float()
        self.sum["score_by_rank"] = self.sum.get("score_by_rank", 0) + s_sorted.sum(0).double().cpu()
        self.add("score_mean", scores.sum())
        self.add("score_gt_half", (scores > 0.5).float().sum())
        self.add("score_rank1", s_sorted[:, 0].sum())
        self.add("score_rank4", s_sorted[:, self.k - 1].sum())
        self.add("score_rank5", s_sorted[:, self.k].sum())
        self.add("score_rank24", s_sorted[:, -1].sum())
        self.add("choice_gap45", (c_sorted[:, self.k - 1] - c_sorted[:, self.k]).sum())
        self.add("gap_lt_fp32_ulp", ((c_sorted[:, self.k - 1] - c_sorted[:, self.k]) <
                                     torch.finfo(torch.float32).eps * b.abs().max()).float().sum())
        self.add("w_max", w.max(-1)[0].sum())
        self.add("w_min", w.min(-1)[0].sum())
        self.add("w_sum", w.sum(-1).sum())
        # the chosen weights against a softmax over the same experts' logits, scaled the same way
        soft = logits.gather(1, idx_model).softmax(-1) * mod.routed_scaling_factor
        self.add("softmax_diff", ((w - soft).abs() / mod.routed_scaling_factor).mean(-1).sum())
        self.add("logit_mean", logits.mean(-1).sum())
        self.add("logit_sq", logits.pow(2).mean(-1).sum())
        z = torch.logsumexp(logits, dim=-1)
        self.add("z_sq", z.pow(2).sum())
        # DeepSeek-V3's sequence-wise balance term, exactly as Router._gate computes it, per window of SEQ
        onehot = F.one_hot(idx_model, self.E).sum(1).float()
        f = onehot.view(-1, SEQ, self.E).mean(1) * (self.E / self.k)
        P = (scores / scores.sum(-1, keepdim=True)).view(-1, SEQ, self.E).mean(1)
        self.add("seq_balance", (f * P).sum(-1).sum())
        self.add("seq_windows", f.shape[0])

    def result(self):
        n = self.n
        idx = {k: torch.cat(v) for k, v in self.idx.items()}
        st = idx["stored"]
        b = self.r.expert_bias.double().cpu()
        res = {k: v / n for k, v in self.sum.items()
               if k not in ("seq_balance", "seq_windows", "gap_lt_fp32_ulp", "score_by_rank")}
        res["score_by_rank"] = (self.sum["score_by_rank"] / n).tolist()
        res["score_sum"] = sum(res["score_by_rank"])
        res["score_gt_half"] /= self.E
        res["score_mean"] /= self.E
        res["logit_std"] = math.sqrt(res.pop("logit_sq") - res["logit_mean"] ** 2)
        res["seq_balance"] = self.sum["seq_balance"] / self.sum["seq_windows"]
        res["gap_lt_fp32_ulp_frac"] = self.sum["gap_lt_fp32_ulp"] / n
        res["tokens"] = n
        res["bias_mean"] = b.mean().item()
        res["bias_spread"] = (b.max() - b.min()).item()
        res["bias_std"] = b.std().item()
        res["bias_centred"] = (b - b.mean()).tolist()
        res["fp32_ulp_at_bias"] = (torch.nextafter(torch.tensor(b.mean().item(), dtype=torch.float32),
                                                   torch.tensor(1e9)) - torch.tensor(b.mean().item())).item()
        res["bf16_ulp_at_bias"] = (torch.nextafter(torch.tensor(b.mean().item(), dtype=torch.bfloat16),
                                                   torch.tensor(1e9, dtype=torch.bfloat16)).float()
                                   - torch.tensor(b.mean().item()).bfloat16().float()).item()
        res["bf16_bias_distinct_values"] = len(set(self.r.expert_bias.bfloat16().float().tolist()))
        for name, v in idx.items():
            res[f"imbalance_{name}"] = imbalance(v, self.E)
            if name != "stored":
                res[f"changed_{name}"] = (v != st).any(-1).double().mean().item()
                # share of individual (token, choice) slots whose expert is not in the stored set
                res[f"changed_slots_{name}"] = 1 - (v.unsqueeze(-1) == st.unsqueeze(-2)).any(-1).double().mean().item()
        res["changed_fp64_vs_centred"] = (idx["fp64"] != idx["centred"]).any(-1).double().mean().item()
        return res


def swiglu_hook(store):
    def hook(mod, inp, out):
        u = inp[0]
        with torch.no_grad():
            g = mod.gate_proj(u)
            up = mod.up_proj(u)
            h = F.silu(g) * up                                           # (B, S, I)
            store["n"] += g.numel() // g.shape[-1]
            store["gate_neg"] += (g < 0).float().mean(-1).sum().item()
            store["gate_lt_m4"] += (g < -4).float().mean(-1).sum().item()
            h2 = h.pow(2)
            pr = h2.sum(-1).pow(2) / h2.pow(2).sum(-1)                   # participation ratio, channels in play
            store["in_play"] += pr.sum().item()
            top = h2.sort(-1, descending=True)[0].cumsum(-1) / h2.sum(-1, keepdim=True)
            store["ch90"] += ((top < 0.9).sum(-1) + 1).float().sum().item()
            store["cum"] = store.get("cum", 0) + top[..., :256].reshape(-1, 256).sum(0).double().cpu()
            store["out_norm"] += out.norm(dim=-1).sum().item()
            store["silu_min_seen"] = min(store["silu_min_seen"], F.silu(g).min().item())
            flat = h2.reshape(-1, h2.shape[-1])
            store["per_channel"] = store.get("per_channel", 0) + flat.sum(0).double().cpu()
            store["top_channel"] = store.get("top_channel", 0) + torch.bincount(
                flat.argmax(-1), minlength=flat.shape[-1]).cpu()
            store["h_absmax"] = max(store.get("h_absmax", 0.0), h.abs().max().item())
    return hook


def resid_hook(store, key):
    def hook(mod, inp, out):
        store[key] += inp[0].norm(dim=-1).sum().item()
    return hook


def dispatch(m, x_flat, dev):
    """The three paths on one real MoE layer's real inputs."""
    moe = m.layers[10].mlp
    out = {}
    with torch.no_grad():
        idx, w, _ = moe.router(x_flat)
        ys = moe._route_sparse(x_flat, idx, w)
        yd = moe._route_dense(x_flat, idx, w)
        out["dense_vs_sparse_fp32"] = (ys - yd).abs().max().item()
        out["scale"] = ys.abs().max().item()
        if hasattr(F, "grouped_mm"):
            yg = moe._route_grouped(x_flat, idx, w)
            out["grouped_vs_sparse_fp32"] = (yg - ys).abs().max().item()
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                ysb = moe._route_sparse(x_flat, idx, w)
            out["grouped_vs_sparse_bf16"] = (yg.float() - ysb.float()).abs().max().item()
            out["sparse_bf16_vs_fp32"] = (ysb.float() - ys).abs().max().item()
        ctx = lambda: torch.autocast(device_type=dev, dtype=torch.bfloat16)   # every path, as in training
        runs = {name: [] for name in ("sparse", "dense", "grouped")}
        for _ in range(7):                       # 7 rounds of 20 timed calls each, paths interleaved
            for name in runs:
                fn = getattr(moe, f"_route_{name}")
                for _ in range(3):               # warm-up; grouped's stacked weights are cached in eval mode
                    with ctx():
                        fn(x_flat, idx, w)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                reps = 20
                for _ in range(reps):
                    with ctx():
                        fn(x_flat, idx, w)
                if dev == "cuda":
                    torch.cuda.synchronize()
                runs[name].append((time.perf_counter() - t) / reps * 1e3)
        out["ms_per_call_bf16_autocast_runs"] = runs
        out["ms_per_call_bf16_autocast"] = {k: statistics.median(v) for k, v in runs.items()}
        out["tokens"] = x_flat.shape[0]
        out["device"] = torch.cuda.get_device_name() if dev == "cuda" else "cpu"
    return out


def read_log():
    p = ROOT / "v2_train_run.log"
    if not p.exists():
        return None
    pat = re.compile(r"^step\s+(\d+) \| loss ([\d.]+) \| aux ([\d.]+) \| .*imbalance\s+([\d.]+)x \(median ([\d.]+)x\)")
    rows = [tuple(float(g) for g in mt.groups()) for line in p.read_text(encoding="utf-8", errors="replace").splitlines()
            if (mt := pat.match(line))]
    rows.sort()
    dedup = {int(r[0]): r for r in rows}                   # a resumed run logs some steps twice; keep the last
    rows = [dedup[k] for k in sorted(dedup)]
    last = rows[-100:]
    return {"file": "v2_train_run.log", "logged_steps": len(rows), "first": rows[0], "last": rows[-1],
            "last100_aux_mean": statistics.mean(r[2] for r in last),
            "last100_imbalance_mean": statistics.mean(r[3] for r in last),
            "last100_median_mean": statistics.mean(r[4] for r in last),
            "last100_imbalance_range": [min(r[3] for r in last), max(r[3] for r in last)],
            "step_1000": next((r for r in rows if r[0] == 1000), None),
            "step_10000": next((r for r in rows if r[0] == 10000), None),
            "step_100000": next((r for r in rows if r[0] == 100000), None)}


def main():
    torch.manual_seed(SEED)
    m, cfg, dev = load()
    E, k = cfg.num_experts, cfg.num_experts_per_tok
    print(f"device {dev}; E = {E}, A = {k}, routed_scaling_factor {cfg.routed_scaling_factor}, "
          f"bias_update_rate {cfg.bias_update_rate}, moe_impl {cfg.moe_impl}, "
          f"seq_balance_alpha {cfg.seq_balance_alpha}, router_z_alpha {cfg.router_z_alpha}")
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(val) - SEQ - 1, size=WINDOWS)
    stats = {i: RouterStats(l.mlp.router, E, k) for i, l in enumerate(m.layers) if l.is_moe}
    sw = {"n": 0, "gate_neg": 0.0, "gate_lt_m4": 0.0, "in_play": 0.0, "ch90": 0.0, "out_norm": 0.0,
          "resid_norm": 0.0, "silu_min_seen": 0.0}
    moe_norm = {"out": 0.0, "resid": 0.0}
    hooks = [m.layers[0].mlp.register_forward_hook(swiglu_hook(sw)),
             m.layers[0].post_attention_layernorm.register_forward_hook(resid_hook(sw, "resid_norm")),
             m.layers[10].post_attention_layernorm.register_forward_hook(resid_hook(moe_norm, "resid")),
             m.layers[10].mlp.register_forward_hook(
                 lambda mod, i, o: moe_norm.__setitem__("out", moe_norm["out"] + o[0].norm(dim=-1).sum().item()))]
    hooks += [m.layers[i].mlp.router.register_forward_hook(s) for i, s in stats.items()]
    keep = {}
    hooks.append(m.layers[10].mlp.router.register_forward_hook(      # a hook must return None
        lambda mod, i, o: None if "x" in keep else keep.__setitem__("x", i[0].clone())))
    losses, tokens = [], 0
    try:
        with torch.no_grad():
            for b in range(0, WINDOWS, BATCH):
                rows = np.stack([val[s: s + SEQ + 1] for s in starts[b: b + BATCH]]).astype(np.int64)
                x = torch.from_numpy(rows[:, :-1]).to(dev)
                y = torch.from_numpy(rows[:, 1:]).to(dev)
                y = y.masked_fill(x == EOS, -1)
                _, loss = m(x, y)
                n = int((y != -1).sum())
                losses.append(loss.item() * n)
                tokens += n
    finally:
        for h in hooks:
            h.remove()
    res = {"checkpoint": "ckpt_base_v2.pt", "data": "data/v2/val.u16", "windows": WINDOWS, "seq": SEQ,
           "seed": SEED, "precision": "no autocast; fp32 except the experts (bf16 on the released grouped path)", "loss": sum(losses) / tokens, "graded_tokens": tokens,
           "config": {"E": E, "A": k, "routed_scaling_factor": cfg.routed_scaling_factor,
                      "bias_update_rate": cfg.bias_update_rate, "moe_impl": cfg.moe_impl,
                      "seq_balance_alpha": cfg.seq_balance_alpha, "router_z_alpha": cfg.router_z_alpha,
                      "n_group": cfg.n_group, "num_shared_experts": cfg.num_shared_experts,
                      "intermediate_size": cfg.intermediate_size, "moe_intermediate_size": cfg.moe_intermediate_size}}
    n = sw.pop("n")
    res["layer0_swiglu"] = {"tokens": n, "gate_negative": sw["gate_neg"] / n, "gate_below_minus4": sw["gate_lt_m4"] / n,
                            "channels_in_play": sw["in_play"] / n, "channels_for_90pct": sw["ch90"] / n,
                            "out_norm": sw["out_norm"] / n, "resid_norm": sw["resid_norm"] / n,
                            "silu_min_seen": sw["silu_min_seen"]}
    pc, tc = sw["per_channel"], sw["top_channel"]
    order = pc.argsort(descending=True)
    res["layer0_swiglu"].update({
        "top_channels": order[:8].tolist(), "top_channel_share": (pc[order[:8]] / pc.sum()).tolist(),
        "top1_share": (pc[order[0]] / pc.sum()).item(), "top5_share": (pc[order[:5]].sum() / pc.sum()).item(),
        "tokens_whose_largest_is_top5": (tc[order[:5]].sum() / tc.sum()).item(),
        "tokens_whose_largest_is_top1": (tc[order[0]] / tc.sum()).item(), "h_absmax": sw["h_absmax"],
        "cumulative_share_top_n": (sw["cum"] / n).tolist()})
    res["layer10_moe"] = {"out_norm": moe_norm["out"] / n, "resid_norm": moe_norm["resid"] / n}
    res["routers"] = {str(i): s.result() for i, s in stats.items()}
    r = res["routers"]
    res["aux_on_heldout"] = sum(cfg.seq_balance_alpha * v["seq_balance"] + cfg.router_z_alpha * v["z_sq"]
                                for v in r.values())
    res["dispatch_layer10"] = dispatch(m, keep["x"], dev)
    res["training_log"] = read_log()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    report(res)


def report(res):
    print(f"loss {res['loss']:.4f} on {res['graded_tokens']:,} graded tokens ({res['precision']})")
    s = res["layer0_swiglu"]
    print(f"layer 0 SwiGLU over {s['tokens']:,} tokens: gate input < 0 for {s['gate_negative']:.3f} of channels, "
          f"< -4 for {s['gate_below_minus4']:.3f}; channels in play {s['channels_in_play']:.1f} of 2816; "
          f"channels for 90% of sum h^2 {s['channels_for_90pct']:.1f}; |out| {s['out_norm']:.2f} vs |resid| "
          f"{s['resid_norm']:.2f} (ratio {s['out_norm'] / s['resid_norm']:.3f}); smallest SiLU seen {s['silu_min_seen']:.4f}")
    print(f"  per-channel share of sum h^2: top channels {s['top_channels'][:5]} take "
          f"{[round(v, 3) for v in s['top_channel_share'][:5]]}; top 1 {s['top1_share']:.3f}, top 5 {s['top5_share']:.3f}; "
          f"largest channel of a token is the top channel for {s['tokens_whose_largest_is_top1']:.3f} of tokens, "
          f"one of the top 5 for {s['tokens_whose_largest_is_top5']:.3f}; largest |h| {s['h_absmax']:.1f}")
    c = s["cumulative_share_top_n"]
    print("  mean share of a token's sum h^2 in its largest n channels: "
          + ", ".join(f"n={k} {c[k - 1]:.3f}" for k in (1, 2, 5, 10, 20, 50, 100, 256)))
    t = res["layer10_moe"]
    print(f"layer 10 MoE: |out| {t['out_norm']:.2f} vs |resid| {t['resid_norm']:.2f} (ratio {t['out_norm'] / t['resid_norm']:.3f})")
    r = res["routers"]
    print("layer | bias mean  spread  fp32ulp bf16ulp distinct | score mean  >.5  r1    r4    r5    r24  gap45 | "
          "w max  w min | z^2   seqbal | imb stored nobias bf16b | changed: nobias bf16bias centred fp64 autocast")
    for i, v in r.items():
        print(f"{int(i):5d} | {v['bias_mean']:8.2f} {v['bias_spread']:.4f} {v['fp32_ulp_at_bias']:.1e} "
              f"{v['bf16_ulp_at_bias']:4.1f} {v['bf16_bias_distinct_values']:3d} | {v['score_mean']:.3f} "
              f"{v['score_gt_half']:.3f} {v['score_rank1']:.3f} {v['score_rank4']:.3f} {v['score_rank5']:.3f} "
              f"{v['score_rank24']:.3f} {v['choice_gap45']:.4f} | {v['w_max']:.3f} {v['w_min']:.3f} | "
              f"{v['z_sq']:.3f} {v['seq_balance']:.3f} | {v['imbalance_stored']:.2f} {v['imbalance_no_bias']:.2f} "
              f"{v['imbalance_bf16_bias']:.2f} | {v['changed_no_bias']:.3f} {v['changed_bf16_bias']:.3f} "
              f"{v['changed_centred']:.5f} {v['changed_fp64']:.5f} {v['changed_bf16_autocast']:.4f}")

    def agg(key, f=statistics.median):
        return f([v[key] for v in r.values()])
    for i in ("1", "3", "10", "19"):
        print(f"  layer {i} mean score by rank: " + " ".join(f"{v:.3f}" for v in r[i]["score_by_rank"])
              + f"  (sum {r[i]['score_sum']:.3f})")
    for key in ("score_sum", "bias_mean", "bias_spread", "score_mean", "score_gt_half", "score_rank1", "score_rank4", "score_rank5",
                "choice_gap45", "w_max", "w_min", "w_sum", "softmax_diff", "logit_mean", "logit_std", "z_sq", "seq_balance",
                "imbalance_stored", "imbalance_no_bias", "imbalance_bf16_bias", "changed_no_bias",
                "changed_bf16_bias", "changed_slots_no_bias", "changed_slots_bf16_bias", "changed_centred", "changed_fp64", "changed_fp64_vs_centred",
                "changed_bf16_autocast", "gap_lt_fp32_ulp_frac"):
        print(f"  {key:24s} median {agg(key):.5f}  min {agg(key, min):.5f}  max {agg(key, max):.5f}  "
              f"mean {agg(key, statistics.mean):.5f}")
    seq = sum(res["config"]["seq_balance_alpha"] * v["seq_balance"] for v in r.values())
    print(f"aux the released run's regularisers would add on these windows: {res['aux_on_heldout']:.5f} "
          f"(sequence balance {seq:.5f}, z-loss {res['aux_on_heldout'] - seq:.5f})")
    d = res["dispatch_layer10"]
    print(f"dispatch on layer 10's inputs ({d['tokens']} tokens, {d['device']}): dense vs sparse fp32 "
          f"{d['dense_vs_sparse_fp32']:.2e}; grouped vs sparse fp32 {d.get('grouped_vs_sparse_fp32', float('nan')):.2e}, "
          f"vs sparse under bf16 {d.get('grouped_vs_sparse_bf16', float('nan')):.2e}; sparse bf16 vs fp32 "
          f"{d.get('sparse_bf16_vs_fp32', float('nan')):.2e}; output scale {d['scale']:.2f}")
    print("  ms per call under bf16 autocast, median [min, max] of 7 rounds of 20 calls:",
          {k2: f"{statistics.median(v2):.2f} [{min(v2):.2f}, {max(v2):.2f}]"
           for k2, v2 in d["ms_per_call_bf16_autocast_runs"].items()})
    lg = res["training_log"]
    if lg:
        print(f"training log {lg['file']}: {lg['logged_steps']} logged steps; first {lg['first']}; step 1000 "
              f"{lg['step_1000']}; step 10000 {lg['step_10000']}; step 100000 {lg['step_100000']}; last {lg['last']}")
        print(f"  last 100 logs: aux mean {lg['last100_aux_mean']:.4f}, worst-layer imbalance mean "
              f"{lg['last100_imbalance_mean']:.2f}x (range {lg['last100_imbalance_range']}), median-layer mean "
              f"{lg['last100_median_mean']:.2f}x")


if __name__ == "__main__":
    main()
