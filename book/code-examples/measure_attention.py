"""
Book example, not repository code.

Chapters 11 and 20: what the released model actually does on held-out text.

    python book/code-examples/measure_attention.py        # GPU if free, else CPU (slow)

Runs `ckpt_base_v2.pt` through `model.py` unchanged over windows of the base
run's own held-out split (`data/v2/val.u16`, documents hashed out of training)
and records, for every layer and head:

- the mean weight on the previous token, on the token itself and on the first
  token of the window;
- the mean "keys in play", e^entropy of a weight row, against the same number
  for uniform attention over the same keys;
- the mean distance, in tokens, between a query and what it attends to.

It also records the loss on those windows and how evenly each MoE layer spreads
its tokens over the experts, and the weights of one sentence written for the
book. No text from the held-out split is printed or saved: only statistics.

Attention weights are not something `model.py` computes (it calls a fused
kernel), so the script wraps `model._sdpa`: the wrapper calls the original and,
beside it, recomputes softmax(q kᵀ · scale + mask) from the same arguments.
Results go to ./attention.json, or to book/measurements/attention.json with --save.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = ROOT / "book" / "measurements" / "attention.json"
WINDOWS, SEQ, BATCH, SEED = 64, 512, 4, 0
SENTENCE = "The dog bit the man because it was hungry."


def load():
    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    m = M.AnuLM(cfg)
    m.load_state_dict(ck["model"], strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return m.to(dev).eval(), cfg, dev


class Recorder:
    """Wraps model._sdpa. One call per layer per forward, in layer order."""

    def __init__(self, n_layer):
        self.n_layer, self.calls, self.keep = n_layer, 0, None
        self.stats = None

    def __call__(self, q, k, v, mask, S, scale, drop):
        y = self.orig(q, k, v, mask, S, scale, drop)
        layer = self.calls % self.n_layer
        self.calls += 1
        T = k.shape[-2]
        s = (q.float() @ k.float().transpose(-2, -1)) * scale                  # (B, H, S, T)
        if mask is None:
            mask = torch.ones(S, T, dtype=torch.bool, device=q.device).tril(T - S) if S > 1 else None
        if mask is not None:
            s = s.masked_fill(~mask, float("-inf"))
        w = s.softmax(-1)
        # the recomputed weights must reproduce the kernel's output, or every statistic below is wrong
        err = ((w @ v.float()) - y.float()).abs().max().item()
        assert err < 1e-3, f"layer {layer}: recomputed attention differs from _sdpa by {err:.2e}"
        self.max_err = max(getattr(self, "max_err", 0.0), err)
        if self.keep is not None:
            self.keep[layer] = w[0].cpu()
        if self.stats is not None:
            self.stats.add(layer, w, mask)
        return y


class Stats:
    def __init__(self, L, H, S):
        z = lambda: torch.zeros(L, H, dtype=torch.float64)
        self.prev, self.self_, self.first, self.inplay, self.uniform, self.dist = z(), z(), z(), z(), z(), z()
        self.first_visible = z()                      # weight on position 0, only over queries that can see it
        self.n = 0
        self.S = S
        self.lo = 64                                  # queries from position 64 on, so every row has many keys

    def add(self, layer, w, mask):
        B, H, S, T = w.shape
        pos = torch.arange(S, device=w.device)
        rows = w[:, :, self.lo:, :]                                                   # (B, H, S-lo, T)
        qpos = pos[self.lo:]
        self.prev[layer] += rows.gather(-1, (qpos - 1).view(1, 1, -1, 1).expand(B, H, -1, 1)).mean((0, 2, 3)).double().cpu()
        self.self_[layer] += rows.gather(-1, qpos.view(1, 1, -1, 1).expand(B, H, -1, 1)).mean((0, 2, 3)).double().cpu()
        self.first[layer] += rows[..., 0].mean((0, 2)).double().cpu()
        sees0 = mask[self.lo:, 0] if mask is not None else torch.ones_like(qpos, dtype=torch.bool)
        self.first_visible[layer] += rows[:, :, sees0, 0].mean((0, 2)).double().cpu()
        ent = -(rows * rows.clamp_min(1e-30).log()).sum(-1)                           # (B, H, S-lo)
        self.inplay[layer] += ent.exp().mean((0, 2)).double().cpu()
        allowed = mask[self.lo:].sum(-1).float() if mask is not None else (qpos + 1).float()
        self.uniform[layer] += allowed.mean().double().cpu().expand(H)                # uniform over allowed keys
        d = (qpos.view(-1, 1) - pos.view(1, -1)).clamp_min(0).float()                  # (S-lo, T)
        self.dist[layer] += (rows * d).sum(-1).mean((0, 2)).double().cpu()
        if layer == 0:
            self.n += 1

    def result(self):
        f = lambda t: (t / self.n).tolist()
        return {"prev": f(self.prev), "self": f(self.self_), "first": f(self.first),
                "first_where_visible": f(self.first_visible), "keys_in_play": f(self.inplay),
                "uniform_keys": f(self.uniform), "mean_distance": f(self.dist)}


def main():
    torch.manual_seed(SEED)
    m, cfg, dev = load()
    L, H = cfg.n_layer, cfg.n_head
    print(f"device {dev}; {L} layers × {H} heads; window of attention on layers 0–{cfg.max_window_layers - 1}: "
          f"{cfg.sliding_window}")
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, len(val) - SEQ - 1, size=WINDOWS)
    rec = Recorder(L)
    rec.orig = M._sdpa
    rec.stats = Stats(L, H, SEQ)
    loads = torch.zeros(L, cfg.num_experts, dtype=torch.float64)
    hooks = []

    def count(i):
        def hook(mod, inp, out):                   # must return None: a returned value replaces the output
            loads[i].add_(torch.bincount(out[0].flatten(), minlength=cfg.num_experts).double().cpu())
        return hook

    for i, layer in enumerate(m.layers):
        if layer.is_moe:
            hooks.append(layer.mlp.router.register_forward_hook(count(i)))
    M._sdpa = rec
    losses, tokens = [], 0
    try:
        with torch.no_grad():
            for b in range(0, WINDOWS, BATCH):
                rows = np.stack([val[s: s + SEQ + 1] for s in starts[b: b + BATCH]]).astype(np.int64)
                x = torch.from_numpy(rows[:, :-1]).to(dev)
                y = torch.from_numpy(rows[:, 1:]).to(dev)
                y = y.masked_fill(x == 32767, -1)            # do not grade the token after a document end
                _, loss = m(x, y)
                n = int((y != -1).sum())
                losses.append(loss.item() * n)
                tokens += n
            assert rec.calls == L * (WINDOWS // BATCH), \
                f"{rec.calls} attention calls for {WINDOWS // BATCH} forwards of {L} layers: is NATIVE_GQA on?"
            st = rec.stats.result()
            rec.stats = None
            for h in hooks:                                  # the sentence must not count towards expert loads
                h.remove()
            hooks = []
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(ROOT / "release" / "AnuLM-Base-v2-400M" / "tokenizer.json"))
            enc = tok.encode(SENTENCE)
            rec.keep = {}
            m(torch.tensor([enc.ids], device=dev))
            sentence_w = {l: w for l, w in rec.keep.items()}
    finally:
        M._sdpa = rec.orig
        for h in hooks:
            h.remove()

    res = {"checkpoint": "ckpt_base_v2.pt", "data": "data/v2/val.u16", "windows": WINDOWS, "seq": SEQ, "seed": SEED,
           "precision": "fp32 without autocast, except the experts: the checkpoint's moe_impl='grouped' runs them in bf16 (model.py 716, 743)", "queries_from_position": 64, "graded_tokens": tokens,
           "ungraded_after_eos": WINDOWS * SEQ - tokens, "loss": sum(losses) / tokens,
           "ln_vocab": math.log(cfg.vocab_size), "max_recompute_error": rec.max_err}
    res.update(st)
    frac = (loads / loads.sum(-1, keepdim=True).clamp_min(1)).tolist()
    res["expert_load"] = {str(i): frac[i] for i in range(L) if m.layers[i].is_moe}
    words = [t.replace("Ġ", " ").strip() or t for t in enc.tokens]
    res["sentence"] = {"text": SENTENCE, "tokens": words,
                       "weights": {str(l): w.tolist() for l, w in sentence_w.items()}}
    out = OUT if "--save" in sys.argv else Path("attention.json")   # the book's copy changes only with --save
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res), encoding="utf-8")
    report(res, cfg)



def report(res, cfg):
    L, H = cfg.n_layer, cfg.n_head
    print(f"loss on {res['graded_tokens']:,} held-out tokens ({res['ungraded_after_eos']} targets after an end of "
          f"document not graded; {res['precision']}): {res['loss']:.4f} nats (ln V = {res['ln_vocab']:.4f})")
    print(f"recomputed weights · v against _sdpa's output: largest difference {res['max_recompute_error']:.1e}")
    fv = np.array(res["first_where_visible"])
    print("first-position weight over the queries that can see position 0, mean over heads [max head]: "
          + "; ".join(f"L{l} {fv[l].mean():.3f} [{fv[l].max():.3f}]" for l in range(L)))
    prev, first, selfw = np.array(res["prev"]), np.array(res["first"]), np.array(res["self"])
    inplay, uni, dist = np.array(res["keys_in_play"]), np.array(res["uniform_keys"]), np.array(res["mean_distance"])
    print("per layer: mean over heads [max head] of: previous-token weight | first-token weight | keys in play "
          "(uniform) | mean distance")
    for l in range(L):
        print(f"  layer {l:2d}: prev {prev[l].mean():.3f} [{prev[l].max():.3f} h{prev[l].argmax()}] | "
              f"first {first[l].mean():.3f} [{first[l].max():.3f}] | in play {inplay[l].mean():6.1f} "
              f"[min {inplay[l].min():5.1f}] ({uni[l][0]:.0f}) | dist {dist[l].mean():6.1f}")
    flat = [(prev[l, h], l, h) for l in range(L) for h in range(H)]
    flat.sort(reverse=True)
    print("strongest previous-token heads:", ", ".join(f"L{l}.h{h} {p:.3f}" for p, l, h in flat[:5]))
    print(f"heads with previous-token weight > 0.5: {sum(p > 0.5 for p, _, _ in flat)} of {L * H}")
    sink = [(first[l, h], l, h) for l in range(L) for h in range(H)]
    sink.sort(reverse=True)
    print(f"heads putting > 0.5 of their weight on the first token: {sum(s > 0.5 for s, _, _ in sink)} of {L * H}; "
          "strongest: " + ", ".join(f"L{l}.h{h} {s:.3f}" for s, l, h in sink[:5]))
    for i, f in res["expert_load"].items():
        f = np.array(f)
        print(f"  MoE layer {int(i):2d}: busiest expert {f.max():.3f}, idlest {f.min():.3f}, "
              f"ratio {f.max() / max(f.min(), 1e-9):.2f} (uniform {1 / len(f):.3f})")
    s = res["sentence"]
    toks = s["tokens"]
    print("sentence tokens:", toks)
    bit, it = toks.index("bit"), toks.index("it")
    best = []
    for l, w in s["weights"].items():
        w = np.array(w)                                   # (H, S, S)
        for h in range(H):
            best.append((w[h, bit, toks.index("dog")], w[h, it, toks.index("dog")], w[h, it, toks.index("man")], int(l), h))
    b = max(best)
    print(f"largest weight from 'bit' onto 'dog': L{b[3]}.h{b[4]} {b[0]:.3f}; row: "
          + ", ".join(f"{t} {x:.2f}" for t, x in zip(toks, np.array(s['weights'][str(b[3])])[b[4], bit])))
    c = max(best, key=lambda r: r[1])
    print(f"largest weight from 'it' onto 'dog': L{c[3]}.h{c[4]} {c[1]:.3f} (onto 'man' in that head {c[2]:.3f}); row: "
          + ", ".join(f"{t} {x:.2f}" for t, x in zip(toks, np.array(s['weights'][str(c[3])])[c[4], it])))
    d = max(best, key=lambda r: r[2])
    print(f"largest weight from 'it' onto 'man': L{d[3]}.h{d[4]} {d[2]:.3f}")


if __name__ == "__main__":
    main()
