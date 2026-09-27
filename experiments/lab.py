"""AnuLM lab: fast, fair A/B runs of new training ideas on the real 3-language corpus.

    python lab.py NAME [--tokens 25e6] [--seed 0] [idea flags...]

Every run: same proxy MoE (8 layers, 512 wide, 16 experts top-2 + shared, vocab
32,768), same data order for a given seed, same WSD schedule, same 256 strided
held-out windows. Records held-out loss at 10 points with the TRAINING wall time
at each (eval time excluded), so ideas are compared both per token and per
GPU-second. One JSON line per run in lab_results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import AnuLM, AnuLMConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "ctx_mix.multi32k.bin")
TOK = os.path.join(ROOT, "data", "multi32k.json")
V = 32768
DEV = "cuda"


def proxy_cfg(**kw):
    c = dict(vocab_size=V, block_size=512, n_layer=8, hidden_size=512, n_head=8, head_dim=64,
             n_kv_head=2, num_experts=16, num_experts_per_tok=2, num_shared_experts=1,
             moe_intermediate_size=256, intermediate_size=1536, rope_theta=1e6,
             moe_impl="grouped", bias_update_rate=3e-3)
    c.update(kw)
    return AnuLMConfig(**c)


# ---------------------------------------------------------------- idea: n-gram prior
def build_ngram_prior(train_ids: torch.Tensor, K: int):
    """Unigram log-probs (V,) and, per previous token, the top-K next tokens with
    their bigram log-prob MINUS the unigram log-prob (a residual), from counts."""
    ids = train_ids.to(DEV).long()
    uni = torch.bincount(ids, minlength=V).float() + 0.5
    log_uni = (uni / uni.sum()).log()
    pairs = ids[:-1] * V + ids[1:]
    keys, cnt = torch.unique(pairs, return_counts=True)
    prev, nxt = keys // V, keys % V
    prev_tot = torch.bincount(ids[:-1], minlength=V).float()
    # Interpolated bigram: lambda = n / (n + 5), so rare contexts lean on unigram.
    n = prev_tot[prev]
    lam = n / (n + 5.0)
    p_bi = lam * cnt.float() / n.clamp(min=1) + (1 - lam) * uni[nxt] / uni.sum()
    resid = p_bi.log() - log_uni[nxt]
    # Top-K per prev: sort by (prev, -count) and take the first K of each run.
    order = torch.argsort(prev * (cnt.max() + 1) - cnt, stable=True)
    prev_s, nxt_s, res_s = prev[order], nxt[order], resid[order]
    start = torch.searchsorted(prev_s, torch.arange(V, device=DEV))
    rank = torch.arange(len(prev_s), device=DEV) - start[prev_s]
    keep = rank < K
    tab_idx = torch.zeros(V, K, dtype=torch.long, device=DEV)
    tab_val = torch.zeros(V, K, device=DEV)
    tab_idx[prev_s[keep], rank[keep]] = nxt_s[keep]
    tab_val[prev_s[keep], rank[keep]] = res_s[keep]
    return log_uni, tab_idx, tab_val


# ---------------------------------------------------------------- idea: merge-tree embeddings
def merge_levels():
    merges = json.load(open(TOK, encoding="utf-8"))["merges"]
    depth = [0] * 256
    parents = {}
    for i, (a, b) in enumerate(merges):
        t = 256 + i
        depth.append(1 + max(depth[a], depth[b]))
        parents[t] = (a, b)
    levels = {}
    for t, (a, b) in parents.items():
        levels.setdefault(depth[t], []).append((t, a, b))
    out = []
    for d in sorted(levels):
        t, a, b = zip(*levels[d])
        out.append(tuple(torch.tensor(x, device=DEV) for x in (t, a, b)))
    return out


def merge_matrix():
    """Sparse (V, V): row t holds t's own weight 1 plus 0.5**k for each ancestor
    k levels up (summed over paths), i.e. E = P @ W unrolls E[t] = W[t] + (E[a] + E[b]) / 2."""
    merges = json.load(open(TOK, encoding='utf-8'))['merges']
    coef = [{i: 1.0} for i in range(256)]
    for a_, b_ in merges:
        c = {len(coef): 1.0}
        for q in (a_, b_):
            for k, v in coef[q].items():
                c[k] = c.get(k, 0.0) + 0.5 * v
        coef.append(c)
    while len(coef) < V:
        coef.append({len(coef): 1.0})
    r, cidx, val = [], [], []
    for t, c in enumerate(coef):
        for k, v in c.items():
            r.append(t); cidx.append(k); val.append(v)
    return torch.sparse_coo_tensor(torch.tensor([r, cidx]), torch.tensor(val), (V, V)).coalesce().to(DEV)


class Lab(nn.Module):
    def __init__(self, a):
        super().__init__()
        self.a = a
        self.base = AnuLM(proxy_cfg(n_layer=a.layers)).to(DEV)
        D = self.base.cfg.hidden_size
        if a.tie:
            self.base.lm_head.weight = self.base.embed_tokens.weight
        if a.hash_rows:
            self.h2 = nn.Embedding(a.hash_rows, a.hash_dim)
            self.h3 = nn.Embedding(a.hash_rows, a.hash_dim)
            nn.init.normal_(self.h2.weight, std=0.02)
            nn.init.normal_(self.h3.weight, std=0.02)
            self.hproj = nn.Linear(a.hash_dim, D, bias=False)
            nn.init.zeros_(self.hproj.weight)
        if a.merge_tree:
            self.levels = merge_levels()
            self.merge_tables()
            self.mgate = nn.Parameter(torch.tensor(a.merge_tree))
        if a.merge_out:
            self.P = merge_matrix()
        if a.prior_k:
            self.prior_w = nn.Parameter(torch.tensor([1.0, 1.0]))
        if a.cache_prior:
            self.cache_w = nn.Parameter(torch.tensor(a.cache_prior))
        self.schedule = list(range(a.layers))
        if a.loop:
            lo, hi = map(int, a.loop.split(":"))
            self.schedule = list(range(hi)) + list(range(lo, hi)) + list(range(hi, a.layers))
        if a.grow:
            # shallow phase: the dense layer 0, then every odd MoE layer; each
            # odd layer i is later copied into i+1 (both MoE, same shapes).
            self.schedule = [0] + list(range(1, a.layers, 2))

    def embed(self, x):
        W = self.base.embed_tokens.weight
        if self.a.merge_tree and self.a.merge_full:
            E = W
            for t, pa, pb in self.levels:
                E = E.index_copy(0, t, W[t] + self.mgate * 0.5 * (E[pa] + E[pb]))
            h = F.embedding(x, E)
        elif self.a.merge_tree:
            h = self.merge_embed(x, W)
        else:
            h = F.embedding(x, W)
        if self.a.hash_rows:
            R = self.a.hash_rows
            p1 = F.pad(x[:, :-1], (1, 0))
            p2 = F.pad(x[:, :-2], (2, 0))
            b2 = (x * 1000003 + p1 * 999331) % R
            b3 = (x * 1000003 + p1 * 999331 + p2 * 998353) % R
            h = h + self.hproj(self.h2(b2) + self.h3(b3))
        return h

    def merge_tables(self):
        """Per-token parents (-1 for bytes / EOS) and depth, as GPU tensors."""
        merges = json.load(open(TOK, encoding="utf-8"))["merges"]
        pa = torch.full((V,), -1, dtype=torch.long)
        pb = torch.full((V,), -1, dtype=torch.long)
        depth = torch.zeros(V, dtype=torch.long)
        for i, (a, b) in enumerate(merges):
            t = 256 + i
            pa[t], pb[t] = a, b
            depth[t] = 1 + max(depth[a], depth[b])
        self.mt = (pa.to(DEV), pb.to(DEV), depth.to(DEV), int(depth.max()))

    def merge_embed(self, x, W):
        """The same E[t] = W[t] + g/2 (E[a] + E[b]) recursion, computed only for
        the tokens in this batch and their ancestors instead of the whole
        vocabulary: a few thousand rows per level instead of 32,768."""
        pa, pb, depth, maxd = self.mt
        S = torch.unique(x)
        frontier = S
        for _ in range(maxd):
            f = frontier[pa[frontier] >= 0]
            if f.numel() == 0:
                break
            frontier = torch.unique(torch.cat([pa[f], pb[f]]))
            S = torch.cat([S, frontier])
        S = torch.unique(S)
        loc = torch.full((V,), -1, dtype=torch.long, device=x.device)
        loc[S] = torch.arange(len(S), device=x.device)
        E = W[S]
        dS = depth[S]
        for lv in range(1, maxd + 1):
            rows = (dS == lv).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            t = S[rows]
            E = E.index_copy(0, rows, W[t] + self.mgate * 0.5 * (E[loc[pa[t]]] + E[loc[pb[t]]]))
        return E[loc[x]]

    def ngram_logp(self, x):
        lu, ti, tv = self.prior
        z = lu.expand(x.shape[0], x.shape[1], V).scatter_add(-1, ti[x], tv[x])
        return F.log_softmax(z, dim=-1)

    def forward(self, x, y=None, per_token=False, teach=0.0):
        b = self.base
        h = self.embed(x)
        S = x.shape[1]
        cos, sin = b.rotary(S, h.device, h.dtype)
        for i in self.schedule:
            if self.a.ckpt and self.training:
                h, _ = torch.utils.checkpoint.checkpoint(b.layers[i], h, cos, sin, use_reentrant=False)
            else:
                h, _ = b.layers[i](h, cos, sin)
        h = b.norm(h)
        if self.a.merge_out:
            # Output rows composed through the merge tree too: a rare token's
            # classifier row is its own row plus half of each parent's (recursively).
            with torch.autocast('cuda', enabled=False):
                Wout = torch.sparse.mm(self.P, b.lm_head.weight.float())
            logits = F.linear(h, Wout.to(h.dtype)).float()
        else:
            logits = b.lm_head(h).float()
        if self.a.prior_k:
            lu, ti, tv = self.prior
            bias = lu.expand_as(logits) * self.prior_w[0]
            bias = bias.scatter_add(-1, ti[x], tv[x] * self.prior_w[1])
            logits = logits + bias
        if self.a.cache_prior:
            # Every token already seen in the window (inputs up to and including
            # position t) gets +w per occurrence at position t.
            Bq, Sq = x.shape
            idx = x[:, None, :].expand(Bq, Sq, Sq)
            src = torch.ones(Sq, Sq, device=x.device).tril().expand(Bq, Sq, Sq) * self.cache_w
            logits = logits.scatter_add(-1, idx, src)
        loss = F.cross_entropy(logits.view(-1, V), y.reshape(-1), reduction="none")
        if teach > 0:
            kd = F.kl_div(F.log_softmax(logits, -1), self.ngram_logp(x), log_target=True,
                          reduction="none").sum(-1).view(-1)
            loss = loss + teach * kd
        return loss if per_token else loss.mean()

    def grow_now(self, opt):
        """Shallow -> full depth: layer i (even) is copied into i+1, weights and
        AdamW moments both, so the deep model starts as the shallow one with
        every odd block applied twice."""
        L = self.base.layers
        for i in range(1, self.a.layers - 1, 2):
            for (n, ps), (_, pd) in zip(L[i].named_parameters(), L[i + 1].named_parameters()):
                pd.data.copy_(ps.data)
                for o in opt:
                    if ps in o.state:
                        o.state[pd] = {k: (v.clone() if torch.is_tensor(v) else v)
                                       for k, v in o.state[ps].items()}
            for (n, bs), (_, bd) in zip(L[i].named_buffers(), L[i + 1].named_buffers()):
                bd.data.copy_(bs.data)
        self.schedule = list(range(self.a.layers))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--tokens", type=float, default=25e6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--ckpt", action="store_true")
    p.add_argument("--tie", action="store_true")
    p.add_argument("--hash-rows", type=int, default=0)
    p.add_argument("--hash-dim", type=int, default=128)
    p.add_argument("--merge-tree", type=float, default=0.0)
    p.add_argument("--merge-out", action="store_true", help="merge-tree composition on the output head")
    p.add_argument("--merge-full", action="store_true", help="old whole-table merge-tree path")
    p.add_argument("--prior-k", type=int, default=0)
    p.add_argument("--loop", default="", help="LO:HI -- blocks LO..HI-1 run twice")
    p.add_argument("--grow", type=float, default=0.0, help="fraction of tokens at half depth")
    p.add_argument("--hard-frac", type=float, default=0.0, help="drop this easiest fraction of tokens from the loss")
    p.add_argument("--muon", action="store_true")
    p.add_argument("--cache-prior", type=float, default=0.0, help="init weight of the in-window token boost")
    p.add_argument("--teacher", type=float, default=0.0,
                   help="KL weight toward the bigram model's distribution, linearly to 0 by --teacher-until")
    p.add_argument("--teacher-until", type=float, default=0.5)
    p.add_argument("--emb-nodecay", action="store_true", help="no weight decay on embedding tables")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    blob = torch.load(DATA, weights_only=False)
    ids = blob["ids"]
    # The corpus file is ORDERED (mostly Hindi first, English at the end), so a
    # tail split is English-only. Interleave instead: every 10th chunk of 256k
    # tokens is held out, so the split has the corpus's own language mix.
    CH = 262144
    chunks = list(ids.split(CH))
    val = torch.cat(chunks[5::10])
    train = torch.cat([c for i, c in enumerate(chunks) if i % 10 != 5])
    T, B = 512, a.batch
    steps = int(a.tokens // (B * T))

    m = Lab(a).to(DEV)
    if a.prior_k or a.teacher:
        m.prior = build_ngram_prior(train, a.prior_k or 32)
    params = [q for q in m.parameters() if q.requires_grad]
    if a.muon:
        from muon import build_optimizers
        from train import OptimizerSet
        opts, _ = build_optimizers(m, muon_lr=0.02, adamw_lr=a.lr, weight_decay=0.1, fused=True)
        opt = OptimizerSet(opts)
    else:
        from train import OptimizerSet
        emb_ids = {id(m.base.embed_tokens.weight)} if a.emb_nodecay else set()
        if a.emb_nodecay and a.hash_rows:
            emb_ids |= {id(m.h2.weight), id(m.h3.weight)}
        dec = [q for q in params if q.dim() >= 2 and id(q) not in emb_ids]
        nod = [q for q in params if q.dim() < 2 or id(q) in emb_ids]
        opt = OptimizerSet(torch.optim.AdamW([{"params": dec, "weight_decay": 0.1},
                                              {"params": nod, "weight_decay": 0.0}],
                                             lr=a.lr, betas=(0.9, 0.95), fused=True))
    nparam = sum(q.numel() for q in {id(q): q for q in m.parameters()}.values())

    g = torch.Generator().manual_seed(1000 + a.seed)            # data order: same per seed
    val_idx = torch.linspace(0, len(val) - T - 2, 384).long()
    # Label each held-out window hindi / english / code once, so the loss can be
    # reported per language and as their macro mean (the headline number).
    from bpe import BPE
    tok = BPE.load(TOK)
    lang = []
    for j in val_idx.tolist():
        t = tok.decode(val[j:j + T].long().tolist())
        dev = sum("ऀ" <= ch <= "ॿ" for ch in t)
        lat = sum(ch.isascii() and ch.isalpha() for ch in t)
        code = sum(t.count(k) for k in ("def ", "self.", "import ", "return ", "):\n"))
        lang.append("code" if code >= 3 else ("hi" if dev > lat else "en"))
    lang_t = torch.tensor([["hi", "en", "code"].index(l) for l in lang])
    print("held-out windows:", {k: lang.count(k) for k in ("hi", "en", "code")}, flush=True)

    @torch.no_grad()
    def evaluate():
        m.eval()
        per = []
        for i in range(0, len(val_idx), 32):
            ix = val_idx[i:i + 32]
            x = torch.stack([val[j:j + T] for j in ix]).long().to(DEV)
            y = torch.stack([val[j + 1:j + 1 + T] for j in ix]).long().to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                per.append(m(x, y, per_token=True).view(len(ix), T).mean(1).float().cpu())
        m.train()
        per = torch.cat(per)
        by = {k: float(per[lang_t == i].mean()) for i, k in enumerate(("hi", "en", "code"))}
        by["macro"] = sum(by.values()) / 3
        return by

    warm, decay_start = 200, int(steps * 0.8)
    marks = {int(steps * f) - 1 for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
    curve, train_time = [], 0.0
    grown = not a.grow
    for step in range(steps):
        if not grown and step >= int(steps * a.grow):
            m.grow_now(opt.opts)
            grown = True
        lr_scale = min((step + 1) / warm, 1.0) if step < decay_start else \
            1.0 - 0.9 * (step - decay_start) / max(steps - decay_start, 1)
        opt.set_lr_scale(lr_scale)
        torch.cuda.synchronize(); t0 = time.time()
        ix = torch.randint(len(train) - T - 1, (B,), generator=g)
        x = torch.stack([train[j:j + T] for j in ix]).long().to(DEV, non_blocking=True)
        y = torch.stack([train[j + 1:j + 1 + T] for j in ix]).long().to(DEV, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if a.hard_frac:
                lt = m(x, y, per_token=True)
                k = int(lt.numel() * (1 - a.hard_frac))
                loss = lt.detach().topk(k).indices
                loss = lt[loss].mean()
            else:
                teach = a.teacher * max(0.0, 1 - step / (a.teacher_until * steps)) if a.teacher else 0.0
                loss = m(x, y, teach=teach)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        m.base.update_expert_biases()
        torch.cuda.synchronize(); train_time += time.time() - t0
        if step in marks:
            by = evaluate()
            v = by["macro"]
            curve.append({"tokens": (step + 1) * B * T, "sec": round(train_time, 1), "val": round(v, 4),
                          **{k: round(x, 4) for k, x in by.items() if k != "macro"}})
            print(f"{a.name:28s} step {step+1:5d}/{steps}  {(step+1)*B*T/1e6:5.1f}M tok  "
                  f"{train_time:6.0f}s  macro {v:.4f}  hi {by['hi']:.3f} en {by['en']:.3f} "
                  f"code {by['code']:.3f}", flush=True)
    rec = {"name": a.name, "args": vars(a), "params_M": round(nparam / 1e6, 1),
           "tok_per_s": round(steps * B * T / train_time), "final": curve[-1]["val"], "curve": curve,
           "peak_GB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "mgate": round(float(m.mgate), 3) if a.merge_tree else None}
    with open(os.path.join(HERE, "lab_results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"RESULT {a.name}: final val {rec['final']:.4f}  {rec['tok_per_s']} tok/s  "
          f"{rec['params_M']}M params  peak {rec['peak_GB']} GB", flush=True)


if __name__ == "__main__":
    main()
