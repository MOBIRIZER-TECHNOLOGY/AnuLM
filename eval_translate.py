"""
Translation quality: chrF on FLORES-200 devtest, both directions.

    python eval_translate.py ckpt_translate.pt --device cuda
    python eval_translate.py ckpt_multi36k.pt ckpt_translate.pt --limit 200

chrF (Popovic 2015; the chrF2 variant sacrebleu reports) compares character
n-grams up to 6 between hypothesis and reference with recall weighted 2x.
It is the standard metric for Indic MT because it does not depend on word
segmentation. Reference points on FLORES devtest, English -> Hindi: Google
Translate and IndicTrans2 report chrF around 58-62; NLLB-600M around 55.
Hindi -> English scores run higher for every system.

Greedy decoding, one sentence at a time, cut at EOS or the first newline.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from bpe import BPE
from make_qa import PROMPTS
from model import AnuLM


def chrf(hyps: list[str], refs: list[str], n: int = 6, beta: float = 2.0) -> float:
    """Corpus chrF: n-gram counts pooled over the corpus, then one F-beta per
    order averaged -- sacrebleu's chrF2 definition (word order 0)."""
    def grams(s, k):
        s = s.replace(" ", "")
        return Counter(s[i:i + k] for i in range(len(s) - k + 1))
    prec, rec = [], []
    for k in range(1, n + 1):
        match = hyp_total = ref_total = 0
        for h, r in zip(hyps, refs):
            hg, rg = grams(h, k), grams(r, k)
            match += sum((hg & rg).values())
            hyp_total += sum(hg.values())
            ref_total += sum(rg.values())
        prec.append(match / hyp_total if hyp_total else 0.0)
        rec.append(match / ref_total if ref_total else 0.0)
    p, r = sum(prec) / n, sum(rec) / n
    if p + r == 0:
        return 0.0
    return 100 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


@torch.no_grad()
def score(ckpt: str, items: list[dict], device: str, max_new: int, show: int) -> dict:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if device.startswith("cuda"):
        cfg = replace(cfg, moe_impl="grouped")
    m = AnuLM(cfg).to(device).eval()
    m.load_state_dict(ck["model"])
    tok = BPE.load(cfg.tokenizer_path)
    templates = {**PROMPTS, **(ck.get("qa_templates") or {})}
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))
    bs = cfg.block_size
    out = {"en-hi": ([], []), "hi-en": ([], [])}
    samples = []
    t0 = time.time()
    for i, it in enumerate(items):
        text = templates[it["lang"]].format(q=it["question"])
        ids = tok.encode(text)[-(bs - max_new):]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with ac:
            gen = m.generate(x, max_new, temperature=1.0, top_k=1, eos_id=tok.eos_id)
        hyp = tok.decode(gen[0, len(ids):].tolist()).strip().split("\n")[0].strip()
        out[it["lang"]][0].append(hyp)
        out[it["lang"]][1].append(it["answer"])
        if len(samples) < show:
            samples.append((it["lang"], it["question"], hyp, it["answer"]))
        print(f"\r  {Path(ckpt).name}: {i + 1}/{len(items)}  ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    return {"ckpt": ckpt, "samples": samples,
            **{d: chrf(h, r) for d, (h, r) in out.items() if h}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--data", default="data/flores_devtest.jsonl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="sentences per direction")
    p.add_argument("--max-new", type=int, default=128)
    p.add_argument("--show", type=int, default=6)
    args = p.parse_args()
    # split("\n"): splitlines() also breaks on U+2028 and friends, which
    # json.dumps(ensure_ascii=False) leaves raw inside strings (see finetune.py).
    items = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").split("\n") if l.strip()]
    if args.limit:
        items = [it for d in ("en-hi", "hi-en") for it in [x for x in items if x["lang"] == d][: args.limit]]
    print(f"{args.data}: {len(items)} items, greedy, chrF, device {args.device}\n")
    results = [score(c, items, args.device, args.max_new, args.show) for c in args.ckpts]
    print(f"\n{'checkpoint':26s} {'en->hi chrF':>12s} {'hi->en chrF':>12s}")
    print("-" * 52)
    for r in results:
        print(f"{Path(r['ckpt']).name:26s} {r.get('en-hi', 0):12.1f} {r.get('hi-en', 0):12.1f}")
    print("\nreference (FLORES devtest, en->hi): NLLB-600M ~55, IndicTrans2 / Google ~60")
    for r in results:
        print(f"\n=== {Path(r['ckpt']).name}")
        for lang, q, hyp, ref in r["samples"]:
            print(f"  [{lang}] {q}\n     -> {hyp}\n     ref {ref}")


if __name__ == "__main__":
    main()
