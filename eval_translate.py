"""
Translation quality: chrF on FLORES-200 devtest, both directions.

    python eval_translate.py ckpt_translate.pt --device cuda
    python eval_translate.py ckpt_multi36k.pt ckpt_translate.pt --limit 200

chrF (Popovic 2015; the chrF2 variant sacrebleu reports) compares character
n-grams up to 6 between hypothesis and reference with recall weighted 2x.
It is the standard metric for Indic MT because it does not depend on word
segmentation. Reference points on FLORES devtest, English -> Hindi: Google
Translate and IndicTrans2 around 58-62, NLLB-600M around 55. Those published
figures are **chrF++** (character n-grams plus word uni- and bigrams), not the
chrF this file computes, so they are a neighbouring metric rather than the same
one -- and the two do not move together in a fixed direction. When sacrebleu is
installed this file reports chrF++ as well, for a like-for-like comparison.
Hindi -> English scores run higher for every system.

The chrF here is verified against sacrebleu.corpus_chrf: identical to three
decimals on all 1,012 FLORES devtest sentences in both directions.

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
from model import AnuLM, load_checkpoint


def chrf(hyps: list[str], refs: list[str], n: int = 6, beta: float = 2.0) -> float:
    """Corpus chrF: character n-gram counts pooled over the corpus, precision
    and recall computed per order and averaged over orders 1..n, then ONE
    F-beta from those averages -- sacrebleu 2.x's chrF2 (word order 0, no eps
    smoothing). Averaging per-order F-scores instead is the older smoothed
    variant and gives a different number."""
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
    ck = load_checkpoint(ckpt, "cpu")
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
    res = {"ckpt": ckpt, "samples": samples,
           **{d: chrf(h, r) for d, (h, r) in out.items() if h}}
    pp = chrf_pp(out)
    if pp:
        res["chrf++"] = pp
    return res


def chrf_pp(out: dict) -> dict | None:
    """chrF++ per direction via sacrebleu, if it is installed.

    Published FLORES figures for NLLB and IndicTrans2 are chrF++, so this is
    the number to set beside them. It stays optional -- the project's scoring
    needs only torch -- and chrF above is computed natively either way.
    """
    try:
        import sacrebleu
    except ImportError:
        return None
    return {d: sacrebleu.corpus_chrf(h, [r], word_order=2).score
            for d, (h, r) in out.items() if h}


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
    have_pp = all("chrf++" in r for r in results)
    head = f"{'checkpoint':26s} {'en->hi chrF':>12s} {'hi->en chrF':>12s}"
    if have_pp:
        head += f" {'en->hi chrF++':>14s} {'hi->en chrF++':>14s}"
    print("\n" + head)
    print("-" * len(head))
    for r in results:
        row = f"{Path(r['ckpt']).name:26s} {r.get('en-hi', 0):12.1f} {r.get('hi-en', 0):12.1f}"
        if have_pp:
            pp = r["chrf++"]
            row += f" {pp.get('en-hi', 0):14.1f} {pp.get('hi-en', 0):14.1f}"
        print(row)
    print("\nreference (FLORES devtest, en->hi, chrF++): NLLB-600M ~55, IndicTrans2 / Google ~60")
    if not have_pp:
        print("  (those are chrF++; pip install sacrebleu to report it beside chrF)")
    for r in results:
        print(f"\n=== {Path(r['ckpt']).name}")
        for lang, q, hyp, ref in r["samples"]:
            print(f"  [{lang}] {q}\n     -> {hyp}\n     ref {ref}")


if __name__ == "__main__":
    main()
