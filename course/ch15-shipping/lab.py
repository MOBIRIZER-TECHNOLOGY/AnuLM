"""
Chapter 15 lab: release a model, check what came back, and decide whether to publish.

    python lab.py                      # your course/build/post.py
    python lab.py --impl solution      # the reference solution

Needs ../ch09-sft/sft.pt (Chapter 9's sums model). CPU, under a minute.

1. Export the sums model three ways and load each back:
     export_hf's rule   bfloat16 only for floating tensors of 100,000+ numbers
     a lower bar        bfloat16 for floating tensors of 1,000+ numbers
     all bfloat16       every floating tensor, the experts' biases and norms too
   For each: the folder's size, the largest change in any next-byte
   probability, how many expert choices changed, and the sums it still gets
   right.
2. The release decision for the first AnuLM-Omni run, from its measured
   results (logs/omni_one/REPORT.md) and experiments/omni_one.py's gates.
Writes results.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
ROOT = COURSE.parent
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lab9 = load(COURSE / "ch09-sft" / "lab.py")
omni_one = load(ROOT / "experiments" / "omni_one.py")
OMNI = {"chat_heldout_loss": 5.3181, "read_squad_en": 24.6, "intent_banking77": 90.0, "wer_en": 12.2,
        "wer_hi_fleurs": 48.8, "wer_hi_kathbath": 35.5, "wer_hi_mucs": 54.4, "vqa_acc": 17.6}   # REPORT.md


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    a = p.parse_args()
    sol = a.impl == "solution"
    Bm = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S9 = load(COURSE / "build" / ("solutions/ch09.py" if sol else "post.py"))
    S = load(COURSE / "build" / ("solutions/ch15.py" if sol else "post.py"))
    sft = torch.load(COURSE / "ch09-sft" / "sft.pt", weights_only=False)
    ck = {"model": sft["model"], "cfg": sft["sizes"], "step": 1500, "val_loss": None}
    pt_bytes = (COURSE / "ch09-sft" / "sft.pt").stat().st_size

    rnd = random.Random(0)
    lab9.make_items(6000, rnd)
    test = lab9.make_items(200, rnd)                                      # Chapter 9's held-out sums
    enc = lambda s: list(s.encode("utf-8"))
    prompts = torch.tensor([enc(S9.PROMPT_EN.format(q=q)) for q, _ in test])

    def build(state):
        m = Bm.MoELM(**ck["cfg"])
        m.load_state_dict(state)
        return m.eval()

    @torch.no_grad()
    def measure(m):
        seen = []
        hooks = [r.register_forward_hook(lambda mod, inp, out: seen.append(out[0].sort(-1).values))
                 for r in m.modules() if type(r).__name__ == "Router"]
        logits = m(prompts)[0]
        for h in hooks:
            h.remove()
        out = m.generate(prompts, 5, temperature=1e-5, eos_id=S9.BYTE_EOS)[:, prompts.shape[1]:].tolist()
        right = sum(bytes(t for t in o if t < 256).decode("utf-8", "replace").strip() == ans
                    for o, (_, ans) in zip(out, test))
        return logits, torch.cat([s.reshape(-1, s.shape[-1]) for s in seen]), right / len(test)

    ref_logits, ref_routes, ref_acc = measure(build(ck["model"]))
    print(f"the sums model: {sum(v.numel() for v in ck['model'].values()):,} numbers, {pt_bytes / 1e6:.2f} MB as a .pt, "
          f"{ref_acc:.1%} of 200 held-out sums right")

    def lower_bar(sd):
        return {k: (v.detach().to(torch.bfloat16) if v.is_floating_point() and v.numel() >= 1000 else v.detach())
                for k, v in sd.items()}

    def all_bf16(sd):
        return {k: (v.detach().to(torch.bfloat16) if v.is_floating_point() else v.detach()) for k, v in sd.items()}

    out = {"pt_bytes": pt_bytes, "reference_accuracy": ref_acc, "exports": {}}
    tmp = Path(tempfile.mkdtemp())
    try:
        for name, rule in (("export_hf's rule", None), ("a lower bar", lower_bar), ("all bfloat16", all_bf16)):
            original = S.to_bf16
            try:
                if rule is not None:
                    S.to_bf16 = rule                     # the same exporter, a different precision rule
                folder = S.export_folder(ck, tmp / name.replace(" ", "_").replace("'", ""))
                back = S.load_folder(folder)
            except NotImplementedError as e:
                sys.exit(f"course/build/post.py is not finished yet ({e}). Run course/checks/ch15.py, or use --impl solution.")
            finally:
                S.to_bf16 = original
            size = sum(f.stat().st_size for f in folder.iterdir())
            logits, routes, acc = measure(build(back["model"]))
            gap = (logits.softmax(-1) - ref_logits.softmax(-1)).abs().max().item()
            flips = (routes != ref_routes).any(-1).float().mean().item()
            n16 = sum(v.numel() for k, v in ck["model"].items()
                      if (rule or S.to_bf16)({k: v})[k].dtype == torch.bfloat16)
            out["exports"][name] = {"bytes": size, "bf16_numbers": n16, "max_prob_change": gap,
                                    "routing_changed": flips, "accuracy": acc}
            print(f"{name:17s} {size / 1e6:5.2f} MB, {n16:,} numbers in bfloat16; largest probability change {gap:.4f}; "
                  f"{flips:.2%} of expert choices changed; {acc:.1%} of sums right")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    res = dict(OMNI)
    g = omni_one.gates(res)                        # it adds the Hindi mean to res, which the report must show
    text, ok = S.release_report(res, omni_one.GATES, g)
    out["omni"] = {"results": res, "gates": g, "publish": ok, "report": text}
    print("\nthe first AnuLM-Omni run:\n" + text)
    (HERE / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
