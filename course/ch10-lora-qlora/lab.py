"""
Chapter 10 lab: teach Chapter 8's English (Shakespeare) model a new language
-- a full fine-tune, LoRA, and QLoRA -- with the code you wrote in
course/build/post.py, and measure what each learns, forgets, and costs.

Four runs, 1,000 steps each:
  full              every weight trains
  lora              adapters on the attention projections (lora.py's default)
  lora + rows       the same, and the embedding and output tables train too:
                    most of the new language's bytes were never seen, so
                    their rows hold nothing yet
  qlora + rows      lora + rows, with the attention weights stored in NF4

    python lab.py                          # your course/build/post.py and anulm.py
    python lab.py --impl solution          # the reference solutions
    python lab.py --data my_language.txt   # any text file of your own

Needs ../ch08-pretrain/base.pt (run that lab first). CPU, about ten minutes.
After the three runs, one more measurement: rebuild the LoRA model from a fresh
base twice -- from the adapters alone, and from the adapters plus the experts'
balancing biases -- and see which one is the model you trained. Writes
results.json next to this file for the chapter's figures.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
for _s in (sys.stdout, sys.stderr):              # Hindi on a Windows console
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lab1 = load(COURSE / "ch01-smallest-model" / "lab.py")
CTX, BATCH = 128, 16


def stored_bytes(model: nn.Module) -> int:
    """Memory the frozen weights take: fp32 frozen parameters plus every buffer (the 4-bit codes)."""
    n = sum(p.numel() * p.element_size() for p in model.parameters() if not p.requires_grad)
    return n + sum(b.numel() * b.element_size() for b in model.buffers())


@torch.no_grad()
def choices(model, x):
    """The experts each token is sent to, in every MoE layer: the routing decision itself."""
    seen = []
    hooks = [m.register_forward_hook(lambda mod, inp, out: seen.append(out[0].sort(-1).values))
             for m in model.modules() if type(m).__name__ == "Router"]
    model.eval()
    model(x)
    for h in hooks:
        h.remove()
    return torch.cat([s.reshape(-1, s.shape[-1]) for s in seen])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    p.add_argument("--data", default=str(COURSE.parent / "data" / "bench_hindi.txt"))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--prompt", default="भारत ")
    a = p.parse_args()
    sol = a.impl == "solution"
    B = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S = load(COURSE / "build" / ("solutions/ch10.py" if sol else "post.py"))
    base = torch.load(COURSE / "ch08-pretrain" / "base.pt", weights_only=False)
    new_train, new_val = lab1.load_bytes(a.data)
    _, old_val = lab1.load_bytes(str(COURSE.parent / "data" / "input.txt"))

    def fresh():
        m = B.MoELM(**base["sizes"])
        m.load_state_dict(base["model"])
        return m.train()

    def losses(m):
        g = torch.Generator().manual_seed(0)                  # the same windows for every model
        return lab1.evaluate(m, new_val, CTX, g), lab1.evaluate(m, old_val, CTX, g)

    def prepare(method):
        m = fresh()
        if method != "full":
            try:
                S.apply_lora(m, a.rank, 2.0 * a.rank, quantize=method.startswith("qlora"))
            except NotImplementedError as e:
                sys.exit(f"course/build/post.py is not finished yet ({e}). Run course/checks/ch10.py, "
                         "or use --impl solution.")
        if method.endswith("+ rows"):                         # the tables that hold one row per byte
            for t in (m.embed_tokens.weight, m.lm_head.weight):
                t.requires_grad_(True)
        return m

    seen = set((COURSE.parent / "data" / "input.txt").read_bytes())
    used = set(new_train.unique().tolist())
    unseen = len(used - seen)
    print(f"{unseen} of the {len(used)} byte values in {Path(a.data).name} never occur in pre-training")
    m0 = fresh()
    base_new, base_old = losses(m0)
    n_all = sum(t.numel() for t in m0.parameters())
    q_new, q_old = losses(prepare("qlora"))                   # 4 bits, adapters still zero
    print(f"base model: new language {base_new:.3f}, Shakespeare {base_old:.3f}")
    print(f"attention stored in NF4, before tuning: {q_new:.3f}, {q_old:.3f} (the cost of compression alone)")
    out = {"data": Path(a.data).name, "unseen_bytes": unseen, "used_bytes": len(used), "steps": a.steps, "rank": a.rank, "params": n_all,
           "base_new": round(base_new, 4), "base_old": round(base_old, 4),
           "nf4_only_new": round(q_new, 4), "nf4_only_old": round(q_old, 4), "methods": {}}
    trained = {}
    for method, lr in (("full", 3e-4), ("lora", 2e-3), ("lora + rows", 5e-4), ("qlora + rows", 5e-4)):
        torch.manual_seed(0)
        g = torch.Generator().manual_seed(0)
        m = prepare(method)
        train = [t for t in m.parameters() if t.requires_grad]
        n_train = sum(t.numel() for t in train)
        opt = B.make_optimizer(m, lr, weight_decay=0.0)
        curve, t0 = [(0, round(base_new, 4))], time.time()
        for step in range(1, a.steps + 1):
            B.train_step(m, opt, [lab1.batch(new_train, CTX, BATCH, g)], lr * min(1.0, step / 50))
            if step % 100 == 0:
                curve.append((step, round(lab1.evaluate(m, new_val, CTX, torch.Generator().manual_seed(0)), 4)))
        secs = time.time() - t0
        new, old = losses(m)
        m.eval()
        torch.manual_seed(0)
        ids = m.generate(torch.tensor([list(a.prompt.encode())]), 120, temperature=0.8, top_k=40)
        sample = bytes([t for t in ids[0].tolist() if t < 256]).decode("utf-8", "replace")
        keep = n_train * 4 if method != "full" else n_all * 4                # what you save afterwards
        out["methods"][method] = {
            "lr": lr, "trainable": n_train, "trainable_pct": round(100 * n_train / n_all, 2),
            "new_loss": round(new, 4), "old_loss": round(old, 4), "seconds": round(secs, 1),
            "frozen_bytes": stored_bytes(m) if method != "full" else 0, "saved_bytes": keep,
            "curve": curve, "sample": sample}
        trained[method] = m
        print(f"\n{method:5s}  trains {n_train:,} parameters ({out['methods'][method]['trainable_pct']}%) in {secs:.0f}s\n"
              f"       new language {base_new:.3f} -> {new:.3f}   Shakespeare {base_old:.3f} -> {old:.3f}\n"
              f"       {sample!r}")

    # What do you have to keep? Rebuild the LoRA model on a fresh base, two ways.
    lora = trained["lora"].eval()
    state = S.adapter_state(lora)
    only = {k: v for k, v in state.items() if "lora_" in k}
    x, _ = lab1.batch(new_val, CTX, 8, torch.Generator().manual_seed(1))
    want = choices(lora, x)
    rebuilt = {}
    for name, sd in (("adapters only", only), ("adapters + router biases", state)):
        m = fresh()
        S.apply_lora(m, a.rank, 2.0 * a.rank)
        m.load_state_dict(sd, strict=False)
        m.eval()
        with torch.no_grad():
            gap = (m(x)[0] - lora(x)[0]).abs().max().item()
        flips = (choices(m, x) != want).any(-1).float().mean().item()
        loss = losses(m)[0]
        rebuilt[name] = {"keys": len(sd), "max_logit_gap": gap, "routing_changed": flips, "new_loss": round(loss, 4)}
        print(f"\nrebuilt from {name} ({len(sd)} tensors): Hindi loss {loss:.4f} (trained {out['methods']['lora']['new_loss']:.4f}), "
              f"{flips:.1%} of expert choices changed, largest logit gap {gap:.2e}")
    drift = [b.abs().max().item() for n, b in lora.named_buffers() if n.endswith("expert_bias")]
    out["rebuild"] = rebuilt
    out["bias_drift_max"] = max(drift)

    merged = trained["lora"]
    with torch.no_grad():
        before = merged(x)[0]
        n = S.merge_lora(merged)
        after = merged(x)[0]
    out["merge"] = {"layers": n, "max_gap": (after - before).abs().max().item()}
    print(f"\nmerged {n} adapters into the weights: largest change in any logit {out['merge']['max_gap']:.1e}")
    (HERE / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
