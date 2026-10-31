"""
Chapter 13 lab: give Chapter 8's model eyes, and test whether it looks.

    python lab.py                      # your course/build/post.py
    python lab.py --impl solution      # the reference solution

Needs ../ch08-pretrain/base.pt, and the repository's cached image features:
data/omni/caption_train.jsonl and caption_train.feats.f16 (SigLIP patch vectors
for the Open Images pictures behind Localized Narratives; experiments/omni_one.py
builds them). CPU, about eight minutes, and about 1 GB of memory for features.

The model grows two ids, <|image|> and <|/image|>, and a projector turns each
picture's 196 patch vectors into 49 vectors of the model's size. Two runs,
1,200 steps each, train the projector and the model together to write the
caption after the picture:

  matched     the projector's output scaled to the size of a word's embedding
  unit gain   the projector left at LayerNorm's default scale, about 11 here

Then, on 200 pictures neither run has seen: the caption loss with the right
picture, with another picture, and with no picture at all. If the model uses
what it sees, the right picture must beat the wrong one. Writes results.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
DATA = COURSE.parent / "data" / "omni"
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


lab1 = load(COURSE / "ch01-smallest-model" / "lab.py")
N_TRAIN, N_TEST, CAP, STEPS, BATCH, LR = 2000, 200, 160, 1200, 16, 3e-4
EOS = 257


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["build", "solution"], default="build")
    a = p.parse_args()
    sol = a.impl == "solution"
    Bm = load(COURSE / "build" / ("solutions/ch08.py" if sol else "anulm.py"))
    S = load(COURSE / "build" / ("solutions/ch13.py" if sol else "post.py"))
    base = torch.load(COURSE / "ch08-pretrain" / "base.pt", weights_only=False)
    _, shakespeare_val = lab1.load_bytes(str(COURSE.parent / "data" / "input.txt"))

    # the pictures: the cache holds every distinct image of the manifest, in sorted order
    rows = [json.loads(l) for l in (DATA / "caption_train.jsonl").read_text(encoding="utf-8").splitlines()]
    uniq = sorted({r["image"] for r in rows})
    cache = np.memmap(DATA / "caption_train.feats.f16", dtype="float16", mode="r", shape=(len(uniq), 196, 768))
    first = {}
    for r in rows:
        first.setdefault(r["image"], r["text"])
    pick = random.Random(0).sample(range(len(uniq)), N_TRAIN + N_TEST)
    t0 = time.time()
    feats = torch.from_numpy(np.stack([cache[i] for i in sorted(pick)]))          # read in file order
    where = {i: k for k, i in enumerate(sorted(pick))}
    items = [(where[i], list(first[uniq[i]].encode("utf-8"))[:CAP] + [EOS]) for i in pick]
    train, test = items[:N_TRAIN], items[N_TRAIN:]
    print(f"{len(uniq):,} pictures in the cache; {N_TRAIN} for training and {N_TEST} held out, read in "
          f"{time.time() - t0:.0f}s ({feats.numel() * 2 / 1e9:.2f} GB)")
    print("for example:", bytes(t for t in test[0][1] if t < 256).decode("utf-8", "replace"))

    def fresh(match):
        torch.manual_seed(0)
        m = Bm.MoELM(**base["sizes"])
        m.load_state_dict(base["model"])
        try:
            vocab = S.grow_vocab(m, 2)
            proj = S.ImageProjector(768, base["sizes"]["hidden_size"], 2, hidden=256)
        except NotImplementedError as e:
            sys.exit(f"course/build/post.py is not finished yet ({e}). Run course/checks/ch13.py, or use --impl solution.")
        scale = S.match_scale(proj, m.embed_tokens.weight[:259]) if match else None
        return m.train(), proj.train(), vocab - 2, vocab - 1, scale

    def batch(m, proj, bos, eos, chosen, image_of=None, no_image=False):
        built = []
        for k, (fi, ids) in enumerate(chosen):
            pf = feats[image_of[k] if image_of else fi].float()
            if no_image:
                e = m.embed_tokens(torch.tensor([bos, eos] + ids))
                t = torch.full((len(ids) + 2,), -1)
                t[1:1 + len(ids)] = torch.tensor(ids)
                built.append((e, t))
            else:
                built.append(S.build_caption(m, proj, pf, bos, eos, ids))
        L = max(e.shape[0] for e, _ in built)
        pad = m.embed_tokens.weight[EOS]
        x = torch.stack([torch.cat([e, pad.expand(L - e.shape[0], -1)]) for e, _ in built])
        y = torch.stack([torch.cat([t, torch.full((L - t.shape[0],), -1)]) for _, t in built])
        return x, y

    @torch.no_grad()
    def held(m, proj, bos, eos, mode):
        m.eval(), proj.eval()
        other = [test[(k + 1) % len(test)][0] for k in range(len(test))]          # a different picture
        total = 0.0
        for i in range(0, len(test), 25):
            chunk = test[i:i + 25]
            x, y = batch(m, proj, bos, eos, chunk, image_of=other[i:i + 25] if mode == "wrong picture" else None,
                         no_image=mode == "no picture")
            total += S.forward_embeds(m, x, y)[1].item() * len(chunk)
        m.train(), proj.train()
        return total / len(test)

    @torch.no_grad()
    def describe(m, proj, bos, eos, k):
        m.eval(), proj.eval()
        e = torch.cat([m.embed_tokens(torch.tensor([bos])), proj(feats[test[k][0]].float()),
                       m.embed_tokens(torch.tensor([eos]))])
        out = []
        for _ in range(CAP):
            nxt = S.forward_embeds(m, e[None])[0][0, -1].argmax().item()
            if nxt == EOS or nxt >= 256:
                break
            out.append(nxt)
            e = torch.cat([e, m.embed_tokens(torch.tensor([nxt]))])
        m.train(), proj.train()
        return bytes(out).decode("utf-8", "replace")

    def shakespeare(m):
        m.eval()
        g = torch.Generator().manual_seed(0)
        loss = sum(m(*lab1.batch(shakespeare_val, 128, 32, g))[1].item() for _ in range(20)) / 20
        m.train()
        return loss

    out = {"pictures": len(uniq), "train": N_TRAIN, "test": N_TEST, "steps": STEPS,
           "example": bytes(t for t in test[0][1] if t < 256).decode("utf-8", "replace")}
    for name, match in (("matched", True), ("unit gain", False)):
        m, proj, bos, eos, scale = fresh(match)
        word = m.embed_tokens.weight[:259].norm(dim=-1).mean().item()
        with torch.no_grad():
            start_norm = proj(feats[0].float()).norm(dim=-1).mean().item()
        before = shakespeare(m)
        opt = Bm.make_optimizer(torch.nn.ModuleList([m, proj]), LR)
        g = random.Random(1)
        curve, t0 = [], time.time()
        for step in range(1, STEPS + 1):
            x, y = batch(m, proj, bos, eos, g.sample(train, BATCH))
            for grp in opt.param_groups:
                grp["lr"] = LR * min(1.0, step / 50)
            loss = S.forward_embeds(m, x, y)[1]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(m.parameters()) + list(proj.parameters()), 1.0)
            opt.step()
            m.update_expert_biases()
            if step % 200 == 0:
                curve.append((step, held(m, proj, bos, eos, "right picture")))
                print(f"   {name:9s} step {step:5d}  held-out caption loss {curve[-1][1]:.3f}  ({time.time() - t0:.0f}s)")
        res = {mode: held(m, proj, bos, eos, mode) for mode in ("right picture", "wrong picture", "no picture")}
        res.update(curve=curve, word_norm=word, projector_norm_at_start=start_norm, scale=scale,
                   shakespeare_before=before, shakespeare_after=shakespeare(m),
                   samples=[(describe(m, proj, bos, eos, k),
                             bytes(t for t in test[k][1] if t < 256).decode("utf-8", "replace")) for k in range(3)])
        out[name] = res
        print(f"{name}: picture vectors start at norm {start_norm:.2f} (a word's: {word:.2f}); held-out caption loss "
              f"right picture {res['right picture']:.3f}, wrong picture {res['wrong picture']:.3f}, "
              f"none {res['no picture']:.3f}; Shakespeare {before:.3f} -> {res['shakespeare_after']:.3f}")
        for got, want in res["samples"]:
            print(f"   wrote: {got!r}\n   truth: {want!r}")
    (HERE / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved results.json")


if __name__ == "__main__":
    main()
