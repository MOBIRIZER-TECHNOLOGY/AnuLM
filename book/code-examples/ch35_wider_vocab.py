"""
Book example, not repository code.

Chapter 35: a wider vocabulary and inputs that are not tokens.

    python book/code-examples/ch35_wider_vocab.py

It imports the repository's `speech_vocab.py`, `audio_codec.py`,
`speech_encoder.py` and `model.py` unchanged and checks, on the small test
model `tiny()`:

1. the grown vocabulary's layout, for the released model's 32,768 text ids;
2. that SNAC's three levels flatten into 7 slot blocks and back exactly;
3. that growing both tables leaves every text logit unchanged;
4. what growing costs the released model, total and active;
5. that `embeds=` is the embedding lookup, moved outside the model;
6. that `generate` never samples a non-text id once `text_vocab` is set;
7. where `ContinuousSpeech.build` puts the targets.

    python book/code-examples/ch35_wider_vocab.py --measure [--save]

also grows the released checkpoint `ckpt_base_v2.pt` with the repository's
`_grow` (seed 0) and measures its loss on 64 held-out windows of 512 tokens of
`data/v2/val.u16` (default_rng(0), as Chapter 20), over the full grown softmax
and over the text ids only. The MoE runs on its "sparse" path, so everything
is in float32. No held-out text is printed. Results: ./part_viii_ix_grow.json,
or book/measurements/ with --save.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anulm import model as M                                   # noqa: E402  repository code, unchanged
from anulm.multimodal import speech_vocab as SV                           # noqa: E402
from anulm.multimodal.audio_codec import AUDIO_VOCAB, FRAME_SLOTS, LEVEL_VOCAB, flatten, unflatten  # noqa: E402
from anulm.multimodal.speech_encoder import ContinuousSpeech         # noqa: E402
from trace_forward import released                  # noqa: E402  the released 400M config

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def tiny(**kw) -> M.AnuLMConfig:
    """test_model.py's tiny(), copied so this script needs no test harness."""
    base = dict(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8,
                n_kv_head=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
                intermediate_size=64, n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


def layout() -> None:
    v = SV.SpeechVocab(32_768)                       # the released model's text vocabulary
    print("1. layout for the released model")
    print("  ", v.describe())
    print(f"   audio ids: {FRAME_SLOTS} slots x {LEVEL_VOCAB} codes = {AUDIO_VOCAB}; "
          f"control tokens: {SV.N_CONTROL} {SV.CONTROL}")
    assert v.total == 32_768 + 28_672 + 5 == 61_445
    b = SV.SpeechVocab(259)                          # exercise 1: a byte-level checkpoint
    print(f"   for 259 text ids: <|speech|> {b.speech_bos}, <|/image|> {b.image_eos}")
    assert (b.speech_bos, b.image_eos) == (28_931, 28_935)


def snac_round_trip() -> None:
    rng = np.random.default_rng(0)
    t = 6                                            # coarse frames
    codes = [rng.integers(0, LEVEL_VOCAB, (1, t * r)) for r in (1, 2, 4)]
    ids = flatten(codes)
    slots = [ids[j::7] for j in range(7)]
    lo = [int(s.min()) // LEVEL_VOCAB for s in slots]
    back = unflatten(ids)
    exact = all(np.array_equal(a, b) for a, b in zip(codes, back))
    print(f"\n2. SNAC: levels of {t}, {2 * t}, {4 * t} codes -> {ids.size} flat ids; "
          f"slot j lives in block {lo}; round trip exact: {exact}")
    assert lo == list(range(7)) and exact


def grow(model: M.AnuLM, n_new: int, seed: int = 0) -> M.AnuLM:
    """speech_vocab.grow_checkpoint's middle, on a live model instead of a file."""
    from dataclasses import replace
    gen = torch.Generator().manual_seed(seed)
    sd = model.state_dict()
    for key in ("embed_tokens.weight", "lm_head.weight"):
        sd[key] = SV._grow(sd[key].float(), n_new, gen)        # the repository's own _grow
    big = M.AnuLM(replace(model.cfg, vocab_size=model.cfg.vocab_size + n_new)).eval()
    big.load_state_dict(sd)
    return big


def grow_and_compare() -> M.AnuLM:
    torch.manual_seed(0)
    small = M.AnuLM(tiny()).eval()
    v = SV.SpeechVocab(64)
    big = grow(small, v.total - 64)
    x = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        a, _ = small(x)
        b, _ = big(x)
    d = (a - b[..., :64]).abs().max().item()
    old = small.embed_tokens.weight.norm(dim=-1)
    new = big.embed_tokens.weight[64:].norm(dim=-1)
    print(f"\n3. grown tiny(): vocab 64 -> {big.cfg.vocab_size}; logits {tuple(a.shape)} -> {tuple(b.shape)}; "
          f"text logits differ by at most {d:.1e}")
    print(f"   row norms: text rows mean {old.mean():.3f}, new rows mean {new.mean():.3f} "
          f"(min {new.min():.3f}, max {new.max():.3f})")
    assert d < 1e-6
    return big


def cost_at_scale() -> None:
    from dataclasses import replace
    cfg = released()
    with torch.device("meta"):                       # shapes only: no 1.8 GB of memory
        t0, a0 = M.AnuLM(cfg).num_params()
        t1, a1 = M.AnuLM(replace(cfg, vocab_size=SV.SpeechVocab(cfg.vocab_size).total)).num_params()
    n_new = SV.SpeechVocab(cfg.vocab_size).total - cfg.vocab_size
    print(f"\n4. released model: +{n_new:,} ids x D {cfg.hidden_size} x 2 tables = {t1 - t0:,} parameters")
    print(f"   total  {t0:,} -> {t1:,}  (+{100 * (t1 / t0 - 1):.1f}%)")
    print(f"   active {a0:,} -> {a1:,}  (+{100 * (a1 / a0 - 1):.1f}%)")
    assert t1 - t0 == a1 - a0 == 2 * n_new * cfg.hidden_size


def embeds_is_the_lookup(big: M.AnuLM) -> None:
    x = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        a, _ = big(x)
        b, _ = big(None, embeds=big.embed_tokens(x))
    print(f"\n5. forward(idx) and forward(None, embeds=embed_tokens(idx)) equal: {torch.equal(a, b)}")
    assert torch.equal(a, b)


def text_only_generation(big: M.AnuLM) -> None:
    x = torch.randint(0, 64, (1, 4))
    torch.manual_seed(1)
    free = big.generate(x, 200, temperature=1.0)[0, 4:]
    big.text_vocab = 64                              # what serve.Engine sets from speech_text_vocab
    torch.manual_seed(1)
    held = big.generate(x, 200, temperature=1.0)[0, 4:]
    del big.text_vocab
    print(f"\n6. 200 sampled ids: without text_vocab {int((free >= 64).sum())} are audio or control ids; "
          f"with it {int((held >= 64).sum())}")
    assert int((held >= 64).sum()) == 0


def asr_targets(big: M.AnuLM) -> None:
    v = SV.SpeechVocab(64)
    sm = ContinuousSpeech(big, v, in_dim=8, stack=4)
    states = torch.randn(10, 8)                      # 10 encoder frames: 2 whole groups of 4, 2 dropped
    answer = [5, 6, 7, 1]                            # three text ids and an EOS
    with torch.no_grad():
        embeds, targets = sm.build(states, [], answer)
    print(f"\n7. ContinuousSpeech.build: states {tuple(states.shape)} -> embeds {tuple(embeds.shape)}")
    print("   position:", list(range(embeds.shape[0])))
    print("   targets: ", targets.tolist())
    assert embeds.shape == (1 + 2 + 1 + 4, 32)
    assert targets.tolist() == [-1, -1, -1, 5, 6, 7, 1, -1]


def measure_grown() -> None:
    import json
    from dataclasses import replace
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = M.load_checkpoint(str(ROOT / "ckpt_base_v2.pt"), "cpu")
    base = M.AnuLM(replace(ck["cfg"], moe_impl="sparse")).eval()
    base.load_state_dict(ck["model"])
    del ck
    V = base.cfg.vocab_size
    big = grow(base, SV.SpeechVocab(V).total - V).to(dev)     # the same _grow, seed 0
    base = base.to(dev)
    val = np.memmap(ROOT / "data" / "v2" / "val.u16", dtype=np.uint16, mode="r")
    starts = np.random.default_rng(0).integers(0, len(val) - 513, size=64)
    rows = np.stack([val[s: s + 513] for s in starts]).astype(np.int64)
    sums, n = {"text model": 0.0, "grown, full softmax": 0.0, "grown, text ids only": 0.0}, 0
    with torch.no_grad():
        for b in range(0, 64, 8):
            x = torch.from_numpy(rows[b: b + 8, :-1]).to(dev)
            y = torch.from_numpy(rows[b: b + 8, 1:]).to(dev).masked_fill(x == 32767, -1)
            k = int((y != -1).sum())
            _, l0 = base(x, y)
            logits, l1 = big(x, y)
            l2 = torch.nn.functional.cross_entropy(logits[..., :V].reshape(-1, V).float(), y.reshape(-1),
                                                   ignore_index=-1)
            for key, l in zip(sums, (l0, l1, l2)):
                sums[key] += l.item() * k
            n += k
    res = {key: round(v / n, 6) for key, v in sums.items()}
    print(f"\n8. ckpt_base_v2.pt grown to {big.cfg.vocab_size:,} ids, held-out loss over {n:,} tokens (fp32):")
    for key, v in res.items():
        print(f"   {key:22s} {v:.5f}  ({v - res['text model']:+.5f})")
    res.update(graded_tokens=n, sample="64 x 512 of data/v2/val.u16, default_rng(0); moe_impl sparse; seed 0 growth")
    out = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_viii_ix_grow.json"
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"   -> {out}")


if __name__ == "__main__":
    layout()
    snac_round_trip()
    big = grow_and_compare()
    cost_at_scale()
    embeds_is_the_lookup(big)
    text_only_generation(big)
    asr_targets(big)
    if "--measure" in sys.argv:
        measure_grown()
    print("\nall checks passed")
