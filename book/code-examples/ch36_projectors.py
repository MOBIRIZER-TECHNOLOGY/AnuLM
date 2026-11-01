"""
Book example, not repository code.

Chapter 36: frozen encoders and the projectors that feed them to AnuLM.

    python book/code-examples/ch36_projectors.py            # checks on tiny(), CPU, seconds
    python book/code-examples/ch36_projectors.py --measure [--save]  # + the measurements below

It imports `speech_encoder.py`, `vision_encoder.py`, `speech_vocab.py` and
`model.py` unchanged, and checks:

1. the audio projector's shapes and size, and the frames it drops;
2. that the image projector pools 2x2 neighbourhoods of the patch grid, against
   a plain loop, and how often a flat stride would join two rows;
3. where `VisionLanguage.build` puts the targets;
4. the size of one image in the frozen-feature cache;
5. the scale a fresh projector hands the backbone.

With --measure it also runs the repository's own `WhisperEncoder` (Whisper-small,
CPU) on synthetic tones, and measures on the released checkpoints the norms of
the text embeddings and of the trained projectors' outputs, using 64 images'
cached SigLIP features (seed 0). No text from any dataset is read or printed.
Results: ./part_viii_ix_scale.json, or book/measurements/ with --save.
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
from anulm import model as M                                               # noqa: E402  repository code, unchanged
from anulm.multimodal.speech_encoder import STACK, AudioProjector                # noqa: E402
from anulm.multimodal.speech_vocab import SpeechVocab                            # noqa: E402
from anulm.multimodal.vision_encoder import POOL, ImageProjector, VisionLanguage  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT = (ROOT / "book" / "measurements" if "--save" in sys.argv else Path.cwd()) / "part_viii_ix_scale.json"


def tiny(**kw) -> M.AnuLMConfig:
    base = dict(vocab_size=64, block_size=128, n_layer=3, hidden_size=32, n_head=4, head_dim=8,
                n_kv_head=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
                intermediate_size=64, n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def audio() -> None:
    torch.manual_seed(0)
    proj = AudioProjector(768, 1024, STACK).eval()              # Whisper-small width, released D
    with torch.no_grad():
        y = proj(torch.randn(1, 501, 768))                      # 10 s of Whisper frames
    print(f"1. AudioProjector(768 -> 1024, stack {STACK}): {n_params(proj):,} parameters")
    print(f"   (1, 501, 768) -> {tuple(y.shape)}: {501 % STACK} frame dropped")
    assert y.shape == (1, 125, 1024) and n_params(proj) == 3072 * 3072 + 3072 + 3072 * 1024 + 1024 + 2 * 1024


def image() -> None:
    torch.manual_seed(0)
    proj = ImageProjector(768, 1024, POOL).eval()
    print(f"\n2. ImageProjector(768 -> 1024, pool {POOL}x{POOL}): {n_params(proj):,} parameters")
    # Replace the network by "show me the pooled input" to test the pooling alone.
    D, g = 3, 14
    patches = torch.arange(g * g * D, dtype=torch.float32).reshape(g * g, D)
    probe = ImageProjector(D, 4, POOL)
    probe.net = torch.nn.Identity()
    pooled = probe(patches)                                     # (49, 4*D)
    naive = torch.stack([torch.cat([patches[(2 * i + a) * g + 2 * j + c] for a in (0, 1) for c in (0, 1)])
                         for i in range(g // 2) for j in range(g // 2)])
    flat = patches.reshape(-1, 4 * D)                           # the bug the docstring warns about
    rows = [{k // g for k in range(4 * r, 4 * r + 4)} for r in range(g * g // 4)]
    print(f"   196 patches -> {tuple(pooled.shape)}; equals a plain 2x2 loop: {torch.equal(pooled, naive)}")
    print(f"   a flat stride of 4 would instead join two grid rows in {sum(len(s) > 1 for s in rows)} of "
          f"{len(rows)} groups")
    assert torch.equal(pooled, naive) and not torch.equal(flat, naive)


def vl_targets() -> None:
    torch.manual_seed(0)
    v = SpeechVocab(64)
    m = M.AnuLM(tiny(vocab_size=v.total)).eval()
    vl = VisionLanguage(m, v, in_dim=8, pool=POOL)
    with torch.no_grad():
        e, t = vl.build(torch.randn(196, 8), answer_ids=[9, 1], prompt_ids=[20, 21, 22])
    print(f"\n3. VisionLanguage.build: 196 patches, a 3-id question, a 2-id answer -> embeds {tuple(e.shape)}")
    print(f"   graded positions {[(i, x) for i, x in enumerate(t.tolist()) if x != -1]}")
    assert e.shape == (1 + 49 + 1 + 3 + 2, 32) and t.tolist()[53:55] == [9, 1] and (t != -1).sum() == 2


def cache_size() -> None:
    per = 196 * 768 * 2
    print(f"\n4. one cached image: 196 x 768 float16 = {per:,} bytes; "
          f"float16 keeps 11 significant bits, a relative rounding error of at most {2 ** -11:.1e}")
    for name, n in (("caption", 49_345), ("vqa", 21_953)):
        f = ROOT / "data" / "omni" / f"{name}_train.feats.f16"
        if f.exists():
            print(f"   {f.name}: {f.stat().st_size:,} bytes = {f.stat().st_size // per:,} images")
            assert f.stat().st_size == n * per


def fresh_scale() -> float:
    torch.manual_seed(0)
    proj = AudioProjector(768, 1024, STACK).eval()
    with torch.no_grad():
        y = proj(torch.randn(1, 400, 768))
    norm = y.norm(dim=-1).mean().item()
    print(f"\n5. a fresh projector's output norm: {norm:.2f} (LayerNorm, gain 1, bias 0: sqrt(1024) = 32)")
    assert abs(norm - 32) < 0.01
    return norm


def measure() -> dict:
    from anulm.multimodal.speech_encoder import AudioProjector as AP
    res = {}
    # (a) the repository's frozen ear on synthetic tones (no speech data)
    try:
        from anulm.multimodal.audio_codec import WhisperEncoder
        enc = WhisperEncoder("small", "cpu")                    # CPU: ctranslate2 runs it in int8
        frames = {}
        for sec in (3.0, 10.0, 30.0, 40.0):
            t = np.arange(int(16_000 * sec)) / 16_000
            frames[sec] = int(enc.encode((0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)).shape[0])
        res["whisper_small_frames"] = {str(k): v for k, v in frames.items()}
        print(f"\n6. WhisperEncoder('small', 'cpu') frames for a 3 / 10 / 30 / 40 s tone: "
              f"{' / '.join(str(v) for v in frames.values())}")
    except Exception as e:                                      # faster-whisper not installed
        print(f"\n6. Whisper skipped: {type(e).__name__}: {e}")
    # (b) text embedding norms, and the trained projectors' output norms
    base = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", mmap=True, weights_only=False)
    emb = base["model"]["embed_tokens.weight"].float()
    res["base_v2_text_embedding_norm_mean"] = round(emb.norm(dim=-1).mean().item(), 4)
    omni = torch.load(ROOT / "ckpt_omni.pt", map_location="cpu", mmap=True, weights_only=False)
    V = omni["speech_text_vocab"]
    e2 = omni["model"]["embed_tokens.weight"].float()
    res["omni_text_embedding_norm_mean"] = round(e2[:V].norm(dim=-1).mean().item(), 4)
    res["omni_new_row_norm_mean"] = round(e2[V:].norm(dim=-1).mean().item(), 4)
    vp = ImageProjector(768, omni["cfg"].hidden_size, omni.get("pool", POOL)).eval()
    vp.load_state_dict(omni["vision_proj"])
    mm = np.memmap(ROOT / "data" / "omni" / "caption_train.feats.f16", dtype="float16", mode="r",
                   shape=(49_345, 196, 768))
    pick = np.sort(np.random.default_rng(0).choice(49_345, 64, replace=False))
    with torch.no_grad():
        out = torch.cat([vp(torch.from_numpy(np.asarray(mm[i])).float()) for i in pick])
    res["omni_vision_proj_output_norm_mean"] = round(out.norm(dim=-1).mean().item(), 4)
    res["siglip_patch_norm_mean"] = round(float(np.linalg.norm(
        np.asarray(mm[pick], dtype=np.float32), axis=-1).mean()), 4)
    ap = AP(768, omni["cfg"].hidden_size, omni.get("stack", STACK)).eval()
    ap.load_state_dict(omni["proj"])
    ln = ap.net[-1]
    res["omni_audio_proj_ln_expected_norm"] = round(
        (ln.weight.float().pow(2).sum() + ln.bias.float().pow(2).sum()).sqrt().item(), 4)
    ln = vp.net[-1]
    res["omni_vision_proj_ln_expected_norm"] = round(
        (ln.weight.float().pow(2).sum() + ln.bias.float().pow(2).sum()).sqrt().item(), 4)
    res["fresh_over_base_text"] = round(32.0 / res["base_v2_text_embedding_norm_mean"], 1)
    res["fresh_over_omni_text"] = round(32.0 / res["omni_text_embedding_norm_mean"], 1)
    res["trained_vision_over_omni_text"] = round(res["omni_vision_proj_output_norm_mean"]
                                                 / res["omni_text_embedding_norm_mean"], 1)
    res["sample"] = "64 images of data/omni/caption_train.feats.f16, numpy default_rng(0); checkpoints ckpt_base_v2.pt, ckpt_omni.pt"
    print("7. norms (mean over rows or vectors):")
    for k, v in res.items():
        if k not in ("whisper_small_frames", "sample"):
            print(f"   {k:38s} {v}")
    return res


if __name__ == "__main__":
    audio()
    image()
    vl_targets()
    cache_size()
    fresh_scale()
    if "--measure" in sys.argv:
        r = measure()
        OUT.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        print(f"   -> {OUT}")
    print("\nall checks passed")
