"""
Book example, not repository code.

Chapter 15: normalisation and the residual stream.

    python book/code-examples/ch15_norms.py

15.1  The toy's three vectors through RMSNorm and through LayerNorm, by hand,
      and `model.RMSNorm` against PyTorch's own `rms_norm` and `LayerNorm`.
15.2  Why the arithmetic runs in fp32: a bf16 input normalised inside fp32
      (as `model.py` does) and inside bf16, against an fp64 reference; and the
      dtype that comes out.
15.5  The scaled initialisation at the released size: which matrices it
      touches, their standard deviation, and how the residual stream grows
      through 20 freshly initialised layers with it and without it.

`model.py` is imported unchanged.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged
from trace_forward import released  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def by_hand() -> None:
    print("15.1  The toy's vectors, normalised (illustrative)")
    X = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])           # The, dog, bit
    rms = X.pow(2).mean(-1, keepdim=True).sqrt()
    norm = M.RMSNorm(2)                                         # gain 1, eps 1e-6
    ln = torch.nn.LayerNorm(2, elementwise_affine=False, eps=1e-6)
    print("  rms of each row:", [round(v, 3) for v in rms.flatten().tolist()])
    print("  RMSNorm:  ", [[round(v, 3) for v in r] for r in norm(X).tolist()])
    print("  LayerNorm:", [[round(v, 3) for v in r] for r in ln(X).tolist()])
    x = torch.randn(4, 1024)
    print(f"  model.RMSNorm vs F.rms_norm: {(M.RMSNorm(1024)(x) - F.rms_norm(x, (1024,), eps=1e-6)).abs().max():.1e}")


def precision() -> None:
    print("\n15.2  Normalising 16-bit inputs: inside fp32 (model.py) and inside the input's own dtype")
    torch.manual_seed(0)
    norm = M.RMSNorm(1024)
    for dtype, scale in ((torch.float16, 300), (torch.bfloat16, 30)):
        x = (torch.randn(4096, 1024) * scale).to(dtype)        # large activations, stored in 16 bits
        ref = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-6)
        naive = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)   # every step in 16 bits
        for name, y in (("inside fp32", norm(x)), ("inside " + str(dtype)[6:], naive)):
            rel = ((y.double() - ref).norm(dim=-1) / ref.norm(dim=-1)).mean()
            print(f"  {str(dtype)[6:]:8s} input × {scale}, {name:15s}: output {str(y.dtype)[6:]}, "
                  f"mean relative error {rel:.1e}")
    x = (torch.randn(4096, 1024) * 30).to(torch.bfloat16)
    ref = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-6)
    y = F.rms_norm(x.float(), (1024,), eps=1e-6)                # the same, without the cast back to bf16
    print(f"  bf16 input, fp32 inside and no cast back: {((y.double() - ref).norm(dim=-1) / ref.norm(dim=-1)).mean():.1e}")
    print(f"  the gain is {str(norm.weight.dtype)[6:]}; bf16 × fp32 gives {str((x * norm.weight).dtype)[6:]}")


def scaled_init() -> None:
    print("\n15.5  The scaled initialisation, released size")
    torch.manual_seed(0)
    cfg = released()
    model = M.AnuLM(cfg).eval()
    names = [n for n, _ in model.named_parameters() if n.endswith(("o_proj.weight", "dense.weight", "down_proj.weight"))]
    std = cfg.initializer_range / math.sqrt(2 * cfg.n_layer)
    kinds = sorted({n.split(".")[-2] for n in names})
    print(f"  {len(names)} matrices ({', '.join(kinds)}); std {cfg.initializer_range} / √{2 * cfg.n_layer} = {std:.5f}; "
          f"measured, mean over the matrices {sum(dict(model.named_parameters())[n].std().item() for n in names) / len(names):.5f}")
    x = torch.randint(0, cfg.vocab_size, (1, 128))
    for label in ("scaled (model.py)", "all 0.02"):
        if label == "all 0.02":
            with torch.no_grad():
                for n in names:
                    dict(model.named_parameters())[n].normal_(0.0, cfg.initializer_range)
        rms = []
        hooks = [b.register_forward_pre_hook(lambda mod, args: rms.append(args[0].pow(2).mean().sqrt().item()))
                 for b in model.layers]
        hooks.append(model.norm.register_forward_pre_hook(lambda mod, args: rms.append(args[0].pow(2).mean().sqrt().item())))
        with torch.no_grad():
            model(x)
        for h in hooks:
            h.remove()
        print(f"  {label:18s}: stream rms at layers 0, 5, 10, 15 and the end: "
              + ", ".join(f"{rms[i]:.4f}" for i in (0, 5, 10, 15, 20))
              + f"  (growth {rms[20] / rms[0]:.2f}×)")


def exercises() -> None:
    print("\n15.7  Answers")
    cfg = released()
    print(f"  1. norms in the released model: {2 * cfg.n_layer + 1} RMSNorms of width {cfg.hidden_size}, "
          f"{(2 * cfg.n_layer + 1) * cfg.hidden_size:,} gains; plus {2 * cfg.n_layer} QK-norms of width {cfg.head_dim}")
    print(f"  2. scaled std for a 40-layer model: {cfg.initializer_range / math.sqrt(80):.5f}")


def blind_spot() -> None:
    print("\n15.5  What the loss check in initial_loss_is_ln_vocab cannot see (tiny(), its own settings)")
    from trace_forward import tiny
    for label, std in (("as model.py initialises", None), ("embeddings at std 1.0", 1.0)):
        torch.manual_seed(0)
        cfg = tiny()
        model = M.AnuLM(cfg).eval()
        if std is not None:
            torch.nn.init.normal_(model.embed_tokens.weight, std=std)
        x = torch.randint(0, cfg.vocab_size, (4, 32))
        with torch.no_grad():
            _, loss = model(x, x)
        emb = model.embed_tokens.weight.std().item()
        print(f"  {label:24s}: initial loss {loss.item():.3f} (ln V = {math.log(cfg.vocab_size):.3f}; "
              f"the loss check allows 0.1); embedding std {emb:.4f}")


if __name__ == "__main__":
    by_hand()
    precision()
    scaled_init()
    blind_spot()
    exercises()
