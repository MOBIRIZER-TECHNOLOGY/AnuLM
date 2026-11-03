"""Measure the noise floor of train.py's val-loss estimate.

evaluate() averages `iters=20` batches of `batch_size` windows -- 160 windows
of 512 tokens on the coder run, 82k tokens out of a 32M-token val split. The
seed is fixed so successive evals see the SAME windows, which makes them
comparable; it does not make them accurate. If the standard error across
window-sets is comparable to the deltas being read off the curve, then the
"plateaus" are the instrument, not the model.

    python tools/eval_noise.py        # needs ckpt_coder.pt and data/coder.code32k.bin
"""
import sys
from dataclasses import replace
from pathlib import Path

import torch

# The repository root, wherever it is checked out, so `python tools/eval_noise.py`
# finds the package and the checkpoint without a machine-specific path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anulm.model import AnuLM, safe_load           # noqa: E402
from anulm.training.train import eval_windows, make_batch        # noqa: E402

DEV = "cuda"
CKPT = ROOT / "ckpt_coder.pt"
BIN = ROOT / "data" / "coder.code32k.bin"


@torch.no_grad()
def val_loss(model, data, batch_size, block_size, iters, seed):
    losses = torch.zeros(iters)
    for i, ix in enumerate(eval_windows(data, batch_size, block_size, iters, seed=seed)):
        x, y = make_batch(data, ix, block_size, DEV)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses[i] = loss.item()
    return losses


def main():
    ck = safe_load(CKPT, map_location=DEV)
    cfg = replace(ck["cfg"], moe_impl="grouped")
    model = AnuLM(cfg).to(DEV).eval()
    model.load_state_dict(ck["model"])
    print(f"checkpoint step {ck['step'] + 1:,}, recorded val {ck['val_loss']:.4f}")

    blob = safe_load(BIN)
    data = blob["ids"]
    n = int(0.9 * len(data))
    val = data[n:]
    print(f"val split: {len(val)/1e6:.1f}M tokens; production eval sees "
          f"{20 * 8 * 512 / 1e6:.3f}M of them ({20*8*512/len(val)*100:.2f}%)\n")

    print("production setting (iters=20, batch 8) with different window sets:")
    means = []
    for seed in range(6):
        ls = val_loss(model, val, 8, 512, 20, seed)
        means.append(ls.mean().item())
        print(f"  seed {seed}: val {means[-1]:.4f}   (per-batch std {ls.std().item():.3f})")
    m = torch.tensor(means)
    print(f"\n  mean of means {m.mean():.4f}, spread {m.max()-m.min():.4f}, "
          f"std across window-sets {m.std():.4f}")

    print("\nbigger sample, for reference:")
    for iters in (100, 400):
        ls = val_loss(model, val, 8, 512, iters, 0)
        se = ls.std().item() / (iters ** 0.5)
        print(f"  iters={iters:4d} ({iters*8*512/1e6:.2f}M tokens): "
              f"val {ls.mean().item():.4f}  +/- {se:.4f} (1 s.e.)")


if __name__ == "__main__":
    main()
