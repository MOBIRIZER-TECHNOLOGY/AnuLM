"""
Book example, not repository code.

Chapter 22: the training loop of train.py, checked in two ways.

1. `train.main()` itself, run with the released model's command line and
   `--steps 0` on the CPU: it builds the data, the model and the optimiser,
   prints the same header the real run printed, and takes no step.
2. One optimiser step written out in the order of train.py lines 689-712, on
   the test model, with the repository's own sampler, batches, schedule and
   optimiser. Two micro-batches must give the gradient of one batch of twice
   the size; forgetting to divide by M doubles it.

Part 1 reads data/v2/meta.json from the repository root (found from this
file's location, so the script runs from any directory). Without the v2
corpus, part 1 is skipped and part 2 still runs.

    python book/code-examples/ch22_one_step.py
"""

from __future__ import annotations

import contextlib
import copy
import io
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as Mdl  # noqa: E402  the repository's model.py, unchanged
from anulm.training import train as T  # noqa: E402  the repository's train.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
META = ROOT / "data" / "v2" / "meta.json"          # repository-relative: runs from any directory
RELEASED = ("--preset 350m --moe-impl grouped --block-size 1024 --batch-size 4 "
            "--steps 0 --seed 2026 --cfg num_experts=24 moe_intermediate_size=192 num_experts_per_tok=4 "
            "num_shared_experts=0 sliding_window=256 max_window_layers=10 bias_update_rate=3e-3")


def dry_run():
    """train.main() with the released arguments, zero steps, on the CPU."""
    with tempfile.TemporaryDirectory() as d:
        sys.argv = ["train.py", "--device", "cpu", "--out", str(Path(d) / "ckpt.pt"),
                    "--data", str(META)] + RELEASED.split()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            T.main()
    lines = [l for l in buf.getvalue().splitlines() if l.startswith(("sampler", "config", "params", "optimizer"))]
    assert "397.66M total, 173.53M active/token" in buf.getvalue()
    return lines


def tiny():
    return Mdl.AnuLMConfig(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8,
                           n_kv_head=2, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
                           intermediate_size=64, n_group=1, topk_group=1,
                           seq_balance_alpha=1e-4, router_z_alpha=1e-3)    # regularisers on, as in the run


def step(model, opt, batches, args, step_no, divide=True):
    """train.py lines 686-712 for one step, without the teacher, logging or eval."""
    lr = T.lr_at(step_no, args)
    opt.set_lr_scale(lr / args.lr)
    opt.zero_grad(set_to_none=True)
    loss_acc = aux_acc = 0.0
    for x, y in batches:
        _, loss = model(x, y)
        (loss / len(batches) if divide else loss).backward()
        loss_acc += loss.item() / len(batches)
        aux_acc += model.aux_loss.item() / len(batches)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)   # returns the norm before clipping
    opt.step()
    imbalance = model.update_expert_biases()
    return loss_acc, aux_acc, float(norm), lr, imbalance


def main():
    if META.exists():
        for line in dry_run():
            print(line)
    else:
        print(f"skipping part 1: {META} not found (it needs the v2 corpus, local only)")

    torch.manual_seed(0)
    cfg = tiny()
    data = torch.randint(0, cfg.vocab_size, (5_000,))
    sampler = T.WindowSampler(len(data), 16, 4, seed=0)
    batches = [T.make_batch(data, sampler.next(), 16, "cpu") for _ in range(2)]        # M = 2, B = 4
    whole = [(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches]))]   # M = 1, B = 8
    args = SimpleNamespace(lr=3e-4, min_lr=3e-5, warmup=100, steps=2000, weight_decay=0.1, grad_clip=1.0,
                           device="cpu", optimizer="adamw", schedule="cosine")
    base = Mdl.AnuLM(cfg).train()
    runs = {}
    for name, b, divide in (("M=2, divided", batches, True), ("M=1, B=8", whole, True),
                            ("M=2, not divided", batches, False)):
        m = copy.deepcopy(base)
        with contextlib.redirect_stdout(io.StringIO()):                    # make_optimizer prints a line
            opt = T.make_optimizer(m, args)
        loss, aux, norm, lr, imb = step(m, opt, b, args, step_no=0, divide=divide)
        runs[name] = (m, loss, aux, norm)
        print(f"{name:17s} | loss {loss - aux:6.4f} | aux {aux:.4f} | lr {lr:.2e} | "
              f"imbalance {imb:4.2f}x | grad norm {norm:.4f}")

    a, b, c = (runs[k] for k in ("M=2, divided", "M=1, B=8", "M=2, not divided"))
    assert abs(a[1] - b[1]) < 1e-5 and abs(a[3] - b[3]) < 1e-5
    print(f"\naccumulated vs one big batch: loss differs by {abs(a[1] - b[1]):.1e}, grad norm by {abs(a[3] - b[3]):.1e}")
    print(f"not dividing by M: grad norm x{c[3] / a[3]:.3f}")
    dw = max((p - q).abs().max().item() for p, q in zip(a[0].parameters(), c[0].parameters()))
    moved = max(p.sub(q).abs().max().item() for p, q in zip(a[0].parameters(), base.parameters()))
    print(f"after clip + one AdamW step the weights differ by at most {dw:.1e} "
          f"(the step itself moved them by up to {moved:.1e})")

    x, y = batches[0]
    with torch.no_grad():
        m = copy.deepcopy(base).train()
        logits, loss = m(x, y)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.reshape(-1))
    print(f"model loss {loss.item():.6f} = cross-entropy {ce.item():.6f} + aux {m.aux_loss.item():.6f}")
    assert abs(loss.item() - ce.item() - m.aux_loss.item()) < 1e-6


if __name__ == "__main__":
    main()
