"""A first CPU training run using the public model source; no data downloads.

    python book/tools/first_prediction.py
    python tools/first_prediction.py --source-root ../AnuLM

This book example trains only on repeated toy sentences. It writes no files.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2],
                        help="companion directory containing anulm/model.py (legacy model.py also supported)")
    parser.add_argument("--steps", type=int, default=80, help="training steps (default: 80)")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    source = args.source_root.resolve() / "anulm" / "model.py"
    if not source.is_file():
        source = args.source_root.resolve() / "model.py"
    if not source.is_file():
        parser.error(f"Cannot find {source}. Follow docs/reader/start-here.md and pass --source-root PATH.")
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        parser.error("PyTorch is missing. Activate your environment and run: python -m pip install torch numpy")
    # Load precisely the file the reader selected, without depending on the book's location.
    sys.path.insert(0, str(args.source_root.resolve()))
    spec = importlib.util.spec_from_file_location("book_first_model", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(7)
    torch.set_num_threads(2)
    cfg = module.AnuLMConfig(vocab_size=259, block_size=32, n_layer=2,
                            hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
                            intermediate_size=64, first_k_dense_replace=2,
                            num_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1)
    model = module.AnuLM(cfg).cpu()
    corpus = ("the sun is warm. the sky is blue. the sun is bright. " * 24).encode("utf-8")
    tokens = torch.tensor(list(corpus), dtype=torch.long)
    starts = torch.arange(8) * 7
    x = torch.stack([tokens[s:s + 32] for s in starts])
    y = torch.stack([tokens[s + 1:s + 33] for s in starts])

    def score() -> float:
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, y)
            return F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)).item()

    initial = score()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for _ in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final = score()
    model.eval()
    prompt = "the sun "
    with torch.no_grad():
        out = model.generate(torch.tensor([list(prompt.encode())]), 32, top_k=1, use_cache=False)
    continuation = bytes(t for t in out[0].tolist() if t < 256).decode("utf-8", errors="replace")
    total, _ = model.num_params()
    print(f"Companion source: {source}")
    print(f"Runtime: Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CPU")
    print(f"Model: 2 dense layers, {total:,} parameters; no routed experts")
    print(f"Initial loss: {initial:.4f} nats/byte")
    print(f"Final loss after {args.steps} steps: {final:.4f} nats/byte")
    print(f"Continuation: {continuation!r}")
    if final >= initial:
        print("CHECK FAILED: loss did not decrease. Check the runtime and configuration.")
        return 1
    print("PASS: loss decreased")
    print("This memorises toy text; it is not a general-purpose or production-ready model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
