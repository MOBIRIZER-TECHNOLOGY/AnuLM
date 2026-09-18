"""
Sample many prompts from one checkpoint, loading the model once.

    python sample_many.py --ckpt ckpt_coder.pt
    python sample_many.py --ckpt ckpt_coder.pt --tokens 120 --only python

`sample.py` reloads a 1.6 GB checkpoint per prompt, which dominates the
wall clock when you want a dozen. This keeps the model resident and walks a
prompt list -- the default set probes each language in the coder mix
(65% Python, 30% English, 5% Hindi) plus the HumanEval-style docstring
shape the benchmark actually asks for. Code prompts decode at temperature
0.2, prose at 0.7, which is where each reads best.

Same decode path as sample.py: grouped dispatch on CUDA, KV cache on, EOS
honoured when the checkpoint's vocab carries the slot.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import torch

from model import AnuLM, load_checkpoint

PROMPTS = [
    ("python", 0.2, "def add(a, b):"),
    ("python", 0.2, "def reverse_string(s):"),
    ("python", 0.2, "def count_vowels(text):"),
    ("python", 0.2, 'def is_palindrome(s):\n    """Return True if s reads the same forwards and backwards."""\n'),
    ("python", 0.2, "import os\n\n\ndef list_python_files(directory):"),
    ("python", 0.2, "class Stack:\n    def __init__(self):"),
    ("english", 0.7, "The capital of France is"),
    ("english", 0.7, "Machine learning is a field of study that"),
    ("hindi", 0.7, "भारत एक देश है"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt_coder.pt")
    p.add_argument("--tokens", type=int, default=100)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--only", default=None, help="run just one group: python, english, hindi")
    p.add_argument("--rep", type=float, default=1.0, help="repetition penalty (CTRL rule); 1.0 = off")
    p.add_argument("--temp", type=float, default=None, help="override the per-prompt temperature")
    args = p.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ck = load_checkpoint(args.ckpt, args.device)
    cfg = replace(ck["cfg"], moe_impl="grouped" if args.device.startswith("cuda") else ck["cfg"].moe_impl)
    model = AnuLM(cfg).to(args.device).eval()
    model.load_state_dict(ck["model"])
    print(f"{args.ckpt}: step {ck['step'] + 1:,}, val loss {ck['val_loss']:.4f}, "
          f"{cfg.moe_impl} dispatch, {args.device}\n")

    tok, eos_id = None, 257
    if getattr(cfg, "tokenizer_path", None):
        from bpe import BPE
        tok = BPE.load(cfg.tokenizer_path)
        eos_id = tok.eos_id if cfg.vocab_size > tok.eos_id else None

    for group, temp, prompt in PROMPTS:
        if args.only and group != args.only:
            continue
        torch.manual_seed(args.seed)
        ids = tok.encode(prompt) if tok is not None else list(prompt.encode("utf-8"))
        x = torch.tensor([ids or [10]], dtype=torch.long, device=args.device)
        t0 = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=args.device.startswith("cuda")):
            out = model.generate(x, args.tokens, temperature=args.temp or temp, top_k=args.top_k,
                                 eos_id=eos_id, use_cache=True,
                                 repetition_penalty=args.rep)
        n_new = out.shape[1] - x.shape[1]
        text = (tok.decode(out[0].tolist()) if tok is not None
                else bytes(i for i in out[0].tolist() if i < 256).decode("utf-8", errors="replace"))
        print("=" * 72)
        print(f"[{group}, temperature {args.temp or temp}, rep {args.rep}, {n_new} tokens in {time.time() - t0:.1f}s"
              f"{', stopped at EOS' if n_new < args.tokens else ''}]")
        print(text)
        print()


if __name__ == "__main__":
    main()
