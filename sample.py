"""
Generate from a AnuLM checkpoint.

    python sample.py --ckpt ckpt.pt --prompt "भारत" --tokens 300
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import torch

from model import AnuLM, load_checkpoint

# Windows consoles default to cp1252, which cannot encode Devanagari (or Tamil,
# or most of what this model is for). Force UTF-8 on the way out.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ckpt.pt")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-cache", action="store_true",
                   help="recompute the whole prefix every token (the pre-KV-cache path)")
    p.add_argument("--moe-impl", choices=["auto", "sparse", "dense", "grouped"], default="auto",
                   help="auto = grouped on CUDA, the checkpoint's own setting on CPU")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ck = load_checkpoint(args.ckpt, args.device)
    cfg = ck["cfg"]
    # The dispatch implementation is not part of the weights, so a checkpoint
    # trained with `sparse` can decode with `grouped`. It should: sparse does a
    # `nonzero` + `.numel()` per expert per layer, i.e. a GPU sync each -- ~240
    # per generated token on the 350M -- and that, not attention, is what makes
    # sparse decoding 17 tok/s regardless of the KV cache. On CPU sparse stays.
    if args.moe_impl == "auto":
        args.moe_impl = "grouped" if args.device.startswith("cuda") else cfg.moe_impl
    cfg = replace(cfg, moe_impl=args.moe_impl)
    model = AnuLM(cfg).to(args.device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded step {ck['step']}, val loss {ck['val_loss']:.4f}  "
          f"(moe_impl={cfg.moe_impl}, cache={'off' if args.no_cache else 'on'})\n")

    tok = None
    if getattr(ck["cfg"], "tokenizer_path", None):
        from bpe import BPE
        tok = BPE.load(ck["cfg"].tokenizer_path)
        prompt_ids = tok.encode(args.prompt) or tok.encode(" ")
        # Only a model whose vocab includes the EOS slot was trained to emit it;
        # older checkpoints (vocab == merges + 256) simply never stop early.
        eos_id = tok.eos_id if ck["cfg"].vocab_size > tok.eos_id else None
    else:
        # Byte-level: the prompt is its own UTF-8 encoding. 257 is EOS.
        prompt_ids = list(args.prompt.encode("utf-8")) or [10]
        eos_id = 257
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
        out = model.generate(ids, args.tokens, temperature=args.temperature, top_k=args.top_k,
                             eos_id=eos_id, use_cache=not args.no_cache)
    n_new = out.shape[1] - ids.shape[1]
    print(f"[{n_new} tokens in {time.time()-t0:.1f}s, {n_new/max(time.time()-t0,1e-9):.0f} tok/s"
          f"{', stopped at EOS' if n_new < args.tokens else ''}]\n")

    if tok is not None:
        print(tok.decode(out[0].tolist()))
    else:
        # Two ways a byte model produces something that is not a byte string:
        #  * ids 256-258 (BOS/EOS/PAD) are in the 259-token vocab and never
        #    appear in training data, but the head still scores them and top-k
        #    can pick one -- `bytes()` raises on anything above 255, so drop
        #    them here rather than crashing after a minute of generation;
        #  * invalid UTF-8 mid-sequence, especially early in training --
        #    multi-byte scripts need several correct bytes in a row.
        raw = bytes(i for i in out[0].tolist() if i < 256)
        print(raw.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
