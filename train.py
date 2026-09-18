"""
AnuLM training loop.

    python train.py --preset 30b --steps 2000
    python train.py --preset 105b --data mycorpus.txt --device cuda --compile

Byte-level tokenisation, so any script works out of the box -- Devanagari,
Tamil, Latin, code-mixed Hinglish -- without a tokenizer to train first. That is
a deliberate stand-in for the real thing: Sarvam's actual contribution at this
layer is a 262144-token tokenizer covering 22 languages across 12 scripts, whose
fertility (tokens per word) on Indic text is the reason their models are cheap
to run in those languages. Bytes are the honest placeholder -- worst possible
fertility, zero training cost -- not an imitation of it.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch

from bpe import DOC_SEP
from model import AnuLM, AnuLMConfig, Router


def atomic_save(obj, path) -> None:
    """torch.save, but a kill mid-write cannot destroy the existing file.

    `<out>.last` is 4.8 GB on the 400M coder and takes seconds to write.
    Killed in the middle -- and on Windows a scheduled task shares the
    interactive console, so Ctrl+C reaches it -- `torch.save` leaves a
    truncated zip that `torch.load` refuses with "failed finding central
    directory", i.e. the resume point is gone. That happened at step
    265,000 of the coder run on 2026-09-12 and cost the optimizer state.
    Write beside it and rename: `os.replace` is atomic within a volume, so
    the old checkpoint survives until the new one is complete.
    """
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

HERE = Path(__file__).parent

# A tiny multilingual fallback so the script runs with no data and no network.
FALLBACK = """भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं।
India is a vast country where many languages are spoken.
தமிழ் ஒரு பழமையான மொழி ஆகும்.
ಕನ್ನಡ ಭಾಷೆಯು ಸುಂದರವಾಗಿದೆ.
আমি বাংলায় গান গাই।
Yaar, this code-mixing is totally normal na?
मैं office जा रहा हूँ, meeting है आज।
The quick brown fox jumps over the lazy dog.
"""


def load_bytes(path: str | None) -> torch.Tensor:
    """Return the corpus as a tensor of ids. Bytes 0..255 map to ids 0..255.
    At every DOC_SEP an EOS (257) is inserted, which needs int16 storage; a
    corpus without separators stays uint8 -- one document, no EOS."""
    if path:
        raw = Path(path).read_bytes()
    else:
        default = HERE / "data" / "input.txt"
        if default.exists():
            raw = default.read_bytes()
        else:
            default.parent.mkdir(parents=True, exist_ok=True)
            try:
                import urllib.request
                url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
                       "master/data/tinyshakespeare/input.txt")
                raw = urllib.request.urlopen(url, timeout=15).read()
                default.write_bytes(raw)
                print(f"downloaded tinyshakespeare -> {default}")
            except Exception as e:
                print(f"no data and download failed ({type(e).__name__}); "
                      f"using the built-in fallback corpus")
                raw = (FALLBACK * 400).encode("utf-8")
                default.write_bytes(raw)
    # Text files written on Windows carry CRLF; the tokenizer path reads them
    # in text mode and never sees it, so the byte path normalises too -- one
    # byte per newline, and DOC_SEP matches whichever platform wrote the file.
    raw = raw.replace(b"\r\n", b"\n")
    sep = DOC_SEP.encode("utf-8")
    if sep not in raw:
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    parts = []
    for doc in raw.split(sep):
        if doc.strip():
            parts.append(torch.frombuffer(bytearray(doc), dtype=torch.uint8).to(torch.int16))
            parts.append(torch.tensor([BYTE_EOS], dtype=torch.int16))
    return torch.cat(parts)


BYTE_EOS = 257          # byte mode: 256 BOS, 257 EOS, 258 PAD


class WindowSampler:
    """Training windows without replacement, in epochs.

    Each epoch lays a grid of non-overlapping `block_size` windows over the
    data from a fresh random offset -- so window boundaries move between
    epochs -- and visits them in a fresh random order. Every token is seen once
    per epoch (bar the sub-window remainder at each end). Sampling with
    replacement, which this replaces, sees some windows three times and others
    never over the 1.25-2.4 epochs the 350M runs make.

    Restartable: `state()` / `load_state()` ride in the `.last` checkpoint so
    `--resume` continues the same permutation rather than restarting the epoch.
    """

    def __init__(self, n_tokens: int, block_size: int, batch_size: int, seed: int):
        self.n, self.bs, self.B = n_tokens, block_size, batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self._new_epoch()

    def _new_epoch(self):
        offset = int(torch.randint(self.bs, (1,), generator=self.g))
        n_win = (self.n - 1 - offset) // self.bs          # each window needs block_size + 1 ids
        starts = offset + self.bs * torch.arange(n_win)
        self.order = starts[torch.randperm(n_win, generator=self.g)]
        self.pos = 0

    def next(self) -> torch.Tensor:
        if self.pos + self.B > len(self.order):
            self.epoch += 1
            self._new_epoch()
        ix = self.order[self.pos: self.pos + self.B]
        self.pos += self.B
        return ix

    @property
    def epochs_done(self) -> float:
        return self.epoch + self.pos / max(len(self.order), 1)

    def state(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "order": self.order, "g": self.g.get_state()}

    def load_state(self, s: dict):
        self.epoch, self.pos, self.order = s["epoch"], s["pos"], s["order"]
        self.g.set_state(s["g"])


def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int, device: str):
    x = torch.stack([data[i: i + block_size] for i in ix]).long()
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix]).long()
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, iters: int, seed: int = 0):
    """The SAME windows every evaluation: a private generator, reseeded on each
    call. Two evals then differ only by the model, and evaluation no longer
    consumes the global RNG the training data order depends on.

    Kept exactly as it was, because every val loss in docs/RESULTS.md was read
    through it and changing which windows it returns would silently move all of
    them. `--eval-windows N` selects `strided_windows` instead."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(len(data) - block_size - 1, (batch_size,), generator=g)
            for _ in range(iters)]


def strided_windows(data: torch.Tensor, batch_size: int, block_size: int, n: int):
    """`n` windows spread evenly across the whole split, in batches.

    Two things this fixes about the random set above. It **covers** the split
    instead of sampling it, so the same number of windows carries less
    variance -- random draws clump, and a clump of one register (all code, or
    all headings) is exactly the accident that moves a small estimate. And it
    does not depend on `batch_size`: the random set redraws when the batch
    size changes, so two runs that differ only in `--batch-size` cannot be
    compared. Here `n` windows are `n` windows.

    Deterministic by construction, so there is nothing to cache: the same
    corpus and the same `n` give the same windows on any machine.
    """
    span = len(data) - block_size - 1
    assert span > 0, "validation split is shorter than one window"
    n = min(n, span)
    step = span / n
    # round-half-down on the midpoint of each cell: evenly spaced, no duplicates
    # until n approaches span, and independent of how they are then batched.
    idx = torch.tensor([int(i * step + step / 2) for i in range(n)], dtype=torch.long)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


def lr_at(step: int, args, start_step: int = 0) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / max(args.warmup, 1)
    progress = min((step - args.warmup) / max(args.steps - args.warmup, 1), 1.0)
    lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))
    # Extending a finished run (--resume with a larger --steps) lands mid-cosine,
    # several times the lr the weights last saw. --rewarmup ramps back up to
    # the schedule over N steps instead of jumping.
    rewarmup = getattr(args, "rewarmup", 0)
    if rewarmup and start_step and step < start_step + rewarmup:
        lr *= (step - start_step + 1) / rewarmup
    return lr


class OptimizerSet:
    """One handle over possibly several optimizers.

    Muon needs a companion AdamW for the 1D and embedding parameters, and the
    two run at very different base learning rates (0.02 vs 3e-4). The schedule
    scales every group by the same *factor* rather than assigning one absolute
    lr, so the ratio between them is preserved.
    """

    def __init__(self, opts):
        self.opts = opts if isinstance(opts, (list, tuple)) else [opts]
        self._base = [[g["lr"] for g in o.param_groups] for o in self.opts]

    def set_lr_scale(self, scale: float):
        for o, bases in zip(self.opts, self._base):
            for g, b in zip(o.param_groups, bases):
                g["lr"] = b * scale

    @property
    def lr(self):                       # representative lr, for logging
        return self.opts[0].param_groups[0]["lr"]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd):
            o.load_state_dict(s)


def make_optimizer(model: AnuLM, args):
    """Weight-decay the matmul weights, not the norms/biases/router bias.

    The router's `expert_bias` is a buffer rather than a parameter precisely so
    it cannot end up here -- it is updated by its own rule, and letting AdamW
    or weight decay touch it would fight the balancing.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = args.device.startswith("cuda") and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    if args.optimizer == "muon":
        from muon import build_optimizers
        opts, desc = build_optimizers(model, muon_lr=args.muon_lr, adamw_lr=args.lr,
                                      weight_decay=args.weight_decay, fused=fused)
        print(f"optimizer: {desc}")
        return OptimizerSet(opts)
    adamw = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                              **({"fused": True} if fused else {}))
    print(f"optimizer: AdamW{' (fused)' if fused else ''}, "
          f"{sum(p.numel() for p in decay+no_decay)/1e6:.1f}M params")
    return OptimizerSet(adamw)


@torch.no_grad()
def evaluate(model, data, args, iters: int | None = None) -> float:
    """Mean LM loss over a fixed set of windows (see eval_windows). In eval mode
    the router regularisers are off, so this is pure cross-entropy.

    **How many windows this needs.** The default of 20 iterations is 20 x
    batch_size windows -- 160 on the coder run, 82k tokens, 0.03% of a 278M
    token val split. Measured on `ckpt_coder.pt` at step 265,000, six
    different window sets of that size gave 3.11, 3.22, 3.26, 3.32, 3.37,
    3.48: a spread of 0.36 and a standard deviation of 0.125. At iters=100
    the standard error is 0.039, at 400 it is 0.019.

    The fixed seed means successive evals see the *same* windows, so the
    curve is internally consistent and its large moves are real -- phase 2
    of the coder run fell 0.160, well outside the noise. But a delta of
    0.01-0.03 read off this curve says nothing about the model, and
    best-checkpoint selection is picking between numbers that differ by
    less than the sampling error of the estimate. Use `--eval-iters 200`
    on a new run; it costs ~30 s per eval, under 2% of a 33-minute eval
    interval. It is not changed mid-run, because adding windows shifts the
    mean and breaks comparability with the run's earlier points.

    **Returns `(mean, standard_error)`,** and the log prints both, so the
    size of the delta you are reading is on the screen beside it rather
    than in this docstring. The error is the standard error of the mean
    across batches -- it describes how much this estimate would move if it
    had drawn different windows, which is the question being asked when two
    checkpoints are 0.01 apart. It does not describe how much the model
    would move on a different corpus.

    `--eval-windows N` swaps the random set for `strided_windows`, which
    covers the split evenly and does not change when the batch size does.
    """
    model.eval()
    fixed = getattr(args, "eval_windows", 0)
    if fixed:
        batches = strided_windows(data, args.batch_size, args.block_size, fixed)
    else:
        iters = iters if iters is not None else getattr(args, "eval_iters", 20)
        batches = eval_windows(data, args.batch_size, args.block_size, iters)
    losses = torch.zeros(len(batches))
    counts = torch.zeros(len(batches))
    for i, ix in enumerate(batches):
        x, y = make_batch(data, ix, args.block_size, args.device)
        with args.autocast:
            _, loss = model(x, y)
        losses[i] = loss.item()
        counts[i] = len(ix)
    model.train()
    # Weighted by windows, because a strided set can end in a short batch.
    mean = float((losses * counts).sum() / counts.sum())
    if len(batches) > 1:
        var = float((counts * (losses - mean) ** 2).sum() / counts.sum())
        sem = (var / len(batches)) ** 0.5
    else:
        sem = float("nan")
    return mean, sem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["30b", "105b", "350m"], default="30b",
                   help="30b = GQA + high rope_theta; 105b = MLA + YaRN-ready; "
                        "350m = 353M params, needs a GPU and --grad-ckpt")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-windows", type=int, default=0,
                   help="evaluate on N windows spread evenly over the split instead of "
                        "--eval-iters random batches. Deterministic, independent of "
                        "--batch-size, and lower variance for the same N. Off by default "
                        "so existing runs keep reading the same number; 400 is a good "
                        "value on a large split")
    p.add_argument("--eval-iters", type=int, default=20,
                   help="batches of --batch-size windows per evaluation. The "
                        "default 20 is cheap and NOISY: +/-0.125 on the coder "
                        "run's val split, so only moves larger than ~0.03 mean "
                        "anything. 200 costs ~30 s and cuts that ~3x. Changing "
                        "it mid-run breaks the curve's comparability.")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--moe-impl", choices=["sparse", "dense", "grouped"], default="sparse",
                   help="sparse: one GEMM per expert (default, best in fp32 eager); "
                        "grouped: one grouped GEMM for all experts, bf16, static "
                        "shapes -- the one to use on a GPU and under --compile; "
                        "dense: every expert on every token, static but E/k x FLOPs")
    p.add_argument("--seq-balance-alpha", type=float, default=0.0,
                   help="DeepSeek-V3 sequence-wise balance loss weight (they use 1e-4)")
    p.add_argument("--router-z-alpha", type=float, default=0.0,
                   help="router z-loss weight (ST-MoE uses 1e-3)")
    p.add_argument("--sliding-window", type=int, default=None,
                   help="window size for the first --max-window-layers layers")
    p.add_argument("--max-window-layers", type=int, default=0)
    p.add_argument("--sample-with-replacement", action="store_true",
                   help="the pre-WindowSampler behaviour: random windows with "
                        "replacement from the global RNG. Kept for A/B runs.")
    p.add_argument("--cfg", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override any AnuLMConfig field on top of the preset, "
                        "e.g. --cfg num_shared_experts=0 num_experts=24. Values are "
                        "cast to the field's type; this is how the ablations run.")
    p.add_argument("--yarn", action="store_true",
                   help="enable YaRN band-blending (for context extension runs)")
    p.add_argument("--out", type=str, default=str(HERE / "ckpt.pt"))
    p.add_argument("--resume", action="store_true",
                   help="continue from <out>.last (weights + optimizer state)")
    p.add_argument("--stop-at", type=int, default=None,
                   help="end this invocation once this step is reached, after an eval "
                        "and a <out>.last save, so a multi-day run proceeds in phases "
                        "between which the machine may sleep; rerun with --resume to continue")
    p.add_argument("--rewarmup", type=int, default=0,
                   help="on --resume, ramp the lr from 0 back to the schedule over "
                        "this many steps -- for extending a finished run with a "
                        "larger --steps, where the cosine would otherwise jump")
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                   help="muon = Muon on 2D hidden weights + AdamW on the rest; "
                        "one momentum buffer instead of two moments, so 12 vs 16 "
                        "bytes/param, which is what makes 443M fit 8 GB")
    p.add_argument("--muon-lr", type=float, default=0.02,
                   help="not comparable to --lr; Muon updates are orthogonalised")
    p.add_argument("--grad-ckpt", action="store_true",
                   help="recompute activations in backward; ~30%% slower, much "
                        "less activation memory")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dtype = (torch.bfloat16 if args.device.startswith("cuda")
             and torch.cuda.is_bf16_supported() else torch.float32)
    args.autocast = (torch.autocast(device_type="cuda", dtype=dtype)
                     if args.device.startswith("cuda") and dtype is torch.bfloat16
                     else torch.autocast(device_type="cpu", enabled=False))

    # --- data --------------------------------------------------------------
    # Two formats: raw text (byte-level, vocab 259) or a .bin from
    # `bpe.py encode` (token ids + metadata). bits/byte stays comparable across
    # both because the .bin remembers how many bytes each token covers.
    tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0
    if args.data and args.data.endswith(".bin"):
        blob = torch.load(args.data, weights_only=False)
        data = blob["ids"]
        vocab_size = blob["vocab_size"]
        tokenizer_path = blob["tokenizer"]
        args.bytes_per_token = blob["bytes_per_token"]
        print(f"token data: {len(data)/1e6:.1f}M tokens, vocab {vocab_size}, "
              f"{args.bytes_per_token:.2f} bytes/token ({tokenizer_path})")
    else:
        data = load_bytes(args.data)
        n_docs = int((data == BYTE_EOS).sum()) if data.dtype != torch.uint8 else 0
        if n_docs:
            print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each")
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus {len(data)/1e6:.2f}M {'tokens' if tokenizer_path else 'bytes'}"
          f"  ->  train {len(train_data)/1e6:.2f}M / val {len(val_data)/1e6:.2f}M")
    sampler = WindowSampler(len(train_data), args.block_size, args.batch_size, args.seed)
    print(f"sampler: {len(sampler.order)} windows/epoch, no replacement; "
          f"{args.steps * args.grad_accum / len(sampler.order) * args.batch_size:.2f} epochs planned")

    # --- model -------------------------------------------------------------
    builder = {"30b": AnuLMConfig.nano_30b,
               "105b": AnuLMConfig.nano_105b,
               "350m": AnuLMConfig.nano_350m}[args.preset]
    overrides = {}
    for kv in args.cfg:
        key, _, val = kv.partition("=")
        field = AnuLMConfig.__dataclass_fields__.get(key)
        assert field is not None and _, f"--cfg: unknown field {kv!r}"
        cast = {"int": int, "float": float, "bool": lambda s: s.lower() in ("1", "true", "yes"),
                "str": str}.get(str(field.type).replace("Optional[", "").rstrip("]"), str)
        overrides[key] = None if val.lower() == "none" else cast(val)
    if overrides:
        print(f"config overrides: {overrides}")
    kw = dict(block_size=args.block_size, yarn=args.yarn,
              yarn_original_context=args.block_size, moe_impl=args.moe_impl,
              vocab_size=vocab_size, tokenizer_path=tokenizer_path,
              seq_balance_alpha=args.seq_balance_alpha, router_z_alpha=args.router_z_alpha,
              sliding_window=args.sliding_window, max_window_layers=args.max_window_layers)
    kw.update(overrides)                      # --cfg wins over the named flags
    cfg = builder(**kw)
    if args.moe_impl == "grouped" and not args.device.startswith("cuda"):
        print("note: --moe-impl grouped computes in bf16 by the kernel's contract; "
              "on CPU without autocast that is slower than sparse, not faster")
    model = AnuLM(cfg).to(args.device)
    if args.grad_ckpt:
        model.enable_gradient_checkpointing()
    total, active = model.num_params()
    n_routers = sum(1 for m in model.modules() if isinstance(m, Router))
    print(f"nano_{args.preset}: attn={cfg.attn}  layers={cfg.n_layer}  "
          f"experts={cfg.num_experts} top-{cfg.num_experts_per_tok} "
          f"(+{cfg.num_shared_experts} shared)  moe layers={n_routers}")
    print(f"params: {total/1e6:.2f}M total, {active/1e6:.2f}M active/token "
          f"({100*active/total:.1f}%)  |  device={args.device} dtype={dtype}")

    raw_model = model
    if args.compile:
        if args.moe_impl == "sparse":
            print("warning: --compile with --moe-impl sparse gives ~8 graph breaks "
                  "per forward\n         (data-dependent `nonzero`), so it buys "
                  "little. Use --moe-impl dense\n         for a single captured graph.")
        print("compiling ...")
        compiled = torch.compile(model)
        # Inductor generates C++ and needs a host compiler (MSVC `cl` on Windows,
        # gcc/clang elsewhere). It only discovers that on the first real forward,
        # so probe here rather than letting it explode 200 steps in.
        try:
            # eval(): nn.Module starts in training mode, and a probe forward
            # in it would bank 8 tokens of router load_counts before step 0.
            model.eval()
            with torch.no_grad():
                compiled(torch.zeros(1, 8, dtype=torch.long, device=args.device))
            model = compiled
            print("  compiled backend ready")
        except Exception as e:
            print(f"  compile unavailable ({type(e).__name__}: {str(e).splitlines()[0][:80]})")
            print("  falling back to eager")

    opt = make_optimizer(raw_model, args)

    # --- resume --------------------------------------------------------------
    # A long run that dies at step 3000 should not start over. `--out` holds the
    # best-val checkpoint (what you sample from); `<out>.last` holds the most
    # recent step with optimizer state, which is what resuming actually needs --
    # AdamW's moments matter as much as the weights.
    # How the val loss is measured. Recorded in every checkpoint, because a
    # curve read on one window set cannot be compared with a curve read on
    # another -- and the random set silently changes when --batch-size does.
    eval_spec = ({"mode": "strided", "windows": args.eval_windows,
                  "block_size": args.block_size}
                 if args.eval_windows else
                 {"mode": "random-seed0", "iters": args.eval_iters,
                  "batch_size": args.batch_size, "block_size": args.block_size})

    start_step, best_val = 0, float("inf")
    last_path = Path(str(args.out) + ".last")
    if args.resume and last_path.exists():
        # Load to CPU, not the training device. Optimizer.load_state_dict moves
        # state tensors to each param's device by itself; loading the whole
        # 4.6 GB blob straight onto an 8 GB GPU -- and then keeping `ck`
        # referenced for the life of the run -- forced WDDM spillover and a 20x
        # slowdown. Measured the hard way.
        try:
            ck = torch.load(last_path, map_location="cpu", weights_only=False)
        except Exception as e:
            # A resume point written before atomic_save existed, or truncated
            # some other way, is unreadable. Do not start from scratch and do
            # not stop: fall back to the best-val checkpoint, which is a
            # separate smaller file and survives when the big one does not.
            # The cost is the optimizer moments and the sampler position --
            # expect a visible dent in the loss for a few hundred steps.
            print(f"WARNING: {last_path} is unreadable ({type(e).__name__}: "
                  f"{str(e)[:80]}).")
            if not Path(args.out).exists():
                raise
            ck = torch.load(args.out, map_location="cpu", weights_only=False)
            ck = {"model": ck["model"], "step": ck["step"],
                  "best_val": ck.get("val_loss", float("inf"))}
            print(f"WARNING: falling back to {args.out} at step {ck['step'] + 1} "
                  f"WITHOUT optimizer state or sampler position.")
        raw_model.load_state_dict(ck["model"])
        if "opt" in ck:
            assert len(ck["opt"]) == len(opt.opts), (
                f"{last_path} holds state for {len(ck['opt'])} optimizer(s) but --optimizer "
                f"{args.optimizer} builds {len(opt.opts)}; resume with the optimizer the run used")
            opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        best_val = ck.get("best_val", float("inf"))
        if "sampler" in ck:
            sampler.load_state(ck["sampler"])
        prev = ck.get("eval_spec")
        if prev and prev != eval_spec:
            print(f"  NOTE: this run measures val loss differently from the one it "
                  f"resumes ({prev} -> {eval_spec}). The curve's earlier points are "
                  f"not comparable with the ones from here.")
        print(f"resuming from {last_path} at step {start_step} "
              f"(best val so far {best_val:.4f}, epoch {sampler.epochs_done:.2f})")
        del ck
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    elif args.resume:
        print(f"--resume given but {last_path} does not exist; starting fresh")

    # --- loop ---------------------------------------------------------------
    t0 = time.time()
    tokens_seen = 0
    aux_on = args.seq_balance_alpha > 0 or args.router_z_alpha > 0
    model.train()

    end = min(args.steps, args.stop_at) if args.stop_at else args.steps
    for step in range(start_step, end):
        t_step = time.time()
        lr = lr_at(step, args, start_step)
        opt.set_lr_scale(lr / args.lr)

        opt.zero_grad(set_to_none=True)
        loss_acc = aux_acc = 0.0
        for _ in range(args.grad_accum):
            ix = (torch.randint(len(train_data) - args.block_size - 1, (args.batch_size,))
                  if args.sample_with_replacement else sampler.next())
            x, y = make_batch(train_data, ix, args.block_size, args.device)
            with args.autocast:
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
            loss_acc += loss.item() / args.grad_accum
            if raw_model.aux_loss is not None:
                aux_acc += raw_model.aux_loss.item() / args.grad_accum
            tokens_seen += x.numel()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        opt.step()
        # Out-of-band, after the gradient step, never part of the loss.
        imbalance = raw_model.update_expert_biases()

        if step % args.log_every == 0 or step == args.steps - 1:
            dt = time.time() - t0
            # `loss` is the LM cross-entropy alone, so it stays comparable
            # across runs with and without the regularisers.
            print(f"step {step:5d} | loss {loss_acc - aux_acc:6.4f} | "
                  f"{f'aux {aux_acc:.4f} | ' if aux_on else ''}lr {lr:.2e} | "
                  f"imbalance {imbalance:5.2f}x | {tokens_seen/max(dt,1e-9):8.0f} tok/s | "
                  f"{(time.time()-t_step)*1e3:5.0f} ms/step | epoch {sampler.epochs_done:.2f}")

        if (step + 1) % args.eval_every == 0 or step == end - 1:
            val, val_sem = evaluate(raw_model, val_data, args)
            flag = ""
            if val < best_val:
                best_val = val
                atomic_save({"model": raw_model.state_dict(), "cfg": cfg,
                             "step": step, "val_loss": val, "val_sem": val_sem,
                             "eval_spec": eval_spec}, args.out)
                flag = "  <- saved"
            # Always write the resume point, best or not: it exists to recover
            # from a kill, and the newest step is what we want to continue from.
            atomic_save({"model": raw_model.state_dict(), "opt": opt.state_dict(),
                         "cfg": cfg, "step": step, "val_loss": val, "val_sem": val_sem,
                         "eval_spec": eval_spec, "best_val": best_val,
                         "sampler": sampler.state()}, last_path)
            bpb = val / math.log(2) / args.bytes_per_token
            print(f"  eval @ {step:5d} | val loss {val:6.4f} +/- {val_sem:.4f} | "
                  f"bits/byte {bpb:5.3f}{flag}")

    if end < args.steps:
        print(f"\nphase done at step {end} of {args.steps} in {time.time()-t0:.0f}s | best val {best_val:.4f}"
              f" | resume point {last_path}\ncontinue with:  the same command plus --resume "
              f"(--stop-at {end} raised, or dropped for the rest of the run)")
        return
    print(f"\ndone in {time.time()-t0:.0f}s | best val {best_val:.4f} | ckpt {args.out}")
    print("sample with:  python sample.py --ckpt", args.out)


if __name__ == "__main__":
    main()
