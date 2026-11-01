"""
Book example, not repository code.

Chapter 12: causal masking.

    python book/code-examples/ch12_masks.py

12.2  The toy `The dog bit`: the mask added as −∞ before the softmax, and what
      goes wrong when it is applied after it instead.
12.3  What `model._attn_mask` returns in each case it handles.
12.4  The offset case: two new tokens against a cache that holds one. The
      explicit mask reproduces the full forward; SDPA's `is_causal` does not.
12.5  The sliding window on the toy, and its arithmetic at the released size.
12.6  Four wrong masks swapped into `model.py`, and which tests catch each.

`model.py` and `test_model.py` are imported unchanged; the mutants are swapped
in for one test call each and restored.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged
from tests import test_model as T  # noqa: E402  the repository's tests, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
cpu = torch.device("cpu")


def grid(mask) -> str:
    """A boolean (S, T) mask as rows of 1 (may attend) and . (masked)."""
    return "\n".join("    " + " ".join("1" if v else "." for v in row) for row in mask.tolist())


def toy():
    """Chapter 11's toy: Q, K, V for The, dog, bit (d = 2)."""
    X = torch.tensor([[1., 0.], [0., 2.], [1., 1.]])
    W_q, W_k, W_v = torch.tensor([[1., 0.], [0., 2.]]), torch.eye(2), torch.tensor([[0., 1.], [1., 0.]])
    return X @ W_q, X @ W_k, X @ W_v


def before_and_after() -> None:
    print("12.2  −∞ before the softmax, not 0 after it (toy, illustrative)")
    Q, K, V = toy()
    s = Q @ K.T / math.sqrt(2)                                  # (S, T) = (3, 3)
    allowed = torch.ones(3, 3, dtype=torch.bool).tril()         # True = may attend
    right = s.masked_fill(~allowed, float("-inf")).softmax(-1)
    wrong = s.softmax(-1) * allowed                             # zeroed after the softmax
    print("  −∞ before:  rows", [round(r, 3) for r in right[0].tolist()],
          "sums", [round(x, 3) for x in right.sum(-1).tolist()])
    print("  0 after:    rows", [round(r, 3) for r in wrong[0].tolist()],
          "sums", [round(x, 3) for x in wrong.sum(-1).tolist()])
    print("  a row with every key masked:", torch.full((3,), float("-inf")).softmax(-1).tolist())


def cases() -> None:
    print("\n12.3  What _attn_mask returns (S queries, T keys, start, window)")
    for S, T_, start, window in ((3, 3, 0, None), (1, 3, 2, None), (2, 3, 1, None),
                                 (3, 3, 0, 2), (1, 3, 2, 2)):
        m = M._attn_mask(S, T_, start, window, cpu)
        what = "None" if m is None else f"{tuple(m.shape)}\n{grid(m)}"
        print(f"  S={S} T={T_} start={start} window={window}: {what}")


def offset() -> None:
    print("\n12.4  Two new tokens (dog, bit) against a cache holding The")
    Q, K, V = (t[None, None] for t in toy())                   # (B=1, H=1, 3, d=2)
    scale = 2 ** -0.5
    full = M._sdpa(Q, K, V, None, 3, scale, 0.0)[0, 0]          # whole sequence: is_causal
    q = Q[:, :, 1:]                                             # the S = 2 new queries
    mask = M._attn_mask(2, 3, 1, None, cpu)                     # start = 1 token cached
    explicit = M._sdpa(q, K, V, mask, 2, scale, 0.0)[0, 0]
    top_left = M._sdpa(q, K, V, None, 2, scale, 0.0)[0, 0]      # what is_causal would do
    r = lambda t: [[round(x, 3) for x in row] for row in t.tolist()]
    print("  full forward, rows dog and bit:", r(full[1:]))
    print("  explicit offset mask:          ", r(explicit))
    print("  is_causal (top-left):          ", r(top_left))
    print("  the mask is_causal builds for (2, 3):\n" + grid(torch.ones(2, 3, dtype=torch.bool).tril()))
    assert torch.allclose(explicit, full[1:], atol=1e-6)
    assert not torch.allclose(top_left, full[1:], atol=1e-2)


def window() -> None:
    print("\n12.5  A window of 2 on the toy, and a window of 256 at the released size")
    Q, K, V = (t[None, None] for t in toy())
    mask = M._attn_mask(3, 3, 0, 2, cpu)
    s = (Q @ K.transpose(-2, -1) / math.sqrt(2)).masked_fill(~mask, float("-inf"))
    w = s.softmax(-1)[0, 0]
    print("  weights, window 2:", [[round(x, 3) for x in r] for r in w.tolist()])
    print("  bit's output:", [round(x, 3) for x in (w @ V[0, 0])[2].tolist()])
    m = M._attn_mask(1024, 1024, 0, 256, cpu)
    causal = 1024 * 1025 // 2
    print(f"  S = T = 1,024: keys per query {int(m.sum(-1).min())}…{int(m.sum(-1).max())}; "
          f"allowed pairs {int(m.sum()):,} of {causal:,} causal ({m.sum().item() / causal:.1%})")
    torch.manual_seed(0)
    model = M.AnuLM(T.tiny(sliding_window=4, max_window_layers=2)).eval()
    cache = model.new_cache()
    with torch.no_grad():
        for i in range(12):
            model(torch.randint(0, 64, (1, 1)), cache=cache, start=i)
    print(f"  tiny() with a window of 4, after 12 cached steps: layer 0 caches "
          f"{cache[0]['k'].shape[2]} positions")


def wrong_masks():
    """Four mutants of _attn_mask: (name, function)."""
    real = M._attn_mask

    def window_off_by_one(S, T_, start, window, device):
        m = real(S, T_, start, None, device)
        if window is None:
            return m
        q = torch.arange(start, start + S, device=device)[:, None]
        k = torch.arange(T_, device=device)[None, :]
        return (k <= q) & (k >= q - window)                     # window + 1 keys

    def inverted(S, T_, start, window, device):
        m = real(S, T_, start, window, device)
        return None if m is None else ~m

    def no_offset(S, T_, start, window, device):
        return real(S, T_, 0, window, device)                   # ignores the cache

    def reversed_always(S, T_, start, window, device):           # no early return, comparison reversed
        q = torch.arange(start, start + S, device=device)[:, None]
        k = torch.arange(T_, device=device)[None, :]
        return k >= q

    return [("window one too wide", window_off_by_one), ("returned mask inverted", inverted),
            ("offset ignored", no_offset), ("reversed, always built", reversed_always)]


def mutants() -> None:
    print("\n12.6  Wrong masks swapped into model.py: which tests fail")
    tests = [T.causal_mask_does_not_leak_future, T.incremental_forward_matches_full_forward,
             T.sliding_window_limits_reach, T.sliding_window_agrees_with_cache]
    real = M._attn_mask
    for name, fn in [("the real _attn_mask", real)] + wrong_masks():
        failed = []
        for t in tests:
            M._attn_mask = fn
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    t()
            except Exception:                                   # AssertionError, or a crash
                failed.append(t.__name__)
            finally:
                M._attn_mask = real
        print(f"  {name:22s} fails: {', '.join(failed) or 'nothing'}")


def exercises() -> None:
    print("\n12.8  Answers")
    print("  1. S=2, T=4, start=2:\n" + grid(M._attn_mask(2, 4, 2, None, cpu)))
    row = M._attn_mask(1, 601, 600, 256, cpu)[0]
    seen = row.nonzero().flatten()
    print(f"  2. query at 600, window 256: {int(row.sum())} keys, positions {int(seen[0])}–{int(seen[-1])}")
    m = M._attn_mask(512, 512, 0, 256, cpu)
    causal = 512 * 513 // 2
    print(f"  4. S = T = 512, window 256: {int(m.sum()):,} of {causal:,} causal pairs ({m.sum().item() / causal:.1%})")


if __name__ == "__main__":
    before_and_after()
    cases()
    offset()
    window()
    mutants()
    exercises()
