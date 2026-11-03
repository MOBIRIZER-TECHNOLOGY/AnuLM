"""
Book example, not repository code.

Chapter 26: bugs that training does not catch, put back into the repository's
code one at a time, and the repository's own tests catching them.

    python book/code-examples/ch26_silent_failures.py
    python book/code-examples/ch26_silent_failures.py --save   # also refresh book/measurements/part_v_b_silent_failures.json

Part 1 trains test_model.py's tiny() on random ids twice: once as committed,
once with the causal mask removed. The broken model's loss falls far below
the floor that random data allows, which is the whole problem: the loss says
"better" for a model that is cheating.

Part 2 is a mutation run. Each mutant reintroduces a failure the repository
met (the code comments and docs/RESULTS.md record them), as a temporary
monkeypatch; the matching test from test_model.py is run against the mutant
and then against the committed code. A good check fails on the mutant and
passes on the original.

One story has no test in the suite: bf16 autocast inside the router. For it
the script writes the check a test would make, router_choices_survive_autocast
(marked EXTENSION in the book), and runs it like the others. Part 3 shows why
casting the router's inputs to fp32 was not enough.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "book" / "code-examples"))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged
from tests import test_model as TM  # noqa: E402  the repository's tests, unchanged
from anulm.training import train as T  # noqa: E402
from trace_forward import released  # noqa: E402  Chapter 20's copy of the released configuration

warnings.filterwarnings("ignore", message="Sparse invariant checks")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# Results go to book/measurements/ only with --save, so a reader's rerun never overwrites them.
SAVE = "--save" in sys.argv[1:]
OUT = (ROOT / "book" / "measurements" if SAVE else Path.cwd()) / "part_v_b_silent_failures.json"


def leaky_sdpa(q, k, v, mask, S, scale, dropout_p):
    """Mutant: attention with no causal mask -- every position sees the whole window."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False,
                                          scale=scale, dropout_p=dropout_p)


def train_curve(steps=200):
    torch.manual_seed(0)
    m = M.AnuLM(TM.tiny())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(1)
    out = []
    for _ in range(steps):
        x = torch.randint(0, 64, (16, 33), generator=g)
        _, loss = m(x[:, :-1], x[:, 1:])           # targets are the inputs shifted by one
        opt.zero_grad()
        loss.backward()
        opt.step()
        out.append(loss.item())
    return out


class patched:
    """Set obj.name = value for the duration of a with-block."""
    def __init__(self, obj, name, value):
        self.obj, self.name, self.value = obj, name, value

    def __enter__(self):
        self.old = getattr(self.obj, self.name)
        setattr(self.obj, self.name, self.value)

    def __exit__(self, *exc):
        setattr(self.obj, self.name, self.old)


def run(test):
    try:
        test()
        return "PASS", ""
    except AssertionError as e:
        return "FAIL", str(e).splitlines()[0][:72]


def tail_split(data, chunk=262144, every=10):           # the split train.py used before --val-interleave
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def train_keeps_cache(self, mode=True):                  # AnuLM.train without dropping the eval cache
    return torch.nn.Module.train(self, mode)


_gqa_init = M.GQAttention.__init__


def gqa_init_without_temperature(self, cfg, rotary):     # GQA before YaRN's temperature reached it
    _gqa_init(self, cfg, rotary)
    self.scale = cfg.head_dim ** -0.5


def router_forward_under_autocast(self, x_flat, seq_len=None):   # Router.forward before the fix
    topk_idx, w, aux = self._gate(x_flat.float(), seq_len)
    return topk_idx, w.to(x_flat.dtype), aux


def router_choices_survive_autocast():
    """EXTENSION: the check the suite lacks. bf16 autocast must not change which experts a token gets."""
    torch.manual_seed(0)
    r = M.Router(released()).eval()                  # the released router's shape: 24 experts, top-4
    x = torch.randn(4096, r.weight.shape[1])
    want = r(x)[0].sort(-1)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = r(x)[0].sort(-1)[0]
    changed = (want != got).any(-1).float().mean().item()
    assert changed == 0.0, f"{100 * changed:.1f}% of expert sets changed under autocast"


def main():
    # GQAttention calls _sdpa only while NATIVE_GQA is off (model.py 454-463); the leaky mutant
    # patches _sdpa, so with native GQA on it would silently test nothing.
    assert not M.NATIVE_GQA, "the leaky mutant patches _sdpa, which native GQA bypasses"
    R = {}
    # ---- Part 1: the loss goes down, and the model is wrong
    ok = train_curve()
    with patched(M, "_sdpa", leaky_sdpa):
        leak = train_curve()
    print(f"Part 1. tiny() on uniform random ids; ln V = {torch.log(torch.tensor(64.)).item():.3f}")
    for s in (0, 49, 99, 149, 199):
        print(f"  step {s + 1:3d}: as committed {ok[s]:.3f}   mask removed {leak[s]:.3f}")
    R["curves"] = {"committed": ok, "leaky": leak}

    # ---- Part 2: each mutant against the test written for it
    mutants = [
        ("causal mask removed", (M, "_sdpa", leaky_sdpa), TM.causal_mask_does_not_leak_future),
        ("merge-tree cache kept across train()", (M.AnuLM, "train", train_keeps_cache),
         TM.merge_tree_eval_cache_sees_optimizer_updates),
        ("validation = last 10% of the file", (T, "interleaved_split", tail_split),
         TM.interleaved_split_samples_the_whole_corpus),
        ("YaRN temperature dropped from GQA", (M.GQAttention, "__init__", gqa_init_without_temperature),
         TM.yarn_temperature_reaches_attention),
        ("router gate left under bf16 autocast", (M.Router, "forward", router_forward_under_autocast),
         router_choices_survive_autocast),
    ]
    print("\nPart 2. mutant -> its test (four from test_model.py, one proposed here)")
    R["mutants"] = []
    for name, (obj, attr, val), test in mutants:
        with patched(obj, attr, val):
            bad, msg = run(test)
        good, _ = run(test)
        print(f"  {name:38s} mutant {bad}   committed {good}   [{test.__name__}]")
        if msg:
            print(f"      {msg}")
        R["mutants"].append({"mutant": name, "test": test.__name__, "on_mutant": bad, "on_committed": good,
                             "message": msg})

    # ---- Part 3: why casting the inputs was not enough
    x, w = torch.randn(4, 8), torch.randn(3, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        print(f"\nPart 3. under autocast, F.linear on fp32 tensors returns {F.linear(x.float(), w.float()).dtype}")
    OUT.write_text(json.dumps(R, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
