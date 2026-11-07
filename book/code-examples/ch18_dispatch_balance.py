"""Book example, not repository code.

Chapter 18: what happens after the router has chosen, and how the choice is
kept balanced.

    python book/code-examples/ch18_dispatch_balance.py

1. Dispatch by hand: five tokens, two experts each, four experts. The index
   tensors each path builds, then the three paths of `model.MoE` against a
   naive per-token loop.
2. The sign rule of `update_expert_biases`, run on loads set by hand: the
   step, the imbalance it reports, and the drift of the biases' mean.
3. What a large common bias does to number formats: fp32 and bf16 spacing.
4. The sequence-balance loss and the router z-loss, recomputed by hand and
   compared with `Router`'s aux.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm import model as M  # noqa: E402  the repository's model.py, unchanged

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.set_printoptions(precision=4, sci_mode=False)


def tiny(**kw) -> M.AnuLMConfig:
    """test_model.py's tiny(), copied so this script needs no test harness."""
    base = dict(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
                num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
                n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


def dispatch_by_hand():
    torch.manual_seed(0)
    moe = M.MoE(tiny(num_experts=4, num_experts_per_tok=2, num_shared_experts=0)).eval()
    x = torch.randn(5, 32)                                           # N = 5 tokens
    idx = torch.tensor([[0, 2], [2, 3], [0, 1], [2, 0], [3, 2]])     # (N, A): the router's choice
    w = torch.tensor([[1.5, 1.0], [2.0, 0.5], [1.25, 1.25], [1.0, 1.5], [0.5, 2.0]])
    flat_expert = idx.reshape(-1)                                    # (N*A,) one slot per (token, choice)
    tok_of_slot = torch.arange(5).repeat_interleave(2)               # which token owns each slot
    order = torch.argsort(flat_expert, stable=True)                  # slots grouped by expert
    counts = torch.zeros(4, dtype=torch.int64).scatter_add_(0, flat_expert, torch.ones_like(flat_expert))
    offs = counts.cumsum(0)
    print("flat_expert", flat_expert.tolist())
    print("tok_of_slot", tok_of_slot.tolist())
    print("order      ", order.tolist(), "-> experts", flat_expert[order].tolist())
    print("counts", counts.tolist(), "offs", offs.tolist())

    with torch.no_grad():
        ref = torch.stack([sum(w[n, s] * moe.experts[idx[n, s]](x[n]) for s in range(2)) for n in range(5)])
        for name in ("sparse", "dense"):
            y = getattr(moe, f"_route_{name}")(x, idx, w)
            print(f"{name:7s} vs naive loop: {(y - ref).abs().max():.1e}")
        if hasattr(F, "grouped_mm"):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                yg = moe._route_grouped(x, idx, w)
                ys = moe._route_sparse(x, idx, w)
            print(f"grouped vs sparse, both under bf16 autocast: {(yg.float() - ys.float()).abs().max():.1e}; "
                  f"grouped vs fp32 loop: {(yg.float() - ref).abs().max():.1e} (output scale {ref.abs().max():.2f})")


def sign_rule():
    torch.manual_seed(0)
    cfg = tiny(num_experts=24, num_experts_per_tok=4, bias_update_rate=3e-3)
    m = M.AnuLM(cfg)
    routers = [mod for mod in m.modules() if isinstance(mod, M.Router)]
    loads = torch.tensor([30., 26, 22, 18] + [8.] * 20)              # four busy experts, twenty starved
    for r in routers:
        r.load_counts.copy_(loads)
    worst = m.update_expert_biases()
    b = routers[0].expert_bias
    print(f"loads sum {loads.sum():.0f}, mean {loads.mean():.2f}; returned imbalance max/mean = {worst:.3f}; "
          f"median layer {m.imbalance_median:.3f}")
    print("bias after one step:", [round(v, 4) for v in b[:6].tolist()], "...")
    under, over = int((loads < loads.mean()).sum()), int((loads > loads.mean()).sum())
    print(f"{under} experts under the mean, {over} over: the biases' mean moved by {b.mean():.6f} "
          f"= 3e-3 * ({under} - {over}) / 24 = {3e-3 * (under - over) / 24:.6f}")
    print("load_counts after the update:", routers[0].load_counts.sum().item())


def formats():
    for v in (216.0, 430.0, 628.8):
        t = torch.tensor(v)
        f32 = (torch.nextafter(t, torch.tensor(1e9)) - t).item()
        tb = t.bfloat16()
        b16 = (torch.nextafter(tb, torch.tensor(1e9, dtype=torch.bfloat16)).float() - tb.float()).item()
        print(f"at {v:6.1f}: fp32 spacing {f32:.2e}, bf16 spacing {b16:g}")
    t = torch.tensor(628.8)
    print(f"628.8 + 0.003 in fp32 moves it by {(t + 3e-3 - t).item():.6f}")
    biases = torch.tensor([628.7236, 628.7774, 628.8133, 628.8492])  # four of layer 2's stored biases
    print("those biases in bf16:", biases.bfloat16().float().tolist())


def regularisers():
    torch.manual_seed(0)
    E, A, T = 4, 2, 3                                                # one sequence of 3 tokens
    r = M.Router(tiny(num_experts=E, num_experts_per_tok=A, seq_balance_alpha=1.0)).train()
    x = torch.randn(T, 32)
    idx, w, aux = r(x, seq_len=T)
    s = torch.sigmoid(F.linear(x, r.weight))                         # (T, E)
    f = F.one_hot(idx, E).sum(1).float().mean(0) * (E / A)          # share of slots, 1.0 if uniform
    P = (s / s.sum(-1, keepdim=True)).mean(0)                        # mean normalised score
    print("chosen", idx.tolist())
    print("f", f, "P", P.detach(), "sum f*P", f"{(f * P).sum():.6f}", "Router aux", f"{aux.item():.6f}")
    rz = M.Router(tiny(num_experts=E, num_experts_per_tok=A, router_z_alpha=1.0)).train()
    rz.load_state_dict(r.state_dict())
    _, _, auxz = rz(x, seq_len=T)
    z = torch.logsumexp(F.linear(x, rz.weight), -1).pow(2).mean()
    print(f"z-loss by hand {z:.6f}, Router aux {auxz.item():.6f}")
    print("load_counts accumulated in training mode:", r.load_counts.tolist())


def graph_breaks():
    import torch._dynamo as dynamo
    for name, cfg in (("tiny()", tiny(moe_impl="sparse")), ("tiny_gqa", M.AnuLMConfig.tiny_gqa(moe_impl="sparse"))):
        torch.manual_seed(0)
        m = M.AnuLM(cfg).eval()
        x = torch.randint(0, cfg.vocab_size, (2, 16))
        dynamo.reset()
        e = dynamo.explain(lambda t: m(t, t))(x)
        print(f"sparse path, {name}, torch {torch.__version__}: {e.graph_break_count} graph breaks, "
              f"{e.graph_count} graphs")
    dynamo.reset()


def exercises():
    loads = torch.tensor([12., 4, 4, 4])
    print("exercise 1: imbalance", (loads.max() / loads.mean()).item(), "bias step", [round(v, 4) for v in (1e-3 * torch.sign(loads.mean() - loads)).tolist()])
    print("exercise 2: a gap of 1.0 at rate 3e-3 needs about", round(1.0 / (2 * 3e-3)), "steps")
    print("slots per MoE layer in one step of the released run (batch 4 x 1,024 tokens, A = 4):", 4 * 1024 * 4)


if __name__ == "__main__":
    dispatch_by_hand()
    sign_rule()
    formats()
    regularisers()
    graph_breaks()
    exercises()
