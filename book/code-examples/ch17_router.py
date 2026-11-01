"""Book example, not repository code.

Chapter 17: the router, by hand and in `model.py`.

    python book/code-examples/ch17_router.py

1. Six experts, two chosen, one token: sigmoid scores, the balancing bias,
   top-k, the unbiased weights, normalising, x 2.5 (illustrative numbers);
   then the same numbers through `model.Router` to check them.
2. Group-limited top-k on sixteen experts in four groups: the best single
   expert can lose because its group is weak.
3. The routed scaling factor is the same as scaling every expert's down_proj.
4. The router's initial weights, which `AnuLM._init_weights` does not touch.
5. Why the gate runs with autocast off: autocast re-casts F.linear to bf16.
6. sigmoid(x) against e^x where the trained logits sit.
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
torch.set_printoptions(precision=3, sci_mode=False)


def tiny(**kw) -> M.AnuLMConfig:
    """test_model.py's tiny(), copied so this script needs no test harness."""
    base = dict(vocab_size=64, block_size=32, n_layer=3, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
                num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
                n_group=1, topk_group=1)
    base.update(kw)
    return M.AnuLMConfig(**base)


def router_with_logits(logits, **kw):
    """A Router whose input IS its logits: D = E and an identity weight matrix."""
    E = len(logits)
    r = M.Router(tiny(hidden_size=E, num_experts=E, **kw)).eval()
    with torch.no_grad():
        r.weight.copy_(torch.eye(E))
    return r, torch.tensor([logits])


def by_hand():
    logits = torch.tensor([0.0, 1.0, -1.0, 0.5, 2.0, -0.5])          # one token, E = 6
    bias = torch.tensor([0.0, -0.2, 0.0, 0.15, 0.0, 0.0])           # the balancing bias
    s = torch.sigmoid(logits)                                        # independent scores in (0, 1)
    choice = (s + bias).topk(2).indices                              # the bias picks ...
    w = s[choice] / s[choice].sum()                                  # ... the unbiased scores weight
    w = w * 2.5                                                      # routed_scaling_factor
    print("scores        ", s)
    print("scores + bias ", s + bias)
    print("top-2 without bias:", s.topk(2).indices.tolist(), " with bias:", choice.tolist())
    print("weights", w, "sum", f"{w.sum():.3f}")

    r, x = router_with_logits(logits.tolist(), num_experts_per_tok=2)
    with torch.no_grad():
        r.expert_bias.copy_(bias)
        idx, wr, aux = r(x)
    assert idx[0].tolist() == choice.tolist() and torch.allclose(wr[0], w), (idx, wr)
    print("model.Router:", idx[0].tolist(), wr[0], "(matches)")


def groups():
    #           group 0                group 1              group 2              group 3
    logits = [3.0, -3., -3., -3.,  1.0, 1.0, -3., -3.,  0.8, 0.8, -3., -3.,  -.5, -.5, -3., -3.]
    r, x = router_with_logits(logits, num_experts_per_tok=2, n_group=4, topk_group=2)
    s = torch.sigmoid(x[0])
    g_score = s.view(4, 4).topk(2, dim=-1).values.sum(-1)           # a group is worth its best two
    print("group scores", g_score, "-> groups", g_score.topk(2).indices.tolist())
    with torch.no_grad():
        idx, w, _ = r(x)
    print("best single expert:", s.argmax().item(), f"({s.max():.3f});",
          "chosen:", idx[0].tolist(), "weights", w[0])
    flat, _ = router_with_logits(logits, num_experts_per_tok=2)      # n_group = 1: no limit
    with torch.no_grad():
        print("without groups:", flat(x)[0][0].tolist())
    try:
        M.AnuLMConfig(num_experts=16, n_group=4, topk_group=1, num_experts_per_tok=8)
    except AssertionError as e:
        print("config refused:", e)


def scaling_factor():
    torch.manual_seed(0)
    a = M.MoE(tiny(num_shared_experts=0)).eval()                     # routed_scaling_factor 2.5
    b = M.MoE(tiny(num_shared_experts=0, routed_scaling_factor=1.0)).eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        for e in b.experts:
            e.down_proj.weight.mul_(2.5)                             # fold 2.5 into every expert
        x = torch.randn(2, 5, 32)
        diff = (a(x)[0] - b(x)[0]).abs().max().item()
    print(f"factor 2.5 vs factor 1.0 with down_proj x 2.5: largest difference {diff:.1e}")
    assert diff < 1e-6


def init():
    cfg = released()
    r = M.Router(cfg)
    D = cfg.hidden_size
    print(f"router weight: max |w| {r.weight.abs().max():.5f}, bound 1/sqrt(D) = {1 / math.sqrt(D):.5f}; "
          f"std {r.weight.std():.5f}, 1/sqrt(3D) = {1 / math.sqrt(3 * D):.5f}")
    torch.manual_seed(0)
    m = M.AnuLM(tiny())                                              # _init_weights runs here
    rw = m.layers[1].mlp.router.weight
    print(f"tiny(), after AnuLM.__init__: router max |w| {rw.abs().max():.4f} "
          f"(uniform bound {1 / math.sqrt(32):.4f}); a Linear's std {m.layers[1].mlp.experts[0].up_proj.weight.std():.4f}")
    torch.manual_seed(0)
    u = torch.randn(4096, D)
    u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))             # RMS-normalised, as the router sees it
    logits = F.linear(u, r.weight)
    s = torch.sigmoid(logits)
    print(f"at init: logit std {logits.std():.3f} (1/sqrt(3) = {1 / math.sqrt(3):.3f}); "
          f"mean score {s.mean():.3f}; sum of 24 scores {s.sum(-1).mean():.2f}")


def autocast():
    x, W = torch.randn(4, 32), torch.randn(8, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        inside = F.linear(x.float(), W.float()).dtype
        with torch.autocast("cpu", enabled=False):
            off = F.linear(x.float(), W.float()).dtype
    print(f"F.linear on fp32 inputs under bf16 autocast: {inside}; with autocast disabled: {off}")


def biased_weights_mistake():
    torch.manual_seed(0)
    r = M.Router(tiny(num_experts=24, num_experts_per_tok=4)).eval()
    with torch.no_grad():
        r.expert_bias.copy_(430.0 + 0.03 * torch.rand(24))         # trained-like: a large common offset
        x = torch.randn(3, 32)
        idx, w, _ = r(x)
        s = torch.sigmoid(F.linear(x, r.weight))
        wrong = (s + r.expert_bias).gather(1, idx)                   # the mistake: gather the biased scores
        wrong = wrong / wrong.sum(-1, keepdim=True) * 2.5
    print("correct weights, token 0:", [round(v, 3) for v in w[0].tolist()])
    print("biased-score weights, token 0:", [round(v, 3) for v in wrong[0].tolist()], "(all near 2.5/4 = 0.625)")


def sigmoid_tail():
    for x in (-3.4, math.log(0.186 / (1 - 0.186))):                   # the mean logit; the mean top score's logit
        print(f"logit {x:.2f}: sigmoid {torch.sigmoid(torch.tensor(x)):.4f}, exp {math.exp(x):.4f}, "
              f"exp/sigmoid = {math.exp(x) / torch.sigmoid(torch.tensor(x)).item():.3f}")


def exercises():
    s = torch.sigmoid(torch.tensor([0.0, 1.0, -1.0, 0.5, 2.0, -0.5]))
    bias = torch.tensor([0.0, -0.2, 0.0, 0.15, 0.0, 0.0])
    c = (s + bias).topk(3).indices
    print("exercise 1, top-3:", c.tolist(), "weights", s[c] / s[c].sum() * 2.5)
    bias2 = bias.clone()
    bias2[4] -= 0.5
    c2 = (s + bias2).topk(2).indices
    print("exercise 2, expert 4's bias lowered by 0.5:", c2.tolist(), "weights", s[c2] / s[c2].sum() * 2.5)
    print("exercise 3, E = 16, n_group = 4, topk_group = 2: largest legal A =", 16 // 4 * 2)


if __name__ == "__main__":
    by_hand()
    groups()
    scaling_factor()
    init()
    autocast()
    biased_weights_mistake()
    sigmoid_tail()
    exercises()
