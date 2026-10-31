"""
Chapter 7 checks: the router, the experts, balancing, and the whole model.

    python ch07.py                  # your course/build/anulm.py
    python ch07.py --impl solution  # the reference solution
    python ch07.py --self-test      # proves each check catches the bug it is for

Mixture-of-experts bugs are quiet by nature: a router that always picks the
same two experts, or a balancer pushing the wrong way, gives a model that
trains, with most of its parameters doing nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, with_replaced  # noqa: E402
import model as ref  # noqa: E402  the repository's model.py

CHECKS = []
D = 32


def check(fn):
    CHECKS.append(fn)
    return fn


def rcfg(**kw):
    base = dict(hidden_size=D, n_head=4, n_kv_head=2, head_dim=8, num_experts=8, num_experts_per_tok=2,
                moe_intermediate_size=16, intermediate_size=64, vocab_size=64, block_size=64)
    base.update(kw)
    return ref.AnuLMConfig(**base)


def router_pair(M, **kw):
    torch.manual_seed(0)
    c = rcfg(**kw)
    theirs = ref.Router(c)
    with torch.no_grad():
        theirs.expert_bias.uniform_(-0.2, 0.2)                 # a non-zero bias, as in training
    mine = M.Router(D, c.num_experts, c.num_experts_per_tok, c.n_group, c.topk_group, c.routed_scaling_factor,
                    c.seq_balance_alpha, c.router_z_alpha)
    mine.load_state_dict(theirs.state_dict())
    return theirs, mine


# --- the router -----------------------------------------------------------------------------

@check
def router_matches_model(M):
    """Same weight and bias, same choices and weights, with and without group-limited routing."""
    x = torch.randn(64, D)
    for kw in (dict(), dict(num_experts=16, n_group=4, topk_group=2, num_experts_per_tok=3)):
        theirs, mine = router_pair(M, **kw)
        theirs.eval(), mine.eval()
        (i1, w1, _), (i2, w2, _) = theirs(x), mine(x)
        assert torch.equal(i1, i2), f"expert choices differ from model.py ({kw or 'no groups'})"
        torch.testing.assert_close(w2, w1, msg=f"combine weights differ ({kw or 'no groups'})")


@check
def bias_picks_but_never_weights(M):
    """The balancing bias changes which experts win, never how much they count. A bias that
    leaks into the weights still trains, and quietly distorts every output."""
    _, r = router_pair(M)
    r.eval()
    x = torch.randn(32, D)
    with torch.no_grad():
        r.expert_bias.zero_()
    i0, w0, _ = r(x)
    with torch.no_grad():
        r.expert_bias[3] += 5.0
    i1, w1, _ = r(x)
    assert (i1 == 3).any(), "a huge bias on expert 3 never selected it"
    same = (i0 == i1).all(-1)
    assert same.any() and (w0[same] - w1[same]).abs().max().item() < 1e-6, \
        "the bias changed the weights of tokens whose choices did not change"


@check
def router_respects_group_limits(M):
    _, r = router_pair(M, num_experts=16, n_group=4, topk_group=2)
    r.eval()
    idx, w, _ = r(torch.randn(64, D))
    for n in range(64):
        assert len({e // 4 for e in idx[n].tolist()}) <= 2, f"token {n} used more than 2 groups"
    assert (w.sum(-1) - 2.5).abs().max() < 1e-5, "weights do not sum to routed_scaling_factor (2.5)"


@check
def expert_bias_is_a_buffer_not_a_parameter(M):
    """A Parameter would get gradients and weight decay; the optimizer would fight the balancer."""
    _, r = router_pair(M)
    assert "expert_bias" not in dict(r.named_parameters()), "expert_bias is a Parameter"
    assert "expert_bias" in dict(r.named_buffers()) and "load_counts" in dict(r.named_buffers()), \
        "expert_bias and load_counts must be buffers (saved, no gradient)"


@check
def load_is_counted_in_training_only(M):
    _, r = router_pair(M)
    x = torch.randn(40, D)
    r.eval()
    r(x)
    assert r.load_counts.sum().item() == 0, "load was counted in eval mode"
    r.train()
    r(x)
    assert r.load_counts.sum().item() == 40 * 2, f"counted {r.load_counts.sum().item():.0f} slots, expected 80"


@check
def routing_stays_float32_under_autocast(M):
    """Top-k is a discrete choice: bfloat16 rounding of the scores flips some of them.
    model.py once routed in bfloat16 on every GPU run (see the chapter)."""
    torch.manual_seed(1)
    c = rcfg(num_experts=64, num_experts_per_tok=6)
    theirs = ref.Router(c).eval()
    mine = M.Router(D, 64, 6).eval()
    mine.load_state_dict(theirs.state_dict())
    x = torch.randn(4096, D)
    want, _, _ = theirs(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got, w, _ = mine(x)
    flipped = (got.sort(-1)[0] != want.sort(-1)[0]).any(-1).float().mean().item()
    assert flipped == 0.0, f"under bfloat16 autocast {flipped:.1%} of tokens chose different experts"


@check
def balance_and_z_losses_match_model(M):
    torch.manual_seed(0)
    x = torch.randn(8 * 4, D)
    theirs, mine = router_pair(M, seq_balance_alpha=0.01, router_z_alpha=0.001)
    theirs.train(), mine.train()
    a1, a2 = theirs(x, seq_len=4)[2], mine(x, seq_len=4)[2]
    assert a2 is not None, "no auxiliary loss returned although both regularisers are on"
    torch.testing.assert_close(a2, a1, msg="sequence-balance + z-loss differ from model.py")


# --- the MoE layer ----------------------------------------------------------------------------

def moe_pair(M, **kw):
    torch.manual_seed(0)
    c = rcfg(**kw)
    theirs = ref.MoE(c).eval()
    mine = M.MoE(D, c.moe_intermediate_size, c.num_experts, c.num_experts_per_tok, c.num_shared_experts).eval()
    mine.load_state_dict(theirs.state_dict())
    return theirs, mine


@check
def moe_matches_model(M):
    theirs, mine = moe_pair(M)
    x = torch.randn(2, 9, D)
    with torch.no_grad():
        torch.testing.assert_close(mine(x)[0], theirs(x)[0], rtol=1e-4, atol=1e-5, msg="MoE output differs from model.py")


@check
def moe_matches_a_per_token_loop(M):
    """The slow, obviously-right way: for each token, each chosen expert, add weight x output."""
    _, mine = moe_pair(M, num_experts_per_tok=3)
    x = torch.randn(2, 5, D)
    with torch.no_grad():
        got, _ = mine(x)
        flat = x.view(-1, D)
        idx, w, _ = mine.router(flat)
        want = torch.zeros_like(flat)
        for n in range(flat.shape[0]):
            for s in range(3):
                want[n] += w[n, s] * mine.experts[idx[n, s]](flat[n])
        want = want.view_as(x) + mine.shared_expert(x)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5, msg="MoE differs from the per-token loop")


@check
def shared_expert_sees_every_token(M):
    _, mine = moe_pair(M)
    with torch.no_grad():
        for e in mine.experts:
            e.down_proj.weight.zero_()                         # silence every routed expert
        x = torch.randn(2, 7, D)
        torch.testing.assert_close(mine(x)[0], mine.shared_expert(x), msg="output is not the shared expert's")


# --- the whole model ----------------------------------------------------------------------------

# Every field set explicitly: model.py's config and MoELM have different defaults
# (rope_theta 1e4 against the 30b preset's 1e6), and a check that relied on either
# would compare two different models.
SMALL = dict(vocab_size=64, block_size=64, n_layer=3, hidden_size=D, n_head=4, head_dim=8, n_kv_head=2,
             num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
             rope_theta=10000.0)


@check
def expert_block_matches_model_layer(M):
    torch.manual_seed(0)
    c = rcfg()
    rot = ref.Rotary(c, 8)
    theirs = ref.Block(c, 1, rot).eval()                     # layer 1: an MoE layer
    moe_kw = dict(moe_intermediate_size=16, num_experts=8, top_k=2, num_shared_experts=1)
    mine = M.ExpertBlock(D, 4, 2, 8, moe_kw).eval()
    mine.load_state_dict(theirs.state_dict())
    x = torch.randn(2, 9, D)
    cos, sin = rot(9, x.device, x.dtype)
    with torch.no_grad():
        torch.testing.assert_close(mine(x, cos, sin)[0], theirs(x, cos, sin)[0], rtol=1e-4, atol=1e-5,
                                   msg="ExpertBlock differs from model.py's layer 1")


@check
def your_model_is_model_py(M):
    """The payoff: your weights load into model.py's AnuLM with nothing missing or extra, and
    give the same logits and the same loss, regulariser included."""
    torch.manual_seed(0)
    kw = dict(SMALL, seq_balance_alpha=0.01)
    mine = M.MoELM(**kw)
    theirs = ref.AnuLM(ref.AnuLMConfig(**kw))
    theirs.load_state_dict(mine.state_dict(), strict=True)
    x = torch.randint(0, 64, (2, 20))
    for mode in ("eval", "train"):
        getattr(mine, mode)(), getattr(theirs, mode)()
        with torch.no_grad():
            (a, la), (b, lb) = mine(x, x), theirs(x, x)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5, msg=f"logits differ from model.py ({mode})")
        torch.testing.assert_close(la, lb, rtol=1e-4, atol=1e-5, msg=f"loss differs from model.py ({mode})")


@check
def thirty_b_preset_has_the_right_parameter_count(M):
    """The defaults are the `30b` preset: 17,440,384 parameters, 6,602,368 active per token."""
    total, active = M.MoELM().num_params()
    assert (total, active) == (17_440_384, 6_602_368), \
        f"(total, active) = ({total:,}, {active:,}), expected (17,440,384, 6,602,368)"


@check
def config_rejects_unreachable_top_k(M):
    """Group limits can make top_k impossible: 16 experts in 4 groups, 1 group allowed, top 6."""
    try:
        M.MoELM(**dict(SMALL, num_experts=16, n_group=4, topk_group=1, num_experts_per_tok=6))
    except (AssertionError, ValueError):
        return
    raise AssertionError("an unreachable top_k was accepted")


def train_steps(m, n, x):
    for _ in range(n):
        m(x, x)[1].backward()
        m.zero_grad(set_to_none=True)
        yield m


@check
def bias_update_moves_the_right_way(M):
    """Overloaded experts' bias goes down by exactly the rate, starved ones' up, and the counts
    reset. Pushing the wrong way still trains: it just makes the favourites more favoured."""
    torch.manual_seed(0)
    m = M.MoELM(**dict(SMALL, bias_update_rate=0.05)).train()
    r = next(mod for mod in m.modules() if type(mod).__name__ == "Router")
    x = torch.randint(0, 64, (8, 32))
    for _ in train_steps(m, 6, x):
        load, before = r.load_counts.clone(), r.expert_bias.clone()
        m.update_expert_biases()
        delta, mean = r.expert_bias - before, load.mean()
        assert (delta[load > mean] < 0).all() and (delta[load < mean] > 0).all(), "bias moved the wrong way"
        assert torch.allclose(delta.abs()[load != mean], torch.full_like(delta[load != mean], 0.05)), \
            "the step is not bias_update_rate"
        assert r.load_counts.sum().item() == 0, "load_counts were not reset after the update"


@check
def balancer_recovers_from_collapse(M):
    """Force the router onto two experts; 200 steps of the bias rule must spread the load."""
    torch.manual_seed(0)
    m = M.MoELM(**dict(SMALL, bias_update_rate=0.01)).train()
    r = next(mod for mod in m.modules() if type(mod).__name__ == "Router")
    with torch.no_grad():
        r.expert_bias[0], r.expert_bias[1] = 1.0, 0.85
    x = torch.randint(0, 64, (8, 32))
    hist = []
    for _ in train_steps(m, 200, x):
        load = r.load_counts.clone()
        hist.append((load.max() / load.mean()).item())
        m.update_expert_biases()
    early, late = sum(hist[:10]) / 10, sum(hist[-20:]) / 20
    assert late < early * 0.6, f"imbalance went {early:.2f}x -> {late:.2f}x; the balancer is not working"


# --- broken versions for --self-test ----------------------------------------------------------

NAMES = ("Router", "MoE", "ExpertBlock", "MoELM")


def MUTANTS(sol):
    def router(**bug):
        class Buggy(sol.Router):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                if bug.get("bias_param"):
                    del self.expert_bias
                    self.expert_bias = nn.Parameter(torch.zeros(self.num_experts))

            def forward(self, x_flat, seq_len=None):
                if bug.get("bf16"):                              # no float32 guard
                    idx, w, aux = self._gate(x_flat, seq_len)
                    return idx, w.to(x_flat.dtype), aux
                if bug.get("count_always") and not self.training:
                    self.train()
                    out = super().forward(x_flat, seq_len)
                    self.eval()
                    return out
                return super().forward(x_flat, seq_len)

            def _gate(self, x, seq_len):
                if not (bug.get("softmax") or bug.get("biased_weights") or bug.get("no_groups")
                        or bug.get("bf16")):
                    return super()._gate(x, seq_len)
                logits = F.linear(x, self.weight if bug.get("bf16") else self.weight.float())
                scores = F.softmax(logits, -1) if bug.get("softmax") else torch.sigmoid(logits)
                choice = scores + self.expert_bias[None, :]
                if self.n_group > 1 and not bug.get("no_groups"):
                    N, per = scores.shape[0], self.num_experts // self.n_group
                    g = choice.view(N, self.n_group, per).topk(min(2, per), dim=-1)[0].sum(-1)
                    keep = torch.zeros_like(g).scatter_(1, g.topk(self.topk_group, dim=-1)[1], 1.0)
                    keep = keep.unsqueeze(-1).expand(N, self.n_group, per).reshape(N, -1)
                    choice = choice.masked_fill(~keep.bool(), float("-inf"))
                idx = choice.topk(self.top_k, dim=-1)[1]
                w = (choice if bug.get("biased_weights") else scores).gather(1, idx)
                w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.routed_scaling_factor
                if self.training:
                    with torch.no_grad():
                        self.load_counts += torch.bincount(idx.flatten(), minlength=self.num_experts).to(
                            self.load_counts.dtype)
                return idx, w, None
        return Buggy

    class Unweighted(sol.MoE):
        def forward(self, x):
            B, S, D_ = x.shape
            flat = x.view(-1, D_)
            idx, w, aux = self.router(flat, seq_len=S)
            y = torch.zeros_like(flat)
            for n in range(flat.shape[0]):
                for s in range(self.top_k):
                    y[n] += self.experts[idx[n, s]](flat[n])          # weights forgotten
            return y.view(B, S, D_) + self.shared_expert(x), aux

    class WrongTokenOfSlot(sol.MoE):
        def forward(self, x):
            B, S, D_ = x.shape
            flat = x.view(-1, D_)
            idx, w, aux = self.router(flat, seq_len=S)
            N = flat.shape[0]
            fe = idx.reshape(-1)
            tok = torch.arange(N).repeat(self.top_k)                   # repeat, not repeat_interleave
            out = flat.new_zeros(N * self.top_k, D_)
            for e in range(self.num_experts):
                sel = (fe == e).nonzero(as_tuple=True)[0]
                if sel.numel():
                    out[sel] = self.experts[e](flat[tok[sel]])
            y = (out.view(N, self.top_k, D_) * w.unsqueeze(-1)).sum(1).view(B, S, D_)
            return y + self.shared_expert(x), aux

    class NoShared(sol.MoE):
        def forward(self, x):
            y, aux = super().forward(x)
            return y - self.shared_expert(x), aux

    class WrongWayBalancer(sol.MoELM):
        @torch.no_grad()
        def update_expert_biases(self):
            for m in self.modules():
                if isinstance(m, sol.Router) and m.load_counts.sum() > 0:
                    m.expert_bias += self.bias_update_rate * torch.sign(m.load_counts - m.load_counts.mean())
                    m.load_counts.zero_()
            return 1.0

    class NeverResets(sol.MoELM):
        @torch.no_grad()
        def update_expert_biases(self):
            for m in self.modules():
                if isinstance(m, sol.Router) and m.load_counts.sum() > 0:
                    m.expert_bias += self.bias_update_rate * torch.sign(m.load_counts.mean() - m.load_counts)
            return 1.0

    class AllActive(sol.MoELM):
        def num_params(self):
            n = sum(p.numel() for p in self.parameters())
            return n, n

    class AuxDropped(sol.MoELM):
        def forward(self, idx, targets=None, cache=None, start=0):
            logits, _ = super().forward(idx)
            return logits, (None if targets is None else F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1))

    class NoTopkGuard(sol.MoELM):
        def __init__(self, *a, n_group=1, topk_group=1, num_experts_per_tok=2, num_experts=16, **k):
            ok = num_experts_per_tok <= (num_experts // n_group) * topk_group
            super().__init__(*a, n_group=n_group if ok else 1, topk_group=topk_group if ok else 1,
                             num_experts_per_tok=num_experts_per_tok, num_experts=num_experts, **k)

    bugs = [("router using softmax instead of sigmoid", dict(Router=router(softmax=True))),
            ("weights taken from the biased scores", dict(Router=router(biased_weights=True))),
            ("expert_bias as a trainable Parameter", dict(Router=router(bias_param=True))),
            ("routing in bfloat16 under autocast", dict(Router=router(bf16=True))),
            ("group limits ignored", dict(Router=router(no_groups=True))),
            ("load counted in eval mode too", dict(Router=router(count_always=True))),
            ("expert outputs summed without weights", dict(MoE=Unweighted)),
            ("slots mapped to the wrong tokens", dict(MoE=WrongTokenOfSlot)),
            ("shared expert left out", dict(MoE=NoShared)),
            ("balancer pushing the wrong way", dict(MoELM=WrongWayBalancer)),
            ("load counts never reset", dict(MoELM=NeverResets)),
            ("every expert counted as active", dict(MoELM=AllActive)),
            ("router regulariser not added to the loss", dict(MoELM=AuxDropped)),
            ("unreachable top_k accepted", dict(MoELM=NoTopkGuard))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(7, CHECKS, MUTANTS, __doc__)
