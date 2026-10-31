"""
Chapter 13 checks: hearing and seeing.

    python ch13.py                  # your course/build/post.py
    python ch13.py --impl solution  # the reference solution
    python ch13.py --self-test      # proves each check catches the bug it is for

A model fed pictures has more ways to be quietly wrong than one fed text:
new rows far from the old ones, patches pooled across the wrong edge,
captions graded on the picture, an input a hundred times louder than a word.
Each of these still trains.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import main, solution, with_replaced  # noqa: E402
import omni_train  # noqa: E402
import speech_vocab  # noqa: E402
import vision_encoder  # noqa: E402

CHECKS = []
B = solution(8)
KW = dict(vocab_size=64, block_size=80, n_layer=2, hidden_size=32, n_head=4, head_dim=8, n_kv_head=2,
          num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, intermediate_size=64,
          rope_theta=10000.0)


def check(fn):
    CHECKS.append(fn)
    return fn


def model():
    torch.manual_seed(0)
    return B.MoELM(**KW).eval()


@check
def new_rows_match_speech_vocab(M):
    torch.manual_seed(0)
    w = torch.randn(50, 32) * 0.3 + 0.1
    got = M.grow_rows(w, 7, torch.Generator().manual_seed(3))
    want = speech_vocab._grow(w, 7, torch.Generator().manual_seed(3))
    torch.testing.assert_close(got, want, msg="new rows differ from speech_vocab._grow")


@check
def growing_keeps_every_old_id(M):
    """Both tables grow, and on the old ids the model gives exactly what it gave before."""
    m = model()
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        before = m(x)[0]
        size = M.grow_vocab(m, 5)
        after = m(x)[0]
    assert size == 69 and m.embed_tokens.weight.shape[0] == 69, "the embedding table did not grow to 69"
    assert m.lm_head.weight.shape[0] == 69, "the output head did not grow: the model could read the new ids but never write them"
    torch.testing.assert_close(after[..., :64], before, msg="growing the vocabulary changed the model on old ids")


@check
def embeddings_in_give_the_same_model(M):
    """Fed the embeddings of some ids, the forward pass from vectors equals the forward pass from ids."""
    m = model()
    x = torch.randint(0, 64, (2, 20))
    with torch.no_grad():
        want_logits, want_loss = m(x[:, :-1], x[:, 1:])
        got_logits, got_loss = M.forward_embeds(m, m.embed_tokens(x[:, :-1]), x[:, 1:])
    torch.testing.assert_close(got_logits, want_logits, rtol=1e-4, atol=1e-5, msg="logits from embeddings differ")
    assert abs(got_loss.item() - want_loss.item()) < 1e-4, "the loss from embeddings differs"


@check
def projector_matches_vision_encoder(M):
    """Same layers, same 2-D pooling: a state dict loads into either and gives the same output."""
    torch.manual_seed(0)
    theirs = vision_encoder.ImageProjector(12, 32, 2, hidden=40)
    mine = M.ImageProjector(12, 32, 2, hidden=40)
    mine.load_state_dict(theirs.state_dict())
    patches = torch.randn(36, 12)                      # a 6 x 6 grid
    with torch.no_grad():
        torch.testing.assert_close(mine(patches), theirs(patches), msg="projector output differs from vision_encoder's")


@check
def pooling_joins_grid_neighbours(M):
    """Each pooled vector is one 2 x 2 square of the grid, in reading order: top-left, top-right,
    bottom-left, bottom-right."""
    proj = M.ImageProjector(1, 4, 2, hidden=4)
    seen = []
    proj.net[0].register_forward_hook(lambda mod, inp, out: seen.append(inp[0]))
    grid = torch.arange(16.0)[:, None]                 # a 4 x 4 grid holding its own index
    with torch.no_grad():
        proj(grid)
    assert seen[0][0].tolist() == [0, 1, 4, 5] and seen[0][1].tolist() == [2, 3, 6, 7], \
        f"first pooled squares {seen[0][:2].tolist()}; want [0, 1, 4, 5] and [2, 3, 6, 7]"


def fake_vl(m, proj, bos, eos):
    return types.SimpleNamespace(model=m, proj=proj, vocab=types.SimpleNamespace(image_bos=bos, image_eos=eos))


@check
def captions_are_built_like_vision_encoder(M):
    """The picture is read, never graded; the answer is graded, one position early."""
    m = model()
    torch.manual_seed(1)
    proj = M.ImageProjector(12, 32, 2, hidden=40)
    patches = torch.randn(16, 12)
    for prompt in (None, [7, 8, 9]):
        with torch.no_grad():
            got = M.build_caption(m, proj, patches, 60, 61, [3, 4, 5, 6], prompt)
            want = vision_encoder.VisionLanguage.build(fake_vl(m, proj, 60, 61), patches, [3, 4, 5, 6], prompt)
        torch.testing.assert_close(got[0], want[0], msg="embeddings differ from VisionLanguage.build")
        assert got[1].tolist() == want[1].tolist(), f"targets {got[1].tolist()}; VisionLanguage.build's {want[1].tolist()}"


@check
def projected_vectors_arrive_at_a_words_size(M):
    """After match_scale, the projector's output is as large as a text embedding, not sqrt(hidden)."""
    torch.manual_seed(0)
    emb = torch.randn(64, 32) * 0.05
    mine = M.ImageProjector(12, 32, 2, hidden=40)
    theirs = vision_encoder.ImageProjector(12, 32, 2, hidden=40)
    theirs.load_state_dict(mine.state_dict())
    with torch.no_grad():
        mine.net[-1].bias.normal_(0, 0.5)
        theirs.net[-1].bias.copy_(mine.net[-1].bias)
    before, target = M.match_scale(mine, emb)
    fake = types.SimpleNamespace(model=types.SimpleNamespace(embed_tokens=nn.Embedding.from_pretrained(emb)),
                                 vocab=types.SimpleNamespace(text=64))
    with contextlib.redirect_stdout(io.StringIO()):                # it prints what it did
        omni_train.Omni._match_scale(fake, theirs, "test")
    torch.testing.assert_close(mine.net[-1].weight, theirs.net[-1].weight, msg="gain differs from omni_train's")
    torch.testing.assert_close(mine.net[-1].bias, theirs.net[-1].bias, msg="bias differs from omni_train's")
    with torch.no_grad():
        out = mine(torch.randn(400, 12)).norm(dim=-1).mean().item()
    assert abs(out / target - 1) < 0.15, f"projected vectors have norm {out:.3f}; text embeddings {target:.3f}"


# --- broken versions for --self-test ----------------------------------------------------

NAMES = ("grow_rows", "grow_vocab", "forward_embeds", "ImageProjector", "build_caption", "match_scale")


def MUTANTS(sol):
    def fresh_rows(weight, n_new, gen):
        return torch.cat([weight, torch.randn(n_new, weight.shape[1], generator=gen) * 0.02])

    @torch.no_grad()
    def embeddings_only(model, n_new, seed=0):
        gen = torch.Generator().manual_seed(seed)
        e = sol.grow_rows(model.embed_tokens.weight.float(), n_new, gen)
        model.embed_tokens = nn.Embedding.from_pretrained(e, freeze=False)
        return e.shape[0]

    def no_final_norm(model, embeds, targets=None):
        x = embeds
        cos, sin = model.rotary(x.shape[1], x.device, x.dtype)
        for layer in model.layers:
            out = layer(x, cos, sin)
            x = out[0] if isinstance(out, tuple) else out
        logits = model.lm_head(x)
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                                            ignore_index=-1)
        return logits, loss

    def no_positions(model, embeds, targets=None):
        x = embeds
        cos, sin = model.rotary(x.shape[1], x.device, x.dtype)
        cos, sin = torch.ones_like(cos), torch.zeros_like(sin)
        for layer in model.layers:
            out = layer(x, cos, sin)
            x = out[0] if isinstance(out, tuple) else out
        logits = model.lm_head(model.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                                            ignore_index=-1)
        return logits, loss

    class FlatPool(sol.ImageProjector):
        def forward(self, patches):
            P, D = patches.shape
            return self.net(patches.reshape(P // self.pool ** 2, D * self.pool ** 2))

    class NoLayerNorm(sol.ImageProjector):
        def __init__(self, in_dim, out_dim, pool=2, hidden=None):
            super().__init__(in_dim, out_dim, pool, hidden)
            self.net[-1] = nn.LayerNorm(out_dim, elementwise_affine=True)
            self.net[-1].forward = lambda x: x                       # the norm is skipped

    def image_graded(model, proj, patches, image_bos, image_eos, answer_ids, prompt_ids=None):
        embeds, targets = sol.build_caption(model, proj, patches, image_bos, image_eos, answer_ids, prompt_ids)
        targets[0] = image_bos
        return embeds, targets

    def unshifted(model, proj, patches, image_bos, image_eos, answer_ids, prompt_ids=None):
        embeds, targets = sol.build_caption(model, proj, patches, image_bos, image_eos, answer_ids, prompt_ids)
        t = torch.full_like(targets, -1)
        t[-len(answer_ids):] = torch.tensor(answer_ids)
        return embeds, t

    def no_image_eos(model, proj, patches, image_bos, image_eos, answer_ids, prompt_ids=None):
        emb = model.embed_tokens
        parts = [emb(torch.tensor([image_bos])), proj(patches)]
        if prompt_ids:
            parts.append(emb(torch.tensor(prompt_ids)))
        parts.append(emb(torch.tensor(answer_ids)))
        embeds = torch.cat(parts)
        targets = torch.full((embeds.shape[0],), -1, dtype=torch.long)
        start = embeds.shape[0] - len(answer_ids)
        targets[start - 1: start - 1 + len(answer_ids)] = torch.tensor(answer_ids)
        return embeds, targets

    @torch.no_grad()
    def gain_only(proj, text_embeddings):
        target = text_embeddings.float().norm(dim=-1).mean()
        ln = proj.net[-1]
        now = ln.weight.float().norm()
        ln.weight.mul_(target / now)
        return now.item(), target.item()

    @torch.no_grad()
    def per_element(proj, text_embeddings):
        target = text_embeddings.float().abs().mean()
        ln = proj.net[-1]
        now = (ln.weight.float().pow(2).sum() + ln.bias.float().pow(2).sum()).sqrt()
        ln.weight.mul_(target / now)
        ln.bias.mul_(target / now)
        return now.item(), target.item()

    bugs = [("new rows drawn from N(0, 0.02)", dict(grow_rows=fresh_rows)),
            ("only the embeddings grown", dict(grow_vocab=embeddings_only)),
            ("final norm skipped", dict(forward_embeds=no_final_norm)),
            ("no positions for embedded inputs", dict(forward_embeds=no_positions)),
            ("patches pooled with a flat stride", dict(ImageProjector=FlatPool)),
            ("projector without its LayerNorm", dict(ImageProjector=NoLayerNorm)),
            ("the image graded", dict(build_caption=image_graded)),
            ("caption targets not shifted", dict(build_caption=unshifted)),
            ("no end-of-image marker", dict(build_caption=no_image_eos)),
            ("scale applied to the gain only", dict(match_scale=gain_only)),
            ("scale matched per element, not per vector", dict(match_scale=per_element))]
    return [(d, with_replaced(sol, NAMES, **r)) for d, r in bugs]


if __name__ == "__main__":
    main(13, CHECKS, MUTANTS, __doc__, learner="post.py")
