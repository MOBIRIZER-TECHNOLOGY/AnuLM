"""Exact deterministic greedy decoding without changing the native model."""
import torch


@torch.no_grad()
def greedy_generate(model, idx, max_new_tokens, eos_id=None, repetition_penalty=1., on_token=None):
    done = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
    cache, start, pending = None, 0, idx
    for _ in range(max_new_tokens):
        if cache is None or start + pending.shape[1] > model.cfg.block_size:
            cache, start, pending = model.new_cache(), 0, idx[:, -model.cfg.block_size:]
        logits, _ = model(pending, cache=cache, start=start)
        start += pending.shape[1]
        logits = logits[:, -1].float()
        if getattr(model, 'text_vocab', None):
            logits[:, model.text_vocab:] = -float('inf')
        if repetition_penalty != 1.:
            seen = logits.gather(1, idx)
            logits.scatter_(1, idx, torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty))
        token = logits.argmax(dim=-1, keepdim=True)
        if eos_id is not None:
            token = torch.where(done[:, None], torch.full_like(token, eos_id), token)
        idx = torch.cat([idx, token], dim=1)
        pending = token
        if on_token is not None and on_token(token):
            break
        if eos_id is not None:
            done |= token[:, 0] == eos_id
            if bool(done.all()):
                break
    return idx
