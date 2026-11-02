"""
Book example, not repository code.

Chapter 32: a classification head that scores every option in one pass, and
the calibration that makes its probabilities mean what they say. Checked
against decide.py.

    python book/code-examples/ch32_decide_calibration.py

1. decide.encode on a made-up message and three options: where each option's
   last token sits.
2. `score_options` by hand (hidden states, gather, linear head) equals
   decide.Decider's forward on the tiny test model; in causal mode an
   option's score cannot see the options after it, in bidirectional mode it can.
3. Calibration on made-up, overconfident logits (2,000 examples, 10 classes,
   seed 0): `ece_by_hand` equals decide.metrics, decide.fit_temperature finds
   the temperature, and the argmax never changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.tasks import decide as D  # noqa: E402  the repository's decide.py, unchanged
from anulm.tokenization.bpe import BPE  # noqa: E402
from anulm.model import AnuLM  # noqa: E402
from tests.test_model import tiny  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def score_options(model, head, idx, pos):
    """idx (B, S) ids; pos (B, K) positions of each option's last token -> (B, K) logits."""
    x = F.embedding(idx, model._table("in"))                 # (B, S, D): the causal backbone
    cos, sin = model.rotary(idx.shape[1], x.device, x.dtype)
    for layer in model.layers:
        x, _ = layer(x, cos, sin)
    h = model.norm(x)                                        # (B, S, D)
    g = h.gather(1, pos[..., None].expand(-1, -1, h.shape[-1]))   # (B, K, D)
    return head(g).squeeze(-1)                               # (B, K)


def ece_by_hand(probs, y, bins=15):
    """Expected calibration error: bin by confidence, weight |confidence - accuracy| by bin size."""
    conf, pred = probs.max(1)
    right = (pred == y).float()
    ece = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            ece += m.float().mean() * (conf[m].mean() - right[m].mean()).abs()
    return float(ece)


def main() -> None:
    # 1. where the options end
    tok = BPE.train("Message: which of these does it ask for options card arrival "
                    "lost card refund exchange rate\n- " * 30, 300, verbose=False)
    opts = ["card_arrival", "lost_card", "refund"]
    ids, pos = D.encode(tok, "where is my card", opts)
    print(f"{len(ids)} ids; option last-token positions {pos}")
    for k, o in enumerate(opts):
        print(f"  ids[:{pos[k] + 1}] ends with {tok.decode(ids[:pos[k] + 1]).splitlines()[-1]!r}")

    # 2. one pass, K logits, against Decider
    torch.manual_seed(0)
    backbone = AnuLM(tiny(vocab_size=300, block_size=128)).eval()
    dec = D.Decider(backbone).eval()
    x, p, _ = D.batchify([(ids, pos, 0, opts)], tok, tok.eos_id)
    with torch.no_grad():
        theirs = dec(x, p)
        mine = score_options(backbone, dec.head, x, p)
    print(f"Decider logits {tuple(theirs.shape)}: {[round(v, 4) for v in theirs[0].tolist()]}; "
          f"by hand: largest difference {(theirs - mine).abs().max():.1e}")
    swapped = opts[:2] + ["exchange_rate"]                   # change only the LAST option
    ids2, pos2 = D.encode(tok, "where is my card", swapped)
    x2, p2, _ = D.batchify([(ids2, pos2, 0, swapped)], tok, tok.eos_id)
    with torch.no_grad():
        after = dec(x2, p2)
        dec.kind = "bidir"
        b1, b2 = dec(x, p), dec(x2, p2)
        dec.kind = "causal"
    print(f"change the last option: first two logits move by {(after[0, :2] - theirs[0, :2]).abs().max():.1e} "
          f"(causal), {(b2[0, :2] - b1[0, :2]).abs().max():.1e} (bidirectional)")

    # 3. calibration on made-up overconfident logits
    g = torch.Generator().manual_seed(0)
    N, K = 2000, 10
    true_logits = torch.randn(N, K, generator=g) * 1.5
    y = torch.multinomial(F.softmax(true_logits, 1), 1, generator=g).squeeze(1)
    logits = 3.0 * true_logits                               # the model is three times too sure
    tau = D.fit_temperature(logits, y)
    for name, t in (("raw", 1.0), (f"tau {tau:.2f}", tau)):
        probs = F.softmax(logits / t, 1)
        m = D.metrics(probs, y)
        assert abs(ece_by_hand(probs, y) - m["ece"]) < 1e-6
        sel = m["answer_if_conf>=0.9"]
        print(f"{name:9s} accuracy {100 * m['accuracy']:.1f}%  mean confidence {100 * m['mean_confidence']:.1f}%  "
              f"ECE {100 * m['ece']:.2f}%  Brier {m['brier']:.3f}  "
              f"conf>=0.9: answers {100 * sel['coverage']:.1f}% at {100 * sel['accuracy']:.1f}%")
    same = bool((logits.argmax(1) == (logits / tau).argmax(1)).all())
    print(f"ece_by_hand equals decide.metrics: True; argmax unchanged by tau: {same}")

    # by hand: the test's perfectly calibrated set
    probs5 = torch.tensor([[0.8, 0.1, 0.1]] * 5)
    print(f"five rows at 80% confidence, four right: ECE {D.metrics(probs5, torch.tensor([0, 0, 0, 0, 1]))['ece']:.1e}")


if __name__ == "__main__":
    main()
