"""Draws Chapter 15's figures; the evidence ones from the lab's results.json.

    python lab.py && python make_figures.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "figures"))
from fig import Fig, P  # noqa: E402

OUT = HERE / "figures"
RES = HERE / "results.json"
R = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else None


def fig_folder():
    """15.1: a training checkpoint against a release folder."""
    f = Fig(720, 330)
    f.text(24, 34, "training checkpoint (.pt)", 13, P.ink, weight=700)
    items = [("weights, float32", P.mist), ("buffers: expert biases", P.mist), ("projector, head, adapters", P.mist),
             ("optimizer moments, step", P.stone_soft), ("Python objects (pickle)", P.rose_soft)]
    for i, (label, fill) in enumerate(items):
        f.rect(24, 48 + i * 36, 230, 30, fill, P.stone, r=4, sw=1)
        f.text(36, 68 + i * 36, label, 12, P.ink)
    f.text(24, 246, "loading a pickle runs code: only", 11, P.rose)
    f.text(24, 262, "open one you made yourself", 11, P.rose)
    f.arrow(262, 128, 330, 128, P.stone)
    f.text(296, 118, "export", 11, P.stone, "middle")
    f.text(340, 34, "release folder", 13, P.ink, weight=700)
    files = [("model.safetensors", "big matrices in bfloat16, small ones as they were", P.teal_soft, P.teal),
             ("extras.safetensors", "projector, heads, adapters", P.teal_soft, P.teal),
             ("config.json", "sizes, step, loss, prompt templates, extras", P.indigo_soft, P.indigo),
             ("tokenizer", "the one it was trained with", P.indigo_soft, P.indigo),
             ("README.md", "the model card: licence, data, measured results", P.saffron_soft, P.saffron)]
    for i, (name, what, fill, stroke) in enumerate(files):
        y = 48 + i * 44
        f.rect(340, y, 360, 38, fill, stroke, r=5, sw=1.2)
        f.text(352, y + 16, name, 12, P.ink, weight=700, mono=True)
        f.text(352, y + 31, what, 11, P.stone)
    f.text(340, 286, "no optimizer, no pickle, about half the size", 12, P.teal, weight=600)
    f.caption("Figure 15.1  What export_hf.py writes. The folder holds data only, and loads anywhere a .pt does.")
    f.save(OUT / "fig15_1_folder.svg")


def fig_precision():
    """15.2: three precision rules on the same model."""
    e = R["exports"]
    names = list(e)
    f = Fig(720, 260)
    cols = [("size", lambda d: f"{d['bytes'] / 1e6:.2f} MB"), ("in bfloat16", lambda d: f"{d['bf16_numbers']:,}"),
            ("experts changed", lambda d: f"{d['routing_changed']:.2%}"),
            ("max prob. change", lambda d: f"{d['max_prob_change']:.3f}"),
            ("sums right", lambda d: f"{d['accuracy']:.1%}")]
    xs = [200, 290, 420, 545, 650]
    for x, (head, _) in zip(xs, cols):
        f.text(x, 40, head, 11, P.stone, "middle", 600)
    for i, n in enumerate(names):
        y = 70 + i * 40
        f.rect(16, y - 20, 688, 32, P.teal_soft if i == 0 else P.paper, P.mist, r=4, sw=1)
        f.text(28, y + 1, n, 12, P.ink, weight=700)
        for x, (_, fmt) in zip(xs, cols):
            f.text(x, y + 1, fmt(e[n]), 12, P.ink, "middle")
    f.text(16, 210, f"The .pt file: {R['pt_bytes'] / 1e6:.2f} MB. Chapter 9's model got {R['reference_accuracy']:.1%} of "
                    "200 held-out sums right before export.", 12, P.ink)
    f.caption("Figure 15.2  The same exporter with three precision rules, on Chapter 9's sums model. Measured in the lab.")
    f.save(OUT / "fig15_2_precision.svg")


def fig_gates():
    """15.3: the first AnuLM-Omni run's release gates."""
    o = R["omni"]
    f = Fig(720, 320)
    f.text(16, 32, "the first AnuLM-Omni run, 30 October 2026: one model, seven tasks", 13, P.ink, weight=700)
    rows = list(o["gates"].items())
    labels = {"wer_en": "English speech, WER %", "wer_hi_mean": "Hindi speech, mean WER %",
              "wer_hi_fleurs": "Hindi speech, FLEURS WER %", "chat_heldout_loss": "chat, held-out loss",
              "read_squad_en": "reading, SQuAD F1", "intent_banking77": "intents, BANKING77 %",
              "vqa_acc": "image questions, accuracy %"}
    for i, (k, ok) in enumerate(rows):
        y = 50 + i * 30
        f.rect(16, y, 688, 26, P.teal_soft if ok else P.rose_soft, "none", r=4)
        f.text(28, y + 18, labels.get(k, k), 12, P.ink)
        f.text(400, y + 18, str(o["results"].get(k, "-")), 12, P.ink, "end", 700)
        f.text(560, y + 18, "pass" if ok else "fail", 12, P.teal if ok else P.rose, "middle", 700)
    y = 50 + len(rows) * 30 + 26
    f.text(16, y, "Not published: four of seven gates failed, although the run's own score had improved by 41%.",
           12, P.rose, weight=600)
    f.caption("Figure 15.3  experiments/omni_one.py's gates on the run's measurements (logs/omni_one/REPORT.md). "
              "--publish uploads only if every gate passes.")
    f.save(OUT / "fig15_3_gates.svg")


if __name__ == "__main__":
    fig_folder()
    if R:
        fig_precision()
        fig_gates()
    print("figures written to", OUT)
