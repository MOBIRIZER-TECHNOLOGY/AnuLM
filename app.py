"""
Gradio front end for AnuLM: a laptop demo, a Colab cell, or a Hugging Face Space.

    pip install gradio huggingface_hub
    python app.py                                        # the picker, nothing loaded yet
    python app.py --ckpt ckpt_coder_sft.pt               # a .pt or an exported folder
    python app.py --host 0.0.0.0                         # reachable from your phone on the same wifi
    ANULM_REPO=toonist/AnuLM-Coder-400M python app.py    # download the weights from the Hub first

The page holds all four released checkpoints in a dropdown and loads one on
demand, because they do different things and the interesting thing about this
project is the comparison: the same 398M network writes Python, translates,
answers questions or continues prose depending only on what it was tuned on.

**One at a time.** Each model is 1.6 GB in float32 and switching evicts the
previous one, which keeps this inside Colab's free tier. Loading takes a
moment the first time (the weights download) and a second or two after that.

The modes, the examples and the defaults all follow the loaded checkpoint,
exactly as serve.py's page does: the coder offers "write a function" and
"continue the code", the translator "translate", the question answerer
"answer a question", the base model only "continue the text".

On a Space, set the variable ANULM_REPO to preselect one, and add `gradio`
and `huggingface_hub` to requirements.txt.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import gradio as gr
import torch

from serve import Engine

MODE_LABELS = {
    "question": "answer a question",
    "translate": "translate",
    "continue": "continue the text",
}
CODER_LABELS = {"question": "write a function", "continue": "continue the code"}

# label -> Hub repo. The order is the order of the README's table.
MODELS = {
    "Python coder — writes small functions (MBPP 12.5%)": "toonist/AnuLM-Coder-400M",
    "Translator — English ↔ Hindi (chrF 41.5 / 43.4)": "toonist/AnuLM-Translate-400M",
    "Question answerer — Hindi, English, Python": "toonist/AnuLM-Hindi-QA-400M",
    "Base — continues text in three languages": "toonist/AnuLM-Base-400M",
    "Base 2K — the same, at a 2,048-token context": "toonist/AnuLM-Base-2K-400M",
}

MODES = {"coder": ["question", "continue"], "translate": ["translate", "continue"],
         "qa": ["question", "continue"], "base": ["continue"]}

EXAMPLES = {
    "coder": {
        "question": ["Write a Python function to check whether a number is prime.",
                     "Write a function that removes duplicate items from a list while keeping their order.",
                     "Write a Python function to count the vowels in a string."],
        "continue": ['def is_palindrome(s: str) -> bool:\n    """Return True if s reads the same forwards and backwards."""\n',
                     "# Compute the factorial of n\ndef factorial(n):\n"],
    },
    "translate": {
        "translate": ["Where is the nearest hospital?", "Please send me the report by Friday.",
                      "मुझे कल दिल्ली जाना है।"],
        "continue": ["भारत की राजधानी"],
    },
    "qa": {
        "question": ["काशी क्या है?", "योग क्या है?", "What is the Ganges?"],
        "continue": ["मुंबई\n\n", "हिन्दी साहित्य\n\n"],
    },
    "base": {
        "continue": ["मुंबई\n\n", "हिन्दी साहित्य\n\n", "def fibonacci(n):\n"],
    },
}
SUBTITLE = {"coder": "writes Python", "translate": "translates English ↔ Hindi",
            "qa": "answers questions in Hindi, English and Python",
            "base": "continues text"}


def kind_of(info: dict) -> str:
    if info.get("coder"):
        return "coder"
    if info.get("translate"):
        return "translate"
    if info.get("qa_template"):
        return "qa"
    return "base"


def labels_for(kind: str) -> dict[str, str]:
    return {m: (CODER_LABELS.get(m) if kind == "coder" else None) or MODE_LABELS[m]
            for m in MODES[kind]}


def examples_for(kind: str) -> list[list[str]]:
    labels = labels_for(kind)
    return [[p, labels[m]] for m in MODES[kind] for p in EXAMPLES.get(kind, {}).get(m, [])]


def default_temp(kind: str) -> float:
    return 0.2 if kind == "coder" else 0.3 if kind in ("qa", "translate") else 0.8


class Loader:
    """Holds at most one Engine. Switching evicts the previous one, because
    four of these in float32 is 6.4 GB and Colab's free tier has 12."""

    def __init__(self, device: str):
        self.device = device
        self.engine: Engine | None = None
        self.source: str | None = None

    def load(self, source: str) -> Engine:
        if self.engine is not None and self.source == source:
            return self.engine
        self.engine = None                       # drop the old weights first
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        path = source
        if "/" in source and not os.path.exists(source):
            from huggingface_hub import snapshot_download
            path = snapshot_download(source)
        self.engine = Engine(path, self.device)
        # snapshot_download returns .../snapshots/<commit sha>, so Engine takes
        # the sha as the checkpoint's name and the page header showed a hash
        # for every model loaded from the Hub. Say what was asked for instead.
        self.engine.info["ckpt"] = source
        self.source = source
        return self.engine


def header_for(engine: Engine) -> str:
    i = engine.info
    return (f"**{i['ckpt']}** · {i['params_total_m']}M params ({i['params_active_m']}M active) · "
            f"{i['attn'].upper()} × {i['layers']} layers · {i['experts']} experts top-{i['top_k']} · "
            f"context {i['block_size']} · {i['device']}")


def build(loader: Loader, initial: str | None = None) -> gr.Blocks:
    with gr.Blocks(title="AnuLM") as demo:
        gr.Markdown("# AnuLM · a small Sarvam-shaped MoE\n"
                    "398M parameters, 174M active per token, trained from scratch on one "
                    "consumer GPU for Hindi, English and Python. Pick a checkpoint; they are "
                    "the same network tuned on different data.")
        with gr.Row():
            picker = gr.Dropdown(list(MODELS), label="checkpoint",
                                 value=initial or next(iter(MODELS)), scale=3)
            go_load = gr.Button("Load", variant="secondary", scale=1)
        status = gr.Markdown("Nothing loaded yet — press **Load**. "
                             "The first load downloads about 0.8 GB.")

        with gr.Row():
            mode = gr.Radio(list(labels_for("coder").values()),
                            value=CODER_LABELS["question"], label="mode")
            greedy = gr.Checkbox(value=True, label="greedy (as evaluated)")
        prompt = gr.Textbox(lines=5, label="prompt")
        ex = gr.Examples(examples_for("coder"), inputs=[prompt, mode], label="examples")
        with gr.Row():
            max_tokens = gr.Slider(10, 400, value=200, step=10, label="tokens")
            temperature = gr.Slider(0.1, 1.5, value=0.2, step=0.05, label="temperature")
            # Default 1.0 -- no change -- because every number in the model
            # cards was measured without it. Raise it when a base model starts
            # repeating itself, which at 398M it will.
            rep_penalty = gr.Slider(1.0, 1.6, value=1.0, step=0.05,
                                    label="repetition penalty")
        run_btn = gr.Button("Generate", variant="primary")
        out = gr.Textbox(lines=10, label="output")
        meta = gr.Markdown()

        def do_load(label):
            engine = loader.load(MODELS[label])
            kind = kind_of(engine.info)
            labels = labels_for(kind)
            first = list(labels.values())[0]
            return (gr.update(choices=list(labels.values()), value=first),
                    gr.update(value=default_temp(kind)),
                    gr.update(value=(kind == "coder")),
                    gr.update(value=200 if kind == "coder" else 120),
                    f"Loaded — {SUBTITLE[kind]}.\n\n{header_for(engine)}",
                    gr.update(value=""))

        outputs = [mode, temperature, greedy, max_tokens, status, out]
        go_load.click(do_load, [picker], outputs)
        picker.change(lambda: gr.update(value="Press **Load** to switch."), None, [status])

        def run(text, mode_label, n_tokens, temp, is_greedy, penalty):
            if loader.engine is None:
                return "", "Nothing loaded — press **Load** first."
            engine = loader.engine
            kind = kind_of(engine.info)
            inv = {v: k for k, v in labels_for(kind).items()}
            m = inv.get(mode_label, "continue")
            # The question mode's 1.3 is what eval_qa.py and serve.py use, so
            # the page reproduces the cards; anything the user sets wins.
            default_pen = 1.3 if (kind == "qa" and m == "question") else 1.0
            r = engine.generate(text, int(n_tokens), float(temp), 1 if is_greedy else 50, 1337,
                                mode=m,
                                repetition_penalty=max(float(penalty), default_pen))
            note = f"{r['tokens']} tokens in {r['seconds']}s, mode {r['mode']}"
            if r["stopped_at_eos"]:
                note += ", stopped at end of answer"
            return r["completion"], note

        inputs = [prompt, mode, max_tokens, temperature, greedy, rep_penalty]
        run_btn.click(run, inputs, [out, meta])
        prompt.submit(run, inputs, [out, meta])
        gr.Markdown(
            "These are 398M base and fine-tuned models: they repeat themselves, especially "
            "when continuing text. Raise the **repetition penalty** to about 1.2 when that "
            "happens — it is left at 1.0 because every number in the model cards was "
            "measured without it.\n\n"
            "Independent academic project, not affiliated with Sarvam AI, BharatGen, AI4Bharat "
            "or the Government of India. Every checkpoint invents things: run the code, check "
            "the facts. Licences and data are on each model card.")

        if initial:
            demo.load(lambda: do_load(initial), None, outputs)
    return demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="a .pt or an exported folder to preload")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    p.add_argument("--host", default=None,
                   help="0.0.0.0 to reach it from other devices on the same network; "
                        "the default binds to this machine only")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    loader = Loader(args.device)
    initial = None
    repo = os.environ.get("ANULM_REPO")
    # Both --ckpt and ANULM_REPO become an entry in the picker and are then
    # loaded by the same path as a click, so the page always describes what is
    # actually loaded rather than the default.
    if args.ckpt:
        label = f"{Path(args.ckpt).name} (local)"
        MODELS[label] = args.ckpt
        initial = label
    elif repo:
        initial = next((k for k, v in MODELS.items() if v == repo), None)
        if initial is None:            # a repo not in the list still works
            label = f"{repo} (from ANULM_REPO)"
            MODELS[label] = repo
            initial = label

    host = args.host or ("0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1")
    if host == "0.0.0.0":
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:                                # the address other devices should use
            s.connect(("8.8.8.8", 80))
            print(f"\n  on this network:  http://{s.getsockname()[0]}:{args.port}\n")
        except OSError:
            pass
        finally:
            s.close()
    build(loader, initial).launch(server_name=host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
