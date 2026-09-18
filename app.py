"""
Gradio front end for AnuLM: a laptop demo or a Hugging Face Space.

    pip install gradio huggingface_hub
    python app.py --ckpt ckpt_coder_sft.pt                  # a .pt or an exported folder
    ANULM_REPO=<hf-owner>/AnuLM-Coder-400M python app.py    # download the weights from the Hub first

On a Space, set the variable ANULM_REPO to the model repo and add
`gradio` and `huggingface_hub` to requirements.txt. The page shows the
modes the loaded checkpoint supports, exactly like serve.py's web page:
the coder offers "write a function" and "continue the code", the
translator "translate", the question-answering model "answer a question".
"""

from __future__ import annotations

import argparse
import os

import gradio as gr
import torch

from serve import Engine

MODE_LABELS = {
    "question": "answer a question",
    "translate": "translate",
    "continue": "continue the text",
}
CODER_LABELS = {"question": "write a function", "continue": "continue the code"}

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


def resolve_checkpoint(path: str | None) -> str:
    repo = os.environ.get("ANULM_REPO")
    if repo:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo)
    return path or "ckpt_coder_sft.pt"


def kind_of(info: dict) -> str:
    if info.get("coder"):
        return "coder"
    if info.get("translate"):
        return "translate"
    if info.get("qa_template"):
        return "qa"
    return "base"


def build(engine: Engine) -> gr.Blocks:
    info = engine.info
    kind = kind_of(info)
    modes = {"coder": ["question", "continue"], "translate": ["translate", "continue"],
             "qa": ["question", "continue"], "base": ["continue"]}[kind]
    labels = {m: (CODER_LABELS.get(m) if kind == "coder" else None) or MODE_LABELS[m] for m in modes}
    inv = {v: k for k, v in labels.items()}
    default_temp = 0.2 if kind == "coder" else 0.3 if kind in ("qa", "translate") else 0.8

    def run(prompt, mode_label, max_tokens, temperature, greedy):
        mode = inv[mode_label]
        r = engine.generate(prompt, int(max_tokens), float(temperature), 1 if greedy else 50, 1337,
                            mode=mode, repetition_penalty=1.3 if (kind == "qa" and mode == "question") else 1.0)
        meta = f"{r['tokens']} tokens in {r['seconds']}s, mode {r['mode']}" + (", stopped at end of answer" if r["stopped_at_eos"] else "")
        return r["completion"], meta

    subtitle = {"coder": "writes Python", "translate": "translates English ↔ Hindi",
                "qa": "answers questions in Hindi, English and Python", "base": "continues text"}[kind]
    header = (f"**{info['ckpt']}** · {info['params_total_m']}M params ({info['params_active_m']}M active) · "
              f"{info['attn'].upper()} × {info['layers']} layers · {info['experts']} experts top-{info['top_k']} · "
              f"context {info['block_size']} · {info['device']}")

    with gr.Blocks(title="AnuLM") as demo:
        gr.Markdown(f"# AnuLM · a small Sarvam-shaped MoE that {subtitle}\n{header}")
        with gr.Row():
            mode = gr.Radio(list(labels.values()), value=list(labels.values())[0], label="mode")
            greedy = gr.Checkbox(value=(kind == "coder"), label="greedy (as evaluated)")
        prompt = gr.Textbox(lines=5, label="prompt")
        with gr.Row():
            max_tokens = gr.Slider(10, 400, value=200 if kind == "coder" else 120, step=10, label="tokens")
            temperature = gr.Slider(0.1, 1.5, value=default_temp, step=0.05, label="temperature")
        go = gr.Button("Generate", variant="primary")
        out = gr.Code(language="python", label="output") if kind == "coder" else gr.Textbox(lines=8, label="output")
        meta = gr.Markdown()
        go.click(run, [prompt, mode, max_tokens, temperature, greedy], [out, meta])
        prompt.submit(run, [prompt, mode, max_tokens, temperature, greedy], [out, meta])
        ex = [[p, labels[m]] for m in modes for p in EXAMPLES.get(kind, {}).get(m, [])]
        if ex:
            gr.Examples(ex, inputs=[prompt, mode], label="examples")
        gr.Markdown(
            "Independent academic project, not affiliated with Sarvam AI, BharatGen, AI4Bharat or the Government of India. "
            "Every checkpoint invents things: run the code, check the facts. Licences and data: see the model card.")
    return demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="a .pt or an exported folder; ANULM_REPO overrides it")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    p.add_argument("--share", action="store_true")
    args = p.parse_args()
    engine = Engine(resolve_checkpoint(args.ckpt), args.device)
    build(engine).launch(server_name="0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1",
                         server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
