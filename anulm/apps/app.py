"""
Gradio front end for AnuLM: a laptop demo, a Colab cell, or a Hugging Face Space.

    pip install gradio huggingface_hub
    python -m anulm app                                        # the picker, nothing loaded yet
    python -m anulm app --ckpt ckpt_coder_sft.pt               # a .pt or an exported folder
    python -m anulm app --host 0.0.0.0                         # reachable from your phone on the same wifi
    ANULM_REPO=toonist/AnuLM-Coder-400M python -m anulm app    # download the weights from the Hub first

The page holds all five released checkpoints in a dropdown and loads one on
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
import json
import os
from pathlib import Path

import gradio as gr
import torch

from anulm.inference.serve import Engine

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


def load_samples() -> dict:
    """Real replies from tools_sweep.py, written by tools_samples.py.

    The page shows these before anything is loaded, so a visitor sees what
    the models produce without first waiting for 0.8 GB of weights. They are
    selected -- nearest the median repetition among replies that stopped on
    their own -- and the aggregate numbers for the whole sweep are shown
    beside them, because a page of hand-picked best cases is not evidence.
    """
    f = Path(__file__).resolve().parents[2] / "docs" / "samples.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def samples_markdown(data: dict, per_segment: int = 4) -> str:
    """Markdown for the samples accordion: for each segment (code, translate,
    continue) its title, blurb and sweep statistics, then its first
    `per_segment` samples with prompt and completion truncated."""
    out = []
    for seg in ("code", "translate", "continue"):
        d = data.get(seg)
        if not d:
            continue
        st = d["stats"]
        out.append(
            "### {}\n\n{}\n\n*Over {:,} held-out prompts: {} errors, {} empty "
            "replies, {}% stopped on their own, {}% repeated 4-grams on average, "
            "{} ms per reply.*\n".format(
                d["title"], d["blurb"], st["prompts"], st["errors"], st["empty"],
                st["stopped_pct"], st["rep4_mean_pct"], st["ms_mean"]))
        for sample in d["samples"][:per_segment]:
            out.append("**{}**\n\n```\n{}\n```\n".format(
                sample["prompt"].strip()[:160], sample["completion"].strip()[:420]))
    return "\n".join(out)


def kind_of(info: dict) -> str:
    """The page kind of a loaded checkpoint -- coder, translate, qa or base -- read off
    Engine's info flags."""
    if info.get("coder"):
        return "coder"
    if info.get("translate"):
        return "translate"
    if info.get("qa_template"):
        return "qa"
    return "base"


def labels_for(kind: str) -> dict[str, str]:
    """Mode name -> label shown on the page, for the modes `kind` offers; the coder has its
    own labels."""
    return {m: (CODER_LABELS.get(m) if kind == "coder" else None) or MODE_LABELS[m]
            for m in MODES[kind]}


def examples_for(kind: str) -> list[list[str]]:
    """The example rows, [prompt, mode label], for a page kind."""
    labels = labels_for(kind)
    return [[p, labels[m]] for m in MODES[kind] for p in EXAMPLES.get(kind, {}).get(m, [])]


def default_penalty(kind: str, mode: str) -> float:
    """The starting repetition penalty for a page kind and mode: 1.3 for the
    question answerer's question mode (what eval_qa.py and serve.py use, so the
    page reproduces the cards), 1.0 -- no penalty -- everywhere else."""
    return 1.3 if (kind == "qa" and mode == "question") else 1.0


def repetition_penalty(kind: str, mode: str, user: float | None) -> float:
    """The penalty a generation runs with: the user's value whenever they have
    set one, the mode's default only when they have not (None)."""
    return default_penalty(kind, mode) if user is None else float(user)


def is_loopback(host: str | None) -> bool:
    """True when `host` only accepts connections from this machine (None is
    Gradio's default, which is loopback)."""
    return host is None or host in ("localhost", "::1") or host.startswith("127.")


def exposure_warning(host: str | None, share: bool = False, auth=None) -> str | None:
    """The start-up warning for a page other machines can reach with no
    password, or None. A warning, not a refusal: a phone on the same wifi is
    what --host 0.0.0.0 is for."""
    if auth or (is_loopback(host) and not share):
        return None
    return (f"warning: {'a public share link' if share else f'listening on {host}'} with no password: "
            "anyone who can reach it can use the model and this machine's GPU.")


def default_temp(kind: str) -> float:
    """The starting temperature for a page kind: 0.2 coder, 0.3 qa and translate, 0.8 base."""
    return 0.2 if kind == "coder" else 0.3 if kind in ("qa", "translate") else 0.8


class Loader:
    """Holds at most one Engine. Switching evicts the previous one, because
    five of these in float32 is 8 GB and Colab's free tier has 12."""

    def __init__(self, device: str):
        self.device = device
        self.engine: Engine | None = None
        self.source: str | None = None

    def load(self, source: str) -> Engine:
        """The Engine for `source` -- a .pt, an exported folder, or a Hub repo id,
        downloaded first -- reusing the loaded one for the same source and evicting
        it otherwise."""
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
    """One line of Markdown describing a loaded engine: name, parameters, attention,
    layers, experts, context and device."""
    i = engine.info
    return (f"**{i['ckpt']}** · {i['params_total_m']}M params ({i['params_active_m']}M active) · "
            f"{i['attn'].upper()} × {i['layers']} layers · {i['experts']} experts top-{i['top_k']} · "
            f"context {i['block_size']} · {i['device']}")


def build(loader: Loader, initial: str | None = None) -> gr.Blocks:
    """The Gradio page: checkpoint picker, mode, prompt, sampling controls and
    output, plus the recorded samples when docs/samples.json exists. `initial`
    is a picker label to load when the page opens."""
    with gr.Blocks(title="AnuLM") as demo:
        gr.Markdown("# AnuLM · a small mixture-of-experts model for Hindi, English and Python\n"
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
                    gr.update(value=""),
                    # The examples belong to the checkpoint too. Without this
                    # the page went on offering "def is_palindrome(...)" after
                    # a switch to the translator: Python prompts for a model
                    # that only translates.
                    gr.update(samples=examples_for(kind)))

        outputs = [mode, temperature, greedy, max_tokens, status, out, ex.dataset]
        go_load.click(do_load, [picker], outputs)
        picker.change(lambda: gr.update(value="Press **Load** to switch."), None, [status])

        # The penalty slider shows the mode's default until the user moves it;
        # from then on their value is used, whatever the mode or checkpoint.
        pen_touched = gr.State(False)
        rep_penalty.input(lambda: True, None, pen_touched)

        def follow_mode(mode_label, touched):
            if touched or loader.engine is None:
                return gr.update()
            kind = kind_of(loader.engine.info)
            m = {v: k for k, v in labels_for(kind).items()}.get(mode_label, "continue")
            return gr.update(value=default_penalty(kind, m))

        mode.change(follow_mode, [mode, pen_touched], rep_penalty)

        def run(text, mode_label, n_tokens, temp, is_greedy, penalty):
            if loader.engine is None:
                return "", "Nothing loaded — press **Load** first."
            if not (text or "").strip():
                # serve.py's HTTP API answers 400 here; the page should say
                # something rather than return a column of spaces.
                return "", "Type a prompt first."
            engine = loader.engine
            kind = kind_of(engine.info)
            inv = {v: k for k, v in labels_for(kind).items()}
            m = inv.get(mode_label, "continue")
            # The slider starts at the mode's default (follow_mode); whatever it
            # says now is the user's choice and wins. It used to be
            # max(penalty, default), so nobody could go below 1.3 in question mode.
            r = engine.generate(text, int(n_tokens), float(temp), 1 if is_greedy else 50, 1337,
                                mode=m,
                                repetition_penalty=repetition_penalty(kind, m, penalty))
            note = f"{r['tokens']} tokens in {r['seconds']}s, mode {r['mode']}"
            if r["stopped_at_eos"]:
                note += ", stopped at end of answer"
            return r["completion"], note

        inputs = [prompt, mode, max_tokens, temperature, greedy, rep_penalty]
        run_btn.click(run, inputs, [out, meta])
        prompt.submit(run, inputs, [out, meta])
        samples = load_samples()
        if samples:
            with gr.Accordion("What these models produced on thousands of held-out "
                              "prompts \u2014 real replies, nothing to load", open=False):
                gr.Markdown(samples_markdown(samples))
        gr.Markdown(
            "These are 398M base and fine-tuned models: they repeat themselves, especially "
            "when continuing text. Raise the **repetition penalty** to about 1.2 when that "
            "happens — it is left at 1.0 because every number in the model cards was "
            "measured without it.\n\n"
            "Independent project, not affiliated with any company or government body. "
            "Every checkpoint invents things: run the code, check "
            "the facts. Licences and data are on each model card.")

        if initial:
            demo.load(lambda: do_load(initial), None, outputs)
    return demo


def main() -> None:
    """Command line: add `--ckpt` or ANULM_REPO to the picker as the initial
    model, print the LAN address when serving on 0.0.0.0, and launch the page."""
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
    warning = exposure_warning(host, args.share)
    if warning and not os.environ.get("SPACE_ID"):       # a Space is public on purpose
        print(warning, flush=True)
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
