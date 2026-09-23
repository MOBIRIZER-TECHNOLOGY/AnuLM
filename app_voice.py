"""
Talk to AnuLM: the cascade demo, in a browser.

    pip install gradio faster-whisper piper-tts
    python app_voice.py                                  # pick a checkpoint on the page
    python app_voice.py --ckpt toonist/AnuLM-Translate-400M
    python app_voice.py --host 0.0.0.0                   # reachable from your phone

Press record, talk, get speech back:

    microphone -> faster-whisper -> AnuLM -> Piper -> your speakers

Everything runs on this machine. `voice.py` holds the three stages and its
docstring explains the pieces; this file is only the page around them, in the
same shape as `app.py`'s (the same checkpoint picker, one model resident at a
time, the same eviction so it fits Colab's free tier).

**This is the baseline, not the goal.** It is a cascade: three separate models
in a row, and the text in the middle is the only part this project trained.
`audio_codec.py` is the other road -- audio as tokens the model itself emits --
and the honest way to judge that work later is against the latency and the
word error rate this page reports now.

The stage timings under each reply are the point of the page as much as the
audio is: on an RTX 5070 Ti the 400M model is the cheapest of the three.
"""

from __future__ import annotations

import argparse
import gc
import os

import gradio as gr
import numpy as np
import torch

from app import MODELS, header_for
from voice import ASR, TTS, Voice, mode_for

MODE_LABELS = {"question": "answer it", "translate": "translate it",
               "continue": "carry on from it"}


class VoiceLoader:
    """One checkpoint at a time, like app.py's Loader, but the ear and the
    mouth are loaded once and shared -- they do not change with the model."""

    def __init__(self, device: str, whisper: str):
        self.device = device
        self.asr = ASR(whisper)
        self.tts = TTS()
        self.voice: Voice | None = None
        self.source: str | None = None

    def load(self, source: str) -> Voice:
        if self.voice is not None and self.source == source:
            return self.voice
        from voice import load_engine
        self.voice = None                        # drop the old weights first
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        self.voice = Voice(load_engine(source, self.device), self.asr, self.tts)
        self.source = source
        return self.voice


def build(loader: VoiceLoader, initial: str | None = None) -> gr.Blocks:
    with gr.Blocks(title="AnuLM — voice") as demo:
        gr.Markdown("## AnuLM — talk to it\n"
                    "Record, and it answers out loud. Whisper hears, AnuLM thinks, "
                    "Piper speaks; all three run locally.")
        with gr.Row():
            picker = gr.Dropdown(list(MODELS), label="checkpoint",
                                 value=initial or list(MODELS)[2], scale=3)
            go_load = gr.Button("Load", variant="secondary", scale=1)
        header = gr.Markdown("*nothing loaded yet*")

        with gr.Row():
            with gr.Column():
                mic = gr.Audio(sources=["microphone", "upload"], type="numpy",
                               label="say something")
                with gr.Row():
                    max_tokens = gr.Slider(20, 300, value=120, step=10, label="tokens")
                    temperature = gr.Slider(0.1, 1.5, value=0.3, step=0.05, label="temperature")
                run = gr.Button("Answer", variant="primary")
            with gr.Column():
                heard = gr.Textbox(label="heard", lines=2)
                said = gr.Textbox(label="said", lines=6)
                reply = gr.Audio(label="reply", autoplay=True)
                timing = gr.Markdown()

        def do_load(source):
            v = loader.load(source)
            mode = mode_for(v.engine.info)
            return (f"{header_for(v.engine)}\n\n"
                    f"*heard speech becomes a prompt, and the model will "
                    f"{MODE_LABELS[mode]}.*")

        def do_run(audio, source, toks, temp):
            if audio is None:
                return "", "", None, "*record something first*"
            v = loader.load(source)
            rate, data = audio
            r = v.reply((rate, data), int(toks), float(temp))
            if not r.heard:
                return "", "", None, "*heard only silence*"
            t = r.timings
            total = sum(t.values())
            bar = (f"**{total:.1f}s** end to end — "
                   f"hear {t['hear']}s · think {t['think']}s · speak {t['say']}s"
                   f"  ·  *{r.mode}*, {r.language}")
            out = (r.rate, r.audio) if r.audio.size else None
            return r.heard, r.said, out, bar

        go_load.click(do_load, picker, header)
        run.click(do_run, [mic, picker, max_tokens, temperature],
                  [heard, said, reply, timing])
        demo.load(do_load, picker, header)
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="Talk to AnuLM in a browser.")
    p.add_argument("--ckpt", help="preselect a checkpoint (label or Hub repo id)")
    p.add_argument("--whisper", default="small", help="tiny/base/small/medium/large-v3")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7861)
    p.add_argument("--share", action="store_true", help="public gradio link (Colab)")
    args = p.parse_args()

    initial = args.ckpt or os.environ.get("ANULM_REPO")
    if initial and initial not in MODELS:            # a repo id, not a label
        for label, repo in MODELS.items():
            if repo == initial:
                initial = label
                break
    print(f"loading whisper-{args.whisper} and the Piper voices on {args.device}...")
    loader = VoiceLoader(args.device, args.whisper)
    build(loader, initial).launch(server_name=args.host, server_port=args.port,
                                  share=args.share)


if __name__ == "__main__":
    main()
