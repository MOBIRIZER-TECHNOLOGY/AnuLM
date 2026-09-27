"""
AnuLM Omni: listen, look and talk on one page.

    pip install gradio faster-whisper piper-tts timm pillow
    python app_omni.py                       # everything on the GPU
    python app_omni.py --host 0.0.0.0        # reachable from a phone on the same wifi

Three tabs, each using this project's own trained models:

    Listen   speech -> AnuLM speech recognition (English or Hindi)
                    -> the question-answering model -> Piper voice
    Look     image  -> AnuLM captioner, or image + question -> AnuLM VQA
    Chat     text   -> the question-answering model

Only the voice (Piper) and the frozen encoders underneath (Whisper's encoder
for audio, SigLIP for images) are other people's models. The words heard, the
descriptions seen and the answers given all come from the 400M backbone this
repository trains. docs/SPEECH.md has the numbers behind every tab: WER 8.2%
English / 48.4% Hindi on held-out speech, 40.9% 4-way on A-OKVQA.

Models load on first use and stay resident. All of them together (three
backbones for hearing/seeing, one for answering, plus the encoders) fit a
16 GB card; use --device cpu to try it without one.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import gradio as gr
import torch

HERE = Path(__file__).parent

CKPTS = {
    "asr_en": "ckpt_asr_ls100_e3.pt",
    "asr_hi": "ckpt_asr_hi_mix2.pt",
    "caption": "ckpt_caption_f30k_e2.pt",
    "vqa": "ckpt_vqa_f30k.pt",
}
BRAIN = "release/AnuLM-Hindi-QA-400M"
BRAIN_HUB = "toonist/AnuLM-Hindi-QA-400M"


class Models:
    """Lazy loader: nothing touches the GPU until a tab first needs it."""

    def __init__(self, device: str, brain: str):
        self.device = device
        self.brain_source = brain
        self._m: dict[str, object] = {}

    def _get(self, key, make):
        if key not in self._m:
            t0 = time.time()
            self._m[key] = make()
            print(f"loaded {key} in {time.time() - t0:.1f}s", flush=True)
        return self._m[key]

    def ear(self, lang: str):
        from voice import OwnASR
        key = "asr_hi" if lang == "hi" else "asr_en"
        return self._get(key, lambda: OwnASR(CKPTS[key], self.device))

    def eye(self, task: str):
        from vision_encoder import load_trained
        return self._get(task, lambda: load_trained(CKPTS[task], self.device))

    def brain(self):
        from voice import load_engine
        src = self.brain_source
        if not os.path.exists(src) and src == BRAIN:
            src = BRAIN_HUB
        return self._get("brain", lambda: load_engine(src, self.device))

    def mouth(self):
        from voice import TTS
        return self._get("tts", TTS)


def answer_text(models: Models, text: str, max_tokens: int = 120) -> tuple[str, float]:
    from voice import speakable
    eng = models.brain()
    out = eng.generate(text, max_tokens, 0.3, 40, 0, mode="question", repetition_penalty=1.15)
    return speakable(out["completion"]), out["seconds"]


def build(models: Models) -> gr.Blocks:
    with gr.Blocks(title="AnuLM Omni") as demo:
        gr.Markdown(
            "# AnuLM Omni\n"
            "A 400M sparse mixture-of-experts model, trained from scratch on one consumer "
            "GPU, that **listens**, **looks** and **answers** in English and Hindi. "
            "The hearing, seeing and answering are all this project's own models.")

        with gr.Tab("Listen & reply"):
            with gr.Row():
                with gr.Column():
                    audio = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                     label="speak or upload a clip")
                    lang = gr.Radio([("English", "en"), ("हिन्दी", "hi")], value="en",
                                    label="language you speak")
                    speak_back = gr.Checkbox(value=True, label="answer the question out loud")
                    go_listen = gr.Button("Listen", variant="primary")
                with gr.Column():
                    heard = gr.Textbox(label="AnuLM heard", lines=2)
                    said = gr.Textbox(label="AnuLM answered", lines=4)
                    voice_out = gr.Audio(label="spoken answer", autoplay=True)
                    listen_meta = gr.Markdown()

            def do_listen(a, lg, speak):
                if a is None:
                    return "", "", None, "*record or upload a clip first*"
                t0 = time.time()
                h = models.ear(lg).hear(a)
                t_hear = time.time() - t0
                if not h["text"]:
                    return "", "", None, f"heard nothing ({t_hear:.2f}s)"
                if not speak:
                    return h["text"], "", None, f"hear {t_hear:.2f}s"
                reply, t_think = answer_text(models, h["text"])
                t1 = time.time()
                rate, samples = models.mouth().say(reply) if reply else (22050, None)
                t_say = time.time() - t1
                return (h["text"], reply, (rate, samples) if samples is not None else None,
                        f"hear {t_hear:.2f}s · think {t_think:.2f}s · speak {t_say:.2f}s")

            go_listen.click(do_listen, [audio, lang, speak_back],
                            [heard, said, voice_out, listen_meta])
            gr.Markdown("*Speech recognition: WER 8.2% English (LibriSpeech), 48.4% Hindi "
                        "(FLEURS, unseen speakers). It struggles most with names on very "
                        "short clips.*")

        with gr.Tab("Look"):
            with gr.Row():
                with gr.Column():
                    img = gr.Image(type="filepath", label="image", height=320)
                    question = gr.Textbox(label="question (leave empty for a caption)",
                                          placeholder="What is the man holding?")
                    spoken_q = gr.Audio(sources=["microphone", "upload"], type="numpy",
                                        label="...or ask it out loud (English)")
                    go_look = gr.Button("Look", variant="primary")
                with gr.Column():
                    seen = gr.Textbox(label="AnuLM sees", lines=3)
                    look_voice = gr.Audio(label="spoken", autoplay=True)
                    look_meta = gr.Markdown()

            def do_look(image, q, sq):
                from vision_encoder import answer, caption
                if image is None:
                    return q, "", None, "*drop an image in first*"
                t0 = time.time()
                parts = []
                if sq is not None:
                    q = models.ear("en").hear(sq)["text"]
                    parts.append(f"heard {time.time() - t0:.2f}s")
                t1 = time.time()
                if q and q.strip():
                    vl, tower, tok = models.eye("vqa")
                    out = answer(vl, tower, tok, image, q.strip(), device=models.device)
                    parts.append(f"answer (A-OKVQA-trained) {time.time() - t1:.2f}s")
                else:
                    vl, tower, tok = models.eye("caption")
                    out = caption(vl, tower, tok, image, device=models.device)
                    parts.append(f"caption (Flickr30k-trained) {time.time() - t1:.2f}s")
                out = out or "(nothing)"
                spoken = models.mouth().say(out) if out != "(nothing)" else None
                return q, out, spoken, " · ".join(parts)

            go_look.click(do_look, [img, question, spoken_q], [question, seen, look_voice, look_meta])
            examples = sorted((HERE / "data" / "flickr30k_img").glob("10*.jpg"))[:6] \
                if (HERE / "data" / "flickr30k_img").exists() else []
            if examples:
                gr.Examples([[str(e), ""] for e in examples], [img, question],
                            label="held-out Flickr30k test images")

        with gr.Tab("Chat"):
            q_in = gr.Textbox(label="ask in English, Hindi or about Python", lines=2,
                              placeholder="भारत की राजधानी क्या है?")
            go_chat = gr.Button("Ask", variant="primary")
            a_out = gr.Textbox(label="AnuLM", lines=6)
            chat_meta = gr.Markdown()

            def do_chat(q):
                if not q or not q.strip():
                    return "", ""
                reply, t = answer_text(models, q.strip(), max_tokens=200)
                return reply, f"{t:.2f}s · {models.brain_source}"

            go_chat.click(do_chat, [q_in], [a_out, chat_meta])
            gr.Markdown("*The question-answering model was tuned on questions mined from "
                        "Wikipedia and Python docstrings. Open-ended chat is not trained yet; "
                        "see TODO.md item 4.*")
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="AnuLM Omni: listen, look and talk.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--brain", default=BRAIN, help="the text model that answers")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7863)
    p.add_argument("--share", action="store_true", help="public gradio link")
    args = p.parse_args()
    missing = [c for c in CKPTS.values() if not (HERE / c).exists()]
    if missing:
        print(f"note: {', '.join(missing)} not found; those tabs will fail until trained "
              f"(docs/SPEECH.md has the commands)")
    build(Models(args.device, args.brain)).launch(server_name=args.host, server_port=args.port,
                                                  share=args.share)


if __name__ == "__main__":
    main()
