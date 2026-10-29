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
import threading
import time
from pathlib import Path

import gradio as gr
import torch

HERE = Path(__file__).parent

# Retrained on base v2 (experiments/omni_v3.py, logs/omni_v3/), each beating the
# model it replaced on the same eval: WER en 8.4 -> 7.1%, hi 51.0 -> 35.0%;
# caption val loss 2.704 -> 2.617; VQA multiple choice 40.9 -> 43.5%.
CKPTS = {
    "asr_en": "ckpt_asr_en_v3.pt",
    "asr_hi": "ckpt_asr_hi_v3.pt",
    "caption": "ckpt_caption_v3.pt",
    "vqa": "ckpt_vqa_v3.pt",
}
# Newest measured models first, older ones as fallbacks (docs/RESULTS.md section 35).
_HERE = Path(__file__).parent
# The chat model and its Decide adapters go together: an adapter attaches only
# to the exact chat model it was trained on (Decide.on refuses any other).
# v3 is the chat model fine-tuned on base v2 (held-out answer loss 3.1445 vs
# 3.2000), with recipe-F adapters (dense, rank 64; logs/post_v2, logs/indic_v4):
# BANKING77 89.03% at 56 ms, MASSIVE-hi 84.46% at 54 ms, vs 86.4% / 82.6% for v2.
if (_HERE / "ckpt_chat_v3.pt").exists() or not (_HERE / "ckpt_chat_v2.pt").exists():   # or a clone: the Hub's
    BRAIN, DECIDE_ADAPTER, DECIDE_ADAPTER_HI = "ckpt_chat_v3.pt", "ckpt_decide_lora_F.pt", "ckpt_decide_lora_hi_F.pt"
    DECIDE_ADAPTER_IN = "ckpt_decide_lora_india_F.pt"
else:
    BRAIN = "ckpt_chat_v2.pt" if (_HERE / "ckpt_chat_v2.pt").exists() else "release/AnuLM-Hindi-QA-400M"
    DECIDE_ADAPTER, DECIDE_ADAPTER_HI = "ckpt_decide_chat_lora.pt", "ckpt_decide_lora_hi.pt"
    DECIDE_ADAPTER_IN = "ckpt_decide_lora_india_F.pt"     # trained on v3 only: absent here, the task hides
READER = "ckpt_rc_pointer_v2.pt" if (_HERE / "ckpt_rc_pointer_v2.pt").exists() else "ckpt_rc_pointer.pt"
# Decide backbones (docs/RESULTS.md section 36). ModernBERT is an open
# pretrained encoder, not this project's model, and is labelled so in the UI.
# AnuLM decides with the chat model itself plus a LoRA adapter (section 37):
# one copy of the weights for both tabs.
DECIDE_OWN = "AnuLM (our own, from scratch; the chat model + adapter)"
DECIDE_MB = "ModernBERT-base (borrowed open model, not from scratch)"
# Two routing tasks, each an adapter on the same chat model (section 38).
TASK_BANK = "Banking support (English, 77 intents)"
TASK_HI = "Hindi assistant commands (MASSIVE, 60 intents)"
# SYNTHETIC: experiments/make_india_banking.py, written for the demo (UPI, IFSC, KYC, ...).
TASK_IN = "Indian banking: UPI, KYC, Aadhaar, EMI (English, Hindi, Hinglish; 20 intents, synthetic data)"
DECIDE_TASKS = {
    TASK_BANK: {DECIDE_OWN: DECIDE_ADAPTER if (_HERE / DECIDE_ADAPTER).exists() else "ckpt_decide.pt",
                DECIDE_MB: "ckpt_decide_modernbert.pt"},
    TASK_HI: {DECIDE_OWN: DECIDE_ADAPTER_HI, DECIDE_MB: "ckpt_decide_mb_hi.pt"},
    TASK_IN: {DECIDE_OWN: DECIDE_ADAPTER_IN, DECIDE_MB: "ckpt_decide_mb_india.pt"},
}
DECIDE_ADAPTERS = {DECIDE_ADAPTER, DECIDE_ADAPTER_HI, DECIDE_ADAPTER_IN}     # --mode lora on BRAIN
BRAIN_HUB = "toonist/AnuLM-Hindi-QA-400M"
# The one model (omni_train.py, docs/OMNI.md): every tab on one decoder. Used
# when this checkout has it, or has none of the separate models above -- a
# fresh clone, which then downloads it from the Hub on first use.
# ANULM_OMNI=<path or Hub id> forces it.
OMNI_HUB = "toonist/AnuLM-Omni-400M"
OMNI = os.environ.get("ANULM_OMNI") or (
    str(HERE / "ckpt_omni.pt") if (HERE / "ckpt_omni.pt").exists() else OMNI_HUB)
USE_OMNI = (bool(os.environ.get("ANULM_OMNI")) or (HERE / "ckpt_omni.pt").exists()
            or not (HERE / "ckpt_chat_v3.pt").exists())
OMNI_TASKS = {TASK_BANK: "banking", TASK_HI: "massive-hi"}     # the intents it was trained on


def have(name: str) -> bool:
    if USE_OMNI:
        return name in set(CKPTS.values()) | {READER} | {DECIDE_TASKS[t][DECIDE_OWN] for t in OMNI_TASKS}
    return (HERE / name).exists()


class Models:
    """Lazy loader: nothing touches the GPU until a tab first needs it."""

    def __init__(self, device: str, brain: str):
        self.device = device
        self.brain_source = brain
        self._m: dict[str, object] = {}
        # Chat and Decide share the chat model's weights; Decide switches its
        # adapters on for a forward pass, so the two never run at once.
        self.shared = threading.Lock()

    def _get(self, key, make):
        if key not in self._m:
            t0 = time.time()
            self._m[key] = make()
            print(f"loaded {key} in {time.time() - t0:.1f}s", flush=True)
        return self._m[key]

    def _half(self, module):
        """bf16 weights on the GPU: five resident backbones in fp32 filled the
        16 GB card to 15,986 MiB, the edge of the silent spill to system RAM."""
        if self.device.startswith("cuda"):
            from model import Rotary
            module.to(torch.bfloat16)
            # .to(dtype) also casts RoPE's inv_freq buffer; in bf16 the angles
            # drift ~0.8 rad by position 200. Rebuild it in fp32.
            for m in module.modules():
                if isinstance(m, Rotary):
                    m.inv_freq = Rotary._build_inv_freq(m.cfg, m.dim).to(self.device)
                    m._cached_len, m._cached_key = 0, None
        return module

    def omni(self):
        from omni import Omni
        return self._get("omni", lambda: Omni(OMNI, self.device))

    def ear(self, lang: str):
        from voice import OwnASR
        if USE_OMNI:                      # one ear for both languages, on the shared decoder
            return self.omni().ear()

        def make():
            asr = OwnASR(str(HERE / CKPTS[key]), self.device)
            self._half(asr.sm)
            return asr
        key = "asr_hi" if lang == "hi" else "asr_en"
        return self._get(key, make)

    def eye(self, task: str):
        from vision_encoder import load_trained

        if USE_OMNI:
            return self.omni().eye()

        def make():
            vl, tower, tok = load_trained(str(HERE / CKPTS[task]), self.device)
            return self._half(vl), tower, tok
        return self._get(task, make)

    def reader(self):
        """Our own from-scratch reader (rc_pointer.py) over dense retrieval of
        Simple English + Hindi Wikipedia (rag.py). None if either is missing."""
        if not have(READER) or not (HERE / "data" / "rag" / "dense_e5s.npy").exists():
            return None
        if USE_OMNI:
            return self.omni().reader()
        from rc_pointer import PointerQA
        return self._get("reader", lambda: PointerQA(str(HERE / READER), self.device))

    def brain(self):
        from voice import load_engine
        if USE_OMNI and self.brain_source == BRAIN:
            return self.omni().engine
        src = self.brain_source
        if not os.path.exists(src) and src == BRAIN:
            src = BRAIN_HUB
        return self._get("brain", lambda: load_engine(src, self.device))

    def mouth(self):
        from voice import TTS
        return self._get("tts", TTS)

    def decider(self, which: str = DECIDE_OWN, task: str = TASK_BANK):
        """Our model's adapters attach to the loaded chat model; each has its
        own hooks, switched on only for its own forward pass."""
        from decide import Decide
        if USE_OMNI:
            return self.omni().decider(OMNI_TASKS[task])
        name = DECIDE_TASKS[task][which]
        path, key = str(HERE / name), f"decide:{task}:{which}"
        if which == DECIDE_OWN and name in DECIDE_ADAPTERS and self.brain_source == BRAIN:
            def make():
                eng = self.brain()
                return Decide.on(eng.model, eng.tok, path, self.device)
            return self._get(key, make)
        return self._get(key, lambda: Decide(path, self.device))


def as_question(heard: str) -> str:
    """Speech recognition returns "what animal is this"; the VQA model was
    trained on A-OKVQA's "What animal is this?" and answered the heard form
    "bee" where the typed one got "dog". Restore the written form."""
    q = heard.strip()
    if not q:
        return q
    q = q[0].upper() + q[1:]
    return q if q[-1] in "?.!।" else q + "?"


def speech_or_none(models: Models, text: str):
    """(rate, samples) for Gradio, or None when nothing could be voiced -- e.g.
    a Hindi reply on a machine with no Hindi voice once Piper is blocked."""
    if not text:
        return None
    rate, samples = models.mouth().say(text)
    return (rate, samples) if len(samples) else None


def chat_text(models: Models, text: str, max_tokens: int = 200) -> tuple[str, float]:
    """The chat model in its own words (no lookup)."""
    from voice import speakable
    t0 = time.time()
    eng = models.brain()
    with models.shared:
        out = eng.generate(text, max_tokens, 0.3, 40, 0, mode="question", repetition_penalty=1.15)
    return speakable(out["completion"]), time.time() - t0


def answer_text(models: Models, text: str, max_tokens: int = 120) -> tuple[str, float]:
    """Look it up first: retrieve Wikipedia passages and let our reader point
    at the answer (docs/RESULTS.md section 33). Only when it finds none does
    the question-answering model answer from memory, and the reply says so."""
    from voice import speakable
    t0 = time.time()
    qa = models.reader()
    if qa is not None:
        out = qa.ask(text)
        if out["answer"]:
            return f"{out['answer']}  (from Wikipedia: {out['sources'][0]})", time.time() - t0
    eng = models.brain()
    with models.shared:
        out = eng.generate(text, max_tokens, 0.3, 40, 0, mode="question", repetition_penalty=1.15)
    return "(not found in Wikipedia; from memory) " + speakable(out["completion"]), time.time() - t0


def build(models: Models) -> gr.Blocks:
    with gr.Blocks(title="AnuLM Omni") as demo:
        gr.Markdown(
            "# AnuLM Omni\n"
            "A 400M sparse mixture-of-experts model, trained from scratch on one consumer "
            "GPU, that **listens**, **looks** and **answers** in English and Hindi. "
            "The language model behind every tab is this project's own; hearing and seeing "
            "read frozen pretrained encoders (Whisper-small, SigLIP), lookup uses the "
            "multilingual-e5-small retriever, and the voice is Piper.")

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
                reply, t_think = answer_text(models, as_question(h["text"]))
                t1 = time.time()
                spoken = speech_or_none(models, reply)
                t_say = time.time() - t1
                note = "" if spoken else " · no voice for this language installed (text only)"
                return (h["text"], reply, spoken,
                        f"hear {t_hear:.2f}s · think {t_think:.2f}s · speak {t_say:.2f}s{note}")

            go_listen.click(do_listen, [audio, lang, speak_back],
                            [heard, said, voice_out, listen_meta])
            gr.Markdown("*Speech recognition: WER 7.0% English (100 held-out LibriSpeech clips), "
                        "37.4% Hindi (100 FLEURS clips, unseen speakers), about 0.11 x real time; "
                        "the Benchmark tab has every transcript. It struggles most with names.*")

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
                    q = as_question(models.ear("en").hear(sq)["text"])
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
                spoken = speech_or_none(models, out) if out != "(nothing)" else None
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
            chat_mode = gr.Radio([("Look it up (facts, from Wikipedia)", "lookup"),
                                  ("Just talk (the chat model's own words)", "chat")],
                                 value="lookup", label="how to answer")
            go_chat = gr.Button("Ask", variant="primary")
            a_out = gr.Textbox(label="AnuLM", lines=6)
            chat_meta = gr.Markdown()

            def do_chat(q, mode):
                if not q or not q.strip():
                    return "", ""
                if mode == "chat":
                    reply, t = chat_text(models, q.strip())
                    return reply, f"{t:.2f}s · chat model {Path(models.brain_source).name}"
                reply, t = answer_text(models, q.strip(), max_tokens=200)
                return reply, f"{t:.2f}s · reader {READER}"

            go_chat.click(do_chat, [q_in, chat_mode], [a_out, chat_meta])
            gr.Markdown("*Look it up: our reader points at the answer inside Wikipedia passages "
                        "(answers facts it was never told). Just talk: the chat model, tuned on "
                        "23k human-written instructions; fluent and on topic, but its facts from "
                        "memory are unreliable.*")

        with gr.Tab("Decide"):
            gr.Markdown("A *System One* decision, like TypeSafe's Jev: no text is generated. The "
                        "message and every option go through the model **once**; it returns a "
                        "calibrated probability for each, and hands the case to a person when it "
                        "is not confident enough. **AnuLM** is the same model as the Chat tab, "
                        "with a small LoRA adapter per task. **ModernBERT-base** is an open English "
                        "model used as a reference.\n\n"
                        "| unseen test messages | AnuLM | ModernBERT |\n|---|---|---|\n"
                        "| Banking, English (3,080) | 89.0% | **90.8%** |\n"
                        "| Assistant commands, **Hindi** (2,033) | **84.5%** | 67.5% |\n\n"
                        "AnuLM answers in about 55 ms, ModernBERT in about 20; in Hindi ModernBERT "
                        "reads byte fragments, 3.3x the tokens.")
            with gr.Row():
                with gr.Column():
                    task = gr.Radio([k for k, v in DECIDE_TASKS.items() if have(v[DECIDE_OWN])],
                                    value=TASK_BANK, label="task")
                    msg = gr.Textbox(label="message", lines=3,
                                     placeholder="I still haven't received my new card")
                    backbone = gr.Radio([DECIDE_OWN, DECIDE_MB], value=DECIDE_OWN, label="model")
                    thr = gr.Slider(0.3, 0.99, value=0.7, step=0.01, label="hand to a human below this confidence")
                    go_dec = gr.Button("Decide", variant="primary")
                    gr.Examples([["I still haven't received my new card", TASK_BANK],
                                 ["Why was I charged twice for the same coffee?", TASK_BANK],
                                 ["How do I top up with Apple Pay?", TASK_BANK],
                                 ["Someone is using my card right now, block it!", TASK_BANK],
                                 ["कल सुबह सात बजे का अलार्म लगा दो", TASK_HI],
                                 ["आज की ताज़ा खबरें सुनाओ", TASK_HI],
                                 # From MASSIVE's Hindi test set: AnuLM right at >= 0.9,
                                 # ModernBERT wrong (162 such messages; 9 the other way).
                                 ["क्या धूप वाला दिन है", TASK_HI],
                                 ["दस बजे मुझे उठाओ", TASK_HI],
                                 ["मेरी अगली बैठक कब है", TASK_HI],
                                 ["तीन गुना सात क्या है", TASK_HI],
                                 ["नए ईमेल की जाँच करें", TASK_HI]]
                                + ([  # Indian banking, phrased unlike anything in its training set;
                                    # the aadhaar one is right but unsure (0.40) and goes to a person
                                    ["paisa account se debit ho gaya lekin transaction failed dikha raha", TASK_IN],
                                    ["GPay pe payment fail ho raha hai baar baar", TASK_IN],
                                    ["मेरा यूपीआई पिन ब्लॉक हो गया है, नया कैसे बनाऊं", TASK_IN],
                                    ["My KYC is pending and the account got frozen", TASK_IN],
                                    ["gas subsidy nahi aa rahi, aadhaar link check karna hai", TASK_IN],
                                    ["एटीएम से कैश नहीं निकला पर 5000 कट गए", TASK_IN]]
                                   if have(DECIDE_ADAPTER_IN) else []), [msg, task])
                with gr.Column():
                    choice = gr.Label(label="intent (top 3)", num_top_classes=3)
                    action = gr.Markdown()
                    typed = gr.JSON(label="typed output")

            def do_decide(m, t, which, tk=TASK_BANK):
                if not m or not m.strip():
                    return None, "", None
                which, tk = which or DECIDE_OWN, tk or TASK_BANK
                if not have(DECIDE_TASKS[tk][which]):
                    return None, f"{which} has no trained model for this task here", None
                d = models.decider(which, tk)
                d.threshold = float(t)
                t0 = time.time()
                with models.shared:
                    out = d(m.strip())
                ms = (time.time() - t0) * 1000
                if out["action"] == "auto-route":
                    verdict = "**auto-route** to `" + out["choice"] + "`"
                else:
                    verdict = "**send to a human** (not confident enough)"
                return ({k.replace("_", " "): v for k, v in out["probabilities"].items()},
                        f"{verdict} · confidence {out['confidence']:.2f} · {ms:.0f} ms · {which}",
                        {**out, "model": which, "task": tk})

            go_dec.click(do_decide, [msg, thr, backbone, task], [choice, action, typed])

        import app_benchmark
        app_benchmark.wire_initial(demo, app_benchmark.add_tab())
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="AnuLM Omni: listen, look and talk.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--brain", default=BRAIN, help="the text model that answers")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7863)
    p.add_argument("--share", action="store_true", help="public gradio link")
    args = p.parse_args()
    # Checkpoints and their tokenizers are saved with repo-relative paths;
    # launched from elsewhere, the chat model silently fell back to the Hub one.
    os.chdir(HERE)
    missing = [c for c in CKPTS.values() if not have(c)]
    if missing:
        print(f"note: {', '.join(missing)} not found; those tabs will fail until trained "
              f"(docs/SPEECH.md has the commands)")
    build(Models(args.device, args.brain)).launch(server_name=args.host, server_port=args.port,
                                                  share=args.share)


if __name__ == "__main__":
    main()
