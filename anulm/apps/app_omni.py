"""
AnuLM Omni: listen, look and talk on one page.

    pip install gradio faster-whisper piper-tts timm pillow
    python -m anulm app_omni                       # everything on the GPU
    python -m anulm app_omni --host 0.0.0.0 --auth-file .demo_auth   # a phone on the same wifi, with a password

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

Each tab has a gallery from examples/samples/ (held-out rows, one click runs
them, the reference answer shown beside the output);
experiments/demo_samples_check.py scores all of them against a running page.

Models load on first use and stay resident. All of them together (three
backbones for hearing/seeing, one for answering, plus the encoders) fit a
16 GB card; use --device cpu to try it without one.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
from pathlib import Path

import gradio as gr
import torch

HERE = Path(__file__).resolve().parents[2]

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
# The chat model and its Decide adapters go together: an adapter attaches only
# to the exact chat model it was trained on (Decide.on refuses any other).
# v3 is the chat model fine-tuned on base v2 (held-out answer loss 3.1445 vs
# 3.2000), with recipe-F adapters (dense, rank 64; logs/post_v2, logs/indic_v4):
# BANKING77 89.03% at 56 ms, MASSIVE-hi 84.46% at 54 ms, vs 86.4% / 82.6% for v2.
if (HERE / "ckpt_chat_v3.pt").exists() or not (HERE / "ckpt_chat_v2.pt").exists():   # or a clone: the Hub's
    BRAIN, DECIDE_ADAPTER, DECIDE_ADAPTER_HI = "ckpt_chat_v3.pt", "ckpt_decide_lora_F.pt", "ckpt_decide_lora_hi_F.pt"
    DECIDE_ADAPTER_IN = "ckpt_decide_lora_india_F.pt"
else:
    BRAIN = "ckpt_chat_v2.pt" if (HERE / "ckpt_chat_v2.pt").exists() else "release/AnuLM-Hindi-QA-400M"
    DECIDE_ADAPTER, DECIDE_ADAPTER_HI = "ckpt_decide_chat_lora.pt", "ckpt_decide_lora_hi.pt"
    DECIDE_ADAPTER_IN = "ckpt_decide_lora_india_F.pt"     # trained on v3 only: absent here, the task hides
READER = "ckpt_rc_pointer_v2.pt" if (HERE / "ckpt_rc_pointer_v2.pt").exists() else "ckpt_rc_pointer.pt"
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
    TASK_BANK: {DECIDE_OWN: DECIDE_ADAPTER if (HERE / DECIDE_ADAPTER).exists() else "ckpt_decide.pt",
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

# What the page says about the one model comes from experiments/omni_one.py's
# own records (results.json, the training log, the curve), so a retrain
# updates the page without anyone editing it. A clone without them points at
# the Hub card, which carries the same table.
OMNI_LOGS = HERE / "logs" / "omni_one"
MEASURES = {        # key -> (what it is, unit, higher is better)
    "chat_heldout_loss": ("Chat: held-out answer loss", "", False),
    "read_squad_en": ("Reading, English: SQuAD F1", "", True),
    "read_mlqa_hi": ("Reading, Hindi: MLQA F1", "", True),
    "intent_banking77": ("Banking intents, English: BANKING77 accuracy", "%", True),
    "intent_massive_hi": ("Assistant commands, Hindi: MASSIVE accuracy", "%", True),
    "wer_en": ("English speech: WER, LibriSpeech held-out", "%", False),
    "wer_hi_fleurs": ("Hindi speech: WER, FLEURS", "%", False),
    "wer_hi_kathbath": ("Hindi speech: WER, Kathbath", "%", False),
    "wer_hi_mucs": ("Hindi speech: WER, MUCS", "%", False),
    "wer_hi_mean": ("Hindi speech: WER, mean of the three", "%", False),
    "vqa_acc": ("Image questions: TextVQA accuracy", "%", True),
    "caption_heldout_loss": ("Captions: held-out loss", "", False),
}


def _omni_one():
    """experiments/omni_one.py as a module: its gates and task mix are the truth."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("omni_one", HERE / "experiments" / "omni_one.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def omni_results() -> dict:
    res = _json(OMNI_LOGS / "results.json")
    if res:
        _omni_one().gates(res)          # adds wer_hi_mean
    return res


def num(res: dict, key: str) -> str:
    v = res.get(key)
    return "–" if v is None else f"{v:g}{MEASURES[key][1]}"


def omni_status(res: dict, pointer: bool = True) -> str:
    if not res:
        return f"Measured numbers are on the model card: https://huggingface.co/{OMNI_HUB}"
    g = _omni_one().gates(dict(res))
    short = {"chat_heldout_loss": "chat", "read_squad_en": "English reading", "wer_en": "English speech",
             "wer_hi_fleurs": "Hindi speech on FLEURS", "wer_hi_mean": "Hindi speech",
             "intent_banking77": "banking intents", "vqa_acc": "image questions"}
    failed = [short.get(k, k) for k, ok in g.items() if not ok]
    when = time.strftime("%Y-%m-%d", time.localtime((OMNI_LOGS / "results.json").stat().st_mtime))
    if not failed:
        return f"Measured {when}: all {len(g)} release gates pass."
    return (f"Measured {when}: **not released yet**. {len(g) - len(failed)} of {len(g)} release gates "
            f"pass; still short on {', '.join(dict.fromkeys(failed))}."
            + (" The Model & data tab has every number." if pointer else ""))


def have(name: str) -> bool:
    if USE_OMNI and name not in {v[DECIDE_MB] for v in DECIDE_TASKS.values()}:
        return name in set(CKPTS.values()) | {READER} | {DECIDE_TASKS[t][DECIDE_OWN] for t in OMNI_TASKS}
    return (HERE / name).exists()


class Models:
    """Lazy loader: nothing touches the GPU until a tab first needs it."""

    def __init__(self, device: str, brain: str):
        self.device = device
        self.brain_source = brain
        self._m: dict[str, object] = {}
        # One lock for everything on the shared decoder. Chat and Decide share
        # the chat model's weights (Decide switches its adapters on for a
        # forward pass); with the one model, hearing, looking and reading run
        # on that same decoder too. Gradio serves visitors from a thread pool,
        # so without it two requests interleave on one module. It also guards
        # loading: two first requests used to load the model twice. Re-entrant,
        # because one loader calls another (Decide.on -> brain) and every
        # caller holding it may load on first use.
        self.shared = threading.RLock()

    def _get(self, key, make):
        with self.shared:
            if key not in self._m:
                t0 = time.time()
                self._m[key] = make()
                print(f"loaded {key} in {time.time() - t0:.1f}s", flush=True)
            return self._m[key]

    def _half(self, module):
        """bf16 weights on the GPU: five resident backbones in fp32 filled the
        16 GB card to 15,986 MiB, the edge of the silent spill to system RAM."""
        if self.device.startswith("cuda"):
            from anulm.model import Rotary, Router
            routers = {m: {k: v.detach().clone() for k, v in m.state_dict().items()}
                       for m in module.modules() if isinstance(m, Router)}
            module.to(torch.bfloat16)
            # The routers stay fp32: bf16 cannot hold their biases
            # (per-layer means of 216-798 with spreads under 0.3 round to one or
            # two values), and Router.forward computes in fp32 regardless, so
            # mixed dtypes cannot reach a matmul. Everything else stays bf16:
            # an fp32 bias beside a bf16 weight breaks F.linear outside autocast.
            for m, state in routers.items():     # restored from the copy: casting back
                m.float()                         # would only re-widen the rounded values
                m.load_state_dict(state)
            # .to(dtype) also casts RoPE's inv_freq buffer; in bf16 the angles
            # drift 0.19 rad by position 200 and 0.94 by 1,000, the end of the
            # context (course/ch05-positions/lab.py). Rebuild it in fp32.
            for m in module.modules():
                if isinstance(m, Rotary):
                    m.inv_freq = Rotary._build_inv_freq(m.cfg, m.dim).to(self.device)
                    m._cached_len, m._cached_key = 0, None
        return module

    def omni(self):
        from anulm.multimodal.omni import Omni
        return self._get("omni", lambda: Omni(OMNI, self.device))

    def ear(self, lang: str):
        from anulm.inference.voice import OwnASR
        if USE_OMNI:                      # one ear for both languages, on the shared decoder
            return self.omni().ear()

        def make():
            asr = OwnASR(str(HERE / CKPTS[key]), self.device)
            self._half(asr.sm)
            return asr
        key = "asr_hi" if lang == "hi" else "asr_en"
        return self._get(key, make)

    def eye(self, task: str):
        from anulm.multimodal.vision_encoder import load_trained

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
        from anulm.tasks.rc_pointer import PointerQA
        return self._get("reader", lambda: PointerQA(str(HERE / READER), self.device))

    def brain(self):
        from anulm.inference.voice import load_engine
        if USE_OMNI and self.brain_source == BRAIN:
            return self.omni().engine
        src = self.brain_source
        if not os.path.exists(src) and src == BRAIN:
            src = BRAIN_HUB
        return self._get("brain", lambda: load_engine(src, self.device))

    def mouth(self):
        from anulm.inference.voice import TTS
        return self._get("tts", TTS)

    def decider(self, which: str = DECIDE_OWN, task: str = TASK_BANK):
        """Our model's adapters attach to the loaded chat model; each has its
        own hooks, switched on only for its own forward pass."""
        from anulm.tasks.decide import Decide
        if USE_OMNI and which == DECIDE_OWN:      # ModernBERT stays a separate reference model
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
    from anulm.inference.voice import speakable
    t0 = time.time()
    with models.shared:
        eng = models.brain()
        out = eng.generate(text, max_tokens, 0.3, 40, 0, mode="question", repetition_penalty=1.15)
    return speakable(out["completion"]), time.time() - t0


def hear(models: Models, lang: str, audio) -> dict:
    """models.ear(lang).hear(audio), under the shared-decoder lock."""
    with models.shared:
        return models.ear(lang).hear(audio)


def look(models: Models, image, question: str | None) -> str:
    """A caption, or the answer to `question` about `image`, under the
    shared-decoder lock."""
    from anulm.multimodal.vision_encoder import answer, caption
    with models.shared:
        if question and question.strip():
            vl, tower, tok = models.eye("vqa")
            return answer(vl, tower, tok, image, question.strip(), device=models.device)
        vl, tower, tok = models.eye("caption")
        return caption(vl, tower, tok, image, device=models.device)


def decide(models: Models, message: str, threshold: float, which: str = DECIDE_OWN,
           task: str = TASK_BANK) -> dict:
    """One decision. The decider is shared by every visitor, so its threshold
    is set inside the lock, right before the call that reads it: set outside,
    one visitor's slider could decide another visitor's action."""
    with models.shared:
        d = models.decider(which, task)
        d.threshold = float(threshold)
        return d(message.strip())


def answer_text(models: Models, text: str, max_tokens: int = 120) -> tuple[str, float]:
    """Look it up first: retrieve Wikipedia passages and let our reader point
    at the answer (docs/RESULTS.md section 33). Only when it finds none does
    the question-answering model answer from memory, and the reply says so."""
    from anulm.inference.voice import speakable
    t0 = time.time()
    with models.shared:                     # the reader is a view of the shared decoder too
        qa = models.reader()
        out = qa.ask(text) if qa is not None else {"answer": ""}
    if out["answer"]:
        return f"{out['answer']}  (from Wikipedia: {out['sources'][0]})", time.time() - t0
    with models.shared:
        eng = models.brain()
        out = eng.generate(text, max_tokens, 0.3, 40, 0, mode="question", repetition_penalty=1.15)
    return "(not found in Wikipedia; from memory) " + speakable(out["completion"]), time.time() - t0


def omni_look_examples() -> tuple[list, str]:
    """Held-out photos the one model never trained on, all commercially
    licensed: DOCCI for captions, TextVQA (Open Images) for questions."""
    def rows(name):
        p = HERE / "data" / "omni" / name
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []
    base = HERE / "data" / "omni"
    caps = [r for r in rows("caption_test.jsonl") if r.get("source") == "docci"][:3]
    # Questions whose answer is written in the photo -- what TextVQA teaches.
    wanted = ("what is the brand of this camera?", "what is the time?", "what number is on the player's jersey?")
    qs = [r for r in rows("vqa_test.jsonl") if r["question"] in wanted][:3]
    ex = [[str((base / r["image"]).resolve()), ""] for r in caps]
    ex += [[str((base / r["image"]).resolve()), r["question"]] for r in qs]
    ex = [e for e in ex if Path(e[0]).exists()]
    authors = {}
    att = base / "ATTRIBUTION_textvqa.csv"
    if att.exists():
        with open(att, encoding="utf-8") as f:
            authors = {r["image_id"]: r["author"] for r in csv.DictReader(f)}
    by = sorted({authors[Path(r["image"]).stem] for r in qs if Path(r["image"]).stem in authors})
    note = ("Example photos, none seen in training: DOCCI test set (Google, CC BY 4.0); TextVQA "
            "photos from Open Images" + (f" by {', '.join(by)}" if by else "") + " (CC BY 2.0).")
    return ex, note


# The sample library (examples/samples/, built by experiments/make_demo_samples.py):
# 117 held-out rows the one model never trained on, each with its reference
# answer and the credit its licence asks for. experiments/demo_samples_check.py
# runs every one of them through this page's API, so what a visitor clicks is
# what the check scores.
SAMPLES = HERE / "examples" / "samples"
SAMPLE_TASKS = {"banking": TASK_BANK, "massive-hi": TASK_HI}


def sample_library(*kinds: str) -> list[dict]:
    """The manifest's rows of these kinds whose media exist; [] without the library."""
    items = _json(SAMPLES / "samples.json").get("items", [])
    return [i for i in items if i["kind"] in kinds
            and all((SAMPLES / v).exists() for k, v in i["inputs"].items() if k in ("audio", "image"))]


def sample_file(item: dict, key: str) -> str:
    return str((SAMPLES / item["inputs"][key]).resolve())


def sample_reference(item: dict) -> str:
    """What the reference box shows: the dataset's answer, then the source and
    credit (CC BY asks for the credit wherever the material is shown)."""
    ref, kind = item.get("reference"), item["kind"]
    if kind == "caption":
        head = f"A person wrote: {ref}"
    elif kind == "vqa":
        others = [a for a in item.get("answers", []) if a != ref]
        head = f"Most annotators answered: {ref}" + (f" (also given: {', '.join(others)})" if others else "")
    elif kind == "lookup":
        head = "Expected: " + " / ".join(item.get("answers") or [ref])
    elif kind == "chat":
        head = "No single right answer: a prompt written for this library."
    elif kind.startswith("intent"):
        head = f"Labelled intent: {ref}"
    else:
        head = ref or ""
    return f"{head}\n\n{item['source']} · {item['license']} · {item['attribution']}"


def sample_label(item: dict, n: int = 56) -> str:
    """A short button label: text where the input is text, where-from for media
    (an audio player or thumbnail per row would load every file with the page)."""
    kind = item["kind"]
    if kind.startswith("asr"):
        where = item["source"].split(",")[0].replace(" hi_in test", "").replace(" Hindi held-out (valid split)", "")
        return f"{'English' if item['language'] == 'en' else 'हिन्दी'} · {where} · {item.get('seconds', 0):.0f} s"
    if kind == "caption":
        return "describe: " + item["source"].replace("DOCCI test, ", "DOCCI ")
    text = item["inputs"].get("question") or item["inputs"].get("message") or item["id"]
    text = text if len(text) <= n else text[:n - 1] + "…"
    return {"lookup": "look up: ", "vqa": "ask: "}.get(kind, "") + text


def whole_sentences(text: str) -> str:
    """A caption cut at the evaluation's token limit, ended at its last full
    sentence (or marked with an ellipsis when that would drop most of it)."""
    text = text.strip()
    if not text or text[-1] in '.!?"':
        return text
    cut = text.rfind(". ")
    return text[:cut + 1] + " …" if cut > len(text) // 3 else text.rstrip(",;: ") + " …"


def best_step(df) -> tuple[int, int] | None:
    """(step kept, last step) from the curve: omni_train.py keeps the lowest
    combined score, the first one on a tie."""
    if "score" not in df.columns or len(df) < 2:
        return None
    return int(df.loc[df["score"].idxmin(), "step"]), int(df["step"].iloc[-1])


def omni_tab(res: dict) -> None:
    """The Model & data tab: what the one model scores, how it got there, on what."""
    gr.Markdown("## The one model\n" + omni_status(res, pointer=False))
    if res:
        oo = _omni_one()
        g = oo.gates(dict(res))
        start = _json(OMNI_LOGS / "eval_base_chat.json")
        rows = ["| task and measure | AnuLM-Omni | chat model it started from | separate model before | "
                "release gate | |", "|---|---|---|---|---|---|"]
        for k, (what, unit, up) in MEASURES.items():
            gate = oo.GATES.get(k)
            mark = ("pass" if g[k] else "**not yet**") if k in g else ""
            rows.append(f"| {what} ({'higher' if up else 'lower'} is better) | {num(res, k)} | "
                        f"{num(start, k) if k in start else '–'} | {oo.EARLIER.get(k, '–')} | "
                        f"{f'{gate[0]} {gate[1]}' if gate else '–'} | {mark} |")
        gr.Markdown("\n".join(rows) + "\n\n*The chat model it started from had never seen the intent "
                    "labels, so its intent scores are close to guessing: they show what "
                    "the training added.*")
        if res.get("caption_samples"):
            gr.Markdown("### Captions on held-out DOCCI photos, beside what a person wrote\n"
                        "*The evaluation stops each caption at a fixed length; a cut is marked …*\n\n" + "\n".join(
                f"- **person:** {s['reference']}\n  **AnuLM:** {whole_sentences(s['model'])}"
                for s in res["caption_samples"]))

    curve = HERE / "ckpt_omni_curve.csv"
    if curve.exists():
        import pandas as pd
        df = pd.read_csv(curve)
        tasks = [c for c in df.columns if c not in ("step", "score")]
        first = df.iloc[0]
        long = pd.concat([pd.DataFrame({"step": df["step"], "task": t, "relative_loss": df[t] / first[t]})
                          for t in tasks])
        kept = best_step(df)
        gr.Markdown("### Training, per task\nHeld-out loss divided by its value before training: "
                    "below 1 the task has improved, above 1 it has got worse."
                    + (f" The run trained {kept[1]:,} steps and kept the checkpoint with the best "
                       f"combined score, **step {kept[0]:,}**: every number on this page is that "
                       f"checkpoint's." if kept else ""))
        gr.LinePlot(long, x="step", y="relative_loss", color="task", height=320)

    # The mix of the run that made this checkpoint, from its own log (the
    # pipeline's settings may have moved on since); the pipeline's as fallback.
    mix_rows, accum = {}, 1
    log = OMNI_LOGS / "6_train.log"
    if log.exists():
        text = log.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"^task (\S+)\s+(\S+)\s+(\d+) train\s+(\d+) held\s+batch (\d+)\s+weight (\S+)", text, re.M):
            mix_rows[m.group(1)] = (m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5)), float(m.group(6)))
        sums = re.findall(r"^each optimizer step sums (\d+) batch", text, re.M)
        accum = int(sums[-1]) if sums else 1          # runs before --accum existed: one batch per step
    if not mix_rows:
        oo = _omni_one()
        mix_rows = {n: (k, None, None, b, w) for n, k, _p, w, b in oo.TASKS}
        accum = getattr(oo, "ACCUM", 1)
    total = sum(r[4] for r in mix_rows.values())
    mix = ["| task | input | examples (train / held out) | share of batches | batch |", "|---|---|---|---|---|"]
    for name, (kind, n_tr, n_held, b, w) in mix_rows.items():
        mix.append(f"| {name} | {kind} | {f'{n_tr:,} / {n_held:,}' if n_tr else '–'} | "
                   f"{100 * w / total:.0f}% | {b} |")
    gr.Markdown(f"### How it was trained\nOne decoder, all seven tasks at once, starting from the chat "
                f"model (itself base v2, pretrained here from scratch on 11.3B tokens). Each optimizer step "
                f"{'sums ' + str(accum) + ' batches' if accum > 1 else 'takes one batch'}, drawn at random "
                f"by the shares below. To train it yourself, from a 17M toy model up to every stage "
                f"from nothing, follow `docs/TRAIN_FROM_SCRATCH.md`.\n\n" + "\n".join(mix))
    card = HERE / "docs" / "MODEL_CARD_OMNI.md"
    if card.exists():
        text = card.read_text(encoding="utf-8")
        m = re.search(r"^## Training data\n(.*?)(?=^## )", text, re.S | re.M)
        if m:
            gr.Markdown("### Training data\n" + m.group(1).strip())


def build(models: Models) -> gr.Blocks:
    res = omni_results() if USE_OMNI else {}
    with gr.Blocks(title="AnuLM Omni") as demo:
        if USE_OMNI:
            gr.Markdown(
                "# AnuLM-Omni-400M\n"
                "**One model behind every tab.** A single 456M mixture-of-experts decoder (232M active "
                "per token), pretrained from scratch on one consumer GPU, then trained on seven tasks at "
                "once: it **listens** (English and Hindi speech), **looks** (captions, questions about a "
                "photo), **reads**, **routes** messages and **chats**. Trained only on data that allows "
                "commercial use. Hearing and seeing go through frozen borrowed encoders (Whisper-small, "
                "SigLIP), lookup uses the multilingual-e5-small retriever, and the voice is Piper.\n\n"
                + omni_status(res))
        else:
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
                    listen_lib = sample_library("asr_en", "asr_hi")
                    listen_ref = gr.Textbox(label="what was actually said (library samples)", lines=3,
                                            interactive=False, visible=bool(listen_lib))

            def do_listen(a, lg, speak):
                if a is None:
                    return "", "", None, "*record or upload a clip first*"
                t0 = time.time()
                h = hear(models, lg, a)
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
            if listen_lib:
                # The clips are read sentences, not questions, so a sample turns
                # "answer out loud" off: the transcript is what the reference checks.
                ex = gr.Examples([[sample_file(i, "audio"), i["inputs"]["lang"], False, sample_reference(i)]
                                  for i in listen_lib], [audio, lang, speak_back, listen_ref],
                                 label=f"{len(listen_lib)} held-out clips: one click transcribes",
                                 example_labels=[sample_label(i) for i in listen_lib], examples_per_page=10)
                # Chained on the click rather than run_on_click, so the sample
                # runs through the same handler, and settings, as the button.
                ex.load_input_event.then(do_listen, [audio, lang, speak_back],
                                         [heard, said, voice_out, listen_meta], api_visibility="undocumented")
                # A clip of the visitor's own makes the shown reference stale.
                for ev in (audio.upload, audio.stop_recording):
                    ev(lambda: "", None, listen_ref, queue=False, api_visibility="private")
            if USE_OMNI and res:
                gr.Markdown(f"*Speech recognition by the one model, word error rate (lower is better): "
                            f"{num(res, 'wer_en')} English (100 held-out LibriSpeech clips); Hindi "
                            f"{num(res, 'wer_hi_fleurs')} FLEURS, {num(res, 'wer_hi_kathbath')} Kathbath, "
                            f"{num(res, 'wer_hi_mucs')} MUCS (held-out clips). Trained on LibriSpeech, "
                            f"Kathbath, MUCS and FLEURS. It struggles most with names.*")
            elif not USE_OMNI:
                gr.Markdown("*Speech recognition: WER 7.0% English (100 held-out LibriSpeech clips), "
                            "37.4% Hindi (100 held-out FLEURS clips), about 0.11 x real time; "
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
                    look_lib = sample_library("caption", "vqa") if USE_OMNI else []
                    look_ref = gr.Textbox(label="reference (library samples)", lines=4,
                                          interactive=False, visible=bool(look_lib))

            def do_look(image, q, sq):
                if image is None:
                    return q, "", None, "*drop an image in first*"
                t0 = time.time()
                parts = []
                if sq is not None:
                    q = as_question(hear(models, "en", sq)["text"])
                    parts.append(f"heard {time.time() - t0:.2f}s")
                t1 = time.time()
                out = look(models, image, q)
                if q and q.strip():
                    parts.append(f"answer ({'AnuLM-Omni' if USE_OMNI else 'A-OKVQA-trained'}) "
                                 f"{time.time() - t1:.2f}s")
                else:
                    parts.append(f"caption ({'AnuLM-Omni' if USE_OMNI else 'Flickr30k-trained'}) "
                                 f"{time.time() - t1:.2f}s")
                out = out or "(nothing)"
                spoken = speech_or_none(models, out) if out != "(nothing)" else None
                return q, out, spoken, " · ".join(parts)

            go_look.click(do_look, [img, question, spoken_q], [question, seen, look_voice, look_meta])
            def look_sample(image, q):
                """A sample asks with its typed question; a spoken one left in
                the box from earlier would otherwise replace it."""
                return do_look(image, q, None)

            if look_lib:
                ex = gr.Examples([[sample_file(i, "image"), i["inputs"]["question"], sample_reference(i)]
                                  for i in look_lib], [img, question, look_ref],
                                 label=f"{len(look_lib)} held-out photos: one click describes it, or asks "
                                       f"what is written",
                                 example_labels=[sample_label(i) for i in look_lib], examples_per_page=11)
                ex.load_input_event.then(look_sample, [img, question], [question, seen, look_voice, look_meta],
                                         api_visibility="undocumented")
                img.upload(lambda: "", None, look_ref, queue=False, api_visibility="private")
                question.input(lambda: "", None, look_ref, queue=False, api_visibility="private")
                gr.Markdown("*" + ("Captions learned from Localized Narratives and DOCCI; questions from "
                                   f"TextVQA, {num(res, 'vqa_acc')} right on 1,000 held-out questions. "
                                   if res else "") + "Sample photos, none seen in training: DOCCI test set "
                            "(Google, CC BY 4.0) and TextVQA photos from Open Images (CC BY 2.0), each "
                            "credited in its reference box and in `examples/samples/ATTRIBUTION.md`.*")
            elif USE_OMNI:
                ex, credit = omni_look_examples()
                if ex:
                    gr.Examples(ex, [img, question], label="held-out photos: describe, or ask what is written")
                gr.Markdown("*" + ("Captions learned from Localized Narratives and DOCCI; questions from "
                                   f"TextVQA, {num(res, 'vqa_acc')} right on 1,000 held-out questions. "
                                   if res else "") + credit + "*")
            else:
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
            chat_lib = sample_library("lookup") + sample_library("chat")     # the default mode's first
            chat_ref = gr.Textbox(label="reference (library samples)", lines=2, interactive=False,
                                  visible=bool(chat_lib))

            def do_chat(q, mode):
                if not q or not q.strip():
                    return "", ""
                if mode == "chat":
                    reply, t = chat_text(models, q.strip())
                    omni = USE_OMNI and models.brain_source == BRAIN      # Models.brain() uses the one model
                    return reply, f"{t:.2f}s · " + ("AnuLM-Omni" if omni else
                                                    f"chat model {Path(models.brain_source).name}")
                reply, t = answer_text(models, q.strip(), max_tokens=200)
                return reply, f"{t:.2f}s · " + ("AnuLM-Omni reader" if USE_OMNI else f"reader {READER}")

            go_chat.click(do_chat, [q_in, chat_mode], [a_out, chat_meta])
            if chat_lib:
                ex = gr.Examples([[i["inputs"]["question"], i["inputs"]["mode"], sample_reference(i)]
                                  for i in chat_lib], [q_in, chat_mode, chat_ref],
                                 label=f"{len(chat_lib)} samples: Wikipedia lookups, then English, Hindi "
                                       f"and Python prompts (one click asks)",
                                 example_labels=[sample_label(i) for i in chat_lib], examples_per_page=10)
                ex.load_input_event.then(do_chat, [q_in, chat_mode], [a_out, chat_meta],
                                         api_visibility="undocumented")
                q_in.input(lambda: "", None, chat_ref, queue=False, api_visibility="private")
            if USE_OMNI:
                start = _json(OMNI_LOGS / "eval_base_chat.json")
                gr.Markdown("*Look it up: the one model reads the Wikipedia passages the retriever finds "
                            "and writes the answer"
                            + (f" (SQuAD F1 {num(res, 'read_squad_en')})" if res else "")
                            + ". Just talk: its own words, learned from 23k human-written instructions"
                            + (f"; held-out answer loss {num(res, 'chat_heldout_loss')}, against "
                               f"{num(start, 'chat_heldout_loss')} for the chat model it started from "
                               f"(lower is better)" if res and start else "")
                            + ". Its facts from memory are unreliable.*")
            else:
                gr.Markdown("*Look it up: our reader points at the answer inside Wikipedia passages "
                            "(answers facts it was never told). Just talk: the chat model, tuned on "
                            "23k human-written instructions; fluent and on topic, but its facts from "
                            "memory are unreliable.*")

        with gr.Tab("Decide"):
            if USE_OMNI:
                gr.Markdown("A *System One* decision: no text is generated. The message and every "
                            "option go through the model **once**; it returns a probability for each, and "
                            "hands the case to a person when it is not confident enough. **AnuLM-Omni** is "
                            "the same one model as every other tab, with no adapter: it learned these "
                            "intents alongside everything else. **ModernBERT-base** is an open English "
                            "model kept as a reference.\n\n"
                            "| unseen test messages | AnuLM-Omni | AnuLM chat + adapter (before) | ModernBERT |\n"
                            "|---|---|---|---|\n"
                            f"| Banking, English | {num(res, 'intent_banking77')} | 89.0% | 90.8% |\n"
                            f"| Assistant commands, **Hindi** | {num(res, 'intent_massive_hi')} | 84.5% | 67.5% |")
            else:
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
                    backbone = gr.Radio([("AnuLM-Omni (the one model, no adapter)" if USE_OMNI else DECIDE_OWN,
                                          DECIDE_OWN), (DECIDE_MB, DECIDE_MB)], value=DECIDE_OWN, label="model")
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
                    decide_lib = [i for i in sample_library("intent_banking", "intent_massive_hi")
                                  if have(DECIDE_TASKS[SAMPLE_TASKS[i["inputs"]["task"]]][DECIDE_OWN])]
                    decide_ref = gr.Textbox(label="reference (library samples)", lines=2, interactive=False,
                                            visible=bool(decide_lib))
                    typed = gr.JSON(label="typed output")

            def do_decide(m, t, which, tk=TASK_BANK):
                if not m or not m.strip():
                    return None, "", None
                which, tk = which or DECIDE_OWN, tk or TASK_BANK
                name = "AnuLM-Omni" if USE_OMNI and which == DECIDE_OWN else which
                if not have(DECIDE_TASKS[tk][which]):
                    return None, f"{name} has no trained model for this task here", None
                with models.shared:                  # load first, so the time below is the decision's
                    models.decider(which, tk)
                t0 = time.time()
                out = decide(models, m, t, which, tk)
                ms = (time.time() - t0) * 1000
                if out["action"] == "auto-route":
                    verdict = "**auto-route** to `" + out["choice"] + "`"
                else:
                    verdict = "**send to a human** (not confident enough)"
                return ({k.replace("_", " "): v for k, v in out["probabilities"].items()},
                        f"{verdict} · confidence {out['confidence']:.2f} · {ms:.0f} ms · {name}",
                        {**out, "model": name, "task": tk})

            go_dec.click(do_decide, [msg, thr, backbone, task], [choice, action, typed])
            if decide_lib:
                # Fills message and task only: the model and threshold stay as
                # the visitor set them, and the chained run reads them.
                ex = gr.Examples([[i["inputs"]["message"], SAMPLE_TASKS[i["inputs"]["task"]], sample_reference(i)]
                                  for i in decide_lib], [msg, task, decide_ref],
                                 label=f"{len(decide_lib)} held-out test messages (BANKING77, MASSIVE hi-IN): "
                                       f"one click decides",
                                 example_labels=[sample_label(i) for i in decide_lib], examples_per_page=10)
                ex.load_input_event.then(do_decide, [msg, thr, backbone, task], [choice, action, typed],
                                         api_visibility="undocumented")
                msg.input(lambda: "", None, decide_ref, queue=False, api_visibility="private")

        if USE_OMNI:
            with gr.Tab("Model & data"):
                omni_tab(res)

        from anulm.apps import app_benchmark
        # The recording is of whichever models the API served when it was made;
        # until it is re-made with the one model, say so rather than let it pass as Omni's.
        note = ("> **An earlier recording.** These requests went to the *separate* models, one per task, "
                "that AnuLM-Omni replaces (including test sets it does not use, such as Flickr30k and "
                "A-OKVQA). The one model's own numbers are on the **Model & data** tab."
                if USE_OMNI else "")
        app_benchmark.wire_initial(demo, app_benchmark.add_tab(note=note))
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="AnuLM Omni: listen, look and talk.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--brain", default=BRAIN, help="the text model that answers")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7863)
    p.add_argument("--share", action="store_true", help="public gradio link")
    p.add_argument("--auth-file", default=os.environ.get("ANULM_DEMO_AUTH_FILE"),
                   help="a file with one line, user:password; the page asks for it before anything "
                        "else. Use it with --host 0.0.0.0 or --share, or anyone who can reach the port "
                        "can use this machine's GPU. Keep the file out of git (.demo_auth is ignored)")
    args = p.parse_args()
    from anulm.apps.app import is_loopback
    auth = None
    if args.auth_file:
        user, _, pw = Path(args.auth_file).read_text(encoding="utf-8").strip().partition(":")
        if not user or not pw:
            raise SystemExit(f"{args.auth_file}: expected one line, user:password")
        auth = (user, pw)
    elif args.share or not is_loopback(args.host):
        print("warning: reachable from other machines with no password; see --auth-file", flush=True)
    # Checkpoints and their tokenizers are saved with repo-relative paths;
    # launched from elsewhere, the chat model silently fell back to the Hub one.
    os.chdir(HERE)
    missing = [c for c in CKPTS.values() if not have(c)]
    if missing:
        print(f"note: {', '.join(missing)} not found; those tabs will fail until trained "
              f"(docs/SPEECH.md has the commands)")
    build(Models(args.device, args.brain)).launch(server_name=args.host, server_port=args.port,
                                                  share=args.share, auth=auth)


if __name__ == "__main__":
    main()
