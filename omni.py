"""
The one model (omni_train.py) behind every tab of app_omni.py, one copy of it.

Each separate model came with its own loader, and each loader built its own
decoder. The joint model has one decoder, so here the chat engine loads it
once and the ear, the eye, the reader and the intent classifier are views of
that same module: a projector, a prompt, or a scoring rule on top.

    from omni import Omni
    om = Omni("ckpt_omni.pt", "cuda")        # or "toonist/AnuLM-Omni-400M"
    om.engine.generate("What is UPI?", 60, 0.3, 40, 0, mode="question")
    om.ear().hear("clip.wav");  om.eye()       # (vl, tower, tok) for vision_encoder
    om.reader().ask("Who wrote Godan?")        # retrieval + answer-from-passages
    om.decider("banking")("my card has not arrived")
"""

from __future__ import annotations

import re
import time

import torch
import torch.nn.functional as F

DEV = re.compile(r"[ऀ-ॿ]")

# The prompts omni_data.py writes the intent rows with; scoring must use the same.
INTENT_PROMPT = {"banking": ("en", "Which banking intent is this? {m}"),
                 "massive-hi": ("hi", "यह किस काम का आदेश है? {m}")}
INTENT_TASK = {"banking": "banking77", "massive-hi": "massive-hi"}


class Omni:
    def __init__(self, src: str, device: str):
        from model import load_checkpoint
        from serve import Engine
        self.src, self.device = src, device
        self.engine = Engine(src, device)
        self.model, self.tok = self.engine.model, self.engine.tok
        ck = load_checkpoint(src, "cpu")
        self.meta = {k: ck[k] for k in ("speech_text_vocab", "whisper", "stack", "tower", "pool") if k in ck}
        self.proj, self.vision_proj = ck.get("proj"), ck.get("vision_proj")
        del ck
        self._ear = self._eye = self._reader = None
        self._decide = {}

    def ear(self):
        """voice.OwnASR's interface over the shared decoder."""
        if self._ear is None:
            from audio_codec import WhisperEncoder
            from hf_tok import load_tokenizer
            from speech_encoder import ContinuousSpeech
            from speech_vocab import vocab_of
            from voice import OwnASR
            enc = WhisperEncoder(self.meta.get("whisper", "small"), self.device)
            sm = ContinuousSpeech(self.model, vocab_of(self.meta), enc.dim, self.meta.get("stack", 4)).to(self.device)
            sm.proj.load_state_dict(self.proj)
            sm.eval()
            asr = OwnASR.__new__(OwnASR)
            asr.device, asr.ckpt = self.device, self.src
            asr.sm, asr.enc, asr.tok = sm, enc, load_tokenizer(self.model.cfg.tokenizer_path)
            self._ear = asr
        return self._ear

    def eye(self):
        """vision_encoder.load_trained's (vl, tower, tok) over the shared decoder."""
        if self._eye is None:
            from hf_tok import load_tokenizer
            from speech_vocab import vocab_of
            from vision_encoder import VisionLanguage, VisionTower
            tower = VisionTower(self.meta["tower"], self.device)
            vl = VisionLanguage(self.model, vocab_of(self.meta), tower.dim, self.meta.get("pool", 2)).to(self.device)
            vl.proj.load_state_dict(self.vision_proj)
            vl.eval()
            self._eye = (vl, tower, load_tokenizer(self.model.cfg.tokenizer_path))
        return self._eye

    def reader(self, retriever=None):
        if self._reader is None:
            self._reader = Reader(self, retriever)
        return self._reader

    def decider(self, task: str, threshold: float = 0.7):
        if task not in self._decide:
            self._decide[task] = Intent(self, task, threshold)
        return self._decide[task]


class Reader:
    """rc_pointer.PointerQA.ask's interface: retrieve passages (rag.py), then
    the model writes the answer from them, in the prompt it was trained on
    (make_qa.PROMPTS rc-en / rc-hi, make_rc.py's passage layout)."""

    def __init__(self, om: Omni, retriever=None):
        self.om, self.r = om, retriever

    def ask(self, question: str, k: int = 3) -> dict:
        from make_qa import PROMPTS
        from rag import Retriever
        if self.r is None:
            self.r = Retriever()
        t0 = time.time()
        hits = self.r.search(question, k)
        t1 = time.time()
        hi = bool(DEV.search(question))
        body = "\n\n".join(f"[{i + 1}] {' '.join(h['text'].split()[:120])}" for i, h in enumerate(hits))
        q = f"{body}\n\n{'प्रश्न' if hi else 'Question'}: {question.strip()}"
        prompt = PROMPTS["rc-hi" if hi else "rc-en"].format(q=q)
        out = self.om.engine.generate(prompt, 40, 0.0, 1, 0, mode="continue")
        return {"answer": out["completion"].strip(), "sources": [h["title"] for h in hits],
                "retrieve_s": round(t1 - t0, 3), "read_s": round(time.time() - t1, 3)}


class Intent:
    """decide.Decide's output, from the model's own likelihood of each label
    (length-normalised), softmaxed over the label set."""

    def __init__(self, om: Omni, task: str, threshold: float):
        from decide import labels
        self.om, self.threshold = om, threshold
        self.lang, self.template = INTENT_PROMPT[task]
        self.labels = labels(INTENT_TASK[task])

    @torch.no_grad()
    def __call__(self, message: str, top: int = 3) -> dict:
        from make_qa import PROMPTS
        tok, m, dev = self.om.tok, self.om.model, self.om.device
        p = tok.encode(PROMPTS[self.lang].format(q=self.template.format(m=message.strip())))
        lab = [tok.encode(" " + l) + [tok.eos_id] for l in self.labels]
        S = len(p) + max(len(l) for l in lab)
        x = torch.zeros(len(lab), S, dtype=torch.long, device=dev)
        for i, l in enumerate(lab):
            x[i, : len(p) + len(l)] = torch.tensor(p + l, device=dev)
        logits, _ = m(x, x)
        lp = F.log_softmax(logits.float(), -1)
        score = torch.stack([lp[i, torch.arange(len(p) - 1, len(p) - 1 + len(l), device=dev),
                                torch.tensor(l, device=dev)].mean() for i, l in enumerate(lab)])
        prob = F.softmax(score * 10, 0)        # mean log-probs are close together; sharpen
        k = prob.argsort(descending=True)[:top]
        conf = float(prob[k[0]])
        return {"choice": self.labels[int(k[0])],
                "probabilities": {self.labels[int(i)]: round(float(prob[i]), 4) for i in k},
                "confidence": round(conf, 4),
                "action": "auto-route" if conf >= self.threshold else "send to human"}
