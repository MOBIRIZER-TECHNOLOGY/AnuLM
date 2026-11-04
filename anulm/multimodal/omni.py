"""
The one model (omni_train.py) behind every tab of app_omni.py, one copy of it.

Each separate model came with its own loader, and each loader built its own
decoder. The joint model has one decoder, so here the chat engine loads it
once and the ear, the eye, the reader and the intent classifier are views of
that same module: a projector, a prompt, or a scoring rule on top.

    from anulm.multimodal.omni import Omni
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
        from anulm.model import load_checkpoint
        from anulm.inference.serve import Engine
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
            from .audio_codec import WhisperEncoder
            from anulm.hf_tok import load_tokenizer
            from .speech_encoder import ContinuousSpeech
            from .speech_vocab import vocab_of
            from anulm.inference.voice import OwnASR
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
            from anulm.hf_tok import load_tokenizer
            from .speech_vocab import vocab_of
            from .vision_encoder import VisionLanguage, VisionTower
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

    def read(self, question: str, hits: list[dict]) -> str:
        """One greedy answer from `hits`, numbered in the order given."""
        from anulm.data.make_qa import PROMPTS
        hi = bool(DEV.search(question))
        body = "\n\n".join(f"[{i + 1}] {' '.join(h['text'].split()[:120])}" for i, h in enumerate(hits))
        q = f"{body}\n\n{'प्रश्न' if hi else 'Question'}: {question.strip()}"
        prompt = PROMPTS["rc-hi" if hi else "rc-en"].format(q=q)
        return self.om.engine.generate(prompt, 40, 0.0, 1, 0, mode="continue")["completion"].strip()

    def ask(self, question: str, k: int = 3) -> dict:
        """Retrieve `k` passages and read the answer. An answer that only
        repeats the question ("Hamlet" for "Who wrote Hamlet?") or says
        not-found is read again with the passages rotated, up to k reads: on
        the 20 demo questions the answer was in the three passages every time,
        and these were the reader's commonest misses. The answer is "" when no
        read found one (callers then fall back, as for rc_pointer.PointerQA),
        and `sources` starts with the passage the answer was taken from."""
        from anulm.inference.rag import NOT_FOUND, Retriever, cite, echoes
        if self.r is None:
            self.r = Retriever(device=self.om.device)       # e5 on the model's device, not always CUDA
        t0 = time.time()
        hits = self.r.search(question, k)
        t1 = time.time()
        ans = ""
        for turn in range(len(hits)):
            a = self.read(question, hits[turn:] + hits[:turn])
            if a not in NOT_FOUND and not echoes(a, question):
                ans = a
                break
        titles = [h["title"] for h in hits]
        src = cite(ans, hits) if ans else (titles[0] if titles else "")
        return {"answer": ans, "sources": [src] + [t for t in titles if t != src] if titles else [],
                "retrieve_s": round(t1 - t0, 3), "read_s": round(time.time() - t1, 3)}


class Intent:
    """decide.Decide's output, from the model's own likelihood of each label
    (length-normalised), softmaxed over the label set."""

    def __init__(self, om: Omni, task: str, threshold: float):
        from anulm.tasks.decide import labels
        self.om, self.threshold = om, threshold
        self.lang, self.template = INTENT_PROMPT[task]
        self.labels = labels(INTENT_TASK[task])

    def choices(self, options: list[str] | None) -> list[str]:
        """The labels to score. decide.Decide takes any option list, zero-shot;
        the joint model only ever learned to write its own trained labels after
        this prompt, so a subset of them is honest (the softmax is renormalised
        over that subset) but a label it never saw would get a confident-looking
        probability with nothing measured behind it. That is refused with a
        ValueError, which api.py turns into a 400."""
        if options is None:
            return self.labels
        known = set(self.labels)
        unknown = [o for o in options if o not in known]
        if unknown:
            raise ValueError("custom options are not supported by the joint model: it scores only "
                             f"its trained labels, and {unknown[:3]} are not among them. Pass a subset "
                             "of the trained labels, or none for all of them.")
        return list(dict.fromkeys(options))

    @torch.no_grad()
    def __call__(self, message: str, options: list[str] | None = None, top: int = 3) -> dict:
        """decide.Decide's call signature, so api.py's /v1/decide can call either.
        `options`: see `choices`."""
        from anulm.data.make_qa import PROMPTS
        tok, m, dev = self.om.tok, self.om.model, self.om.device
        labels = self.choices(options)
        p = tok.encode(PROMPTS[self.lang].format(q=self.template.format(m=message.strip())))
        lab = [tok.encode(" " + l) + [tok.eos_id] for l in labels]
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
        return {"choice": labels[int(k[0])],
                "probabilities": {labels[int(i)]: round(float(prob[i]), 4) for i in k},
                "confidence": round(conf, 4),
                "action": "auto-route" if conf >= self.threshold else "send to human"}
