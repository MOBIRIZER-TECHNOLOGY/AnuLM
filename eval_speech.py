"""
Scoring a speech model without a human in the loop.

    python eval_speech.py tts --ckpt ckpt_tts.pt --texts data/tts_eval.txt
    python eval_speech.py asr --ckpt ckpt_asr.pt --prefix data/speech_ls100 --limit 200

The trick both directions rely on is that Whisper is a much better listener
than this project's model is a speaker, so it can be the judge:

    tts   text -> AnuLM -> SNAC decode -> Whisper -> compare to the text
    asr   audio -> AnuLM -> text       ->            compare to the transcript

Both come out as a word error rate, which is a number that can go in
docs/RESULTS.md beside MBPP and chrF and be argued with. For TTS it is an
*intelligibility* score and nothing more: it says the words survived, not that
the speech sounds good, and a voice can be robotic, toneless and badly paced
while scoring well. Prosody needs ears. Use `--keep` and listen.

Two honest caveats about the TTS number:

  * Whisper is generous. It has a language model inside it and will repair a
    mangled word from context, so a low WER flatters the speaker. Comparing two
    checkpoints with the same judge is fair; comparing this WER to a published
    TTS system's is not.
  * Whisper's own errors are in the score. The floor is not zero: this file
    prints the WER of the *reference* audio through the same judge when it can,
    and that is the number to measure against.

Generation is unconstrained on purpose. A trained model should learn to stay
inside the audio block after `<|speech|>`, and how often it escapes is a
diagnostic worth seeing (reported as `stray`), not something to hide by masking
the text logits. Masking them is the obvious next move once the failure mode is
understood -- it would make a weak checkpoint sound better than it is.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from audio_codec import AUDIO_VOCAB, FRAME_SLOTS, SnacCodec
from speech_vocab import vocab_of

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def normalise(s: str) -> str:
    """Lowercase, strip punctuation, collapse space. WER is meant to score
    words, not whether the model guessed a comma.

    Devanagari needs the explicit mark categories. A matra or a virama is not
    `isalnum()`, so the obvious one-liner silently deletes them and turns
    नमस्ते into नमसत -- every Hindi word collapses towards its bare consonants,
    distinct words become equal, and the WER comes out flattering and
    meaningless. Mn is a non-spacing mark (ि, ्), Mc a spacing one (ा, ी).
    """
    import unicodedata
    keep = [c for c in s.lower()
            if c.isalnum() or c.isspace() or unicodedata.category(c) in ("Mn", "Mc")]
    return " ".join("".join(keep).replace("।", " ").split())


def wer(ref: str, hyp: str) -> tuple[int, int]:
    """(edits, reference length) by Levenshtein over words, so a corpus total
    is a sum of edits over a sum of lengths rather than a mean of ratios."""
    r, h = normalise(ref).split(), normalise(hyp).split()
    if not r:
        return (len(h), 0)
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[len(h)], len(r)


class SpeechModel:
    """A grown checkpoint, loaded for generation in either direction."""

    def __init__(self, ckpt: str, device: str | None = None):
        from dataclasses import replace

        from bpe import BPE
        from model import AnuLM, load_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = load_checkpoint(ckpt, self.device)
        cfg = replace(ck["cfg"], moe_impl="grouped" if self.device.startswith("cuda") else "sparse")
        self.cfg = cfg
        self.vocab = vocab_of(ck)
        self.tok = BPE.load(cfg.tokenizer_path)
        self.model = AnuLM(cfg).to(self.device).eval()
        self.model.load_state_dict(ck["model"])

    def _generate(self, ids: list[int], max_tokens: int, eos: int | None,
                  temperature: float, top_k: int) -> list[int]:
        ids = ids[-(self.cfg.block_size - 1):]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
            out = self.model.generate(x, max_tokens, temperature=max(temperature, 1e-3),
                                      top_k=top_k or None, eos_id=eos)
        return out[0, len(ids):].tolist()

    def speak(self, text: str, max_seconds: float = 12.0,
              temperature: float = 0.7, top_k: int = 50) -> tuple[list[int], int]:
        """text -> generated ids. Also returns how many were not audio."""
        v = self.vocab
        prompt = [v.text_bos] + self.tok.encode(" " + text.strip()) + [v.speech_bos]
        budget = int(max_seconds * 83.3 / FRAME_SLOTS) * FRAME_SLOTS
        new = self._generate(prompt, budget, v.speech_eos, temperature, top_k)
        audio = [i - v.audio_base for i in new if v.audio_base <= i < v.audio_base + AUDIO_VOCAB]
        return audio, len(new) - len(audio)

    def transcribe(self, audio_ids, max_tokens: int = 64) -> str:
        """SNAC flat ids -> text the model thinks it heard."""
        v = self.vocab
        prompt = ([v.speech_bos] + [int(i) + v.audio_base for i in audio_ids] + [v.text_bos])
        new = self._generate(prompt, max_tokens, self.tok.eos_id, 0.0, 0)
        text = self.tok.decode([i for i in new if i < v.text])
        return text.strip().split("\n")[0]


def eval_tts(ckpt: str, texts: list[str], judge: str = "small",
             keep: str | None = None, temperature: float = 0.7) -> dict:
    from voice import ASR
    sm = SpeechModel(ckpt)
    codec = SnacCodec()
    asr = ASR(judge)
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)
        import soundfile as sf

    edits = length = stray = silent = off_slot = quiet = 0
    levels = []
    t0 = time.time()
    for i, text in enumerate(texts):
        audio, off_block = sm.speak(text, temperature=temperature)
        stray += off_block
        if len(audio) < FRAME_SLOTS:
            silent += 1
            e, n = wer(text, "")
            edits, length = edits + e, length + n
            continue
        rate, wav = codec.decode(np.array(audio))
        off_slot += getattr(codec, "off_slot", 0)
        # Loudness, because cross-entropy does not measure it and a model that
        # hedges towards the mean emits inaudible noise while scoring *better*.
        # Real LibriSpeech averages rms 0.064; below 0.005 nothing is audible
        # and the WER is 100% for a reason that has nothing to do with words.
        rms = float(np.sqrt((wav.astype(np.float64) ** 2).mean())) if wav.size else 0.0
        levels.append(rms)
        if rms < 0.005:
            quiet += 1
        if keep:
            sf.write(str(Path(keep) / f"{i:04d}.wav"), wav, rate)
        heard = asr.hear((rate, wav))["text"]
        e, n = wer(text, heard)
        edits, length = edits + e, length + n
        if i < 3:
            print(f"  [{i}] said     {text}")
            print(f"      heard    {heard or '(nothing)'}  ({wav.size/rate:.1f}s)")
    return {"wer": edits / max(length, 1), "clips": len(texts), "silent": silent,
            "stray_tokens": stray, "off_slot": off_slot, "quiet": quiet,
            "rms": float(np.mean(levels)) if levels else 0.0,
            "seconds": round(time.time() - t0)}


def eval_asr(ckpt: str, prefix: str, limit: int | None = None) -> dict:
    from speech_data import load_clips
    sm = SpeechModel(ckpt)
    rows, blob = load_clips(prefix)
    if limit:
        rows = rows[:limit]
    edits = length = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        ids = blob[r["offset"]: r["offset"] + r["length"]]
        hyp = sm.transcribe(ids)
        e, n = wer(r["text"], hyp)
        edits, length = edits + e, length + n
        if i < 3:
            print(f"  [{i}] reference {r['text']}")
            print(f"      model     {hyp or '(nothing)'}")
    return {"wer": edits / max(length, 1), "clips": len(rows),
            "seconds": round(time.time() - t0)}


def eval_judge_floor(prefix: str, judge: str = "small", limit: int = 50) -> dict:
    """What the judge scores on the *real* audio: the floor any TTS run is
    measured against, because Whisper's own mistakes are in every number."""
    from voice import ASR
    from speech_data import load_clips
    codec = SnacCodec()
    asr = ASR(judge)
    rows, blob = load_clips(prefix)
    edits = length = 0
    for r in rows[:limit]:
        ids = blob[r["offset"]: r["offset"] + r["length"]]
        rate, wav = codec.decode(np.asarray(ids, dtype=np.int64))
        e, n = wer(r["text"], asr.hear((rate, wav))["text"])
        edits, length = edits + e, length + n
    return {"wer": edits / max(length, 1), "clips": min(limit, len(rows))}


def main() -> None:
    p = argparse.ArgumentParser(description="Word error rate for AnuLM's speech modes.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tts", help="text -> speech, judged by Whisper")
    t.add_argument("--ckpt", required=True)
    t.add_argument("--texts", help="a file of one sentence per line")
    t.add_argument("--prefix", help="use an encoded corpus's transcripts instead")
    t.add_argument("--limit", type=int, default=50)
    t.add_argument("--judge", default="small")
    t.add_argument("--temperature", type=float, default=0.7)
    t.add_argument("--keep", help="write the generated wavs here and listen to them")

    a = sub.add_parser("asr", help="speech -> text")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--prefix", required=True)
    a.add_argument("--limit", type=int, default=200)

    f = sub.add_parser("floor", help="the judge's WER on the reference audio")
    f.add_argument("--prefix", required=True)
    f.add_argument("--judge", default="small")
    f.add_argument("--limit", type=int, default=50)

    args = p.parse_args()

    if args.cmd == "tts":
        if args.texts:
            texts = [l.strip() for l in Path(args.texts).read_text(encoding="utf-8").splitlines()
                     if l.strip()][:args.limit]
        elif args.prefix:
            from speech_data import load_clips
            texts = [r["text"] for r in load_clips(args.prefix)[0][:args.limit]]
        else:
            p.error("tts needs --texts or --prefix")
        r = eval_tts(args.ckpt, texts, args.judge, args.keep, args.temperature)
        print(f"\nTTS  WER {r['wer']*100:.1f}%  over {r['clips']} clips  "
              f"({r['silent']} silent, {r['stray_tokens']} stray non-audio tokens, "
              f"{r['off_slot']} codes out of slot, {r['seconds']}s)")
        print(f"     loudness: mean rms {r['rms']:.4f}, {r['quiet']} of {r['clips']} clips "
              f"inaudible (<0.005); real speech averages 0.064")
    elif args.cmd == "asr":
        r = eval_asr(args.ckpt, args.prefix, args.limit)
        print(f"\nASR  WER {r['wer']*100:.1f}%  over {r['clips']} clips ({r['seconds']}s)")
    else:
        r = eval_judge_floor(args.prefix, args.judge, args.limit)
        print(f"\njudge floor  WER {r['wer']*100:.1f}%  on {r['clips']} reference clips "
              f"(codec-decoded); no TTS run should be expected to beat this")


if __name__ == "__main__":
    main()
