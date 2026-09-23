"""
Voice in, voice out: a local speech front end for any AnuLM checkpoint.

    pip install faster-whisper piper-tts sounddevice
    python voice.py --ckpt toonist/AnuLM-Hindi-QA-400M --wav question.wav
    python voice.py --ckpt toonist/AnuLM-Translate-400M --mic     # push to talk
    python voice.py --say "नमस्ते दुनिया"                          # test the voice alone
    python voice.py --hear clip.wav                               # test the ear alone
    python app_voice.py                                           # the browser version

Three stages, all on this machine, nothing leaves it:

    microphone -> faster-whisper (ASR) -> AnuLM -> Piper (TTS) -> speaker

AnuLM is a text model and stays one: this wraps it rather than changing it.
Whisper decides the language itself, the checkpoint answers in text, and the
voice is picked by script -- any Devanagari in the reply is spoken by a Hindi
voice, anything else by an English one. That rule is also what makes the
translator the best demo here: you speak English and hear Hindi.

**What to expect.** The two speech stages are strong and the 400M model
between them is not, so this mostly gives you a faithful reading of what
AnuLM already says (docs/RESULTS.md §23-25 has the numbers). A voice loop
changes the interface, not the intelligence. Each stage is timed separately
in the reply so you can see where the seconds go; on an RTX 5070 Ti the model
is the cheap part.

Whisper's own Hindi transcription is far better than anything this project
trained, which is worth saying plainly: the ear and the mouth here are other
people's models, and only the sentence in the middle is AnuLM's.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
VOICE_DIR = HERE / "release" / "piper"          # downloaded voices land here
ASR_RATE = 16_000                                # what Whisper wants

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Hub repo for Piper's voices, and the two this uses. Every voice is an .onnx
# plus an .onnx.json beside it; "medium" is the quality tier that sounds good
# and still runs faster than real time on a CPU core.
PIPER_REPO = "rhasspy/piper-voices"
VOICES = {
    "hi": "hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx",
    "en": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def script_of(text: str) -> str:
    """Which voice should read this: Devanagari anywhere means Hindi."""
    return "hi" if _DEVANAGARI.search(text) else "en"


def to_mono16k(rate: int, data: np.ndarray) -> np.ndarray:
    """Whatever the browser or the sound card handed over -> float32 mono at
    16 kHz, which is the only shape Whisper accepts as an array."""
    x = np.asarray(data)
    # Scale before mixing: averaging two int16 channels returns float64, and
    # checking the dtype after that silently skipped the /32767 and handed
    # Whisper audio a thousand times too loud.
    if np.issubdtype(x.dtype, np.integer):       # int16 -> [-1, 1]
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32, copy=False)
    if x.ndim > 1:                               # stereo -> mono
        x = x.mean(axis=1)
    if rate != ASR_RATE and x.size:
        # Linear resampling. Whisper mel-bins the audio anyway, so the
        # difference from a windowed-sinc resampler is inaudible here and it
        # saves a scipy dependency.
        n = int(round(x.size * ASR_RATE / rate))
        x = np.interp(np.linspace(0, x.size - 1, n, dtype=np.float32),
                      np.arange(x.size, dtype=np.float32), x).astype(np.float32)
    return x


class ASR:
    """faster-whisper, loaded once. `small` is the sweet spot for Hindi on a
    consumer card; `base` is quicker and noticeably worse at Devanagari."""

    def __init__(self, size: str = "small", device: str = "auto", compute_type: str | None = None):
        from faster_whisper import WhisperModel
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type is None:
            compute_type = "float16" if device.startswith("cuda") else "int8"
        self.device, self.size = device, size
        self.model = WhisperModel(size, device=device, compute_type=compute_type)

    def hear(self, audio, language: str | None = None) -> dict:
        """A path, or (rate, samples) as Gradio and sounddevice give it."""
        if isinstance(audio, tuple):
            audio = to_mono16k(*audio)
        elif isinstance(audio, (str, Path)):
            audio = str(audio)
        t0 = time.time()
        # vad_filter drops the silence either side of a push-to-talk clip,
        # which is most of it and which Whisper otherwise hallucinates into.
        segments, info = self.model.transcribe(audio, language=language, beam_size=5,
                                               vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        return {"text": text, "language": info.language,
                "seconds": round(time.time() - t0, 2)}


class TTS:
    """Piper: one small ONNX voice per language, held open, CPU-only (the
    onnxruntime wheel on PyPI has no CUDA provider, and medium voices
    synthesise faster than real time on a core anyway)."""

    def __init__(self, voices: dict[str, str] | None = None, voice_dir: Path = VOICE_DIR):
        self.paths = voices or VOICES
        self.dir = Path(voice_dir)
        self.loaded: dict[str, object] = {}

    def _fetch(self, lang: str) -> tuple[Path, Path]:
        from huggingface_hub import hf_hub_download
        rel = self.paths[lang]
        self.dir.mkdir(parents=True, exist_ok=True)
        onnx = hf_hub_download(PIPER_REPO, rel, local_dir=self.dir)
        cfg = hf_hub_download(PIPER_REPO, rel + ".json", local_dir=self.dir)
        return Path(onnx), Path(cfg)

    def voice(self, lang: str):
        if lang not in self.loaded:
            from piper import PiperVoice
            onnx, cfg = self._fetch(lang)
            self.loaded[lang] = PiperVoice.load(onnx, cfg)
        return self.loaded[lang]

    def say(self, text: str, lang: str | None = None) -> tuple[int, np.ndarray]:
        """-> (sample rate, int16 samples), the shape Gradio and wave want."""
        lang = lang or script_of(text)
        v = self.voice(lang)
        chunks = list(v.synthesize(text))
        if not chunks:                           # punctuation only, or empty
            return 22_050, np.zeros(0, dtype=np.int16)
        return chunks[0].sample_rate, np.concatenate([c.audio_int16_array for c in chunks])


def write_wav(path: str | Path, rate: int, samples: np.ndarray) -> Path:
    path = Path(path)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(samples.astype(np.int16).tobytes())
    return path


def mode_for(info: dict) -> str:
    """The one generation mode a checkpoint is worth asking out loud, read off
    serve.Engine's own /info flags so this stays right as checkpoints change."""
    if info.get("translate"):
        return "translate"
    if info.get("qa_template"):
        return "question"
    return "continue"


def speakable(text: str, max_chars: int = 400) -> str:
    """What to hand the voice. A tuned checkpoint stops at EOS and needs none
    of this; a base model runs on until it hits the token budget, and reading
    400 tokens of invented encyclopaedia out loud is a minute of nobody's
    time. Cut at a sentence end if there is one in reach."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind("।"), cut.rfind("."), cut.rfind("?"), cut.rfind("!"))
    return (cut[:end + 1] if end > max_chars // 3 else cut).strip()


@dataclass
class Reply:
    heard: str
    language: str
    said: str
    rate: int
    audio: np.ndarray
    mode: str
    timings: dict

    def summary(self) -> str:
        t = self.timings
        return (f"heard ({self.language}, {t['hear']}s): {self.heard}\n"
                f"said  ({self.mode}, {t['think']}s think, {t['say']}s speak): {self.said}")


class Voice:
    """ASR + an AnuLM Engine + TTS. The Engine is `serve.Engine`, unchanged --
    everything the text demo can do is reachable from here."""

    def __init__(self, engine, asr: ASR | None = None, tts: TTS | None = None,
                 max_say_chars: int = 400):
        self.engine = engine
        self.asr = asr or ASR()
        self.tts = tts or TTS()
        self.max_say_chars = max_say_chars

    def reply(self, audio, max_tokens: int = 120, temperature: float = 0.3,
              top_k: int = 40, seed: int = 0, mode: str | None = None) -> Reply:
        h = self.asr.hear(audio)
        if not h["text"]:
            return Reply("", h["language"], "", 22_050, np.zeros(0, np.int16), "-",
                         {"hear": h["seconds"], "think": 0.0, "say": 0.0})
        mode = mode or mode_for(self.engine.info)
        out = self.engine.generate(h["text"], max_tokens, temperature, top_k, seed, mode=mode)
        said = speakable(out["completion"], self.max_say_chars)
        t0 = time.time()
        rate, samples = self.tts.say(said) if said else (22_050, np.zeros(0, np.int16))
        return Reply(h["text"], h["language"], said, rate, samples, out["mode"],
                     {"hear": h["seconds"], "think": out["seconds"],
                      "say": round(time.time() - t0, 2)})


def load_engine(source: str, device: str | None = None):
    """A .pt, an exported folder, or a Hub repo id -- the same three things
    app.py's picker accepts."""
    import os

    import torch

    from serve import Engine
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = source
    if "/" in source and not os.path.exists(source):
        from huggingface_hub import snapshot_download
        path = snapshot_download(source)
    engine = Engine(path, device)
    engine.info["ckpt"] = source
    return engine


def record(seconds: float | None = None, rate: int = ASR_RATE) -> tuple[int, np.ndarray]:
    """Push to talk: record until Enter, or for a fixed number of seconds."""
    import sounddevice as sd
    if seconds:
        print(f"recording {seconds}s...", flush=True)
        buf = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32")
        sd.wait()
        return rate, buf[:, 0]
    frames: list[np.ndarray] = []
    with sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                        callback=lambda indata, *_: frames.append(indata.copy())):
        input("recording -- press Enter to stop: ")
    return rate, (np.concatenate(frames)[:, 0] if frames else np.zeros(0, np.float32))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="toonist/AnuLM-Hindi-QA-400M",
                   help="a .pt, an exported folder, or a Hub repo id")
    p.add_argument("--wav", help="a wav/mp3/flac to answer instead of the microphone")
    p.add_argument("--mic", action="store_true", help="record from the microphone")
    p.add_argument("--seconds", type=float, help="with --mic: record this long instead of until Enter")
    p.add_argument("--out", default="reply.wav", help="where to write the spoken reply")
    p.add_argument("--play", action="store_true", help="play the reply as well as writing it")
    p.add_argument("--say", help="synthesise this text and exit (no model loaded)")
    p.add_argument("--hear", help="transcribe this file and exit (no model loaded)")
    p.add_argument("--whisper", default="small", help="tiny/base/small/medium/large-v3")
    p.add_argument("--mode", choices=["question", "translate", "continue"],
                   help="override the mode picked from the checkpoint")
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-say-chars", type=int, default=400)
    args = p.parse_args()

    if args.say:                                 # the mouth on its own
        rate, samples = TTS().say(args.say)
        print(f"{write_wav(args.out, rate, samples)}  ({samples.size / rate:.1f}s)")
        if args.play:
            import sounddevice as sd
            sd.play(samples, rate)
            sd.wait()
        return

    if args.hear:                                # the ear on its own
        h = ASR(args.whisper).hear(args.hear)
        print(f"[{h['language']}, {h['seconds']}s] {h['text']}")
        return

    if not args.wav and not args.mic:
        p.error("give --wav a file, or --mic to record (or --say/--hear to test one stage)")

    v = Voice(load_engine(args.ckpt), ASR(args.whisper), TTS(), args.max_say_chars)
    audio = record(args.seconds) if args.mic else args.wav
    r = v.reply(audio, args.max_tokens, args.temperature, args.top_k, args.seed, args.mode)
    print(r.summary())
    if r.audio.size:
        print(f"-> {write_wav(args.out, r.rate, r.audio)}")
        if args.play:
            import sounddevice as sd
            sd.play(r.audio, r.rate)
            sd.wait()


if __name__ == "__main__":
    main()
