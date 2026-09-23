"""
The two ways audio can enter AnuLM, and the one way it leaves.

    python audio_codec.py --probe clip.wav      # both paths, with shapes and rates

Speech has to become something a transformer can read. There are two answers
and this project is going to measure both rather than argue about them
(docs/RESULTS.md style: run it, print the number):

  * **discrete** -- SNAC turns 24 kHz audio into 3 levels of codebook indices
    at 11.9 / 23.8 / 47.6 Hz. Flattened Orpheus-style that is 7 tokens per
    coarse frame, 83.3 tokens per second, and they go straight into the
    vocabulary beside the BPE tokens. One model, one softmax, and the only
    path that can also *emit* audio.

  * **continuous** -- Whisper's encoder turns the same audio into 50 Hz of
    768-dim states. Nothing is quantised, so nothing is thrown away, and a
    small projector maps it into the model's hidden size. This is the input
    side of how Qwen3-Omni is built (its AuT encoder, our borrowed Whisper),
    and it cannot generate audio: it is an ear, not a mouth.

So the shape that falls out is asymmetric, which is the point: continuous in,
discrete out. Generation needs a vocabulary; understanding does not.

**Measured on an RTX 5070 Ti** (a 3.87 s Hindi clip, this file's --probe):

    SNAC     322 tokens (46 / 92 / 184)   83.3 tok/s   decode 42 ms
    Whisper  194 frames x 768             50.1 fr/s    fp16

    Round-tripping that clip through SNAC and back left it *more* legible to
    Whisper, not less, so the codec is not the quality ceiling here. The 400M
    model in the middle is.

**Neither path imports `transformers`.** Not a style preference: Smart App
Control on the training box blocks scipy's unsigned `_rigid_transform_cy.pyd`,
and `transformers` reaches scipy through its object-detection loss, so every
model class there fails to import. SNAC is pure torch and Whisper arrives via
ctranslate2, so both survive. See `ensure_cuda_dlls` for the other half of
that fight.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SNAC_REPO = "hubertsiuzdak/snac_24khz"
SNAC_RATE = 24_000
LEVEL_VOCAB = 4096          # codes per SNAC codebook
FRAME_SLOTS = 7             # tokens per coarse (11.9 Hz) frame once flattened
AUDIO_VOCAB = FRAME_SLOTS * LEVEL_VOCAB      # 28,672 ids to add to the tokenizer

ASR_RATE = 16_000           # Whisper's input rate
WHISPER_DIM = {"tiny": 384, "base": 512, "small": 768, "medium": 1024, "large-v3": 1280}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def ensure_cuda_dlls() -> None:
    """Put torch's bundled CUDA libraries on the DLL search path.

    ctranslate2 links its own cuBLAS and looks for `cublas64_12.dll` on PATH.
    A torch installed from the cu128 wheel ships exactly that file inside
    torch/lib but does not export the directory, so `Whisper.encode` dies with
    "Library cublas64_12.dll is not found or cannot be loaded" on a machine
    where CUDA plainly works -- and confusingly `transcribe` may still run,
    because it reaches the GPU by a different route. Call this before the
    first ctranslate2 CUDA call. It is a no-op off Windows.
    """
    if os.name != "nt":
        return
    import torch
    lib = Path(torch.__file__).parent / "lib"
    if lib.is_dir():
        os.add_dll_directory(str(lib))


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Linear resampling, which is inaudible once the signal is either mel-binned
    or codec-quantised and saves a scipy dependency we could not import anyway."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if src == dst or x.size == 0:
        return x
    n = int(round(x.size * dst / src))
    return np.interp(np.linspace(0, x.size - 1, n),
                     np.arange(x.size), x).astype(np.float32)


def read_wav(path: str | Path, rate: int) -> np.ndarray:
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32")
    return resample(data, sr, rate)


# --------------------------------------------------------------------------
# discrete: SNAC codes <-> a flat token stream
# --------------------------------------------------------------------------

def flatten(codes: list) -> np.ndarray:
    """SNAC's three ragged levels -> one sequence, 7 tokens per coarse frame.

    The levels run at 1x, 2x and 4x the coarse rate, so a frame is
    interleaved coarse-to-fine and each of the 7 slots gets its own 4096-wide
    block of ids. Slot-specific offsets mean the model never has to infer
    which level a code came from -- position in the frame says it.
    """
    l0, l1, l2 = (np.asarray(c).reshape(-1) for c in codes)
    t = l0.size
    out = np.empty(t * FRAME_SLOTS, dtype=np.int64)
    out[0::7] = l0 + 0 * LEVEL_VOCAB
    out[1::7] = l1[0::2] + 1 * LEVEL_VOCAB
    out[2::7] = l2[0::4] + 2 * LEVEL_VOCAB
    out[3::7] = l2[1::4] + 3 * LEVEL_VOCAB
    out[4::7] = l1[1::2] + 4 * LEVEL_VOCAB
    out[5::7] = l2[2::4] + 5 * LEVEL_VOCAB
    out[6::7] = l2[3::4] + 6 * LEVEL_VOCAB
    return out


def unflatten(ids: np.ndarray, report: bool = False):
    """The inverse. A generated stream is truncated to whole frames first,
    because a model that stops mid-frame would otherwise desynchronise every
    level after it.

    Codes are clamped into their slot's range, and that is not defensive
    paranoia: slot j's ids live in [j*4096, (j+1)*4096), and a *trained* model
    respects that, but an undertrained one emits a slot-0 code at position 3
    and the offset subtraction sends it negative. The negative index reaches
    SNAC's embedding lookup and fails as `CUDA error: device-side assert
    triggered` inside the decoder's noise layer -- several frames later, in a
    function that has nothing to do with the mistake. Clamping keeps the audio
    flowing; `report=True` returns how many codes were out of slot, which is
    the honest measure of how well the model has learned the frame structure.
    """
    ids = np.asarray(ids, dtype=np.int64)
    ids = ids[: ids.size - ids.size % FRAME_SLOTS]
    t = ids.size // FRAME_SLOTS
    if t == 0:
        return ([np.zeros((1, 0), np.int64)] * 3, 0) if report else [np.zeros((1, 0), np.int64)] * 3
    slot = [ids[i::7] - i * LEVEL_VOCAB for i in range(FRAME_SLOTS)]
    off = int(sum(int(((s < 0) | (s >= LEVEL_VOCAB)).sum()) for s in slot))
    slot = [np.clip(s, 0, LEVEL_VOCAB - 1) for s in slot]
    l0 = slot[0]
    l1 = np.empty(t * 2, np.int64); l1[0::2], l1[1::2] = slot[1], slot[4]
    l2 = np.empty(t * 4, np.int64)
    l2[0::4], l2[1::4], l2[2::4], l2[3::4] = slot[2], slot[3], slot[5], slot[6]
    codes = [l0[None, :], l1[None, :], l2[None, :]]
    return (codes, off) if report else codes


class SnacCodec:
    """The mouth, and the cheap ear. Loaded once, kept on the GPU."""

    def __init__(self, device: str | None = None):
        import torch
        from snac import SNAC
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SNAC.from_pretrained(SNAC_REPO).eval().to(self.device)
        self.rate = self.model.sampling_rate

    def encode(self, audio) -> np.ndarray:
        """wav path or float samples at 24 kHz -> flat token ids."""
        if isinstance(audio, (str, Path)):
            audio = read_wav(audio, self.rate)
        x = self.torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
        with self.torch.inference_mode():
            codes = self.model.encode(x[None, None, :].to(self.device))
        return flatten([c.cpu().numpy() for c in codes])

    def decode(self, ids) -> tuple[int, np.ndarray]:
        """flat token ids -> (rate, float samples). Ids outside the audio
        block are dropped, so a model that mixes text and audio still decodes."""
        ids = np.asarray(ids, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < AUDIO_VOCAB)]
        raw, self.off_slot = unflatten(ids, report=True)
        codes = [self.torch.from_numpy(c).to(self.device) for c in raw]
        if codes[0].numel() == 0:
            return self.rate, np.zeros(0, np.float32)
        with self.torch.inference_mode():
            wav = self.model.decode(codes)
        return self.rate, wav.squeeze().float().cpu().numpy()


# --------------------------------------------------------------------------
# continuous: Whisper's encoder, reached without transformers
# --------------------------------------------------------------------------

class WhisperEncoder:
    """Whisper's encoder stack only -- no decoder, no text. 50 Hz of states.

    ctranslate2 runs it as a frozen graph, so gradients stop here: this feeds
    a projector that is trained, while the ear itself stays fixed. That is the
    cheap and usually correct choice at this scale, and it is also the only
    one available through this runtime.
    """

    def __init__(self, size: str = "small", device: str = "auto", compute_type: str | None = None):
        ensure_cuda_dlls()
        import ctranslate2
        from faster_whisper import WhisperModel
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type is None:
            compute_type = "float16" if device.startswith("cuda") else "int8"
        self.ct2 = ctranslate2
        self.size, self.device = size, device
        self.whisper = WhisperModel(size, device=device, compute_type=compute_type)
        self.dim = WHISPER_DIM.get(size, 768)

    def encode(self, audio) -> np.ndarray:
        """wav path or 16 kHz samples -> (frames, dim) float32, ~50 Hz."""
        if isinstance(audio, (str, Path)):
            audio = read_wav(audio, ASR_RATE)
        feats = self.whisper.feature_extractor(np.ascontiguousarray(audio, dtype=np.float32))
        # Whisper's own 30 s window; longer clips are chunked by the caller.
        sv = self.ct2.StorageView.from_array(np.ascontiguousarray(feats[np.newaxis, :, :3000]))
        out = self.whisper.model.encode(sv).to_device(self.ct2.Device.cpu)
        return np.asarray(out)[0].astype(np.float32)


def probe(path: str) -> None:
    """Both paths over one clip, with the numbers that decide the design."""
    dur = read_wav(path, ASR_RATE).size / ASR_RATE
    print(f"{path}: {dur:.2f}s\n")

    codec = SnacCodec()
    ids = codec.encode(read_wav(path, SNAC_RATE))
    frames = ids.size // FRAME_SLOTS
    print(f"  SNAC     {ids.size} tokens ({frames} frames x {FRAME_SLOTS})  "
          f"{ids.size/dur:.1f} tok/s  vocab {AUDIO_VOCAB}")
    rate, wav = codec.decode(ids)
    print(f"           round trip decodes to {wav.size/rate:.2f}s at {rate} Hz")
    back = flatten(unflatten(ids))
    print(f"           flatten/unflatten is exact: {np.array_equal(ids, back)}")

    enc = WhisperEncoder()
    states = enc.encode(read_wav(path, ASR_RATE))
    print(f"\n  Whisper  {states.shape[0]} frames x {states.shape[1]}  "
          f"{states.shape[0]/dur:.1f} fr/s  ({enc.size}, frozen)")

    print(f"\n  context: 2048 positions = {2048/(ids.size/dur):.1f}s discrete, "
          f"{2048/(states.shape[0]/dur):.1f}s continuous (1 frame per position)")


def main() -> None:
    p = argparse.ArgumentParser(description="Audio in and out of AnuLM.")
    p.add_argument("--probe", metavar="WAV", help="measure both paths on a clip")
    p.add_argument("--roundtrip", nargs=2, metavar=("IN", "OUT"),
                   help="encode a wav to SNAC tokens and decode it back")
    args = p.parse_args()
    if args.probe:
        probe(args.probe)
    elif args.roundtrip:
        import soundfile as sf
        codec = SnacCodec()
        ids = codec.encode(read_wav(args.roundtrip[0], SNAC_RATE))
        rate, wav = codec.decode(ids)
        sf.write(args.roundtrip[1], wav, rate)
        print(f"{ids.size} tokens -> {args.roundtrip[1]} ({wav.size/rate:.2f}s)")
    else:
        p.error("give --probe a wav, or --roundtrip IN OUT")


if __name__ == "__main__":
    main()
