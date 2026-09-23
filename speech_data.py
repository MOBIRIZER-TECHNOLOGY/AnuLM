"""
Speech corpora -> token pairs AnuLM can train on.

    python speech_data.py encode --librispeech D:/corpora/LibriSpeech/train-clean-100 \
                                 --out data/speech_ls100
    python speech_data.py encode --manifest clips.tsv --out data/speech_hi
    python speech_data.py stat   --prefix data/speech_ls100

One pass over the audio produces two files, because SNAC encoding is the slow
part and both training tasks want the same tokens:

    data/speech_xx.bin      every clip's SNAC ids, uint16, back to back
    data/speech_xx.jsonl    one row per clip: text, offset, length, seconds

Then a task turns them into (prompt, answer) pairs, which is all
`finetune.py` has ever wanted -- it packs pairs into rows and masks the loss to
the answer, and it does not care whether the ids came from a tokenizer or a
codec:

    tts   <|text|> transcript <|speech|>  ->  audio ... <|/speech|>
    asr   <|speech|> audio <|text|>       ->  transcript ... EOS

So the same corpus trains a mouth and an ear, in the same vocabulary, on the
same backbone, and the direction is set by which control token the prompt ends
with. That is the whole trick, and it is why the audio vocabulary was worth
adding rather than bolting a second model on the side.

**The two tasks are not equally efficient, and the packer shows it.** A TTS row
is about half masked -- short text prompt, long audio answer -- so half its
positions carry loss. An ASR row is ~98% masked: the audio sits in the prompt,
the transcript is a dozen tokens, and the GPU spends almost all of its time on
positions that teach nothing. Measured on the synthetic corpus at block 2048.
So expect ASR to need far more wall clock per unit of learning than TTS, and
consider giving the audio prompt its own (unmasked) language-model loss so the
compute is not simply thrown away. That experiment is not run here.

**Sizing, measured.** Audio costs 83.3 tokens per second (`audio_codec.py`), so
an hour of speech is ~300k tokens and LibriSpeech's train-clean-100 is ~30M --
about 49 minutes per epoch at the 10.2k tok/s this box trains a 398M model at
(docs/RESULTS.md §27). The 2,048-token context holds 24.6 s of audio, so clips
longer than about 20 s are dropped rather than truncated: half an utterance
teaches the model to stop mid-word.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from audio_codec import ASR_RATE, FRAME_SLOTS, SNAC_RATE, SnacCodec, read_wav, resample

MAX_SECONDS = 20.0          # a 2048 context holds 24.6 s; leave room for text
MIN_SECONDS = 0.4

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


# --------------------------------------------------------------------------
# finding clips
# --------------------------------------------------------------------------

def from_librispeech(root: str | Path):
    """LibriSpeech's layout: <spk>/<chapter>/<spk>-<chapter>.trans.txt lists
    `<utt-id> THE TRANSCRIPT IN CAPITALS` beside the .flac files."""
    root = Path(root)
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt, _, text = line.partition(" ")
            audio = trans.parent / f"{utt}.flac"
            if text.strip() and audio.exists():
                # LibriSpeech ships capitals with no punctuation; lowercase it
                # so the model is not asked to learn a shouting register.
                yield audio, text.strip().lower()


def from_manifest(path: str | Path):
    """A tsv or jsonl of your own: `<path>\\t<transcript>`, or
    {"audio": ..., "text": ...}. Relative paths resolve beside the manifest."""
    path = Path(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            audio, text = row.get("audio") or row.get("path"), row.get("text", "")
        else:
            audio, _, text = line.partition("\t")
        if not audio or not text.strip():
            continue
        p = Path(audio)
        yield (p if p.is_absolute() else path.parent / p), text.strip()


def from_parquet(path: str | Path, tmp: str | Path | None = None):
    """Hugging Face audio parquet: one row per clip, the audio inline as
    encoded bytes under `audio`, the transcript under `text`.

    `path` may be one shard, a glob, or a directory -- a real split arrives as
    many shards (train-clean-100 is 14) and asking for them one at a time would
    mean fourteen runs and fourteen manifests to stitch.

    This is how LibriSpeech actually arrives when openslr.org is serving at
    35 kB/s and the Hub is serving at 260. The bytes are decoded straight from
    memory -- soundfile reads a file-like object, so nothing is written to disk
    and there is no tarball to unpack.
    """
    import io

    import pyarrow.parquet as pq
    import soundfile as sf

    import glob as _glob
    p = Path(path)
    shards = (sorted(p.glob("*.parquet")) if p.is_dir()
              else sorted(Path(x) for x in _glob.glob(str(path))) or [p])
    for shard in shards:
        table = pq.ParquetFile(str(shard))
        for batch in table.iter_batches(batch_size=64):
            rows = batch.to_pylist()
            for r in rows:
                audio, text = r.get("audio"), (r.get("text") or "").strip()
                if not audio or not text:
                    continue
                raw = audio.get("bytes") if isinstance(audio, dict) else audio
                if not raw:
                    continue
                # Decoded samples, not a path; encode() accepts both.
                data, sr = sf.read(io.BytesIO(raw), dtype="float32")
                yield (data, sr, str(r.get("id") or "")), text.lower()


def from_directory(root: str | Path):
    """A folder of clips, each with a .txt of the same name beside it."""
    root = Path(root)
    for audio in sorted(p for ext in ("*.wav", "*.flac", "*.mp3")
                        for p in root.rglob(ext)):
        txt = audio.with_suffix(".txt")
        if txt.exists():
            text = txt.read_text(encoding="utf-8").strip()
            if text:
                yield audio, text


# --------------------------------------------------------------------------
# encoding
# --------------------------------------------------------------------------

def encode(clips, out_prefix: str, limit: int | None = None,
           max_seconds: float = MAX_SECONDS) -> dict:
    """Encode every clip once. Resumable only by rerunning: the .bin is written
    straight through, so a killed run leaves a truncated pair of files that the
    next run overwrites."""
    import soundfile as sf
    out = Path(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    clip_dir = out.parent / f"{out.name}_clips"
    codec = SnacCodec()

    kept = skipped = 0
    seconds = 0.0
    t0 = time.time()
    with open(out.with_suffix(".bin"), "wb") as fbin, \
         open(out.with_suffix(".jsonl"), "w", encoding="utf-8") as fjsonl:
        offset = 0
        for audio, text in clips:
            if limit and kept >= limit:
                break
            try:
                if isinstance(audio, tuple):
                    # From a parquet shard: samples already in memory. Write a
                    # 16 kHz clip so the continuous path (speech_encoder.py)
                    # has a waveform to re-encode; it cannot read the .bin,
                    # which holds SNAC codes rather than audio.
                    data, sr, uid = audio
                    clip_dir.mkdir(parents=True, exist_ok=True)
                    dest = clip_dir / f"{uid or f'{kept:06d}'}.wav"
                    sf.write(str(dest), resample(data, sr, ASR_RATE), ASR_RATE,
                             subtype="PCM_16")
                    wav = resample(data, sr, SNAC_RATE)
                    audio = dest
                else:
                    wav = read_wav(audio, SNAC_RATE)
            except Exception as e:
                print(f"  skip {getattr(audio, 'name', audio)}: {type(e).__name__}")
                skipped += 1
                continue
            dur = wav.size / SNAC_RATE
            if not (MIN_SECONDS <= dur <= max_seconds):
                skipped += 1
                continue
            ids = codec.encode(wav)
            fbin.write(ids.astype(np.uint16).tobytes())
            # The path is kept because the continuous path (speech_encoder.py)
            # needs the waveform again: Whisper states are 153 kB per second
            # against SNAC's 167 bytes, so they are recomputed on the fly
            # rather than cached, and that needs the original file.
            fjsonl.write(json.dumps({"text": text, "audio": str(Path(audio).resolve()),
                                     "offset": offset, "length": int(ids.size),
                                     "seconds": round(dur, 2)},
                                    ensure_ascii=False) + "\n")
            offset += int(ids.size)
            kept += 1
            seconds += dur
            if kept % 200 == 0:
                rate = seconds / (time.time() - t0)
                print(f"\r  {kept} clips, {seconds/3600:.2f} h audio, "
                      f"{offset/1e6:.1f}M tokens ({rate:.0f}x real time)",
                      end="", flush=True)
    print()
    stats = {"clips": kept, "skipped": skipped, "hours": round(seconds / 3600, 3),
             "tokens": offset, "wall_seconds": round(time.time() - t0)}
    print(f"{kept} clips ({skipped} skipped), {stats['hours']} h, "
          f"{offset/1e6:.2f}M audio tokens -> {out}.bin/.jsonl")
    return stats


def load_clips(prefix: str) -> tuple[list[dict], np.ndarray]:
    """The manifest and a memory-mapped view of every clip's tokens."""
    p = Path(prefix)
    rows = [json.loads(l) for l in
            p.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    blob = np.memmap(p.with_suffix(".bin"), dtype=np.uint16, mode="r")
    return rows, blob


# --------------------------------------------------------------------------
# pairs
# --------------------------------------------------------------------------

def make_pairs(prefix: str, task: str, tok, vocab, limit: int | None = None):
    """-> [(prompt ids, answer ids)], ready for finetune.pack.

    The answer carries the closing control token so the model learns to stop;
    without it generation runs to the token budget every time and the last
    frame is always cut mid-word.
    """
    rows, blob = load_clips(prefix)
    if limit:
        rows = rows[:limit]
    pairs = []
    for r in rows:
        audio = [int(i) + vocab.audio_base
                 for i in blob[r["offset"]: r["offset"] + r["length"]]]
        if len(audio) % FRAME_SLOTS:
            continue                                  # never train on a part frame
        text = tok.encode(" " + r["text"].strip())
        if task == "tts":
            prompt = [vocab.text_bos] + text + [vocab.speech_bos]
            answer = audio + [vocab.speech_eos]
        elif task == "asr":
            prompt = [vocab.speech_bos] + audio + [vocab.text_bos]
            answer = text + [tok.eos_id]
        else:
            raise SystemExit(f"unknown task {task!r}; use tts or asr")
        pairs.append((prompt, answer))
    return pairs


def main() -> None:
    p = argparse.ArgumentParser(description="Turn speech into AnuLM tokens.")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="encode a corpus to .bin/.jsonl")
    src = e.add_mutually_exclusive_group(required=True)
    src.add_argument("--librispeech", help="a LibriSpeech split directory")
    src.add_argument("--manifest", help="a tsv/jsonl of path + transcript")
    src.add_argument("--dir", help="a folder of clips with .txt beside them")
    src.add_argument("--parquet", help="a HF parquet shard, a glob, or a directory of them")
    e.add_argument("--out", required=True, help="output prefix, e.g. data/speech_ls100")
    e.add_argument("--limit", type=int, help="stop after this many clips")
    e.add_argument("--max-seconds", type=float, default=MAX_SECONDS)

    s = sub.add_parser("stat", help="summarise an encoded corpus")
    s.add_argument("--prefix", required=True)

    args = p.parse_args()

    if args.cmd == "encode":
        clips = (from_librispeech(args.librispeech) if args.librispeech
                 else from_manifest(args.manifest) if args.manifest
                 else from_parquet(args.parquet) if args.parquet
                 else from_directory(args.dir))
        encode(clips, args.out, args.limit, args.max_seconds)
    else:
        rows, blob = load_clips(args.prefix)
        hours = sum(r["seconds"] for r in rows) / 3600
        toks = sum(r["length"] for r in rows)
        print(f"{len(rows)} clips, {hours:.2f} h, {toks/1e6:.2f}M audio tokens")
        print(f"median {np.median([r['seconds'] for r in rows]):.1f}s, "
              f"longest {max(r['seconds'] for r in rows):.1f}s")
        print(f"one epoch at 10.2k tok/s ~ {toks/10_200/60:.0f} min (audio alone)")


if __name__ == "__main__":
    main()
