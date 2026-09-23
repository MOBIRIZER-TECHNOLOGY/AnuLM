"""
Speech corpora, fetched and unpacked.

    python fetch_speech.py --split dev-clean          # 5.4 h, 337 MB -- start here
    python fetch_speech.py --split train-clean-100    # 100.6 h, 6.3 GB
    python fetch_speech.py --list

LibriSpeech (openslr.org/12, CC BY 4.0) is read English, sixteen kilohertz,
already split by speaker, and free of the licence questions that make the
Hindi corpora slower to start with. It is the shakedown corpus: get the
pipeline producing a real word error rate on read English, then argue about
Hindi data.

Everything lands in `data/librispeech/<split>/`, which `.gitignore` already
excludes, and `speech_data.py encode --librispeech` reads that layout
directly.

**On Hindi.** The obvious candidates are IndicVoices (AI4Bharat, ~1,600 h),
Shrutilipi (~6,400 h of mined news audio) and Common Voice. None of them are a
plain HTTPS tarball: Common Voice wants an account and an emailed link,
IndicVoices and Shrutilipi come through Hugging Face with terms to accept. So
this file does not pretend to fetch them. Download one by hand, write a
two-column manifest, and point `speech_data.py encode --manifest` at it --
that path exists precisely because this one cannot cover everything.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

BASE = "https://www.openslr.org/resources/12/"
SPLITS = {
    "dev-clean": (337, 5.4, "the quick one: enough for a real number in an hour"),
    "dev-other": (314, 5.3, "accented and noisier; a harder eval set"),
    "test-clean": (346, 5.4, "hold this back if you want an untouched test set"),
    "train-clean-100": (6387, 100.6, "the first real training run"),
    "train-clean-360": (23049, 363.6, "when 100 hours stops being the limit"),
}
ROOT = Path(__file__).parent / "data" / "librispeech"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


def fetch(split: str, root: Path = ROOT, keep_archive: bool = True) -> Path:
    if split not in SPLITS:
        raise SystemExit(f"unknown split {split!r}; one of {', '.join(SPLITS)}")
    root.mkdir(parents=True, exist_ok=True)
    out = root / "LibriSpeech" / split
    if out.is_dir() and any(out.rglob("*.flac")):
        print(f"{split} already unpacked at {out}")
        return out

    archive = root / f"{split}.tar.gz"
    mb, hours, _ = SPLITS[split]
    if not archive.exists() or archive.stat().st_size < mb * 1_000_000 * 0.95:
        url = BASE + f"{split}.tar.gz"
        print(f"downloading {split} ({mb} MB, {hours} h of audio)")
        t0 = time.time()

        def progress(block, size, total):
            done = block * size
            if total > 0 and block % 200 == 0:
                pct = 100 * done / total
                rate = done / max(time.time() - t0, 1e-6) / 1e6
                print(f"\r  {pct:5.1f}%  {done/1e6:7.0f}/{total/1e6:.0f} MB  "
                      f"{rate:.1f} MB/s", end="", flush=True)

        urllib.request.urlretrieve(url, archive, reporthook=progress)
        print(f"\r  done in {time.time() - t0:.0f}s" + " " * 30)

    print(f"unpacking {archive.name}...")
    with tarfile.open(archive) as tf:
        # LibriSpeech archives are well formed, but a tarball is still an
        # instruction list: refuse any member that would land outside root.
        for m in tf.getmembers():
            target = (root / m.name).resolve()
            if not str(target).startswith(str(root.resolve())):
                raise SystemExit(f"refusing path outside the corpus dir: {m.name}")
        tf.extractall(root)
    if not keep_archive:
        archive.unlink()

    n = sum(1 for _ in out.rglob("*.flac"))
    print(f"{split}: {n} clips at {out}")
    print(f"\nnext:\n  python speech_data.py encode --librispeech {out} "
          f"--out data/speech_{split.replace('-', '_')}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch LibriSpeech.")
    p.add_argument("--split", default="dev-clean")
    p.add_argument("--list", action="store_true", help="show the splits and their sizes")
    p.add_argument("--drop-archive", action="store_true",
                   help="delete the .tar.gz once unpacked")
    args = p.parse_args()

    if args.list:
        print(f"{'split':<18} {'MB':>6} {'hours':>7}  note")
        for name, (mb, hours, note) in SPLITS.items():
            print(f"{name:<18} {mb:>6} {hours:>7.1f}  {note}")
        print(f"\nat 83.3 audio tokens/s, an hour is ~300k tokens; "
              f"dev-clean is ~1.6M, train-clean-100 ~30M")
        return
    fetch(args.split, keep_archive=not args.drop_archive)


if __name__ == "__main__":
    main()
