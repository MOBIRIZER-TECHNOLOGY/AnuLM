"""
Every text and speech set the joint model (omni_train.py) trains and is scored
on, fetched from its own public source and turned into the exact files the
existing builders and trainers read.

    python omni_data.py list               # every source: licence, size, outputs, present/missing
    python omni_data.py fetch chat read    # or `fetch all`; a source whose outputs exist is skipped
    python omni_data.py check              # outputs exist, row counts as expected; exit 1 if not

Then, as omni_train.py's docstring has it, with these paths:

    --task chat=text:data/chat_train.jsonl --held chat=data/chat_heldout.jsonl
    --task read=text:data/rc_train.jsonl::4 --held read=data/rc_heldout.jsonl
    --task asr_en=asr:data/speech_ls100 --task asr_hi=asr:data/omni/hindi_train
    --task intent=text:data/omni/intent_train.jsonl

Nothing here re-implements a builder. Each source downloads into the raw layout
that the existing script already expects and then calls that script:
make_chat.py, make_qa.py and fetch_hindi.py (chat), rag.py and make_rc.py
(reading), fetch_speech.py (LibriSpeech), experiments/speech_v5.py's
prep_kathbath / prep_mucs (Kathbath, MUCS). The few files no script wrote
reproducibly are rebuilt here, and were checked against the copies on the
training box (2026-10-29):

  * dolly_train / dolly_heldout: Dolly-15k as {question, answer, lang, category},
    the context appended under the instruction, Random(0) shuffle, first 300
    held out. Row-for-row identical to the files chat v3 was tuned on.
  * fleurs_hi/*.jsonl: identical to the files behind the 36.5% Hindi WER.
  * massive_<lang>/*.jsonl: byte-identical (the Hub's parquet of
    AmazonScience/massive, written as pandas would).
  * banking77/*: byte-identical (PolyAI's GitHub).
  * speech_ls100.jsonl: written from the LibriSpeech flac and transcripts
    instead of through `speech_data.py encode`, which SNAC-encodes every clip
    for TTS -- hours of codec work the ASR path never reads. soundfile reads
    flac, so `audio` points straight at the .flac and nothing is converted.
    Row order puts the same 11 chapters first that the original (built from
    the Hub parquet) had first, so the first 2% -- what omni_train.py and
    speech_encoder.py hold out -- is the same 570 clips. Like the original it
    is not quite speaker-disjoint: chapter 1088-134315 straddles the cut
    (54 clips held out, 5 trained on).

Two cannot be reproduced bit for bit, and say so: the Hindi Wikipedia QA pairs
come from the *latest* hiwiki dump (fetch_hindi.py), so make_chat.py's 4,000
Hindi pairs are drawn from a newer encyclopedia than chat v3's; and the
reading set's hard negatives depend on the index rag.py builds.

**Left out on purpose.** IndicTTS-Hindi (INCLUDE_INDICTTS below) until its
commercial terms are checked; Monsoon en-IN, which is evaluation only and
carries speakers' personal fields (experiments/speech_v5.py prep_monsoon reads
only audio and text). Images (captions, VQA): `fetch images` runs omni_images.py.

**Disk.** The speech sets dominate: Kathbath is 16.6 GB of parquet plus 16 GB
of 16 kHz clips, MUCS 7.8 GB of tarballs that grow to 28 GB unpacked and cut,
LibriSpeech 6.4 GB, FLEURS 1.6 GB (+1.9 GB of clips). The text sets together
are under 1 GB.

**Gated.** Kathbath only: accept its terms on the Hub and `hf auth login`
first; kathbath_hi() checks and says how if it cannot read the repo.

Licences are those in docs/DATASETS.md and NOTICE. One set of joint weights
inherits every one of them -- read those before training anything meant for
commercial use.
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
sys.path.append(str(ROOT / "experiments"))       # speech_v5, for its Kathbath / MUCS prep

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# IndicTTS-Hindi (SPRINGLab, CC BY 4.0 per its card, derived from IIT Madras's
# Indic TTS database) helped Hindi ASR before (RESULTS 40), but the database's
# own terms for commercial use are still being checked. Off until they are;
# when True, hindi_mix() adds data/speech_hi_tts.jsonl (speaker 0, 5,851
# clips, made by `speech_data.py encode --parquet ... --where gender=0`).
INCLUDE_INDICTTS = False


@dataclass
class Out:
    path: str                    # relative to the repo
    rows: int | None = None      # expected rows (lines; csv data rows; json list length)
    tol: float = 0.0             # allowed relative difference
    fields: tuple = ()           # keys the first row must have


@dataclass
class Source:
    licence: str
    size: str                    # download, then what the build adds on disk
    outs: list[Out] = field(default_factory=list)
    needs: tuple = ()            # sources whose outputs this one reads


QA = ("question", "answer", "lang")
AUDIO = ("audio", "text")
SOURCES: dict[str, Source] = {
    "chat": Source(
        "Dolly-15k CC BY-SA 3.0; oasst2 Apache 2.0; Aya Apache 2.0; Hindi Wikipedia QA CC BY-SA 4.0",
        "~450 MB (oasst2 54 MB, Aya 137 MB, Dolly 13 MB, the 242 MB hiwiki dump streamed)",
        [Out("data/chat_train.jsonl", 23022, 0.0, QA), Out("data/chat_heldout.jsonl", 300, 0.0, QA),
         Out("data/dolly_train.jsonl", 14711, 0.0, QA), Out("data/dolly_heldout.jsonl", 300, 0.0, QA),
         Out("data/qa_lora.jsonl", 17143, 0.3, QA)]),
    "read": Source(
        "SQuAD CC BY-SA 4.0; MLQA CC BY-SA 3.0; XQuAD CC BY-SA 4.0; IndicQA see its card (eval only); "
        "Wikipedia CC BY-SA 4.0",
        "~440 MB (Wikipedia hi + simple 395 MB, the QA sets 42 MB), then rag.py's BM25 index",
        [Out("data/rc_train.jsonl", 59400, 0.0, QA), Out("data/rc_heldout.jsonl", 600, 0.0, QA),
         Out("data/rc_eval_squad_en.jsonl", 500, 0.0, QA), Out("data/rc_eval_mlqa_hi.jsonl", 500, 0.0, QA),
         Out("data/rc_eval_xquad_hi.jsonl", 500, 0.0, QA), Out("data/rc_eval_indicqa_hi.jsonl", 350, 0.0, QA)]),
    "librispeech": Source(
        "CC BY 4.0", "6.4 GB tarball (train-clean-100, 100.6 h); the manifest points at its flac",
        [Out("data/speech_ls100.jsonl", 28538, 0.001, AUDIO)]),
    "fleurs_hi": Source(
        "CC BY 4.0", "1.6 GB (google/fleurs hi_in), +1.9 GB of extracted wav",
        [Out("data/fleurs_hi/train.jsonl", 2120, 0.0, AUDIO), Out("data/fleurs_hi/dev.jsonl", 239, 0.0, AUDIO),
         Out("data/fleurs_hi/test.jsonl", 418, 0.0, AUDIO), Out("data/fleurs_hi/devtest.jsonl", 657, 0.0, AUDIO)]),
    "kathbath_hi": Source(
        "CC BY 4.0; GATED (accept terms on the Hub)", "16.6 GB parquet (34 files), +16 GB of 16 kHz clips",
        [Out("data/kathbath_hi/train.jsonl", 91751, 0.001, AUDIO),
         Out("data/kathbath_hi/test.jsonl", 3151, 0.001, AUDIO)]),
    "mucs_hi": Source(
        "CC BY-SA 4.0", "7.8 GB of tarballs (OpenSLR 104), 28 GB after unpacking and cutting",
        [Out("data/mucs_hi_en/train.jsonl", 52479, 0.001, AUDIO),
         Out("data/mucs_hi_en/test.jsonl", 3099, 0.001, AUDIO)]),
    "hindi_mix": Source(
        "as its parts: Kathbath CC BY 4.0, MUCS CC BY-SA 4.0, FLEURS CC BY 4.0", "none (derived, 40 MB)",
        [Out("data/omni/hindi_train.jsonl", 146350, 0.001, AUDIO)],
        needs=("fleurs_hi", "kathbath_hi", "mucs_hi")),
    "banking77": Source(
        "CC BY 4.0", "1.1 MB (PolyAI's GitHub)",
        [Out("data/banking77/train.csv", 10003, 0.0, ("text", "category")),
         Out("data/banking77/test.csv", 3080, 0.0, ("text", "category")),
         Out("data/banking77/categories.json", 77)]),
    "massive_hi": Source(
        "CC BY 4.0", "2.1 MB (AmazonScience/massive hi-IN and en-US, the Hub's parquet)",
        [Out(f"data/decide_general/massive_{lang}/{split}.jsonl", n, 0.0, ("text", "label_text"))
         for lang in ("hi-IN", "en-US") for split, n in (("train", 11514), ("validation", 2033))]),
    "intent": Source(
        "as its parts: BANKING77 CC BY 4.0, MASSIVE CC BY 4.0", "none (derived, 3 MB)",
        [Out("data/omni/intent_train.jsonl", 21517, 0.0, QA), Out("data/omni/intent_test.jsonl", 5113, 0.0, QA)],
        needs=("banking77", "massive_hi")),
}

UA = "anulm-data/0.1 (educational; local LLM training)"


# --------------------------------------------------------------------- helpers
def write_jsonl(path: Path, rows, **dump) -> None:
    """Written beside the target and renamed into place, so a run killed half
    way leaves no file that `check` or a later `fetch` would take as done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=dump.get("ensure_ascii", False),
                               separators=dump.get("separators")) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def download(url: str, dest: Path, size: int | None = None) -> Path:
    """Plain HTTPS to `dest`, resuming a `.part` with a Range request: the
    MUCS train tarball is 7.3 GB and OpenSLR is not always fast."""
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": UA, **({"Range": f"bytes={have}-"} if have else {})}
    t0, start, last = time.time(), have, have
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        if have and r.status != 206:              # the server ignored the range: start over
            have = start = last = 0
        total = have + int(r.headers.get("Content-Length") or 0)
        print(f"  {url} -> {dest.relative_to(ROOT)} ({total / 1e6:,.0f} MB"
              + (f", resuming at {have / 1e6:,.0f}" if have else "") + ")", flush=True)
        with open(part, "ab" if have else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                have += len(chunk)
                if have - last >= 200 << 20:
                    last = have
                    rate = (have - start) / max(time.time() - t0, 1e-6)
                    print(f"\r    {have / 1e6:8,.0f} / {total / 1e6:,.0f} MB  {rate / 1e6:5.1f} MB/s",
                          end="", flush=True)
    print()
    if size is not None and part.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: got {part.stat().st_size:,} bytes, expected {size:,}; rerun to resume")
    os.replace(part, dest)
    return dest


def hub(repo: str, filename: str, local_dir: Path, revision: str | None = None) -> Path:
    """One file of a Hugging Face dataset repo, kept at `local_dir/filename`
    -- the layout every builder here was written against."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_dir=str(local_dir)))


def run(script: str, *args: str) -> None:
    """A repo script as its own process, exactly as its docstring runs it."""
    cmd = [sys.executable, "-u", script, *args]
    print("  $ python " + " ".join([script, *args]), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def untar_wavs(tar: Path, out: Path) -> int:
    """Every .wav in `tar`, flattened into `out` by basename, as
    experiments/indic_v4.py prep_fleurs did. Nothing else in the archive is
    written, so no member can land outside `out`."""
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar) as t:
        for m in t:
            if m.isfile() and m.name.endswith(".wav"):
                dest = out / Path(m.name).name
                if not dest.exists():
                    dest.write_bytes(t.extractfile(m).read())
                n += 1
    return n


def speech_v5():
    import speech_v5 as sv               # experiments/, on sys.path above; ROOT is its parent's parent
    return sv


# --------------------------------------------------------------------- 1. chat
DOLLY = ("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl")


def dolly_rows(raw: Path) -> tuple[list[dict], list[dict]]:
    """(train, held-out) as data/dolly_*.jsonl hold them: 15,011 rows, the
    context (closed QA, summarisation, extraction) below the instruction."""
    rows = []
    for r in read_jsonl(raw):
        ctx = r["context"].strip()
        rows.append({"question": r["instruction"].strip() + (f"\n\n{ctx}" if ctx else ""),
                     "answer": r["response"].strip(), "lang": "en", "category": r["category"]})
    random.Random(0).shuffle(rows)
    return rows[300:], rows[:300]


def chat() -> None:
    """make_chat.py reads data/dolly_train.jsonl and data/qa_lora.jsonl and
    downloads oasst2 and Aya itself (into data/raw/). qa_lora is make_qa.py's
    Hindi Wikipedia pairs mined from data/hindi_ctx.txt, which is fetch_hindi.py
    streaming the whole hiwiki dump and keeping 30% of articles (seed 1337) --
    193 MB of prose and 17,143 pairs on 2026-09-19; the same commands as
    docs/RESULTS.md section 28:

        python fetch_hindi.py --mb 300 --keep-prob 0.3 --out data/hindi_ctx.txt
        python make_qa.py --data data/hindi_ctx.txt --out data/qa_lora.jsonl \\
                          --heldout data/qa_lora_heldout.jsonl
        python make_chat.py
    """
    tr, he = DATA / "dolly_train.jsonl", DATA / "dolly_heldout.jsonl"
    if not (tr.exists() and he.exists()):
        train, held = dolly_rows(hub(*DOLLY, DATA / "dolly"))
        write_jsonl(he, held)
        write_jsonl(tr, train)
        print(f"  dolly: {len(train)} train, {len(held)} held out")
    if not (DATA / "qa_lora.jsonl").exists():
        if not (DATA / "hindi_ctx.txt").exists():
            run("fetch_hindi.py", "--mb", "300", "--keep-prob", "0.3", "--out", "data/hindi_ctx.txt")
        run("make_qa.py", "--data", "data/hindi_ctx.txt", "--out", "data/qa_lora.jsonl",
            "--heldout", "data/qa_lora_heldout.jsonl")
    run("make_chat.py")


# --------------------------------------------------------------------- 2. read
# (repo, file in repo, revision): make_rc.py reads data/rc/<owner>__<name>/<file>.
# MLQA and IndicQA are loading-script datasets; their files are the Hub's own
# parquet conversion of them (refs/convert/parquet).
RC_FILES = [
    ("rajpurkar/squad", "plain_text/train-00000-of-00001.parquet", None),
    ("rajpurkar/squad", "plain_text/validation-00000-of-00001.parquet", None),
    ("facebook/mlqa", "mlqa-translate-train.hi/train/0000.parquet", "refs/convert/parquet"),
    ("facebook/mlqa", "mlqa.hi.hi/test/0000.parquet", "refs/convert/parquet"),
    ("google/xquad", "xquad.hi/validation-00000-of-00001.parquet", None),
    ("ai4bharat/IndicQA", "indicqa.hi/test/0000.parquet", "refs/convert/parquet"),
]
# rag.py build reads data/rag/wiki/20231101.{simple,hi}/*.parquet
WIKI_FILES = ["20231101.hi/train-00000-of-00002.parquet", "20231101.hi/train-00001-of-00002.parquet",
              "20231101.simple/train-00000-of-00001.parquet"]


def read() -> None:
    """Reading comprehension: three passages (the right one and two hard
    negatives that rag.py's BM25 retrieves for the question) and a question.
    Fetch the QA sets and the two Wikipedias, `rag.py build` the BM25 index
    (932,648 chunks on the training box), then make_rc.py."""
    for repo, fname, rev in RC_FILES:
        hub(repo, fname, DATA / "rc" / repo.replace("/", "__"), rev)
    for fname in WIKI_FILES:
        hub("wikimedia/wikipedia", fname, DATA / "rag" / "wiki")
    rag_dir = DATA / "rag"
    if (rag_dir / "dense_e5s.npy").exists():
        # rag.Retriever switches to dense e5 retrieval when this file exists,
        # and make_rc.py would load it into each of its 6 workers. The
        # negatives make_rc.py documents (and rc_train.jsonl was built with)
        # are BM25's, so stop rather than silently build a different set.
        raise RuntimeError("data/rag/dense_e5s.npy exists: make_rc.py would take dense negatives, not BM25. "
                           "Move it aside for this build and put it back afterwards.")
    if not (rag_dir / "meta.json").exists():
        run("rag.py", "build")
    run("make_rc.py")


# --------------------------------------------------------------------- 3. LibriSpeech
# The chapters the original data/speech_ls100.jsonl opened with (the Hub
# parquet's order, which is the tarball's). Putting them first keeps its
# held-out 570 clips -- and so every English ASR number so far -- comparable.
HELD_CHAPTERS = ("374-180298", "374-180299", "7800-283478", "7800-283492", "7800-283493", "2514-149482",
                 "3240-131231", "3240-131232", "1088-129236", "1088-134318", "1088-134315")


def librispeech_rows(split_dir: Path) -> tuple[list[dict], int]:
    """{text, audio, seconds} per utterance, lowercased as speech_data.py's
    from_librispeech does, kept to speech_data's 0.4-20 s window (the
    original kept 28,538 of 28,539). Chapters in HELD_CHAPTERS order, then
    sorted; utterances in transcript order."""
    import soundfile as sf
    from speech_data import MAX_SECONDS, MIN_SECONDS
    chapters = {t.name[: -len(".trans.txt")]: t for t in split_dir.rglob("*.trans.txt")}
    rest = sorted((c for c in chapters if c not in HELD_CHAPTERS), key=lambda c: tuple(map(int, c.split("-"))))
    rows, skipped = [], 0
    for c in [c for c in HELD_CHAPTERS if c in chapters] + rest:
        for line in chapters[c].read_text(encoding="utf-8").splitlines():
            utt, _, text = line.partition(" ")
            flac = chapters[c].parent / f"{utt}.flac"
            if not text.strip() or not flac.exists():
                skipped += 1
                continue
            info = sf.info(str(flac))
            dur = info.frames / info.samplerate
            if not (MIN_SECONDS <= dur <= MAX_SECONDS):
                skipped += 1
                continue
            rows.append({"text": text.strip().lower(), "audio": str(flac.resolve()), "seconds": round(dur, 2)})
    return rows, skipped


def librispeech() -> None:
    """fetch_speech.py's train-clean-100 (openslr.org/12), then the manifest
    speech_encoder.load_manifest reads. No SNAC .bin: the TTS experiments
    still need `speech_data.py encode --librispeech ...` for that."""
    from fetch_speech import fetch
    split_dir = fetch("train-clean-100")
    rows, skipped = librispeech_rows(split_dir)
    write_jsonl(DATA / "speech_ls100.jsonl", rows)
    print(f"  speech_ls100: {len(rows)} clips ({skipped} skipped), "
          f"{sum(r['seconds'] for r in rows) / 3600:.1f} h")


# --------------------------------------------------------------------- 4. FLEURS
FLEURS_SRC = DATA / "fleurs" / "data" / "hi_in"
FLEURS_FILES = [f"data/hi_in/{s}.tsv" for s in ("train", "dev", "test")] + \
               [f"data/hi_in/audio/{s}.tar.gz" for s in ("train", "dev", "test")]
# Which tsv column became `text`. Train and dev were written from the
# normalised transcription (3); test, later, by indic_v4.prep_fleurs from the
# raw one (2, with punctuation). Kept as they are so the 657-clip WER is scored
# against the same references; eval_speech.wer strips punctuation first anyway.
FLEURS_TEXT_COL = {"train": 3, "dev": 3, "test": 2}


def fleurs_rows(split: str) -> list[dict]:
    col, clips = FLEURS_TEXT_COL[split], DATA / "fleurs_hi" / split
    rows = []
    for line in open(FLEURS_SRC / f"{split}.tsv", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) > col and (clips / p[1]).exists():
            rows.append({"audio": str((clips / p[1]).resolve()), "text": p[col].strip() if col == 3 else p[col]})
    return rows


def fleurs_hi() -> None:
    """google/fleurs hi_in into data/fleurs/ (the Hub layout indic_v4.py
    reads), wavs to data/fleurs_hi/<split>/, one manifest per split and
    devtest = dev + test (239 + 418 = 657 clips, the Hindi WER set)."""
    for f in FLEURS_FILES:
        hub("google/fleurs", f, DATA / "fleurs")
    out = DATA / "fleurs_hi"
    for split in ("train", "dev", "test"):
        n = untar_wavs(FLEURS_SRC / "audio" / f"{split}.tar.gz", out / split)
        rows = fleurs_rows(split)
        write_jsonl(out / f"{split}.jsonl", rows)
        print(f"  fleurs {split}: {len(rows)} rows ({n} wavs)")
    dev = (out / "dev.jsonl").read_text(encoding="utf-8")
    test = (out / "test.jsonl").read_text(encoding="utf-8")
    tmp = out / "devtest.jsonl.tmp"
    tmp.write_text(dev + test, encoding="utf-8", newline="\n")
    os.replace(tmp, out / "devtest.jsonl")


# --------------------------------------------------------------------- 5. Kathbath
KATHBATH = "ai4bharat/Kathbath"
KATHBATH_HELP = """
Kathbath is gated. Once, by hand:
  1. open https://huggingface.co/datasets/ai4bharat/Kathbath while logged in,
     and accept the terms ("Agree and access repository"; access is automatic)
  2. make a read token at https://huggingface.co/settings/tokens
  3. hf auth login            (paste the token)
then rerun: python omni_data.py fetch kathbath_hi
"""


def kathbath_hi() -> None:
    """The 34 Hindi parquet files (32 train, 2 valid; m4a inside) flat into
    data/kathbath_hi/, where speech_v5.prep_kathbath looks, then that prep:
    PyAV decodes to 16 kHz clips in data/kathbath_hi_clips/, 8 processes.
    Measured: 91,751 train clips (134.7 h, 101 speakers) and 3,151 valid
    (4.8 h, 20 speakers -- all 20 also in train) in 89 s."""
    from huggingface_hub import HfApi, hf_hub_download
    try:
        import av  # noqa: F401  -- prep_kathbath decodes the m4a with it
    except ImportError:
        raise RuntimeError("Kathbath's audio is m4a; prep_kathbath needs PyAV (`av`). Install it, then rerun.")
    api = HfApi()
    try:
        api.auth_check(KATHBATH, repo_type="dataset")
    except Exception as e:
        print(KATHBATH_HELP)
        raise RuntimeError(f"no access to {KATHBATH} ({type(e).__name__})") from None
    src, stage = DATA / "kathbath_hi", DATA / "kathbath_hi" / "_hub"
    files = sorted(f for f in api.list_repo_files(KATHBATH, repo_type="dataset")
                   if f.startswith("hindi/") and f.endswith(".parquet"))
    for i, f in enumerate(files, 1):
        dest = src / Path(f).name
        if dest.exists():
            continue
        print(f"  [{i}/{len(files)}] {f}", flush=True)
        p = hf_hub_download(KATHBATH, f, repo_type="dataset", local_dir=str(stage))
        os.replace(p, dest)                       # flat, as prep_kathbath globs it
    print("  " + speech_v5().prep_kathbath())


# --------------------------------------------------------------------- 6. MUCS
MUCS_URL = "https://www.openslr.org/resources/104/"
MUCS_SIZES = {"test": 443_929_204, "train": 7_332_628_101}     # Content-Length, 2026-10-29


def mucs_hi() -> None:
    """MUCS 2021 Hindi-English (OpenSLR 104): Kaldi segments over long
    recordings. Tarballs into data/mucs_hi_en/, then speech_v5.prep_mucs
    unpacks each and cuts utterance clips; code-switched text, Hindi in
    Devanagari, English in Latin. Measured: test 3,099 clips (4.9 h) in 7 s,
    train 52,479 clips (87.1 h) in 40 min."""
    base = DATA / "mucs_hi_en"
    for split in ("test", "train"):
        download(f"{MUCS_URL}Hindi-English_{split}.tar.gz", base / f"Hindi-English_{split}.tar.gz", MUCS_SIZES[split])
        print("  " + speech_v5().prep_mucs(split))


# --------------------------------------------------------------------- 7. the Hindi mix
def hindi_mix() -> None:
    """data/omni/hindi_train.jsonl: Kathbath train + MUCS train + FLEURS train,
    shuffled with Random(0) -- hindi_mix5 without IndicTTS. Held-out sets stay
    their own files (Kathbath test, MUCS test, FLEURS devtest); omni_train.py's
    first-2% split of this one is clip-random, not speaker-disjoint."""
    parts = [("kathbath", DATA / "kathbath_hi" / "train.jsonl"), ("mucs", DATA / "mucs_hi_en" / "train.jsonl"),
             ("fleurs", DATA / "fleurs_hi" / "train.jsonl")]
    if INCLUDE_INDICTTS:
        parts.append(("indictts", DATA / "speech_hi_tts.jsonl"))
    loaded = [(name, read_jsonl(p)) for name, p in parts]
    mix = [r for _, rows in loaded for r in rows]
    random.Random(0).shuffle(mix)
    write_jsonl(DATA / "omni" / "hindi_train.jsonl", mix)
    print("  hindi_train: " + " + ".join(f"{n} {len(r)}" for n, r in loaded) + f" = {len(mix)} clips")


# --------------------------------------------------------------------- 8. BANKING77
BANKING_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"


def banking77() -> None:
    """PolyAI's own CSVs (text,category) and the 77 label names, as
    decide.py load_split / labels read them."""
    for f in ("train.csv", "test.csv", "categories.json"):
        download(BANKING_URL + f, DATA / "banking77" / f)


# --------------------------------------------------------------------- 9. MASSIVE
MASSIVE = "AmazonScience/massive"


def massive_rows(parquet: Path) -> list[dict]:
    """{id, label, text, label_text}: the utterance and its intent, the label
    name taken from the ClassLabel stored in the parquet's own schema."""
    import pyarrow.parquet as pq
    t = pq.read_table(parquet)
    names = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
    return [{"id": r["id"], "label": r["intent"], "text": r["utt"], "label_text": names[r["intent"]]}
            for r in t.select(["id", "intent", "utt"]).to_pylist()]


def massive_hi() -> None:
    """MASSIVE hi-IN (the Hindi Decide task and the intent task) and en-US
    (decide.py's general tasks). AmazonScience/massive is a loading script over
    Amazon's S3 tarball; the Hub's parquet of it reproduces the files on the
    training box byte for byte. Only train and validation: decide.py scores on
    validation."""
    for lang in ("hi-IN", "en-US"):
        for split in ("train", "validation"):
            p = hub(MASSIVE, f"{lang}/{split}/0000.parquet", DATA / "raw" / "massive", "refs/convert/parquet")
            rows = massive_rows(p)
            write_jsonl(DATA / "decide_general" / f"massive_{lang}" / f"{split}.jsonl", rows,
                        ensure_ascii=True, separators=(",", ":"))
            print(f"  massive {lang} {split}: {len(rows)}")


# --------------------------------------------------------------------- 10. intent
INTENT_Q = {"en": "Which banking intent is this? {t}", "hi": "यह किस काम का आदेश है? {t}"}


def intent() -> None:
    """Intent classification as the joint model's text task: {question,
    answer, lang} rows that omni_train.py wraps in make_qa.PROMPTS[lang] like
    any other pair, the answer the label string. BANKING77 in English, MASSIVE
    hi-IN in Hindi; train from each train split, test from BANKING77 test and
    MASSIVE validation (its test split is not on disk; decide.py does the
    same). Both files are shuffled with Random(0) so a first-2% or first-300
    slice holds both tasks, not only the one written first."""
    import decide
    out = {}
    for split, b_split in (("train", "train"), ("test", "test")):
        rows = [{"question": INTENT_Q["en"].format(t=t), "answer": l, "lang": "en"}
                for t, l in decide.load_split(b_split, "banking77")]
        rows += [{"question": INTENT_Q["hi"].format(t=t), "answer": l, "lang": "hi"}
                 for t, l in decide.load_split(b_split, "massive-hi")]
        random.Random(0).shuffle(rows)
        write_jsonl(DATA / "omni" / f"intent_{split}.jsonl", rows)
        out[split] = rows
    print(f"  intent: train {len(out['train'])}, test {len(out['test'])} "
          f"({sum(r['lang'] == 'hi' for r in out['test'])} Hindi)")


# --------------------------------------------------------------------- images
def images() -> None:
    # captions and VQA from commercially usable sources live in omni_images.py (~26 GB)
    run("omni_images.py", "fetch")


# --------------------------------------------------------------------- list / fetch / check
def count(path: Path) -> int:
    if path.suffix == ".csv":
        return sum(1 for _ in csv.DictReader(open(path, encoding="utf-8")))
    if path.suffix == ".json":
        return len(json.load(open(path, encoding="utf-8")))
    with open(path, "rb") as f:
        return sum(1 for l in f if l.strip())


def status(src: Source) -> str:
    have = sum((ROOT / o.path).exists() for o in src.outs)
    return "present" if have == len(src.outs) else "missing" if have == 0 else f"partial {have}/{len(src.outs)}"


def cmd_list() -> int:
    for name, s in SOURCES.items():
        print(f"{name:12s} {status(s):12s} {s.size}\n{'':12s} licence: {s.licence}")
        for o in s.outs:
            mark = "+" if (ROOT / o.path).exists() else "-"
            print(f"{'':14s}{mark} {o.path}" + (f"  ({o.rows:,} rows)" if o.rows else ""))
    print("\n  images       captions + VQA: `python omni_images.py check`; fetched by `fetch images`")
    return 0


def cmd_fetch(names: list[str]) -> int:
    if names == ["all"]:
        names = list(SOURCES)        # images stay their own step: omni_images.py, ~26 GB
    bad = [n for n in names if n not in SOURCES and n != "images"]
    if bad:
        raise SystemExit(f"unknown source(s) {bad}; one of {', '.join(SOURCES)}, images, all")
    order: list[str] = []

    def add(n):
        for d in SOURCES[n].needs if n in SOURCES else ():
            add(d)
        if n not in order:
            order.append(n)
    for n in names:
        add(n)
    failed = []
    for n in order:
        if n in SOURCES and status(SOURCES[n]) == "present":
            print(f"--- {n}: present, skipped")
            continue
        if n in SOURCES and any(d in failed for d in SOURCES[n].needs):
            print(f"--- {n}: skipped, an input failed")
            failed.append(n)
            continue
        print(f"--- {n}", flush=True)
        t0 = time.time()
        try:
            globals()[n]()
            print(f"--- {n}: done in {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"--- {n}: FAILED: {type(e).__name__}: {e}")
            failed.append(n)
    if failed:
        print(f"\nfailed: {', '.join(failed)}")
    return 1 if failed else 0


def cmd_check() -> int:
    """Each output exists, its row count is the expected one (within `tol`),
    its first row has the fields its reader needs, and for a speech manifest a
    sample of 200 rows (every n-th) points at audio that exists: the paths are
    absolute, so a moved checkout breaks them."""
    bad = 0
    for name, s in SOURCES.items():
        for o in s.outs:
            p, msg = ROOT / o.path, ""
            if not p.exists():
                msg = "MISSING"
            else:
                n = count(p)
                if o.rows and abs(n - o.rows) > o.tol * o.rows:
                    msg = f"{n:,} rows, expected {o.rows:,}" + (f" +-{o.tol:.1%}" if o.tol else "")
                elif o.fields:
                    if p.suffix == ".csv":
                        first = next(csv.DictReader(open(p, encoding="utf-8")), {})
                    else:
                        first = json.loads(open(p, encoding="utf-8").readline())
                    if missing := [f for f in o.fields if f not in first]:
                        msg = f"first row lacks {missing}"
                    elif "audio" in o.fields:
                        rows = read_jsonl(p)
                        gone = sum(not Path(r["audio"]).exists() for r in rows[:: max(1, len(rows) // 200)])
                        msg = f"{gone} sampled audio paths do not exist" if gone else ""
                msg = msg or f"ok  {n:,} rows"
            ok = msg.startswith("ok")
            bad += not ok
            print(f"{'  ' if ok else '!!'} {name:12s} {o.path:48s} {msg}")
    print(f"\n{'all outputs look right' if not bad else f'{bad} output(s) missing or wrong'}")
    return 1 if bad else 0


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="every source, its licence, size, outputs and whether they exist")
    f = sub.add_parser("fetch", help="download and build sources (skips any already present)")
    f.add_argument("names", nargs="+", help=f"{', '.join(SOURCES)}, or all")
    sub.add_parser("check", help="verify outputs and row counts; exit 1 if anything is off")
    a = p.parse_args()
    sys.exit({"list": cmd_list, "check": cmd_check}[a.cmd]() if a.cmd != "fetch" else cmd_fetch(a.names))


if __name__ == "__main__":
    main()
