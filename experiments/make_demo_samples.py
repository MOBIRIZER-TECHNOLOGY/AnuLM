"""
Build examples/samples/: the demo's one-click sample library.

    python experiments/make_demo_samples.py            # rewrites samples.json and media/

Every sample is a held-out row the one model (experiments/omni_one.py) never
trained on, from a source that allows commercial use and redistribution with
credit. Rows are drawn with a fixed seed from the eligible pool, not picked
for looking good: the library shows the model as it is, misses included.
Each pool is checked against the matching training file, so a row that
also sits in training data is dropped rather than trusted.

Audio is kept as 16 kHz mono 16-bit WAV (every source already is, or is
converted); photos are shrunk to at most 640 px and re-saved as JPEG, which
keeps the folder small enough to ship in the repository.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "examples" / "samples"
MEDIA = OUT / "media"
SEED = 20261104

CC_BY_4 = "CC BY 4.0"
OURS = "MIT (written for this project)"


def rows(path: Path) -> list[dict]:
    if not path.exists():       # a checkout that links only part of data/ to the main one
        alt = Path(os.environ.get("ANULM_DATA", ROOT.parent / "AnuLM" / "data")) / path.relative_to(DATA)
        path = alt if alt.exists() else path
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def wav(src: str, dst: Path) -> float:
    """Copy as 16 kHz mono PCM16; returns seconds. FLEURS is float WAV, the rest PCM16."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:            # none of today's sources need it; linear is enough for speech at 16k
        n = int(round(len(x) * 16000 / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype("float32")
        sr = 16000
    sf.write(dst, x, sr, subtype="PCM_16")
    return round(len(x) / sr, 2)


def jpeg(src: Path, dst: Path) -> None:
    from PIL import Image
    im = Image.open(src).convert("RGB")
    im.thumbnail((640, 640))
    im.save(dst, "JPEG", quality=82, optimize=True)


def short(seconds: float, lo=2.5, hi=9.0) -> bool:
    """Short clips: quick on a CPU and quick to listen to."""
    return lo <= seconds <= hi


def speech_en(rng) -> list[dict]:
    """LibriSpeech dev-clean; the one model trained on train-clean-100 only."""
    import soundfile as sf  # noqa: F401  (fail early if missing)
    train = {Path(r["audio"]).name for r in rows(DATA / "speech_ls100.jsonl")} \
        if (DATA / "speech_ls100.jsonl").exists() else set()
    pool = [r for r in rows(DATA / "speech_devclean.jsonl")
            if short(r["seconds"]) and Path(r["audio"]).name not in train]
    by_speaker = {}
    for r in pool:                         # one clip per speaker: 25 voices, not 25 sentences of one book
        by_speaker.setdefault(Path(r["audio"]).stem.split("-")[0], []).append(r)
    picks = [rng.choice(v) for _, v in sorted(by_speaker.items())]
    rng.shuffle(picks)
    out = []
    for i, r in enumerate(picks[:25], 1):
        sid = f"listen-en-{i:02d}"
        utt = Path(r["audio"]).stem
        secs = wav(r["audio"], MEDIA / f"{sid}.wav")
        out.append({"id": sid, "tab": "listen", "kind": "asr_en", "language": "en",
                    "inputs": {"audio": f"media/{sid}.wav", "lang": "en"}, "seconds": secs,
                    "reference": r["text"], "source": f"LibriSpeech dev-clean, utterance {utt}",
                    "license": CC_BY_4,
                    "attribution": "LibriSpeech ASR corpus (Panayotov, Chen, Povey, Khudanpur, 2015), "
                                   "openslr.org/12, read from LibriVox audiobooks; trimmed to 16 kHz WAV"})
    return out


def speech_hi(rng) -> list[dict]:
    """FLEURS hi_in test and Kathbath Hindi held-out; the model trained on
    FLEURS train, Kathbath train and MUCS train (data/omni/hindi_train.jsonl)."""
    import soundfile as sf
    train = rows(DATA / "omni" / "hindi_train.jsonl")
    seen_audio = {Path(r["audio"]).name for r in train}
    seen_text = {r["text"].strip() for r in train}

    def ok(r):
        return Path(r["audio"]).name not in seen_audio and r["text"].strip() not in seen_text

    fleurs = [r for r in rows(DATA / "fleurs_hi" / "test.jsonl") if ok(r)]
    fleurs = [r for r in fleurs if short(sf.info(r["audio"]).duration)]
    kath = [r for r in rows(DATA / "kathbath_hi" / "test.jsonl") if ok(r) and short(r["seconds"])]
    picks = [("fleurs", r) for r in rng.sample(fleurs, 13)] + [("kathbath", r) for r in rng.sample(kath, 12)]
    out = []
    for i, (src, r) in enumerate(picks, 1):
        sid = f"listen-hi-{i:02d}"
        secs = wav(r["audio"], MEDIA / f"{sid}.wav")
        if src == "fleurs":
            source = f"FLEURS hi_in test, clip {Path(r['audio']).stem}"
            att = ("FLEURS (Conneau et al., 2022), Google, huggingface.co/datasets/google/fleurs; "
                   "converted to 16 kHz 16-bit WAV")
        else:
            source = f"Kathbath Hindi held-out (valid split), clip {Path(r['audio']).stem}"
            att = ("Kathbath (Javed et al., 2023), AI4Bharat, huggingface.co/datasets/ai4bharat/Kathbath; "
                   "16 kHz WAV")
        out.append({"id": sid, "tab": "listen", "kind": "asr_hi", "language": "hi",
                    "inputs": {"audio": f"media/{sid}.wav", "lang": "hi"}, "seconds": secs,
                    "reference": r["text"], "source": source, "license": CC_BY_4, "attribution": att})
    return out


def images(rng) -> list[dict]:
    """DOCCI test (captions) and TextVQA val (questions); the model trained on
    caption_train / vqa_train. Flickr30k and COCO are left out: not for commercial use."""
    base = DATA / "omni"
    cap_train = {r["image"] for r in rows(base / "caption_train.jsonl")}
    vqa_train = {r["image"] for r in rows(base / "vqa_train.jsonl")}
    out = []
    docci = [r for r in rows(base / "caption_test.jsonl")
             if r.get("source") == "docci" and r["image"] not in cap_train]
    for i, r in enumerate(rng.sample(docci, 10), 1):
        sid = f"look-caption-{i:02d}"
        jpeg(base / r["image"], MEDIA / f"{sid}.jpg")
        out.append({"id": sid, "tab": "look", "kind": "caption", "language": "en",
                    "inputs": {"image": f"media/{sid}.jpg", "question": ""},
                    "reference": r["text"], "source": f"DOCCI test, {Path(r['image']).name}",
                    "license": CC_BY_4,
                    "attribution": "DOCCI (Onoe et al., 2024), Google, google.github.io/docci; "
                                   "photo resized to 640 px"})
    authors = {}
    with open(base / "ATTRIBUTION_textvqa.csv", encoding="utf-8") as f:
        for a in csv.DictReader(f):
            if "/licenses/by/2.0" in a["license"]:
                authors[a["image_id"]] = a
    vqa = []
    for r in rows(base / "vqa_test.jsonl"):
        stem = Path(r["image"]).stem
        top, votes = Counter(r["answers"]).most_common(1)[0]
        # A clear reference only: most annotators agree, and the photo's
        # author and licence are on record.
        # TextVQA's "unanswerable" marker is not an answer anyone could check against.
        if (stem in authors and r["image"] not in vqa_train and votes >= 6
                and "does not require" not in top and top != "unanswerable"):
            vqa.append(r)
    seen = set()
    picks = []
    for r in rng.sample(vqa, len(vqa)):
        if r["image"] not in seen:
            seen.add(r["image"])
            picks.append(r)
        if len(picks) == 12:
            break
    for i, r in enumerate(picks, 1):
        sid = f"look-vqa-{i:02d}"
        a = authors[Path(r["image"]).stem]
        jpeg(base / r["image"], MEDIA / f"{sid}.jpg")
        out.append({"id": sid, "tab": "look", "kind": "vqa", "language": "en",
                    "inputs": {"image": f"media/{sid}.jpg", "question": r["question"]},
                    "reference": r["text"], "answers": sorted(set(r["answers"])),
                    "source": f"TextVQA val, image {Path(r['image']).stem} (Open Images)",
                    "license": "question and answers CC BY 4.0; photo CC BY 2.0",
                    "attribution": f"Photo by {a['author']} ({a['url']}), CC BY 2.0, via Open Images; "
                                   f"resized to 640 px. Question from TextVQA (Singh et al., 2019), textvqa.org"})
    return out


CHAT = [  # Written for this library; none of them is in chat_train.jsonl (checked below).
    ("en", "Explain photosynthesis in two sentences."),
    ("en", "Give me three tips for a job interview."),
    ("en", "What is the difference between weather and climate?"),
    ("en", "Write a short, polite email asking a colleague to move our meeting to Friday."),
    ("en", "Why do we see lightning before we hear thunder?"),
    ("hi", "सुबह जल्दी उठने के तीन फायदे बताइए।"),
    ("hi", "बारिश कैसे होती है? सरल शब्दों में समझाइए।"),
    ("hi", "एक छोटी कहानी लिखिए जिसमें एक बिल्ली और एक चूहा दोस्त बनते हैं।"),
    ("hi", "स्वस्थ रहने के लिए हमें क्या खाना चाहिए?"),
    ("hi", "पानी बचाने के चार आसान तरीके बताइए।"),
    ("python", "Write a Python function that reverses a string."),
    ("python", "Write a Python function to check whether a number is prime."),
    ("python", "How do I read a CSV file in Python?"),
    ("python", "Write a Python function that returns the factorial of n."),
    ("python", "What does a list comprehension do in Python? Give an example."),
]


def chat() -> list[dict]:
    train = {r.get("question", r.get("instruction", "")).strip() for r in rows(DATA / "chat_train.jsonl")}
    out = []
    n = Counter()
    for lang, q in CHAT:
        assert q not in train, q
        n[lang] += 1
        sid = f"chat-{lang}-{n[lang]:02d}"
        out.append({"id": sid, "tab": "chat", "kind": "chat", "language": "en" if lang == "python" else lang,
                    "topic": lang, "inputs": {"question": q, "mode": "chat"}, "reference": None,
                    "source": "written for this sample library", "license": OURS,
                    "attribution": "The AnuLM authors"})
    return out


def lookup() -> list[dict]:
    """data/rag_demo_questions.jsonl: five English, five Hindi."""
    qs = rows(DATA / "rag_demo_questions.jsonl")
    en = [r for r in qs if not re.search(r"[ऀ-ॿ]", r["q"])][:5]
    hi = [r for r in qs if re.search(r"[ऀ-ॿ]", r["q"])][:5]
    out = []
    for i, r in enumerate(en + hi, 1):
        lang = "hi" if r in hi else "en"
        out.append({"id": f"lookup-{i:02d}", "tab": "chat", "kind": "lookup", "language": lang,
                    "inputs": {"question": r["q"], "mode": "lookup"}, "reference": r["a"][0], "answers": r["a"],
                    "source": "data/rag_demo_questions.jsonl; answers looked up in Wikipedia",
                    "license": OURS,
                    "attribution": "Questions by the AnuLM authors; the demo retrieves from Wikipedia "
                                   "(CC BY-SA 4.0)"})
    return out


HI_PROMPT = "यह किस काम का आदेश है?"


def intents(rng) -> list[dict]:
    """BANKING77 test and MASSIVE hi-IN held-out (its validation split, the
    2,033 messages the Decide tab's numbers are on)."""
    train = {r["question"] for r in rows(DATA / "omni" / "intent_train.jsonl")}
    with open(DATA / "banking77" / "test.csv", encoding="utf-8") as f:
        bank = [r for r in csv.DictReader(f)]
    rng.shuffle(bank)
    out, cats = [], set()
    for r in bank:
        if r["category"] in cats or len(r["text"]) > 90 or f"Which banking intent is this? {r['text']}" in train:
            continue
        cats.add(r["category"])
        sid = f"decide-bank-{len(cats):02d}"
        out.append({"id": sid, "tab": "decide", "kind": "intent_banking", "language": "en",
                    "inputs": {"message": r["text"], "task": "banking"}, "reference": r["category"],
                    "source": "BANKING77 test", "license": CC_BY_4,
                    "attribution": "BANKING77 (Casanueva et al., 2020), PolyAI, "
                                   "huggingface.co/datasets/PolyAI/banking77"})
        if len(cats) == 10:
            break
    hi = [r for r in rows(DATA / "omni" / "intent_test.jsonl") if r["lang"] == "hi" and r["question"] not in train]
    rng.shuffle(hi)
    cats = set()
    for r in hi:
        if r["answer"] in cats:
            continue
        cats.add(r["answer"])
        msg = r["question"].removeprefix(HI_PROMPT).strip()     # the training prompt, not the message
        sid = f"decide-hi-{len(cats):02d}"
        out.append({"id": sid, "tab": "decide", "kind": "intent_massive_hi", "language": "hi",
                    "inputs": {"message": msg, "task": "massive-hi"}, "reference": r["answer"],
                    "source": "MASSIVE hi-IN validation (held out)", "license": CC_BY_4,
                    "attribution": "MASSIVE (FitzGerald et al., 2022), Amazon Science, "
                                   "huggingface.co/datasets/AmazonScience/massive"})
        if len(cats) == 10:
            break
    return out


SOURCES = [  # (kind prefix, dataset, licence, citation and link)
    ("asr_en", "LibriSpeech ASR corpus, dev-clean", CC_BY_4,
     "V. Panayotov, G. Chen, D. Povey, S. Khudanpur. LibriSpeech: an ASR corpus based on public domain "
     "audio books. ICASSP 2015. https://www.openslr.org/12/"),
    ("asr_hi", "FLEURS, Hindi (hi_in) test", CC_BY_4,
     "A. Conneau et al. FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech. "
     "SLT 2022. https://huggingface.co/datasets/google/fleurs"),
    ("asr_hi", "Kathbath, Hindi held-out (valid) split", CC_BY_4,
     "T. Javed et al. IndicSUPERB: A Speech Processing Universal Performance Benchmark for Indian "
     "Languages. AAAI 2023. AI4Bharat, https://huggingface.co/datasets/ai4bharat/Kathbath"),
    ("caption", "DOCCI test", CC_BY_4,
     "Y. Onoe et al. DOCCI: Descriptions of Connected and Contrasting Images. ECCV 2024. Google, "
     "https://google.github.io/docci/"),
    ("vqa", "TextVQA val (questions, answers)", CC_BY_4 + "; photos CC BY 2.0 from Open Images, authors below",
     "A. Singh et al. Towards VQA Models That Can Read. CVPR 2019. https://textvqa.org; photos via "
     "Open Images, https://storage.googleapis.com/openimages/web/index.html"),
    ("intent_banking", "BANKING77 test", CC_BY_4,
     "I. Casanueva et al. Efficient Intent Detection with Dual Sentence Encoders. NLP4ConvAI 2020. "
     "PolyAI, https://huggingface.co/datasets/PolyAI/banking77"),
    ("intent_massive_hi", "MASSIVE hi-IN validation", CC_BY_4,
     "J. FitzGerald et al. MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset "
     "with 51 Typologically-Diverse Languages. ACL 2023. Amazon Science, "
     "https://huggingface.co/datasets/AmazonScience/massive"),
]


def write_attribution(items: list[dict]) -> None:
    n = Counter(i["kind"] for i in items)
    lines = ["# Sample library: sources, licences, credits", "",
             "Generated by `experiments/make_demo_samples.py` with `samples.json`; edit the script, not this file.",
             "", "Every sample is a held-out row that AnuLM-Omni-400M did not train on. Media were converted "
             "(audio to 16 kHz mono 16-bit WAV, photos resized to at most 640 px and re-saved as JPEG); "
             "the words, transcripts and labels are unchanged. Flickr30k and COCO are deliberately absent: "
             "their terms do not allow commercial use.", "",
             "| samples | dataset | licence | credit |", "|---|---|---|---|"]
    for kind, name, lic, cite in SOURCES:
        count = sum(1 for i in items if i["kind"] == kind and name.split(",")[0].split(" ")[0] in i["source"])
        lines.append(f"| {count} | {name} | {lic} | {cite} |")
    lines.append(f"| {n['chat'] + n['lookup']} | chat prompts and Wikipedia lookup questions | MIT, like the "
                 f"code | written by the AnuLM authors; the lookup answers come from Wikipedia "
                 f"(CC BY-SA 4.0) at run time |")
    lines += ["", "## TextVQA photos (CC BY 2.0)", "",
              "Each photo is by the person named, licensed under "
              "[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), resized for this library.", ""]
    for i in items:
        if i["kind"] == "vqa":
            lines.append(f"- `{i['inputs']['image']}`: {i['attribution'].split(', CC BY 2.0')[0]}")
    lines += ["", "## DOCCI photos (CC BY 4.0)", "",
              "All DOCCI photos were taken by one Google researcher and are released with their "
              "descriptions under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).", ""]
    for i in items:
        if i["kind"] == "caption":
            lines.append(f"- `{i['inputs']['image']}`: {i['source']}")
    (OUT / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if MEDIA.exists():
        shutil.rmtree(MEDIA)
    MEDIA.mkdir(parents=True)
    rng = random.Random(SEED)
    items = speech_en(rng) + speech_hi(rng) + images(rng) + chat() + lookup() + intents(rng)
    manifest = {"version": 1, "seed": SEED, "built_by": "experiments/make_demo_samples.py",
                "note": "Held-out rows the one model never trained on, drawn at random from each pool.",
                "items": items}
    (OUT / "samples.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_attribution(items)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{len(items)} samples, {Counter(i['kind'] for i in items)}, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
