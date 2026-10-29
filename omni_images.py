"""
Images for the joint model, from sources that allow commercial use.

    python omni_images.py fetch            # ~26 GB at the defaults; resumable
    python omni_images.py fetch --vizwiz   # + VizWiz-VQA, another ~15 GB
    python omni_images.py check

The first captioner and VQA model trained on Flickr30k and on A-OKVQA's COCO
photos. Flickr30k is for "non-commercial research and/or educational
purposes" only, and COCO's photos carry their Flickr owners' licences, some of
them non-commercial; a model that is meant to be sold cannot learn from either.
These replace them:

  caption  Localized Narratives (Google, annotations CC BY 4.0) over Open Images
           V7 photos (each CC BY 2.0 per Open Images' own list; rows with any
           other licence are dropped, and every kept photo's author and URL go
           into data/omni/ATTRIBUTION_open_images.csv, which CC BY requires).
           The narratives are transcribed speech ("In this image I can see
           ..."), so the spoken opener is cut and each is trimmed to its first
           two sentences.
           DOCCI (Google, images and descriptions CC BY 4.0, all photos taken by
           one researcher): long descriptions, first two sentences kept.
  vqa      TextVQA (Facebook, CC BY 4.0; Open Images photos, again only those
           licensed CC BY 2.0, authors in ATTRIBUTION_textvqa.csv): questions
           about text in the picture, ten human answers each, the most common
           kept as the target and all ten for scoring.
           VizWiz-VQA (CC BY 4.0; photos and questions from blind users), opt-in.

Output, in the trainer's manifest format (vision_encoder.load_manifest; image
paths relative to the manifest so a clone can move the folder):
  data/omni/caption_train.jsonl  caption_test.jsonl
  data/omni/vqa_train.jsonl      vqa_test.jsonl
Pictures land in data/omni_img/.

Bandwidth decides the schedule: 2 MB/s here, so the defaults are a few hours.
Every download resumes from where it stopped, and finished pieces are skipped.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import random
import re
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).parent
OUT = HERE / "data" / "omni"
IMG = HERE / "data" / "omni_img"
RAW = IMG / "raw"

LN_URL = ("https://storage.googleapis.com/localized-narratives/annotations/"
          "open_images_train_v6_localized_narratives-0000{}-of-00010.jsonl")
OI_INFO = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
OI_IMAGE = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"
OI_ALL = "https://storage.googleapis.com/openimages/2018_04/image_ids_and_rotation.csv"
CC_BY_2 = "https://creativecommons.org/licenses/by/2.0/"
DOCCI_DESC = "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"
DOCCI_IMAGES = "https://storage.googleapis.com/docci/data/docci_images.tar.gz"
TEXTVQA = "https://dl.fbaipublicfiles.com/textvqa/"
VIZWIZ = "https://vizwiz.cs.colorado.edu/VizWiz_final/"


def get(url: str, dest: Path, quiet: bool = False) -> Path:
    """Download with resume: a partial file continues with a Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "AnuLM-omni_images",
                                               **({"Range": f"bytes={have}-"} if have else {})})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if have else "wb") as f:
        total = have + int(r.headers.get("Content-Length") or 0)
        done, last = have, 0.0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if not quiet and time.time() - last > 30:
                last = time.time()
                rate = (done - have) / max(last - t0, 1e-6) / 1e6
                print(f"  {dest.name}: {done/1e9:.2f}/{total/1e9:.2f} GB  {rate:.1f} MB/s", flush=True)
    part.replace(dest)
    return dest


def first_sentences(text: str, n: int = 2, max_words: int = 60) -> str:
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(parts[:n])
    words = out.split()
    return " ".join(words[:max_words]).rstrip(",;") + ("" if len(words) <= max_words else ".")


OPENER = re.compile(r"^(in (this|the) (image|picture|photo)[, ]*(we|i)? *(can )?(see|observe)?[, ]*"
                    r"(there (is|are))?[, ]*)", re.I)


def tidy_narrative(text: str) -> str:
    t = OPENER.sub("", " ".join(text.split())).strip()
    t = first_sentences(t)
    return t[:1].upper() + t[1:] if t else ""


def write(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def rel(p: Path) -> str:
    """Image path as the manifest stores it: relative to data/omni."""
    return Path("..", p.relative_to(HERE / "data")).as_posix()


# ------------------------------------------------------------------ caption
def localized_narratives(n_images: int) -> list[dict]:
    done = OUT / "_ln.jsonl"
    if done.exists():
        return [json.loads(l) for l in open(done, encoding="utf-8")]
    info = get(OI_INFO, RAW / "train-images-boxable-with-rotation.csv")
    lic = {}
    with open(info, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            lic[r["ImageID"]] = (r["License"], r["Author"], r["OriginalLandingURL"])
    caps: dict[str, str] = {}
    for shard in range(10):
        if len(caps) >= n_images:
            break
        path = get(LN_URL.format(shard), RAW / f"ln_train_{shard}.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                iid = r["image_id"]
                if iid in caps or lic.get(iid, ("",))[0] != CC_BY_2:
                    continue                       # one narrative per photo; CC BY 2.0 only
                c = tidy_narrative(r["caption"])
                if len(c.split()) >= 5:
                    caps[iid] = c
                if len(caps) >= n_images:
                    break
    ids = sorted(caps)
    dest = IMG / "open_images"
    dest.mkdir(parents=True, exist_ok=True)

    def one(iid):
        p = dest / f"{iid}.jpg"
        try:
            get(OI_IMAGE.format(iid), p, quiet=True)
            return iid
        except Exception:
            return None
    t0, ok = time.time(), []
    with ThreadPoolExecutor(32) as ex:
        for i, r in enumerate(ex.map(one, ids)):
            if r:
                ok.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  open images: {i+1}/{len(ids)}  {(i+1)/(time.time()-t0):.0f} img/s", flush=True)
    with open(OUT / "ATTRIBUTION_open_images.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "license", "author", "url"])
        for iid in ok:
            w.writerow([iid, *lic[iid]])
    rows = [{"image": rel(dest / f"{iid}.jpg"), "text": caps[iid], "source": "localized_narratives"} for iid in ok]
    write(done, rows)
    return rows


def docci() -> tuple[list[dict], list[dict]]:
    desc = get(DOCCI_DESC, RAW / "docci_descriptions.jsonlines")
    dest = IMG / "docci"
    if not dest.exists() or not any(dest.iterdir()):
        import tarfile
        tar = get(DOCCI_IMAGES, RAW / "docci_images.tar.gz")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as t:
            for m in t.getmembers():
                if m.isfile() and m.name.endswith(".jpg"):
                    m.name = Path(m.name).name
                    t.extract(m, dest)
    train, test = [], []
    for line in open(desc, encoding="utf-8"):
        r = json.loads(line)
        p = dest / r["image_file"]
        if not p.exists():
            continue
        row = {"image": rel(p), "text": first_sentences(r["description"]), "source": "docci"}
        (test if r["split"] == "test" else train).append(row)
    return train, test


# ------------------------------------------------------------------ vqa
def majority(answers: list[str]) -> str:
    return collections.Counter(a.strip().lower() for a in answers).most_common(1)[0][0]


def oi_licences(ids: set[str]) -> dict[str, tuple[str, str, str]]:
    """(licence, author, page) for the given Open Images ids, from the full
    9M-row list (3.3 GB; the boxable list Localized Narratives uses does not
    cover every photo TextVQA asks about)."""
    path = get(OI_ALL, RAW / "image_ids_and_rotation.csv")
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["ImageID"] in ids:
                out[r["ImageID"]] = (r["License"], r["Author"], r["OriginalLandingURL"])
    return out


def textvqa() -> tuple[list[dict], list[dict]]:
    """TextVQA's annotations are CC BY 4.0, but its photos are Open Images
    photos, each under its own licence: only CC BY 2.0 ones are kept, and
    their authors go to ATTRIBUTION_textvqa.csv, as for the captions."""
    dest = IMG / "textvqa"
    if not dest.exists() or not any(dest.iterdir()):
        z = get(TEXTVQA + "images/train_val_images.zip", RAW / "textvqa_train_val_images.zip")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith(".jpg"):
                    (dest / Path(n).name).write_bytes(zf.read(n))
    splits = {s: json.load(open(get(TEXTVQA + f"data/TextVQA_0.5.1_{s}.json", RAW / f"textvqa_{s}.json"),
                                encoding="utf-8"))["data"] for s in ("train", "val")}
    lic = oi_licences({r["image_id"] for d in splits.values() for r in d})
    out, kept = [], set()
    for split, data in splits.items():
        rows = []
        for r in data:
            p = dest / f"{r['image_id']}.jpg"
            if p.exists() and r.get("answers") and lic.get(r["image_id"], ("",))[0] == CC_BY_2:
                kept.add(r["image_id"])
                rows.append({"image": rel(p), "question": r["question"], "text": majority(r["answers"]),
                             "answers": r["answers"], "source": "textvqa"})
        out.append(rows)
    with open(OUT / "ATTRIBUTION_textvqa.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "license", "author", "url"])
        for iid in sorted(kept):
            w.writerow([iid, *lic[iid]])
    print(f"textvqa: {len(kept)} of {len(lic)} photos are CC BY 2.0 and kept", flush=True)
    return out[0], out[1]


def vizwiz() -> tuple[list[dict], list[dict]]:
    ann = get(VIZWIZ + "vqa_data/Annotations.zip", RAW / "vizwiz_annotations.zip")
    out = []
    for split in ("train", "val"):
        dest = IMG / "vizwiz" / split
        if not dest.exists() or not any(dest.iterdir()):
            z = get(VIZWIZ + f"images/{split}.zip", RAW / f"vizwiz_{split}.zip")
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if n.endswith(".jpg"):
                        (dest / Path(n).name).write_bytes(zf.read(n))
        with zipfile.ZipFile(ann) as zf:
            js = json.loads(zf.read(f"{split}.json"))
        rows = []
        for r in js:
            p = dest / r["image"]
            answers = [a["answer"] for a in r.get("answers", [])]
            if p.exists() and answers:
                rows.append({"image": rel(p), "question": r["question"], "text": majority(answers),
                             "answers": answers, "source": "vizwiz"})
        out.append(rows)
    return out[0], out[1]


def fetch(a) -> None:
    rng = random.Random(0)
    ln = localized_narratives(a.ln_images)
    d_train, d_test = docci()
    ln_test, ln_train = ln[:500], ln[500:]
    cap_train = ln_train + d_train
    rng.shuffle(cap_train)
    print(f"caption: train {write(OUT / 'caption_train.jsonl', cap_train)} "
          f"(Localized Narratives {len(ln_train)}, DOCCI {len(d_train)}), "
          f"test {write(OUT / 'caption_test.jsonl', d_test[:500] + ln_test)}", flush=True)
    t_train, t_val = textvqa()
    v_train, v_val = vizwiz() if a.vizwiz else ([], [])
    vqa_train = t_train + v_train
    rng.shuffle(vqa_train)
    vqa_test = t_val[:1000] + v_val[:1000]
    print(f"vqa: train {write(OUT / 'vqa_train.jsonl', vqa_train)} (TextVQA {len(t_train)}, "
          f"VizWiz {len(v_train)}), test {write(OUT / 'vqa_test.jsonl', vqa_test)}", flush=True)


def check() -> int:
    bad = 0
    for name in ("caption_train", "caption_test", "vqa_train", "vqa_test"):
        p = OUT / f"{name}.jsonl"
        n = sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0
        miss = 0
        if n:
            for line in open(p, encoding="utf-8"):
                if not (OUT / json.loads(line)["image"]).exists():
                    miss += 1
        print(f"{name:14s} {n:7d} rows  {miss} images missing")
        bad += (n == 0) + (miss > 0)
    return 1 if bad else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--ln-images", type=int, default=40000, help="Open Images photos with a narrative")
    f.add_argument("--vizwiz", action="store_true", help="also VizWiz-VQA (~15 GB more)")
    sub.add_parser("check")
    a = p.parse_args()
    if a.cmd == "fetch":
        fetch(a)
    sys.exit(check())


if __name__ == "__main__":
    main()
