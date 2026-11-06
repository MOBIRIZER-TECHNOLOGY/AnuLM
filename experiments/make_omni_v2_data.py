"""Build the run-2 training manifests: the current ones plus data/next.

    python experiments/make_omni_v2_data.py [--next C:/workspace/AnuLM/data/next]

Writes (each source's terms and dedupe are in data/next/MANIFEST.md):

  data/speech_ls460.jsonl          LibriSpeech train-clean-100 + train-clean-360 (CC BY 4.0)
  data/omni/vqa_train_v2.jsonl     TextVQA train + VizWiz-VQA train, answerable rows only (CC BY 4.0)
  data/omni/caption_train_v2.jsonl Localized Narratives, 49k + 60k Open Images photos (CC BY 4.0 / BY 2.0)
  data/rc_train_v2.jsonl           reading + MLQA hi.hi dev and unused translate-train hi (CC BY-SA 3.0)

Image paths are written absolute, so the merged files can sit anywhere. Rows whose
image is in a test set (vqa_test, caption_test) or whose question is in a reading
eval set are dropped again here, whatever the builders already removed.

VizWiz keeps only rows a majority marked answerable and whose answer is not
"unanswerable": 7,588 of its 20,523 training rows say the photo cannot answer the
question, and teaching that answer would leak into TextVQA, where every question
has an answer.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def absolute(r: dict, base: Path) -> dict:
    img = Path(r["image"])
    return {**r, "image": str((img if img.is_absolute() else base / img).resolve())}


def write(path: Path, out: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--next", default="C:/workspace/AnuLM/data/next")
    a = ap.parse_args()
    nxt, data, omni = Path(a.next), ROOT / "data", ROOT / "data" / "omni"
    report = {}

    # Speech: both LibriSpeech manifests point at absolute flac/wav paths.
    speech = list(rows(data / "speech_ls100.jsonl"))
    seen = {r["audio"] for r in speech}
    extra = [r for r in rows(nxt / "speech_ls360.jsonl") if r["audio"] not in seen]
    write(data / "speech_ls460.jsonl", speech + extra)
    report["speech_ls460"] = {"ls100": len(speech), "ls360": len(extra)}

    # Images: never train on a test photo.
    test_imgs = {absolute(r, omni)["image"] for f in ("vqa_test.jsonl", "caption_test.jsonl") for r in rows(omni / f)}
    for name, extra_file, keep in (
            ("vqa_train", "vqa_vizwiz_train.jsonl",
             lambda r: r.get("answerable") == 1 and r["text"].strip().lower() != "unanswerable"),
            ("caption_train", "caption_ln_extra.jsonl", lambda r: True)):
        c = Counter()
        old = [absolute(r, omni) for r in rows(omni / f"{name}.jsonl")]
        new = []
        for r in rows(nxt / extra_file):
            r = absolute(r, nxt)
            if not keep(r):
                c["not answerable"] += 1
            elif r["image"] in test_imgs:
                c["test photo"] += 1
            elif not Path(r["image"]).exists():
                c["photo missing"] += 1
            else:
                new.append(r)
        write(omni / f"{name}_v2.jsonl", old + new)
        report[f"{name}_v2"] = {"old": len(old), "new": len(new), "dropped": dict(c)}

    # Reading: drop any question that is in an eval or held-out set.
    held = {r["question"] for f in ("rc_eval_squad_en.jsonl", "rc_eval_mlqa_hi.jsonl", "rc_heldout.jsonl")
            for r in rows(data / f)}
    old = list(rows(data / "rc_train.jsonl"))
    have = {r["question"] for r in old}
    new = [r for r in rows(nxt / "rc_hi_extra.jsonl") if r["question"] not in held and r["question"] not in have]
    write(data / "rc_train_v2.jsonl", old + new)
    report["rc_train_v2"] = {"old": len(old), "new": len(new)}

    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
