"""
The second batch of Indian-language data -> training-ready jsonl, one file per
source, in chat_train.jsonl's format ({question, answer, lang, source}).

    python experiments/prep_indic_extra.py          # writes data/extra_hi/*.jsonl + README.md

Inputs (downloaded 2026-10-18, each in its own folder under data/):
  hinglish/hinglish_upload_v1.json    findnitai/english-to-hinglish, Apache-2.0
  indic_instruct/{hh-rlhf,flan_v2}-hi.parquet, nmt-en-hi.parquet
                                       ai4bharat/indic-instruct-data-v0.1 (per-subset licences,
                                       arXiv 2401.15006: hh-rlhf MIT, FLAN Apache-2.0, NMT CC BY 4.0)
  indic_align/...anudesh1, ...oasst, ...wiki_chat_0000_of_0031
                                       ai4bharat/indic-align, CC BY 4.0

Every file drops questions that appear in data/chat_heldout.jsonl (the chat
held-out set stays unseen) and questions already in data/chat_train_v4.jsonl.
Nothing here trains anything; finetune.py --qa takes any of these files, or a
concatenation of them.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
OUT = D / "extra_hi"
MAX_Q, MAX_A = 1500, 3000
DEVA = re.compile(r"[ऀ-ॿ]")


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def deva_share(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    return sum(bool(DEVA.match(c)) for c in letters) / max(1, len(letters))


# Marathi, Nepali and Sanskrit are Devanagari too: a script test lets them through.
# Common function words tell Hindi from Marathi (anudesh holds both).
HI_WORDS = {"है", "हैं", "का", "की", "के", "में", "और", "नहीं", "क्या", "को", "से", "यह", "था", "थे", "कैसे", "लिए"}
MR_WORDS = {"आहे", "आहेत", "आणि", "नाही", "काय", "कसा", "कसे", "मध्ये", "होते", "होतो", "आपण", "त्या", "हे", "ला"}


def is_hindi(s: str) -> bool:
    words = re.findall(r"[ऀ-ॿ]+", s)
    hi, mr = sum(w in HI_WORDS for w in words), sum(w in MR_WORDS for w in words)
    return hi >= 2 and hi > 2 * mr


def first_turn(turns):
    """[[q, a], ...] or [{'role','content'}, ...] -> (q, a) or None."""
    if not turns:
        return None
    if isinstance(turns[0], dict):
        if len(turns) >= 2 and turns[0].get("role") in ("user", "prompter") and turns[1].get("role") == "assistant":
            return turns[0].get("content") or "", turns[1].get("content") or ""
        return None
    t = turns[0]
    return (t[0] or "", t[1] or "") if t is not None and len(t) >= 2 else None


class Writer:
    def __init__(self, blocked: set[str]):
        self.blocked, self.stats = blocked, {}

    def write(self, name: str, rows, source: str, licence: str, note: str):
        kept, seen, dropped = [], set(), 0
        for q, a, lang in rows:
            q, a = (q or "").strip(), (a or "").strip()
            k = norm(q)
            if not q or not a or len(q) > MAX_Q or len(a) > MAX_A or k in self.blocked or k in seen:
                dropped += 1
                continue
            seen.add(k)
            kept.append({"question": q, "answer": a, "lang": lang, "source": source})
        random.Random(0).shuffle(kept)
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                                           encoding="utf-8")
        self.stats[name] = {"rows": len(kept), "dropped": dropped, "source": source, "licence": licence, "note": note}
        print(f"{name:32s} {len(kept):7,d} kept  {dropped:6,d} dropped", flush=True)


def used_questions() -> set[str]:
    """Questions chat_train_v4.jsonl holds: chat_train.jsonl plus the first user turn of
    the four indic-instruct Hindi subsets (read from the sources, so this works
    before experiments/indic_v4.py has built the file)."""
    v4 = D / "chat_train_v4.jsonl"
    if v4.exists():
        return {norm(json.loads(l)["question"]) for l in open(v4, encoding="utf-8")}
    used = {norm(json.loads(l)["question"]) for l in open(D / "chat_train.jsonl", encoding="utf-8")}
    for name in ("anudesh", "wikihow", "dolly", "oasst1"):
        p = D / "indic_instruct" / f"{name}-hi.parquet"
        if not p.exists():
            continue
        for r in pq.read_table(p).to_pylist():
            if "instruction" in r:                       # dolly's columns
                ctx = (r.get("context") or "").strip()
                used.add(norm((r.get("instruction") or "").strip() + (f"\n\n{ctx}" if ctx else "")))
            elif (ft := first_turn(r.get("messages") or [])):
                used.add(norm(ft[0]))
    return used


def main():
    global ROOT, D, OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ROOT = ap.parse_args().root.resolve()
    D, OUT = ROOT / "data", ROOT / "data" / "extra_hi"
    OUT.mkdir(exist_ok=True)
    blocked = {norm(json.loads(l)["question"]) for l in open(D / "chat_heldout.jsonl", encoding="utf-8")}
    w = Writer(blocked | used_questions())

    # Hinglish <-> English translation pairs
    pairs = [json.loads(l)["translation"] for l in open(D / "hinglish" / "hinglish_upload_v1.json", encoding="utf-8")
             if l.strip()]
    rng = random.Random(1)
    rows = []
    for p in pairs:
        en, hg = p.get("en", ""), p.get("hi_ng", "")
        if rng.random() < 0.5:
            rows.append((f"Translate to Hinglish: {en}", hg, "hinglish"))
        else:
            rows.append((f"Translate to English: {hg}", en, "hinglish"))
    w.write("hinglish_translate", rows, "findnitai/english-to-hinglish", "Apache-2.0",
            "189k sentence pairs, half each direction; spelling is informal and noisy")

    # English -> Hindi translation (seed set, human-curated)
    t = pq.read_table(D / "indic_instruct" / "nmt-en-hi.parquet").to_pylist()
    rows = [(f"इस वाक्य का हिंदी में अनुवाद करें: {r['input_text']}", r["output_text"], "hi") for r in t]
    w.write("nmt_en_hi", rows, "ai4bharat/indic-instruct-data-v0.1 nmt", "CC BY 4.0", "English -> Hindi sentences")

    # hh-rlhf, Hindi (machine-translated helpful-assistant dialogues), first turn
    t = pq.read_table(D / "indic_instruct" / "hh-rlhf-hi.parquet").to_pylist()
    rows = [(*ft, "hi") for r in t if (ft := first_turn(r["messages"]))]
    w.write("hh_rlhf_hi", rows, "ai4bharat/indic-instruct-data-v0.1 hh-rlhf", "MIT",
            "machine-translated Anthropic HH helpful dialogues, first turn")

    # FLAN v2, Hindi: short-answer tasks; kept apart (it teaches one-word replies)
    t = pq.read_table(D / "indic_instruct" / "flan_v2-hi.parquet", columns=["inputs", "targets"]).to_pylist()
    w.write("flan_v2_hi", [(r["inputs"], r["targets"], "hi") for r in t],
            "ai4bharat/indic-instruct-data-v0.1 flan_v2", "Apache-2.0",
            "machine-translated FLAN tasks with short answers; mix in sparingly, not for chat style")

    # indic-align Anudesh: crowd-written prompts in many languages; keep the Hindi ones
    t = pq.read_table(D / "indic_align" / "indicalign-instruct__anudesh__anudesh1.parquet").to_pylist()
    rows = []
    for r in t:
        ft = first_turn(r["interactions"])
        if ft and deva_share(ft[0]) > 0.6 and deva_share(ft[1]) > 0.6 and is_hindi(ft[0] + " " + ft[1]):
            rows.append((*ft, "hi"))
    w.write("anudesh_hi", rows, "ai4bharat/indic-align anudesh", "CC BY 4.0",
            "crowd-written prompts, Hindi only (Devanagari, and Hindi not Marathi function words); "
            "LLM-written answers")

    # indic-align WikiChat (grounded in Wikipedia) and OASST: Hindi script and romanized Hindi
    for shard, name, cols in (("indicalign-instruct__wiki_chat__wiki_chat_0000_of_0031.parquet", "wikichat",
                               (("hin_Deva", "hi"), ("hin_Latn", "hinglish"))),
                              ("indicalign-instruct__oasst__oasst.parquet", "oasst",
                               (("hin_Latn", "hinglish"),))):        # hin_Deva oasst is already in v4
        t = pq.read_table(D / "indic_align" / shard, columns=[c for c, _ in cols]).to_pylist()
        for col, lang in cols:
            rows = [(*ft, lang) for r in t if (ft := first_turn(r[col]))]
            w.write(f"{name}_{'hi' if lang == 'hi' else 'hinglish'}", rows, f"ai4bharat/indic-align {name} {col}",
                    "CC BY 4.0", ("questions generated from Wikipedia passages, answers LLM-written; "
                                  if name == "wikichat" else "OASST conversations, ")
                    + ("Hindi script" if lang == "hi" else "LOW QUALITY: romanized Hindi (Hinglish) machine-transliterated from Devanagari, odd spellings ('haye joe' for 'hai jo'); opt-in only")
                    + ("; 1 of 32 shards" if name == "wikichat" else ""))

    lines = ["# data/extra_hi: Indian-language training data, ready for finetune.py --qa", "",
             "Built by `experiments/prep_indic_extra.py`. Each file has rows of "
             "`{question, answer, lang, source}`; chat_heldout questions and questions already in "
             "chat_train_v4.jsonl are removed.", "",
             "| file | rows | lang | source | licence | note |", "|---|---:|---|---|---|---|"]
    for name, s in w.stats.items():
        lang = "hinglish" if "hinglish" in name else "hi"
        lines.append(f"| `{name}.jsonl` | {s['rows']:,} | {lang} | {s['source']} | {s['licence']} | {s['note']} |")
    lines += ["", f"Total: {sum(s['rows'] for s in w.stats.values()):,} rows."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "stats.json").write_text(json.dumps(w.stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
