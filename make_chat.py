"""
Instruction-tuning set for the chat model (finetune.py's {question, answer, lang}).

    python make_chat.py        # -> data/chat_train.jsonl, data/chat_heldout.jsonl

Sources, all Apache 2.0 or CC BY-SA, all human-written:
  * databricks-dolly-15k        English instructions (data/dolly_train.jsonl)
  * OpenAssistant oasst2         first turn of each conversation tree, with its
                                  rank-0 (best-rated) assistant reply
  * CohereForAI/aya_dataset      the Hindi rows -- native Hindi, not translated
  * Hindi Wikipedia QA           4,000 of make_qa.py's mined pairs (Aya has only ~1.1k Hindi rows)
  * identity                     what the model is, in both languages, so a demo
                                  question "who are you?" gets a true answer

Replies longer than 1,500 characters are dropped (finetune.py packs pairs into
the model's block and drops what does not fit anyway). 300 random pairs are
held out.
"""

from __future__ import annotations

import gzip
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
DEV = re.compile(r"[ऀ-ॿ]")
MAX_ANSWER = 1500

IDENTITY_EN = [
    ("Who are you?", "I am AnuLM, a small language model trained from scratch on one consumer GPU. I read Hindi, English and Python."),
    ("What is your name?", "My name is AnuLM."),
    ("Who made you?", "I was trained by the AnuLM project, from scratch, on a single RTX 5070 Ti graphics card."),
    ("What languages do you speak?", "I understand and write Hindi and English, and I can write simple Python code."),
    ("Are you ChatGPT?", "No. I am AnuLM, a much smaller model trained from scratch. I can make mistakes, especially about facts."),
    ("Can I trust your answers?", "Not fully. I am a small model and I can be confidently wrong. For facts, check a reliable source; my Wikipedia mode looks answers up."),
]
IDENTITY_HI = [
    ("आप कौन हैं?", "मैं AnuLM हूँ, एक छोटा भाषा मॉडल जिसे एक ही GPU पर शुरू से प्रशिक्षित किया गया है। मैं हिन्दी, अंग्रेज़ी और Python समझता हूँ।"),
    ("आपका नाम क्या है?", "मेरा नाम AnuLM है।"),
    ("आपको किसने बनाया?", "मुझे AnuLM परियोजना ने एक RTX 5070 Ti ग्राफ़िक्स कार्ड पर शुरू से प्रशिक्षित किया है।"),
    ("आप कौन सी भाषाएँ बोलते हैं?", "मैं हिन्दी और अंग्रेज़ी समझता और लिखता हूँ, और साधारण Python कोड लिख सकता हूँ।"),
    ("क्या मैं आपके उत्तरों पर भरोसा कर सकता हूँ?", "पूरी तरह नहीं। मैं एक छोटा मॉडल हूँ और गलत हो सकता हूँ। तथ्यों के लिए किसी विश्वसनीय स्रोत से जाँच करें।"),
]


def dolly():
    p = DATA / "dolly_train.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def oasst2():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("OpenAssistant/oasst2", "2023-11-05_oasst2_ready.trees.jsonl.gz",
                           repo_type="dataset", local_dir=str(DATA / "raw" / "oasst2"))
    out = []
    for line in gzip.open(path, "rt", encoding="utf-8"):
        t = json.loads(line)
        root = t["prompt"]
        if root.get("lang") not in ("en", "hi"):
            continue
        replies = [r for r in root.get("replies", []) if r.get("role") == "assistant"]
        ranked = [r for r in replies if r.get("rank") == 0] or replies[:1]
        if ranked and ranked[0].get("text"):
            out.append({"question": root["text"].strip(), "answer": ranked[0]["text"].strip(),
                        "lang": "hi" if DEV.search(root["text"]) else "en", "source": "oasst2"})
    return out


def aya_hindi():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("CohereForAI/aya_dataset", "data/train-00000-of-00001.parquet",
                           repo_type="dataset", local_dir=str(DATA / "raw" / "aya"))
    df = pd.read_parquet(path)
    df = df[df["language"].str.lower() == "hindi"]
    return [{"question": q.strip(), "answer": a.strip(), "lang": "hi", "source": "aya"}
            for q, a in zip(df["inputs"], df["targets"])]


def hindi_wiki_qa(n: int, rng) -> list[dict]:
    """Questions mined from Hindi Wikipedia by make_qa.py ("X क्या है?" -> the
    article's lead). Aya has only ~1.1k Hindi rows, so these carry most of the
    Hindi signal; capped so they do not swamp the conversational sources."""
    p = DATA / "qa_lora.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    rng.shuffle(rows)
    return [{"question": r["question"], "answer": r["answer"], "lang": "hi", "source": "hi_wiki_qa"}
            for r in rows[:n]]


def main():
    rng = random.Random(0)
    parts = {"dolly": [{**r, "source": "dolly"} for r in dolly()], "oasst2": oasst2(), "aya_hi": aya_hindi(),
             "hi_wiki_qa": hindi_wiki_qa(4000, rng)}
    ident = [{"question": q, "answer": a, "lang": "en", "source": "identity"} for q, a in IDENTITY_EN] + \
            [{"question": q, "answer": a, "lang": "hi", "source": "identity"} for q, a in IDENTITY_HI]
    parts["identity"] = ident * 20                         # rare but must be learnt
    rows = []
    for name, items in parts.items():
        kept = [r for r in items if r["question"] and r["answer"] and len(r["answer"]) <= MAX_ANSWER]
        print(f"{name:9s} {len(items):6d} rows, {len(kept):6d} kept")
        rows += kept
    rng.shuffle(rows)
    held = [r for r in rows if r["source"] != "identity"][:300]
    held_ids = {id(r) for r in held}
    train = [r for r in rows if id(r) not in held_ids]
    for name, rs in (("chat_train", train), ("chat_heldout", held)):
        with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({"question": r["question"], "answer": r["answer"], "lang": r["lang"]},
                                   ensure_ascii=False) + "\n")
    hi = sum(r["lang"] == "hi" for r in train)
    print(f"train {len(train)} ({hi} Hindi), held-out {len(held)}")


if __name__ == "__main__":
    main()
