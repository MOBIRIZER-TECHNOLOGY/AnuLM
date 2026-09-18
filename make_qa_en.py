"""
Build an English question-answer set from the English Wikipedia articles.

    python make_qa_en.py --data data/english.txt --out data/qa_english.jsonl

make_qa.py's recipe, for English: one question per article about its
subject, answered by the article's own lead sentence(s).

    Question: What is Digimon?
    Answer: Digimon, short for "Digital Monsters", is a Japanese media franchise ...

Only the Wikipedia documents are used (C4 web pages have no subject line).
The scrubber drops the bold title from an English lead, so a lead that does
not start with a capital gets the title put back in front of it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from bpe import DOC_SEP

TEMPLATES = [
    "What is {t}?",
    "Tell me about {t}.",
    "Give a short introduction to {t}.",
    "Describe {t}.",
    "What do you know about {t}?",
    "Explain what {t} is.",
]

# A sentence ends at . ! or ? only when a capital, digit, quote or bracket
# follows: "i.e., it" and "U.S. Army" do not end sentences, "II. The" does.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_SPACE_PUNCT = re.compile(r"\s+([,.;:)])")
_EMPTY_PARENS = re.compile(r"\(\s*[,;\s]*\)")


def lead_paragraph(body: str, title: str) -> str | None:
    for para in body.split("\n\n"):
        p = _EMPTY_PARENS.sub("", para.strip().replace("\n", " "))
        p = re.sub(r"\s{2,}", " ", _SPACE_PUNCT.sub(r"\1", p))
        if len(p) < 80 or p[-1] not in ".!?":
            continue
        if not p[0].isupper() and not p[0].isdigit():     # bold title was stripped
            p = title + ("" if p[0] in ",;:" else " ") + p
        return p
    return None


def answer_from(lead: str, max_chars: int) -> str | None:
    sents = [s.strip() for s in _SENTENCE_END.split(lead) if s.strip()]
    if not sents or not sents[0][0].isalnum():
        return None
    ans = sents[0]
    if len(ans) < 40:
        if len(sents) < 2:
            return None
        ans = ans + " " + sents[1]
    elif len(sents) > 1 and len(ans) + 1 + len(sents[1]) <= max_chars:
        ans = ans + " " + sents[1]
    if len(ans) > max_chars or len(ans) < 40:
        return None
    return ans


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/english.txt")
    p.add_argument("--out", default="data/qa_english.jsonl")
    p.add_argument("--heldout", default="data/qa_english_heldout.jsonl")
    p.add_argument("--heldout-frac", type=float, default=0.02)
    p.add_argument("--max-answer-chars", type=int, default=300)
    p.add_argument("--max-title-chars", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = random.Random(args.seed)

    docs = [d for d in Path(args.data).read_text(encoding="utf-8").split(DOC_SEP)
            if d.strip() and not d.startswith("http")]
    items, skipped = [], 0
    for d in docs:
        title, _, body = d.partition("\n\n")
        title = title.strip()
        if not title or len(title) > args.max_title_chars or "(" in title or ":" in title:
            skipped += 1
            continue
        lead = lead_paragraph(body, title)
        ans = answer_from(lead, args.max_answer_chars) if lead else None
        if not ans:
            skipped += 1
            continue
        q = rng.choice(TEMPLATES).format(t=title)
        items.append({"title": title, "question": q, "answer": ans, "lang": "en"})

    rng.shuffle(items)
    n_held = int(len(items) * args.heldout_frac)
    held, train = items[:n_held], items[n_held:]
    for path, rows in ((args.out, train), (args.heldout, held)):
        with Path(path).open("w", encoding="utf-8", newline="\n") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    mean_a = sum(len(i["answer"]) for i in items) / max(len(items), 1)
    print(f"{len(docs)} articles -> {len(items)} QA pairs ({skipped} skipped), mean answer {mean_a:.0f} chars")
    print(f"train {len(train)} -> {args.out}   held-out {len(held)} -> {args.heldout}")
    for it in train[:3]:
        print(f"\n  Question: {it['question']}\n  Answer: {it['answer']}")


if __name__ == "__main__":
    main()
