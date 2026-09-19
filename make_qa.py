"""
Build a Hindi question-answer set from the Wikipedia corpus.

    python make_qa.py --data data/hindi_full.txt --out data/qa_hindi.jsonl

The pretrained model continues text; it has never seen a question. This is
the smallest honest instruction set that teaches it the *format* without
inventing knowledge it does not have: for every article, a question about
its subject and the article's own lead sentence(s) as the answer.

    प्रश्न: काशी क्या है?
    उत्तर: काशी नगरी वर्तमान वाराणसी शहर में स्थित पौराणिक नगरी है।

Six question templates, chosen per article by a seeded RNG, so the model
learns that different phrasings ask the same thing. Answers are the first
one or two sentences of the lead paragraph, capped by length. Articles are
split 98 / 2 into train and held-out *by article*, so the held-out questions
are about subjects whose article the model saw in pretraining but whose
question it never did -- which is exactly what "answering a question" means
for a model of this size: recall plus format, not reasoning.

Wikipedia only: the literary half of the corpus has no subject-definition
structure to mine.
"""

from __future__ import annotations

try:                                    # Devanagari on a cp1252 console
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


import argparse
import json
import random
import re
from pathlib import Path

from bpe import DOC_SEP

TEMPLATES = [
    "{t} क्या है?",
    "{t} के बारे में बताइए।",
    "{t} के बारे में जानकारी दीजिए।",
    "{t} का परिचय दीजिए।",
    "मुझे {t} के बारे में बताओ।",
    "{t} किसे कहते हैं?",
]
PROMPT = "प्रश्न: {q}\nउत्तर:"          # the answer follows a single space, then EOS
# One template per language. Python questions are written in English, so
# they share its template; the model tells them apart from the content.
PROMPT_EN = "Question: {q}\nAnswer:"
# Translation pairs carry their direction as `lang`; serve.py's translate
# mode picks the direction from the script of the input.
PROMPTS = {"hi": PROMPT, "en": PROMPT_EN, "py": PROMPT_EN,
           "en-hi": "English: {q}\nHindi:", "hi-en": "Hindi: {q}\nEnglish:"}
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def template_for(question: str, templates: dict | None = None) -> str:
    """The prompt template a question should be wrapped in, chosen by script:
    any Devanagari means Hindi, anything else gets the English template. A
    checkpoint carries the templates it was tuned with as `qa_templates`."""
    t = templates or PROMPTS
    return t["hi"] if _DEVANAGARI.search(question) else t.get("en", t["hi"])


def translation_lang(text: str) -> str:
    """Direction for a translate request: Devanagari in -> English out."""
    return "hi-en" if _DEVANAGARI.search(text) else "en-hi"

_EMPTY_PARENS = re.compile(r"\(\s*[,;\s]*\)")
_LEADING_PUNCT = re.compile(r"\(\s*[,;]+\s*")     # "(; जन्म 1962)" -> "(जन्म 1962)"
_SPACE_PUNCT = re.compile(r"\s+([,।;:])")
_SENTENCE = re.compile(r"[^।]*।")


def lead_paragraph(body: str, title: str) -> str | None:
    """The first paragraph that reads as a lead: two sentences or 80+ chars,
    ends on a danda, and is not a bare caption line."""
    for para in body.split("\n\n"):
        p = _EMPTY_PARENS.sub("", para.strip().replace("\n", " "))
        p = _SPACE_PUNCT.sub(r"\1", _LEADING_PUNCT.sub("(", p))
        p = re.sub(r"\s{2,}", " ", p)
        if not p.endswith("।") or len(p) < 80:
            continue
        if p.count("।") < 2 and len(p) < 120:
            continue
        return p
    return None


def answer_from(lead: str, max_chars: int) -> str | None:
    sents = [s.strip() for s in _SENTENCE.findall(lead)]
    if not sents:
        return None
    ans = sents[0]
    if len(ans) < 40:                       # too short to be an answer; take two
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
    p.add_argument("--data", default="data/hindi_full.txt")
    p.add_argument("--out", default="data/qa_hindi.jsonl")
    p.add_argument("--heldout", default="data/qa_hindi_heldout.jsonl")
    p.add_argument("--heldout-frac", type=float, default=0.02)
    p.add_argument("--max-answer-chars", type=int, default=260)
    p.add_argument("--max-title-chars", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = random.Random(args.seed)

    docs = [d for d in Path(args.data).read_text(encoding="utf-8").split(DOC_SEP) if d.strip()]
    items, skipped = [], 0
    for d in docs:
        title, _, body = d.partition("\n\n")
        title = title.strip()
        if not title or len(title) > args.max_title_chars or "(" in title:
            skipped += 1
            continue
        lead = lead_paragraph(body, title)
        ans = answer_from(lead, args.max_answer_chars) if lead else None
        if not ans:
            skipped += 1
            continue
        q = rng.choice(TEMPLATES).format(t=title)
        items.append({"title": title, "question": q, "answer": ans, "lang": "hi"})

    rng.shuffle(items)
    n_held = int(len(items) * args.heldout_frac)
    held, train = items[:n_held], items[n_held:]
    for path, rows in ((args.out, train), (args.heldout, held)):
        with Path(path).open("w", encoding="utf-8", newline="\n") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    mean_a = sum(len(i["answer"]) for i in items) / max(len(items), 1)
    print(f"{len(docs)} articles -> {len(items)} QA pairs ({skipped} skipped), "
          f"mean answer {mean_a:.0f} chars")
    print(f"train {len(train)} -> {args.out}   held-out {len(held)} -> {args.heldout}")
    for it in train[:3]:
        print(f"\n  {PROMPT.format(q=it['question'])} {it['answer']}")


if __name__ == "__main__":
    main()
