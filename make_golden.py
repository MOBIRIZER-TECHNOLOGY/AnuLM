"""
Build the golden set: Hindi cloze items with a known correct answer.

    python make_golden.py --out golden/golden_hindi.jsonl

bits/byte says how well a model compresses text; it never says whether the
model got anything *right*. This file is the first accuracy axis in the
project: every item is a passage prefix plus one blanked content word, and
the gold answer is the word that is actually there in the held-out text.

Two item types, in equal number, labelled in the file:

  in_context  the gold word already occurred in the prefix (the last
              --context-words words before the blank, which is exactly what
              the scorer feeds the model). A
              competent reader can answer these from the passage alone;
              a model answers them by copying/coreference (LAMBADA's design).
  novel       the gold word has NOT occurred in the prefix. The model has
              to know the language and the register -- these are harder,
              and a fair share are not answerable with certainty by anyone.

Every item also carries three distractors: Devanagari words from the same
benchmark file with similar frequency and length, absent from the prefix.
That gives a 4-way multiple-choice score with chance at 25%, which is robust
to the tokenizer and to "right word, wrong spelling" -- see eval_golden.py.

Sources are the two shared benchmark files, which no checkpoint trained on:
Wikipedia (bench_hindi.txt) and Wikisource literature (bench_wikisource.txt).
The build is deterministic (--seed) and one item is taken per document.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from bpe import DOC_SEP

SOURCES = {"wikipedia": "data/bench_hindi.txt", "wikisource": "data/bench_wikisource.txt"}

# Function words, pronouns, auxiliaries, common light verbs: predictable from
# grammar alone, so blanking them measures nothing about content.
STOPWORDS = set("""
है हैं था थे थी थीं हो हों होता होती होते हुआ हुई हुए होना होने और या एवं तथा
का की के को में से पर ने तक ही भी तो कि जो जब तब अब यहाँ वहाँ यह वह ये वे इस उस इन उन
इसे उसे इसका उसका इसकी उसकी इसके उसके अपना अपनी अपने मैं हम तुम आप वो मेरा मेरी मेरे
हमारा हमारी हमारे तुम्हारा तुम्हारी तुम्हारे उनका उनकी उनके एक दो तीन चार पाँच कुछ सब सभी
बहुत कई कोई किसी कौन क्या क्यों कैसे कहाँ नहीं न ना मत रहा रही रहे रहना गया गयी गई गए
गये कर करना करने करता करती करते किया किये किए दिया दिये दिए लिया लिये लिए जाता जाती जाते
जाना जाने सकता सकती सकते चाहिए पड़ा पड़ी पड़े लगा लगी लगे साथ बाद पहले फिर अभी यहां वहां
कहा कहते कहती बोला बोली द्वारा लिये लिए ओर तरह प्रकार रूप बारे अनुसार वाला वाली वाले
""".split())

DEVANAGARI_WORD = re.compile(r"^[ऀ-ॿ]+$")
STRIP_PUNCT = re.compile(r"^[\"'(\[“‘]+|[\"')\]”’,;:।.!?]+$")
SENTENCE_END = re.compile(r"[।.!?]['\")\]”’]*\s*$")
DIGITS = re.compile(r"[0-9०-९]")
# Marathi shares the script and shows up in Hindi Wikipedia (film and play
# lists). These function words are Marathi, not Hindi; two hits and the
# document is out.
MARATHI = re.compile(r"(?<!\S)(आणि|आहे|आहेत|होते|म्हणून|त्यांनी|येथे)(?!\S)")


def clean(word: str) -> str:
    return STRIP_PUNCT.sub("", word)


def prose_fraction(doc: str) -> float:
    lines = [l.strip() for l in doc.split("\n") if l.strip()]
    if not lines:
        return 0.0
    return sum(1 for l in lines if SENTENCE_END.search(l)) / len(lines)


def sentence_line(doc: str, s: int, e: int) -> str | None:
    """The line containing doc[s:e] if it reads as a sentence: at least ten
    words, ends on a sentence mark, and is not a table or an index (digits
    in more than a tenth of its characters). None otherwise."""
    ls = doc.rfind("\n", 0, s) + 1
    le = doc.find("\n", e)
    line = doc[ls: le if le >= 0 else len(doc)]
    if len(line.split()) < 10 or not SENTENCE_END.search(line):
        return None
    if len(DIGITS.findall(line)) > len(line) / 10:
        return None
    return line


def is_content_word(w: str) -> bool:
    return (DEVANAGARI_WORD.match(w) is not None and 3 <= len(w) <= 14
            and w not in STOPWORDS and not any("०" <= ch <= "९" for ch in w))


def build(text: str, source: str, per_type: int, context_words: int, min_words: int,
          rng: random.Random) -> list[dict]:
    docs = [d for d in text.split(DOC_SEP) if d.strip()]
    rng.shuffle(docs)

    # Frequency table for distractor sampling, over the whole file.
    freq = Counter(clean(w) for d in docs for w in d.split() if is_content_word(clean(w)))
    by_len: dict[int, list[str]] = {}
    for w in freq:
        by_len.setdefault(len(w), []).append(w)

    def distractors(gold: str, prefix_words: set[str], k: int = 3) -> list[str]:
        f = freq[gold]
        pool = [w for L in range(len(gold) - 2, len(gold) + 3) for w in by_len.get(L, ())
                if w != gold and w not in prefix_words and f / 3 <= freq[w] <= f * 3]
        if len(pool) < k:                      # widen the frequency band
            pool = [w for L in range(len(gold) - 2, len(gold) + 3) for w in by_len.get(L, ())
                    if w != gold and w not in prefix_words]
        return rng.sample(pool, k) if len(pool) >= k else []

    items = {"in_context": [], "novel": []}
    for doc in docs:
        if all(len(v) >= per_type for v in items.values()):
            break
        # Word spans in the original string, so the prefix keeps its newlines.
        spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", doc)]
        if len(spans) < min_words:
            continue
        # Prose documents only: half the lines must end sentences (index
        # pages, filmographies and station lists fail this), and no Marathi.
        if prose_fraction(doc) < 0.5 or len(MARATHI.findall(doc)) >= 2:
            continue
        candidates = []
        for i in range(min_words // 2, len(spans)):
            s, e, raw = spans[i]
            gold = clean(raw)
            # Exactly one space before the word, so the candidate is " " + word
            # for every tokenizer and the prefix ends on the previous word.
            if not is_content_word(gold) or doc[spans[i - 1][1]:s] != " ":
                continue
            # The blank must sit inside a real sentence, with at least four
            # words of that sentence already on the page before it.
            line = sentence_line(doc, s, e)
            if line is None or len(doc[doc.rfind("\n", 0, s) + 1: s].split()) < 4:
                continue
            # "Seen before" means seen in the prefix the item will actually
            # carry -- the last `context_words` words -- not anywhere earlier
            # in the document. The first build checked the whole document and
            # labelled 64 of 300 in_context items whose gold word had
            # scrolled out of the prefix; those were novel items in disguise.
            prev = {clean(w) for _, _, w in spans[max(0, i - context_words):i]}
            kind = "in_context" if gold in prev else "novel"
            if len(items[kind]) < per_type:
                candidates.append((i, gold, kind, prev))
        if not candidates:
            continue
        i, gold, kind, prev = rng.choice(candidates)
        a = spans[max(0, i - context_words)][0]
        prefix = doc[a: spans[i - 1][1]]
        ds = distractors(gold, prev)
        if not ds:
            continue
        choices = [gold] + ds
        rng.shuffle(choices)
        items[kind].append({
            "id": f"{source}-{len(items['in_context']) + len(items['novel']):04d}",
            "source": source, "type": kind,
            "title": doc.split("\n", 1)[0].strip(),
            "prefix": prefix, "answer": gold,
            "choices": choices, "answer_idx": choices.index(gold),
            "raw_next": doc[spans[i][0]: spans[min(i + 6, len(spans) - 1)][1]],
        })
    return items["in_context"] + items["novel"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="golden/golden_hindi.jsonl")
    p.add_argument("--per-type", type=int, default=150,
                   help="items per (source, type) cell; 4 cells")
    p.add_argument("--context-words", type=int, default=120)
    p.add_argument("--min-words", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = random.Random(args.seed)
    items = []
    for source, path in SOURCES.items():
        text = Path(path).read_text(encoding="utf-8")
        got = build(text, source, args.per_type, args.context_words, args.min_words, rng)
        items.extend(got)
        n_ic = sum(1 for it in got if it["type"] == "in_context")
        print(f"{source}: {len(got)} items ({n_ic} in_context, {len(got) - n_ic} novel)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {out}: {len(items)} items, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
