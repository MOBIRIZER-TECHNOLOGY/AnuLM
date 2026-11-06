"""Build data/chat_train_v6.jsonl: the chat model's pairs plus human-written extras.

    python experiments/make_chat_v6.py [--extra C:/workspace/AnuLM/data/next/chat_extra.jsonl]

chat_train.jsonl (23,022 pairs) is what ckpt_chat_v3 learned, and the 3,125-step
Omni run reads it about twice over; chat held-out loss rose while the other tasks
pulled the shared decoder. chat_extra.jsonl adds 13,163 human-written English
pairs (oasst2 later turns and alternative rank-1 replies, Aya English rows, both
Apache 2.0), already deduplicated against chat_heldout and the Aya test split
(data/next/MANIFEST.md). This script re-checks that, drops exact duplicates and
keeps only the question/answer/lang fields the trainer reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=str(ROOT / "data" / "chat_train.jsonl"))
    ap.add_argument("--extra", default="C:/workspace/AnuLM/data/next/chat_extra.jsonl")
    ap.add_argument("--heldout", default=str(ROOT / "data" / "chat_heldout.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "chat_train_v6.jsonl"))
    a = ap.parse_args()

    held = {norm(json.loads(l)["question"]) for l in open(a.heldout, encoding="utf-8") if l.strip()}
    seen, rows, drops = set(), [], {"heldout question": 0, "duplicate": 0, "empty": 0}
    for src in (a.base, a.extra):
        for line in open(src, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            q, ans = (r.get("question") or "").strip(), (r.get("answer") or "").strip()
            if not q or not ans:
                drops["empty"] += 1
                continue
            if src == a.extra and any(h in norm(q) for h in held if len(h) > 30) or norm(q) in held:
                drops["heldout question"] += 1
                continue
            key = (norm(q), norm(ans))
            if key in seen:
                drops["duplicate"] += 1
                continue
            seen.add(key)
            rows.append({"question": q, "answer": ans, "lang": r.get("lang", "en")})
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {a.out}; dropped {drops}")


if __name__ == "__main__":
    main()
