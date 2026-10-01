"""Typed outputs of a decide.py checkpoint on banking messages and on tasks it
never trained on (sentiment, news topic, Hindi banking).

    python experiments/decide_samples.py ckpt_decide_general.pt
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import os  # noqa: E402

os.chdir(ROOT)
import torch  # noqa: E402

from decide import Decide  # noqa: E402

ck = sys.argv[1] if len(sys.argv) > 1 else "ckpt_decide_general.pt"
d = Decide(ck, "cuda" if torch.cuda.is_available() else "cpu", threshold=0.7)
cases = [
    ("banking (77 intents, never trained on)", None, [
        "I still haven't received my new card", "My card got swallowed by the ATM, what do I do?",
        "Why was I charged twice for the same coffee?", "can u change my pin pls",
        "मेरा कार्ड अभी तक नहीं आया", "What's the weather like in Delhi today?"]),
    ("sentiment (never trained on)", ["positive", "negative", "neutral"], [
        "I love this app, it works perfectly!", "This is the worst bank I have ever used.",
        "I opened the app today."]),
    ("news topic (never trained on)", ["sports", "politics", "technology", "cooking", "health"], [
        "India won the cricket match by six wickets.", "The new phone has a faster chip and a better camera.",
        "Add the onions and fry until golden.", "Parliament passed the new budget today."]),
    ("support urgency (never trained on)", ["urgent", "not urgent"], [
        "Someone is using my card right now, block it!", "How do I change the app's colour theme?"]),
]
for title, opts, msgs in cases:
    print(f"\n=== {title}")
    for m in msgs:
        out = d(m, opts)
        print(f"> {m}\n  {json.dumps(out, ensure_ascii=False)}")
