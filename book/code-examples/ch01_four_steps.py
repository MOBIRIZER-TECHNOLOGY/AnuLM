"""
Book example, not repository code.

Chapter 1: the four-step loop (read, score, pick, repeat) written out by hand
around model.py's AnuLM, and checked token for token against AnuLM.generate.
Then a map of the repository at the pin, by role, and the check behind every
code citation in the book.

    python book/code-examples/ch01_four_steps.py

It imports model.py and test_model.py unchanged.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.model import AnuLM          # noqa: E402  the repository's model.py, unchanged
from tests.test_model import tiny      # noqa: E402  the test suite's small configuration

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PIN = "773b6a8"


def four_steps(model: AnuLM, ids: torch.Tensor, n: int, show: bool = False) -> torch.Tensor:
    """Greedy decoding with no cache: the loop at its plainest."""
    for step in range(n):
        logits, _ = model(ids)                        # 1. read: the whole sequence so far
        probs = F.softmax(logits[:, -1, :], dim=-1)   # 2. score: one probability per vocabulary id
        nxt = probs.argmax(dim=-1, keepdim=True)      # 3. pick: here, the most likely id
        if show and step == 0:
            print(f"read   ids    {tuple(ids.shape)}    logits {tuple(logits.shape)}")
            print(f"score  probs  {tuple(probs.shape)}  sum {probs.sum().item():.4f}")
            print(f"pick   nxt    {tuple(nxt.shape)}   id {nxt.item()}, p = {probs.max().item():.4f}")
        ids = torch.cat([ids, nxt], dim=1)            # 4. repeat: append, and go again
    return ids


def loop_matches_generate() -> None:
    torch.manual_seed(0)
    cfg = tiny()
    model = AnuLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        mine = four_steps(model, prompt.clone(), 12, show=True)
    ours = model.generate(prompt.clone(), 12, top_k=1)                    # KV cache on
    ours_nc = model.generate(prompt.clone(), 12, top_k=1, use_cache=False)
    print(f"by hand        {mine[0].tolist()}")
    print(f"generate       {ours[0].tolist()}")
    assert torch.equal(mine, ours) and torch.equal(mine, ours_nc), "the hand loop and generate disagree"
    print("by-hand loop == generate(top_k=1), with and without the cache: True")


ROLES = [  # (role, test on the file name); the first match wins
    ("the model", lambda f: f == "model.py"),
    ("tests", lambda f: f == "test_model.py"),
    ("training and tokenizer", lambda f: f in ("train.py", "muon.py", "bigram_teacher.py", "bpe.py")),
    ("post-training", lambda f: f in ("finetune.py", "lora.py", "decide.py", "rag.py", "rc_pointer.py",
                                      "bidirectional.py")),
    ("speech and vision", lambda f: f.startswith(("omni", "speech_", "vision_", "audio_", "voice"))),
    ("serving and apps", lambda f: f in ("serve.py", "api.py", "ask.py", "sample.py", "sample_many.py",
                                         "quickstart.py", "langchain_anulm.py") or f.startswith("app")),
    ("export", lambda f: f in ("export_hf.py", "modeling_anulm.py", "configuration_anulm.py", "hf_tok.py")
     or f.startswith("tools_hf")),
    ("evaluation", lambda f: f.startswith(("eval_", "make_bench", "make_golden")) or f == "tools_eval_noise.py"),
    ("data", lambda f: f.startswith(("fetch_", "convert_", "make_", "mix_", "filter_"))),
    ("tools", lambda f: True),
]


def at_pin(path: str) -> list[str]:
    out = subprocess.run(["git", "show", f"{PIN}:{path}"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", check=True).stdout
    return out.splitlines()


def repo_map() -> dict:
    names = subprocess.run(["git", "ls-tree", "--name-only", PIN], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    files = [f for f in names if f.endswith(".py")]
    groups: dict[str, list] = {}
    for f in files:
        role = next(r for r, test in ROLES if test(f))
        groups.setdefault(role, []).append((f, len(at_pin(f))))
    print(f"\nPython files at the repository root @{PIN}: {len(files)}")
    for role, _ in ROLES:
        fs = groups.get(role, [])
        print(f"  {role:24s} {len(fs):3d} files {sum(n for _, n in fs):6,d} lines")
    return groups


def subdirectories() -> None:
    """The Python outside the root: the lab book, run scripts, the reference code, examples."""
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", PIN], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    py = [f for f in names if f.endswith(".py")]
    print(f"\nPython files in the whole tree @{PIN}: {len(py)}")
    for d in ("course", "experiments", "sarvam", "examples"):
        fs = [f for f in py if f.startswith(d + "/")]
        print(f"  {d + '/':24s} {len(fs):3d} files {sum(len(at_pin(f)) for f in fs):6,d} lines")


def citation_holds(path: str, first: int, last: int, quoted: list[str]) -> bool:
    """What the book's checker does for every excerpt: each quoted line, stripped,
    must occur inside the cited range of the file at the pin."""
    window = [s.strip() for s in at_pin(path)[first - 1:last]]
    return all(q.strip() in window for q in quoted if q.strip() and q.strip() != "# ...")


if __name__ == "__main__":
    loop_matches_generate()
    repo_map()
    subdirectories()
    excerpt = ["logits = logits / max(temperature, 1e-5)",
               "probs = F.softmax(logits, dim=-1)",
               "nxt = torch.multinomial(probs, num_samples=1)              # (B, 1)"]
    print(f"\ncitation model.py 1100-1105 holds: {citation_holds('model.py', 1100, 1105, excerpt)}")
    print(f"citation model.py 1000-1005 holds: {citation_holds('model.py', 1000, 1005, excerpt)}")
