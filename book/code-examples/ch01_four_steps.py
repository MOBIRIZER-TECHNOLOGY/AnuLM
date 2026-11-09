"""
Book example, not repository code.

Chapter 1: the four-step loop (read, score, pick, repeat) written out by hand
around anulm/model.py's AnuLM, and checked token for token against
AnuLM.generate. Then a map of the repository at the pin, by role, and the check
behind every code citation in the book.

    python book/code-examples/ch01_four_steps.py

It imports anulm/model.py and tests/test_model.py unchanged.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from anulm.model import AnuLM          # noqa: E402  the repository's anulm/model.py, unchanged
from tests.test_model import tiny      # noqa: E402  the test suite's small configuration

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
PIN = "2e80b3e"


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


ROLES = [  # (role, test on the path at the pin); the first match wins
    ("the model", lambda f: f == "anulm/model.py"),
    ("tests", lambda f: f.startswith("tests/")),
    ("training and tokenizer", lambda f: f.startswith("anulm/training/")
     or f in ("anulm/bpe.py", "anulm/muon.py", "anulm/bigram_teacher.py")),
    ("data", lambda f: f.startswith("anulm/data/")),
    ("evaluation", lambda f: f.startswith("anulm/eval/")),
    ("post-training", lambda f: f.startswith("anulm/tasks/") or f in ("anulm/lora.py", "anulm/bidirectional.py")),
    ("speech and vision", lambda f: f.startswith("anulm/multimodal/")),
    ("serving and apps", lambda f: f.startswith(("anulm/inference/", "anulm/apps/")) or f == "anulm/quickstart.py"),
    ("export", lambda f: f.startswith("anulm/hf/") or f == "anulm/hf_tok.py"),
    ("package entry", lambda f: f in ("anulm/__init__.py", "anulm/__main__.py", "model.py")),
    ("tools", lambda f: f.startswith("tools/")),
]
FOLDERS = ("course", "experiments", "examples", "book", "space_omni")


def at_pin(path: str) -> list[str]:
    out = subprocess.run(["git", "show", f"{PIN}:{path}"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", check=True).stdout
    return out.splitlines()


def tree_at_pin() -> list[str]:
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", PIN], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    # A folder of third-party reference code was at the pin and has since been
    # removed from the repository; count only what a clone still holds.
    return [f for f in names if f.endswith(".py") and (ROOT / f.split("/")[0]).exists()]


def repo_map() -> dict:
    """The package, its tests and tools (everything outside the four folders below), by role."""
    files = [f for f in tree_at_pin() if not f.startswith(tuple(d + "/" for d in FOLDERS))]
    groups: dict[str, list] = {}
    for f in files:
        role = next(r for r, test in ROLES if test(f))
        groups.setdefault(role, []).append((f, len(at_pin(f))))
    print(f"\nPython files in anulm/, tests/, tools/ and the root @{PIN}: {len(files)}")
    for role, _ in ROLES:
        fs = groups.get(role, [])
        print(f"  {role:24s} {len(fs):3d} files {sum(n for _, n in fs):6,d} lines")
    return groups


def subdirectories() -> None:
    """The rest of the Python: the lab book, run scripts, examples, the book."""
    py = tree_at_pin()
    print(f"\nPython files in the whole tree @{PIN}: {len(py)}")
    for d in FOLDERS:
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
    excerpt = ["logits = logits / temperature",
               "probs = F.softmax(logits, dim=-1)",
               "nxt = torch.multinomial(probs, num_samples=1)          # (B, 1)"]
    print(f"\ncitation anulm/model.py 1174-1179 holds: {citation_holds('anulm/model.py', 1174, 1179, excerpt)}")
    print(f"citation anulm/model.py 1064-1069 holds: {citation_holds('anulm/model.py', 1064, 1069, excerpt)}")
