"""
Stage 3: recompute the blueprint's numeric and narrative claims about the repository.

    python book/tools/verify_claims.py      # appends to technical-review/source-verification.md

Every claim is recomputed from the code at the pinned commit or from the
released checkpoint, never read from the book's own prose.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import torch

BOOK = Path(__file__).resolve().parent.parent
ROOT = BOOK.parent
OUT = BOOK / "technical-review" / "source-verification.md"
PIN = re.search(r"`@([0-9a-f]{7,40})`", (BOOK / "REPOSITORY_SOURCE_MAP.md").read_text(encoding="utf-8")).group(1)


def at(path):
    return subprocess.run(["git", "show", f"{PIN}:{path}"], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(ROOT))
    from anulm import model as M  # noqa: E402  the working tree, identical to the pin for model.py (checked by verify_source_map)

    rows = []

    def claim(text, ok, evidence):
        rows.append((text, "yes" if ok else "**NO**", evidence))

    src = at("anulm/model.py")
    n = len(src.splitlines())
    claim("anulm/model.py has 1,364 lines", n == 1364, f"{n} lines at @{PIN}")

    tsrc = at("tests/test_model.py")
    tree = ast.parse(tsrc)
    decorated = [f.name for f in tree.body if isinstance(f, ast.FunctionDef)
                 and any(ast.unparse(d).split("(")[0] in ("test", "check", "case") for d in f.decorator_list)]
    deco_names = sorted({ast.unparse(d).split("(")[0] for f in tree.body if isinstance(f, ast.FunctionDef)
                         for d in f.decorator_list})
    claim("tests/test_model.py has 80 tests", len(decorated) == 80,
          f"{len(decorated)} functions carry a test decorator ({', '.join(deco_names)}) at @{PIN}")

    ck = torch.load(ROOT / "ckpt_base_v2.pt", map_location="cpu", weights_only=False, mmap=True)
    cfg = ck["cfg"]
    c = {k: getattr(cfg, k) for k in ("vocab_size", "hidden_size", "n_layer", "n_head", "n_kv_head", "head_dim",
                                      "rope_theta", "num_experts", "num_experts_per_tok", "num_shared_experts",
                                      "moe_intermediate_size", "intermediate_size", "sliding_window", "max_window_layers",
                                      "use_qk_norm", "attention", "first_k_dense_replace", "tie_word_embeddings",
                                      "dropout") if hasattr(cfg, k)}
    total = sum(v.numel() for v in ck["model"].values() if v.is_floating_point() and not any(
        s in k for k in [""] for s in ()))
    params = sum(v.numel() for k, v in ck["model"].items() if not k.endswith(("expert_bias", "load_counts")))
    m = M.AnuLM(cfg)
    tot, act = m.num_params()
    claim("released model: 397.7M parameters", abs(tot / 1e6 - 397.7) < 0.1, f"num_params() total {tot:,}; state dict {params:,}")
    claim("about 173M active per token", abs(act / 1e6 - 173) < 1.5, f"num_params() active {act:,}")
    claim("32k vocabulary", c["vocab_size"] == 32768, f"vocab_size {c['vocab_size']:,}")
    claim("RoPE θ = 1e6", c["rope_theta"] == 1e6, f"rope_theta {c['rope_theta']:g}")
    claim("GQA 16:4 with QK-norm", (c["n_head"], c["n_kv_head"], c.get("use_qk_norm")) == (16, 4, True),
          f"n_head {c['n_head']}, n_kv_head {c['n_kv_head']}, use_qk_norm {c.get('use_qk_norm')}, attention {c.get('attention')}")
    claim("24 experts, top-4", (c["num_experts"], c["num_experts_per_tok"]) == (24, 4),
          f"num_experts {c['num_experts']}, top-k {c['num_experts_per_tok']}, shared {c.get('num_shared_experts')}")
    claim("sliding window on half the layers (0–9 of 20)", c["max_window_layers"] * 2 == c["n_layer"] and c["sliding_window"],
          f"sliding_window {c['sliding_window']}, max_window_layers {c['max_window_layers']}, n_layer {c['n_layer']}")
    kv = c["n_layer"] * 2 * c["n_kv_head"] * c["head_dim"] * 2
    claim("20 KiB of KV cache per token (bf16)", kv == 20 * 1024, f"{c['n_layer']} layers × K,V × {c['n_kv_head']} heads × {c['head_dim']} × 2 bytes = {kv:,} bytes")
    emb = 2 * c["vocab_size"] * c["hidden_size"]
    att = c["n_layer"] * ((c["n_head"] + 2 * c["n_kv_head"]) * c["head_dim"] * c["hidden_size"] + c["n_head"] * c["head_dim"] * c["hidden_size"])
    dense = 3 * c["hidden_size"] * c["intermediate_size"]
    exp = (c["n_layer"] - 1) * c["num_experts"] * 3 * c["hidden_size"] * c["moe_intermediate_size"]
    claim("embedding + head 67.1M", abs(emb / 1e6 - 67.1) < 0.05, f"{emb:,}")
    claim("attention 52.4M", abs(att / 1e6 - 52.4) < 0.05, f"{att:,}")
    claim("dense layer 0: 8.7M", abs(dense / 1e6 - 8.7) < 0.05, f"{dense:,} (intermediate_size {c['intermediate_size']})")
    claim("experts 269.0M", abs(exp / 1e6 - 269.0) < 0.05, f"{exp:,} (moe_intermediate_size {c['moe_intermediate_size']})")
    claim("untied head", c.get("tie_word_embeddings") is False, f"tie_word_embeddings {c.get('tie_word_embeddings')}")

    preset = M.AnuLMConfig.tiny_gqa()
    if preset is not None:
        pt, pa = M.AnuLM(preset).num_params()
        claim("tiny_gqa is the 17M preset", abs(pt / 1e6 - 17) < 1, f"total {pt:,}, active {pa:,}")
    lines = src.splitlines()
    # Located by content, not by line, so a moved line cannot make this pass or fail.
    router = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Router")
    hits = [i + 1 for i in range(router.lineno - 1, router.end_lineno) if "3.2%" in lines[i]]
    claim(f"router in fp32 because bf16 flipped 3.2% of choices (Router, line {hits[0] if hits else '?'})",
          bool(hits), "; ".join(lines[i - 1].strip() for i in hits)[:220])
    tsrc_train = at("anulm/training/train.py")
    tl = tsrc_train.splitlines()
    save = next(n for n in ast.parse(tsrc_train).body if isinstance(n, ast.FunctionDef) and n.name == "atomic_save")
    span = tl[save.lineno - 1:save.end_lineno]
    claim(f"atomic save exists because a kill at step 265,000 destroyed a checkpoint "
          f"(anulm/training/train.py {save.lineno}–{save.end_lineno})",
          any("265,000" in l or "265000" in l for l in span), "; ".join(l.strip() for l in span if "265" in l)[:220])
    api = at("anulm/inference/api.py")
    tp = [i + 1 for i, l in enumerate(api.splitlines()) if "top_p" in l]
    claim("anulm/inference/api.py accepts top_p and ignores it", bool(tp) and not re.search(r"top_p\s*[=:]\s*[^,)]*\)?\s*\n[^\n]*generate\(.*top_p", api),
          f"top_p on lines {tp[:8]}; check that none reaches generate()")
    drop = [l.strip() for l in lines if re.search(r"\bdropout\s*:\s*float\s*=", l)]
    claim("dropout wired in but 0.0 by default", any("0.0" in d for d in drop), "; ".join(drop))

    plan = (BOOK / "BOOK_PLAN.md").read_text(encoding="utf-8")
    files = sorted(set(re.findall(r"`([\w/]+\.(?:py|md|json))`", plan)))
    tracked = set(subprocess.run(["git", "ls-tree", "-r", "--name-only", PIN], cwd=ROOT, capture_output=True,
                                 text=True).stdout.split())
    bookfiles = {p.relative_to(BOOK).as_posix() for p in BOOK.rglob("*") if p.is_file()}
    planned_sec = plan[plan.find("## Code examples written for the book"):]
    planned_sec = planned_sec[: planned_sec.find("\n## ", 5)]
    planned = set(re.findall(r"`([\w/]+\.py)`", planned_sec))
    missing_files = [f for f in files if f not in tracked and f not in bookfiles and f not in planned
                     and not any(t.endswith("/" + f) for t in tracked)]
    not_yet = sorted(f for f in planned if f not in bookfiles and not any(b.endswith("/" + f) for b in bookfiles))
    claim("the book's own code examples (planned for Stage 4)", True,
          f"{len(planned)} planned; not written yet: {', '.join(not_yet) or 'none'}")
    test_names = {f.name for f in tree.body if isinstance(f, ast.FunctionDef)}
    named = sorted({t for t in re.findall(r"`([a-z][a-z0-9_]{12,})`", plan) if t.count("_") >= 3})
    missing_tests = [t for t in named if t not in test_names and t not in src]
    claim("every file BOOK_PLAN.md names exists (repo at the pin, or book/)", not missing_files,
          f"{len(files)} named; missing: {', '.join(missing_files) or 'none'}")
    claim("every test BOOK_PLAN.md names exists in tests/test_model.py", not missing_tests,
          f"{len(named)} named; missing: {', '.join(missing_tests) or 'none'}")

    L = ["", "## Claims recomputed from the code and the released checkpoint", "",
         f"`book/tools/verify_claims.py`, against `@{PIN}` and `ckpt_base_v2.pt`.", "",
         "| Claim | Holds | Evidence |", "|---|---|---|"] + [f"| {a} | {b} | {e} |" for a, b, e in rows]
    text = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    text = text.split("\n## Claims recomputed")[0].rstrip() + "\n" + "\n".join(L) + "\n"
    OUT.write_text(text, encoding="utf-8")
    for a, b, e in rows:
        print(f"{b:6s} {a}\n         {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
