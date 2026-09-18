"""
Does the model write working Python? pass@1 on HumanEval and MBPP.

    python eval_code.py ckpt_coder.pt ckpt_coder_sft.pt --device cuda
    python eval_code.py ckpt_coder.pt --bench mbpp --limit 50

The standard code tests: each problem is a function to write, and the
generated code either passes the hidden tests when executed or it does not.
Greedy decoding, one sample per problem (pass@1), which is the number every
coding-model paper reports.

Two prompting modes, chosen automatically:
  continue  base checkpoints get the function signature and docstring and
            complete it; generation is cut at the first top-level statement
            after the function (the usual HumanEval stop sequences).
  question  fine-tuned checkpoints (those carrying qa_templates) are asked
            in the QA format and produce the whole function, ended by EOS.

Generated code is executed. It is the model's own output on a fixed public
test set, run in a subprocess with a timeout and no inherited environment,
in a scratch directory -- but it is still arbitrary code; do not point this
at a checkpoint you do not trust.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch

from bpe import BPE
from make_qa import PROMPTS, template_for
from model import AnuLM, load_checkpoint

HUMANEVAL = "data/raw/openai__openai_humaneval/openai_humaneval/test-00000-of-00001.parquet"
MBPP = "data/raw/google-research-datasets__mbpp/sanitized/test-00000-of-00001.parquet"
STOPS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\nassert "]


def load_problems(bench: str) -> list[dict]:
    import pyarrow.parquet as pq
    if bench == "humaneval":
        rows = pq.read_table(HUMANEVAL).to_pylist()
        return [{"id": r["task_id"], "prompt": r["prompt"], "entry": r["entry_point"],
                 "test": r["test"] + f"\n\ncheck({r['entry_point']})\n"} for r in rows]
    rows = pq.read_table(MBPP).to_pylist()
    out = []
    for r in rows:
        entry = r["test_list"][0].split("(")[0].replace("assert ", "").strip()
        out.append({"id": str(r["task_id"]), "desc": r["prompt"], "entry": entry,
                    "prompt": f'"""\n{r["prompt"]}\n{chr(10).join(r["test_list"])}\n"""\n',
                    "test": "\n".join(r.get("test_imports") or []) + "\n" + "\n".join(r["test_list"]) + "\n"})
    return out


def run_program(src: str, workdir: str, timeout: float = 10.0) -> bool:
    path = os.path.join(workdir, "prog.py")
    Path(path).write_text(src, encoding="utf-8")
    try:
        r = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=timeout,
                           cwd=workdir, env={"PATH": os.environ.get("PATH", "")})
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def cut_at_stops(text: str) -> str:
    end = len(text)
    for s in STOPS:
        i = text.find(s)
        if i != -1:
            end = min(end, i)
    return text[:end]


@torch.no_grad()
def score(ckpt: str, problems: list[dict], device: str, max_new: int, workdir: str, verbose: int,
          mode_override: str | None = None) -> dict:
    ck = load_checkpoint(ckpt, "cpu")
    cfg = ck["cfg"]
    if device.startswith("cuda"):
        cfg = replace(cfg, moe_impl="grouped")
    m = AnuLM(cfg).to(device).eval()
    m.load_state_dict(ck["model"])
    tok = BPE.load(cfg.tokenizer_path)
    templates = ck.get("qa_templates")
    # `question` whenever the checkpoint knows a template, unless --mode forces
    # it. That default is right for MBPP, whose items are prose descriptions,
    # and wrong for HumanEval, whose native form IS a continuation: signature
    # plus docstring, complete the body. Measured on ckpt_coder_sft: 0.0% asked
    # as a question against 4.9% for the same weights' base asked to continue,
    # because in question mode the model writes a FRESH function rather than
    # continuing the given one. The default stays (it is what the checkpoint was
    # tuned for) but the other way is now reachable -- say which you used.
    mode = mode_override or ("question" if templates else "continue")
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))
    bs = cfg.block_size
    passed, t0 = 0, time.time()
    for i, pr in enumerate(problems):
        if mode == "question":
            q = ("Complete the following Python function.\n" + pr["prompt"] if "desc" not in pr
                 else f"Write a Python function `{pr['entry']}`: {pr['desc']}")
            text = {**PROMPTS, **templates}["en"].format(q=q)
        else:
            text = pr["prompt"]
        ids = tok.encode(text)[-(bs - max_new):]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with ac:
            out = m.generate(x, max_new, temperature=1.0, top_k=1, eos_id=tok.eos_id)
        gen = tok.decode(out[0, len(ids):].tolist())
        if mode == "question":
            body = gen.strip()
            program = body if f"def {pr['entry']}" in body else pr["prompt"] + body
        else:
            program = pr["prompt"] + cut_at_stops(gen)
        ok = run_program(program + "\n\n" + pr["test"], workdir)
        passed += ok
        if verbose and (ok or verbose > 1):
            print(f"\n--- {pr['id']} {'PASS' if ok else 'FAIL'}\n{program[:800]}")
        print(f"\r  {ckpt}: {i + 1}/{len(problems)}  pass {passed}  ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    return {"ckpt": ckpt, "mode": mode, "n": len(problems), "pass": passed, "pass_at_1": passed / len(problems)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--bench", choices=["humaneval", "mbpp"], default="humaneval")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mode", choices=["continue", "question"], default=None,
                   help="force the prompting style instead of inferring it from the "
                        "checkpoint; see the note in score()")
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--verbose", "-v", action="count", default=0, help="-v prints passing programs, -vv all")
    args = p.parse_args()
    problems = load_problems(args.bench)[: args.limit]
    print(f"{args.bench}: {len(problems)} problems, greedy, pass@1, device {args.device}\n")
    with tempfile.TemporaryDirectory() as workdir:
        results = [score(c, problems, args.device, args.max_new, workdir, args.verbose, args.mode) for c in args.ckpts]
    print(f"\n{'checkpoint':26s} {'mode':>9s} {'pass@1':>8s}")
    print("-" * 46)
    for r in results:
        print(f"{Path(r['ckpt']).name:26s} {r['mode']:>9s} {100 * r['pass_at_1']:7.1f}%   ({r['pass']}/{r['n']})")
    print("\nreference: CodeParrot-110M ~4%, SantaCoder-1.1B ~18%, phi-1-small-350M ~45% (HumanEval)")


if __name__ == "__main__":
    main()
