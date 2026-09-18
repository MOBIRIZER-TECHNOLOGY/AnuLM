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

Generated code is executed. That is the only way to score it, and it means
this script runs a language model's unreviewed output on your machine. What
contains it -- see `run_program` for the detail:

  * a fresh empty directory per problem, deleted afterwards, so nothing one
    program writes can be imported or read by the next (a generated file
    called `random.py` in a shared directory would shadow the standard
    library for every problem after it);
  * `python -I -B`: no environment variables, no user site-packages, no
    bytecode written. Not `-S`: dropping site-packages would fail any
    solution that imports numpy, changing the pass@1 this script exists to
    measure, and a sandbox that moves the number it guards is a bad trade;
  * an environment holding only PATH and SystemRoot, and no network setup;
  * a wall-clock timeout, and the whole process *tree* killed when it
    expires, so a spawned child cannot outlive it;
  * on Linux and macOS, address space, CPU time, file size, process count
    and core dumps are capped with `setrlimit`.

**On Windows there are no resource limits** -- `resource.setrlimit` does not
exist there, so a program can still exhaust memory or spawn processes until
the timeout kills the tree. The isolation, the per-problem directory and the
tree kill all work; the limits do not. Run untrusted checkpoints on Linux.

None of this is a security boundary against code that is trying to escape.
It is enough for a small model's attempts at `is_prime`, which is what this
script is for; do not point it at a checkpoint someone else trained.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


POSIX = os.name == "posix"
MEMORY_MB = 2048            # generated code has no business needing more
FILE_CAP = 16 * 1024 * 1024  # anything it writes, including its own output
MAX_PROCS = 64              # enough for a subprocess, not for a fork bomb


def _rlimits(timeout: float):
    """Applied in the child between fork and exec. POSIX only."""
    import resource

    def apply():
        os.setsid()                       # its own process group, so we can kill the tree
        b = MEMORY_MB * 1024 * 1024
        for what, limit in ((resource.RLIMIT_AS, b),
                            (resource.RLIMIT_DATA, b),
                            (resource.RLIMIT_CPU, int(timeout) + 1),
                            (resource.RLIMIT_FSIZE, FILE_CAP),
                            (resource.RLIMIT_NPROC, MAX_PROCS),
                            (resource.RLIMIT_CORE, 0)):
            try:
                resource.setrlimit(what, (limit, limit))
            except (ValueError, OSError):
                pass                      # a limit the platform will not set is not fatal
    return apply


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the process and anything it started. A bare proc.kill() leaves
    grandchildren running, which on a benchmark of 257 problems is how a
    machine ends up with a hundred orphaned interpreters."""
    if POSIX:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError):
            pass
    else:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, check=False)
    try:
        proc.kill()
    except ProcessLookupError:
        pass


def run_program(src: str, workdir: str, timeout: float = 10.0) -> bool:
    """Run `src` in a throwaway directory and say whether it exited 0.

    `workdir` is the parent; each call gets a fresh child directory inside it
    which is removed afterwards, so programs cannot see each other's files.
    """
    cell = tempfile.mkdtemp(dir=workdir)
    try:
        path = os.path.join(cell, "prog.py")
        Path(path).write_text(src, encoding="utf-8")
        # Output goes to a file, not a pipe: a program that prints in a loop
        # would otherwise fill this process's memory rather than its own, and
        # on POSIX the file is capped by RLIMIT_FSIZE.
        out_path = os.path.join(cell, "out.txt")
        env = {"PATH": os.environ.get("PATH", "")}
        if not POSIX:                       # Windows needs this to start python at all
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        kw = {}
        if POSIX:
            kw["preexec_fn"] = _rlimits(timeout)
        else:
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        with open(out_path, "wb") as sink:
            # -I (no environment, no user site) and -B (no .pyc left behind),
            # but deliberately NOT -S: that would drop site-packages, so a
            # solution importing numpy would fail here and pass everywhere
            # else. The published pass@1 numbers were measured without it, and
            # a sandbox that quietly changes the score it is guarding is worse
            # than the isolation it buys.
            proc = subprocess.Popen([sys.executable, "-I", "-B", "prog.py"],
                                    cwd=cell, env=env, stdin=subprocess.DEVNULL,
                                    stdout=sink, stderr=subprocess.STDOUT, **kw)
            try:
                return proc.wait(timeout=timeout) == 0
            except subprocess.TimeoutExpired:
                _kill_tree(proc)
                proc.wait(timeout=5)
                return False
    finally:
        shutil.rmtree(cell, ignore_errors=True)


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
