"""Run every main user flow once, end to end, on a tiny model on the CPU.

    python -m tools.smoke            # about 65-80 seconds on a laptop CPU
    python -m tools.smoke --keep     # keep the work folder to look at what each step wrote

The unit tests (`python -m tests.test_model`) check the parts; this checks that
the commands a newcomer types still connect: train a model, sample from it,
fine-tune it on question-answer pairs, ask it questions, score it, export it to
a Hugging Face folder and load that folder without this repository, and serve
it over HTTP. Each step runs as its own process, exactly as documented, and the
run ends with a pass/fail table. Nothing needs a GPU or a download except
tinyshakespeare (1 MB), which `train` fetches on first use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
# Children import this checkout, never another copy of anulm installed elsewhere.
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT)}


def run(cmd: list[str], timeout: float = 600) -> tuple[bool, str]:
    """Run one command from the repository root; (succeeded, last lines of its output)."""
    r = subprocess.run([PY, "-u", *cmd], cwd=ROOT, env=ENV, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    out = (r.stdout + r.stderr).strip()
    return r.returncode == 0 and "Traceback" not in out, out


def free_port() -> int:
    """A TCP port nothing is listening on, so the server step cannot collide."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(ckpt: Path) -> tuple[bool, str]:
    """Start the web server, request /info and one /generate, then stop it."""
    port = free_port()
    proc = subprocess.Popen([PY, "-u", "-m", "anulm", "serve", "--ckpt", str(ckpt), "--port", str(port),
                             "--device", "cpu"], cwd=ROOT, env=ENV, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):                       # the model loads before the port opens
            try:
                info = json.loads(urllib.request.urlopen(base + "/info", timeout=2).read())
                break
            except OSError:
                if proc.poll() is not None:
                    return False, proc.stdout.read()
                time.sleep(0.5)
        else:
            return False, "server did not answer /info within 60 s"
        body = json.dumps({"prompt": "KING RICHARD II:", "max_tokens": 12, "seed": 1}).encode()
        req = urllib.request.Request(base + "/generate", body, {"Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=60).read())
        ok = "error" not in res and bool(res)
        return ok, f"/info keys {sorted(info)[:6]} | /generate -> {json.dumps(res, ensure_ascii=False)[:160]}"
    finally:
        proc.terminate()
        proc.wait(timeout=30)


def export_loads_without_repo(folder: Path) -> tuple[bool, str]:
    """Load an exported folder through transformers with `anulm` made unimportable."""
    code = ("import sys, torch\n"
            "sys.modules['anulm'] = None\n"
            "from transformers import AutoModelForCausalLM\n"
            f"m = AutoModelForCausalLM.from_pretrained(r'{folder}', trust_remote_code=True)\n"
            "print('generated', tuple(m.generate(torch.tensor([[65, 66]]), max_new_tokens=4, do_sample=False).shape))\n")
    env = {k: v for k, v in ENV.items() if k != "PYTHONPATH"}
    r = subprocess.run([PY, "-c", code], cwd=folder, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300)
    out = (r.stdout + r.stderr).strip()
    return r.returncode == 0 and "generated" in out, out


def main() -> int:
    # The table quotes each step's output, which can hold Devanagari or other
    # characters a Windows console (cp1252) cannot encode: print them, never crash.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--keep", action="store_true", help="keep the work folder and print its path")
    args = p.parse_args()
    work = Path(tempfile.mkdtemp(prefix="anulm_smoke_"))
    ckpt, qa_ckpt = work / "ckpt.pt", work / "qa.pt"
    qa = work / "qa.jsonl"
    qa.write_text("".join(json.dumps({"question": f"What is {a} plus {b}?", "answer": str(a + b), "lang": "en"}) + "\n"
                          for a in range(8) for b in range(8)), encoding="utf-8")
    held = work / "held.jsonl"
    # The last row repeats a training question (in different case and spacing):
    # finetune must drop it from the held-out side and say so.
    held.write_text("".join(json.dumps({"question": q, "answer": a, "lang": "en"}) + "\n"
                            for q, a in [*((f"What is {a} plus {a}?", str(2 * a)) for a in range(8, 12)),
                                         ("what is 2  PLUS 3?", "5")]), encoding="utf-8")

    def finetune() -> tuple[bool, str]:
        ok, out = run(["-m", "anulm", "finetune", "--ckpt", str(ckpt), "--qa", str(qa), "--heldout", str(held),
                       "--out", str(qa_ckpt), "--epochs", "1", "--batch-size", "8", "--device", "cpu"])
        if ok and "held-out: dropped 1 of 5 rows" not in out:
            return False, out + "\nexpected finetune to drop the held-out row that repeats a training question"
        return ok, out

    def text_tail() -> list[str]:
        bench = work / "bench.txt"
        bench.write_text((ROOT / "data" / "input.txt").read_text(encoding="utf-8")[-20000:], encoding="utf-8")
        return ["-m", "anulm", "eval_bench", str(ckpt), "--bench", str(bench), "--device", "cpu"]

    corpus, tok = ROOT / "data" / "input.txt", work / "tok.json"
    steps = [
        ("train bytes (fetches data)", lambda: run(["-m", "anulm", "train", "--preset", "30b", "--steps", "10",
                                                    "--eval-every", "10", "--out", str(work / "bytes.pt"),
                                                    "--device", "cpu"])),
        ("train a BPE tokenizer", lambda: run(["-m", "anulm", "bpe", "train", "--data", str(corpus), "--vocab", "512",
                                               "--out", str(tok)])),
        ("encode the corpus", lambda: run(["-m", "anulm", "bpe", "encode", "--tokenizer", str(tok), "--data", str(corpus),
                                           "--out", str(work / "corpus.bin"), "--workers", "1"])),
        ("train 30 steps on BPE", lambda: run(["-m", "anulm", "train", "--preset", "30b", "--steps", "30",
                                               "--eval-every", "15", "--data", str(work / "corpus.bin"),
                                               "--out", str(ckpt), "--device", "cpu"])),
        ("sample", lambda: run(["-m", "anulm", "sample", "--ckpt", str(ckpt), "--prompt", "KING", "--tokens", "20",
                                "--device", "cpu"])),
        ("fine-tune on Q&A", finetune),
        ("ask questions", lambda: run(["-m", "anulm", "ask", "--ckpt", str(qa_ckpt), "--mode", "question",
                                       "--tokens", "8", "--device", "cpu", "What is 2 plus 3?"])),
        ("score held-out text", lambda: run(text_tail())),
        ("export to HF folder", lambda: run(["-m", "anulm", "export_hf", str(ckpt), str(work / "export"),
                                             "--license", "mit"])),
        ("load export without repo", lambda: export_loads_without_repo(work / "export")),
        ("serve over HTTP", lambda: serve(ckpt)),
    ]
    rows, t0 = [], time.time()
    for name, step in steps:
        t = time.time()
        try:
            ok, out = step()
        except Exception as e:                        # a crash in one step must not hide the others
            ok, out = False, f"{type(e).__name__}: {e}"
        rows.append((name, ok, time.time() - t, out.splitlines()[-1][:110] if out else ""))
        print(f"{'PASS' if ok else 'FAIL'}  {name:26s} {rows[-1][2]:6.1f}s  {rows[-1][3]}", flush=True)
        if not ok:
            print("      " + "\n      ".join(out.splitlines()[-15:]), flush=True)
    failed = [r[0] for r in rows if not r[1]]
    print(f"\n{len(rows) - len(failed)} of {len(rows)} steps passed in {time.time() - t0:.0f} s"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    if args.keep:
        print("work folder:", work)
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
