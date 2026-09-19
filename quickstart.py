"""Check this machine, then do the first useful thing on it.

    python quickstart.py            what works here, and the next command to run
    python quickstart.py --train    train a 17M model from scratch and sample it
    python quickstart.py --demo     download a released 400M checkpoint and talk to it

Written for someone who has just cloned the repository and does not yet know
whether their install is right. It needs nothing but the standard library to
*report*; torch and the rest are only needed for what it offers to run. Every
failure it finds prints the exact command that fixes it.

Nothing here is required to use the project -- docs/TUTORIAL.md is the guided
path and README.md is the reference. This is the shortcut.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CPU_WHEEL = "pip install --index-url https://download.pytorch.org/whl/cpu torch"
CUDA_WHEEL = "pip install torch --index-url https://download.pytorch.org/whl/cu128"

try:                                    # Devanagari in a Windows console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

OK, WARN, BAD = "ok", "--", "!!"


def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def n_tests() -> int:
    """Counted, not written down: this line has been wrong twice already,
    and counting decorators needs neither torch nor an import."""
    try:
        return sum(1 for line in (HERE / "test_model.py").read_text(encoding="utf-8").splitlines()
                   if line.strip() == "@test")
    except OSError:
        return 0


def row(status: str, what: str, detail: str) -> None:
    print(f"  [{status:2s}] {what:22s} {detail}")


def check() -> dict:
    """Print a report of this machine and return what the caller can rely on."""
    state = {"torch": False, "cuda": False, "grouped": False, "safetensors": False,
             "hub": False, "blockers": [], "hints": []}

    print(f"\nAnuLM quickstart -- {platform.system()} {platform.machine()}, "
          f"Python {sys.version.split()[0]}\n")
    print("environment")

    py = sys.version_info
    if py >= (3, 10):
        row(OK, "python", f"{py.major}.{py.minor}.{py.micro}")
    else:
        row(BAD, "python", f"{py.major}.{py.minor} -- 3.10 or newer is required")
        state["blockers"].append("Install Python 3.10+ (3.12 or 3.13 are the safe choices).")

    if not have("torch"):
        row(BAD, "torch", "not installed")
        state["blockers"].append(
            f"Install torch. CPU only:\n      {CPU_WHEEL}\n"
            f"    NVIDIA GPU (pick the wheel matching your driver at\n"
            f"    https://pytorch.org/get-started/locally/):\n      {CUDA_WHEEL}")
    else:
        import torch
        v = torch.__version__
        state["torch"] = True
        old = tuple(int(x) for x in v.split(".")[:2]) < (2, 1)
        row(BAD if old else OK, "torch", v)
        if old:
            state["blockers"].append(
                f"torch {v} is too old: both attention paths pass scale= to "
                f"scaled_dot_product_attention, which needs 2.1+.\n      {CPU_WHEEL}")

        if torch.cuda.is_available():
            state["cuda"] = True
            free = ""
            try:
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                free = f", {total:.0f} GB"
            except Exception:
                pass
            row(OK, "cuda", f"{torch.cuda.get_device_name(0)}{free}")
        elif "+cpu" in v:
            row(WARN, "cuda", "no -- this is the CPU-only wheel")
            state["hints"].append(
                "You have the CPU wheel. That is fine for everything small, but if\n"
                "    this machine has an NVIDIA card it is idle. Check with nvidia-smi,\n"
                f"    then: {CUDA_WHEEL} --force-reinstall")
        else:
            row(WARN, "cuda", "no -- CPU only")

        state["grouped"] = hasattr(torch.nn.functional, "grouped_mm")
        row(OK if state["grouped"] else WARN, "grouped_mm",
            "yes -- --moe-impl grouped available" if state["grouped"]
            else f"no (torch {v} < 2.11) -- use the default --moe-impl sparse")

    state["safetensors"] = have("safetensors")
    row(OK if state["safetensors"] else WARN, "safetensors",
        "yes" if state["safetensors"] else "no -- needed to load an exported/downloaded model")
    if not state["safetensors"]:
        state["hints"].append("pip install -r requirements.txt   # safetensors, and pyarrow for the converters")

    state["hub"] = have("huggingface_hub")
    row(OK if state["hub"] else WARN, "huggingface_hub",
        "yes" if state["hub"] else "no -- needed only to download the released weights")

    for mod, why in (("pyarrow", "the parquet converters and the MBPP loader"),
                     ("gradio", "app.py, the optional Gradio demo")):
        row(OK if have(mod) else WARN, mod, "yes" if have(mod) else f"no -- optional, for {why}")

    print("\nrepository")
    for f, what in (("model.py", "the architecture"), ("train.py", "the training loop"),
                    ("test_model.py", "the test suite")):
        row(OK if (HERE / f).exists() else BAD, f, what)
        if not (HERE / f).exists():
            state["blockers"].append(f"{f} is missing -- run this from the repository root.")
    toks = sorted((HERE / "data").glob("*.json")) if (HERE / "data").is_dir() else []
    row(OK if toks else WARN, "data/*.json",
        f"{len(toks)} tokenizers" if toks else "none -- they ship with the repo; re-clone if missing")
    return state


def report(state: dict) -> bool:
    """Print what to do next. Returns True if the machine is ready to run."""
    if state["blockers"]:
        print("\nfix these first\n")
        for b in state["blockers"]:
            print(f"  *  {b}")
        return False
    if state["hints"]:
        print("\nworth knowing\n")
        for h in state["hints"]:
            print(f"  *  {h}")

    gpu = state["cuda"]
    print("\nwhat you can run now\n")
    print("  python quickstart.py --train      train a 17M model from scratch"
          f"  ({'~30 s' if gpu else '3-6 min'})")
    if state["safetensors"] and state["hub"]:
        print("  python quickstart.py --demo       download a real 400M checkpoint and talk to it")
    print(f"  python test_model.py              {n_tests()} tests, ~2 min")
    print("  python model.py                   build both presets, forward + backward, no data")
    if not gpu:
        print("\n  The 30b and 105b presets run on any CPU. The 400M presets need a GPU")
        print("  (8 GB with --grad-ckpt); on CPU you can still *run* the released")
        print("  checkpoints, at a few tokens per second.")
    print("\n  docs/TUTORIAL.md goes from here to every checkpoint in the project.")
    return True


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd, cwd=HERE)


def train(state: dict) -> int:
    """The smallest honest run: 200 steps of the Sarvam-30B-shaped preset."""
    if not state["torch"]:
        print("\ntorch is not installed; see above.")
        return 1
    print("\n" + "=" * 70)
    print("Training nano_30b (17.4M parameters, 6.6M active per token) for 200")
    print("steps on tinyshakespeare, which train.py downloads if it is not there.")
    print("Expect val loss around 1.85, and expert imbalance that RISES to 4-8x")
    print("before falling under 2x -- that rise is correct, and README.md")
    print('"Status: trained and verified" explains why.')
    print("=" * 70)
    rc = run([sys.executable, "-u", "train.py", "--preset", "30b", "--steps", "200",
              "--eval-every", "50", "--out", "ckpt_quickstart.pt"])
    if rc:
        print("\nTraining failed. The full log is above; docs/TUTORIAL.md has a")
        print("troubleshooting section for the usual causes.")
        return rc
    rc = run([sys.executable, "sample.py", "--ckpt", "ckpt_quickstart.pt",
              "--prompt", "KING RICHARD II:", "--tokens", "120"])
    print("\nThat is a 17M model after 200 steps: Shakespeare-shaped, and made of")
    print("non-words. 1,000 steps reaches val 1.52 and reads far better:")
    print("\n  python train.py --preset 30b --steps 1000 --out ckpt.pt")
    print("\nNext: docs/TUTORIAL.md section 4 trains on Hindi instead, and section 5")
    print("is the recipe every released checkpoint uses.")
    return rc


def demo(state: dict) -> int:
    """Fetch a real checkpoint and serve it, which is the shortest route to output."""
    missing = [m for m, ok in (("safetensors", state["safetensors"]),
                               ("huggingface_hub", state["hub"])) if not ok]
    if missing:
        print(f"\nNeed {' and '.join(missing)} first:\n")
        print("  pip install -r requirements.txt huggingface_hub")
        return 1
    repo = "toonist/AnuLM-Coder-400M"
    print(f"\nDownloading {repo} (~0.8 GB, cached afterwards) ...")
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo)
    print(f"  -> {local}")
    print("\nA 398M mixture-of-experts model, 174M active per token, that writes")
    print("small Python functions (MBPP 12.5% pass@1). Starting the web page;")
    print("open http://127.0.0.1:8000 and pick 'write a function'. Ctrl+C to stop.")
    return run([sys.executable, "serve.py", "--ckpt", local])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train", action="store_true", help="train a 17M model and sample it")
    p.add_argument("--demo", action="store_true", help="download a released 400M checkpoint and serve it")
    args = p.parse_args()

    state = check()
    ready = report(state)
    if not ready:
        return 1
    if args.train:
        return train(state)
    if args.demo:
        return demo(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
