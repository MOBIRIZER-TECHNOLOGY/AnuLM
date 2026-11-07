"""Assemble the AnuLM-Omni Hugging Face Space as a folder, ready to upload.

    python tools/build_space_omni.py --out build/space_omni
    python tools/build_space_omni.py --out build/space_omni --results logs/run1_keep/demo_results.json

The folder is repo-shaped, because the demo finds its files relative to the
repository root (`anulm/apps/app_omni.py`: HERE = two levels up):

    app.py, README.md, requirements.txt     from space_omni/
    anulm/                                  the package, without __pycache__
    experiments/omni_one.py                 the recipe: the page reads its release gates
    examples/samples/                       the one-click sample library (about 10 MB)
    results.json                            the served model's measured results (the page's numbers)

Nothing is uploaded; `huggingface-cli upload <space> build/space_omni . --repo-type space`
(or HfApi.upload_folder) does that, once someone decides to publish.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATED = ("chat_heldout_loss", "wer_en", "wer_hi_mean", "intent_banking77", "read_squad_en", "vqa_acc")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "build" / "space_omni"))
    ap.add_argument("--results", default=str(ROOT / "logs" / "run1_keep" / "demo_results.json"),
                    help="the served model's results (exported copy), with wer_hi_mean")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in ("app.py", "README.md", "requirements.txt"):
        shutil.copy2(ROOT / "space_omni" / f, out / f)
    shutil.copytree(ROOT / "anulm", out / "anulm", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # The page reads the release gates and the task mix from the recipe itself.
    (out / "experiments").mkdir()
    shutil.copy2(ROOT / "experiments" / "omni_one.py", out / "experiments" / "omni_one.py")
    samples = ROOT / "examples" / "samples"
    shutil.copytree(samples, out / "examples" / "samples",
                    ignore=shutil.ignore_patterns("results.json", "results.md"))
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    missing = [k for k in GATED if k not in res]
    if missing:
        raise SystemExit(f"{a.results} lacks {missing}: the page's numbers would be blank")
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    files = sum(1 for p in out.rglob("*") if p.is_file())
    print(f"{out}: {files} files, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
