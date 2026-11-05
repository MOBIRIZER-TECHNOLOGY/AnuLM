"""Blend the trained Omni model back toward the chat model it started from.

    python experiments/blend_weights.py --alpha 0.1 --out ckpt_omni_blend10.pt
    python experiments/blend_weights.py --eval 0.05 0.1 0.15 0.2     # make and measure each

Every shared weight becomes (1 - alpha) * omni + alpha * chat. Two kinds of
weight are kept from Omni untouched: the embedding and output rows for the
audio and image ids (the chat model only has their initial values), and the
parts the chat model does not have at all (audio and vision projectors).

Why: the 2026-11-05 run passed every release gate but chat, by 0.009 (held-out
loss 3.309 against 3.30). Nearly all of chat's rise came in the first 625
steps, while the other tasks moved the shared decoder. Mixing a little of the
starting weights back usually recovers some of the original skill at a small
cost to the new ones. The evaluation says whether that cost stays inside the
other gates.

--eval runs the text and image evaluation (omni_eval.py) for each alpha,
writing logs/blend/<alpha>.json; speech is measured only with --speech,
because it takes about 35 minutes per model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))      # the checkpoints pickle anulm config classes
LOGS = ROOT / "logs" / "blend"
SPEECH = (("wer_en", "data/speech_ls100", 100), ("wer_hi_fleurs", "data/fleurs_hi/devtest", 1000),
          ("wer_hi_kathbath", "data/kathbath_hi/test", 800), ("wer_hi_mucs", "data/mucs_hi_en/test", 800))


def blend(omni_path: Path, chat_path: Path, alpha: float, out: Path) -> None:
    omni = torch.load(omni_path, map_location="cpu", weights_only=False)
    chat = torch.load(chat_path, map_location="cpu", weights_only=False, mmap=True)
    text_rows = int(omni["speech_text_vocab"])
    sd, ref = omni["model"], chat["model"]
    if set(sd) != set(ref):
        raise SystemExit("the two checkpoints do not have the same decoder weights")
    for k, w in sd.items():
        r = ref[k].to(w.dtype)
        if r.shape != w.shape:
            raise SystemExit(f"{k}: {tuple(r.shape)} vs {tuple(w.shape)}")
        if not w.is_floating_point():
            continue
        if k in ("embed_tokens.weight", "lm_head.weight"):
            w[:text_rows] = (1 - alpha) * w[:text_rows] + alpha * r[:text_rows]
        else:
            w.copy_((1 - alpha) * w + alpha * r)
    omni["blend"] = {"alpha": alpha, "omni": str(omni_path), "chat": str(chat_path)}
    torch.save(omni, out)


def run(args: list[str], log: Path) -> str:
    r = subprocess.run([sys.executable, "-u", *args], cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    log.write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise SystemExit(f"failed; see {log}")
    return r.stdout


def measure(ckpt: str, tag: str, speech: bool) -> dict:
    out = LOGS / f"{tag}.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if "chat_heldout_loss" not in res:
        run(["-m", "anulm.multimodal.omni_eval", "--ckpt", ckpt, "--vision", "--out", str(LOGS / f"{tag}.text.json")],
            LOGS / f"{tag}.text.log")
        res.update({k: v for k, v in json.loads((LOGS / f"{tag}.text.json").read_text(encoding="utf-8")).items()
                    if k not in ("ckpt", "seconds") and not isinstance(v, list)})
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for name, prefix, limit in SPEECH if speech else ():
        if name not in res:
            text = run(["-m", "anulm.multimodal.speech_encoder", "eval", "--ckpt", ckpt, "--prefix", prefix,
                        "--limit", str(limit)], LOGS / f"{tag}.{name}.log")
            res[name] = float(re.search(r"WER ([\d.]+)%\s+over", text).group(1))
            out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    hi = [res.get(k) for k in ("wer_hi_fleurs", "wer_hi_kathbath", "wer_hi_mucs")]
    if all(v is not None for v in hi):
        res["wer_hi_mean"] = round(sum(hi) / 3, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omni", default="ckpt_omni.pt")
    ap.add_argument("--chat", default="ckpt_chat_mm.pt")
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--out")
    ap.add_argument("--eval", type=float, nargs="+", metavar="ALPHA")
    ap.add_argument("--eval-path", help="measure this checkpoint or exported folder as it is (no blending)")
    ap.add_argument("--tag", default="model", help="name of the --eval-path results in logs/blend/")
    ap.add_argument("--speech", action="store_true", help="also measure the four speech test sets")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.alpha is not None:
        blend(Path(a.omni), Path(a.chat), a.alpha, Path(a.out or f"ckpt_omni_blend{round(a.alpha * 100)}.pt"))
    LOGS.mkdir(parents=True, exist_ok=True)
    if a.eval_path:
        res = measure(a.eval_path, a.tag, a.speech)
        print(f"{a.eval_path}: " + "  ".join(f"{k} {v}" for k, v in res.items()), flush=True)
    for alpha in a.eval or []:
        tag = f"blend{round(alpha * 100):02d}"
        ckpt = f"ckpt_omni_{tag}.pt"
        if not Path(ckpt).exists():
            blend(Path(a.omni), Path(a.chat), alpha, Path(ckpt))
        res = measure(ckpt, tag, a.speech)
        print(f"alpha {alpha:.2f}: " + "  ".join(f"{k} {v}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
