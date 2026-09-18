"""
Ask a checkpoint questions from the command line, the way serve.py does.

    python ask.py --ckpt ckpt_multi_qa.pt          # the built-in question set
    python ask.py --ckpt ckpt_translate.pt --mode translate
    python ask.py --ckpt ckpt_coder.pt "def add(a, b):"   # a base model just continues

Loads the model once and walks a list of prompts. A checkpoint carrying
`qa_templates` (anything finetune.py wrote) has each prompt wrapped in the
template for its language, picked by `make_qa.template_for` exactly as
serve.py picks it, and is decoded with a repetition penalty -- the setting
the web page uses, and the one these models need. A base checkpoint has no
templates, so the prompt goes in raw and the model continues it; that
contrast is the point of `--mode continue` on a tuned checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace

import torch

from model import AnuLM, load_checkpoint

QUESTIONS = {
    "hi": ["काशी क्या है?", "गंगा नदी के बारे में बताइए।", "क्रिकेट किसे कहते हैं?"],
    "en": ["What is Digimon?", "What is a volcano?"],
    "py": ["Write a Python function `add`: return the sum of two numbers.",
           "Write a Python function `is_prime`: return True if n is prime."],
}
TRANSLATE = ["The train to Mumbai leaves at six in the morning.",
             "Where is the nearest hospital?",
             "मुझे कल दिल्ली जाना है।"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("prompts", nargs="*", help="ask these instead of the built-in set")
    p.add_argument("--ckpt", default="ckpt_multi_qa.pt")
    p.add_argument("--mode", default="question", choices=["question", "translate", "continue"])
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--rep", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ck = load_checkpoint(args.ckpt, args.device)
    cfg = replace(ck["cfg"], moe_impl="grouped" if args.device.startswith("cuda") else ck["cfg"].moe_impl)
    model = AnuLM(cfg).to(args.device).eval()
    model.load_state_dict(ck["model"])

    templates = ck.get("qa_templates") or ({"hi": ck["qa_template"]} if ck.get("qa_template") else None)
    print(f"{args.ckpt}: step {ck['step'] + 1:,}, val {ck['val_loss']:.4f}, "
          f"base {ck.get('base_ckpt', '-')}, templates {sorted(templates) if templates else 'none (base model)'}\n")

    from bpe import BPE
    tok = BPE.load(cfg.tokenizer_path)
    eos_id = tok.eos_id if cfg.vocab_size > tok.eos_id else None

    if args.prompts:
        prompts = args.prompts
    elif args.mode == "translate":
        prompts = TRANSLATE
    else:
        prompts = [q for group in ("hi", "en", "py") for q in QUESTIONS[group]]

    mode = args.mode
    if mode != "continue" and not templates:
        print("(this checkpoint carries no templates, so every prompt is continued, not answered)\n")
        mode = "continue"

    for raw in prompts:
        prompt = raw
        if mode != "continue":
            from make_qa import template_for, translation_lang
            key = translation_lang(raw) if mode == "translate" else None
            tmpl = templates.get(key) if key else template_for(raw.strip(), templates)
            prompt = (tmpl or "{q}").format(q=" ".join(raw.split()) if mode == "translate" else raw.strip())
        torch.manual_seed(args.seed)
        ids = tok.encode(prompt) or tok.encode(" ")
        x = torch.tensor([ids], dtype=torch.long, device=args.device)
        t0 = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=args.device.startswith("cuda")):
            out = model.generate(x, args.tokens, temperature=args.temperature, top_k=args.top_k,
                                 eos_id=eos_id, use_cache=True, repetition_penalty=args.rep)
        new = out[0, len(ids):].tolist()
        answer = tok.decode(new)
        if mode == "translate":
            answer = answer.strip().split("\n")[0]
        stopped = eos_id is not None and eos_id in new
        print("=" * 72)
        print(f"Q: {raw}")
        print(f"A: {answer.strip()}")
        print(f"   [{len(new)} tokens, {time.time() - t0:.1f}s{', stopped at EOS' if stopped else ''}]")
        print()


if __name__ == "__main__":
    main()
