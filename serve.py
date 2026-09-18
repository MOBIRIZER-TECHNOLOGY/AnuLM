"""
A tiny web UI for talking to a AnuLM checkpoint.

    python serve.py --ckpt ckpt_hindi_mixed18k.pt
    # then open http://localhost:8000

Stdlib only (http.server + json), no Flask. Loads the checkpoint once, serves
web/index.html at /, model facts at /info, and generation at POST /generate
with a JSON body {prompt, max_tokens, temperature, top_k, seed}.

What to expect depends on the checkpoint, and the page adapts from /info:

  * a base model continues whatever you type (Hindi article style);
  * a question-tuned or translation checkpoint offers those modes;
  * the Python coder (ckpt_coder_sft.pt, docs/MODEL_CARD.md) offers
    "write a function" (the MBPP setting, 12.5% pass@1) and "continue the
    code" (the HumanEval setting), greedy by default as it was evaluated.

Every checkpoint invents things; run the code, check the facts.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from make_qa import template_for, translation_lang
from model import AnuLM, load_checkpoint

HERE = Path(__file__).parent
INDEX = HERE / "web" / "index.html"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass


class Engine:
    """One loaded checkpoint plus a lock: the GPU model is not re-entrant."""

    def __init__(self, ckpt: str, device: str):
        ck = load_checkpoint(ckpt, device)          # a .pt, or a folder from export_hf.py
        cfg = ck["cfg"]
        if device.startswith("cuda"):
            cfg = replace(cfg, moe_impl="grouped")
        self.cfg, self.device = cfg, device
        self.model = AnuLM(cfg).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.tok = None
        if getattr(cfg, "tokenizer_path", None):
            from bpe import BPE
            self.tok = BPE.load(cfg.tokenizer_path)
            self.eos_id = self.tok.eos_id if cfg.vocab_size > self.tok.eos_id else None
        else:
            self.eos_id = 257
        # A fine-tuned checkpoint (finetune.py) carries the prompt template it
        # was trained on; with it the page offers a question mode.
        self.qa_template = ck.get("qa_template")
        # Newer fine-tunes carry one template per language; older ones a
        # single Hindi template, which template_for falls back to.
        self.qa_templates = ck.get("qa_templates") or ({"hi": self.qa_template} if self.qa_template else None)
        # The coder was tuned with the code32k tokenizer on Python pairs. It
        # inherits every template in make_qa.PROMPTS (finetune.py saves them
        # all), so without this flag the page would offer translation.
        tok_name = Path(cfg.tokenizer_path).stem if self.tok else ""
        self.coder = (tok_name.startswith("code")
                      or str(ck.get("base_ckpt") or "").startswith("ckpt_coder")
                      or Path(ckpt).name.startswith("ckpt_coder"))
        total, active = self.model.num_params()
        self.info = {
            "qa_template": self.qa_template, "base_ckpt": ck.get("base_ckpt"),
            "coder": self.coder,
            "translate": bool(self.qa_templates and "en-hi" in self.qa_templates and not self.coder),
            "ckpt": Path(ckpt).name, "step": ck.get("step"), "val_loss": ck.get("val_loss"),
            "params_total_m": round(total / 1e6, 1), "params_active_m": round(active / 1e6, 1),
            "attn": cfg.attn, "layers": cfg.n_layer, "experts": cfg.num_experts,
            "top_k": cfg.num_experts_per_tok, "block_size": cfg.block_size,
            "tokenizer": Path(cfg.tokenizer_path).stem if self.tok else "bytes",
            "vocab": cfg.vocab_size, "device": device,
        }
        self.lock = threading.Lock()
        # Warm up: the first CUDA call and the grouped-GEMM weight stack are slow.
        self.generate("def f(" if self.coder else "भारत", 4, 0.8, 50, 0)



    def encode(self, s: str) -> list[int]:
        if self.tok:
            return self.tok.encode(s) or self.tok.encode(" ")
        return list(s.encode("utf-8")) or [10]

    def decode(self, ids) -> str:
        if self.tok:
            return self.tok.decode(ids)
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_k: int, seed: int,
                 mode: str = "continue", repetition_penalty: float = 1.0):
        if mode == "translate" and self.qa_templates and "en-hi" in self.qa_templates and not self.coder:
            prompt = self.qa_templates[translation_lang(prompt)].format(q=" ".join(prompt.split()))
        elif mode == "question" and self.qa_templates:
            prompt = template_for(prompt.strip(), self.qa_templates).format(q=prompt.strip())
        elif mode in ("question", "translate"):
            mode = "continue"                       # no template: report what actually ran
        ids = self.encode(prompt)[-(self.cfg.block_size - 1):]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self.lock:
            torch.manual_seed(seed)
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                out = self.model.generate(x, max_tokens, temperature=max(temperature, 1e-3),
                                          top_k=top_k or None, eos_id=self.eos_id,
                                          repetition_penalty=repetition_penalty)
            dt = time.time() - t0
        new = out[0, len(ids):].tolist()
        stopped = self.eos_id is not None and self.eos_id in new
        text = self.decode(new)
        if mode == "translate":
            text = text.strip().split("\n")[0]      # one sentence out
        elif mode == "question" and not self.coder:
            # A tuned model ends its answer with EOS; only when it ran on
            # without EOS is the first line the answer.
            text = text.strip() if stopped else text.strip().split("\n")[0]
        elif mode == "question":
            text = text.strip("\n")                  # code spans lines: keep all of it
        return {"prompt": prompt, "completion": text, "tokens": len(new),
                "seconds": round(dt, 2), "stopped_at_eos": stopped, "mode": mode}


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/info":
                self._json(engine.info)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/generate":
                return self._json({"error": "not found"}, 404)
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                prompt = str(req.get("prompt", "")).strip()
                if not prompt:
                    return self._json({"error": "empty prompt"}, 400)
                res = engine.generate(
                    prompt,
                    max_tokens=max(1, min(int(req.get("max_tokens", 150)), 400)),
                    temperature=float(req.get("temperature", 0.8)),
                    top_k=int(req.get("top_k", 50)),
                    seed=int(req.get("seed", 1337)),
                    mode=req.get("mode") if req.get("mode") in ("question", "translate") else "continue",
                    repetition_penalty=float(req.get("repetition_penalty", 1.0)),
                )
                self._json(res)
            except Exception as e:                       # report, don't die
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def log_message(self, fmt, *args):             # one quiet line per request
            print(f"{self.address_string()} {fmt % args}")

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt_hindi_mixed18k.pt")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"loading {args.ckpt} on {args.device} ...")
    engine = Engine(args.ckpt, args.device)
    i = engine.info
    print(f"  {i['params_total_m']}M params ({i['params_active_m']}M active), "
          f"{i['attn']} x{i['layers']}, {i['experts']} experts top-{i['top_k']}, "
          f"tokenizer {i['tokenizer']}, step {i['step']}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"serving on http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
