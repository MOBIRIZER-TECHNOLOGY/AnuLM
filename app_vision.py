"""
Show AnuLM a picture: the captioning demo.

    pip install gradio timm pillow
    python app_vision.py                              # loads ckpt_caption_f8k.pt
    python app_vision.py --ckpt ckpt_caption_p1.pt     # a different checkpoint
    python app_vision.py --host 0.0.0.0               # reachable on the same wifi

This is the only multimodal path in the project that works, and until now there
was no way to look at it. `docs/SPEECH.md` has the numbers: on held-out
Flickr8k images it produced "A brown dog is running on the beach" for a
reference reading "A brown dog is running along a beach", recognised a skier as
skiing, and got roughly five of ten clearly right.

The pipeline is three pieces and only the middle one was trained here:

    image -> SigLIP ViT-B/16 (frozen) -> projector (12.6M, trained) -> AnuLM

**What to expect.** It gets the subject and the setting reliably -- dog, boy,
skier, snow, beach, field -- and often the action. It hallucinates colours, and
falls back on "playing with a toy" when unsure. That is a 353M model with a
12.6M projector trained for sixteen minutes on six thousand photographs, so the
surprise is that it works at all rather than that it errs.

**Why this works when speech does not.** Captioning emits about twelve tokens
in a vocabulary the model already speaks, and needs only the projector. Speech
had to learn 28,672 new vocabulary entries from scratch and emit six hundred
tokens in a language it had never spoken. Same backbone, opposite outcomes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gradio as gr
import torch

HERE = Path(__file__).parent
EXAMPLE_DIR = HERE / "data" / "flickr8k_images"


def find_examples(n: int = 6) -> list[str]:
    """Held-out images if the corpus is on disk: the manifest's first 600 rows
    are the held-out slice, which is images 000000-000119, so anything below
    000120 is a picture the model was never trained on."""
    if not EXAMPLE_DIR.is_dir():
        return []
    files = sorted(EXAMPLE_DIR.glob("0000*.jpg"))[:120]
    step = max(1, len(files) // max(n, 1))
    return [str(f) for f in files[::step][:n]]


class Captioner:
    """Backbone, projector and frozen tower, loaded once."""

    def __init__(self, ckpt: str, device: str):
        from vision_encoder import load_trained
        self.device = device
        self.vl, self.tower, self.tok = load_trained(ckpt, device)
        self.ckpt = ckpt
        total, active = self.vl.model.num_params()
        pool = self.vl.proj.pool
        patches = self.tower.encode(find_examples(1)[0]).shape[0] if find_examples(1) else 196
        self.info = (
            f"**{Path(ckpt).name}** · {total/1e6:.0f}M params ({active/1e6:.0f}M active) · "
            f"tower `{self.tower.name}` frozen, {self.tower.dim}-dim · "
            f"projector {sum(p.numel() for p in self.vl.proj.parameters())/1e6:.1f}M, "
            f"pool {pool}×{pool} → {patches // (pool * pool)} of 2048 positions per image"
        )

    def caption(self, image, max_tokens: int) -> tuple[str, str]:
        from vision_encoder import caption as gen
        if image is None:
            return "", "*drop an image in, or pick an example*"
        t0 = time.time()
        text = gen(self.vl, self.tower, self.tok, image,
                   max_tokens=int(max_tokens), device=self.device)
        dt = time.time() - t0
        return (text or "(nothing)"), f"{dt:.2f}s · greedy · {int(max_tokens)} token budget"


def build(cap: Captioner) -> gr.Blocks:
    with gr.Blocks(title="AnuLM — vision") as demo:
        gr.Markdown("## AnuLM — show it a picture\n"
                    "A frozen SigLIP tower, a 12.6M projector trained on Flickr8k, "
                    "and the same 353M backbone that writes Python.")
        gr.Markdown(cap.info)
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="filepath", label="image", height=320)
                tokens = gr.Slider(8, 60, value=30, step=2, label="max tokens")
                go = gr.Button("Caption", variant="primary")
            with gr.Column():
                out = gr.Textbox(label="caption", lines=3)
                meta = gr.Markdown()
                gr.Markdown("*Subject and setting are usually right; colours are "
                            "often invented. See `docs/SPEECH.md` for the held-out "
                            "numbers and the failure modes.*")
        examples = find_examples()
        if examples:
            gr.Examples(examples=[[e] for e in examples], inputs=[img],
                        label="held-out images (never trained on)")
        go.click(cap.caption, [img, tokens], [out, meta])
        img.change(cap.caption, [img, tokens], [out, meta])
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="Caption images with AnuLM.")
    p.add_argument("--ckpt", default="ckpt_caption_f8k.pt",
                   help="a checkpoint from vision_encoder.py train")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7862)
    p.add_argument("--share", action="store_true", help="public gradio link (Colab)")
    args = p.parse_args()

    if not Path(args.ckpt).exists():
        raise SystemExit(f"no {args.ckpt}; train one with:\n"
                         f"  python vision_encoder.py train --ckpt ckpt_mm_init.pt "
                         f"--manifest data/flickr8k.jsonl --out {args.ckpt} --epochs 1")
    print(f"loading {args.ckpt} on {args.device}...")
    build(Captioner(args.ckpt, args.device)).launch(
        server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
