---
title: AnuLM
emoji: ⚛️
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.28.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: Small Sarvam-shaped MoE for Hindi, English and Python
models:
  - toonist/AnuLM-Coder-400M
  - toonist/AnuLM-Translate-400M
  - toonist/AnuLM-Hindi-QA-400M
  - toonist/AnuLM-Base-400M
---

# AnuLM demo

A 398M-parameter mixture-of-experts language model (174M active per
token) in the shape of Sarvam 30B, trained from scratch on one consumer
GPU. This Space loads the checkpoint named by the Space variable
`ANULM_REPO` — set it to one of the four models below, e.g.
`toonist/AnuLM-Coder-400M` — and runs it on the free CPU tier, at a few
tokens per second. Without that variable `app.py` looks for a local
`ckpt_coder_sft.pt` instead, which is the laptop case, not the Space one.

- Code, tutorial, recipes: https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM
- Weights and model cards: the four models linked above
- Independent academic project; not affiliated with Sarvam AI, BharatGen,
  AI4Bharat or the Government of India. The model invents things: run the
  code, check the facts.
