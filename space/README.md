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
  - toonist/AnuLM-Base-2K-400M
---

# AnuLM demo

A 398M-parameter mixture-of-experts language model (174M active per
token) in the shape of Sarvam 30B, trained from scratch on one consumer
GPU. The page holds all five models below in a dropdown and loads one on
demand, one at a time, on the free CPU tier at a few tokens per second. Set
the Space variable `ANULM_REPO` to preselect one; without it the page opens
on the picker with nothing loaded, which costs nothing until someone
chooses.

- Code, tutorial, recipes: https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM
- Weights and model cards: the four models linked above
- Independent academic project; not affiliated with Sarvam AI, BharatGen,
  AI4Bharat or the Government of India. The model invents things: run the
  code, check the facts.
