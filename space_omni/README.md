---
title: AnuLM-Omni-400M
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
license: cc-by-sa-4.0
models:
  - toonist/AnuLM-Omni-400M
short_description: One small model that chats, listens and sees (Hindi, English)
---

# AnuLM-Omni-400M: live demo

One 400M mixture-of-experts model, trained from scratch on one consumer GPU,
that chats, answers from passages, routes a message to an intent, transcribes
English and Hindi speech, and describes and answers questions about photos.
Every sample under each tab runs in one click, with the reference answer beside
the model's.

- Model and card: https://huggingface.co/toonist/AnuLM-Omni-400M (CC BY-SA 4.0)
- Code, training recipes, book and course: https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM (MIT)
- Hearing and seeing read frozen borrowed encoders (Whisper-small, MIT; SigLIP,
  Apache 2.0); the voice is Piper. Everything else is AnuLM's own weights.
- The Wikipedia lookup of the full demo needs a 1.3 GB index and is off here.

It is a 400M model: fluent and often wrong on facts. Check what it says.
