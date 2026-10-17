"""
The same AnuLM API through the official OpenAI SDK and LangChain, unchanged
except for base_url. Start the server first (python api.py), then:

    python examples/api_demo/openai_client.py
    python examples/api_demo/openai_client.py --base-url http://10.0.0.5:8000/v1

The output is saved to responses/openai_client.txt.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    a = p.parse_args()
    key = os.environ.get("ANULM_API_KEY", "not-needed")       # any string when the server has no key
    client = OpenAI(base_url=a.base_url, api_key=key)
    log = io.StringIO()

    def show(title, text):
        block = f"### {title}\n{text}\n"
        print(block)
        log.write(block + "\n")

    show("models.list()", ", ".join(m.id for m in client.models.list().data))

    r = client.chat.completions.create(
        model="anulm-chat", max_tokens=60, temperature=0.7, seed=7,
        messages=[{"role": "user", "content": "Give me three tips for saving money every month."}],
        extra_body={"repetition_penalty": 1.3})                # AnuLM extension, passed through
    show("chat.completions.create(anulm-chat)", f"{r.choices[0].message.content}\nusage: {r.usage}")

    parts = []
    for chunk in client.chat.completions.create(
            model="anulm-chat", max_tokens=40, temperature=0.7, seed=7, stream=True,
            messages=[{"role": "user", "content": "भारत के बारे में एक वाक्य लिखिए।"}],
            extra_body={"repetition_penalty": 1.3}):
        if chunk.choices and chunk.choices[0].delta.content:
            parts.append(chunk.choices[0].delta.content)
    show("chat.completions.create(stream=True)", f"{len(parts)} chunks: {''.join(parts)}")

    r = client.chat.completions.create(model="anulm-answer",
                                       messages=[{"role": "user", "content": "Who built the Taj Mahal?"}])
    show("chat.completions.create(anulm-answer)",
         f"{r.choices[0].message.content}  (sources: {getattr(r, 'sources', None)})")

    img = base64.b64encode((HERE / "data" / "astronaut.jpg").read_bytes()).decode()
    r = client.chat.completions.create(model="anulm-vision", messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}]}])
    show("chat.completions.create(anulm-vision, image only)", r.choices[0].message.content)

    r = client.completions.create(model="anulm-chat", prompt="The three primary colours are",
                                  max_tokens=32, temperature=0, stop=["."])
    show("completions.create()", repr(r.choices[0].text))

    wav = HERE / "responses" / "openai_speech.wav"
    wav.parent.mkdir(exist_ok=True)
    speech = client.audio.speech.create(model="system-tts", voice="alloy", response_format="wav",
                                        input="Please send me a new debit card.")
    wav.write_bytes(speech.content)
    with wav.open("rb") as f:
        t = client.audio.transcriptions.create(model="anulm-asr", file=f, language="en")
    show("audio.speech.create() -> audio.transcriptions.create()",
         f"said:  Please send me a new debit card.\nheard: {t.text}")

    try:
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    except Exception as e:                                     # openai.NotFoundError
        show("an unknown model raises", f"{type(e).__name__}: {e}")

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="anulm-answer", base_url=a.base_url, api_key=key)
        show("LangChain ChatOpenAI(anulm-answer).invoke()",
             llm.invoke("What is the capital of Japan?").content)
    except ImportError:
        show("LangChain", "langchain-openai is not installed; skipped")

    (HERE / "responses" / "openai_client.txt").write_text(log.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
