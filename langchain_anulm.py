"""
LangChain helpers for the AnuLM API (api.py).

    from langchain_anulm import AnuLMDecide, anulm_chat

    llm = anulm_chat()                                   # a ChatOpenAI on anulm-chat
    route = AnuLMDecide(model="anulm-decide-banking")    # a Runnable: str -> decision dict
    route.invoke("I still haven't received my new card")
    # {'choice': 'card_arrival', 'confidence': 0.92, 'action': 'auto-route', ...}

Chat needs nothing AnuLM-specific -- the API speaks OpenAI's protocol, so
`ChatOpenAI(base_url=..., model="anulm-chat")` is all `anulm_chat` does.
Decisions have no OpenAI equivalent, so `AnuLMDecide` wraps /v1/decide as a
Runnable: it composes with `|`, batches in one HTTP call, and pairs with
RunnableBranch to send each message down the branch its decision names.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx
from langchain_core.runnables import Runnable, RunnableConfig

DEFAULT_BASE = os.environ.get("ANULM_BASE_URL", "http://127.0.0.1:8000/v1")


def anulm_chat(model: str = "anulm-chat", base_url: str = DEFAULT_BASE, api_key: Optional[str] = None,
               **kwargs):
    """ChatOpenAI pointed at the AnuLM API. model: anulm-chat, anulm-answer or
    anulm-vision. Extra keyword arguments go to ChatOpenAI (max_tokens, temperature...)."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, base_url=base_url,
                      api_key=api_key or os.environ.get("ANULM_API_KEY", "none"), **kwargs)


class AnuLMDecide(Runnable[str, dict]):
    """A calibrated decision as a LangChain Runnable.

    Input: the message (str). Output: {"choice", "probabilities", "confidence",
    "action", "input"}, where action is "auto-route" when confidence >=
    threshold and "send to human" otherwise."""

    def __init__(self, model: str = "anulm-decide-banking", options: Optional[list[str]] = None,
                 threshold: float = 0.7, top: int = 3, base_url: str = DEFAULT_BASE,
                 api_key: Optional[str] = None, timeout: float = 120.0):
        self.model, self.options, self.threshold, self.top = model, options, threshold, top
        key = api_key or os.environ.get("ANULM_API_KEY", "none")
        self._http = httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=timeout,
                                  headers={"Authorization": f"Bearer {key}"})

    def _post(self, inputs: list[str]) -> list[dict]:
        body: dict[str, Any] = {"model": self.model, "input": inputs, "threshold": self.threshold,
                                "top": self.top}
        if self.options:
            body["options"] = self.options
        r = self._http.post("decide", json=body)
        if r.status_code != 200:
            raise ValueError(f"AnuLM /v1/decide {r.status_code}: {r.json().get('error', {}).get('message', r.text)}")
        return r.json()["results"]

    def invoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs) -> dict:
        return self._post([input])[0]

    def batch(self, inputs: list[str], config=None, *, return_exceptions: bool = False, **kwargs) -> list[dict]:
        return self._post(list(inputs)) if inputs else []
