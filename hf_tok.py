"""
A Hugging Face `tokenizer.json` behind the same three members as bpe.BPE
(`encode`, `decode`, `eos_id`), so code written for the project's own
tokenizer runs unchanged on a checkpoint converted from another model
(tools_qwen.py). Uses the `tokenizers` package only -- no transformers.
"""

from __future__ import annotations

from pathlib import Path


class HFTokenizer:
    def __init__(self, path: str | Path, eos_token: str = "<|endoftext|>"):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(path))
        self.eos_id = self.tk.token_to_id(eos_token)
        assert self.eos_id is not None, f"{eos_token} not in {path}"

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=False)


class EncoderTokenizer:
    """An encoder's tokenizer.json (ModernBERT, BERT-style) behind the same
    interface: `eos_id` is its [CLS] start token, `pad_id` its padding."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(path))
        self.eos_id = self.tk.token_to_id("[CLS]")
        self.pad_id = self.tk.token_to_id("[PAD]")
        assert self.eos_id is not None and self.pad_id is not None, f"no [CLS]/[PAD] in {path}"

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=False)


def load_tokenizer(path: str | Path):
    """bpe.BPE for the project's own JSON, HFTokenizer for a HF tokenizer.json."""
    if Path(path).name == "tokenizer.json":
        return HFTokenizer(path)
    from bpe import BPE
    return BPE.load(path)
