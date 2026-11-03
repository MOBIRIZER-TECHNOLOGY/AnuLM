"""
A Hugging Face `tokenizer.json` behind the same three members as bpe.BPE
(`encode`, `decode`, `eos_id`), so code written for the project's own
tokenizer runs unchanged on a checkpoint converted from another model
(tools_qwen.py). Uses the `tokenizers` package only -- no transformers.
"""

from __future__ import annotations

import json
from pathlib import Path


def _gpt2_byte_decoder() -> dict[str, int]:
    """The inverse of GPT-2's `bytes_to_unicode`: the printable character a
    ByteLevel tokenizer writes for each byte, mapped back to that byte."""
    values = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    chars = values.copy()
    n = 0
    for b in range(256):
        if b not in values:
            values.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for c, b in zip(chars, values)}


class HFTokenizer:
    """A decoder's tokenizer.json (Qwen-style); `eos_id` is the id of
    `eos_token`, which must exist (ValueError otherwise)."""
    def __init__(self, path: str | Path, eos_token: str = "<|endoftext|>"):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(path))
        self.eos_id = self.tk.token_to_id(eos_token)
        if self.eos_id is None:
            raise ValueError(f"{eos_token} not in {path}")
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        # Only a ByteLevel tokenizer stores bytes we can reassemble exactly;
        # anything else falls back to the library's own decode.
        self.byte_level = (config.get("decoder") or {}).get("type") == "ByteLevel"
        self.added = {t["id"]: t["content"].encode("utf-8") for t in config.get("added_tokens", [])}
        self._bytes_of = _gpt2_byte_decoder() if self.byte_level else None

    def encode(self, text: str) -> list[int]:
        """Ids for `text`, without the tokenizer's added special tokens."""
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        """Text for `ids`; special tokens are kept in the output."""
        return self.tk.decode(ids, skip_special_tokens=False)

    def decode_complete(self, ids: list[int]) -> str:
        """`decode` for text still being generated: a trailing, incomplete UTF-8
        character is held back rather than shown as U+FFFD (bpe.complete_utf8)."""
        if not self.byte_level:
            return self.decode(ids)
        from anulm.bpe import complete_utf8
        parts = []
        for i in ids:
            i = int(i)
            parts.append(self.added[i] if i in self.added
                         else bytes(self._bytes_of[c] for c in self.tk.id_to_token(i)))
        return complete_utf8(b"".join(parts))


class EncoderTokenizer:
    """An encoder's tokenizer.json (ModernBERT, BERT-style) behind the same
    interface: `eos_id` is its [CLS] start token, `pad_id` its padding."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(path))
        self.eos_id = self.tk.token_to_id("[CLS]")
        self.pad_id = self.tk.token_to_id("[PAD]")
        if self.eos_id is None or self.pad_id is None:
            raise ValueError(f"no [CLS]/[PAD] in {path}")

    def encode(self, text: str) -> list[int]:
        """Ids for `text`, without the [CLS]/[SEP] the tokenizer would add."""
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        """Text for `ids`; special tokens are kept in the output."""
        return self.tk.decode(ids, skip_special_tokens=False)


def load_tokenizer(path: str | Path):
    """bpe.BPE for the project's own JSON, HFTokenizer for a HF tokenizer.json.

    Decided by the file's content, not its name: an exported folder, a renamed
    copy or a converted model's tokenizer can be called anything. The project's
    format carries its `type` field; a Hugging Face file has a `model` block.
    Anything else goes to BPE.load, whose error names the file and the type.
    """
    from anulm.bpe import BPE
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("type") in BPE.TYPES:
        return BPE.load(path)
    if isinstance(config.get("model"), dict):
        return HFTokenizer(path)
    return BPE.load(path)
