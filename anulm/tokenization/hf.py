"""
A Hugging Face `tokenizer.json` behind the same three members as bpe.BPE
(`encode`, `decode`, `eos_id`), so code written for the project's own
tokenizer runs unchanged on a checkpoint converted from another model
(tools_qwen.py). Uses the `tokenizers` package only -- no transformers.
"""

from __future__ import annotations

from pathlib import Path
import json


class HFTokenizer:
    def __init__(self, path: str | Path, eos_token: str = "<|endoftext|>"):
        from tokenizers import Tokenizer
        self.tk = Tokenizer.from_file(str(path))
        self.eos_id = self.tk.token_to_id(eos_token)
        assert self.eos_id is not None, f"{eos_token} not in {path}"
        config = json.loads(Path(path).read_text(encoding='utf-8'))
        self.byte_level = (config.get('decoder') or {}).get('type') == 'ByteLevel'
        self.added = {item['id']: item['content'].encode('utf-8') for item in config.get('added_tokens', [])}
        values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        characters = values.copy()
        for value in range(256):
            if value not in values:
                values.append(value)
                characters.append(256 + sum(c >= 256 for c in characters))
        self.byte_decoder = dict(zip(map(chr, characters), values))

    def encode(self, text: str) -> list[int]:
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=False)

    def decode_complete(self, ids):
        if not self.byte_level:
            raise ValueError('Incremental UTF-8 decoding requires a ByteLevel tokenizer.')
        from anulm.tokenization.utf8 import complete_utf8
        chunks = []
        for value in ids:
            value = int(value)
            if value in self.added:
                chunks.append(self.added[value])
            else:
                chunks.append(bytes(self.byte_decoder[c] for c in self.tk.id_to_token(value)))
        return complete_utf8(b''.join(chunks))


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
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if 'model' in config and 'added_tokens' in config:
        return HFTokenizer(path)
    from anulm.tokenization.bpe import BPE
    return BPE.load(path)
