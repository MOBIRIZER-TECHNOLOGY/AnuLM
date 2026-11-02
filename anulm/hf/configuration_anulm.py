"""`transformers` configuration for AnuLM, wrapping the project's own dataclass.

This exists so a released checkpoint loads with

    AutoModelForCausalLM.from_pretrained("toonist/AnuLM-Coder-400M",
                                         trust_remote_code=True)

instead of requiring a clone of this repository. It is a thin adapter, not a
second source of truth: every architectural field lives in `AnuLMConfig` in
`model.py`, and this class carries that dataclass's fields under `config`,
exactly as `export_hf.py` already writes them into config.json. `to_native()`
hands the dataclass back, so the weights are read by the same code that wrote
them.

The one rule to keep: a field added to the dataclass needs no change here,
because the nested dict is passed through whole.
"""

from __future__ import annotations

from dataclasses import asdict, fields

from transformers import PretrainedConfig

# Relative first so transformers' dynamic-module loader pulls model.py down
# beside this file when someone loads from the Hub with trust_remote_code;
# absolute second so the same file works in a clone of the repository.
try:
    from .model import AnuLMConfig as NativeConfig
except ImportError:
    from model import AnuLMConfig as NativeConfig


class AnuLMConfig(PretrainedConfig):
    model_type = "anulm"
    # Names transformers looks for when it wants the shape of a model without
    # understanding it (device maps, pipeline parallelism, generation).
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
        "max_position_embeddings": "block_size",
    }

    def __init__(self, config: dict | None = None, **kwargs):
        # `config` is the nested dict export_hf.py writes: the native dataclass,
        # field for field. Unknown keys are kept rather than dropped, so a
        # checkpoint from a newer model.py still round-trips through here.
        native = dict(config or {})
        known = {f.name for f in fields(NativeConfig)}
        defaults = asdict(NativeConfig())
        self.config = {k: native.get(k, defaults[k]) for k in known}
        self.config.update({k: v for k, v in native.items() if k not in known})

        # Flat mirrors of the fields transformers and the Hub read directly.
        c = self.config
        self.vocab_size = c["vocab_size"]
        self.hidden_size = c["hidden_size"]
        self.n_layer = c["n_layer"]
        self.n_head = c["n_head"]
        self.block_size = c["block_size"]
        self.rope_theta = c["rope_theta"]
        self.tie_word_embeddings = c["tie_word_embeddings"]

        kwargs.setdefault("tie_word_embeddings", self.tie_word_embeddings)
        super().__init__(**kwargs)

    def to_native(self) -> NativeConfig:
        """The dataclass `model.py` builds from, with fields it does not know dropped."""
        known = {f.name for f in fields(NativeConfig)}
        return NativeConfig(**{k: v for k, v in self.config.items() if k in known})

    @classmethod
    def from_native(cls, cfg: NativeConfig, **kwargs) -> "AnuLMConfig":
        return cls(config=asdict(cfg), **kwargs)
