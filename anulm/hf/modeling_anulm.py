"""`transformers` wrapper for AnuLM: `from_pretrained`, `generate`, pipelines.

    from transformers import AutoModelForCausalLM, AutoTokenizer
    m = AutoModelForCausalLM.from_pretrained("toonist/AnuLM-Coder-400M",
                                             trust_remote_code=True)
    t = AutoTokenizer.from_pretrained("toonist/AnuLM-Coder-400M")
    print(t.decode(m.generate(**t("def is_prime(n):", return_tensors="pt"),
                              max_new_tokens=60)[0]))

The architecture itself is not reimplemented here. This class builds the real
modules from `model.py` and adopts them under their own names, so the
parameter names are identical to the ones in `model.safetensors` and no key
remapping happens on load -- the failure mode that silently loads half a model
cannot occur, because a wrong name would be a missing key instead.

What this file adds is only the part transformers needs and `model.py`
deliberately does not have: an HF config, logits for every position (the
native forward returns just the last one when it is not training, which is
the right call for sampling and the wrong shape for `labels=`), a
`CausalLMOutputWithPast`, and the KV cache plumbed through
`prepare_inputs_for_generation`.

`test_model.py` holds this path to the native one: same logits, same greedy
continuation, cached and uncached.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

try:                                    # on the Hub, beside this file
    from .configuration_anulm import AnuLMConfig
    from .model import AnuLM as NativeAnuLM, Rotary as NativeRotary
except ImportError:                     # in a clone of the repository
    from configuration_anulm import AnuLMConfig
    from model import AnuLM as NativeAnuLM, Rotary as NativeRotary


class AnuLMCache:
    """The native cache -- one dict per layer -- with the two methods
    transformers asks a cache for. `model.py` owns what goes in the dicts;
    each attention module reads and writes its own entries."""

    # transformers probes this before deciding whether generate can be
    # compiled. This cache is a list of plain dicts whose contents each
    # attention module owns, so: no.
    is_compileable = False

    def __init__(self, layers: list[dict], seen: int = 0):
        self.layers = layers
        self.seen = seen                     # tokens already in the cache

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def __len__(self) -> int:
        return len(self.layers)


class AnuLMPreTrainedModel(PreTrainedModel):
    config_class = AnuLMConfig
    base_model_prefix = ""                   # the checkpoint has no prefix
    supports_gradient_checkpointing = True
    _no_split_modules = ["Block"]
    _supports_sdpa = True
    # The rotary tables are computed, not trained, so model.py registers them
    # non-persistently and they are not in the checkpoint. transformers builds
    # the model on the meta device and materialises anything the checkpoint
    # does not cover -- which for these buffers means uninitialised memory
    # unless _init_weights rebuilds them. It loaded every weight correctly and
    # produced quietly wrong logits until this was added; the test in
    # test_model.py exists to keep it that way.
    _keys_to_ignore_on_load_missing = [r"rotary\..*"]

    def rebuild_rotary(self, device=None) -> None:
        """Recompute the rotary tables from the config.

        Called after every load rather than left to transformers' missing-key
        handling, which materialises the buffer but does not know how to fill
        it. Cheap -- one arange per model -- and the alternative is a model
        whose weights are all correct and whose positions are noise.
        """
        for module in self.modules():
            if not isinstance(module, NativeRotary):
                continue
            dev = device or module.inv_freq.device
            if getattr(dev, "type", None) == "meta":
                dev = torch.device("cpu")
            module.inv_freq = NativeRotary._build_inv_freq(module.cfg, module.dim).to(dev)
            module.cos_cached = torch.empty(0, device=dev)
            module.sin_cached = torch.empty(0, device=dev)
            module._cached_len = 0
            module._cached_key = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.rebuild_rotary(device=next(model.parameters()).device)
        dtype = next(model.parameters()).dtype
        if dtype in (torch.bfloat16, torch.float16):
            warnings.warn(
                f"AnuLM loaded with {dtype} weights. This model does not "
                "survive that: the aux-loss-free router keeps a per-expert "
                "bias whose differences decide which experts fire, and "
                "rounding it to 16 bits changes the routing -- the output "
                "degrades to repeated tokens rather than getting slightly "
                "worse. The export keeps those tensors in float32 for this "
                "reason. Load with dtype=torch.float32 (the default) and use "
                "torch.autocast for speed, which is how every number in the "
                "model card was measured.",
                RuntimeWarning, stacklevel=2)
        return model

    def _init_weights(self, module):
        if isinstance(module, NativeRotary):
            pass                      # rebuild_rotary owns these; see above
        elif isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            std = self.config.config["initializer_range"]
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)


class AnuLMForCausalLM(AnuLMPreTrainedModel, GenerationMixin):
    def __init__(self, config: AnuLMConfig):
        super().__init__(config)
        native = config.to_native()
        core = NativeAnuLM(native)
        # Adopt the real modules under their own names. `embed_tokens`,
        # `layers`, `norm` and `lm_head` are exactly the top-level names in
        # model.safetensors; `rotary` holds only non-persistent buffers.
        self.cfg = native
        self.embed_tokens = core.embed_tokens
        self.rotary = core.rotary
        self.layers = core.layers
        self.norm = core.norm
        self.lm_head = core.lm_head
        self.grad_ckpt = False
        self.aux_loss = None
        # Weight tying and the bookkeeping transformers does after a model is
        # built. The native modules are already initialised by model.py; this
        # re-runs that and then from_pretrained overwrites it anyway.
        self.post_init()

    # --- transformers plumbing -------------------------------------------
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def _set_gradient_checkpointing(self, enable: bool = True, **kwargs):
        self.grad_ckpt = enable

    # --- the model --------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[AnuLMCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """The native forward loop, with every position's logits kept.

        `attention_mask` is accepted and ignored: this model is trained and
        evaluated on dense blocks with no padding, its mask is built from the
        causal structure and the sliding window inside each attention module,
        and silently honouring a padding mask here would make these numbers
        disagree with every one in docs/RESULTS.md. Left-padding a batch is
        therefore not supported -- feed one sequence at a time, which is what
        serve.py, sample.py and eval_*.py all do.
        """
        if inputs_embeds is not None:
            raise NotImplementedError("AnuLM takes input_ids, not inputs_embeds")
        start = past_key_values.seen if past_key_values is not None else 0
        cache = past_key_values.layers if past_key_values is not None else None

        B, S = input_ids.shape
        if start + S > self.cfg.block_size:
            raise ValueError(
                f"positions up to {start + S} exceed block_size "
                f"{self.cfg.block_size}; this model has no long-context mode "
                f"(see README.md 'Long-context extension')")

        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(S, x.device, x.dtype, start=start)
        for i, layer in enumerate(self.layers):
            if self.grad_ckpt and self.training:
                x, aux = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, use_reentrant=False)
            else:
                x, aux = layer(x, cos, sin, cache[i] if cache is not None else None, start)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100)

        present = None
        if use_cache:
            if past_key_values is None:
                past_key_values = AnuLMCache([dict() for _ in self.layers])
            past_key_values.seen = start + S
            present = past_key_values
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=present)

    def _prepare_cache_for_generation(self, generation_config, model_kwargs,
                                      generation_mode, batch_size, max_cache_length):
        """Install our own cache instead of a DynamicCache.

        transformers would otherwise hand the model a DynamicCache, which holds
        (key, value) pairs per layer. This model's layers do not all cache that
        shape -- MLA caches the `kv_lora_rank` latent plus the single shared
        rope key, which is the whole point of MLA -- so the native per-layer
        dict is what goes in, wrapped in AnuLMCache.
        """
        if not generation_config.use_cache:
            model_kwargs.pop("past_key_values", None)
            return False
        if model_kwargs.get("past_key_values") is None:
            model_kwargs["past_key_values"] = AnuLMCache([dict() for _ in self.layers])
        return True

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, use_cache=True, **kwargs):
        # With a cache, only the tokens it has not seen are new.
        if past_key_values is not None:
            input_ids = input_ids[:, past_key_values.seen:]
        return {"input_ids": input_ids, "past_key_values": past_key_values,
                "use_cache": use_cache}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # Beam search would have to reorder every tensor each attention module
        # put in its dict, and nothing here has been tested against it.
        raise NotImplementedError(
            "beam search is not supported; use greedy or sampling "
            "(num_beams=1), which is how every number in docs/RESULTS.md "
            "was measured")


__all__ = ["AnuLMConfig", "AnuLMForCausalLM", "AnuLMPreTrainedModel", "AnuLMCache"]
