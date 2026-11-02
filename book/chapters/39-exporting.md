---
title: "39 · Exporting to the ecosystem"
part: "IX · To production"
accent: teal
pinned: "@773b6a8"
---

# 39  Exporting to the ecosystem

> **Current companion update, 2 November 2026:** See the [current workflow](../docs/reader/current-workflow.md) and [language coverage](../docs/reader/language-coverage.md) for revised setup, training, evaluation and limitations. This chapter's cited implementation and numerical results remain historical; new workflow checks do not clear its open release gates.

A training checkpoint is a pickled Python dict: loading it runs code, it needs
this repository's classes, and it holds the weights in 32-bit floats.
`export_hf.py` turns it into a folder that anyone can download: weights in
safetensors, a JSON configuration, the tokenizer and a model card. Two thin
files let the Hugging Face `transformers` library load it, reusing
`model.py`'s modules instead of reimplementing them. This chapter reads the
exporter, both loaders and the wrapper, and measures what the export's 16-bit
storage costs the released model.

Prerequisites: checkpoints (Chapter 25), the KV cache (Chapter 27).

---

## 39.1  What a release folder holds

The released base model is published as `toonist/AnuLM-Base-v2-400M` on the
Hugging Face Hub. Its local copy:

```
5. release\AnuLM-Base-v2-400M (local copy; release/ is git-ignored)
           2,301  config.json
           3,124  configuration_anulm.py  identical to @773b6a8
          61,044  model.py  differs from @773b6a8
     796,522,272  model.safetensors
          12,275  modeling_anulm.py  identical to @773b6a8
           6,946  README.md
       1,533,692  tokenizer.json
         413,097  tokenizer.multi32k.json
             232  tokenizer_config.json
```

*measured*: `ch39_export_roundtrip.py --measure`. The two wrapper files equal
the pinned ones. `model.py` does not: the exporter copies the checkout's
`model.py` as it was on the day of the export (39.3), so a release carries a
snapshot. A safetensors file is an 8-byte length, a JSON header naming each
tensor's dtype, shape and offsets, then the raw numbers (39.2 reads one): no
pickle, so reading it runs no code.

![Figure 39.1 · From a training checkpoint to two loaders](../figures/fig39-1-export.svg)

*Figure 39.1 · From a training checkpoint to two loaders.* *verified*: from
`export_hf.py` at @773b6a8 and the folder listing above.

`IN THE REPO` · `MEASURED`

## 39.2  Weights: big tensors in 16 bits, small ones in 32

> **In the repo** · `export_hf.py` · `to_bf16` · lines 50–59 · @773b6a8

```python
def to_bf16(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Weights to bfloat16; small floating tensors (norm gains, router
    biases) stay in float32 because they are cheap and precision-sensitive."""
    out = {}
    for k, v in sd.items():
        v = v.detach().cpu().contiguous()
        if v.is_floating_point() and v.numel() >= 100_000:
            v = v.to(torch.bfloat16)
        out[k] = v
    return out
```

The rule is by size: a floating tensor of 100,000 numbers or more becomes
bfloat16, anything smaller stays float32. Big matrices dominate the bytes;
small tensors cost nothing to keep exact. On the released model:

```
6. model.safetensors: 796,522,272 bytes = 8 + 176,344 header + data
   F32:   138 tensors, 2,049,600 bytes = 512,400 numbers
   BF16:  1413 tensors, 794,296,320 bytes = 397,148,160 numbers
   float32 tensors by kind: {'norm gains': 81, 'router expert_bias': 19, 'router load_counts': 19, 'router weight': 19}; rule numel >= 100,000 <=> BF16 holds for every tensor
   numbers in all: 397,660,560; ckpt_base_v2.pt is 1,591,281,456 bytes, 2.00x the safetensors file
```

*measured*: the file's header only, read by `ch39_export_roundtrip.py`.

The float32 group is every norm gain and all three router tensors of the 19
MoE layers. The router *weight*, 24 × 1,024 = 24,576 numbers, falls under the
threshold too. The bfloat16 group is the 1,413 large matrices: 1,368 expert
matrices, 40 attention matrices, 3 for layer 0's SwiGLU and the two vocabulary
tables. The counts add up to the 397,660,560 numbers of Chapter 20's state
dict, buffers included, and the file is half the size of the `.pt`.

What 16 bits cost, on held-out text. The book example loads each variant with
`moe_impl="sparse"`, so every matrix multiply runs in float32 unless the
variant says otherwise (the checkpoint's own `grouped` path would run the
experts in bfloat16, Chapter 18). A forward hook on each router records which
experts it picks, to compare with the native model:

```python
    variants = [  # name, source, state-dict edit, run the whole model in bfloat16
        ("native .pt, fp32", ROOT / "ckpt_base_v2.pt", None, False),
        ("exported folder (bf16 matrices), fp32 compute", RELEASE, None, False),
        ("  + router biases rounded to bf16", RELEASE, lambda sd: bias(sd, bf), False),
        ("  + router biases set to zero", RELEASE, lambda sd: bias(sd, torch.zeros_like), False),
        ("  + biases centred per layer, then bf16", RELEASE, lambda sd: bias(sd, lambda v: bf(v - v.mean())), False),
        ("native .pt, whole model .to(bfloat16)", ROOT / "ckpt_base_v2.pt", None, True),
    ]
```

```
   router biases, 19 layers: mean 216.0 to 798.1; spread (max - min) 0.018 to 0.281; distinct values per layer after bf16 rounding: 1 to 2 (of 24)
   native .pt, fp32                               loss 3.37908 (+0.00000); expert sets changed: median layer 0.0%, max 0.0%
                                                  greedy, 8 prompts x 128 tokens: distinct-token share 0.218
   exported folder (bf16 matrices), fp32 compute  loss 3.37903 (-0.00004); expert sets changed: median layer 0.9%, max 1.5%
     + router biases rounded to bf16              loss 3.38219 (+0.00312); expert sets changed: median layer 50.9%, max 99.4%
     + router biases set to zero                  loss 3.38572 (+0.00665); expert sets changed: median layer 51.8%, max 99.4%
     + biases centred per layer, then bf16        loss 3.37905 (-0.00003); expert sets changed: median layer 1.1%, max 2.9%
   native .pt, whole model .to(bfloat16)          loss 3.38285 (+0.00377); expert sets changed: median layer 50.9%, max 99.4%
                                                  greedy, 8 prompts x 128 tokens: distinct-token share 0.227
   graded tokens per variant: 32,732
```

*measured*: 64 windows of 512 tokens of `data/v2/val.u16`, seed 0, the
windows of Chapter 20; one sample, one seed. Greedy prompts are the first 64
tokens of 8 of those windows; only statistics are printed.

Three readings:

- **Storing the matrices in bfloat16 is free here**: −0.00004 nats.
- **Rounding the router biases erases them.** Each layer's 24 biases sit
  together around a large common value, 216 to 798, and differ by at most
  0.281. At that size bfloat16 keeps too few bits to tell them apart: they
  round to one or two values per layer, which is the same as no bias at all.
  About half the expert choices change, and the loss rises 0.003 nats; zeroing
  the biases costs 0.007. Centring each layer's biases first, which does not
  change any choice, since top-k ignores a shift shared by every expert, makes
  the rounding free.
- **A whole-bfloat16 model does not collapse.** Casting every parameter and
  buffer to bfloat16 and computing in it cost 0.004 nats, and greedy
  continuations repeat no more than the float32 model's (distinct-token share
  0.227 against 0.218).

![Figure 39.2 · What the released file stores](../figures/fig39-2-dtypes.svg)

*Figure 39.2 · What the released file stores*, per group of tensors, on a
logarithmic axis. *measured*: `book/measurements/part_viii_ix_export.json`.

> **Details.** The exporter's comment (lines 271–274) and the wrapper's
> warning (`modeling_anulm.py` lines 107–120) say that 16-bit router biases
> make the output collapse into repeated tokens. Neither reproduces at
> @773b6a8 with this checkpoint: the measured costs are 0.003 and 0.004 nats.
> They probably predate the fix that runs the router in float32 (Chapter 17);
> that history is *not verified here*. Keeping the small tensors in float32
> remains the safe choice, since it is exact and costs 2 MB.

`IN THE REPO` · `MEASURED`

## 39.3  The configuration, and the code that travels with it

> **In the repo** · `export_hf.py` · `main` · lines 256–303 · @773b6a8

```python
    if cfg is not None:
        cfg_d = dataclasses.asdict(cfg)
        cfg_d["tokenizer_path"] = tok_file  # resolved relative to the folder by load_checkpoint
        meta.update({
            "architectures": ["AnuLMForCausalLM"],
            # trust_remote_code: transformers fetches these two modules from the
            # repo, and they pull model.py down with them.
            "auto_map": {
                "AutoConfig": "configuration_anulm.AnuLMConfig",
                "AutoModelForCausalLM": "modeling_anulm.AnuLMForCausalLM",
            },
            # ...
            "config": cfg_d,
            # ...
            "dtype": "float32",
            "torch_dtype": "float32",
            # ...
        })
    if args.kind != "lm":
        # Not a plain language model: transformers' AutoModelForCausalLM
        # would load the backbone and silently drop what makes it this model.
        for k in ("architectures", "auto_map"):
            meta.pop(k, None)
        meta["kind"] = args.kind
    # ...
    if args.kind == "lm":
        for mod in ("configuration_anulm.py", "modeling_anulm.py", "model.py"):
            shutil.copyfile(Path(__file__).parent / mod, out / mod)
```

Three choices:

- **The whole dataclass, nested.** `config` is `AnuLMConfig` field for field,
  so nothing is renamed. The tokenizer path is made relative to the folder.
- **Load in float32.** `dtype` tells `transformers` to build the model in
  float32, whatever the file stores; the comment gives the router as the
  reason (39.2 measures it).
- **`auto_map` only for plain language models.** `transformers` loads the
  classes it names, from the files copied beside the weights, when the caller
  passes `trust_remote_code=True`. A speech or image checkpoint has no
  `auto_map`, so `AutoModelForCausalLM` cannot load it as a text model and
  silently drop its projector. Its projector and other extras go to
  `extras.safetensors` instead (lines 231–236).

`IN THE REPO`

## 39.4  Loading it back without `transformers`

`load_checkpoint` accepts a `.pt`, a folder or a Hub id:

> **In the repo** · `model.py` · `load_checkpoint` · lines 1169–1189 · @773b6a8

```python
    p = _P(path)
    if not p.exists() and re.fullmatch(r"[\w.-]+/[\w.-]+", str(path).replace("\\", "/")):
        from huggingface_hub import snapshot_download
        p = _P(snapshot_download(str(path).replace("\\", "/")))
    if not p.is_dir():
        return torch.load(str(p), map_location=map_location, weights_only=False)
    from safetensors.torch import load_file
    dev = map_location if isinstance(map_location, str) else str(map_location)
    meta = json.loads((p / "config.json").read_text(encoding="utf-8"))
    ck = {}
    if "config" in meta:
        cfg_d = dict(meta["config"])
        if cfg_d.get("tokenizer_path"):
            cfg_d["tokenizer_path"] = str(p / cfg_d["tokenizer_path"])
        known = {f.name for f in dataclasses.fields(AnuLMConfig)}
        ck["cfg"] = AnuLMConfig(**{k: v for k, v in cfg_d.items() if k in known})
    if (p / "model.safetensors").exists():
        ck["model"] = load_file(str(p / "model.safetensors"), device=dev)
    for k in ("step", "val_loss", "qa_templates", "base_ckpt"):
        if meta.get(k) is not None:
            ck[k] = meta[k]
```

A string shaped `owner/name` that is not a local path is downloaded first. A
file goes to `torch.load`. A folder is rebuilt into the same dict a `.pt`
holds: `cfg`, with unknown fields dropped so an older `model.py` can read a
newer export, and `model`, the tensors in their stored dtypes.
`load_state_dict` copies them into the float32 model, so bfloat16 becomes
float32 on the way in. The test
`an_exported_folder_loads_everywhere_a_pt_does` (`test_model.py` line 914)
holds the round trip, and checks that eleven named scripts (lines 960–962) call
`load_checkpoint` rather than `torch.load`; scripts outside that list are not
checked.

`IN THE REPO`

## 39.5  The `transformers` wrapper reuses the real modules

> **In the repo** · `modeling_anulm.py` · `AnuLMForCausalLM.__init__` · lines 134–146 · @773b6a8

```python
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
```

`to_native()` (`configuration_anulm.py` lines 67–70) hands back the
dataclass. The wrapper builds a native `AnuLM` and adopts its modules under the
same attribute names, so the parameter names match the file and no key is
remapped. Only the outer loop is new:

> **In the repo** · `modeling_anulm.py` · `AnuLMForCausalLM.forward` · lines 195–230 · @773b6a8

```python
        if inputs_embeds is not None:
            raise NotImplementedError("AnuLM takes input_ids, not inputs_embeds")
        start = past_key_values.seen if past_key_values is not None else 0
        cache = past_key_values.layers if past_key_values is not None else None
        # ...
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary(S, x.device, x.dtype, start=start)
        for i, layer in enumerate(self.layers):
            # ...
                x, aux = layer(x, cos, sin, cache[i] if cache is not None else None, start)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        # ...
        if use_cache:
            if past_key_values is None:
                past_key_values = AnuLMCache([dict() for _ in self.layers])
            past_key_values.seen = start + S
            present = past_key_values
```

Three differences from `AnuLM.forward`, each to fit `transformers`:

- **Logits for every position**, (B, S, V). The native forward returns only
  the last position without targets (Chapter 20).
- **Labels are shifted here**, and −100 is ignored. Native callers pass targets
  already shifted, with −1 (Chapter 20).
- **The cache is the native one.** `AnuLMCache` wraps the list of per-layer
  dicts and counts `seen` tokens, the absolute `start` of Chapter 27. MLA
  caches a latent, not keys and values, so `transformers`' own cache would not
  fit (lines 232–247).

The book example exports a small model with `export_hf.main()` and loads it
both ways, then compares:

```python
    hf = AnuLMForCausalLM.from_pretrained(str(folder), dtype=torch.float32).eval()
    ck = M.load_checkpoint(str(folder))
    nat = M.AnuLM(ck["cfg"]).eval()
    nat.load_state_dict(ck["model"])                      # the same bf16-rounded weights as hf
    x = torch.randint(0, native.cfg.vocab_size, (1, 24))
    with torch.no_grad():
        full = hf(input_ids=x).logits
        last, _ = nat(x)
        hf_loss = hf(input_ids=x, labels=x).loss
        _, nat_loss = nat(x[:, :-1], x[:, 1:])
```

```
3. AnuLMForCausalLM.from_pretrained(folder): logits (1, 24, 2048) (every position), native (1, 1, 2048)
   last-position difference 1.5e-07; loss with labels=x 7.589681, native with shifted targets 7.589681
   greedy 12 tokens equal: True
```

*verified*: `ch39_export_roundtrip.py`. The test
`transformers_wrapper_matches_the_native_model` (`test_model.py` line 1271)
holds the same, cached and uncached.

> **Details.** The rotary tables are buffers registered as non-persistent, so
> they are not in the file. `transformers` builds the model on the `meta`
> device and fills in what the file lacks with uninitialised memory; the
> wrapper once loaded every weight correctly and computed with noise for
> positions. `rebuild_rotary` (lines 84–102) recomputes the tables after every
> `from_pretrained`, and the test checks them.

`IN THE REPO`

## 39.6  What the wrapper does not do

> **In the repo** · `modeling_anulm.py` · `AnuLMForCausalLM._reorder_cache` · lines 257–264 · @773b6a8

```python
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # Beam search would have to reorder every tensor each attention module
        # put in its dict, and nothing here has been tested against it.
        raise NotImplementedError(
            "beam search is not supported; use greedy or sampling "
            "(num_beams=1), which is how every number in docs/RESULTS.md "
            "was measured")
```

- **No beam search**: the cache's dicts are not reordered.
- **No padding mask**: `attention_mask` is accepted and ignored (docstring,
  lines 187–193), so a left-padded batch is wrong. Send one sequence at a time.
- **No `inputs_embeds`**: the speech and image models load through this
  repository's own code (Chapter 36), not `AutoModelForCausalLM`.
- **16-bit loading warns**, in `from_pretrained`:

> **In the repo** · `modeling_anulm.py` · `AnuLMPreTrainedModel.from_pretrained` · lines 104–121 · @773b6a8

```python
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
```

  The load rebuilds the rotary tables, then checks the first parameter's dtype.
  The warning's mechanism is right, since rounding does erase the bias
  differences, but its outcome is not: 39.2 measured 0.004 nats, not repeated
  tokens.

```
4. refusals: inputs_embeds: AnuLM takes input_ids, not inputs_embeds
             bfloat16 load: RuntimeWarning 'AnuLM loaded with torch.bfloat16 weights. This model does ...'
```

*verified*: `ch39_export_roundtrip.py`.

> **Build it.** The lab book's Chapter 15 (`course/ch15-shipping`) exports
> your own model and decides whether it is fit to release.

`IN THE REPO`

## 39.7  Exercises

1. A model has D = 512 and V = 1,000. Which of its vocabulary tables does
   `to_bf16` store in bfloat16?
2. Why does the exporter copy `model.py` into the folder, and what is the
   risk?
3. A caller passes `labels=targets` where `targets` is already shifted. What
   does the wrapper compute?

**Answers.** 1. Both: 1,000 × 512 = 512,000 ≥ 100,000. 2. `trust_remote_code`
needs the module beside the weights; the copy freezes the code at export time,
so a later fix to `model.py` does not reach the release. 3. It shifts again
and grades each position against the token two ahead.

*verified*: answer 1 is arithmetic on `to_bf16`'s threshold.

---

## Key points

- `export_hf.py` writes safetensors weights, a JSON config with the whole
  native dataclass, the tokenizer, a card and the code `transformers` needs.
- Tensors of 100,000 numbers or more are stored in bfloat16; norm gains and
  all router tensors stay float32, and the folder is half the `.pt`'s size.
- On held-out text bfloat16 storage moved the loss by −0.00004 nats. Rounding
  the router biases erases them and changes about half the expert choices, yet
  cost only 0.003 nats; a whole-bfloat16 model cost 0.004 and did not collapse.
- `load_checkpoint` turns a folder or Hub id into the same dict a `.pt` gives;
  a test holds eleven named scripts to using it.
- The `transformers` wrapper adopts `model.py`'s modules by name, returns every
  position's logits, shifts labels itself and carries the native cache.
- It has no beam search, no padding mask and no `inputs_embeds`, and it warns
  on 16-bit loading.

> **In one breath.** The exporter stores big matrices in bfloat16 and every
> norm and router tensor in float32, which costs the released model nothing
> measurable (the biases would otherwise round away); one loader reads `.pt`, folder or Hub id alike, and a thin
> `transformers` wrapper reuses the native modules, so its tokens match the
> repository's own.

## References

This chapter cites no external research. Its evidence is the repository at
@773b6a8, the local release folder and the measurements:
- `export_hf.py`: `to_bf16` 50–59, `tensor_groups` 105–109, `main` 169–321
  (weights 216–236, config 254–298, code copy 300–303).
- `model.py`: `load_checkpoint` 1153–1202.
- `configuration_anulm.py`: `AnuLMConfig` 34–74.
- `modeling_anulm.py`: `AnuLMCache` 45–66, `AnuLMPreTrainedModel` 69–130,
  `AnuLMForCausalLM` 133–264.
- `test_model.py`: `an_exported_folder_loads_everywhere_a_pt_does` 914,
  `transformers_wrapper_matches_the_native_model` 1271.
- `release/AnuLM-Base-v2-400M/` (git-ignored local copy of
  huggingface.co/toonist/AnuLM-Base-v2-400M).
- Book example and measurements: `book/code-examples/ch39_export_roundtrip.py`,
  `book/measurements/part_viii_ix_export.json`.
