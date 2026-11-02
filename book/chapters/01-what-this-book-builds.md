---
title: "1 · What this book builds"
part: "I · Foundations"
accent: slate
pinned: "@773b6a8"
---

# 1  What this book builds

**First reading.** Begin with [Start here](../docs/reader/start-here.md) for installation
and a small CPU training run. Then choose a route in
[Learning paths](../docs/reader/learning-paths.md). You can skip the repository inventory
in 1.5 until you need to locate a particular file.

AnuLM is a 400M-parameter language model trained from scratch on Hindi,
English and Python on one consumer GPU. Its architecture follows the
configurations and modelling code of Sarvam AI's released 30B and 105B models
[R49], scaled down. It is a mixture of experts: of its 397.7M parameters, about
174M are used for any one token (Chapter 20). The released model is
`AnuLM-Base-v2-400M`, the file `ckpt_base_v2.pt`, published on Hugging Face as
`toonist/AnuLM-Base-v2-400M`.

A language model reads some text, gives a probability to every possible next
piece, picks one and appends it, over and over. This chapter writes that loop
by hand, finds it in AnuLM's code, runs it on the released model, and tours the
repository the rest of the book reads.

**New words.** Part I starts here; these terms recur through the book.

| Term | Meaning |
|---|---|
| token | one piece of text the model reads or writes: a byte, or a sub-word piece from a tokenizer (Chapters 8–9) |
| vocabulary, V | the set of all token ids; the released model has V = 32,768 |
| B, S, D | batch size (sequences processed together), sequence length in tokens, and the model's width: the length of the vector it keeps per token (D = 1,024 here) |
| forward pass | one call of the model on a batch of ids: ids in, scores out |
| logits | the forward pass's output, one raw score per vocabulary id; larger means more likely |
| softmax | the function that turns logits into probabilities that sum to 1 (Chapter 5) |
| loss | the average of −log p over the true next tokens (Chapter 3) |
| checkpoint | a file of trained weights and their configuration, such as `ckpt_base_v2.pt` |
| preset | a named configuration of sizes in `model.py`, such as `nano_30b` |
| the pin | commit `773b6a8`, which every line number in this book refers to; the code is identical at the public commit `ae59238` (1.7) |

---

## 1.1  The four-step loop, by hand

`book/code-examples/ch01_four_steps.py` (**Book example, not repository
code**) writes the loop in ten lines, picking the most likely id each time:

```python
def four_steps(model: AnuLM, ids: torch.Tensor, n: int, show: bool = False) -> torch.Tensor:
    """Greedy decoding with no cache: the loop at its plainest."""
    for step in range(n):
        logits, _ = model(ids)                        # 1. read: the whole sequence so far
        probs = F.softmax(logits[:, -1, :], dim=-1)   # 2. score: one probability per vocabulary id
        nxt = probs.argmax(dim=-1, keepdim=True)      # 3. pick: here, the most likely id
        if show and step == 0:
            print(f"read   ids    {tuple(ids.shape)}    logits {tuple(logits.shape)}")
            print(f"score  probs  {tuple(probs.shape)}  sum {probs.sum().item():.4f}")
            print(f"pick   nxt    {tuple(nxt.shape)}   id {nxt.item()}, p = {probs.max().item():.4f}")
        ids = torch.cat([ids, nxt], dim=1)            # 4. repeat: append, and go again
    return ids
```

- **Read.** `model(ids)` runs a forward pass on `ids`, a (B, S) tensor of token
  ids. Called without targets, it returns logits for the last position only,
  (B, 1, V) (1.2).
- **Score.** `logits[:, -1, :]` drops the length-1 axis, (B, V), and the
  softmax turns each row into probabilities.
- **Pick.** `argmax` takes the most likely id, (B, 1).
- **Repeat.** `torch.cat` appends it along the sequence axis, and the loop
  reads the longer sequence again.

The example runs it beside the repository's own loop, `generate`, with
`top_k=1`, which leaves only the best id possible:

```python
    torch.manual_seed(0)
    cfg = tiny()
    model = AnuLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        mine = four_steps(model, prompt.clone(), 12, show=True)
    ours = model.generate(prompt.clone(), 12, top_k=1)                    # KV cache on
    ours_nc = model.generate(prompt.clone(), 12, top_k=1, use_cache=False)
    assert torch.equal(mine, ours) and torch.equal(mine, ours_nc), "the hand loop and generate disagree"
```

`tiny()` is the test suite's small configuration (`test_model.py` lines
33–38): V = 64, three layers, D = 32. Untrained, it picks noise, but the same
noise:

```text
read   ids    (1, 4)    logits (1, 1, 64)
score  probs  (1, 64)  sum 1.0000
pick   nxt    (1, 1)   id 18, p = 0.0204
by hand        [38, 2, 0, 59, 18, 11, 25, 14, 29, 27, 25, 14, 17, 44, 46, 37]
generate       [38, 2, 0, 59, 18, 11, 25, 14, 29, 27, 25, 14, 17, 44, 46, 37]
by-hand loop == generate(top_k=1), with and without the cache: True
```

*verified*: printed by `ch01_four_steps.py`. The first pick's 0.0204 is close
to a uniform 1/64 = 0.0156 (Chapter 3 explains why). The test
`generation_is_deterministic_at_top_k_1` (`test_model.py` line 387) requires
two greedy runs to agree.

![Figure 1.1 · The four-step loop](../figures/fig01-1-loop.svg)

*Figure 1.1 · The four-step loop.* One pass, with each tensor's shape.
*verified*: shapes printed by `ch01_four_steps.py` on `tiny()`.

`IN THE REPO`

## 1.2  The same loop in `generate`

The repository's loop is a method on the model, `generate` (line 1059). With
its optional features elided:

> **In the repo** · `model.py` · `AnuLM.generate` · lines 1083–1109 · @773b6a8

```python
        for _ in range(max_new_tokens):
            if not use_cache:
                logits, _ = self(idx[:, -bs:])
            else:
                if cache is None or start + pending.shape[1] > bs:
                    cache, start, pending = self.new_cache(), 0, idx[:, -bs:]
                logits, _ = self(pending, cache=cache, start=start)
                start += pending.shape[1]
            logits = logits[:, -1, :].float()
            # ...
            logits = logits / max(temperature, 1e-5)
            # ...
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)              # (B, 1)
            # ...
            idx = torch.cat((idx, nxt), dim=1)
            pending = nxt
```

Against the hand loop, three things differ:

- **The KV cache.** Without it (`use_cache=False`) the model rereads the last
  `bs` tokens each step (`block_size`, its longest context). With it, the
  model keeps the keys and values its attention layers computed for earlier
  tokens (Chapter 11's terms), and each step reads only `pending`, the tokens
  not seen yet; `start` is how many it has seen (Chapter 27).
- **Temperature.** The logits are divided by it before the softmax: below 1
  sharpens the distribution, above 1 flattens it (Chapter 5).
- **Sampling.** `torch.multinomial` draws one id per row in proportion to its
  probability, instead of taking the argmax.

The elided lines mask audio and image ids in a model whose vocabulary was
grown for them, apply an optional repetition penalty, apply an optional
`top_k` cut that keeps only the k highest scores, and keep finished rows at
the end-of-sequence id (Chapter 28). Reading costs little at the output,
because without targets the forward pass projects only the last position:

> **In the repo** · `model.py` · `AnuLM.forward` · lines 860–863 · @773b6a8

```python
        if targets is None:
            # Inference shortcut: only the last position matters.
            logits = self._head(x[:, -1:, :])
            return logits, None
```

`x` is the final (B, S, D) stack of vectors; slicing with `-1:` keeps the
axis, so the logits are (B, 1, V).

`IN THE REPO`

## 1.3  The loop on the released model

`sample.py` runs `generate` on a checkpoint from the command line:

> **In the repo** · `sample.py` · `main` · lines 66–74 · @773b6a8

```python
    else:
        # Byte-level: the prompt is its own UTF-8 encoding. 257 is EOS.
        prompt_ids = list(args.prompt.encode("utf-8")) or [10]
        eos_id = 257
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
        out = model.generate(ids, args.tokens, temperature=args.temperature, top_k=args.top_k,
                             eos_id=eos_id, use_cache=not args.no_cache)
```

The released model uses a BPE tokenizer (a vocabulary of sub-word pieces
learned from text, Chapter 9), handled by the branch above this one; this
`else` serves byte-level models, whose prompt bytes are their ids. `ids` is a
(1, S) batch of one. On a GPU, autocast runs the matrix multiplies in bf16, a
16-bit float format (Chapter 23). The defaults are temperature 0.8 and
`top_k` 50 (lines 31–32). One run:

```text
$ python -m anulm.inference.sample --ckpt ckpt_base_v2.pt --prompt "The first railway line in India" --tokens 60 --seed 0
loaded step 1899999, val loss 3.3523  (moe_impl=grouped, cache=on)

[60 tokens in 5.2s, 12 tok/s]

The first railway line in India was opened in 1889. The second was extended in 1895. The first rail line was opened in 1883 and the third in 1888.
Highway construction and maintenance of railway lines were carried out in India in the late nineteenth century by the Indian Railway. In 1923 the first rail line
```

*measured*: `sample.py` at the pin, unchanged, on this machine's GPU, seed 0,
bf16 autocast with the experts in bf16 (`moe_impl=grouped`); one sample, not
a benchmark. The text is fluent and on topic, and its dates contradict each
other (first opened in 1889, then in 1883). That is what a base model is: a
next-token predictor with no notion of checking a fact. Its model card says
the same (`docs/MODEL_CARD_BASE_V2.md`).

`IN THE REPO` · `MEASURED`

## 1.4  A tour of `model.py`

The architecture is one file of 1,224 lines, in banner sections:

| Banner | Starts at | What it holds | In one line | Chapters |
|---|---|---|---|---|
| CONFIG | 36 | `AnuLMConfig` and the presets | every size and switch | 1, 20 |
| NORM / MLP | 210 | `RMSNorm`, `SwiGLU` | RMSNorm rescales each vector to unit root mean square; SwiGLU is a gated feed-forward network | 15, 16 |
| ROPE | 240 | YaRN helpers, `Rotary`, `apply_rope`, `_attn_mask`, `NATIVE_GQA`, `merge_tree_matrix`, `_sdpa` | RoPE rotates queries and keys by position; YaRN stretches it to longer contexts; then the attention mask and kernel call | 12, 14, 29 |
| ATTENTION -- GQA | 401 | `GQAttention` | grouped-query attention: groups of query heads share key/value heads | 11, 13 |
| ATTENTION -- MLA | 469 | `MLAttention` | latent attention: keys and values compressed into one small vector | 19 |
| MoE | 547 | `Router`, `MoE` | a mixture of experts: a router sends each token to a few of many feed-forward "experts" | 17, 18 |
| BLOCK / MODEL | 753 | `Block`, `AnuLM`, `load_checkpoint`, `fit_memory` | the layers, the model, loading | 15, 20, 27 |

*verified*: banner lines and contents read at the pin.

The model class builds everything in its first twelve lines:

> **In the repo** · `model.py` · `AnuLM.__init__` · lines 782–793 · @773b6a8

```python
class AnuLM(nn.Module):
    def __init__(self, cfg: AnuLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim
        self.rotary = Rotary(cfg, rope_dim)
        self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
```

In order: an embedding table, one D-wide vector per token id (Chapter 10); a
shared table of rotary position angles (Chapter 14); `n_layer` blocks of
attention and feed-forward (Chapters 11–18); a final normalisation; and the
output head, a D → V linear map that produces the logits. The released model
does not tie the input and output tables (the last two lines).

`IN THE REPO`

## 1.5  The other 70 root files, and the folders

**Historical layout.** This inventory describes the source at the cited pin.
The organized repository groups those files under `anulm/`; use its
`docs/STRUCTURE.md` and `docs/source-paths.json` for current locations.

This is a reference inventory. You do not need to memorise these files before
running a model; continue at 1.6 on your first reading.

Grouped by role, as `ch01_four_steps.py` counts them at the pin:

| Role | Files | Lines | Examples | Part |
|---|---|---|---|---|
| the model | 1 | 1,224 | `model.py` | I–IV |
| tests | 1 | 1,786 | `test_model.py` | every chapter |
| training and tokenizer | 4 | 1,356 | `train.py`, `bpe.py`, `muon.py`, `bigram_teacher.py` | III, V |
| data | 14 | 1,655 | `fetch_web.py`, `make_chat.py`, `mix_corpus.py` | V, VII |
| evaluation | 10 | 1,602 | `eval_golden.py`, `eval_code.py` | V |
| post-training | 6 | 2,251 | `finetune.py`, `lora.py`, `decide.py`, `rag.py`, `bidirectional.py` | VII |
| speech and vision | 11 | 3,970 | `vision_encoder.py`, `speech_encoder.py`, `omni_train.py` | VIII |
| serving and apps | 12 | 2,840 | `sample.py`, `serve.py`, `api.py`, `quickstart.py` | VI, IX |
| export | 6 | 1,088 | `export_hf.py`, `modeling_anulm.py` | IX |
| tools | 6 | 1,134 | `tools_sweep.py`, `update_tasks.py` | |
| `course/` | 66 | 11,032 | the lab book's code: stubs, solutions, checks | Build it boxes |
| `experiments/` | 17 | 3,373 | run scripts, among them the released runs' | V, VII |
| `sarvam/` | 9 | 3,409 | annotated copies of Sarvam's modelling code (Apache 2.0) | IV |
| `examples/` | 3 | 683 | API usage examples | IX |

*verified*: 166 Python files, counted by `ch01_four_steps.py`. The roles are
the book's grouping, by file name; the first test that matches wins:

```python
ROLES = [  # (role, test on the file name); the first match wins
    ("the model", lambda f: f == "model.py"),
    ("tests", lambda f: f == "test_model.py"),
    ("training and tokenizer", lambda f: f in ("train.py", "muon.py", "bigram_teacher.py", "bpe.py")),
    ("post-training", lambda f: f in ("finetune.py", "lora.py", "decide.py", "rag.py", "rc_pointer.py",
                                      "bidirectional.py")),
    ("speech and vision", lambda f: f.startswith(("omni", "speech_", "vision_", "audio_", "voice"))),
    ("serving and apps", lambda f: f in ("serve.py", "api.py", "ask.py", "sample.py", "sample_many.py",
                                         "quickstart.py", "langchain_anulm.py") or f.startswith("app")),
    ("export", lambda f: f in ("export_hf.py", "modeling_anulm.py", "configuration_anulm.py", "hf_tok.py")
     or f.startswith("tools_hf")),
    ("evaluation", lambda f: f.startswith(("eval_", "make_bench", "make_golden")) or f == "tools_eval_noise.py"),
    ("data", lambda f: f.startswith(("fetch_", "convert_", "make_", "mix_", "filter_"))),
    ("tools", lambda f: True),
]
```

The root is counted by this function, where `at_pin` (1.7) reads one file at
the pin:

```python
def repo_map() -> dict:
    names = subprocess.run(["git", "ls-tree", "--name-only", PIN], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    files = [f for f in names if f.endswith(".py")]
    groups: dict[str, list] = {}
    for f in files:
        role = next(r for r, test in ROLES if test(f))
        groups.setdefault(role, []).append((f, len(at_pin(f))))
    print(f"\nPython files at the repository root @{PIN}: {len(files)}")
    for role, _ in ROLES:
        fs = groups.get(role, [])
        print(f"  {role:24s} {len(fs):3d} files {sum(n for _, n in fs):6,d} lines")
    return groups
```

and the folders by this one, which lists the whole tree (`-r`):

```python
def subdirectories() -> None:
    """The Python outside the root: the lab book, run scripts, the reference code, examples."""
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", PIN], cwd=ROOT, capture_output=True,
                           text=True, check=True).stdout.split()
    py = [f for f in names if f.endswith(".py")]
    print(f"\nPython files in the whole tree @{PIN}: {len(py)}")
    for d in ("course", "experiments", "sarvam", "examples"):
        fs = [f for f in py if f.startswith(d + "/")]
        print(f"  {d + '/':24s} {len(fs):3d} files {sum(len(at_pin(f)) for f in fs):6,d} lines")
```

![Figure 1.2 · The repository by role](../figures/fig01-2-repo-map.svg)

*Figure 1.2 · The repository by role.* Lines of Python per role in the root;
the model is 6% of them. *verified*: drawn from the same count.

From a fresh clone, `quickstart.py --train` trains and samples a model:

> **In the repo** · `quickstart.py` · `train` · lines 182–189 · @773b6a8

```python
    rc = run([sys.executable, "-u", "train.py", "--preset", "30b", "--steps", "200",
              "--eval-every", "50", "--out", "ckpt_quickstart.pt"])
    if rc:
        print("\nTraining failed. The full log is above; docs/TUTORIAL.md has a")
        print("troubleshooting section for the usual causes.")
        return rc
    rc = run([sys.executable, "sample.py", "--ckpt", "ckpt_quickstart.pt",
              "--prompt", "KING RICHARD II:", "--tokens", "120"])
```

It runs `train.py` for 200 steps, then `sample.py` on the result. (`--demo`,
line 220, instead downloads a released checkpoint, the coder model, and
serves it.) The `30b`
preset is a method on `AnuLMConfig`:

> **In the repo** · `model.py` · `AnuLMConfig.nano_30b` · lines 147–159 · @773b6a8

```python
    @staticmethod
    def nano_30b(**kw) -> "AnuLMConfig":
        """GQA + high rope_theta, no YaRN. Sarvam 30B's shape."""
        cfg = dict(
            attn="gqa", n_layer=8, hidden_size=384, n_head=6, head_dim=64,
            n_kv_head=2, num_experts=16, num_experts_per_tok=2,
            moe_intermediate_size=96,      # hidden/4, as in the real model
            intermediate_size=768,         # hidden*2
            n_group=1, topk_group=1,       # grouping is a no-op, like the 30B
            rope_theta=1e6, yarn=False,
        )
        cfg.update(kw)
        return AnuLMConfig(**cfg)
```

A preset is a dictionary of sizes; `**kw` overrides any of them, which is how
`train.py` sets the vocabulary and context (Chapter 22). This one has 8 layers
of width 384 and 16 experts, 2 active per token: 17.4M parameters, 6.6M active
(Chapter 20).

`IN THE REPO`

## 1.6  The book's ten parts

| Part | Chapters | What it opens up |
|---|---|---|
| I · Foundations | 1–3 | the loop, the history, the one objective |
| II · Just enough mathematics | 4–7 | shapes, softmax, gradients, optimisers, on AnuLM's own lines |
| III · From text to vectors | 8–10 | bytes, BPE, embeddings |
| IV · Inside the transformer | 11–20 | attention, masks, heads, positions, norms, experts, the whole model |
| V · Training | 21–26 | data, the loop, precision, measurement, checkpoints, debugging |
| VI · Generation | 27–29 | the KV cache, choosing tokens, long context |
| VII · After pre-training | 30–34 | instruction tuning, LoRA, classifiers, retrieval, preference learning |
| VIII · Beyond text | 35–37 | speech and image inputs, one model for many tasks |
| IX · To production | 38–40 | serving, export, what a large stack adds |
| X · Capstone | 41 | train it, trace it, change it |

`IN THE REPO`

## 1.7  How to read a citation, and the evidence labels

A citation, such as 1.2's, names a file, a symbol, the lines and the commit.
The pin, `773b6a8`, lives only on the author's machine; the public repository's
`ae59238` (the tip of `main` on GitHub) differs from it only in `course/` and
an editor setting, so
every line number in this book is valid there too (the front matter has the
details). From a clone:

```bash
git show ae59238:model.py | sed -n '1083,1109p'
```

Excerpts are copied from the pin, with `# ...` for elided lines. A script
checks that every quoted line occurs inside the cited range; its core is this,
from the book example:

```python
def at_pin(path: str) -> list[str]:
    out = subprocess.run(["git", "show", f"{PIN}:{path}"], cwd=ROOT, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", check=True).stdout
    return out.splitlines()


def citation_holds(path: str, first: int, last: int, quoted: list[str]) -> bool:
    """What the book's checker does for every excerpt: each quoted line, stripped,
    must occur inside the cited range of the file at the pin."""
    window = [s.strip() for s in at_pin(path)[first - 1:last]]
    return all(q.strip() in window for q in quoted if q.strip() and q.strip() != "# ...")
```

Three lines of 1.2's excerpt pass at 1100–1105 and fail at 1000–1005:

```text
citation model.py 1100-1105 holds: True
citation model.py 1000-1005 holds: False
```

*verified*: printed by `ch01_four_steps.py`. The check tests membership, not
order; a stricter check, that the lines appear in order and are contiguous
except at `# ...`, also passes for every excerpt in Chapters 1–7.

Every number and section carries a label:

| Label | Meaning |
|---|---|
| *verified* | checked against the code at the pin, or recomputed by a book script |
| *measured* | from a recorded run in `docs/RESULTS.md` or `logs/`, or a book script on the released checkpoint, saved in `book/measurements/` |
| *cited* | from a paper or document whose register entry is CHECKED |
| *illustrative* | made up for teaching, and said so |
| `IN THE REPO` / `NOT IN THE REPO` | the mechanism is, or is not, implemented here |
| `MEASURED` | the section rests on a measurement |
| `EXTENSION`, `ANALOGY`, `RULE OF THUMB`, `IMPLEMENTATION-DEPENDENT` | a proposed change; a teaching shortcut; a heuristic; behaviour that depends on the PyTorch version or hardware |

A bracketed ID such as [R49] cites an entry of the book's research register,
listed at the end of each chapter and in full in Appendix E.

`IN THE REPO`

## 1.8  Common mistakes, and how to catch them

- **Reading line numbers from a newer commit.** Files may change on `main`
  after the book's commit. *Catch it:* `git show ae59238:<file>`.
- **Expecting `generate` to be repeatable.** It samples, from PyTorch's global
  random generator. *Catch it:* pass `top_k=1`, or call `torch.manual_seed`
  first, as `sample.py` does with `--seed`.

`IN THE REPO`

## 1.9  Exercises

1. In `four_steps`, replace `argmax` with `torch.multinomial(probs, 1)`. Does
   it still match `generate(..., top_k=1)`? Why not?
2. What shape does `logits` have inside `generate` after
   `logits[:, -1, :]` for a batch of three prompts and the released model?
3. Which line of `AnuLM.__init__` decides how many blocks the model has?

**Answers.**
1. No. With `top_k=1` every id but the best gets probability 0, so
   `generate` always returns the argmax; an unrestricted draw usually does not.
2. (3, 32,768): B = 3 rows of V scores.
3. `self.layers = nn.ModuleList([... for i in range(cfg.n_layer)])`.

*verified*: against `model.py` at @773b6a8.

> **Build it.** The lab book's Chapter 1 (`course/ch01-smallest-model`) trains
> a 0.9M-parameter model in about two minutes and samples it with this loop.

---

## Key points

- A language model reads a sequence, scores every vocabulary id, picks one and
  appends it; ten lines of Python do it, and `AnuLM.generate` is the same loop
  with a KV cache, a temperature and sampling.
- Called without targets, the forward pass returns logits for the last
  position only, (B, 1, V); the softmax turns them into probabilities.
- With `top_k=1` sampling becomes greedy, and a hand-written loop matches
  `generate` token for token, with and without the cache.
- The released 400M model, run through `sample.py`, writes fluent text whose
  facts it does not check: it is a base model.
- The architecture is `model.py`, 1,224 lines; 70 more root files and four
  folders train, test, serve and adapt it.
- Every citation names a file, a symbol, lines and the pin `773b6a8`, whose
  line numbers hold at the public commit `ae59238`.

> **In one breath.** AnuLM writes text by reading the sequence, scoring all
> 32,768 ids, drawing one and appending it, over and over; `generate` in
> `model.py` is that loop. The rest of the repository trains, tests, serves
> and adapts that one file, and every citation here points at commit 773b6a8.

## References

Research references cited in this chapter (`docs/reference/research-references.md`), each
used only for the claim in its register row:

| ID | Used for | Status |
|---|---|---|
| [R49] | the configurations and modelling code AnuLM follows (Sarvam 30B and 105B) | CHECKED |

Repository evidence, all at @773b6a8:
- `model.py`: banner lines 36, 210, 240, 401, 469, 547, 753;
  `AnuLMConfig.nano_30b` 147–159; `AnuLM.__init__` 782–793;
  `AnuLM.forward` 860–863; `AnuLM.generate` 1059–1116 (loop 1083–1109).
- `test_model.py`: `tiny` 33–38; `generation_is_deterministic_at_top_k_1` 387.
- `sample.py`: `main` 26–90 (defaults 31–32, generation 66–74).
- `quickstart.py`: `train` 170–195 (lines 182–189); `--demo` 220.
- `docs/MODEL_CARD_BASE_V2.md`: the released model; `README.md` line 29: its
  Hugging Face name.
- The file list and line counts: `git ls-tree` and `git show` at the pin, via
  `book/code-examples/ch01_four_steps.py`.
- The public commit: `book/docs/reader/front-matter.md`.

The book example is `code-examples/ch01_four_steps.py`.
