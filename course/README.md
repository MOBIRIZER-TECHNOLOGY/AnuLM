# Build Your Own LLM, from Scratch to Omni

A hands-on course from the AnuLM lab: pre-train a language model on your own
data, then post-train it every way that matters, on one consumer GPU or a
laptop. Every figure is drawn from a real run, and every chapter has a lab
you can finish on a CPU.

**Status:** all fifteen chapters are drafted, each with its lab, figures and
checks; every chapter's `--self-test` passes. `STYLE.md` is the style guide
every chapter follows.

## Two tracks

- **Laptop track (CPU):** every lab's defaults. Tiny models, minutes per run.
- **GPU track (one 16 GB card):** the real 398M AnuLM model, using the released
  base (`toonist/AnuLM-Base-v2-400M`) so nobody has to pre-train for a week.

## Chapters

The full topic plan, with the code and tests behind each chapter, is in
`OUTLINE.md`. Every build chapter teaches this repository's own from-scratch
code, and the test that catches its silent failure.

| | chapter | builds or uses | status |
|---|---|---|---|
| **Part I** | **Foundations** | | |
| 1 | The smallest language model | a self-contained warm-up | **sample written** |
| 2 | Your own data | `build/data.py`, `checks/ch02.py`; `filter_prose.py`, `tools_contamination.py` | **written** |
| 3 | Build the tokenizer | `build/data.py`, `checks/ch03.py`; `bpe.py` | **written** |
| **Part II** | **Build AnuLM from scratch** | | |
| 4 | Embeddings, RMSNorm, SwiGLU | `model.py`, `build/anulm.py`, `checks/ch04.py` | **written** |
| 5 | Positions: RoPE and YaRN | `model.py`, `build/anulm.py`, `checks/ch05.py` | **written** |
| 6 | Attention: GQA, KV cache, sliding window, MLA | `model.py`, `build/anulm.py`, `checks/ch06.py` | **written** |
| 7 | Mixture of experts | `model.py`, `build/anulm.py`, `checks/ch07.py` | **written** |
| 8 | Assemble and pre-train | `model.py`, `train.py`, `build/anulm.py`, `checks/ch08.py` | **written** |
| **Part III** | **Post-training** | | |
| 9 | Supervised fine-tuning | `finetune.py`, `build/post.py`, `checks/ch09.py` | **written** |
| 10 | LoRA and QLoRA | `lora.py`, QLoRA (new), `build/post.py`, `checks/ch10.py` | **written** |
| 11 | Preference tuning (DPO) | DPO (new), `build/post.py`, `checks/ch11.py` | **written** |
| 12 | Decisions and lookup | `decide.py`, `rag.py`, `build/post.py`, `checks/ch12.py` | **written** |
| **Part IV** | **Multimodal and shipping** | | |
| 13 | Hearing and seeing | `speech_vocab.py`, `vision_encoder.py`, `omni_train.py`, `build/post.py`, `checks/ch13.py` | **written** |
| 14 | One model for everything, and debugging it | `omni_train.py`, `experiments/omni_one.py`, `build/post.py`, `checks/ch14.py` | **written** |
| 15 | Shipping | `export_hf.py`, `docs/PUBLISHING.md`, `build/post.py`, `checks/ch15.py` | **written** |

## Run the labs

Each chapter's lab writes `results.json`, and `make_figures.py` draws the
chapter's figures from it. Later labs start from earlier ones' models, so run
them in order:

```bash
cd course/ch01-smallest-model && python lab.py && python make_figures.py
cd ../ch08-pretrain   && python lab.py && python make_figures.py   # writes base.pt: Chapters 9-15 start here
cd ../ch09-sft        && python lab.py && python make_figures.py   # writes sft.pt: Chapters 11 and 15 use it
cd ../ch10-lora-qlora && python lab.py && python make_figures.py
cd ..                 && python build_preview.py                   # -> preview/sample.html
```

Every lab and check runs your code in `build/` by default; add `--impl solution`
to run the reference solutions in `build/solutions/` instead. Check your code
with `python checks/chNN.py`, and see each check catch the bug it is for with
`python checks/chNN.py --self-test`.

Needed: PyTorch, plus numpy (from Chapter 9 on) and safetensors
(Chapter 15). Lab data: Shakespeare (`data/input.txt`, public domain), a
Hindi Wikipedia sample (`data/bench_hindi.txt`, CC BY-SA 4.0), BANKING77
(`data/banking77`, CC BY 4.0), XQuAD and SQuAD passages
(`ch12-decide-rag/data/`, CC BY-SA 4.0), and the repository's cached image
features for Localized Narratives on Open Images (`data/omni/`, CC BY 4.0;
Chapter 13). Chapters 14 and 15 also read the repository's own logs.
