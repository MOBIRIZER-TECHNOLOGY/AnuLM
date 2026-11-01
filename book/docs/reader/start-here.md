# Start here

This book is for Python programmers who have used arrays and want to understand a language model well enough to change it. You do not need a GPU to begin. Chapters 4–7 rebuild the mathematics used later.

**Your first result:** train a very small dense language model on a few repeated sentences, watch its loss fall, and generate a short continuation. This is a learning exercise, not a useful general-purpose model.

## 1. Get the companion code

The organized repository contains both the model and this book. Clone the branch
containing this organization change (or `main` after it is merged):

```bash
git clone https://github.com/MOBIRIZER-TECHNOLOGY/AnuLM.git
cd AnuLM
```

The first example accepts the companion folder explicitly; the chapter examples
use this layout:

```text
AnuLM/
  pyproject.toml
  anulm/
    model.py
    training/
    tokenization/
    inference/
  book/
    docs/reader/start-here.md
    chapters/
    code-examples/
    tools/
```

Run the printed chapter commands from `AnuLM`. If you received only a standalone
book folder, place it at `AnuLM/book` without replacing files you want to keep.

The historical pin `773b6a8` appears throughout the manuscript and is not public.
Public revision `ae592387dc373cc6f579abce444c7f08c9e373a3` contains the original
flat source layout; historical citations refer to that layout. Do not check out
that older revision over the organized book. The project's `docs/source-paths.json`
maps original filenames to current modules. Optional `course/` labs remain
unpublished and are not prerequisites for the first example.

## 2. Prepare Python

Use Python 3.10 or newer and a virtual environment. From the companion repository:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages for the first CPU examples:

```bash
python -m pip install -e .
```

This installs the core package with PyTorch and NumPy. Later chapters name
optional extras. Install your preferred PyTorch build first if you need a
particular CUDA version. Historical measurements used PyTorch 2.11; the revised
first example was checked on CPU with PyTorch 2.14. Timings and last digits can
vary. No pretrained checkpoint, tokenizer download, or corpus is required for
the first CPU example.

## 3. Run your first model

From the companion repository, with the book in the layout above:

```bash
python book/tools/first_prediction.py
```

If the book is elsewhere, run the tool from this book folder and pass the location of the companion source:

```bash
python tools/first_prediction.py --source-root ../AnuLM
```

The tool prints the source location, a two-layer dense model's parameter count, initial and final loss, and a generated continuation. It prints `PASS: loss decreased` when its check succeeds. It keeps all training in memory and writes no checkpoint. Generated text can vary and is not an acceptance test for quality.

**What just happened?** Bytes became token ids, the model predicted the next byte, cross-entropy scored its mistakes, and AdamW updated its weights. Generation then appended one predicted byte at a time. Chapters 3, 8, 11, 16, 20, and 22 unpack these steps. The first model has no routed experts; learn the complete dense path before taking the MoE detour.

## 4. Choose your next step

- Read [Learning paths](learning-paths.md) to choose a route through the book.
- With `book/` inside the companion repository, run `python book/code-examples/trace_forward.py` to see the tensor shapes and parameter arithmetic. It does not need trained weights.
- Run `python book/code-examples/attention_by_hand.py` for the attention example. Trained-checkpoint measurements are an explicit opt-in with `--released`.
- Read [Launch checklist](launch-checklist.md) before treating the demonstration server as an operational service.

## If something goes wrong

| Symptom | What to check |
|---|---|
| `No module named torch` | Activate the environment where you installed the packages; use `python -m pip` with that same Python. |
| `No module named model` | Put `book/` under the companion checkout for chapter examples. For the first tool, pass `--source-root` pointing at the folder containing `model.py`. |
| A checkpoint or `data/v2` file is missing | You selected a released-model measurement. Return to the CPU example, or obtain/rebuild the exact artifact named in that chapter. |
| `bad object 773b6a8` | That pin is unpublished. The first examples use the public implementation; historical Git-based audits cannot yet be reproduced from a public clone. |
| Numbers differ slightly | Check precision, seed, model configuration, package version, and data split before comparing. |
| The toy continuation is repetitive | The first model learns repeated sentences. This demonstrates the training loop; it does not establish generalisation. |

## What this preview promises

It explains the model and provides a small working entry point. The full book has 41 chapters and six appendices. Advanced examples can require extra packages, local artifacts, or unpublished course material; their chapter entry cards identify these requirements. The production chapters describe both implemented behaviour and missing capabilities. This preview is not a production-service certification.
