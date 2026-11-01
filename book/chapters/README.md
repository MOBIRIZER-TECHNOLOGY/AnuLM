# Chapters and appendices

Start with [the reader guide](../docs/reader/start-here.md) or choose a [learning path](../docs/reader/learning-paths.md).

## I · Foundations

- [1 · What this book builds](01-what-this-book-builds.md)
- [2 · From counts to transformers](02-from-counts-to-transformers.md)
- [3 · The one job: a probability for every next token](03-the-one-job.md)
## II · Just enough mathematics

- [4 · Tensors, shapes and the matrix multiply](04-tensors-shapes-and-the-matrix-multiply.md)
- [5 · Softmax, temperature and cross-entropy](05-softmax-temperature-and-cross-entropy.md)
- [6 · Gradients and backpropagation](06-gradients-and-backpropagation.md)
- [7 · Optimisation: AdamW, schedules and clipping](07-optimisation-adamw-schedules-and-clipping.md)
## III · From text to vectors

- [8 · Bytes as tokens](08-bytes-as-tokens.md)
- [9 · Byte-level BPE from scratch](09-byte-level-bpe.md)
- [10 · Embeddings and the output table](10-embeddings-and-the-output-table.md)
## IV · Inside the transformer

- [11 · Attention from first principles](11-attention.md)
- [12 · Causal masking](12-causal-masking.md)
- [13 · Many heads, shared keys: multi-head and grouped-query attention](13-multi-head-and-gqa.md)
- [14 · Word order: rotary position embeddings](14-rotary-positions.md)
- [15 · Normalisation and the residual stream](15-normalisation-and-the-residual-stream.md)
- [16 · The feed-forward network: SwiGLU](16-swiglu.md)
- [17 · Mixture of experts I: the router](17-router.md)
- [18 · Mixture of experts II: dispatch and balance](18-dispatch-and-balance.md)
- [19 · Latent attention (MLA)](19-latent-attention.md)
- [20 · The whole model, traced end to end](20-the-whole-model.md)
## V · Training

- [21 · Data: corpora, documents, windows and splits](21-data.md)
- [22 · The training loop, line by line](22-the-training-loop.md)
- [23 · Precision, memory and speed](23-precision-memory-speed.md)
- [24 · Measuring progress honestly](24-measuring-progress.md)
- [25 · Checkpoints, resuming and failure](25-checkpoints-and-resuming.md)
- [26 · Debugging: when the loss goes down and the model is still wrong](26-debugging-silent-failures.md)
## VI · Generation

- [27 · Autoregressive decoding and the KV cache](27-decoding-and-the-kv-cache.md)
- [28 · Choosing the next token](28-choosing-the-next-token.md)
- [29 · Long context: sliding windows and YaRN](29-long-context.md)
## VII · After pre-training

- [30 · Instruction tuning](30-instruction-tuning.md)
- [31 · LoRA](31-lora.md)
- [32 · Classification heads and calibration](32-classification-and-calibration.md)
- [33 · Retrieval and readers](33-retrieval-and-readers.md)
- [34 · Preference optimisation and reinforcement learning](34-preference-optimisation-and-rl.md)
## VIII · Beyond text

- [35 · A wider vocabulary and inputs that are not tokens](35-wider-vocabulary.md)
- [36 · Seeing and hearing: frozen encoders and projectors](36-frozen-encoders.md)
- [37 · One model, many tasks](37-one-model-many-tasks.md)
## IX · To production

- [38 · Serving: engine, API and streaming](38-serving.md)
- [39 · Exporting to the ecosystem](39-exporting.md)
- [40 · The production gap](40-the-production-gap.md)
## X · Capstone

- [41 · Capstone: train it, open it, change it](41-capstone.md)
## Appendices

- [Appendix A · Tensor shapes for every module](A1-shape-trace.md)
- [Appendix B · The 400M arithmetic](A2-parameter-arithmetic.md)
- [Appendix C · Test index](A3-test-index.md)
- [Appendix D · Glossary](A4-glossary.md)
- [Appendix E · Bibliography](A5-bibliography.md)
- [Appendix F · Reproducing every number](A6-number-audit.md)
