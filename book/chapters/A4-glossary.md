---
title: "Appendix D · Glossary"
part: "Appendices"
accent: slate
pinned: "@b42998d"
---

# Appendix D  Glossary

Each entry gives the term, a definition in the words the chapters use, and the
chapter that introduces it. The definitions repeat the chapters' "New words"
cards where there is one; Chapter 11's are the reference for Part IV. The
symbols are listed in D.2.

## D.1  Terms

| Term | Meaning | Ch |
|---|---|---|
| active parameters | the parameters one token's computation uses: the total minus the idle experts; `num_params()` returns both | 20 |
| adapter | a small trainable module added beside frozen weights | 31 |
| AdamW | Adam with weight decay applied directly to the weights, not through the gradient; it keeps two moment estimates per parameter | 7 |
| attention weight | a score after the softmax: a number in [0, 1], and a row of them sums to 1 | 11 |
| autocast | PyTorch's mechanism for running chosen operations, chiefly matrix multiplies, in 16-bit while norms, softmaxes and losses stay in 32-bit | 20, 23 |
| autoregressive decoding | producing a sequence one token at a time, each step conditioned on everything before it | 27 |
| balancing bias | a per-expert number added to the router's scores for choosing only; moved by the sign rule, never by gradients | 17, 18 |
| bf16 | bfloat16, a 16-bit float with fp32's exponent range and 8 significant bits | 23 |
| bits per byte | the loss in bits, spread over the bytes of text, so models with different vocabularies compare | 8, 24 |
| block size | the longest context a model accepts, `block_size` | 20 |
| BPE | byte-pair encoding: a tokenizer built by repeatedly merging the most frequent adjacent pair | 9 |
| buffer | a tensor a module keeps but does not train. Persistent ones, such as the router's balancing bias and load counts, are saved in the state dict; non-persistent ones, such as the rotary tables, are not. Neither kind is counted by `num_params()` | 20 |
| calibration | how well a stated confidence matches the observed accuracy | 32 |
| causal mask | the rule that a position may attend only to itself and earlier positions; −∞ on the scores of later ones | 12 |
| checkpoint | a file of trained weights and their configuration, such as `ckpt_base_v2.pt` | 1 |
| chunked loss | computing the output head and the cross-entropy a few thousand positions at a time, so the full (B, S, V) logits never exist | 23 |
| codec | a model that turns a waveform into a short sequence of integer codes and back | 35 |
| combine weight | how much of a chosen expert's output a token receives; the weights sum to the routed scaling factor | 17 |
| context slide | keeping only the last `block_size` tokens once the sequence is longer | 27 |
| control token | an id that marks structure, such as where audio starts, rather than content | 35 |
| cross-entropy | the loss −log p of the true next token, averaged over graded positions | 3 |
| decode step | a call that runs one new token against the KV cache | 27 |
| dense layer | a feed-forward network every token passes through in full; layer 0 here | 16 |
| dispatch path | how an MoE layer runs tokens through their experts: `sparse`, `dense` or `grouped` | 18 |
| document separator | `DOC_SEP`, three newlines, the mark between documents in a corpus file | 8 |
| embedding table | the V × D matrix whose row t is token t's vector | 10 |
| EOS | the end-of-sequence id, put after every document | 8 |
| epoch | one pass over every training window | 21 |
| expert | one of the E small SwiGLUs in an MoE layer | 16, 17 |
| extension | a change this book proposes to the repository, drawn dashed and badged `EXTENSION` | 28, 41 |
| feed-forward network (FFN) | the per-token second half of a block; `anulm/model.py` calls it the MLP | 16 |
| fertility | how many tokens a tokenizer spends on a given text | 9 |
| forward hook | a function PyTorch calls after a module runs, with its inputs and output | 20, 41 |
| fp32 | the 32-bit float. Every `.pt` checkpoint here stores its weights in fp32; an exported folder stores its large tensors in bf16 and its small ones in fp32 by default, or every tensor in fp32 with `--weights-dtype float32` (`anulm/hf/export_hf.py` lines 52–61, 231–233) | 23, 39 |
| frozen encoder | a pretrained network whose weights are never updated here; it only turns input into vectors | 36 |
| gate | a per-channel factor, computed from the token, that scales what the channel writes | 16 |
| gradient | the derivative of the loss with respect to each parameter | 6 |
| gradient accumulation | summing the gradients of several micro-batches before one optimiser step | 22 |
| gradient checkpointing | recomputing a layer's activations in the backward pass instead of storing them | 23 |
| GQA | grouped-query attention: a group of query heads shares one key/value head | 13 |
| greedy decoding | always choosing the most likely token; `top_k=1` | 28 |
| group-limited routing | choosing the best groups of experts first, then experts inside them | 17 |
| head | one complete set of queries, keys and values; the model runs several side by side | 11 |
| held-out split | text kept out of training and used only to measure; `data/v2/val.u16` for the released model | 21, 24 |
| imbalance | the busiest expert's load divided by the mean load, in the worst MoE layer; 1.0 is perfect | 18 |
| inner width | how many numbers the FFN expands a token to; `intermediate_size`, or `moe_intermediate_size` for an expert | 16 |
| instruction tuning, SFT | further training on prompt–answer pairs, with the loss on the answers | 30 |
| key | the vector a position uses to say what it contains | 11 |
| known-wrong implementation | a copy of the code with one classic bug planted, used to prove a test can fail | 41 |
| KV cache | the keys and values of tokens already processed, kept per layer so they are not recomputed | 27 |
| learning rate | the size of each optimiser step | 7 |
| logits | one raw score per vocabulary id; larger means more likely | 1 |
| LoRA | a frozen weight plus a trainable low-rank product **BA**, scaled by α/r, with **B** starting at zero | 31 |
| loss | the average of −log p over the true next tokens | 3 |
| MLA | multi-head latent attention: keys and values compressed into one cached latent, plus one shared rotary key | 19 |
| MoE | mixture of experts: a layer that sends each token to A of E experts chosen by a router | 17 |
| nats | the unit of a loss taken with the natural logarithm; divide by ln 2 for bits | 3 |
| nucleus | the set top-p keeps, renormalised to sum to 1 | 28, 41 |
| packing | laying several short examples end to end in one training row | 30 |
| pin, the | commit `b42998d`, the `anulm/` package layout after the code-quality pass; every line number in this book refers to it | 1 |
| pre-norm | normalising a sublayer's input rather than its output | 15 |
| prefill | the first call, which runs the whole prompt at once and fills the cache | 27 |
| projector | the small trained network that maps an encoder's vectors to the model's width D | 36 |
| prompt mask | target −1 on every prompt position, so that only answer tokens are graded | 30 |
| QK-norm | an RMSNorm on each head's queries and keys before the rotation | 11, 13 |
| query | the vector a position uses to ask what it is looking for | 11 |
| repetition penalty | dividing a seen token's positive logit, or multiplying a negative one, by a constant | 28 |
| residual stream | the running sum of vectors that every layer reads from and adds to | 11, 15 |
| resume point | `<out>.last`, the checkpoint that holds optimiser state and sampler position as well as weights | 25, 41 |
| RMSNorm | dividing a vector by its root mean square, then multiplying by a learned gain; no mean is subtracted | 15 |
| RoPE | rotary position embedding: rotating each query and key pair by an angle proportional to its position | 14 |
| routed scaling factor | a constant, 2.5, multiplying every combine weight | 17 |
| router | the module that scores every expert for every token and picks A of them | 17 |
| router z-loss | a regulariser on the size of the router's logits | 18 |
| sampling | drawing the next token from the model's distribution, after reshaping it | 28 |
| score | the dot product of one query with one key: how well they match | 11 |
| SDPA | PyTorch's `scaled_dot_product_attention`, the kernel `_sdpa` calls | 11 |
| sequence-balance loss | a regulariser that penalises uneven expert use within each sequence | 18 |
| shape | the size of each axis, written (B, S, D) | 4 |
| shared expert | an expert every token uses, outside the routing; the released model has none | 17 |
| sign rule | the balancing update: lower an overloaded expert's bias by a fixed step, raise a starved one's | 18 |
| sliding window | a limit on how far back a position may attend; 256 tokens on the released model's layers 0–9 | 12, 29 |
| softmax | the function that turns logits into probabilities that sum to 1 | 5 |
| SwiGLU | the gated feed-forward network: down(SiLU(gate(x)) ⊙ up(x)) | 16 |
| temperature | the number logits are divided by before the softmax; below 1 sharpens, above 1 flattens; 0 or less means greedy, the argmax with no random draw | 5, 28 |
| tensor | an n-dimensional array of numbers with one dtype, such as fp32 or bf16 | 4 |
| token | one unit the model reads or writes; here, one integer id | 8 |
| top-k | sampling only from the k largest logits | 28 |
| top-p | sampling only from the smallest set of most likely tokens whose probabilities add up to at least p | 28, 41 |
| untied tables | separate input embedding and output head matrices (`tie_word_embeddings = False`) | 10 |
| value | the vector a position hands over if it is chosen | 11 |
| vocabulary | the set of ids the model knows; its size is V | 8 |
| window (training) | one stretch of `block_size` + 1 ids, cut into inputs and targets shifted by one | 21 |
| YaRN | a way to extend RoPE past the training length: rescale slow frequencies, keep fast ones, and multiply the attention scores by mscale² > 1, which sharpens the softmax (a lower temperature) | 29 |

## D.2  Symbols

The notation is Part IV's, fixed in Chapter 11 and repeated with both models'
values in Chapter 20.

| Symbol | Meaning | `tiny()` | Released |
|---|---|---|---|
| B | sequences in a batch | 2 | any |
| S | new tokens per sequence in this call | 5 | up to 1,024 |
| T | key length: cached tokens plus the S new ones | 5 | up to 1,024 |
| D | hidden size, `hidden_size` | 32 | 1,024 |
| H | query heads, `n_head` | 4 | 16 |
| H<sub>kv</sub> | key/value heads, `n_kv_head` | 2 | 4 |
| D<sub>h</sub> | width of one head, `head_dim` | 8 | 64 |
| L | layers, `n_layer` | 3 | 20 |
| V | vocabulary size, `vocab_size`; never the value matrix, which is bold **V** | 64 | 32,768 |
| E | routed experts per MoE layer, `num_experts` | 8 | 24 |
| A | experts active per token, `num_experts_per_tok` | 2 | 4 |
| I | inner width of the dense layer 0, `intermediate_size` | 64 | 2,816 |
| I<sub>e</sub> | inner width of one expert, `moe_intermediate_size` | 16 | 192 |
| N | rows the router sees, N = B·S | 10 | B·S |
| **Q**, **K**, **V** | matrices of queries, keys and values | | |
| q, k, v | one query, key or value vector | | |
| d | Chapter 11's short name for D<sub>h</sub> | 8 | 64 |
| γ | an RMSNorm's learned gain | | |
| ε | the constant inside RMSNorm's square root, `rms_norm_eps` | 10⁻⁶ | 10⁻⁶ |
| w<sub>i</sub> | the combine weight of expert i; a token's weights sum to the routed scaling factor | | |
| n<sub>e</sub> | (token, slot) pairs that reach expert e in one call (Appendix A) | data-dependent | data-dependent |
| n<sub>shared</sub> | shared experts per MoE layer, `num_shared_experts` | 1 | 0 |
| D<sub>c</sub> | MLA latent width, `kv_lora_rank` (Chapter 19; Appendix A.3 writes it r) | 16 in A.3 | no MLA |
| D<sub>n</sub>, D<sub>r</sub>, D<sub>v</sub> | MLA's non-rotated q/k width, rotated q/k width and value width (Chapter 19; Appendix A.3 writes them d<sub>n</sub>, d<sub>r</sub>, d<sub>v</sub>) | 8, 8, 8 in A.3 | no MLA |
| p | top-p's probability threshold (Chapters 28, 41) | | |

**Letters that mean different things in different parts.** The notation is
fixed within a part, not across the book:

| Letter | Part IV (Chapters 11–20) | Elsewhere |
|---|---|---|
| B | sequences in a batch | **B**, bold: LoRA's out × r matrix, which starts at zero (Chapter 31) |
| A | experts active per token | **A**, bold: LoRA's r × in matrix (Chapter 31) |
| r | (not used; Chapter 19 writes MLA's latent width D<sub>c</sub>) | MLA's latent width in Appendix A.3; LoRA's rank (Chapter 31) |
| **M**, M | **M**, bold: the attention mask added to the scores (Chapter 11) | M, plain: micro-batches per optimiser step, `--grad-accum` (Chapters 21–22) |
| **V**, V | **V**, bold: the value matrix; V, plain: the vocabulary size | |
| p | a probability, as in p<sub>b,s</sub> | top-p's threshold (Chapters 28, 41) |

*verified*: the values against `tests/test_model.py`'s `tiny()` (line 44) and the
configuration stored in `ckpt_base_v2.pt` (Appendix B.1).
