---
title: "Appendix I · Common mistakes, collected"
part: "Appendices"
accent: slate
pinned: "@2e80b3e"
---

# Appendix I  Common mistakes, collected

The "Common mistakes" sections of the 29 chapters that have one, in one
table: 108 rows, 93 of them with the check that catches the mistake. The
wording is the chapters' own, shortened; the section in the first column has the example
that shows the mistake happening and the test that pins it. A dash means the chapter does
not state that cell. Nearly every row shares the property that made the book collect them:
the code still runs, and the loss still falls.

This table is **generated** by `book/tools/make_mistakes.py` from the chapter files, and
`book/code-examples/ch41_audit.py` regenerates it on every run and compares. To change a
row, change the chapter.

| § | Mistake | What happens, silently | Catch it |
|---|---|---|---|
| 1.8 | Reading line numbers from a newer commit | Files may change on `main` after the book's commit | `git show 2e80b3e:<file>` |
| 1.8 | Expecting `generate` to be repeatable | It samples, from PyTorch's global random generator | pass `temperature=0` (exact greedy, no draw) or `top_k=1`, or call `torch.manual_seed` first, as `anulm/inference/sample.py` does with `--seed` |
| 3.6 | Unshifted targets | `model(x, x)` grades each token against itself; a trained model learns to copy. The tests use it only untrained | check `y[:, :-1] == x[:, 1:]` for a language-model batch |
| 3.6 | Averaging over ungraded positions | Dividing by B·S when some targets are −1 understates the loss | compare with 3.2's check |
| 4.7 | `reshape` or `view` where a `transpose` was meant | `x.view(B, H, S, Dh)` on a (B, S, H·D<sub>h</sub>) tensor has the right shape and scrambles positions with heads | compare one head against its slice of W, as 4.3 does |
| 4.7 | Reducing without `keepdim` | — | assert the reduced tensor's shape before dividing by it |
| 5.6 | Softmax over the wrong axis | Over the batch instead of the vocabulary, rows no longer sum to 1 | `probs.sum(-1)` must be all ones |
| 5.6 | Taking `log(softmax(z))` | A probability that underflows to 0 gives −∞ | use `F.log_softmax` or `F.cross_entropy`, which never form the probability first |
| 7.6 | Decaying the norm gains | Putting every parameter in one decayed group drags each RMSNorm gain toward zero | print each group's tensor names once, as 7.2 does |
| 7.6 | Extending a cosine run without rewarmup | A larger `--steps` moves the whole curve, and the first resumed step jumps to a much higher rate | print `lr_at` at the resume step before resuming, or use WSD |
| 7.6 | Clipping before the last micro-batch | Clipping inside the accumulation loop caps a partial gradient | clip once, after the loop, as `anulm/training/train.py` does |
| 8.7 | EOS cast to a byte | `.to(torch.uint8)` turns 257 into 1; the model never sees an end | assert the dtype after loading, as the test does |
| 8.7 | CRLF left in | `DOC_SEP` never matches; the corpus becomes one document | `anulm/training/train.py` prints an EOS count (line 699) only when there is one: its absence is the symptom |
| 8.7 | nats per token compared across vocabularies | a 259-id and a 32k model are scored in different units | report bits per byte, and bits per character across scripts |
| 8.7 | generated ids decoded with `bytes()` | raises on 256 to 258 after minutes of generation | drop ids above 255 and decode with `errors="replace"` |
| 8.7 | a separator that can occur inside a document | documents split at random | check that no document contains `DOC_SEP` before joining |
| 9.9 | decoding token by token | a token can be part of a character; each piece becomes U+FFFD | join the bytes, then decode once, as `decode` does; while streaming, `decode_complete` |
| 9.9 | encoding by frequency, or left to right | different ids from training; the model reads text it never saw | encode by rank; compare with a naive replay of the merges |
| 9.9 | a second tokenizer that differs on one merge | plausible text, different numbers | compare ids token for token, as `hf_tokenizer_conversion_is_exact` does |
| 9.9 | removing stale heap entries eagerly | correct but slow | skip them on pop, as lines 120–121 do |
| 9.9 | `--vocab N` without the EOS slot | N + 1 ids, one more than the model's table | `vocab_size == 256 + len(merges) + 1`, the test's first assertion |
| 9.9 | comparing losses across tokenizers | different units | bits per byte (Chapter 8) |
| 10.7 | a large `initializer_range` | logits start wide; the first loss sits well above ln V | assert the first loss near ln V and each weight's starting spread, as `initial_loss_is_ln_vocab` does |
| 10.7 | tying by copying instead of sharing | two tensors that drift apart, twice the memory | check `lm_head.weight is embed_tokens.weight`, as `tied_embeddings_share_one_table` does |
| 10.7 | a vocabulary larger than the table | an index error, or a wrong row if ids were remapped | assert `max(ids) < vocab_size` when loading data |
| 10.7 | a cache keyed on `_version` | fused optimizers and `.data` updates do not bump it; stale tables | drop caches on `train()`, as `_table` does |
| 10.7 | reading rare rows' norms as meaning | a row's size reflects how often it got a gradient against decay, not only what it means | compare against token counts, as 10.5 does |
| 11.10 | scale dropped | attention sharpens with width (11.5) | assert `scale` against `head_dim ** -0.5` times the temperature, as `yarn_temperature_reaches_attention` does |
| 11.10 | scale applied twice | scores divided by 64 at width 64: nearly uniform attention | the same assertion |
| 11.10 | softmax over queries | weights still look plausible | every row of weights sums to 1 over the keys |
| 11.10 | dropout at evaluation | PyTorch applies any `dropout_p` it is given [R50] | pass 0.0 outside training (line 492); two evaluation passes must match |
| 11.10 | boolean mask inverted | PyTorch's `True` means *may attend*; `nn.MultiheadAttention`'s padding mask means the opposite [R50] | `incremental_forward_matches_full_forward` (`tests/test_model.py` line 658), `sliding_window_limits_reach` (line 750, shown in 12.6) and `sliding_window_agrees_with_cache` fail with `_attn_mask`'s output inverted |
| 11.10 | comparing kernels exactly | sums in different orders differ in the last bits | compare with a tolerance suited to the dtype |
| 12.7 | Masking after the softmax | — | every row must sum to 1 |
| 12.7 | Trusting `is_causal` with a cache | — | feed a sequence whole and in multi-token pieces, as `incremental_forward_matches_full_forward` does |
| 12.7 | A window off by one | It still trains | perturb exactly `window` back, as `sliding_window_limits_reach` does, and compare the mask with its definition, as `sliding_window_agrees_with_cache` does |
| 12.7 | An empty row | from a mask that hides a query's own key: NaN | the `sliding_window >= 1` check |
| 13.7 | Tiling instead of interleaving | `repeat` pairs query head h with key/value head h mod H<sub>kv</sub> | the index check of 13.4, with H<sub>kv</sub> > 1 |
| 13.7 | Caching the copies | Storing k after `repeat_interleave` multiplies the cache by `n_rep` | `cache_holds_what_it_should` |
| 14.8 | Mixing the two layouts | It runs and gives wrong scores | compare with the original implementation on one input |
| 14.8 | Cached steps that restart at position 0 | — | `incremental_forward_matches_full_forward` (Chapter 12) |
| 14.8 | A stale table | after `.to()` or a dtype change | `rotary_cache_tracks_dtype_and_positions` |
| 15.6 | Statistics in 16 bits | Fine in bf16 at these widths, garbage in fp16 | compare against an fp64 reference on large inputs (15.2) |
| 15.6 | Post-norm by accident | `x = norm(x + f(x))`. It still trains | the lab book's `block_is_a_pre_norm_residual` check |
| 15.6 | Unscaled output projections | — | check the weights' spread; the loss test cannot (15.5) |
| 16.7 | A forgotten transpose | changed the output by more than its own scale (0.5985 against 0.4609), without an error | test with D ≠ I, where the shapes no longer fit |
| 16.7 | Swapped `gate_proj` and `up_proj` | changed it by about a third of its scale (0.1585). Both are (I, D), so no shape test sees it, and a model trained from scratch learns just as well either way | compare the output with the formula, down(silu(gate(x)) · up(x)), as `rmsnorm_and_swiglu_match_their_formulas` (`tests/test_model.py` line 252) does; or with the source on the same input, as `transformers_wrapper_matches_the_native_model` (line 1450) does. That second test returns early, and counts as a pass, when `transformers` is not installed |
| 16.7 | A projection built but never used | still lets the loss fall | `all_parameters_receive_gradients` (`tests/test_model.py` line 415) |
| 17.8 | Weighting with the biased scores | trains, but with the released model's biases near 430 every chosen expert gets about 2.5/4 = 0.625, whatever its score; negative biases could push the sum towards 0 instead. The example shows the flattening: | `router_weights_come_from_unbiased_scores` |
| 17.8 | Routing under autocast | even with fp32 inputs (17.4) | compare `topk_idx` with autocast on and off on real activations |
| 17.8 | Re-initialising the router as a Linear | — | check `max \|w\|` against 1/√D after construction, as the example does |
| 18.8 | Storing the bias in bf16 | (18.6). Many expert choices change, while the loss rises only 0.003 nats (Chapter 39), so the loss will not show it | compare `topk_idx` before and after saving |
| 18.8 | No `update_expert_biases` | in a new loop: nothing fails | log the ratio, as `anulm/training/train.py` does |
| 18.8 | The bias as a `Parameter` | moved by AdamW | `expert_bias_is_not_trainable` (`tests/test_model.py` line 400) |
| 19.6 | Rotating the latent | `kv_b_proj` then mixes rotated dimensions, and scores stop depending on relative position only. A cache test still passes, since cached and uncached passes rotate alike | shift the whole sequence. The example runs the repository's module and a mutant that also rotates c, at positions 0–7 and 100–107: |
| 19.6 | Sizing the rotary tables to `head_dim` | For MLA they must be D<sub>r</sub> wide (line 847) | the shapes fail loudly in `apply_rope` |
| 19.6 | Caching expanded keys | It works and saves nothing | `cache_holds_what_it_should` checks what each attention type stores |
| 20.11 | expecting S rows of logits without targets | the shortcut returns (B, 1, V); the lab book's Chapter 16 check failed this way | assert the logits' shape |
| 20.11 | forgetting the shift | `targets = idx` trains copying; the loss falls for the wrong reason | `targets[:, :-1]` must equal `idx[:, 1:]` |
| 20.11 | counting parameters from the state dict | 397,660,560, not 397,659,648: it includes 912 router buffer numbers | count `model.parameters()` or use `num_params()` |
| 20.11 | routing under 16-bit autocast | discrete choices flip | compare `Router` outputs with autocast on and off |
| 20.11 | a sublayer never added to the stream | the loss still falls through the others | `all_parameters_receive_gradients`; a forward hook as in `trace_forward.py` |
| 21.10 | a tail split of an ordered file | no loss curve shows it; the held-out split tests one source | count the sources in the held-out split |
| 21.10 | `y = x` | trains a copier | assert `torch.equal(x[:, 1:], y[:, :-1])` |
| 22.9 | reading progress from the train loss | its standard deviation over the last 20 log lines was 0.271 | — |
| 22.9 | accumulating without dividing | the loss will not show it | a gradient comparison: `gradient_accumulation_equals_large_batch` |
| 24.8 | judging a move between two evaluations by the printed ± | — | the ± is the level's error; for a move, take the paired error of per-batch differences (24.5), which the log does not print |
| 24.8 | reading a level to 0.01 from the default 80 windows | — | pass `--eval-windows 400` on a new run |
| 24.8 | comparing runs whose window sets differ (`--batch-size` changes the random set) | — | compare the checkpoints' `eval_spec`, or rescore both with `anulm/eval/eval_bench.py` |
| 24.8 | comparing losses across tokenizers | — | bits per byte, on the same clean text |
| 24.8 | quoting one bits per byte for a mixed split as the model's | — | report each source |
| 24.8 | changing `--eval-iters` mid-run | — | never; new windows shift the mean |
| 25.9 | saving weights only, then resuming | — | compare a resumed run with an uninterrupted one, as in 25.5 |
| 25.9 | writing the checkpoint in place | — | kill a save halfway (25.2) and load what is left |
| 25.9 | loading a resume point onto the GPU and keeping it | — | `map_location="cpu"`, then `del` it; watch tokens per second after a resume |
| 25.9 | a restart loop that dies on the same file | — | log each restart's exit; the coder run looped for three hours |
| 25.9 | saving `parameters()` instead of `state_dict()` | — | buffers such as `expert_bias` vanish; compare a resumed run with an uninterrupted one (25.5). `expert_bias_is_not_trainable` checks the model's state dict, not what a writer saves |
| 25.9 | trusting `atomic_save` and the fallback to keep working | — | no test in `tests/test_model.py` covers either; the book example's part 2 and run C are the checks |
| 25.9 | loading a downloaded `.pt` with `torch.load(weights_only=False)` | — | use `safe_load` or `load_checkpoint`; set `ANULM_TRUST_PICKLE=1` only for a file you made |
| 25.9 | catching every load error and falling back | — | a refusal is not damage: re-raise `UnsafeCheckpointError`, as 25.4 does. No repository test covers the resume path's re-raise; the book example's part 3 is the check |
| 26.11 | trusting a falling loss | — | add an invariance or equivalence test for the component |
| 26.11 | a test that has never failed | — | break the code on purpose and watch it fail (26.3) |
| 26.11 | a test with a tolerance wider than the bug | — | pick the tolerance from the bug's size; the causal test allows 10⁻⁵, the leak moved 0.0897 |
| 26.11 | a test that can pass by luck | — | make the luck unlikely: the golden builder's test uses 400 distinct words |
| 26.11 | testing on toy data only | — | measure on the real model too, as 26.4 did for routing |
| 30.7 | Grading the prompt | each still trains, and each is caught by one assertion of the test below | — |
| 30.7 | dropping the answer's space | each still trains, and each is caught by one assertion of the test below | — |
| 30.7 | forgetting EOS | each still trains, and each is caught by one assertion of the test below | — |
| 30.7 | cutting long examples | each still trains, and each is caught by one assertion of the test below | — |
| 30.7 | A split by row, not by question | 30.5. Every copy of a question must sit on one side; check with the question key, not the exact string | — |
| 31.8 | B initialised randomly | step 0 is no longer the base | logits before and after `apply_lora` |
| 31.8 | The scale forgotten when merging | — | with merged vs adapted logits, as in 31.5 |
| 31.8 | Only the adapters saved, on an MoE model | — | by comparing the expert choices of the merged and the trained model (31.4) |
| 32.7 | Reading the score one token early | before the option's name is read | — |
| 32.7 | Options in a fixed order in training | the head learns slots | — |
| 32.7 | Fitting τ on test data, or another task's | RESULTS §36's borrowed encoder kept an ECE of 18.12% zero-shot | — |
| 33.6 | The question after the passages, for a causal reader | no passage token can condition on it | — |
| 33.6 | Fusing scores instead of ranks | the scales differ | a list scaled by 10 must fuse the same |
| 33.6 | Judging retrieval by recall alone | hybrid recalled at least as well as dense (MLQA R@3 45% against 43%, R@10 59% against 56%; demo R@10 100% for both) and answered worse, 2/20 against 5/20 (RESULTS §33) | — |
| 34.7 | scoring an answer on `anulm/model.py` without targets | — | — |
| 34.7 | letting gradient reach the reference | — | — |
| 34.7 | watching only the DPO loss | — | — |
| 34.7 | all-equal GRPO groups | — | — |
| 41.7 | Trusting a bare command's step count | The default is 2,000 steps | read the epochs planned on the `sampler:` line before you wait |
| 41.7 | Comparing a val loss across tokenizers | Nats per token from a 32k tokenizer and nats per byte are different units | compare bits/byte, which `anulm/training/train.py` prints for both (Chapter 24) |
| 41.7 | A test that cannot fail | The sampling test alone passes the unrenormalised top-p | keep a known-wrong version beside each extension, and require that it fail |
| 41.7 | Checking only that sampling runs | Off by one, with p above the top token's probability, runs fine on too small a set, and is right where a cumulative sum equals p | hand-worked sets, at a p below the top probability and one between two cumulative sums |
| 41.7 | Reading "80 passed" without the rest of the line | A test that cannot run is counted as skipped, not passed (41.5) | read the summary for "skipped", and the SKIP lines for why; install the missing package and run again |

## I.1  Reading the table

Three kinds of row recur. **Right size, wrong meaning**: a `view` where a `transpose` was
meant (4.7), `repeat` where `repeat_interleave` was meant (13.7), a mask applied after the
softmax (12.7); the tensor has the right shape and the loss falls anyway. **The wrong thing
saved or loaded**: EOS cast to a byte (8.7), weights saved without the optimiser and then
resumed (25.9), a checkpoint written in place (25.9). **The wrong instrument**: progress read
from the train loss (22.9), a move judged by the printed ± (24.8), a test that has never
been seen to fail (26.11). The catch column is almost always the same move: an equivalence
or invariance check, stated as an assertion, that a falling loss cannot satisfy by accident
(Chapter 26).

`VERIFIED` · generated from the chapters at @2e80b3e.
