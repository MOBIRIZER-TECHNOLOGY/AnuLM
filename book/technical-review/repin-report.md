# Re-pin report: @fdbb09a -> @b42998d

Written by `book/tools/repin.py`. Every line citation it moved, with the code at
the old line and at the new one. Rows marked UNMAPPED, CHECK, EXCERPT or OUT OF
FILE were **not** resolved automatically and need a person.

| kind | count |
|---|---|
| EXCERPT | 3 |
| UNMAPPED | 8 |
| def | 37 |
| edited | 2 |
| edited-quote | 5 |
| override | 78 |
| reattributed | 100 |
| same | 186 |
| sweep | 120 |

## Needs a person (11)

| where | file | old | new | kind | old code | new code |
|---|---|---|---|---|---|---|
| chapters/12-causal-masking.md:378 | tests/test_model.py | 679–686 | ? | UNMAPPED | def sliding_window_agrees_with_cache(): |  |
| chapters/12-causal-masking.md:378 | tests/test_model.py | 679 | ? | EXCERPT | header range unresolved: sliding_window_agrees_with_cache |  |
| chapters/13-multi-head-and-gqa.md:212 | tests/test_model.py | 1602–1623 | ? | UNMAPPED | def native_gqa_matches_repeat_interleave(): |  |
| chapters/31-lora.md:353 | tests/test_model.py | 1111–1115 | ? | UNMAPPED | trained_bias = router[sorted(router)[0]] |  |
| chapters/31-lora.md:353 | tests/test_model.py | 1111 | ? | EXCERPT | header range unresolved: lora_starts_as_the_base_model_and_merges_back |  |
| chapters/37-one-model-many-tasks.md:355 | anulm/multimodal/omni_train.py | 458–465 | ? | UNMAPPED | if step % args.eval_every == 0 or step == args.steps: |  |
| chapters/37-one-model-many-tasks.md:355 | anulm/multimodal/omni_train.py | 458 | ? | EXCERPT | header range unresolved: train |  |
| chapters/37-one-model-many-tasks.md:661 | experiments/omni_one.py | 222–238 | ? | UNMAPPED | def prepare_training(args): |  |
| chapters/37-one-model-many-tasks.md:731 | experiments/omni_one.py | 222–238 | ? | UNMAPPED | def prepare_training(args): |  |
| chapters/A3-test-index.md:136 | tests/test_model.py | 1607 | ? | UNMAPPED | m = AnuLM(tiny(n_kv_head=1, sliding_window=4, max_window_layers=1)).ev |  |
| chapters/A3-test-index.md:137 | tests/test_model.py | 1111–1115 | ? | UNMAPPED | trained_bias = router[sorted(router)[0]] |  |

## Resolved by a person (2026-11-04)

The dry run had 91 rows needing a person: 78 CHECK and 2 OUT OF FILE rows, now
decided in `repin-overrides.tsv` (each names the file the number belongs to,
or `-` for a number that is not a line), and the 11 rows above, fixed by hand
because the code itself changed in 52d26f0 and 16d8f66:

| where | resolution |
|---|---|
| chapters/12-causal-masking.md:378 | excerpt rewritten from `sliding_window_agrees_with_cache` 776–819 (mask checked against its definition; cached-key perturbation; greedy agreement); `ch12_masks.py` rerun, its table and the prose updated (the test now fails all four wrong masks) |
| chapters/13-multi-head-and-gqa.md:212 | excerpt rewritten from 1767–1796 (1, 2 and 3 key/value heads); prose says a tiled pairing now fails |
| chapters/31-lora.md:353 | two excerpts, 1207–1218 (balancer step, drift asserted) and 1247–1258 (naive merge must differ); the "never runs" Details rewritten as history; `lora_scale_is_alpha_over_r` 1262 added |
| chapters/37-one-model-many-tasks.md:355 | excerpt now 460–467, with `or step == args.stop_after`; the `--stop-after` Details says it evaluates at that step |
| chapters/37-one-model-many-tasks.md:661, 731 | `prepare_training` 222–247: a finished training is kept unless `--retrain` (234–237, flag 450) |
| chapters/A3-test-index.md:136, 137 | the four gaps closed by 52d26f0 moved out of C.5 into a paragraph; C.1–C.4 rows rewritten and the four new tests added |

The tool also left some numbers it does not parse: `(N)` cells in the source
map's test table, Appendix C's line column, and a few evidence-list spans
(`flush`, `meta.json` in Chapter 21; Chapter 26's test spans; Chapter 39's).
They were remapped with the same line map and checked by
`verify_source_map.py`, `ch41_audit.py` and `verify_chapters.py`.

## Every change

| where | file | old | new | kind | old code | new code |
|---|---|---|---|---|---|---|
| chapters/01-what-this-book-builds.md:113 | tests/test_model.py | 398 | 495 | def | 'generation_is_deterministic_at_top_k_1' | def generation_is_deterministic_at_top_k_1(): |
| chapters/01-what-this-book-builds.md:127 | anulm/model.py | 1116 |  | reattributed | 'generate' is defined there |  |
| chapters/01-what-this-book-builds.md:212 | anulm/inference/sample.py | 34–35 |  | override | decided in repin-overrides.tsv |  |
| chapters/01-what-this-book-builds.md:365 | anulm/quickstart.py | 226 |  | reattributed | quoted code is in anulm/quickstart.py, not anulm/inference/sample.py |  |
| chapters/01-what-this-book-builds.md:532 | anulm/model.py | 0 | 0 | sweep | layer 0 dens |  |
| chapters/01-what-this-book-builds.md:532 | anulm/model.py | 811 |  | override | decided in repin-overrides.tsv |  |
| chapters/01-what-this-book-builds.md:533 | anulm/model.py | 1144–1173 | 1144–1173 | sweep | d' 915–918; 'AnuLM.generate' 1115–1180 (loop 1144–1173; |  |
| chapters/01-what-this-book-builds.md:534 | anulm/model.py | 1161–1162 | 1161–1162 | sweep | the 'temperature <= 0' greedy branch 1161–1162). |  |
| chapters/01-what-this-book-builds.md:535 | tests/test_model.py | 398 | 495 | same | def generation_is_deterministic_at_top_k_1(): | def generation_is_deterministic_at_top_k_1(): |
| chapters/01-what-this-book-builds.md:536 | anulm/inference/sample.py | 34–35 | 34–35 | sweep | inference/sample.py': 'main' 26–93 (defaults 34–35, gen |  |
| chapters/01-what-this-book-builds.md:536 | anulm/inference/sample.py | 69–77 | 69–77 | sweep | y': 'main' 26–93 (defaults 34–35, generation 69–77). |  |
| chapters/02-from-counts-to-transformers.md:327 | anulm/model.py | 481 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/02-from-counts-to-transformers.md:411 | anulm/bigram_teacher.py | 42–51 | 42–51 | sweep | er.py': 'build_bigram_teacher' 34–62 (counts 42–51), |  |
| chapters/02-from-counts-to-transformers.md:413 | anulm/bidirectional.py | 82–93 | 82–93 | sweep | : 'bidir_hidden' 40–53; 'adapt' 56–105 (MNTP 82–93). |  |
| chapters/03-the-one-job.md:259 | tests/test_model.py | 349–359 | 428–446 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/03-the-one-job.md:259 | tests/test_model.py | 358 | 445 | edited-quote | assert abs(loss.item() - expected) < 0.35, \ | assert abs(loss.item() - expected) < 0.1, \ |
| chapters/03-the-one-job.md:437 | anulm/training/train.py | 906–908 | 906–908 | sweep | 906–908 (bit |  |
| chapters/03-the-one-job.md:438 | tests/test_model.py | 349–359 | 428–446 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:64 | anulm/model.py | 461–463 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not book/code-examples/ch0 |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:214 | anulm/model.py | 493 |  | reattributed | quoted code is in anulm/model.py, not book/code-examples/ch04_shapes.p |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:296 | anulm/model.py | 470–477 |  | reattributed | quoted code is in anulm/model.py, not book/code-examples/ch04_shapes.p |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:361 | anulm/model.py | 461 |  | override | decided in repin-overrides.tsv |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:111 | anulm/training/train.py | 645–650 |  | override | decided in repin-overrides.tsv |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:158 | anulm/model.py | 924 |  | reattributed | quoted code is in anulm/model.py, not book/code-examples/ch05_softmax. |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:160 | anulm/model.py | 1152 |  | override | decided in repin-overrides.tsv |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:224 | anulm/inference/sample.py | 34–35 |  | override | decided in repin-overrides.tsv |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:249 | anulm/model.py | 1166–1167 |  | override | decided in repin-overrides.tsv |  |
| chapters/06-gradients-and-backpropagation.md:243 | tests/test_model.py | 448–466 | 545–563 | same | def gradient_accumulation_equals_large_batch(): | def gradient_accumulation_equals_large_batch(): |
| chapters/06-gradients-and-backpropagation.md:314 | tests/test_model.py | 336–345 | 415–424 | same | def all_parameters_receive_gradients(): | def all_parameters_receive_gradients(): |
| chapters/06-gradients-and-backpropagation.md:237 | anulm/model.py | 1116 |  | reattributed | 'AnuLM.generate' is in anulm/model.py, not anulm/training/train.py |  |
| chapters/06-gradients-and-backpropagation.md:239 | anulm/model.py | 913 |  | reattributed | quoted code is in anulm/model.py, not anulm/training/train.py |  |
| chapters/06-gradients-and-backpropagation.md:443 | tests/test_model.py | 336–345 | 415–424 | same | def all_parameters_receive_gradients(): | def all_parameters_receive_gradients(): |
| chapters/06-gradients-and-backpropagation.md:444 | tests/test_model.py | 448–466 | 545–563 | same | def gradient_accumulation_equals_large_batch(): | def gradient_accumulation_equals_large_batch(): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:310 | tests/test_model.py | 1627–1636 | 1800–1809 | same | def wsd_schedule_is_flat_then_decays_to_min_lr(): | def wsd_schedule_is_flat_then_decays_to_min_lr(): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:465 | anulm/muon.py | 59 |  | reattributed | 'zeropower_batched' is defined there |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:489 | tests/test_model.py | 1729 | 1902 | def | 'batched_muon_matches_per_matrix_muon' | def batched_muon_matches_per_matrix_muon(): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:572 | anulm/training/train.py | 416–420 | 416–420 | sweep | 416–420); 'm |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:572 | anulm/training/train.py | 555–556 | 555–556 | sweep | ); 'make_optimizer' 447–474; 'main' defaults 555–556, sch |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:573 | anulm/training/train.py | 852–853 | 852–853 | sweep | 852–853, cli |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:573 | anulm/training/train.py | 874–878 | 874–878 | sweep | 852–853, clip and step 874–878. |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:576 | anulm/muon.py | 152–165 | 152–165 | sweep | 'build_optimizers' 144–182 (split 152–165). |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:577 | tests/test_model.py | 1627–1636 | 1800–1809 | same | def wsd_schedule_is_flat_then_decays_to_min_lr(): | def wsd_schedule_is_flat_then_decays_to_min_lr(): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:578 | tests/test_model.py | 1729 | 1902 | same | def batched_muon_matches_per_matrix_muon(): | def batched_muon_matches_per_matrix_muon(): |
| chapters/08-bytes-as-tokens.md:261 | tests/test_model.py | 1430–1448 | 1595–1613 | same | def byte_loader_inserts_eos_at_document_boundaries(): | def byte_loader_inserts_eos_at_document_boundaries(): |
| chapters/08-bytes-as-tokens.md:174 | anulm/training/train.py | 68–85 |  | override | decided in repin-overrides.tsv |  |
| chapters/08-bytes-as-tokens.md:329 | anulm/training/train.py | 656 |  | override | decided in repin-overrides.tsv |  |
| chapters/08-bytes-as-tokens.md:331 | anulm/training/train.py | 652–655 |  | override | decided in repin-overrides.tsv |  |
| chapters/08-bytes-as-tokens.md:478 | tests/test_model.py | 1430–1448 | 1595–1613 | same | def byte_loader_inserts_eos_at_document_boundaries(): | def byte_loader_inserts_eos_at_document_boundaries(): |
| chapters/08-bytes-as-tokens.md:478 | tests/test_model.py | 1430–1448 | 1595–1613 | sweep | 1430–1448. |  |
| chapters/09-byte-level-bpe.md:402 | tests/test_model.py | 837–852 | 970–985 | same | def bpe_documents_get_eos_and_the_vocab_is_exact(): | def bpe_documents_get_eos_and_the_vocab_is_exact(): |
| chapters/09-byte-level-bpe.md:429 | tests/test_model.py | 1365–1375 | 1530–1540 | same | tok = BPE.train(FALLBACK * 8, 400, verbose=False) | tok = BPE.train(FALLBACK * 8, 400, verbose=False) |
| chapters/09-byte-level-bpe.md:214 | anulm/bpe.py | 120–121 |  | override | decided in repin-overrides.tsv |  |
| chapters/09-byte-level-bpe.md:427 | tests/test_model.py | 1355–1358 |  | reattributed | quoted code is in tests/test_model.py, not anulm/bpe.py |  |
| chapters/09-byte-level-bpe.md:427 | tests/test_model.py | 1355–1358 | 1520–1523 | same | try: | try: |
| chapters/09-byte-level-bpe.md:464 | anulm/bpe.py | 221–244 |  | reattributed | quoted code is in anulm/bpe.py, not None |  |
| chapters/09-byte-level-bpe.md:465 | anulm/bpe.py | 240–243 |  | reattributed | quoted code is in anulm/bpe.py, not None |  |
| chapters/09-byte-level-bpe.md:467 | anulm/bpe.py | 297–345 |  | reattributed | 'cmd_encode' is in anulm/bpe.py, not None |  |
| chapters/09-byte-level-bpe.md:541 | anulm/bpe.py | 120–121 |  | override | decided in repin-overrides.tsv |  |
| chapters/09-byte-level-bpe.md:603 | anulm/bpe.py | 64–83 | 64–83 | sweep | 64–83, 'BP |  |
| chapters/09-byte-level-bpe.md:605 | anulm/bpe.py | 297–345 | 297–345 | sweep | 297–345, '_s |  |
| chapters/09-byte-level-bpe.md:606 | tests/test_model.py | 837–852 | 970–985 | same | def bpe_documents_get_eos_and_the_vocab_is_exact(): | def bpe_documents_get_eos_and_the_vocab_is_exact(): |
| chapters/09-byte-level-bpe.md:607 | tests/test_model.py | 1352–1375 | 1517–1540 | same | def hf_tokenizer_conversion_is_exact(): | def hf_tokenizer_conversion_is_exact(): |
| chapters/09-byte-level-bpe.md:608 | tests/test_model.py | 1379 | 1544 | same | def bpe_files_load_under_both_format_ids(): | def bpe_files_load_under_both_format_ids(): |
| chapters/10-embeddings-and-the-output-table.md:101 | anulm/model.py | 118 |  | reattributed | quoted code is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/10-embeddings-and-the-output-table.md:226 | anulm/training/train.py | 555 |  | reattributed | quoted code is in anulm/training/train.py, not tools/windows/_v2_run.c |  |
| chapters/10-embeddings-and-the-output-table.md:287 | anulm/model.py | 390 |  | reattributed | 'merge_tree_matrix' is in anulm/model.py, not tools/windows/_v2_run.cm |  |
| chapters/10-embeddings-and-the-output-table.md:290 | anulm/model.py | 997–1013 |  | reattributed | 'bake_merge_tree' is in anulm/model.py, not tools/windows/_v2_run.cmd |  |
| chapters/10-embeddings-and-the-output-table.md:327 | anulm/model.py | 397 |  | override | decided in repin-overrides.tsv |  |
| chapters/10-embeddings-and-the-output-table.md:343 | tests/test_model.py | 1646–1653 | 1819–1826 | same | merges = [[97, 98], [256, 99], [257, 257], [100, 101]]      # ab, abc, | merges = [[97, 98], [256, 99], [257, 257], [100, 101]]      # ab, abc, |
| chapters/10-embeddings-and-the-output-table.md:400 | tests/test_model.py | 1676 | 1849 | def | 'merge_tree_eval_cache_sees_optimizer_updates' | def merge_tree_eval_cache_sees_optimizer_updates(): |
| chapters/10-embeddings-and-the-output-table.md:511 | tests/test_model.py | 349–359 | 428–446 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/10-embeddings-and-the-output-table.md:512 | tests/test_model.py | 1640 | 1813 | same | def merge_tree_composes_trains_and_bakes_losslessly(): | def merge_tree_composes_trains_and_bakes_losslessly(): |
| chapters/10-embeddings-and-the-output-table.md:513 | tests/test_model.py | 1676 | 1849 | same | def merge_tree_eval_cache_sees_optimizer_updates(): | def merge_tree_eval_cache_sees_optimizer_updates(): |
| chapters/11-attention.md:412 | tests/test_model.py | 376–380 |  | reattributed | quoted code is in tests/test_model.py, not anulm/model.py |  |
| chapters/11-attention.md:412 | tests/test_model.py | 376–380 | 473–477 | same | m2.load_state_dict(ck["model"]) | m2.load_state_dict(ck["model"]) |
| chapters/11-attention.md:416 | tests/test_model.py | 653 |  | reattributed | 'sliding_window_limits_reach' is in tests/test_model.py, not anulm/mod |  |
| chapters/11-attention.md:416 | tests/test_model.py | 653 | 750 | same | def sliding_window_limits_reach(): | def sliding_window_limits_reach(): |
| chapters/11-attention.md:422 | anulm/model.py | 446–448 |  | override | decided in repin-overrides.tsv |  |
| chapters/11-attention.md:459 | anulm/model.py | 465–467 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not tests/test_model.py |  |
| chapters/11-attention.md:482 | anulm/model.py | 231 |  | override | decided in repin-overrides.tsv |  |
| chapters/11-attention.md:531 | anulm/model.py | 465 |  | reattributed | quoted code is in anulm/model.py, not book/code-examples/attention_by_ |  |
| chapters/11-attention.md:669 | anulm/model.py | 482 |  | override | decided in repin-overrides.tsv |  |
| chapters/11-attention.md:670 | tests/test_model.py | 561 | 658 | def | 'incremental_forward_matches_full_forward' | def incremental_forward_matches_full_forward(): |
| chapters/11-attention.md:670 | tests/test_model.py | 653 | 750 | same | def sliding_window_limits_reach(): | def sliding_window_limits_reach(): |
| chapters/11-attention.md:751 | tests/test_model.py | 561 | 658 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/11-attention.md:751 | tests/test_model.py | 653–675 | 750–772 | same | def sliding_window_limits_reach(): | def sliding_window_limits_reach(): |
| chapters/12-causal-masking.md:320 | tests/test_model.py | 561–578 | 658–675 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/12-causal-masking.md:346 | tests/test_model.py | 653–675 | 750–772 | same | def sliding_window_limits_reach(): | def sliding_window_limits_reach(): |
| chapters/12-causal-masking.md:378 | tests/test_model.py | 679–686 | ? | UNMAPPED | def sliding_window_agrees_with_cache(): |  |
| chapters/12-causal-masking.md:378 | tests/test_model.py | 679 | ? | EXCERPT | header range unresolved: sliding_window_agrees_with_cache |  |
| chapters/12-causal-masking.md:158 | anulm/model.py | 575 |  | override | decided in repin-overrides.tsv |  |
| chapters/12-causal-masking.md:517 | anulm/model.py | 812–814 | 812–814 | sweep | 812–814, 'An |  |
| chapters/12-causal-masking.md:519 | tests/test_model.py | 561 | 658 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/12-causal-masking.md:519 | tests/test_model.py | 653 | 750 | same | def sliding_window_limits_reach(): | def sliding_window_limits_reach(): |
| chapters/12-causal-masking.md:519 | tests/test_model.py | 679 | 776 | same | def sliding_window_agrees_with_cache(): | def sliding_window_agrees_with_cache(): |
| chapters/13-multi-head-and-gqa.md:139 | tests/test_model.py | 582–590 | 679–687 | same | def cache_holds_what_it_should(): | def cache_holds_what_it_should(): |
| chapters/13-multi-head-and-gqa.md:51 | anulm/model.py | 430–449 |  | reattributed | 'GQAttention.__init__' is in anulm/model.py, not None |  |
| chapters/13-multi-head-and-gqa.md:95 | anulm/model.py | 451–468 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not None |  |
| chapters/13-multi-head-and-gqa.md:210 | anulm/training/train.py | 637–643 |  | override | decided in repin-overrides.tsv |  |
| chapters/13-multi-head-and-gqa.md:212 | tests/test_model.py | 1602–1623 | ? | UNMAPPED | def native_gqa_matches_repeat_interleave(): |  |
| chapters/13-multi-head-and-gqa.md:362 | tests/test_model.py | 582 | 679 | same | def cache_holds_what_it_should(): | def cache_holds_what_it_should(): |
| chapters/13-multi-head-and-gqa.md:362 | tests/test_model.py | 1602 | 1767 | same | def native_gqa_matches_repeat_interleave(): | def native_gqa_matches_repeat_interleave(): |
| chapters/14-rotary-positions.md:158 | anulm/model.py | 837–839 |  | reattributed | 'AnuLM.__init__' is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/14-rotary-positions.md:159 | anulm/model.py | 899 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/14-rotary-positions.md:236 | tests/test_model.py | 636–645 | 733–742 | same | def rotary_cache_tracks_dtype_and_positions(): | def rotary_cache_tracks_dtype_and_positions(): |
| chapters/14-rotary-positions.md:477 | tests/test_model.py | 636 | 733 | same | def rotary_cache_tracks_dtype_and_positions(): | def rotary_cache_tracks_dtype_and_positions(): |
| chapters/15-normalisation-and-the-residual-stream.md:322 | tests/test_model.py | 349–359 | 428–446 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/15-normalisation-and-the-residual-stream.md:322 | tests/test_model.py | 358 | 445 | edited-quote | assert abs(loss.item() - expected) < 0.35, \ | assert abs(loss.item() - expected) < 0.1, \ |
| chapters/15-normalisation-and-the-residual-stream.md:423 | anulm/model.py | 816–824 | 816–824 | sweep | 816–824, 'An |  |
| chapters/15-normalisation-and-the-residual-stream.md:425 | anulm/multimodal/speech_encoder.py | 79–82 | 79–82 | sweep | - 'anulm/multimodal/speech_encoder.py' 79–82 and |  |
| chapters/15-normalisation-and-the-residual-stream.md:425 | anulm/multimodal/vision_encoder.py | 124 | 124 | sweep | –82 and 'anulm/multimodal/vision_encoder.py' 124 (Lay |  |
| chapters/15-normalisation-and-the-residual-stream.md:426 | tests/test_model.py | 349 | 428 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/16-swiglu.md:366 | tests/test_model.py | 1285 | 1450 | same | def transformers_wrapper_matches_the_native_model(): | def transformers_wrapper_matches_the_native_model(): |
| chapters/16-swiglu.md:369 | tests/test_model.py | 336 | 415 | def | 'all_parameters_receive_gradients' | def all_parameters_receive_gradients(): |
| chapters/16-swiglu.md:427 | tests/test_model.py | 336 | 415 | same | def all_parameters_receive_gradients(): | def all_parameters_receive_gradients(): |
| chapters/16-swiglu.md:428 | tests/test_model.py | 1285 | 1450 | same | def transformers_wrapper_matches_the_native_model(): | def transformers_wrapper_matches_the_native_model(): |
| chapters/17-router.md:339 | tests/test_model.py | 423 | 520 | same | def config_rejects_unreachable_topk(): | def config_rejects_unreachable_topk(): |
| chapters/17-router.md:340 | (removed reference file) | 629–638 |  | reattributed | quoted code is in (removed reference file), not tests |  |
| chapters/17-router.md:480 | anulm/model.py | 629–638 | 629–638 | sweep | ard' 611–622, 'Router._gate' 624–645 (groups 629–638). |  |
| chapters/17-router.md:483 | tests/test_model.py | 423 | 520 | same | def config_rejects_unreachable_topk(): | def config_rejects_unreachable_topk(): |
| chapters/17-router.md:485 | (removed reference file) | 276–277 | 276–277 | sweep | '(removed reference file)' 276–277, |  |
| chapters/17-router.md:486 | (removed reference file) | 305–313 | 305–313 | sweep | 305–313: the |  |
| chapters/18-dispatch-and-balance.md:217 | tests/test_model.py | 694–711 | 827–844 | same | def grouped_moe_matches_sparse_under_bf16(): | def grouped_moe_matches_sparse_under_bf16(): |
| chapters/18-dispatch-and-balance.md:461 | tests/test_model.py | 774–790 | 907–923 | same | def seq_balance_loss_matches_the_formula(): | def seq_balance_loss_matches_the_formula(): |
| chapters/18-dispatch-and-balance.md:119 | tests/test_model.py | 506 | 603 | same | def sparse_moe_breaks_the_graph_as_documented(): | def sparse_moe_breaks_the_graph_as_documented(): |
| chapters/18-dispatch-and-balance.md:214 | tests/test_model.py | 474 | 571 | def | 'dense_and_sparse_moe_agree' | def dense_and_sparse_moe_agree(): |
| chapters/18-dispatch-and-balance.md:268 | anulm/model.py | 753–754 |  | override | decided in repin-overrides.tsv |  |
| chapters/18-dispatch-and-balance.md:350 | tests/test_model.py | 282 | 361 | same | def bias_update_recovers_from_collapse(): | def bias_update_recovers_from_collapse(): |
| chapters/18-dispatch-and-balance.md:485 | tests/test_model.py | 750 | 883 | same | def router_regularisers_are_off_by_default_and_work_when_on(): | def router_regularisers_are_off_by_default_and_work_when_on(): |
| chapters/18-dispatch-and-balance.md:512 | tests/test_model.py | 321 | 400 | def | 'expert_bias_is_not_trainable' | def expert_bias_is_not_trainable(): |
| chapters/18-dispatch-and-balance.md:571 | anulm/model.py | 775–790 | 775–790 | sweep | 775–790; 'An |  |
| chapters/18-dispatch-and-balance.md:574 | tests/test_model.py | 282 | 361 | same | def bias_update_recovers_from_collapse(): | def bias_update_recovers_from_collapse(): |
| chapters/18-dispatch-and-balance.md:574 | tests/test_model.py | 321 | 400 | same | def expert_bias_is_not_trainable(): | def expert_bias_is_not_trainable(): |
| chapters/18-dispatch-and-balance.md:575 | tests/test_model.py | 474 | 571 | same | def dense_and_sparse_moe_agree(): | def dense_and_sparse_moe_agree(): |
| chapters/18-dispatch-and-balance.md:576 | tests/test_model.py | 506 | 603 | same | def sparse_moe_breaks_the_graph_as_documented(): | def sparse_moe_breaks_the_graph_as_documented(): |
| chapters/18-dispatch-and-balance.md:576 | tests/test_model.py | 506 | 603 | sweep | 506, 'gr |  |
| chapters/18-dispatch-and-balance.md:576 | tests/test_model.py | 694–711 | 827–844 | same | def grouped_moe_matches_sparse_under_bf16(): | def grouped_moe_matches_sparse_under_bf16(): |
| chapters/18-dispatch-and-balance.md:577 | tests/test_model.py | 750 | 883 | same | def router_regularisers_are_off_by_default_and_work_when_on(): | def router_regularisers_are_off_by_default_and_work_when_on(): |
| chapters/18-dispatch-and-balance.md:578 | tests/test_model.py | 774–790 | 907–923 | same | def seq_balance_loss_matches_the_formula(): | def seq_balance_loss_matches_the_formula(): |
| chapters/19-latent-attention.md:268 | tests/test_model.py | 432–444 | 529–541 | same | def mla_shares_one_rope_key_across_heads(): | def mla_shares_one_rope_key_across_heads(): |
| chapters/19-latent-attention.md:65 | anulm/model.py | 550–551 |  | override | decided in repin-overrides.tsv |  |
| chapters/19-latent-attention.md:419 | anulm/model.py | 837 |  | reattributed | quoted code is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/19-latent-attention.md:477 | anulm/model.py | 516–533 | 516–533 | sweep | 516–533, 'ML |  |
| chapters/19-latent-attention.md:478 | anulm/model.py | 837 |  | override | decided in repin-overrides.tsv |  |
| chapters/19-latent-attention.md:479 | tests/test_model.py | 432–444 | 529–541 | same | def mla_shares_one_rope_key_across_heads(): | def mla_shares_one_rope_key_across_heads(): |
| chapters/19-latent-attention.md:483 | (removed reference file) | 486–499 | 486–499 | sweep | 486–499 (opt |  |
| chapters/20-the-whole-model.md:213 | anulm/model.py | 242–253 |  | reattributed | 'SwiGLU' is in anulm/model.py, not book/code-examples/trace_forward.py |  |
| chapters/20-the-whole-model.md:252 | tests/test_model.py | 474 | 571 | def | 'dense_and_sparse_moe_agree' | def dense_and_sparse_moe_agree(): |
| chapters/20-the-whole-model.md:253 | tests/test_model.py | 694 | 827 | def | 'grouped_moe_matches_sparse_under_bf16' | def grouped_moe_matches_sparse_under_bf16(): |
| chapters/20-the-whole-model.md:319 | tests/test_model.py | 349 | 428 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/20-the-whole-model.md:333 | anulm/model.py | 1152–1169 |  | reattributed | 'AnuLM.generate' is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/20-the-whole-model.md:454 | tests/test_model.py | 410 | 507 | same | def param_count_matches_analysis(): | def param_count_matches_analysis(): |
| chapters/20-the-whole-model.md:460 | tests/test_model.py | 336 | 415 | def | 'all_parameters_receive_gradients' | def all_parameters_receive_gradients(): |
| chapters/20-the-whole-model.md:461 | tests/test_model.py | 349 | 428 | def | 'initial_loss_is_ln_vocab' | def initial_loss_is_ln_vocab(): |
| chapters/20-the-whole-model.md:563 | tests/test_model.py | 336 | 415 | same | def all_parameters_receive_gradients(): | def all_parameters_receive_gradients(): |
| chapters/20-the-whole-model.md:563 | tests/test_model.py | 349 | 428 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/20-the-whole-model.md:564 | tests/test_model.py | 410 | 507 | same | def param_count_matches_analysis(): | def param_count_matches_analysis(): |
| chapters/20-the-whole-model.md:565 | tests/test_model.py | 474 | 571 | same | def dense_and_sparse_moe_agree(): | def dense_and_sparse_moe_agree(): |
| chapters/20-the-whole-model.md:565 | tests/test_model.py | 694 | 827 | same | def grouped_moe_matches_sparse_under_bf16(): | def grouped_moe_matches_sparse_under_bf16(): |
| chapters/21-data.md:41 | experiments/build_v2.py | 74–81 | 76–83 | same | SOURCES = { | SOURCES = { |
| chapters/21-data.md:63 | experiments/build_v2.py | 101–112 | 103–114 | same | def flush(batch): | def flush(batch): |
| chapters/21-data.md:105 | experiments/build_v2.py | 134–138 | 136–140 | same | meta = {"format": "uint16", "vocab_size": 32768, "eos_id": EOS, | meta = {"format": "uint16", "vocab_size": 32768, "eos_id": EOS, |
| chapters/21-data.md:59 | experiments/build_v2.py | 119 |  | override | decided in repin-overrides.tsv |  |
| chapters/21-data.md:59 | experiments/build_v2.py | 119 | 121 | same | if n_tr >= budget: | if n_tr >= budget: |
| chapters/21-data.md:60 | experiments/build_v2.py | 46 | 48 | same | def parquet_docs(files, column="text", min_chars=200): | def parquet_docs(files, column="text", min_chars=200): |
| chapters/21-data.md:60 | experiments/build_v2.py | 56 | 58 | same | def text_docs(files, sep="\n\n\n", min_chars=200): | def text_docs(files, sep="\n\n\n", min_chars=200): |
| chapters/21-data.md:81 | experiments/build_v2.py | 43 | 45 | same | EOS = 32767 | EOS = 32767 |
| chapters/21-data.md:82 | experiments/build_v2.py | 42 |  | reattributed | quoted code is in experiments/build_v2.py, not None |  |
| chapters/21-data.md:82 | experiments/build_v2.py | 42 | 44 | same | TOKENIZER = ROOT / "release" / "AnuLM-Base-400M" / "tokenizer.json" | TOKENIZER = ROOT / "release" / "AnuLM-Base-400M" / "tokenizer.json" |
| chapters/21-data.md:353 | anulm/training/train.py | 590 |  | reattributed | quoted code is in anulm/training/train.py, not book/code-examples/ch21 |  |
| chapters/21-data.md:354 | anulm/training/train.py | 692–693 |  | override | decided in repin-overrides.tsv |  |
| chapters/21-data.md:384 | anulm/model.py | 365 |  | reattributed | '_attn_mask' is defined there |  |
| chapters/21-data.md:471 | tests/test_model.py | 1749 |  | reattributed | 'memmap_batches_match_in_memory_batches' is defined there |  |
| chapters/21-data.md:471 | tests/test_model.py | 1749 | 1922 | def | 'memmap_batches_match_in_memory_batches' | def memmap_batches_match_in_memory_batches(): |
| chapters/21-data.md:473 | tests/test_model.py | 798 |  | reattributed | 'window_sampler_covers_every_window_once_per_epoch' is defined there |  |
| chapters/21-data.md:473 | tests/test_model.py | 798 | 931 | def | 'window_sampler_covers_every_window_once_per_epoch' | def window_sampler_covers_every_window_once_per_epoch(): |
| chapters/21-data.md:474 | tests/test_model.py | 823 |  | reattributed | 'window_sampler_resumes_exactly' is defined there |  |
| chapters/21-data.md:474 | tests/test_model.py | 823 | 956 | def | 'window_sampler_resumes_exactly' | def window_sampler_resumes_exactly(): |
| chapters/21-data.md:475 | tests/test_model.py | 1771 |  | reattributed | 'interleaved_split_samples_the_whole_corpus' is defined there |  |
| chapters/21-data.md:475 | tests/test_model.py | 1771 | 1944 | def | 'interleaved_split_samples_the_whole_corpus' | def interleaved_split_samples_the_whole_corpus(): |
| chapters/21-data.md:525 | experiments/build_v2.py | 42 | 44 | same | TOKENIZER = ROOT / "release" / "AnuLM-Base-400M" / "tokenizer.json" | TOKENIZER = ROOT / "release" / "AnuLM-Base-400M" / "tokenizer.json" |
| chapters/21-data.md:525 | experiments/build_v2.py | 43 | 45 | same | EOS = 32767 | EOS = 32767 |
| chapters/21-data.md:525 | experiments/build_v2.py | 46 | 48 | same | def parquet_docs(files, column="text", min_chars=200): | def parquet_docs(files, column="text", min_chars=200): |
| chapters/21-data.md:525 | experiments/build_v2.py | 56 | 58 | same | def text_docs(files, sep="\n\n\n", min_chars=200): | def text_docs(files, sep="\n\n\n", min_chars=200): |
| chapters/21-data.md:526 | experiments/build_v2.py | 74–81 | 76–83 | same | SOURCES = { | SOURCES = { |
| chapters/21-data.md:526 | experiments/build_v2.py | 84–140 | 86–142 | same | def main(): | def main(): |
| chapters/21-data.md:526 | anulm/training/finetune.py | 101–112 |  | reattributed | 'flush' is in anulm/training/finetune.py, not experiments/build_v2.py |  |
| chapters/21-data.md:526 | experiments/build_v2.py | 119 | 121 | same | if n_tr >= budget: | if n_tr >= budget: |
| chapters/21-data.md:526 | experiments/build_v2.py | 119 | 121 | sweep | 'main' 84–140 ('flush' 101–112, budget check 119, 'me |  |
| chapters/21-data.md:530 | anulm/training/train.py | 351–372 | 351–372 | sweep | 351–372, 'ma |  |
| chapters/21-data.md:532 | tests/test_model.py | 798 | 931 | same | def window_sampler_covers_every_window_once_per_epoch(): | def window_sampler_covers_every_window_once_per_epoch(): |
| chapters/21-data.md:533 | tests/test_model.py | 823 | 956 | same | def window_sampler_resumes_exactly(): | def window_sampler_resumes_exactly(): |
| chapters/21-data.md:534 | tests/test_model.py | 1749 | 1922 | same | def memmap_batches_match_in_memory_batches(): | def memmap_batches_match_in_memory_batches(): |
| chapters/21-data.md:535 | tests/test_model.py | 1771 | 1944 | same | def interleaved_split_samples_the_whole_corpus(): | def interleaved_split_samples_the_whole_corpus(): |
| chapters/22-the-training-loop.md:114 | anulm/training/train.py | 736 |  | override | decided in repin-overrides.tsv |  |
| chapters/22-the-training-loop.md:278 | anulm/training/train.py | 869 |  | override | decided in repin-overrides.tsv |  |
| chapters/22-the-training-loop.md:322 | anulm/training/train.py | 891–905 |  | reattributed | 'main' is in anulm/training/train.py, not book/code-examples/ch22_read |  |
| chapters/22-the-training-loop.md:456 | tests/test_model.py | 448 |  | reattributed | 'gradient_accumulation_equals_large_batch' is defined there |  |
| chapters/22-the-training-loop.md:456 | tests/test_model.py | 448 | 545 | def | 'gradient_accumulation_equals_large_batch' | def gradient_accumulation_equals_large_batch(): |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 646 | 646 | sweep | - 'anulm/training/train.py': 'main' 536–924 (646, 712 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 712–732 | 712–732 | sweep | ulm/training/train.py': 'main' 536–924 (646, 712–732, 736 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 736 | 736 | sweep | ing/train.py': 'main' 536–924 (646, 712–732, 736, 747 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 747 | 747 | sweep | rain.py': 'main' 536–924 (646, 712–732, 736, 747, 749 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 749–772 | 749–772 | sweep | py': 'main' 536–924 (646, 712–732, 736, 747, 749–772, 849 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 849–873 | 849–873 | sweep | n' 536–924 (646, 712–732, 736, 747, 749–772, 849–873, 869 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 869 | 869 | sweep | 4 (646, 712–732, 736, 747, 749–772, 849–873, 869, |  |
| chapters/22-the-training-loop.md:512 | anulm/training/train.py | 874–880 | 874–880 | sweep | 874–880, 882 |  |
| chapters/22-the-training-loop.md:512 | anulm/training/train.py | 882–889 | 882–889 | sweep | 874–880, 882–889, 891 |  |
| chapters/22-the-training-loop.md:512 | anulm/training/train.py | 891–908 | 891–908 | sweep | 874–880, 882–889, 891–908), 'l |  |
| chapters/22-the-training-loop.md:513 | anulm/model.py | 23 | 23 | sweep | el.py': 'from __future__ import annotations' 23, |  |
| chapters/22-the-training-loop.md:515 | tests/test_model.py | 448 | 545 | same | def gradient_accumulation_equals_large_batch(): | def gradient_accumulation_equals_large_batch(): |
| chapters/23-precision-memory-speed.md:421 | tests/test_model.py | 491–503 | 588–600 | same | def dense_moe_compiles_without_graph_breaks(): | def dense_moe_compiles_without_graph_breaks(): |
| chapters/23-precision-memory-speed.md:438 | tests/test_model.py | 506 | 603 | def | 'sparse_moe_breaks_the_graph_as_documented' | def sparse_moe_breaks_the_graph_as_documented(): |
| chapters/23-precision-memory-speed.md:528 | anulm/multimodal/omni_train.py | 194 | 194 | sweep | y' callers: 'anulm/multimodal/omni_train.py' 194, 'an |  |
| chapters/23-precision-memory-speed.md:528 | anulm/tasks/decide.py | 475 | 475 | sweep | /omni_train.py' 194, 'anulm/tasks/decide.py' 475 and |  |
| chapters/23-precision-memory-speed.md:528 | anulm/tasks/decide.py | 553 | 553 | sweep | ain.py' 194, 'anulm/tasks/decide.py' 475 and 553, |  |
| chapters/23-precision-memory-speed.md:529 | anulm/multimodal/speech_encoder.py | 231 | 231 | sweep | 'anulm/multimodal/speech_encoder.py' 231, 'an |  |
| chapters/23-precision-memory-speed.md:529 | anulm/multimodal/vision_encoder.py | 456 | 456 | sweep | y' 231, 'anulm/multimodal/vision_encoder.py' 456. |  |
| chapters/23-precision-memory-speed.md:535 | tests/test_model.py | 491 | 588 | same | def dense_moe_compiles_without_graph_breaks(): | def dense_moe_compiles_without_graph_breaks(): |
| chapters/23-precision-memory-speed.md:536 | tests/test_model.py | 506 | 603 | same | def sparse_moe_breaks_the_graph_as_documented(): | def sparse_moe_breaks_the_graph_as_documented(): |
| chapters/23-precision-memory-speed.md:537 | tests/test_model.py | 1548 | 1713 | same | def chunked_loss_matches_the_full_loss_and_its_gradients(): | def chunked_loss_matches_the_full_loss_and_its_gradients(): |
| chapters/24-measuring-progress.md:57 | tests/test_model.py | 994–998 | 1127–1131 | same | data = torch.arange(100_000) | data = torch.arange(100_000) |
| chapters/24-measuring-progress.md:206 | tests/test_model.py | 1000–1006 | 1133–1139 | same | flat = lambda bs: [int(i) for b in bs for i in b] | flat = lambda bs: [int(i) for b in bs for i in b] |
| chapters/24-measuring-progress.md:450 | anulm/training/train.py | 477–533 | 477–533 | sweep | 477–533, 'ma |  |
| chapters/24-measuring-progress.md:450 | anulm/training/train.py | 906–908 |  | override | decided in repin-overrides.tsv |  |
| chapters/24-measuring-progress.md:450 | anulm/training/train.py | 906–908 | 906–908 | sweep | 477–533, 'main' 782–786 ('eval_spec') and 906–908 (bit |  |
| chapters/24-measuring-progress.md:452 | tests/test_model.py | 980–1026 | 1113–1159 | same | def eval_windows_are_frozen_and_strided_windows_cover_the_split(): | def eval_windows_are_frozen_and_strided_windows_cover_the_split(): |
| chapters/24-measuring-progress.md:452 | tests/test_model.py | 980–1026 | 1113–1159 | sweep | 980–1026, 'go |  |
| chapters/24-measuring-progress.md:452 | tests/test_model.py | 1465–1507 | 1630–1672 | same | def golden_items_are_well_formed_and_the_scorer_runs(): | def golden_items_are_well_formed_and_the_scorer_runs(): |
| chapters/24-measuring-progress.md:454 | anulm/eval/eval_code.py | 220–272 | 221–273 | same | def score(ckpt: str, problems: list[dict], device: str, max_new: int,  | def score(ckpt: str, problems: list[dict], device: str, max_new: int,  |
| chapters/25-checkpoints-and-resuming.md:189 | tests/test_model.py | 823–833 | 956–966 | same | def window_sampler_resumes_exactly(): | def window_sampler_resumes_exactly(): |
| chapters/25-checkpoints-and-resuming.md:172 | anulm/training/train.py | 645 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:173 | anulm/training/train.py | 859 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:207 | tests/test_model.py | 363 | 460 | same | def checkpoint_roundtrip_is_exact(): | def checkpoint_roundtrip_is_exact(): |
| chapters/25-checkpoints-and-resuming.md:349 | anulm/training/train.py | 857–876 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:376 | anulm/training/train.py | 380–391 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:436 | anulm/model.py | 1213 |  | reattributed | quoted code is in anulm/model.py, not model.py |  |
| chapters/25-checkpoints-and-resuming.md:493 | book/code-examples/ch25_resume.py | 85 |  | reattributed | quoted code is in book/code-examples/ch25_resume.py, not tests/test_qu |  |
| chapters/25-checkpoints-and-resuming.md:493 | book/code-examples/ch25_resume.py | 85 | 96 | same | log_c2 = train(C, "--resume") | log_c2 = train(C, "--resume") |
| chapters/25-checkpoints-and-resuming.md:494 | tests/test_quality_loading.py | 124 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:535 | tests/test_model.py | 924 |  | reattributed | 'an_exported_folder_loads_everywhere_a_pt_does' is defined there |  |
| chapters/25-checkpoints-and-resuming.md:535 | tests/test_model.py | 924 | 1057 | def | 'an_exported_folder_loads_everywhere_a_pt_does' | def an_exported_folder_loads_everywhere_a_pt_does(): |
| chapters/25-checkpoints-and-resuming.md:615 | anulm/training/train.py | 152–160 | 152–160 | sweep | 152–160, 'lr |  |
| chapters/25-checkpoints-and-resuming.md:615 | anulm/training/train.py | 849–850 | 849–850 | sweep | 0, 'lr_at' 375–400, 'main' 788–840 (resume), 849–850 ('-- |  |
| chapters/25-checkpoints-and-resuming.md:616 | anulm/training/train.py | 857–876 | 857–876 | sweep | 857–876 (one |  |
| chapters/25-checkpoints-and-resuming.md:616 | anulm/training/train.py | 891–905 | 891–905 | sweep | 857–876 (one step), 891–905 (sav |  |
| chapters/25-checkpoints-and-resuming.md:617 | anulm/model.py | 1213 | 1213 | sweep | - 'anulm/model.py': (pre-rename alias, since dropped) 1213, 'TR |  |
| chapters/25-checkpoints-and-resuming.md:621 | tests/test_model.py | 363–380 | 460–477 | same | def checkpoint_roundtrip_is_exact(): | def checkpoint_roundtrip_is_exact(): |
| chapters/25-checkpoints-and-resuming.md:622 | tests/test_model.py | 823–833 | 956–966 | same | def window_sampler_resumes_exactly(): | def window_sampler_resumes_exactly(): |
| chapters/25-checkpoints-and-resuming.md:623 | tests/test_model.py | 321–328 | 400–407 | same | def expert_bias_is_not_trainable(): | def expert_bias_is_not_trainable(): |
| chapters/25-checkpoints-and-resuming.md:623 | tests/test_model.py | 321–328 | 400–407 | sweep | 321–328, 'an |  |
| chapters/25-checkpoints-and-resuming.md:623 | tests/test_model.py | 924–976 | 1057–1109 | same | def an_exported_folder_loads_everywhere_a_pt_does(): | def an_exported_folder_loads_everywhere_a_pt_does(): |
| chapters/26-debugging-silent-failures.md:148 | tests/test_model.py | 473–487 | 570–584 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:178 | tests/test_model.py | 1770–1783 | 1943–1956 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:359 | tests/test_model.py | 1688–1698 | 1861–1871 | same | x = torch.randint(0, 259, (2, 8)) | x = torch.randint(0, 259, (2, 8)) |
| chapters/26-debugging-silent-failures.md:486 | tests/test_model.py | 2250–2271 | 2434–2455 | same | def main(): | def main(): |
| chapters/26-debugging-silent-failures.md:486 | tests/test_model.py | 2266 | 2450 | edited-quote | # ("76 passed, 0 failed, 76 total"), so it keeps that exact shape. | # ("80 passed, 0 failed, 80 total"), so it keeps that exact shape. |
| chapters/26-debugging-silent-failures.md:532 | tests/test_model.py | 694–699 | 827–832 | same | def grouped_moe_matches_sparse_under_bf16(): | def grouped_moe_matches_sparse_under_bf16(): |
| chapters/26-debugging-silent-failures.md:551 | tests/test_model.py | 1270–1273 | 1435–1438 | same | assert os.path.exists(os.path.join(outside, "spawned.txt")), \ | assert os.path.exists(os.path.join(outside, "spawned.txt")), \ |
| chapters/26-debugging-silent-failures.md:143 | tests/test_model.py | 653 | 750 | def | 'sliding_window_limits_reach' | def sliding_window_limits_reach(): |
| chapters/26-debugging-silent-failures.md:171 | tests/test_model.py | 604 | 701 | same | def cached_generation_matches_uncached(): | def cached_generation_matches_uncached(): |
| chapters/26-debugging-silent-failures.md:172 | tests/test_model.py | 1548 | 1713 | same | def chunked_loss_matches_the_full_loss_and_its_gradients(): | def chunked_loss_matches_the_full_loss_and_its_gradients(): |
| chapters/26-debugging-silent-failures.md:201 | tests/test_model.py | 798 | 931 | same | def window_sampler_covers_every_window_once_per_epoch(): | def window_sampler_covers_every_window_once_per_epoch(): |
| chapters/26-debugging-silent-failures.md:203 | tests/test_model.py | 1511 | 1676 | same | def qa_packing_masks_prompts_and_keeps_examples_whole(): | def qa_packing_masks_prompts_and_keeps_examples_whole(): |
| chapters/26-debugging-silent-failures.md:449 | tests/test_model.py | 980 | 1113 | def | 'eval_windows_are_frozen_and_strided_windows_cover_the_split' | def eval_windows_are_frozen_and_strided_windows_cover_the_split(): |
| chapters/26-debugging-silent-failures.md:459 | tests/test_model.py | 1285 | 1450 | def | 'transformers_wrapper_matches_the_native_model' | def transformers_wrapper_matches_the_native_model(): |
| chapters/26-debugging-silent-failures.md:460 | tests/test_model.py | 1119 | 1284 | def | 'an_exported_qa_checkpoint_still_offers_its_question_mode' | def an_exported_qa_checkpoint_still_offers_its_question_mode(): |
| chapters/26-debugging-silent-failures.md:461 | tests/test_model.py | 1177 | 1342 | def | 'an_export_keeps_what_makes_a_checkpoint_more_than_a_language_model' | def an_export_keeps_what_makes_a_checkpoint_more_than_a_language_model |
| chapters/26-debugging-silent-failures.md:462 | tests/test_model.py | 924 | 1057 | def | 'an_exported_folder_loads_everywhere_a_pt_does' | def an_exported_folder_loads_everywhere_a_pt_does(): |
| chapters/26-debugging-silent-failures.md:463 | tests/test_model.py | 1225 | 1390 | def | 'eval_code_sandbox_contains_generated_code' | def eval_code_sandbox_contains_generated_code(): |
| chapters/26-debugging-silent-failures.md:464 | tests/test_model.py | 1465 |  | reattributed | 'golden_items_are_well_formed_and_the_scorer_runs' is defined there |  |
| chapters/26-debugging-silent-failures.md:464 | tests/test_model.py | 1465 | 1630 | def | 'golden_items_are_well_formed_and_the_scorer_runs' | def golden_items_are_well_formed_and_the_scorer_runs(): |
| chapters/26-debugging-silent-failures.md:465 | tests/test_model.py | 1379 |  | reattributed | 'bpe_files_load_under_both_format_ids' is defined there |  |
| chapters/26-debugging-silent-failures.md:465 | tests/test_model.py | 1379 | 1544 | def | 'bpe_files_load_under_both_format_ids' | def bpe_files_load_under_both_format_ids(): |
| chapters/26-debugging-silent-failures.md:466 | tests/test_model.py | 561 |  | reattributed | 'incremental_forward_matches_full_forward' is defined there |  |
| chapters/26-debugging-silent-failures.md:466 | tests/test_model.py | 561 | 658 | def | 'incremental_forward_matches_full_forward' | def incremental_forward_matches_full_forward(): |
| chapters/26-debugging-silent-failures.md:513 | tests/test_model.py | 34 |  | override | decided in repin-overrides.tsv |  |
| chapters/26-debugging-silent-failures.md:580 | tests/test_model.py | 448 | 545 | same | def gradient_accumulation_equals_large_batch(): | def gradient_accumulation_equals_large_batch(): |
| chapters/26-debugging-silent-failures.md:624 | tests/test_model.py | 1–15 | 1–15 | sweep | - 'tests/test_model.py': module docstring 1–15; 'te |  |
| chapters/26-debugging-silent-failures.md:625 | tests/test_model.py | 693–711 | 826–844 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:625 | tests/test_model.py | 698–699 | 831–832 | same | if not hasattr(F, "grouped_mm"): | if not hasattr(F, "grouped_mm"): |
| chapters/26-debugging-silent-failures.md:625 | tests/test_model.py | 698–699 | 831–832 | sweep | matches_sparse_under_bf16' 693–711 (the skip 698–699); |  |
| chapters/26-debugging-silent-failures.md:626 | tests/test_model.py | 1224–1281 | 1389–1446 | same | @test | @test |
| chapters/27-decoding-and-the-kv-cache.md:463 | tests/test_model.py | 561–578 | 658–675 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/27-decoding-and-the-kv-cache.md:91 | anulm/model.py | 1144–1151 |  | override | decided in repin-overrides.tsv |  |
| chapters/27-decoding-and-the-kv-cache.md:208 | tests/test_model.py | 636 |  | reattributed | quoted code is in tests/test_model.py, not anulm/model.py |  |
| chapters/27-decoding-and-the-kv-cache.md:208 | tests/test_model.py | 636 | 733 | same | def rotary_cache_tracks_dtype_and_positions(): | def rotary_cache_tracks_dtype_and_positions(): |
| chapters/27-decoding-and-the-kv-cache.md:489 | tests/test_model.py | 604 | 701 | same | def cached_generation_matches_uncached(): | def cached_generation_matches_uncached(): |
| chapters/27-decoding-and-the-kv-cache.md:494 | tests/test_model.py | 582 |  | reattributed | 'cache_holds_what_it_should' is defined there |  |
| chapters/27-decoding-and-the-kv-cache.md:494 | tests/test_model.py | 582 | 679 | def | 'cache_holds_what_it_should' | def cache_holds_what_it_should(): |
| chapters/27-decoding-and-the-kv-cache.md:561 | tests/test_model.py | 561 | 658 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/27-decoding-and-the-kv-cache.md:562 | tests/test_model.py | 582 | 679 | same | def cache_holds_what_it_should(): | def cache_holds_what_it_should(): |
| chapters/27-decoding-and-the-kv-cache.md:562 | tests/test_model.py | 604 | 701 | same | def cached_generation_matches_uncached(): | def cached_generation_matches_uncached(): |
| chapters/27-decoding-and-the-kv-cache.md:563 | tests/test_model.py | 636 | 733 | same | def rotary_cache_tracks_dtype_and_positions(): | def rotary_cache_tracks_dtype_and_positions(): |
| chapters/28-choosing-the-next-token.md:90 | anulm/model.py | 1169 |  | override | decided in repin-overrides.tsv |  |
| chapters/28-choosing-the-next-token.md:99 | anulm/model.py | 1161-1162 |  | override | decided in repin-overrides.tsv |  |
| chapters/28-choosing-the-next-token.md:105 | anulm/model.py | 1157-1168 |  | override | decided in repin-overrides.tsv |  |
| chapters/28-choosing-the-next-token.md:139 | tests/test_model.py | 398 |  | reattributed | 'generation_is_deterministic_at_top_k_1' is defined there |  |
| chapters/28-choosing-the-next-token.md:139 | tests/test_model.py | 398 | 495 | def | 'generation_is_deterministic_at_top_k_1' | def generation_is_deterministic_at_top_k_1(): |
| chapters/28-choosing-the-next-token.md:155 | tests/test_release_fixes.py | 112 |  | reattributed | 'test_temperature_zero_reaches_the_model_unclamped' is in tests/test_r |  |
| chapters/28-choosing-the-next-token.md:274 | tests/test_model.py | 620 |  | reattributed | 'generation_stops_at_eos' is defined there |  |
| chapters/28-choosing-the-next-token.md:274 | tests/test_model.py | 620 | 717 | def | 'generation_stops_at_eos' | def generation_stops_at_eos(): |
| chapters/28-choosing-the-next-token.md:487 | anulm/model.py | 1037–1054 | 1037–1054 | sweep | AnuLM.decode_from_embeds' 1015–1062 (excerpt 1037–1054), |  |
| chapters/28-choosing-the-next-token.md:490 | anulm/inference/api.py | 611–613 | 611–613 | sweep | 611–613, com |  |
| chapters/28-choosing-the-next-token.md:490 | anulm/inference/api.py | 649–650 | 649–650 | sweep | 611–613, completion defaults 649–650. |  |
| chapters/28-choosing-the-next-token.md:491 | anulm/inference/serve.py | 158–160 | 158–160 | sweep | ulm/inference/serve.py': the 'generate' call 158–160 (no |  |
| chapters/28-choosing-the-next-token.md:491 | course/checks/ch08.py | 224–229 |  | reattributed | 'generate' is in course/checks/ch08.py, not anulm/inference/serve.py |  |
| chapters/28-choosing-the-next-token.md:491 | anulm/inference/serve.py | 224–229 | 224–229 | sweep | rate' call 158–160 (no clamp), HTTP defaults 224–229. 'an |  |
| chapters/28-choosing-the-next-token.md:491 | anulm/inference/sample.py | 34–35 | 34–35 | sweep | faults 224–229. 'anulm/inference/sample.py': 34–35. |  |
| chapters/28-choosing-the-next-token.md:492 | tests/test_model.py | 398 | 495 | same | def generation_is_deterministic_at_top_k_1(): | def generation_is_deterministic_at_top_k_1(): |
| chapters/28-choosing-the-next-token.md:493 | tests/test_model.py | 620 | 717 | same | def generation_stops_at_eos(): | def generation_stops_at_eos(): |
| chapters/29-long-context.md:183 | anulm/model.py | 288–292 |  | override | decided in repin-overrides.tsv |  |
| chapters/29-long-context.md:204 | tests/test_model.py | 144 |  | reattributed | quoted code is in tests/test_model.py, not docs/RESULTS.md |  |
| chapters/29-long-context.md:416 | anulm/model.py | 109–110 | 109–110 | sweep | - 'anulm/model.py': configuration 109–110; '_y |  |
| chapters/30-instruction-tuning.md:500 | tests/test_model.py | 1525–1544 | 1690–1709 | same | for it, (p, a) in zip(items, pairs): | for it, (p, a) in zip(items, pairs): |
| chapters/30-instruction-tuning.md:498 | tests/test_model.py | 1511 |  | reattributed | 'qa_packing_masks_prompts_and_keeps_examples_whole' is defined there |  |
| chapters/30-instruction-tuning.md:498 | tests/test_model.py | 1511 | 1676 | def | 'qa_packing_masks_prompts_and_keeps_examples_whole' | def qa_packing_masks_prompts_and_keeps_examples_whole(): |
| chapters/30-instruction-tuning.md:588 | anulm/training/finetune.py | 307–320 | 307–320 | sweep | 'main' 222–423 (the held-out guard 307–320, the |  |
| chapters/30-instruction-tuning.md:588 | anulm/training/finetune.py | 360–375 | 360–375 | sweep | 22–423 (the held-out guard 307–320, the step 360–375, 'qa |  |
| chapters/30-instruction-tuning.md:589 | anulm/data/make_chat.py | 176 | 176 | sweep | chat_rows' 119–145, 'main' 148–184 (the call 176). |  |
| chapters/30-instruction-tuning.md:590 | anulm/multimodal/omni_eval.py | 192–195 |  | override | decided in repin-overrides.tsv |  |
| chapters/30-instruction-tuning.md:590 | anulm/multimodal/omni_eval.py | 192–195 | 192–195 | sweep | omni_eval.py': 'chat_loss' 115–131, its call 192–195; |  |
| chapters/30-instruction-tuning.md:593 | anulm/model.py | 926–927 | 926–927 | sweep | 922–928 ('ignore_index=-1'; the regulariser 926–927). |  |
| chapters/30-instruction-tuning.md:594 | tests/test_model.py | 1511 | 1676 | same | def qa_packing_masks_prompts_and_keeps_examples_whole(): | def qa_packing_masks_prompts_and_keeps_examples_whole(): |
| chapters/31-lora.md:330 | tests/test_model.py | 1056–1070 | 1191–1205 | same | model = AnuLM(cfg).eval() | model = AnuLM(cfg).eval() |
| chapters/31-lora.md:353 | tests/test_model.py | 1111–1115 | ? | UNMAPPED | trained_bias = router[sorted(router)[0]] |  |
| chapters/31-lora.md:353 | tests/test_model.py | 1111 | ? | EXCERPT | header range unresolved: lora_starts_as_the_base_model_and_merges_back |  |
| chapters/31-lora.md:348 | tests/test_model.py | 1030 | 1163 | def | 'lora_starts_as_the_base_model_and_merges_back_exactly' | def lora_starts_as_the_base_model_and_merges_back_exactly(): |
| chapters/31-lora.md:480 | anulm/training/finetune.py | 393–400 | 393–400 | sweep | - 'anulm/training/finetune.py': adapter save 393–400; 'up |  |
| chapters/31-lora.md:480 | anulm/training/finetune.py | 375 | 375 | sweep | er save 393–400; 'update_expert_biases' call 375. |  |
| chapters/31-lora.md:483 | tests/test_model.py | 1030 | 1163 | same | def lora_starts_as_the_base_model_and_merges_back_exactly(): | def lora_starts_as_the_base_model_and_merges_back_exactly(): |
| chapters/32-classification-and-calibration.md:335 | tests/test_model.py | 1712–1725 | 1885–1898 | same | ids, pos = D.encode(tok, "where is my card", opts) | ids, pos = D.encode(tok, "where is my card", opts) |
| chapters/32-classification-and-calibration.md:80 | anulm/tasks/decide.py | 120 |  | override | decided in repin-overrides.tsv |  |
| chapters/32-classification-and-calibration.md:82 | anulm/tasks/decide.py | 596 |  | override | decided in repin-overrides.tsv |  |
| chapters/32-classification-and-calibration.md:193 | anulm/bidirectional.py | 56–105 |  | reattributed | 'adapt' is in anulm/bidirectional.py, not anulm/model.py |  |
| chapters/32-classification-and-calibration.md:225 | anulm/model.py | 697 |  | reattributed | quoted code is in anulm/model.py, not anulm/tasks/decide.py |  |
| chapters/32-classification-and-calibration.md:354 | tests/test_model.py | 1702 | 1875 | def | 'decide_scores_each_option_at_its_last_token_and_calibrates' | def decide_scores_each_option_at_its_last_token_and_calibrates(): |
| chapters/32-classification-and-calibration.md:420 | anulm/tasks/decide.py | 545–616 | 545–616 | sweep | 545–616 (los |  |
| chapters/32-classification-and-calibration.md:420 | anulm/tasks/decide.py | 596 | 596 | sweep | 545–616 (loss 596, tem |  |
| chapters/32-classification-and-calibration.md:420 | anulm/tasks/decide.py | 611 | 611 | sweep | 545–616 (loss 596, temperature 611), 'D |  |
| chapters/32-classification-and-calibration.md:424 | tests/test_model.py | 1702 | 1875 | same | def decide_scores_each_option_at_its_last_token_and_calibrates(): | def decide_scores_each_option_at_its_last_token_and_calibrates(): |
| chapters/33-retrieval-and-readers.md:181 | anulm/inference/rag.py | 213 |  | override | decided in repin-overrides.tsv |  |
| chapters/33-retrieval-and-readers.md:186 | anulm/inference/rag.py | 342 |  | reattributed | 'RagQA' is in anulm/inference/rag.py, not None |  |
| chapters/33-retrieval-and-readers.md:247 | anulm/inference/rag.py | 253–254 |  | override | decided in repin-overrides.tsv |  |
| chapters/33-retrieval-and-readers.md:283 | anulm/inference/rag.py | 354–364 |  | override | decided in repin-overrides.tsv |  |
| chapters/33-retrieval-and-readers.md:535 | anulm/inference/rag.py | 42–43 | 42–43 | sweep | - 'anulm/inference/rag.py': constants 42–43, 'wo |  |
| chapters/33-retrieval-and-readers.md:535 | anulm/inference/rag.py | 91 |  | override | decided in repin-overrides.tsv |  |
| chapters/33-retrieval-and-readers.md:535 | anulm/inference/rag.py | 91 | 91 | sweep | rds' 51, 'term_ids' 57, title + text indexed 91 (in |  |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 136–145 | 136–145 | sweep | 136–145), 'D |  |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 184–193 | 184–193 | sweep | 136–145), 'Dense' 174–230 (device 184–193, 'em |  |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 241 | 241 | sweep | embed' 195–205), 'Retriever' 233–308 (device 241, def |  |
| chapters/33-retrieval-and-readers.md:538 | anulm/inference/rag.py | 256 | 256 | sweep | mode 256, 'se |  |
| chapters/33-retrieval-and-readers.md:540 | anulm/tasks/rc_pointer.py | 185–190 | 185–190 | sweep | 'train' 155–229 (loss 185–190). |  |
| chapters/35-wider-vocabulary.md:81 | anulm/multimodal/audio_codec.py | 121–139 | 125–143 | same | def flatten(codes: list) -> np.ndarray: | def flatten(codes: list) -> np.ndarray: |
| chapters/35-wider-vocabulary.md:103 | anulm/multimodal/audio_codec.py | 142–170 | 146–174 | same | def unflatten(ids: np.ndarray, report: bool = False): | def unflatten(ids: np.ndarray, report: bool = False): |
| chapters/35-wider-vocabulary.md:247 | anulm/model.py | 896 |  | override | decided in repin-overrides.tsv |  |
| chapters/35-wider-vocabulary.md:452 | tests/test_model.py | 1577 | 1742 | def | 'cached_embeds_decoding_matches_prefix_replay' | def cached_embeds_decoding_matches_prefix_replay(): |
| chapters/35-wider-vocabulary.md:512 | anulm/multimodal/audio_codec.py | 104–114 | 104–118 | edited | def audio_chunks(audio: np.ndarray, chunk_samples: int = 30 * ASR_RATE | def audio_chunks(audio: np.ndarray, chunk_samples: int = 30 * ASR_RATE |
| chapters/35-wider-vocabulary.md:512 | anulm/multimodal/audio_codec.py | 121–139 | 125–143 | same | def flatten(codes: list) -> np.ndarray: | def flatten(codes: list) -> np.ndarray: |
| chapters/35-wider-vocabulary.md:513 | anulm/multimodal/audio_codec.py | 142–170 | 146–174 | same | def unflatten(ids: np.ndarray, report: bool = False): | def unflatten(ids: np.ndarray, report: bool = False): |
| chapters/35-wider-vocabulary.md:517 | anulm/multimodal/speech_encoder.py | 287 | 287 | sweep | 287, 'tr |  |
| chapters/35-wider-vocabulary.md:519 | tests/test_model.py | 1577 | 1742 | same | def cached_embeds_decoding_matches_prefix_replay(): | def cached_embeds_decoding_matches_prefix_replay(): |
| chapters/36-frozen-encoders.md:57 | anulm/multimodal/audio_codec.py | 220–255 | 224–259 | same | def __init__(self, size: str = "small", device: str = "auto", compute_ | def __init__(self, size: str = "small", device: str = "auto", compute_ |
| chapters/36-frozen-encoders.md:356 | anulm/multimodal/omni_train.py | 502 | 504 | same | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", |
| chapters/36-frozen-encoders.md:445 | anulm/multimodal/speech_encoder.py | 197–203 |  | reattributed | 'batches' is in anulm/multimodal/speech_encoder.py, not None |  |
| chapters/36-frozen-encoders.md:550 | anulm/multimodal/vision_encoder.py | 485–488 | 485–488 | sweep | (optimiser 485–488, def |  |
| chapters/36-frozen-encoders.md:550 | anulm/multimodal/vision_encoder.py | 653–654 | 653–654 | sweep | (optimiser 485–488, defaults 653–654). |  |
| chapters/36-frozen-encoders.md:552 | anulm/multimodal/speech_encoder.py | 249 | 249 | sweep | ' 171–182, 'batches' 185–213, 'train''s call 249. |  |
| chapters/36-frozen-encoders.md:554 | anulm/multimodal/audio_codec.py | 211–255 | 215–259 | same | class WhisperEncoder: | class WhisperEncoder: |
| chapters/37-one-model-many-tasks.md:355 | anulm/multimodal/omni_train.py | 458–465 | ? | UNMAPPED | if step % args.eval_every == 0 or step == args.steps: |  |
| chapters/37-one-model-many-tasks.md:355 | anulm/multimodal/omni_train.py | 458 | ? | EXCERPT | header range unresolved: train |  |
| chapters/37-one-model-many-tasks.md:499 | experiments/omni_one.py | 375–383 | 384–392 | same | def gates(res) -> dict: | def gates(res) -> dict: |
| chapters/37-one-model-many-tasks.md:642 | experiments/omni_one.py | 460–469 | 471–480 | same | args = ["-m", "anulm.multimodal.omni_train", "--ckpt", "ckpt_chat_mm.p | args = ["-m", "anulm.multimodal.omni_train", "--ckpt", "ckpt_chat_mm.p |
| chapters/37-one-model-many-tasks.md:200 | anulm/multimodal/omni_train.py | 33–37 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:204 | experiments/omni_one.py | 77–87 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:288 | anulm/multimodal/omni_train.py | 505 | 507 | same | p.add_argument("--accum", type=int, default=1, help="batches summed in | p.add_argument("--accum", type=int, default=1, help="batches summed in |
| chapters/37-one-model-many-tasks.md:288 | anulm/multimodal/omni_train.py | 517–518 | 519–520 | same | if args.accum < 1: | if args.accum < 1: |
| chapters/37-one-model-many-tasks.md:291 | anulm/multimodal/omni_train.py | 494–495 | 496–497 | same | p.add_argument("--lr", type=float, default=5e-5) | p.add_argument("--lr", type=float, default=5e-5) |
| chapters/37-one-model-many-tasks.md:338 | anulm/multimodal/omni_train.py | 474–476 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:338 | anulm/multimodal/omni_train.py | 474–476 | 476–478 | same | st = om.state(step, score, losses) | st = om.state(step, score, losses) |
| chapters/37-one-model-many-tasks.md:341 | anulm/multimodal/omni_train.py | 404–414 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:343 | anulm/multimodal/omni_train.py | 506 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:343 | anulm/multimodal/omni_train.py | 506 | 508 | same | p.add_argument("--stop-after", type=int, default=0, | p.add_argument("--stop-after", type=int, default=0, |
| chapters/37-one-model-many-tasks.md:345 | anulm/multimodal/omni_train.py | 471–472 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:345 | anulm/multimodal/omni_train.py | 471–472 | 473–474 | same | if args.stop_after and step >= args.stop_after: | if args.stop_after and step >= args.stop_after: |
| chapters/37-one-model-many-tasks.md:430 | experiments/omni_one.py | 456 | 467 | same | run("3_grow", ["-m", "anulm.multimodal.speech_vocab", "--ckpt", "ckpt_ | run("3_grow", ["-m", "anulm.multimodal.speech_vocab", "--ckpt", "ckpt_ |
| chapters/37-one-model-many-tasks.md:515 | experiments/omni_one.py | 94–97 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:661 | experiments/omni_one.py | 222–238 | ? | UNMAPPED | def prepare_training(args): |  |
| chapters/37-one-model-many-tasks.md:724 | anulm/multimodal/omni_train.py | 85–101 | 85–101 | sweep | 85–101, 'pa |  |
| chapters/37-one-model-many-tasks.md:728 | anulm/multimodal/omni_train.py | 362–479 | 362–481 | same | def train(args) -> None: | def train(args) -> None: |
| chapters/37-one-model-many-tasks.md:728 | anulm/multimodal/omni_train.py | 482–519 | 484–521 | same | def main() -> None: | def main() -> None: |
| chapters/37-one-model-many-tasks.md:731 | experiments/omni_one.py | 222–238 | ? | UNMAPPED | def prepare_training(args): |  |
| chapters/37-one-model-many-tasks.md:731 | experiments/omni_one.py | 375–383 | 384–392 | same | def gates(res) -> dict: | def gates(res) -> dict: |
| chapters/37-one-model-many-tasks.md:732 | experiments/omni_one.py | 460–469 | 471–480 | same | args = ["-m", "anulm.multimodal.omni_train", "--ckpt", "ckpt_chat_mm.p | args = ["-m", "anulm.multimodal.omni_train", "--ckpt", "ckpt_chat_mm.p |
| chapters/37-one-model-many-tasks.md:732 | experiments/omni_one.py | 460–469 | 471–480 | sweep | command 460–469. |  |
| chapters/38-serving.md:109 | anulm/inference/serve.py | 152 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:110 | anulm/inference/serve.py | 146–151 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:235 | anulm/model.py | 1174–1175 |  | reattributed | quoted code is in anulm/model.py, not book/code-examples/ch38_streamin |  |
| chapters/38-serving.md:352 | anulm/inference/api.py | 540–557 |  | reattributed | 'one_shot_reply' is in anulm/inference/api.py, not book/code-examples/ |  |
| chapters/38-serving.md:535 | anulm/inference/api.py | 748–751 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:575 | anulm/inference/api.py | 483 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:665 | anulm/inference/api.py | 567 |  | reattributed | quoted code is in anulm/inference/api.py, not docs/RESULTS.md |  |
| chapters/38-serving.md:730 | anulm/inference/api.py | 195–200 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:730 | anulm/inference/api.py | 195–200 | 195–200 | sweep | y' 173–180, 'exposure_warning' 183–188, caps 195–200, |  |
| chapters/39-exporting.md:292 | tests/test_model.py | 924 | 1057 | def | 'an_exported_folder_loads_everywhere_a_pt_does' | def an_exported_folder_loads_everywhere_a_pt_does(): |
| chapters/39-exporting.md:293 | tests/test_model.py | 970–972 |  | override | decided in repin-overrides.tsv |  |
| chapters/39-exporting.md:293 | tests/test_model.py | 970–972 | 1103–1105 | same | for name in ("eval_bench.py", "eval_context.py", "eval_golden.py", "ev | for name in ("eval_bench.py", "eval_context.py", "eval_golden.py", "ev |
| chapters/39-exporting.md:377 | anulm/hf/export_hf.py | 378–379 |  | reattributed | 'main' is in anulm/hf/export_hf.py, not None |  |
| chapters/39-exporting.md:521 | tests/test_model.py | 1285 | 1450 | def | 'transformers_wrapper_matches_the_native_model' | def transformers_wrapper_matches_the_native_model(): |
| chapters/39-exporting.md:528 | anulm/hf/modeling_anulm.py | 91–109 |  | reattributed | 'rebuild_rotary' is in anulm/hf/modeling_anulm.py, not tests/test_mode |  |
| chapters/40-the-production-gap.md:213 | anulm/multimodal/audio_codec.py | 227–231 | 231–235 | same | if compute_type is None: | if compute_type is None: |
| chapters/40-the-production-gap.md:358 | anulm/inference/api.py | 195–200 |  | override | decided in repin-overrides.tsv |  |
| chapters/40-the-production-gap.md:364 | anulm/inference/api.py | 305–323 |  | reattributed | '_any_error' is in anulm/inference/api.py, not anulm/inference/serve.p |  |
| chapters/40-the-production-gap.md:377 | anulm/inference/api.py | 296 |  | reattributed | 'GPU' is in anulm/inference/api.py, not anulm/apps/app_omni.py |  |
| chapters/40-the-production-gap.md:470 | anulm/multimodal/audio_codec.py | 220–232 | 224–236 | same | def __init__(self, size: str = "small", device: str = "auto", compute_ | def __init__(self, size: str = "small", device: str = "auto", compute_ |
| chapters/40-the-production-gap.md:472 | anulm/inference/api.py | 195–200 | 195–200 | sweep | limits 195–200, 'GP |  |
| chapters/40-the-production-gap.md:474 | anulm/inference/serve.py | 212–213 |  | override | decided in repin-overrides.tsv |  |
| chapters/40-the-production-gap.md:474 | anulm/inference/serve.py | 212–213 | 212–213 | sweep | ence/serve.py': 'MAX_BODY_BYTES' 47, checked 212–213. |  |
| chapters/40-the-production-gap.md:475 | anulm/apps/app_omni.py | 173–181 | 173–181 | sweep | pp_omni.py': 'Models.__init__''s shared lock 173–181. |  |
| chapters/41-capstone.md:163 | tests/test_model.py | 349–359 | 428–446 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/41-capstone.md:163 | tests/test_model.py | 358 | 445 | edited-quote | assert abs(loss.item() - expected) < 0.35, \ | assert abs(loss.item() - expected) < 0.1, \ |
| chapters/41-capstone.md:599 | tests/test_model.py | 398–406 | 495–503 | same | def generation_is_deterministic_at_top_k_1(): | def generation_is_deterministic_at_top_k_1(): |
| chapters/41-capstone.md:624 | tests/test_model.py | 2250–2271 | 2434–2455 | same | def main(): | def main(): |
| chapters/41-capstone.md:624 | tests/test_model.py | 2266 | 2450 | edited-quote | # ("76 passed, 0 failed, 76 total"), so it keeps that exact shape. | # ("80 passed, 0 failed, 80 total"), so it keeps that exact shape. |
| chapters/41-capstone.md:50 | anulm/training/train.py | 575–576 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:62 | anulm/training/train.py | 554 |  | reattributed | quoted code is in anulm/training/train.py, not .github/workflows/tests |  |
| chapters/41-capstone.md:146 | anulm/training/train.py | 726–730 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:390 | anulm/model.py | 1168 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:653 | tests/test_model.py | 2296 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:653 | tests/test_model.py | 2296 | 2480 | same | sys.exit(main()) | sys.exit(main()) |
| chapters/41-capstone.md:666 | README.md | 115 | 115 | edited | python -m tests.test_model                 # 76 tests, under a minute, | python -m tests.test_model                 # 80 tests, under a minute, |
| chapters/41-capstone.md:868 | anulm/training/train.py | 712–723 | 712–723 | sweep | ng/train.py': 'main' 536–924 (preset builder 712–723, key |  |
| chapters/41-capstone.md:869 | anulm/training/train.py | 726–731 | 726–731 | sweep | 726–731, the |  |
| chapters/41-capstone.md:869 | anulm/training/train.py | 855–878 | 855–878 | sweep | 726–731, the step 855–878, eva |  |
| chapters/41-capstone.md:869 | anulm/training/train.py | 891–908 | 891–908 | sweep | –731, the step 855–878, evaluation and saves 891–908), 'l |  |
| chapters/41-capstone.md:871 | anulm/training/train.py | 554 | 554 | sweep | '--steps' 2000 (line 548), '--warmup' 100 (554), '- |  |
| chapters/41-capstone.md:871 | anulm/training/train.py | 575–576 | 575–576 | sweep | line 548), '--warmup' 100 (554), '--device' (575–576). |  |
| chapters/41-capstone.md:873 | anulm/model.py | 1161–1169 | 1161–1169 | sweep | uLM.generate' 1115–1180 (greedy and sampling 1161–1169), |  |
| chapters/41-capstone.md:875 | tests/test_model.py | 349 | 428 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/41-capstone.md:876 | tests/test_model.py | 398 | 495 | same | def generation_is_deterministic_at_top_k_1(): | def generation_is_deterministic_at_top_k_1(): |
| chapters/41-capstone.md:877 | tests/test_model.py | 410 | 507 | same | def param_count_matches_analysis(): | def param_count_matches_analysis(): |
| chapters/41-capstone.md:877 | tests/test_model.py | 410 | 507 | sweep | 410, 'gr |  |
| chapters/41-capstone.md:877 | tests/test_model.py | 448 | 545 | same | def gradient_accumulation_equals_large_batch(): | def gradient_accumulation_equals_large_batch(): |
| chapters/41-capstone.md:878 | tests/test_model.py | 561 | 658 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/41-capstone.md:879 | tests/test_model.py | 604 | 701 | same | def cached_generation_matches_uncached(): | def cached_generation_matches_uncached(): |
| chapters/41-capstone.md:879 | tests/test_model.py | 2250–2271 | 2434–2455 | same | def main(): | def main(): |
| chapters/41-capstone.md:880 | tests/test_model.py | 699 | 832 | same | raise unittest.SkipTest("this torch has no F.grouped_mm") | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/41-capstone.md:880 | tests/test_model.py | 699 | 832 | sweep | 'unittest.SkipTest' at 699, 720 |  |
| chapters/41-capstone.md:880 | tests/test_model.py | 720 | 853 | same | raise unittest.SkipTest("this torch has no F.grouped_mm") | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/41-capstone.md:880 | tests/test_model.py | 720 | 853 | sweep | 'unittest.SkipTest' at 699, 720, 738 |  |
| chapters/41-capstone.md:880 | tests/test_model.py | 738 | 871 | same | raise unittest.SkipTest("this torch has no F.grouped_mm") | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/41-capstone.md:880 | tests/test_model.py | 738 | 871 | sweep | 'unittest.SkipTest' at 699, 720, 738, 864 |  |
| chapters/41-capstone.md:880 | tests/test_model.py | 864 | 997 | same | raise unittest.SkipTest("gradio not installed") | raise unittest.SkipTest("gradio not installed") |
| chapters/41-capstone.md:880 | tests/test_model.py | 864 | 997 | sweep | 'unittest.SkipTest' at 699, 720, 738, 864, 901 |  |
| chapters/41-capstone.md:880 | tests/test_model.py | 901 | 1034 | same | raise unittest.SkipTest("safetensors not installed; the Hub-name half  | raise unittest.SkipTest("safetensors not installed; the Hub-name half  |
| chapters/41-capstone.md:880 | tests/test_model.py | 901 | 1034 | sweep | 'unittest.SkipTest' at 699, 720, 738, 864, 901, 940 |  |
| chapters/41-capstone.md:880 | tests/test_model.py | 940 | 1073 | same | raise unittest.SkipTest("safetensors not installed") | raise unittest.SkipTest("safetensors not installed") |
| chapters/41-capstone.md:880 | tests/test_model.py | 940 | 1073 | sweep | ittest.SkipTest' at 699, 720, 738, 864, 901, 940, 113 |  |
| chapters/41-capstone.md:880 | tests/test_model.py | 1134 | 1299 | same | raise unittest.SkipTest("gradio not installed") | raise unittest.SkipTest("gradio not installed") |
| chapters/41-capstone.md:880 | tests/test_model.py | 1134 | 1299 | sweep | t.SkipTest' at 699, 720, 738, 864, 901, 940, 1134, 113 |  |
| chapters/41-capstone.md:880 | tests/test_model.py | 1138 | 1303 | same | raise unittest.SkipTest("safetensors not installed") | raise unittest.SkipTest("safetensors not installed") |
| chapters/41-capstone.md:880 | tests/test_model.py | 1138 | 1303 | sweep | Test' at 699, 720, 738, 864, 901, 940, 1134, 1138, 118 |  |
| chapters/41-capstone.md:880 | tests/test_model.py | 1186 | 1351 | same | raise unittest.SkipTest("safetensors not installed") | raise unittest.SkipTest("safetensors not installed") |
| chapters/41-capstone.md:880 | tests/test_model.py | 1186 | 1351 | sweep | at 699, 720, 738, 864, 901, 940, 1134, 1138, 1186, |  |
| chapters/41-capstone.md:881 | tests/test_model.py | 1299 | 1464 | same | raise unittest.SkipTest("transformers not installed") | raise unittest.SkipTest("transformers not installed") |
| chapters/41-capstone.md:881 | tests/test_model.py | 1299 | 1464 | sweep | 1299, 135 |  |
| chapters/41-capstone.md:881 | tests/test_model.py | 1358 | 1523 | same | raise unittest.SkipTest("tokenizers not installed") | raise unittest.SkipTest("tokenizers not installed") |
| chapters/41-capstone.md:881 | tests/test_model.py | 1358 | 1523 | sweep | 1299, 1358, 197 |  |
| chapters/41-capstone.md:881 | tests/test_model.py | 1977 | 2152 | same | raise unittest.SkipTest(f"API dependency unavailable: {e}") | raise unittest.SkipTest(f"API dependency unavailable: {e}") |
| chapters/41-capstone.md:881 | tests/test_model.py | 1977 | 2152 | sweep | 1299, 1358, 1977 and |  |
| chapters/41-capstone.md:881 | tests/test_model.py | 2028 | 2203 | same | raise unittest.SkipTest(f"app dependency unavailable: {e}") | raise unittest.SkipTest(f"app dependency unavailable: {e}") |
| chapters/41-capstone.md:881 | tests/test_model.py | 2028 | 2203 | sweep | 1299, 1358, 1977 and 2028; 'sy |  |
| chapters/41-capstone.md:881 | tests/test_model.py | 2296 | 2480 | same | sys.exit(main()) | sys.exit(main()) |
| chapters/41-capstone.md:881 | tests/test_model.py | 2296 | 2480 | sweep | 299, 1358, 1977 and 2028; 'sys.exit(main())' 2296; 76 |  |
| chapters/41-capstone.md:881 | tests/test_model.py | 76 | 76 | sweep | 358, 1977 and 2028; 'sys.exit(main())' 2296; 76 test |  |
| chapters/41-capstone.md:883 | tests/test_quality_serving.py | 63 | 63 | sweep | 'tests/test_quality_serving.py': 63 'uni |  |
| chapters/41-capstone.md:883 | tests/test_quality_serving.py | 18 | 18 | sweep | st_quality_serving.py': 63 'unittest' tests (18, 19 |  |
| chapters/41-capstone.md:883 | tests/test_quality_serving.py | 19 | 19 | sweep | uality_serving.py': 63 'unittest' tests (18, 19 and |  |
| chapters/41-capstone.md:883 | tests/test_quality_serving.py | 26 | 26 | sweep | serving.py': 63 'unittest' tests (18, 19 and 26). |  |
| chapters/41-capstone.md:885 | anulm/hf/export_hf.py | 328–339 | 328–339 | sweep | float32 config and its comment 328–339: by |  |
| chapters/41-capstone.md:893 | anulm/bpe.py | 11 | 11 | sweep | 11 and |  |
| chapters/41-capstone.md:893 | anulm/bpe.py | 15 | 15 | sweep | 11 and 15, 'co |  |
| chapters/A2-parameter-arithmetic.md:85 | anulm/model.py | 126–156 |  | reattributed | '__post_init__' is in anulm/model.py, not data/multi32k.json |  |
| chapters/A3-test-index.md:11 | tests/test_model.py | 34 |  | override | decided in repin-overrides.tsv |  |
| chapters/A3-test-index.md:12 | tests/test_model.py | 2250 | 2434 | same | def main(): | def main(): |
| chapters/A3-test-index.md:15 | book/code-examples/ch41_shapes.py | 44 |  | reattributed | quoted code is in book/code-examples/ch41_shapes.py, not tests/test_mo |  |
| chapters/A3-test-index.md:136 | tests/test_model.py | 1607 |  | reattributed | 'native_gqa_matches_repeat_interleave' is in tests/test_model.py, not  |  |
| chapters/A3-test-index.md:136 | tests/test_model.py | 1607 | ? | UNMAPPED | m = AnuLM(tiny(n_kv_head=1, sliding_window=4, max_window_layers=1)).ev |  |
| chapters/A3-test-index.md:137 | tests/test_model.py | 1111–1115 |  | reattributed | 'lora_starts_as_the_base_model_and_merges_back_exactly' is in tests/te |  |
| chapters/A3-test-index.md:137 | tests/test_model.py | 1111–1115 | ? | UNMAPPED | trained_bias = router[sorted(router)[0]] |  |
| chapters/A3-test-index.md:138 | tests/test_model.py | 699, 720, 738 |  | reattributed | quoted code is in tests/test_model.py, not book/code-examples/ch41_sha |  |
| chapters/A3-test-index.md:138 | tests/test_model.py | 699 | 832 | same | raise unittest.SkipTest("this torch has no F.grouped_mm") | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/A3-test-index.md:138 | tests/test_model.py | 720 | 853 | same | raise unittest.SkipTest("this torch has no F.grouped_mm") | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/A3-test-index.md:138 | tests/test_model.py | 738 | 871 | same | raise unittest.SkipTest("this torch has no F.grouped_mm") | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/A3-test-index.md:139 | tests/test_model.py | 901 |  | reattributed | quoted code is in tests/test_model.py, not book/code-examples/ch41_sha |  |
| chapters/A3-test-index.md:139 | tests/test_model.py | 901 | 1034 | same | raise unittest.SkipTest("safetensors not installed; the Hub-name half  | raise unittest.SkipTest("safetensors not installed; the Hub-name half  |
| chapters/A6-number-audit.md:187 | anulm/model.py | 756 |  | override | decided in repin-overrides.tsv |  |
| chapters/A6-number-audit.md:187 | anulm/model.py | 784 |  | override | decided in repin-overrides.tsv |  |
| chapters/A6-number-audit.md:196 | anulm/model.py | 756 and 784 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_SOURCE_MAP.md:92 | anulm/training/train.py | 101 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_SOURCE_MAP.md:154 | anulm/multimodal/vision_encoder.py | 505 |  | reattributed | 'train' is in anulm/multimodal/vision_encoder.py, not anulm/multimodal |  |
| REPOSITORY_SOURCE_MAP.md:155 | anulm/multimodal/omni_train.py | 502 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_SOURCE_MAP.md:155 | anulm/multimodal/omni_train.py | 502 | 504 | same | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", |
| REPOSITORY_SOURCE_MAP.md:167 | anulm/multimodal/audio_codec.py | 227–228 | 231–232 | same | if compute_type is None: | if compute_type is None: |
| REPOSITORY_ARCHITECTURE.md:121 | anulm/model.py | 896 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:121 | anulm/model.py | 966–984 |  | reattributed | '_table' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:122 | anulm/model.py | 339–343 |  | reattributed | 'Rotary.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:123 | anulm/model.py | 816–824 |  | reattributed | 'Block.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:124 | anulm/model.py | 451–494 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not anulm/training/train.p |  |
| REPOSITORY_ARCHITECTURE.md:125 | anulm/model.py | 365–381 |  | reattributed | '_attn_mask' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:125 | anulm/model.py | 414–418 |  | reattributed | '_sdpa' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:126 | anulm/model.py | 253 |  | reattributed | quoted code is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:126 | anulm/model.py | 696–707 |  | reattributed | 'MoE.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:127 | anulm/model.py | 624–672 |  | reattributed | 'Router._gate' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:128 | anulm/model.py | 912–928 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:130 | anulm/training/train.py | 375–400 |  | reattributed | 'lr_at' is in anulm/training/train.py, not anulm/muon.py |  |
| REPOSITORY_ARCHITECTURE.md:131 | anulm/training/train.py | 850–908 |  | reattributed | 'main' is in anulm/training/train.py, not anulm/muon.py |  |
| REPOSITORY_ARCHITECTURE.md:146 | anulm/model.py | 873 |  | reattributed | 'new_cache' is in anulm/model.py, not anulm/inference/sample.py |  |
| REPOSITORY_ARCHITECTURE.md:148 | anulm/model.py | 470–476 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:149 | anulm/model.py | 555–563 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:154 | anulm/model.py | 365–381 |  | reattributed | quoted code is in anulm/model.py, not anulm/inference/sample.py |  |
| REPOSITORY_ARCHITECTURE.md:156 | anulm/model.py | 1148–1151 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:164 | anulm/model.py | 891–894 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:164 | anulm/model.py | 1016–1062 |  | reattributed | 'decode_from_embeds' is in anulm/model.py, not anulm/hf/modeling_anulm |  |
| REPOSITORY_ARCHITECTURE.md:173 | anulm/model.py | 709–790 |  | reattributed | '_route_grouped' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:174 | anulm/model.py | 611–622 |  | reattributed | 'Router.forward' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:175 | anulm/model.py | 647–670 |  | reattributed | 'Router._gate' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:175 | anulm/model.py | 1071–1101 |  | reattributed | 'AnuLM.update_expert_biases' is in anulm/model.py, not anulm/hf/modeli |  |
| REPOSITORY_ARCHITECTURE.md:176 | anulm/model.py | 902–907 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:177 | anulm/model.py | 930–956 |  | reattributed | '_chunked_loss' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:178 | anulm/model.py | 390–411 |  | reattributed | 'merge_tree_matrix' is in anulm/model.py, not anulm/hf/modeling_anulm. |  |
| REPOSITORY_ARCHITECTURE.md:179 | anulm/model.py | 860–863 |  | reattributed | 'AnuLM.__init__' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:182 | anulm/training/train.py | 788–840 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:187 | anulm/model.py | 1345–1364 |  | reattributed | 'fit_memory' is in anulm/model.py, not anulm/bigram_teacher.py |  |
| BOOK_PLAN.md:87 | anulm/model.py | 922–928 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/bidirectional.py |  |
| BOOK_PLAN.md:94 | anulm/model.py | 457–463 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not anulm/training/train.p |  |
| RESEARCH_REFERENCES.md:301 | anulm/model.py | 109–110 |  | override | decided in repin-overrides.tsv |  |
| RESEARCH_REFERENCES.md:301 | anulm/model.py | 449 |  | override | decided in repin-overrides.tsv |  |
| RESEARCH_REFERENCES.md:304 | anulm/model.py | 443–445 |  | override | decided in repin-overrides.tsv |  |
