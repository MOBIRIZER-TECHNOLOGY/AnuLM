# Re-pin report: @084cbdd -> @fdbb09a

Written by `book/tools/repin.py`. Every line citation it moved, with the code at
the old line and at the new one. Rows marked UNMAPPED, CHECK, EXCERPT or OUT OF
FILE were **not** resolved automatically and need a person.

| kind | count |
|---|---|
| CHECK | 67 |
| EXCERPT | 14 |
| OUT OF FILE | 1 |
| UNMAPPED | 6 |
| def | 49 |
| edited | 20 |
| edited-quote | 23 |
| reattributed | 87 |
| same | 871 |
| sweep | 101 |

## Needs a person (88)

| where | file | old | new | kind | old code | new code |
|---|---|---|---|---|---|---|
| chapters/01-what-this-book-builds.md:196 | anulm/inference/sample.py | 34–35 | ? | CHECK | p.add_argument("--temperature", type=float, default=0.8) | no file named on the line and nothing quoted confirms it |
| chapters/05-softmax-temperature-and-cross-entropy.md:163 | anulm/training/train.py | 1140 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/05-softmax-temperature-and-cross-entropy.md:243 | anulm/inference/sample.py | 34–35 | ? | CHECK | p.add_argument("--temperature", type=float, default=0.8) | no file named on the line and nothing quoted confirms it |
| chapters/05-softmax-temperature-and-cross-entropy.md:252 | anulm/inference/sample.py | 1154–1155 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:193 | anulm/muon.py | 13–14 | ? | CHECK | moments per parameter; Muon keeps one momentum buffer. That is 16 vs 1 | no file named on the line and nothing quoted confirms it |
| chapters/08-bytes-as-tokens.md:357 | anulm/training/train.py | 660 | ? | CHECK | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 | no file named on the line and nothing quoted confirms it |
| chapters/09-byte-level-bpe.md:213 | anulm/bpe.py | 120–121 | ? | CHECK | if -neg != pair_counts.get(pair, 0): | no file named on the line and nothing quoted confirms it |
| chapters/09-byte-level-bpe.md:425 | anulm/bpe.py | 1342–1346 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/09-byte-level-bpe.md:537 | docs/RESULTS.md | 120–121 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/10-embeddings-and-the-output-table.md:340 | anulm/model.py | 390 | ? | CHECK | coef = [{i: 1.0} for i in range(256)] | no file named on the line and nothing quoted confirms it |
| chapters/11-attention.md:418 | anulm/model.py | 367 | ? | CHECK | if window is None and (start == 0 or S == 1): | no file named on the line and nothing quoted confirms it |
| chapters/11-attention.md:424 | anulm/model.py | 369–373 | ? | CHECK | q_pos = torch.arange(start, start + S, device=device)[:, None] | no file named on the line and nothing quoted confirms it |
| chapters/11-attention.md:465 | tests/test_model.py | 421–423 | ? | CHECK | def mla_shares_one_rope_key_across_heads(): | no file named on the line and nothing quoted confirms it |
| chapters/11-attention.md:523 | anulm/model.py | 224 | ? | CHECK | self.weight = nn.Parameter(torch.ones(dim)) | no file named on the line and nothing quoted confirms it |
| chapters/12-causal-masking.md:155 | anulm/bidirectional.py | 568 | ? | OUT OF FILE |  |  |
| chapters/13-multi-head-and-gqa.md:31 | anulm/model.py | 144–147 | ? | UNMAPPED | if self.attn == "gqa": |  |
| chapters/13-multi-head-and-gqa.md:31 | anulm/model.py | 144 | ? | EXCERPT | header range unresolved: AnuLMConfig.__post_init__ |  |
| chapters/13-multi-head-and-gqa.md:75 | anulm/model.py | 442 | ? | CHECK | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) | no file named on the line and nothing quoted confirms it |
| chapters/13-multi-head-and-gqa.md:244 | anulm/model.py | 516–522 | ? | CHECK | self.kv_b_proj = nn.Linear( | no file named on the line and nothing quoted confirms it |
| chapters/13-multi-head-and-gqa.md:394 | anulm/model.py | 144–147 | ? | UNMAPPED | if self.attn == "gqa": |  |
| chapters/14-rotary-positions.md:159 | docs/RESULTS.md | 282–283 | ? | CHECK | moves ±0.05 in 50 steps against sigmoid scores spanning (0,1), so the  | no file named on the line and nothing quoted confirms it |
| chapters/17-router.md:336 | tests/test_model.py | 594–603 | ? | CHECK | """Greedy decoding with and without the cache must agree token-for-tok | no file named on the line and nothing quoted confirms it |
| chapters/18-dispatch-and-balance.md:259 | docs/RESULTS.md | 713–714 | ? | CHECK | Balancing got harder: 24 experts finish at 1.76× against 1.42× — the b | no file named on the line and nothing quoted confirms it |
| chapters/19-latent-attention.md:107 | anulm/model.py | 150–156 |  | EXCERPT | assert self.n_head % self.n_kv_head == 0 | quoted line not in the new range |
| chapters/19-latent-attention.md:62 | anulm/model.py | 543–544 | ? | CHECK | # RoPE touches only the rope slice; the latent stays position-free -- | no file named on the line and nothing quoted confirms it |
| chapters/19-latent-attention.md:472 | anulm/model.py | 830 | ? | CHECK | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | no file named on the line and nothing quoted confirms it |
| chapters/20-the-whole-model.md:105 | anulm/model.py | 833–863 |  | EXCERPT | assert cfg.merge_tree in ("in", "out", "both"), cfg.merge_tree | quoted line not in the new range |
| chapters/20-the-whole-model.md:105 | anulm/model.py | 833–863 |  | EXCERPT | merges = _json.load(open(cfg.tokenizer_path, encoding="utf-8"))["merge | quoted line not in the new range |
| chapters/21-data.md:59 | experiments/build_v2.py | 119 | ? | CHECK | if n_tr >= budget: | no file named on the line and nothing quoted confirms it |
| chapters/22-the-training-loop.md:43 | anulm/training/train.py | 712–732 |  | EXCERPT | assert field is not None and _, f"--cfg: unknown field {kv!r}" | quoted line not in the new range |
| chapters/23-precision-memory-speed.md:517 | anulm/training/train.py | 705 | ? | UNMAPPED | assert not args.ce_chunk, "--bigram-teacher needs the logits; drop --c |  |
| chapters/24-measuring-progress.md:478 | anulm/training/train.py | 903–905 | ? | CHECK | bpb = val / math.log(2) / args.bytes_per_token | 'eval_spec' spans [(784, 788, 784)] at the old pin |
| chapters/25-checkpoints-and-resuming.md:209 | anulm/training/train.py | 788–840 |  | EXCERPT | ck = torch.load(last_path, map_location="cpu", weights_only=False) | quoted line not in the new range |
| chapters/25-checkpoints-and-resuming.md:369 | anulm/model.py | 1280–1316 |  | EXCERPT | if not p.exists() and re.fullmatch(r"[\w.-]+/[\w.-]+", str(path).repla | quoted line not in the new range |
| chapters/25-checkpoints-and-resuming.md:166 | anulm/training/train.py | 649 | ? | CHECK | torch.manual_seed(args.seed) | no file named on the line and nothing quoted confirms it |
| chapters/25-checkpoints-and-resuming.md:167 | anulm/training/train.py | 856 | ? | CHECK | if args.sample_with_replacement else sampler.next()) | no file named on the line and nothing quoted confirms it |
| chapters/25-checkpoints-and-resuming.md:355 | anulm/training/train.py | 384–395 | ? | CHECK | if step < args.warmup: | no file named on the line and nothing quoted confirms it |
| chapters/26-debugging-silent-failures.md:523 | tests/test_model.py | 28 | ? | CHECK | def test(fn): | 'TESTS' spans [(25, 25, 25)] at the old pin |
| chapters/27-decoding-and-the-kv-cache.md:91 | anulm/model.py | 1132–1139 | ? | CHECK | for _ in range(max_new_tokens): | no file named on the line and nothing quoted confirms it |
| chapters/28-choosing-the-next-token.md:87 | anulm/model.py | 1157 | ? | CHECK | nxt = torch.multinomial(probs, num_samples=1)          # (B, 1) | no file named on the line and nothing quoted confirms it |
| chapters/28-choosing-the-next-token.md:96 | anulm/model.py | 1149-1150 | ? | CHECK | if temperature <= 0: | no file named on the line and nothing quoted confirms it |
| chapters/28-choosing-the-next-token.md:102 | anulm/model.py | 1145-1156 | ? | CHECK | if repetition_penalty != 1.0: | no file named on the line and nothing quoted confirms it |
| chapters/28-choosing-the-next-token.md:472 | anulm/inference/serve.py | 210–214 | ? | CHECK | temperature=float(req.get("temperature", 0.8)), | 'generate' spans [(134, 171, 134)] at the old pin |
| chapters/29-long-context.md:185 | anulm/model.py | 281–285 | ? | CHECK | # YaRN's temperature, split the way DeepSeek (and so the 105B) splits | no file named on the line and nothing quoted confirms it |
| chapters/30-instruction-tuning.md:583 | anulm/multimodal/omni_eval.py | 192–195 | ? | CHECK | held = HERE / "data" / "chat_heldout.jsonl" | 'chat_loss' spans [(114, 131, 115)] at the old pin |
| chapters/30-instruction-tuning.md:585 | anulm/data/make_chat.py | 109–112 | ? | CHECK | return [] | 'main' spans [(148, 184, 148)] at the old pin |
| chapters/32-classification-and-calibration.md:21 | anulm/tasks/decide.py | 89–101 |  | EXCERPT | option k's name. Pieces are encoded separately so the positions are ex | quoted line not in the new range |
| chapters/32-classification-and-calibration.md:21 | anulm/tasks/decide.py | 89–101 |  | EXCERPT | assert len(ids) <= max_len, f"{len(ids)} tokens; too many options for  | quoted line not in the new range |
| chapters/32-classification-and-calibration.md:74 | anulm/tasks/decide.py | 118 | ? | CHECK | self.head = nn.Linear(hid, 1) | no file named on the line and nothing quoted confirms it |
| chapters/32-classification-and-calibration.md:76 | anulm/tasks/decide.py | 594 | ? | CHECK | loss = F.cross_entropy(logits, y.to(a.device)) | no file named on the line and nothing quoted confirms it |
| chapters/33-retrieval-and-readers.md:237 | anulm/inference/rag.py | 250 | ? | CHECK | # the top 3 every time (BM25: 15/20) and took our own reader from 0/20 | no file named on the line and nothing quoted confirms it |
| chapters/33-retrieval-and-readers.md:272 | anulm/inference/rag.py | 350–360 | ? | CHECK | def read(self, prompt: str, max_tokens: int = 32) -> str: | no file named on the line and nothing quoted confirms it |
| chapters/33-retrieval-and-readers.md:497 | anulm/inference/rag.py | 91 | ? | CHECK | ids = term_ids(title + " " + chunk) | 'term_ids' spans [(57, 60, 57)] at the old pin |
| chapters/35-wider-vocabulary.md:253 | anulm/model.py | 884 | ? | CHECK | x = F.embedding(idx, self._table("in"))               # (B, S, D) | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:299 | anulm/multimodal/omni_train.py | 365–403 |  | EXCERPT | st = torch.load(last, map_location=dev, weights_only=False) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:202 | anulm/multimodal/omni_train.py | 33–37 | ? | CHECK | **Held out.** The first 2% of distinct input groups (normalized media  | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:206 | experiments/omni_one.py | 77–87 | ? | CHECK | TASKS = [ | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:334 | anulm/multimodal/omni_train.py | 474 | ? | CHECK | st.update(opt=opt.state_dict(), first=first, best=best, seen=seen, rec | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:339 | anulm/multimodal/omni_train.py | 403–413 | ? | CHECK | model.load_state_dict(st["model"]) | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:341 | anulm/multimodal/omni_train.py | 505 | ? | CHECK | p.add_argument("--stop-after", type=int, default=0, | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:343 | anulm/multimodal/omni_train.py | 470–471 | ? | CHECK | if args.stop_after and step >= args.stop_after: | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:513 | experiments/omni_one.py | 94–97 | ? | CHECK | # Hindi: the real-speaker data (Kathbath, MUCS) trades FLEURS for ever | no file named on the line and nothing quoted confirms it |
| chapters/38-serving.md:116 | anulm/inference/serve.py | 198–218 | ? | UNMAPPED | def do_POST(self): |  |
| chapters/38-serving.md:116 | anulm/inference/serve.py | 198 | ? | EXCERPT | header range unresolved: make_handler |  |
| chapters/38-serving.md:109 | anulm/inference/serve.py | 147 | ? | CHECK | ids = self.encode(prompt)[-(self.cfg.block_size - 1):] | no file named on the line and nothing quoted confirms it |
| chapters/38-serving.md:110 | anulm/inference/serve.py | 141–146 | ? | CHECK | if mode == "translate" and self.qa_templates and "en-hi" in self.qa_te | no file named on the line and nothing quoted confirms it |
| chapters/39-exporting.md:248 | anulm/model.py | 1296–1319 |  | EXCERPT | if not p.exists() and re.fullmatch(r"[\w.-]+/[\w.-]+", str(path).repla | quoted line not in the new range |
| chapters/39-exporting.md:281 | tests/test_model.py | 961–963 | ? | CHECK | for name in ("eval_bench.py", "eval_context.py", "eval_golden.py", "ev | no file named on the line and nothing quoted confirms it |
| chapters/40-the-production-gap.md:343 | anulm/inference/api.py | 173–180 |  | EXCERPT | if key and authorization != f"Bearer {key}": | quoted line not in the new range |
| chapters/41-capstone.md:77 | anulm/training/train.py | 712–731 |  | EXCERPT | assert field is not None and _, f"--cfg: unknown field {kv!r}" | quoted line not in the new range |
| chapters/41-capstone.md:50 | docs/TUTORIAL.md | 579–580 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/41-capstone.md:140 | anulm/training/train.py | 728–732 | ? | CHECK | kw = dict(block_size=args.block_size, yarn=args.yarn, | no file named on the line and nothing quoted confirms it |
| chapters/41-capstone.md:384 | anulm/model.py | 1156 | ? | CHECK | probs = F.softmax(logits, dim=-1) | no file named on the line and nothing quoted confirms it |
| chapters/41-capstone.md:638 | tests/test_model.py | 2230 | ? | CHECK | sys.exit(main()) | no file named on the line and nothing quoted confirms it |
| chapters/41-capstone.md:693 | data/multi32k.json | 749 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/A3-test-index.md:13 | tests/test_model.py | 33 | ? | CHECK | def tiny(**kw) -> AnuLMConfig: | no file named on the line and nothing quoted confirms it |
| chapters/A3-test-index.md:136 | tests/test_model.py | 1282–1286 | ? | UNMAPPED | try: |  |
| chapters/A3-test-index.md:136 | tests/test_model.py | 1342–1346 | ? | UNMAPPED | try: |  |
| REPOSITORY_SOURCE_MAP.md:92 | anulm/training/train.py | 101 | ? | CHECK | BYTE_EOS = 257          # byte mode: 256 BOS, 257 EOS, 258 PAD | 'load_bytes' spans [(62, 98, 62)] at the old pin |
| REPOSITORY_SOURCE_MAP.md:155 | anulm/multimodal/omni_train.py | 501 | ? | CHECK | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", | 'Omni._match_scale' spans [(249, 263, 250)] at the old pin |
| REPOSITORY_ARCHITECTURE.md:148 | anulm/inference/sample.py | 463–469 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| REPOSITORY_ARCHITECTURE.md:149 | anulm/inference/sample.py | 548–556 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| REPOSITORY_ARCHITECTURE.md:156 | anulm/inference/sample.py | 1136–1139 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| REPOSITORY_ARCHITECTURE.md:164 | anulm/hf/modeling_anulm.py | 879–882 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| REPOSITORY_ARCHITECTURE.md:182 | anulm/training/train.py | 790–837 | ? | CHECK | start_step, best_val = 0, float("inf") | 'atomic_save' spans [(32, 46, 32)] at the old pin |
| RESEARCH_REFERENCES.md:270 | anulm/model.py | 109–110 | ? | CHECK | yarn_beta_fast: float = 32.0 | 'yarn_mscale' spans [(263, 268, 263)] at the old pin |
| RESEARCH_REFERENCES.md:270 | anulm/model.py | 442 | ? | CHECK | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) | 'yarn_mscale' spans [(263, 268, 263)] at the old pin |
| RESEARCH_REFERENCES.md:273 | anulm/model.py | 436–438 | ? | CHECK | if cfg.use_qk_norm: | 'RMSNorm' spans [(218, 232, 218)] at the old pin |

## Every change

| where | file | old | new | kind | old code | new code |
|---|---|---|---|---|---|---|
| chapters/01-what-this-book-builds.md:114 | anulm/model.py | 1132–1161 | 1144–1173 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/01-what-this-book-builds.md:159 | anulm/model.py | 903–906 | 915–918 | same | if targets is None: | if targets is None: |
| chapters/01-what-this-book-builds.md:235 | anulm/model.py | 820–836 | 827–843 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/01-what-this-book-builds.md:367 | anulm/model.py | 152–164 | 159–171 | same | @staticmethod | @staticmethod |
| chapters/01-what-this-book-builds.md:99 | tests/test_model.py | 387 | 398 | def | 'generation_is_deterministic_at_top_k_1' | def generation_is_deterministic_at_top_k_1(): |
| chapters/01-what-this-book-builds.md:111 | anulm/model.py | 1104 |  | reattributed | 'generate' is defined there |  |
| chapters/01-what-this-book-builds.md:111 | anulm/model.py | 1104 | 1116 | def | 'generate' | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/01-what-this-book-builds.md:196 | anulm/inference/sample.py | 34–35 | ? | CHECK | p.add_argument("--temperature", type=float, default=0.8) | no file named on the line and nothing quoted confirms it |
| chapters/01-what-this-book-builds.md:363 | anulm/quickstart.py | 226 |  | reattributed | quoted code is in anulm/quickstart.py, not anulm/inference/sample.py |  |
| chapters/01-what-this-book-builds.md:532 | anulm/model.py | 215 | 222 | same | # NORM / MLP | # NORM / MLP |
| chapters/01-what-this-book-builds.md:532 | anulm/model.py | 250 | 257 | same | # ROPE  (see sarvam/yarn_rotary_annotated.py) | # ROPE  (see sarvam/yarn_rotary_annotated.py) |
| chapters/01-what-this-book-builds.md:532 | anulm/model.py | 415 | 422 | same | # ATTENTION -- GQA  (see sarvam/gqa_forward_annotated.py) | # ATTENTION -- GQA  (see sarvam/gqa_forward_annotated.py) |
| chapters/01-what-this-book-builds.md:532 | anulm/model.py | 491 | 498 | same | # ATTENTION -- MLA  (see sarvam/mla_forward_annotated.py) | # ATTENTION -- MLA  (see sarvam/mla_forward_annotated.py) |
| chapters/01-what-this-book-builds.md:532 | anulm/model.py | 575 | 582 | same | # MoE  (see sarvam/moe_routing_annotated.py) | # MoE  (see sarvam/moe_routing_annotated.py) |
| chapters/01-what-this-book-builds.md:532 | anulm/model.py | 787 | 794 | same | # BLOCK / MODEL | # BLOCK / MODEL |
| chapters/01-what-this-book-builds.md:533 | anulm/model.py | 152–164 | 159–171 | same | @staticmethod | @staticmethod |
| chapters/01-what-this-book-builds.md:533 | anulm/model.py | 820–836 | 827–843 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/01-what-this-book-builds.md:534 | anulm/model.py | 903–906 | 915–918 | same | if targets is None: | if targets is None: |
| chapters/01-what-this-book-builds.md:534 | anulm/model.py | 1104–1168 | 1116–1180 | same | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/01-what-this-book-builds.md:534 | anulm/model.py | 1132–1161 | 1144–1173 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/01-what-this-book-builds.md:534 | anulm/model.py | 1132–1161 | 1144–1173 | sweep | d' 903–906; 'AnuLM.generate' 1104–1168 (loop 1132–1161; |  |
| chapters/01-what-this-book-builds.md:535 | anulm/model.py | 1149–1150 | 1161–1162 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/01-what-this-book-builds.md:535 | anulm/model.py | 1149–1150 | 1161–1162 | sweep | the 'temperature <= 0' greedy branch 1149–1150). |  |
| chapters/01-what-this-book-builds.md:536 | tests/test_model.py | 33–38 | 44–49 | same | def tiny(**kw) -> AnuLMConfig: | def tiny(**kw) -> AnuLMConfig: |
| chapters/01-what-this-book-builds.md:536 | tests/test_model.py | 387 | 398 | same | def generation_is_deterministic_at_top_k_1(): | def generation_is_deterministic_at_top_k_1(): |
| chapters/01-what-this-book-builds.md:537 | anulm/inference/sample.py | 34–35 | 34–35 | sweep | inference/sample.py': 'main' 26–93 (defaults 34–35, gen |  |
| chapters/01-what-this-book-builds.md:537 | anulm/inference/sample.py | 69–77 | 69–77 | sweep | y': 'main' 26–93 (defaults 34–35, generation 69–77). |  |
| chapters/02-from-counts-to-transformers.md:82 | anulm/model.py | 829–834 | 836–841 | same | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) |
| chapters/02-from-counts-to-transformers.md:96 | anulm/model.py | 888–900 | 900–912 | same | aux_total = None | aux_total = None |
| chapters/02-from-counts-to-transformers.md:265 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/02-from-counts-to-transformers.md:280 | anulm/model.py | 463–469 | 470–476 | same | if cache is not None: | if cache is not None: |
| chapters/02-from-counts-to-transformers.md:67 | anulm/bigram_teacher.py | 65–70 |  | reattributed | 'teacher_kl' is in anulm/bigram_teacher.py, not docs/RESULTS.md |  |
| chapters/02-from-counts-to-transformers.md:334 | anulm/model.py | 474 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not anulm/bidirectional.py |  |
| chapters/02-from-counts-to-transformers.md:334 | anulm/model.py | 474 | 481 | same | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star |
| chapters/02-from-counts-to-transformers.md:416 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/02-from-counts-to-transformers.md:416 | anulm/model.py | 463–469 | 470–476 | same | if cache is not None: | if cache is not None: |
| chapters/02-from-counts-to-transformers.md:416 | anulm/model.py | 474 | 481 | same | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star |
| chapters/02-from-counts-to-transformers.md:417 | anulm/model.py | 829–834 | 836–841 | same | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) |
| chapters/02-from-counts-to-transformers.md:417 | anulm/model.py | 888–900 | 900–912 | same | aux_total = None | aux_total = None |
| chapters/02-from-counts-to-transformers.md:418 | anulm/bigram_teacher.py | 42–51 | 42–51 | sweep | er.py': 'build_bigram_teacher' 34–62 (counts 42–51), |  |
| chapters/02-from-counts-to-transformers.md:420 | anulm/bidirectional.py | 82–93 | 82–93 | sweep | : 'bidir_hidden' 40–53; 'adapt' 56–105 (MNTP 82–93). |  |
| chapters/02-from-counts-to-transformers.md:422 | anulm/training/train.py | 241–254 | 237–250 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/02-from-counts-to-transformers.md:422 | anulm/training/train.py | 451–478 | 447–474 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/03-the-one-job.md:22 | anulm/training/train.py | 241–251 | 237–247 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/03-the-one-job.md:47 | anulm/model.py | 910–916 | 922–928 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/03-the-one-job.md:173 | anulm/model.py | 847–851 | 859–863 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/03-the-one-job.md:183 | anulm/model.py | 853–859 | 865–871 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/03-the-one-job.md:257 | tests/test_model.py | 338–348 | 349–359 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/03-the-one-job.md:279 | anulm/model.py | 1184–1193 | 1196–1205 | same | x = torch.randint(0, cfg.vocab_size, (2, 64)) | x = torch.randint(0, cfg.vocab_size, (2, 64)) |
| chapters/03-the-one-job.md:311 | anulm/training/train.py | 903–905 | 906–908 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/03-the-one-job.md:322 | anulm/training/train.py | 660–675 | 656–671 | same | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 |
| chapters/03-the-one-job.md:349 | anulm/training/train.py | 903 | 906 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/03-the-one-job.md:428 | anulm/model.py | 847–851 | 859–863 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/03-the-one-job.md:428 | anulm/model.py | 853–859 | 865–871 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/03-the-one-job.md:429 | anulm/model.py | 910–916 | 922–928 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/03-the-one-job.md:429 | anulm/model.py | 1171–1196 | 1183–1208 | same | if __name__ == "__main__": | if __name__ == "__main__": |
| chapters/03-the-one-job.md:430 | anulm/training/train.py | 241–254 | 237–250 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/03-the-one-job.md:430 | anulm/training/train.py | 660–675 | 656–671 | same | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 |
| chapters/03-the-one-job.md:431 | anulm/training/train.py | 903–905 | 906–908 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/03-the-one-job.md:431 | anulm/training/train.py | 903–905 | 906–908 | sweep | 903–905 (bit |  |
| chapters/03-the-one-job.md:432 | tests/test_model.py | 33–38 | 44–49 | same | def tiny(**kw) -> AnuLMConfig: | def tiny(**kw) -> AnuLMConfig: |
| chapters/03-the-one-job.md:432 | tests/test_model.py | 338–348 | 349–359 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:73 | anulm/model.py | 430–434 | 437–441 | same | # One fused projection: (H + 2*Hkv) * Dh, exactly as the 30B does. | # One fused projection: (H + 2*Hkv) * Dh, exactly as the 30B does. |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:110 | anulm/model.py | 450–456 | 457–463 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:175 | anulm/model.py | 476–487 | 483–494 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:237 | anulm/model.py | 339–355 | 346–362 | same | def rotate_half(x): | def rotate_half(x): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:265 | anulm/model.py | 227–232 | 234–239 | same | def forward(self, x): | def forward(self, x): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:307 | anulm/model.py | 463–470 | 470–477 | same | if cache is not None: | if cache is not None: |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:62 | anulm/model.py | 454–456 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not None |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:62 | anulm/model.py | 454–456 | 461–463 | same | q = q.transpose(1, 2)                         # (B, H,   S, Dh) | q = q.transpose(1, 2)                         # (B, H,   S, Dh) |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:192 | anulm/model.py | 380 | 387 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:228 | anulm/model.py | 486 |  | reattributed | quoted code is in anulm/model.py, not None |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:228 | anulm/model.py | 486 | 493 | same | y = y.transpose(1, 2).contiguous().view(B, S, -1) | y = y.transpose(1, 2).contiguous().view(B, S, -1) |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:428 | anulm/model.py | 227–232 | 234–239 | same | def forward(self, x): | def forward(self, x): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:428 | anulm/model.py | 339–343 | 346–350 | same | def rotate_half(x): | def rotate_half(x): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:429 | anulm/model.py | 346–355 | 353–362 | same | def apply_rope(q, k, cos, sin): | def apply_rope(q, k, cos, sin): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:429 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:429 | anulm/model.py | 430–434 | 437–441 | same | # One fused projection: (H + 2*Hkv) * Dh, exactly as the 30B does. | # One fused projection: (H + 2*Hkv) * Dh, exactly as the 30B does. |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:430 | anulm/model.py | 380 | 387 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:430 | anulm/model.py | 450–456 | 457–463 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:430 | anulm/model.py | 463–470 | 470–477 | same | if cache is not None: | if cache is not None: |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:430 | anulm/model.py | 476–487 | 483–494 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:431 | tests/test_model.py | 33–38 | 44–49 | same | def tiny(**kw) -> AnuLMConfig: | def tiny(**kw) -> AnuLMConfig: |
| chapters/05-softmax-temperature-and-cross-entropy.md:77 | anulm/model.py | 617–619 | 624–626 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/05-softmax-temperature-and-cross-entropy.md:94 | anulm/model.py | 911–913 | 923–925 | same | loss = F.cross_entropy( | loss = F.cross_entropy( |
| chapters/05-softmax-temperature-and-cross-entropy.md:150 | anulm/training/train.py | 649–654 | 645–650 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/05-softmax-temperature-and-cross-entropy.md:172 | anulm/model.py | 1140–1157 | 1152–1169 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:230 | anulm/inference/api.py | 399–401 | 446–448 | same | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/05-softmax-temperature-and-cross-entropy.md:264 | anulm/model.py | 1026–1037 | 1038–1049 | same | scores = logits[0, -1].float() | scores = logits[0, -1].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:394 | anulm/model.py | 932–944 | 944–956 | same | def _chunked_loss(self, x, targets): | def _chunked_loss(self, x, targets): |
| chapters/05-softmax-temperature-and-cross-entropy.md:161 | anulm/model.py | 912 |  | reattributed | quoted code is in anulm/model.py, not anulm/training/train.py |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:161 | anulm/model.py | 912 | 924 | same | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ |
| chapters/05-softmax-temperature-and-cross-entropy.md:163 | anulm/training/train.py | 1140 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/05-softmax-temperature-and-cross-entropy.md:238 | anulm/inference/api.py | 192 | 218 | same | temperature: Optional[float] = Field(None, ge=0, le=2, description="De | temperature: Optional[float] = Field(None, ge=0, le=2, description="De |
| chapters/05-softmax-temperature-and-cross-entropy.md:243 | anulm/inference/sample.py | 34–35 | ? | CHECK | p.add_argument("--temperature", type=float, default=0.8) | no file named on the line and nothing quoted confirms it |
| chapters/05-softmax-temperature-and-cross-entropy.md:252 | anulm/inference/sample.py | 1154–1155 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/05-softmax-temperature-and-cross-entropy.md:485 | anulm/model.py | 617–619 | 624–626 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/05-softmax-temperature-and-cross-entropy.md:485 | anulm/model.py | 911–913 | 923–925 | same | loss = F.cross_entropy( | loss = F.cross_entropy( |
| chapters/05-softmax-temperature-and-cross-entropy.md:486 | anulm/model.py | 932–941 | 944–953 | same | def _chunked_loss(self, x, targets): | def _chunked_loss(self, x, targets): |
| chapters/05-softmax-temperature-and-cross-entropy.md:486 | anulm/model.py | 943–944 | 955–956 | same | def _ce_sum(self, h, t): | def _ce_sum(self, h, t): |
| chapters/05-softmax-temperature-and-cross-entropy.md:487 | anulm/model.py | 1026–1037 | 1038–1049 | same | scores = logits[0, -1].float() | scores = logits[0, -1].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:487 | anulm/model.py | 1140–1157 | 1152–1169 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:488 | anulm/inference/api.py | 399–401 | 446–448 | same | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/05-softmax-temperature-and-cross-entropy.md:489 | anulm/inference/api.py | 192 | 218 | same | temperature: Optional[float] = Field(None, ge=0, le=2, description="De | temperature: Optional[float] = Field(None, ge=0, le=2, description="De |
| chapters/05-softmax-temperature-and-cross-entropy.md:491 | anulm/training/train.py | 649–654 | 645–650 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/06-gradients-and-backpropagation.md:84 | anulm/model.py | 235–246 | 242–253 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/06-gradients-and-backpropagation.md:167 | anulm/model.py | 813–817 | 820–824 | same | x = x + self.attn(self.input_layernorm(x), cos, sin, | x = x + self.attn(self.input_layernorm(x), cos, sin, |
| chapters/06-gradients-and-backpropagation.md:196 | anulm/training/train.py | 852–870 | 855–873 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/06-gradients-and-backpropagation.md:242 | tests/test_model.py | 437–455 | 448–466 | same | def gradient_accumulation_equals_large_batch(): | def gradient_accumulation_equals_large_batch(): |
| chapters/06-gradients-and-backpropagation.md:313 | tests/test_model.py | 325–334 | 336–345 | same | def all_parameters_receive_gradients(): | def all_parameters_receive_gradients(): |
| chapters/06-gradients-and-backpropagation.md:338 | tests/test_model.py | 67–82 | 78–93 | same | def causality_holds_under_gradients_too(): | def causality_holds_under_gradients_too(): |
| chapters/06-gradients-and-backpropagation.md:236 | anulm/training/train.py | 482 | 478 | def | 'evaluate' | def evaluate(model, data, args, iters: int \| None = None) -> tuple[flo |
| chapters/06-gradients-and-backpropagation.md:236 | anulm/model.py | 1104 |  | reattributed | 'AnuLM.generate' is in anulm/model.py, not anulm/training/train.py |  |
| chapters/06-gradients-and-backpropagation.md:236 | anulm/model.py | 1104 | 1116 | same | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/06-gradients-and-backpropagation.md:238 | anulm/model.py | 901 |  | reattributed | quoted code is in anulm/model.py, not anulm/training/train.py |  |
| chapters/06-gradients-and-backpropagation.md:238 | anulm/model.py | 901 | 913 | same | self.aux_loss = aux_total.detach() if aux_total is not None else None | self.aux_loss = aux_total.detach() if aux_total is not None else None |
| chapters/06-gradients-and-backpropagation.md:438 | anulm/model.py | 235–246 | 242–253 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/06-gradients-and-backpropagation.md:438 | anulm/model.py | 813–817 | 820–824 | same | x = x + self.attn(self.input_layernorm(x), cos, sin, | x = x + self.attn(self.input_layernorm(x), cos, sin, |
| chapters/06-gradients-and-backpropagation.md:439 | anulm/model.py | 901 | 913 | same | self.aux_loss = aux_total.detach() if aux_total is not None else None | self.aux_loss = aux_total.detach() if aux_total is not None else None |
| chapters/06-gradients-and-backpropagation.md:439 | anulm/model.py | 1103–1104 | 1115–1116 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/06-gradients-and-backpropagation.md:440 | anulm/training/train.py | 481–482 | 477–478 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/06-gradients-and-backpropagation.md:440 | anulm/training/train.py | 852–870 | 855–873 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/06-gradients-and-backpropagation.md:441 | tests/test_model.py | 67–82 | 78–93 | same | def causality_holds_under_gradients_too(): | def causality_holds_under_gradients_too(): |
| chapters/06-gradients-and-backpropagation.md:442 | tests/test_model.py | 325–334 | 336–345 | same | def all_parameters_receive_gradients(): | def all_parameters_receive_gradients(): |
| chapters/06-gradients-and-backpropagation.md:443 | tests/test_model.py | 437–455 | 448–466 | same | def gradient_accumulation_equals_large_batch(): | def gradient_accumulation_equals_large_batch(): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:76 | anulm/training/train.py | 451–478 | 447–474 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:204 | anulm/training/train.py | 379–404 | 375–400 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:249 | anulm/training/train.py | 849–850 | 852–853 | same | lr = lr_at(step, args, start_step) | lr = lr_at(step, args, start_step) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:256 | anulm/training/train.py | 420–424 | 416–420 | same | def set_lr_scale(self, scale: float): | def set_lr_scale(self, scale: float): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:308 | tests/test_model.py | 1610–1619 | 1627–1636 | same | def wsd_schedule_is_flat_then_decays_to_min_lr(): | def wsd_schedule_is_flat_then_decays_to_min_lr(): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:334 | anulm/training/train.py | 871–875 | 874–878 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:21 | anulm/training/train.py | 559–560 | 555–556 | same | p.add_argument("--weight-decay", type=float, default=0.1) | p.add_argument("--weight-decay", type=float, default=0.1) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:193 | anulm/muon.py | 13–14 | ? | CHECK | moments per parameter; Muon keeps one momentum buffer. That is 16 vs 1 | no file named on the line and nothing quoted confirms it |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:456 | anulm/muon.py | 59 |  | reattributed | 'zeropower_batched' is defined there |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:483 | tests/test_model.py | 1712 | 1729 | def | 'batched_muon_matches_per_matrix_muon' | def batched_muon_matches_per_matrix_muon(): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:563 | anulm/training/train.py | 379–404 | 375–400 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:563 | anulm/training/train.py | 407–448 | 403–444 | same | class OptimizerSet: | class OptimizerSet: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:564 | anulm/training/train.py | 420–424 | 416–420 | same | def set_lr_scale(self, scale: float): | def set_lr_scale(self, scale: float): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:564 | anulm/training/train.py | 420–424 | 416–420 | sweep | 420–424); 'm |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:564 | anulm/training/train.py | 451–478 | 447–474 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:564 | anulm/training/train.py | 559–560 | 555–556 | same | p.add_argument("--weight-decay", type=float, default=0.1) | p.add_argument("--weight-decay", type=float, default=0.1) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:564 | anulm/training/train.py | 559–560 | 555–556 | sweep | ); 'make_optimizer' 451–478; 'main' defaults 559–560, sch |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:565 | anulm/training/train.py | 849–850 | 852–853 | same | lr = lr_at(step, args, start_step) | lr = lr_at(step, args, start_step) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:565 | anulm/training/train.py | 849–850 | 852–853 | sweep | 849–850, cli |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:565 | anulm/training/train.py | 871–875 | 874–878 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:565 | anulm/training/train.py | 871–875 | 874–878 | sweep | 849–850, clip and step 871–875. |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:566 | anulm/muon.py | 13–14 | 13–14 | sweep | - 'anulm/muon.py': module docstring 13–14; 'ze |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:568 | anulm/muon.py | 152–165 | 152–165 | sweep | 'build_optimizers' 144–182 (split 152–165). |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:569 | tests/test_model.py | 1610–1619 | 1627–1636 | same | def wsd_schedule_is_flat_then_decays_to_min_lr(): | def wsd_schedule_is_flat_then_decays_to_min_lr(): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:570 | tests/test_model.py | 1712 | 1729 | same | def batched_muon_matches_per_matrix_muon(): | def batched_muon_matches_per_matrix_muon(): |
| chapters/08-bytes-as-tokens.md:270 | tests/test_model.py | 1413–1431 | 1430–1448 | same | def byte_loader_inserts_eos_at_document_boundaries(): | def byte_loader_inserts_eos_at_document_boundaries(): |
| chapters/08-bytes-as-tokens.md:338 | anulm/training/train.py | 656–660 | 652–656 | same | # --- data ----------------------------------------------------------- | # --- data ----------------------------------------------------------- |
| chapters/08-bytes-as-tokens.md:348 | anulm/training/train.py | 903–905 | 906–908 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/08-bytes-as-tokens.md:357 | anulm/training/train.py | 660 | ? | CHECK | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 | no file named on the line and nothing quoted confirms it |
| chapters/08-bytes-as-tokens.md:409 | anulm/model.py | 1159–1167 | 1171–1179 | same | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) |
| chapters/08-bytes-as-tokens.md:435 | anulm/training/train.py | 692 | 688 | same | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") |
| chapters/08-bytes-as-tokens.md:492 | anulm/model.py | 1159–1167 | 1171–1179 | same | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) |
| chapters/08-bytes-as-tokens.md:496 | anulm/training/train.py | 656–660 | 652–656 | same | # --- data ----------------------------------------------------------- | # --- data ----------------------------------------------------------- |
| chapters/08-bytes-as-tokens.md:496 | anulm/training/train.py | 692 | 688 | same | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") |
| chapters/08-bytes-as-tokens.md:496 | anulm/training/train.py | 903–905 | 906–908 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/08-bytes-as-tokens.md:500 | tests/test_model.py | 1413–1431 | 1430–1448 | same | def byte_loader_inserts_eos_at_document_boundaries(): | def byte_loader_inserts_eos_at_document_boundaries(): |
| chapters/08-bytes-as-tokens.md:500 | tests/test_model.py | 1413–1431 | 1430–1448 | sweep | 1413–1431. |  |
| chapters/09-byte-level-bpe.md:401 | tests/test_model.py | 826–841 | 837–852 | same | def bpe_documents_get_eos_and_the_vocab_is_exact(): | def bpe_documents_get_eos_and_the_vocab_is_exact(): |
| chapters/09-byte-level-bpe.md:427 | tests/test_model.py | 1353–1363 | 1365–1375 | same | tok = BPE.train(FALLBACK * 8, 400, verbose=False) | tok = BPE.train(FALLBACK * 8, 400, verbose=False) |
| chapters/09-byte-level-bpe.md:474 | anulm/bpe.py | 347–355 | 348–356 | same | def _stats(tok, text): | def _stats(tok, text): |
| chapters/09-byte-level-bpe.md:213 | anulm/bpe.py | 120–121 | ? | CHECK | if -neg != pair_counts.get(pair, 0): | no file named on the line and nothing quoted confirms it |
| chapters/09-byte-level-bpe.md:425 | anulm/bpe.py | 1342–1346 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/09-byte-level-bpe.md:462 | anulm/bpe.py | 221–243 |  | reattributed | quoted code is in anulm/bpe.py, not None |  |
| chapters/09-byte-level-bpe.md:462 | anulm/bpe.py | 221–243 | 221–244 | same | # The format id the project wrote before it was renamed on 2026-09-18. | # The format id the project wrote before it was renamed on 2026-09-18. |
| chapters/09-byte-level-bpe.md:463 | anulm/bpe.py | 296–344 |  | reattributed | 'cmd_encode' is in anulm/bpe.py, not None |  |
| chapters/09-byte-level-bpe.md:463 | anulm/bpe.py | 296–344 | 297–345 | same | def cmd_encode(args): | def cmd_encode(args): |
| chapters/09-byte-level-bpe.md:537 | docs/RESULTS.md | 120–121 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/09-byte-level-bpe.md:599 | anulm/bpe.py | 64–83 | 64–83 | sweep | 64–83, 'BP |  |
| chapters/09-byte-level-bpe.md:600 | anulm/bpe.py | 221–243 | 221–244 | same | # The format id the project wrote before it was renamed on 2026-09-18. | # The format id the project wrote before it was renamed on 2026-09-18. |
| chapters/09-byte-level-bpe.md:601 | anulm/bpe.py | 296–344 | 297–345 | same | def cmd_encode(args): | def cmd_encode(args): |
| chapters/09-byte-level-bpe.md:601 | anulm/bpe.py | 296–344 | 297–345 | sweep | 296–344, '_s |  |
| chapters/09-byte-level-bpe.md:601 | anulm/bpe.py | 347–355 | 348–356 | same | def _stats(tok, text): | def _stats(tok, text): |
| chapters/09-byte-level-bpe.md:602 | tests/test_model.py | 826–841 | 837–852 | same | def bpe_documents_get_eos_and_the_vocab_is_exact(): | def bpe_documents_get_eos_and_the_vocab_is_exact(): |
| chapters/09-byte-level-bpe.md:603 | tests/test_model.py | 1339–1363 | 1352–1375 | same | def hf_tokenizer_conversion_is_exact(): | def hf_tokenizer_conversion_is_exact(): |
| chapters/09-byte-level-bpe.md:604 | tests/test_model.py | 1367 | 1379 | same | def bpe_files_load_under_both_format_ids(): | def bpe_files_load_under_both_format_ids(): |
| chapters/10-embeddings-and-the-output-table.md:57 | anulm/model.py | 820–837 | 827–844 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/10-embeddings-and-the-output-table.md:98 | anulm/model.py | 847–859 | 859–871 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/10-embeddings-and-the-output-table.md:124 | tests/test_model.py | 338–348 | 349–359 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/10-embeddings-and-the-output-table.md:192 | anulm/model.py | 884 | 896 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| chapters/10-embeddings-and-the-output-table.md:198 | anulm/model.py | 954–958 | 966–970 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/10-embeddings-and-the-output-table.md:208 | anulm/model.py | 980–983 | 992–995 | same | def _head(self, h): | def _head(self, h): |
| chapters/10-embeddings-and-the-output-table.md:312 | anulm/model.py | 383–404 | 390–411 | same | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: |
| chapters/10-embeddings-and-the-output-table.md:356 | tests/test_model.py | 1629–1636 | 1646–1653 | same | merges = [[97, 98], [256, 99], [257, 257], [100, 101]]      # ab, abc, | merges = [[97, 98], [256, 99], [257, 257], [100, 101]]      # ab, abc, |
| chapters/10-embeddings-and-the-output-table.md:372 | anulm/model.py | 947–952 | 959–964 | same | def _compose(self, W): | def _compose(self, W): |
| chapters/10-embeddings-and-the-output-table.md:385 | anulm/model.py | 959–978 | 971–990 | same | if self.training: | if self.training: |
| chapters/10-embeddings-and-the-output-table.md:419 | anulm/model.py | 985–1001 | 997–1013 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/10-embeddings-and-the-output-table.md:79 | anulm/model.py | 836 | 843 | same | self.lm_head.weight = self.embed_tokens.weight | self.lm_head.weight = self.embed_tokens.weight |
| chapters/10-embeddings-and-the-output-table.md:250 | anulm/training/train.py | 451–478 | 447–474 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/10-embeddings-and-the-output-table.md:252 | anulm/training/train.py | 559 |  | reattributed | quoted code is in anulm/training/train.py, not tools/windows/_v2_run.c |  |
| chapters/10-embeddings-and-the-output-table.md:252 | anulm/training/train.py | 559 | 555 | same | p.add_argument("--weight-decay", type=float, default=0.1) | p.add_argument("--weight-decay", type=float, default=0.1) |
| chapters/10-embeddings-and-the-output-table.md:340 | anulm/model.py | 390 | ? | CHECK | coef = [{i: 1.0} for i in range(256)] | no file named on the line and nothing quoted confirms it |
| chapters/10-embeddings-and-the-output-table.md:413 | tests/test_model.py | 1659 | 1676 | def | 'merge_tree_eval_cache_sees_optimizer_updates' | def merge_tree_eval_cache_sees_optimizer_updates(): |
| chapters/10-embeddings-and-the-output-table.md:519 | anulm/model.py | 383–404 | 390–411 | same | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: |
| chapters/10-embeddings-and-the-output-table.md:520 | anulm/model.py | 820–851 | 827–863 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/10-embeddings-and-the-output-table.md:520 | anulm/model.py | 853–859 | 865–871 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/10-embeddings-and-the-output-table.md:521 | anulm/model.py | 884 | 896 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| chapters/10-embeddings-and-the-output-table.md:521 | anulm/model.py | 947–952 | 959–964 | same | def _compose(self, W): | def _compose(self, W): |
| chapters/10-embeddings-and-the-output-table.md:521 | anulm/model.py | 954–972 | 966–984 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/10-embeddings-and-the-output-table.md:522 | anulm/model.py | 974–978 | 986–990 | same | def train(self, mode: bool = True): | def train(self, mode: bool = True): |
| chapters/10-embeddings-and-the-output-table.md:522 | anulm/model.py | 980–983 | 992–995 | same | def _head(self, h): | def _head(self, h): |
| chapters/10-embeddings-and-the-output-table.md:522 | anulm/model.py | 985–1001 | 997–1013 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/10-embeddings-and-the-output-table.md:523 | anulm/training/train.py | 451–478 | 447–474 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/10-embeddings-and-the-output-table.md:524 | tests/test_model.py | 338–348 | 349–359 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/10-embeddings-and-the-output-table.md:525 | tests/test_model.py | 1623 | 1640 | same | def merge_tree_composes_trains_and_bakes_losslessly(): | def merge_tree_composes_trains_and_bakes_losslessly(): |
| chapters/10-embeddings-and-the-output-table.md:526 | tests/test_model.py | 1659 | 1676 | same | def merge_tree_eval_cache_sees_optimizer_updates(): | def merge_tree_eval_cache_sees_optimizer_updates(): |
| chapters/11-attention.md:275 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/11-attention.md:292 | anulm/model.py | 423–442 | 430–449 | same | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): |
| chapters/11-attention.md:328 | anulm/model.py | 444–469 | 451–476 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/11-attention.md:360 | anulm/model.py | 470–487 | 477–494 | same | T = k.shape[2] | T = k.shape[2] |
| chapters/11-attention.md:396 | anulm/model.py | 358–374 | 365–381 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/11-attention.md:430 | tests/test_model.py | 642–664 | 653–675 | same | def sliding_window_limits_reach(): | def sliding_window_limits_reach(): |
| chapters/11-attention.md:467 | tests/test_model.py | 133–153 | 144–164 | same | def yarn_temperature_reaches_attention(): | def yarn_temperature_reaches_attention(): |
| chapters/11-attention.md:506 | anulm/model.py | 218–232 | 225–239 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/11-attention.md:323 | anulm/model.py | 291 | 298 | same | self.attn_mscale = self.table_mscale = 1.0 | self.attn_mscale = self.table_mscale = 1.0 |
| chapters/11-attention.md:389 | anulm/model.py | 380 | 387 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/11-attention.md:418 | anulm/model.py | 367 | ? | CHECK | if window is None and (start == 0 or S == 1): | no file named on the line and nothing quoted confirms it |
| chapters/11-attention.md:424 | anulm/model.py | 369–373 | ? | CHECK | q_pos = torch.arange(start, start + S, device=device)[:, None] | no file named on the line and nothing quoted confirms it |
| chapters/11-attention.md:465 | tests/test_model.py | 421–423 | ? | CHECK | def mla_shares_one_rope_key_across_heads(): | no file named on the line and nothing quoted confirms it |
| chapters/11-attention.md:502 | anulm/model.py | 458–460 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not tests/test_model.py |  |
| chapters/11-attention.md:502 | anulm/model.py | 458–460 | 465–467 | same | if self.use_qk_norm:                          # before RoPE, as in the | if self.use_qk_norm:                          # before RoPE, as in the |
| chapters/11-attention.md:523 | anulm/model.py | 224 | ? | CHECK | self.weight = nn.Parameter(torch.ones(dim)) | no file named on the line and nothing quoted confirms it |
| chapters/11-attention.md:524 | tests/test_model.py | 109 | 120 | same | def rope_preserves_norms(): | def rope_preserves_norms(): |
| chapters/11-attention.md:571 | anulm/model.py | 458 |  | reattributed | quoted code is in anulm/model.py, not None |  |
| chapters/11-attention.md:571 | anulm/model.py | 458 | 465 | same | if self.use_qk_norm:                          # before RoPE, as in the | if self.use_qk_norm:                          # before RoPE, as in the |
| chapters/11-attention.md:695 | tests/test_model.py | 550 | 561 | def | 'incremental_forward_matches_full_forward' | def incremental_forward_matches_full_forward(): |
| chapters/11-attention.md:695 | tests/test_model.py | 642 | 653 | same | def sliding_window_limits_reach(): | def sliding_window_limits_reach(): |
| chapters/11-attention.md:698 | tests/test_model.py | 46 | 57 | def | 'causal_mask_does_not_leak_future' | def causal_mask_does_not_leak_future(): |
| chapters/11-attention.md:770 | anulm/model.py | 218–232 | 225–239 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/11-attention.md:770 | anulm/model.py | 291 | 298 | same | self.attn_mscale = self.table_mscale = 1.0 | self.attn_mscale = self.table_mscale = 1.0 |
| chapters/11-attention.md:771 | anulm/model.py | 358–374 | 365–381 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/11-attention.md:771 | anulm/model.py | 380 | 387 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/11-attention.md:771 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/11-attention.md:772 | anulm/model.py | 423–442 | 430–449 | same | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): |
| chapters/11-attention.md:772 | anulm/model.py | 458–460 | 465–467 | same | if self.use_qk_norm:                          # before RoPE, as in the | if self.use_qk_norm:                          # before RoPE, as in the |
| chapters/11-attention.md:772 | anulm/model.py | 470–487 | 477–494 | same | T = k.shape[2] | T = k.shape[2] |
| chapters/11-attention.md:774 | tests/test_model.py | 46 | 57 | same | def causal_mask_does_not_leak_future(): | def causal_mask_does_not_leak_future(): |
| chapters/11-attention.md:774 | tests/test_model.py | 109 | 120 | same | def rope_preserves_norms(): | def rope_preserves_norms(): |
| chapters/11-attention.md:775 | tests/test_model.py | 133–153 | 144–164 | same | def yarn_temperature_reaches_attention(): | def yarn_temperature_reaches_attention(): |
| chapters/11-attention.md:776 | tests/test_model.py | 550 | 561 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/11-attention.md:776 | tests/test_model.py | 642–664 | 653–675 | same | def sliding_window_limits_reach(): | def sliding_window_limits_reach(): |
| chapters/12-causal-masking.md:71 | anulm/model.py | 132–134 | 138–140 | edited | if self.sliding_window is not None: | if self.sliding_window is not None: |
| chapters/12-causal-masking.md:71 | anulm/model.py | 133 | 139 | edited-quote | assert self.sliding_window >= 1 and self.max_window_layers >= 1, ( | check(self.sliding_window >= 1 and self.max_window_layers >= 1, |
| chapters/12-causal-masking.md:85 | anulm/model.py | 358–374 | 365–381 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/12-causal-masking.md:143 | anulm/model.py | 470–475 | 477–482 | same | T = k.shape[2] | T = k.shape[2] |
| chapters/12-causal-masking.md:187 | anulm/model.py | 1132–1139 | 1144–1151 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/12-causal-masking.md:225 | anulm/model.py | 805–807 | 812–814 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/12-causal-masking.md:290 | tests/test_model.py | 46–63 | 57–74 | same | def causal_mask_does_not_leak_future(): | def causal_mask_does_not_leak_future(): |
| chapters/12-causal-masking.md:317 | tests/test_model.py | 550–567 | 561–578 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/12-causal-masking.md:343 | tests/test_model.py | 642–664 | 653–675 | same | def sliding_window_limits_reach(): | def sliding_window_limits_reach(): |
| chapters/12-causal-masking.md:375 | tests/test_model.py | 668–675 | 679–686 | same | def sliding_window_agrees_with_cache(): | def sliding_window_agrees_with_cache(): |
| chapters/12-causal-masking.md:155 | anulm/bidirectional.py | 568 | ? | OUT OF FILE |  |  |
| chapters/12-causal-masking.md:201 | anulm/model.py | 1161 | 1173 | same | pending = nxt | pending = nxt |
| chapters/12-causal-masking.md:262 | anulm/model.py | 749 | 756 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/12-causal-masking.md:262 | anulm/model.py | 777 | 784 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/12-causal-masking.md:509 | anulm/model.py | 132–134 | 138–140 | edited | if self.sliding_window is not None: | if self.sliding_window is not None: |
| chapters/12-causal-masking.md:509 | anulm/model.py | 358–374 | 365–381 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/12-causal-masking.md:510 | anulm/model.py | 470–475 | 477–482 | same | T = k.shape[2] | T = k.shape[2] |
| chapters/12-causal-masking.md:510 | anulm/model.py | 568 | 575 | same | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star |
| chapters/12-causal-masking.md:511 | anulm/model.py | 805–807 | 812–814 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/12-causal-masking.md:511 | anulm/model.py | 805–807 | 812–814 | sweep | 805–807, 'An |  |
| chapters/12-causal-masking.md:511 | anulm/model.py | 1132–1139 | 1144–1151 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/12-causal-masking.md:511 | anulm/model.py | 1161 | 1173 | same | pending = nxt | pending = nxt |
| chapters/12-causal-masking.md:513 | tests/test_model.py | 46 | 57 | same | def causal_mask_does_not_leak_future(): | def causal_mask_does_not_leak_future(): |
| chapters/12-causal-masking.md:513 | tests/test_model.py | 550 | 561 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/12-causal-masking.md:513 | tests/test_model.py | 642 | 653 | same | def sliding_window_limits_reach(): | def sliding_window_limits_reach(): |
| chapters/12-causal-masking.md:513 | tests/test_model.py | 668 | 679 | same | def sliding_window_agrees_with_cache(): | def sliding_window_agrees_with_cache(): |
| chapters/13-multi-head-and-gqa.md:31 | anulm/model.py | 144–147 | ? | UNMAPPED | if self.attn == "gqa": |  |
| chapters/13-multi-head-and-gqa.md:31 | anulm/model.py | 144 | ? | EXCERPT | header range unresolved: AnuLMConfig.__post_init__ |  |
| chapters/13-multi-head-and-gqa.md:49 | anulm/model.py | 423–438 | 430–445 | same | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): |
| chapters/13-multi-head-and-gqa.md:111 | anulm/model.py | 444–461 | 451–468 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/13-multi-head-and-gqa.md:148 | anulm/model.py | 463–470 | 470–477 | same | if cache is not None: | if cache is not None: |
| chapters/13-multi-head-and-gqa.md:180 | tests/test_model.py | 571–579 | 582–590 | same | def cache_holds_what_it_should(): | def cache_holds_what_it_should(): |
| chapters/13-multi-head-and-gqa.md:197 | anulm/model.py | 476–487 | 483–494 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/13-multi-head-and-gqa.md:235 | anulm/model.py | 377–380 | 384–387 | same | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa |
| chapters/13-multi-head-and-gqa.md:248 | tests/test_model.py | 1585–1606 | 1602–1623 | same | def native_gqa_matches_repeat_interleave(): | def native_gqa_matches_repeat_interleave(): |
| chapters/13-multi-head-and-gqa.md:75 | anulm/model.py | 442 | ? | CHECK | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) | no file named on the line and nothing quoted confirms it |
| chapters/13-multi-head-and-gqa.md:244 | anulm/model.py | 516–522 | ? | CHECK | self.kv_b_proj = nn.Linear( | no file named on the line and nothing quoted confirms it |
| chapters/13-multi-head-and-gqa.md:311 | anulm/model.py | 749 | 756 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/13-multi-head-and-gqa.md:311 | anulm/model.py | 777 | 784 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/13-multi-head-and-gqa.md:394 | anulm/model.py | 144–147 | ? | UNMAPPED | if self.attn == "gqa": |  |
| chapters/13-multi-head-and-gqa.md:394 | anulm/model.py | 377–380 | 384–387 | same | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa |
| chapters/13-multi-head-and-gqa.md:395 | anulm/model.py | 423–438 | 430–445 | same | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): |
| chapters/13-multi-head-and-gqa.md:395 | anulm/model.py | 442 | 449 | same | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) |
| chapters/13-multi-head-and-gqa.md:395 | anulm/model.py | 444–487 | 451–494 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/13-multi-head-and-gqa.md:396 | anulm/training/train.py | 641–647 | 637–643 | same | p.add_argument("--native-gqa", action="store_true", | p.add_argument("--native-gqa", action="store_true", |
| chapters/13-multi-head-and-gqa.md:397 | tests/test_model.py | 571 | 582 | same | def cache_holds_what_it_should(): | def cache_holds_what_it_should(): |
| chapters/13-multi-head-and-gqa.md:397 | tests/test_model.py | 1585 | 1602 | same | def native_gqa_matches_repeat_interleave(): | def native_gqa_matches_repeat_interleave(): |
| chapters/14-rotary-positions.md:115 | anulm/model.py | 297–302 | 304–309 | same | @staticmethod | @staticmethod |
| chapters/14-rotary-positions.md:162 | anulm/model.py | 275–280 | 282–287 | same | def __init__(self, cfg: AnuLMConfig, dim: int): | def __init__(self, cfg: AnuLMConfig, dim: int): |
| chapters/14-rotary-positions.md:176 | anulm/model.py | 317–330 | 324–337 | same | def _grow(self, seq_len: int, device, dtype): | def _grow(self, seq_len: int, device, dtype): |
| chapters/14-rotary-positions.md:201 | anulm/model.py | 332–336 | 339–343 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| chapters/14-rotary-positions.md:224 | tests/test_model.py | 625–634 | 636–645 | same | def rotary_cache_tracks_dtype_and_positions(): | def rotary_cache_tracks_dtype_and_positions(): |
| chapters/14-rotary-positions.md:243 | anulm/model.py | 339–343 | 346–350 | same | def rotate_half(x): | def rotate_half(x): |
| chapters/14-rotary-positions.md:251 | anulm/model.py | 346–355 | 353–362 | same | def apply_rope(q, k, cos, sin): | def apply_rope(q, k, cos, sin): |
| chapters/14-rotary-positions.md:306 | tests/test_model.py | 90–105 | 101–116 | same | def rope_is_relative(): | def rope_is_relative(): |
| chapters/14-rotary-positions.md:330 | tests/test_model.py | 109–116 | 120–127 | same | def rope_preserves_norms(): | def rope_preserves_norms(): |
| chapters/14-rotary-positions.md:157 | anulm/model.py | 830–832 |  | reattributed | 'AnuLM.__init__' is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/14-rotary-positions.md:157 | anulm/model.py | 830–832 | 837–839 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/14-rotary-positions.md:158 | anulm/model.py | 887 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/14-rotary-positions.md:158 | anulm/model.py | 887 | 899 | same | cos, sin = self.rotary(S, x.device, x.dtype, start=start) | cos, sin = self.rotary(S, x.device, x.dtype, start=start) |
| chapters/14-rotary-positions.md:159 | docs/RESULTS.md | 282–283 | ? | CHECK | moves ±0.05 in 50 steps against sigmoid scores spanning (0,1), so the  | no file named on the line and nothing quoted confirms it |
| chapters/14-rotary-positions.md:354 | anulm/model.py | 749 | 756 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/14-rotary-positions.md:354 | anulm/model.py | 777 | 784 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/14-rotary-positions.md:457 | anulm/model.py | 271–336 | 278–343 | same | class Rotary(nn.Module): | class Rotary(nn.Module): |
| chapters/14-rotary-positions.md:457 | anulm/model.py | 339–343 | 346–350 | same | def rotate_half(x): | def rotate_half(x): |
| chapters/14-rotary-positions.md:457 | anulm/model.py | 346–355 | 353–362 | same | def apply_rope(q, k, cos, sin): | def apply_rope(q, k, cos, sin): |
| chapters/14-rotary-positions.md:458 | anulm/model.py | 749 | 756 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/14-rotary-positions.md:458 | anulm/model.py | 777 | 784 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/14-rotary-positions.md:458 | anulm/model.py | 830–832 | 837–839 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/14-rotary-positions.md:458 | anulm/model.py | 887 | 899 | same | cos, sin = self.rotary(S, x.device, x.dtype, start=start) | cos, sin = self.rotary(S, x.device, x.dtype, start=start) |
| chapters/14-rotary-positions.md:459 | tests/test_model.py | 90 | 101 | same | def rope_is_relative(): | def rope_is_relative(): |
| chapters/14-rotary-positions.md:459 | tests/test_model.py | 109 | 120 | same | def rope_preserves_norms(): | def rope_preserves_norms(): |
| chapters/14-rotary-positions.md:459 | tests/test_model.py | 625 | 636 | same | def rotary_cache_tracks_dtype_and_positions(): | def rotary_cache_tracks_dtype_and_positions(): |
| chapters/15-normalisation-and-the-residual-stream.md:58 | anulm/multimodal/speech_encoder.py | 80–83 | 79–82 | same | hidden = hidden or max(out_dim, d) | hidden = hidden or max(out_dim, d) |
| chapters/15-normalisation-and-the-residual-stream.md:73 | anulm/model.py | 218–232 | 225–239 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/15-normalisation-and-the-residual-stream.md:135 | anulm/model.py | 795–804 | 802–811 | same | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): |
| chapters/15-normalisation-and-the-residual-stream.md:152 | anulm/model.py | 809–817 | 816–824 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/15-normalisation-and-the-residual-stream.md:174 | anulm/model.py | 832–834 | 839–841 | same | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range |
| chapters/15-normalisation-and-the-residual-stream.md:182 | anulm/model.py | 900 | 912 | same | x = self.norm(x) | x = self.norm(x) |
| chapters/15-normalisation-and-the-residual-stream.md:269 | anulm/model.py | 847–851 | 859–863 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/15-normalisation-and-the-residual-stream.md:279 | anulm/model.py | 853–859 | 865–871 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/15-normalisation-and-the-residual-stream.md:323 | tests/test_model.py | 338–348 | 349–359 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/15-normalisation-and-the-residual-stream.md:421 | anulm/model.py | 218–232 | 225–239 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/15-normalisation-and-the-residual-stream.md:421 | anulm/model.py | 795–804 | 802–811 | same | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): |
| chapters/15-normalisation-and-the-residual-stream.md:422 | anulm/model.py | 809–817 | 816–824 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/15-normalisation-and-the-residual-stream.md:422 | anulm/model.py | 809–817 | 816–824 | sweep | 809–817, 'An |  |
| chapters/15-normalisation-and-the-residual-stream.md:422 | anulm/model.py | 832–834 | 839–841 | same | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range |
| chapters/15-normalisation-and-the-residual-stream.md:422 | anulm/model.py | 847–851 | 859–863 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/15-normalisation-and-the-residual-stream.md:422 | anulm/model.py | 853–859 | 865–871 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/15-normalisation-and-the-residual-stream.md:423 | anulm/model.py | 900 | 912 | same | x = self.norm(x) | x = self.norm(x) |
| chapters/15-normalisation-and-the-residual-stream.md:424 | anulm/multimodal/speech_encoder.py | 80–83 | 79–82 | same | hidden = hidden or max(out_dim, d) | hidden = hidden or max(out_dim, d) |
| chapters/15-normalisation-and-the-residual-stream.md:424 | anulm/multimodal/speech_encoder.py | 80–83 | 79–82 | sweep | - 'anulm/multimodal/speech_encoder.py' 80–83 and |  |
| chapters/15-normalisation-and-the-residual-stream.md:424 | anulm/multimodal/vision_encoder.py | 124 | 124 | sweep | –83 and 'anulm/multimodal/vision_encoder.py' 124 (Lay |  |
| chapters/15-normalisation-and-the-residual-stream.md:425 | tests/test_model.py | 338 | 349 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/16-swiglu.md:90 | anulm/model.py | 235–246 | 242–253 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/16-swiglu.md:168 | anulm/model.py | 184–193 | 191–200 | same | cfg = dict( | cfg = dict( |
| chapters/16-swiglu.md:230 | anulm/model.py | 802–804 | 809–811 | same | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in both mod | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in both mod |
| chapters/16-swiglu.md:242 | anulm/model.py | 681–687 | 688–694 | same | self.experts = nn.ModuleList( | self.experts = nn.ModuleList( |
| chapters/16-swiglu.md:260 | anulm/model.py | 848–851 | 860–863 | same | # nanoGPT's scaled init for the residual output projections. | # nanoGPT's scaled init for the residual output projections. |
| chapters/16-swiglu.md:222 | anulm/model.py | 159 | 166 | same | intermediate_size=768,         # hidden*2 | intermediate_size=768,         # hidden*2 |
| chapters/16-swiglu.md:223 | anulm/model.py | 206 | 213 | same | intermediate_size=1536,        # hidden*4 | intermediate_size=1536,        # hidden*4 |
| chapters/16-swiglu.md:372 | tests/test_model.py | 1271 | 1285 | def | 'transformers_wrapper_matches_the_native_model' | def transformers_wrapper_matches_the_native_model(): |
| chapters/16-swiglu.md:376 | tests/test_model.py | 325 | 336 | def | 'all_parameters_receive_gradients' | def all_parameters_receive_gradients(): |
| chapters/16-swiglu.md:431 | anulm/model.py | 159 | 166 | same | intermediate_size=768,         # hidden*2 | intermediate_size=768,         # hidden*2 |
| chapters/16-swiglu.md:431 | anulm/model.py | 184–193 | 191–200 | same | cfg = dict( | cfg = dict( |
| chapters/16-swiglu.md:432 | anulm/model.py | 206 | 213 | same | intermediate_size=1536,        # hidden*4 | intermediate_size=1536,        # hidden*4 |
| chapters/16-swiglu.md:432 | anulm/model.py | 235–246 | 242–253 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/16-swiglu.md:432 | anulm/model.py | 681–687 | 688–694 | same | self.experts = nn.ModuleList( | self.experts = nn.ModuleList( |
| chapters/16-swiglu.md:433 | anulm/model.py | 802–804 | 809–811 | same | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in both mod | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in both mod |
| chapters/16-swiglu.md:433 | anulm/model.py | 848–851 | 860–863 | same | # nanoGPT's scaled init for the residual output projections. | # nanoGPT's scaled init for the residual output projections. |
| chapters/16-swiglu.md:434 | tests/test_model.py | 325 | 336 | same | def all_parameters_receive_gradients(): | def all_parameters_receive_gradients(): |
| chapters/16-swiglu.md:435 | tests/test_model.py | 1271 | 1285 | same | def transformers_wrapper_matches_the_native_model(): | def transformers_wrapper_matches_the_native_model(): |
| chapters/17-router.md:91 | anulm/model.py | 587–602 | 594–609 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/17-router.md:133 | anulm/model.py | 604–615 | 611–622 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/17-router.md:183 | anulm/model.py | 617–638 | 624–645 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/17-router.md:210 | tests/test_model.py | 222–237 | 233–248 | same | def router_weights_come_from_unbiased_scores(): | def router_weights_come_from_unbiased_scores(): |
| chapters/17-router.md:259 | anulm/model.py | 622–631 | 629–638 | same | if self.n_group > 1: | if self.n_group > 1: |
| chapters/17-router.md:305 | tests/test_model.py | 207–218 | 218–229 | same | def router_respects_group_limits(): | def router_respects_group_limits(): |
| chapters/17-router.md:325 | anulm/model.py | 138–142 | 144–148 | edited | per_group = self.num_experts // self.n_group | per_group = self.num_experts // self.n_group |
| chapters/17-router.md:325 | anulm/model.py | 139 | 145 | edited-quote | assert self.num_experts_per_tok <= per_group * self.topk_group, ( | check(self.num_experts_per_tok <= per_group * self.topk_group, |
| chapters/17-router.md:325 | anulm/model.py | 141 | 147 | edited-quote | f"experts reachable through {self.topk_group} of {self.n_group} groups | f"experts reachable through {self.topk_group} of {self.n_group} groups |
| chapters/17-router.md:325 | anulm/model.py | 142 | 148 | edited-quote | ) | check(self.topk_group <= self.n_group, |
| chapters/17-router.md:14 | anulm/model.py | 578–665 | 585–672 | same | class Router(nn.Module): | class Router(nn.Module): |
| chapters/17-router.md:335 | tests/test_model.py | 412 | 423 | def | 'config_rejects_unreachable_topk' | def config_rejects_unreachable_topk(): |
| chapters/17-router.md:336 | tests/test_model.py | 594–603 | ? | CHECK | """Greedy decoding with and without the cache must agree token-for-tok | no file named on the line and nothing quoted confirms it |
| chapters/17-router.md:475 | anulm/model.py | 138–142 | 144–148 | edited | per_group = self.num_experts // self.n_group | per_group = self.num_experts // self.n_group |
| chapters/17-router.md:475 | anulm/model.py | 587–602 | 594–609 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/17-router.md:476 | anulm/model.py | 604–615 | 611–622 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/17-router.md:476 | anulm/model.py | 617–638 | 624–645 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/17-router.md:476 | anulm/model.py | 622–631 | 629–638 | same | if self.n_group > 1: | if self.n_group > 1: |
| chapters/17-router.md:476 | anulm/model.py | 622–631 | 629–638 | sweep | ard' 604–615, 'Router._gate' 617–638 (groups 622–631). |  |
| chapters/17-router.md:477 | tests/test_model.py | 207–218 | 218–229 | same | def router_respects_group_limits(): | def router_respects_group_limits(): |
| chapters/17-router.md:478 | tests/test_model.py | 222–237 | 233–248 | same | def router_weights_come_from_unbiased_scores(): | def router_weights_come_from_unbiased_scores(): |
| chapters/17-router.md:479 | tests/test_model.py | 412 | 423 | same | def config_rejects_unreachable_topk(): | def config_rejects_unreachable_topk(): |
| chapters/17-router.md:481 | sarvam/sarvam-105b/modeling_sarvam_moe.py | 276–277 | 276–277 | sweep | 'sarvam/sarvam-105b/modeling_sarvam_moe.py' 276–277, |  |
| chapters/17-router.md:482 | sarvam/sarvam-105b/modeling_sarvam_moe.py | 305–313 | 305–313 | sweep | 305–313: the |  |
| chapters/18-dispatch-and-balance.md:67 | anulm/model.py | 702–720 | 709–727 | same | def _route_sparse(self, x_flat, topk_idx, topk_w): | def _route_sparse(self, x_flat, topk_idx, topk_w): |
| chapters/18-dispatch-and-balance.md:125 | anulm/model.py | 722–731 | 729–738 | same | def _route_dense(self, x_flat, topk_idx, topk_w): | def _route_dense(self, x_flat, topk_idx, topk_w): |
| chapters/18-dispatch-and-balance.md:144 | anulm/model.py | 768–783 | 775–790 | same | N, D = x_flat.shape | N, D = x_flat.shape |
| chapters/18-dispatch-and-balance.md:175 | anulm/model.py | 743–752 | 750–759 | same | # Keyed on the parameters' in-place version counters, so a | # Keyed on the parameters' in-place version counters, so a |
| chapters/18-dispatch-and-balance.md:217 | tests/test_model.py | 683–700 | 694–711 | same | def grouped_moe_matches_sparse_under_bf16(): | def grouped_moe_matches_sparse_under_bf16(): |
| chapters/18-dispatch-and-balance.md:217 | tests/test_model.py | 688 | 699 | edited-quote | return | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/18-dispatch-and-balance.md:267 | anulm/model.py | 640–645 | 647–652 | same | aux = None | aux = None |
| chapters/18-dispatch-and-balance.md:280 | anulm/training/train.py | 873–875 | 876–878 | same | opt.step() | opt.step() |
| chapters/18-dispatch-and-balance.md:288 | anulm/model.py | 1069–1089 | 1081–1101 | same | ratios = [] | ratios = [] |
| chapters/18-dispatch-and-balance.md:408 | anulm/hf/export_hf.py | 322–332 | 328–338 | same | # Small tensors stay float32 on disk AND must be loaded that way: the | # Small tensors stay float32 on disk AND must be loaded that way: the |
| chapters/18-dispatch-and-balance.md:432 | anulm/model.py | 646–663 | 653–670 | same | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: |
| chapters/18-dispatch-and-balance.md:481 | tests/test_model.py | 763–779 | 774–790 | same | def seq_balance_loss_matches_the_formula(): | def seq_balance_loss_matches_the_formula(): |
| chapters/18-dispatch-and-balance.md:119 | tests/test_model.py | 495 | 506 | same | def sparse_moe_breaks_the_graph_as_documented(): | def sparse_moe_breaks_the_graph_as_documented(): |
| chapters/18-dispatch-and-balance.md:214 | tests/test_model.py | 463 | 474 | def | 'dense_and_sparse_moe_agree' | def dense_and_sparse_moe_agree(): |
| chapters/18-dispatch-and-balance.md:259 | docs/RESULTS.md | 713–714 | ? | CHECK | Balancing got harder: 24 experts finish at 1.76× against 1.42× — the b | no file named on the line and nothing quoted confirms it |
| chapters/18-dispatch-and-balance.md:341 | tests/test_model.py | 271 | 282 | same | def bias_update_recovers_from_collapse(): | def bias_update_recovers_from_collapse(): |
| chapters/18-dispatch-and-balance.md:403 | docs/DEVELOPING.md | 295 | 302 | same | changes which experts a token is routed to, and the output collapses i | changes which experts a token is routed to, and the output collapses i |
| chapters/18-dispatch-and-balance.md:507 | tests/test_model.py | 739 | 750 | same | def router_regularisers_are_off_by_default_and_work_when_on(): | def router_regularisers_are_off_by_default_and_work_when_on(): |
| chapters/18-dispatch-and-balance.md:533 | tests/test_model.py | 310 | 321 | def | 'expert_bias_is_not_trainable' | def expert_bias_is_not_trainable(): |
| chapters/18-dispatch-and-balance.md:590 | anulm/model.py | 640–645 | 647–652 | same | aux = None | aux = None |
| chapters/18-dispatch-and-balance.md:590 | anulm/model.py | 646–663 | 653–670 | same | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: |
| chapters/18-dispatch-and-balance.md:590 | anulm/model.py | 702–720 | 709–727 | same | def _route_sparse(self, x_flat, topk_idx, topk_w): | def _route_sparse(self, x_flat, topk_idx, topk_w): |
| chapters/18-dispatch-and-balance.md:591 | anulm/model.py | 722–731 | 729–738 | same | def _route_dense(self, x_flat, topk_idx, topk_w): | def _route_dense(self, x_flat, topk_idx, topk_w): |
| chapters/18-dispatch-and-balance.md:591 | anulm/model.py | 743–752 | 750–759 | same | # Keyed on the parameters' in-place version counters, so a | # Keyed on the parameters' in-place version counters, so a |
| chapters/18-dispatch-and-balance.md:592 | anulm/model.py | 768–783 | 775–790 | same | N, D = x_flat.shape | N, D = x_flat.shape |
| chapters/18-dispatch-and-balance.md:592 | anulm/model.py | 768–783 | 775–790 | sweep | 768–783; 'An |  |
| chapters/18-dispatch-and-balance.md:592 | anulm/model.py | 1069–1089 | 1081–1101 | same | ratios = [] | ratios = [] |
| chapters/18-dispatch-and-balance.md:593 | anulm/training/train.py | 873–875 | 876–878 | same | opt.step() | opt.step() |
| chapters/18-dispatch-and-balance.md:593 | anulm/hf/export_hf.py | 322–332 | 328–338 | same | # Small tensors stay float32 on disk AND must be loaded that way: the | # Small tensors stay float32 on disk AND must be loaded that way: the |
| chapters/18-dispatch-and-balance.md:594 | anulm/hf/modeling_anulm.py | 120–133 | 120–133 | sweep | 'anulm/hf/modeling_anulm.py' 120–133; 'do |  |
| chapters/18-dispatch-and-balance.md:594 | docs/DEVELOPING.md | 295 | 302 | same | changes which experts a token is routed to, and the output collapses i | changes which experts a token is routed to, and the output collapses i |
| chapters/18-dispatch-and-balance.md:595 | tests/test_model.py | 169 | 180 | same | def moe_matches_naive_per_token_loop(): | def moe_matches_naive_per_token_loop(): |
| chapters/18-dispatch-and-balance.md:596 | tests/test_model.py | 271 | 282 | same | def bias_update_recovers_from_collapse(): | def bias_update_recovers_from_collapse(): |
| chapters/18-dispatch-and-balance.md:596 | tests/test_model.py | 310 | 321 | same | def expert_bias_is_not_trainable(): | def expert_bias_is_not_trainable(): |
| chapters/18-dispatch-and-balance.md:597 | tests/test_model.py | 463 | 474 | same | def dense_and_sparse_moe_agree(): | def dense_and_sparse_moe_agree(): |
| chapters/18-dispatch-and-balance.md:598 | tests/test_model.py | 495 | 506 | same | def sparse_moe_breaks_the_graph_as_documented(): | def sparse_moe_breaks_the_graph_as_documented(): |
| chapters/18-dispatch-and-balance.md:598 | tests/test_model.py | 495 | 506 | sweep | 495, 'gr |  |
| chapters/18-dispatch-and-balance.md:598 | tests/test_model.py | 683–700 | 694–711 | same | def grouped_moe_matches_sparse_under_bf16(): | def grouped_moe_matches_sparse_under_bf16(): |
| chapters/18-dispatch-and-balance.md:599 | tests/test_model.py | 739 | 750 | same | def router_regularisers_are_off_by_default_and_work_when_on(): | def router_regularisers_are_off_by_default_and_work_when_on(): |
| chapters/18-dispatch-and-balance.md:600 | tests/test_model.py | 763–779 | 774–790 | same | def seq_balance_loss_matches_the_formula(): | def seq_balance_loss_matches_the_formula(): |
| chapters/19-latent-attention.md:70 | anulm/model.py | 509–526 | 516–533 | same | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, |
| chapters/19-latent-attention.md:107 | anulm/model.py | 144–149 | 150–156 | same | if self.attn == "gqa": | if self.attn == "gqa": |
| chapters/19-latent-attention.md:107 | anulm/model.py | 150–156 |  | EXCERPT | assert self.n_head % self.n_kv_head == 0 | quoted line not in the new range |
| chapters/19-latent-attention.md:126 | anulm/model.py | 532–546 | 539–553 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/19-latent-attention.md:153 | anulm/model.py | 557–571 | 564–578 | same | T = compressed.shape[1] | T = compressed.shape[1] |
| chapters/19-latent-attention.md:225 | anulm/model.py | 548–556 | 555–563 | same | if cache is not None: | if cache is not None: |
| chapters/19-latent-attention.md:264 | tests/test_model.py | 421–433 | 432–444 | same | def mla_shares_one_rope_key_across_heads(): | def mla_shares_one_rope_key_across_heads(): |
| chapters/19-latent-attention.md:15 | anulm/model.py | 494–571 | 501–578 | same | class MLAttention(nn.Module): | class MLAttention(nn.Module): |
| chapters/19-latent-attention.md:62 | anulm/model.py | 543–544 | ? | CHECK | # RoPE touches only the rope slice; the latent stays position-free -- | no file named on the line and nothing quoted confirms it |
| chapters/19-latent-attention.md:120 | anulm/model.py | 830 | 837 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/19-latent-attention.md:413 | anulm/model.py | 830 |  | reattributed | quoted code is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/19-latent-attention.md:413 | anulm/model.py | 830 | 837 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/19-latent-attention.md:470 | anulm/model.py | 144–149 | 150–156 | same | if self.attn == "gqa": | if self.attn == "gqa": |
| chapters/19-latent-attention.md:471 | anulm/model.py | 509–526 | 516–533 | same | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, |
| chapters/19-latent-attention.md:471 | anulm/model.py | 509–526 | 516–533 | sweep | 509–526, 'ML |  |
| chapters/19-latent-attention.md:471 | anulm/model.py | 532–546 | 539–553 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/19-latent-attention.md:471 | anulm/model.py | 548–556 | 555–563 | same | if cache is not None: | if cache is not None: |
| chapters/19-latent-attention.md:471 | anulm/model.py | 557–571 | 564–578 | same | T = compressed.shape[1] | T = compressed.shape[1] |
| chapters/19-latent-attention.md:472 | anulm/model.py | 830 | ? | CHECK | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | no file named on the line and nothing quoted confirms it |
| chapters/19-latent-attention.md:473 | tests/test_model.py | 421–433 | 432–444 | same | def mla_shares_one_rope_key_across_heads(): | def mla_shares_one_rope_key_across_heads(): |
| chapters/19-latent-attention.md:477 | sarvam/sarvam-105b/modeling_sarvam_moe.py | 486–499 | 486–499 | sweep | 486–499 (opt |  |
| chapters/20-the-whole-model.md:105 | anulm/model.py | 826–851 | 833–863 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/20-the-whole-model.md:105 | anulm/model.py | 833–863 |  | EXCERPT | assert cfg.merge_tree in ("in", "out", "both"), cfg.merge_tree | quoted line not in the new range |
| chapters/20-the-whole-model.md:105 | anulm/model.py | 833–863 |  | EXCERPT | merges = _json.load(open(cfg.tokenizer_path, encoding="utf-8"))["merge | quoted line not in the new range |
| chapters/20-the-whole-model.md:146 | anulm/model.py | 879–887 | 891–899 | same | if embeds is not None: | if embeds is not None: |
| chapters/20-the-whole-model.md:146 | anulm/model.py | 885 | 897 | edited-quote | assert start + S <= self.cfg.block_size, \ | if start + S > self.cfg.block_size:      # an error, not an assert: py |
| chapters/20-the-whole-model.md:146 | anulm/model.py | 886 | 898 | edited-quote | f"positions up to {start + S} exceed block_size {self.cfg.block_size}" | raise ValueError(f"positions up to {start + S} exceed block_size {self |
| chapters/20-the-whole-model.md:165 | anulm/model.py | 317–336 | 324–343 | same | def _grow(self, seq_len: int, device, dtype): | def _grow(self, seq_len: int, device, dtype): |
| chapters/20-the-whole-model.md:206 | anulm/model.py | 888–899 | 900–911 | same | aux_total = None | aux_total = None |
| chapters/20-the-whole-model.md:226 | anulm/model.py | 809–817 | 816–824 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/20-the-whole-model.md:249 | anulm/model.py | 451–487 | 458–494 | same | qkv = self.query_key_value(x) | qkv = self.query_key_value(x) |
| chapters/20-the-whole-model.md:304 | anulm/model.py | 235–246 | 242–253 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/20-the-whole-model.md:324 | anulm/model.py | 689–700 | 696–707 | same | def forward(self, x):                                     # (B, S, D) | def forward(self, x):                                     # (B, S, D) |
| chapters/20-the-whole-model.md:344 | anulm/model.py | 604–615 | 611–622 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/20-the-whole-model.md:370 | anulm/model.py | 617–665 | 624–672 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/20-the-whole-model.md:418 | anulm/model.py | 900–916 | 912–928 | same | x = self.norm(x) | x = self.norm(x) |
| chapters/20-the-whole-model.md:449 | anulm/training/train.py | 241–254 | 237–250 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/20-the-whole-model.md:492 | anulm/model.py | 1140–1157 | 1152–1169 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/20-the-whole-model.md:555 | anulm/model.py | 1092–1101 | 1104–1113 | same | def num_params(self) -> tuple[int, int]: | def num_params(self) -> tuple[int, int]: |
| chapters/20-the-whole-model.md:62 | tests/test_model.py | 33–38 | 44–49 | same | def tiny(**kw) -> AnuLMConfig: | def tiny(**kw) -> AnuLMConfig: |
| chapters/20-the-whole-model.md:284 | anulm/model.py | 380 | 387 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/20-the-whole-model.md:397 | tests/test_model.py | 463 | 474 | def | 'dense_and_sparse_moe_agree' | def dense_and_sparse_moe_agree(): |
| chapters/20-the-whole-model.md:398 | tests/test_model.py | 683 | 694 | def | 'grouped_moe_matches_sparse_under_bf16' | def grouped_moe_matches_sparse_under_bf16(): |
| chapters/20-the-whole-model.md:476 | tests/test_model.py | 338 | 349 | def | 'initial_loss_is_ln_vocab' | def initial_loss_is_ln_vocab(): |
| chapters/20-the-whole-model.md:626 | anulm/model.py | 463–469 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not anulm/inference/api.py |  |
| chapters/20-the-whole-model.md:626 | anulm/model.py | 463–469 | 470–476 | same | if cache is not None: | if cache is not None: |
| chapters/20-the-whole-model.md:634 | tests/test_model.py | 399 | 410 | same | def param_count_matches_analysis(): | def param_count_matches_analysis(): |
| chapters/20-the-whole-model.md:640 | tests/test_model.py | 325 | 336 | def | 'all_parameters_receive_gradients' | def all_parameters_receive_gradients(): |
| chapters/20-the-whole-model.md:641 | tests/test_model.py | 338 | 349 | def | 'initial_loss_is_ln_vocab' | def initial_loss_is_ln_vocab(): |
| chapters/20-the-whole-model.md:734 | anulm/model.py | 218–232 | 225–239 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/20-the-whole-model.md:734 | anulm/model.py | 235–246 | 242–253 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/20-the-whole-model.md:734 | anulm/model.py | 317–336 | 324–343 | same | def _grow(self, seq_len: int, device, dtype): | def _grow(self, seq_len: int, device, dtype): |
| chapters/20-the-whole-model.md:735 | anulm/model.py | 380 | 387 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/20-the-whole-model.md:736 | anulm/model.py | 444–487 | 451–494 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/20-the-whole-model.md:736 | anulm/model.py | 604–615 | 611–622 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/20-the-whole-model.md:737 | anulm/model.py | 617–665 | 624–672 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/20-the-whole-model.md:737 | anulm/model.py | 689–700 | 696–707 | same | def forward(self, x):                                     # (B, S, D) | def forward(self, x):                                     # (B, S, D) |
| chapters/20-the-whole-model.md:737 | anulm/model.py | 809–817 | 816–824 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/20-the-whole-model.md:738 | anulm/model.py | 826–851 | 833–863 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/20-the-whole-model.md:738 | anulm/model.py | 865–916 | 877–928 | same | def forward(self, idx, targets=None, cache: Optional[list] = None, sta | def forward(self, idx, targets=None, cache: Optional[list] = None, sta |
| chapters/20-the-whole-model.md:739 | anulm/model.py | 1092–1101 | 1104–1113 | same | def num_params(self) -> tuple[int, int]: | def num_params(self) -> tuple[int, int]: |
| chapters/20-the-whole-model.md:739 | anulm/model.py | 1104–1168 | 1116–1180 | same | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/20-the-whole-model.md:740 | anulm/training/train.py | 241–254 | 237–250 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/20-the-whole-model.md:742 | tests/test_model.py | 33–38 | 44–49 | same | def tiny(**kw) -> AnuLMConfig: | def tiny(**kw) -> AnuLMConfig: |
| chapters/20-the-whole-model.md:743 | tests/test_model.py | 325 | 336 | same | def all_parameters_receive_gradients(): | def all_parameters_receive_gradients(): |
| chapters/20-the-whole-model.md:743 | tests/test_model.py | 338 | 349 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/20-the-whole-model.md:744 | tests/test_model.py | 399 | 410 | same | def param_count_matches_analysis(): | def param_count_matches_analysis(): |
| chapters/20-the-whole-model.md:745 | tests/test_model.py | 463 | 474 | same | def dense_and_sparse_moe_agree(): | def dense_and_sparse_moe_agree(): |
| chapters/20-the-whole-model.md:745 | tests/test_model.py | 683 | 694 | same | def grouped_moe_matches_sparse_under_bf16(): | def grouped_moe_matches_sparse_under_bf16(): |
| chapters/21-data.md:138 | anulm/training/train.py | 663–676 | 659–672 | same | memmapped = bool(args.data and args.data.endswith("meta.json")) | memmapped = bool(args.data and args.data.endswith("meta.json")) |
| chapters/21-data.md:176 | anulm/training/train.py | 241–254 | 237–250 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/21-data.md:393 | anulm/training/train.py | 693–699 | 689–695 | same | if memmapped or (split_encoded and val_data is not None): | if memmapped or (split_encoded and val_data is not None): |
| chapters/21-data.md:411 | anulm/training/train.py | 206–220 | 202–216 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/21-data.md:59 | experiments/build_v2.py | 119 | ? | CHECK | if n_tr >= budget: | no file named on the line and nothing quoted confirms it |
| chapters/21-data.md:82 | experiments/build_v2.py | 42 |  | reattributed | quoted code is in experiments/build_v2.py, not None |  |
| chapters/21-data.md:353 | anulm/training/train.py | 594 |  | reattributed | quoted code is in anulm/training/train.py, not None |  |
| chapters/21-data.md:353 | anulm/training/train.py | 594 | 590 | same | p.add_argument("--sample-with-replacement", action="store_true", | p.add_argument("--sample-with-replacement", action="store_true", |
| chapters/21-data.md:384 | anulm/model.py | 358 |  | reattributed | '_attn_mask' is defined there |  |
| chapters/21-data.md:384 | anulm/model.py | 358 | 365 | def | '_attn_mask' | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/21-data.md:470 | tests/test_model.py | 1732 |  | reattributed | 'memmap_batches_match_in_memory_batches' is defined there |  |
| chapters/21-data.md:470 | tests/test_model.py | 1732 | 1749 | def | 'memmap_batches_match_in_memory_batches' | def memmap_batches_match_in_memory_batches(): |
| chapters/21-data.md:472 | tests/test_model.py | 787 |  | reattributed | 'window_sampler_covers_every_window_once_per_epoch' is defined there |  |
| chapters/21-data.md:472 | tests/test_model.py | 787 | 798 | def | 'window_sampler_covers_every_window_once_per_epoch' | def window_sampler_covers_every_window_once_per_epoch(): |
| chapters/21-data.md:473 | tests/test_model.py | 812 |  | reattributed | 'window_sampler_resumes_exactly' is defined there |  |
| chapters/21-data.md:473 | tests/test_model.py | 812 | 823 | def | 'window_sampler_resumes_exactly' | def window_sampler_resumes_exactly(): |
| chapters/21-data.md:474 | tests/test_model.py | 1754 |  | reattributed | 'interleaved_split_samples_the_whole_corpus' is defined there |  |
| chapters/21-data.md:474 | tests/test_model.py | 1754 | 1771 | def | 'interleaved_split_samples_the_whole_corpus' | def interleaved_split_samples_the_whole_corpus(): |
| chapters/21-data.md:525 | anulm/training/finetune.py | 101–112 |  | reattributed | 'flush' is in anulm/training/finetune.py, not experiments/build_v2.py |  |
| chapters/21-data.md:525 | experiments/build_v2.py | 119 | 119 | sweep | 'main' 84–140 ('flush' 101–112, budget check 119, 'me |  |
| chapters/21-data.md:528 | anulm/training/train.py | 206–220 | 202–216 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/21-data.md:528 | anulm/training/train.py | 241–254 | 237–250 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/21-data.md:529 | anulm/training/train.py | 355–376 | 351–372 | same | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i |
| chapters/21-data.md:529 | anulm/training/train.py | 355–376 | 351–372 | sweep | 355–376, 'ma |  |
| chapters/21-data.md:529 | anulm/training/train.py | 594 | 590 | same | p.add_argument("--sample-with-replacement", action="store_true", | p.add_argument("--sample-with-replacement", action="store_true", |
| chapters/21-data.md:529 | anulm/training/train.py | 663–676 | 659–672 | same | memmapped = bool(args.data and args.data.endswith("meta.json")) | memmapped = bool(args.data and args.data.endswith("meta.json")) |
| chapters/21-data.md:529 | anulm/training/train.py | 693–699 | 689–695 | same | if memmapped or (split_encoded and val_data is not None): | if memmapped or (split_encoded and val_data is not None): |
| chapters/21-data.md:529 | anulm/training/train.py | 855–856 | 858–859 | same | ix = (torch.randint(len(train_data) - args.block_size - 1, (args.batch | ix = (torch.randint(len(train_data) - args.block_size - 1, (args.batch |
| chapters/21-data.md:530 | anulm/model.py | 358–374 | 365–381 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/21-data.md:531 | tests/test_model.py | 787 | 798 | same | def window_sampler_covers_every_window_once_per_epoch(): | def window_sampler_covers_every_window_once_per_epoch(): |
| chapters/21-data.md:532 | tests/test_model.py | 812 | 823 | same | def window_sampler_resumes_exactly(): | def window_sampler_resumes_exactly(): |
| chapters/21-data.md:533 | tests/test_model.py | 1732 | 1749 | same | def memmap_batches_match_in_memory_batches(): | def memmap_batches_match_in_memory_batches(): |
| chapters/21-data.md:534 | tests/test_model.py | 1754 | 1771 | same | def interleaved_split_samples_the_whole_corpus(): | def interleaved_split_samples_the_whole_corpus(): |
| chapters/22-the-training-loop.md:43 | anulm/training/train.py | 715–734 | 712–732 | same | builder = {"30b": AnuLMConfig.nano_30b, | builder = {"30b": AnuLMConfig.nano_30b, |
| chapters/22-the-training-loop.md:43 | anulm/training/train.py | 712–732 |  | EXCERPT | assert field is not None and _, f"--cfg: unknown field {kv!r}" | quoted line not in the new range |
| chapters/22-the-training-loop.md:115 | anulm/training/train.py | 751–774 | 749–772 | same | raw_model = model | raw_model = model |
| chapters/22-the-training-loop.md:158 | anulm/training/train.py | 846–870 | 849–873 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/22-the-training-loop.md:216 | anulm/training/train.py | 871–877 | 874–880 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/22-the-training-loop.md:284 | anulm/training/train.py | 879–886 | 882–889 | same | if step % args.log_every == 0 or step == args.steps - 1: | if step % args.log_every == 0 or step == args.steps - 1: |
| chapters/22-the-training-loop.md:317 | anulm/training/train.py | 888–905 | 891–908 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/22-the-training-loop.md:360 | anulm/training/train.py | 379–395 | 375–391 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/22-the-training-loop.md:10 | anulm/training/train.py | 540–921 | 536–924 | same | def main(): | def main(): |
| chapters/22-the-training-loop.md:107 | anulm/training/train.py | 650 |  | reattributed | 'dtype' is in anulm/training/train.py, not None |  |
| chapters/22-the-training-loop.md:107 | anulm/training/train.py | 650 | 646 | same | dtype = (torch.bfloat16 if args.device.startswith("cuda") | dtype = (torch.bfloat16 if args.device.startswith("cuda") |
| chapters/22-the-training-loop.md:140 | anulm/training/train.py | 759–761 | 757–759 | same | # Inductor generates C++ and needs a host compiler (MSVC 'cl' on Windo | # Inductor generates C++ and needs a host compiler (MSVC 'cl' on Windo |
| chapters/22-the-training-loop.md:240 | anulm/training/train.py | 849–875 | 852–878 | same | lr = lr_at(step, args, start_step) | lr = lr_at(step, args, start_step) |
| chapters/22-the-training-loop.md:457 | tests/test_model.py | 437 |  | reattributed | 'gradient_accumulation_equals_large_batch' is defined there |  |
| chapters/22-the-training-loop.md:457 | tests/test_model.py | 437 | 448 | def | 'gradient_accumulation_equals_large_batch' | def gradient_accumulation_equals_large_batch(): |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 540–921 | 536–924 | same | def main(): | def main(): |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 650 | 646 | same | dtype = (torch.bfloat16 if args.device.startswith("cuda") | dtype = (torch.bfloat16 if args.device.startswith("cuda") |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 650 | 646 | sweep | - 'anulm/training/train.py': 'main' 540–921 (650, 715 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 715–734 | 712–732 | same | builder = {"30b": AnuLMConfig.nano_30b, | builder = {"30b": AnuLMConfig.nano_30b, |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 715–734 | 712–732 | sweep | ulm/training/train.py': 'main' 540–921 (650, 715–734, 738 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 738 | 736 | same | model = AnuLM(cfg).to(args.device) | model = AnuLM(cfg).to(args.device) |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 738 | 736 | sweep | ing/train.py': 'main' 540–921 (650, 715–734, 738, 749 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 749 | 747 | same | f"({100*active/total:.1f}%)  \|  device={args.device} dtype={dtype}") | f"({100*active/total:.1f}%)  \|  device={args.device} dtype={dtype}") |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 749 | 747 | sweep | rain.py': 'main' 540–921 (650, 715–734, 738, 749, 751 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 751–774 | 749–772 | same | raw_model = model | raw_model = model |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 751–774 | 749–772 | sweep | py': 'main' 540–921 (650, 715–734, 738, 749, 751–774, 846 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 846–870 | 849–873 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 846–870 | 849–873 | sweep | n' 540–921 (650, 715–734, 738, 749, 751–774, 846–870, 866 |  |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 866 | 869 | same | loss_acc += lm_loss.item() / args.grad_accum | loss_acc += lm_loss.item() / args.grad_accum |
| chapters/22-the-training-loop.md:511 | anulm/training/train.py | 866 | 869 | sweep | 1 (650, 715–734, 738, 749, 751–774, 846–870, 866, |  |
| chapters/22-the-training-loop.md:512 | anulm/training/train.py | 871–877 | 874–880 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/22-the-training-loop.md:512 | anulm/training/train.py | 871–877 | 874–880 | sweep | 871–877, 879 |  |
| chapters/22-the-training-loop.md:512 | anulm/training/train.py | 879–886 | 882–889 | same | if step % args.log_every == 0 or step == args.steps - 1: | if step % args.log_every == 0 or step == args.steps - 1: |
| chapters/22-the-training-loop.md:512 | anulm/training/train.py | 879–886 | 882–889 | sweep | 871–877, 879–886, 888 |  |
| chapters/22-the-training-loop.md:512 | anulm/training/train.py | 888–905 | 891–908 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/22-the-training-loop.md:512 | anulm/training/train.py | 888–905 | 891–908 | sweep | 871–877, 879–886, 888–905), 'l |  |
| chapters/22-the-training-loop.md:512 | anulm/training/train.py | 379–404 | 375–400 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/22-the-training-loop.md:512 | anulm/training/train.py | 420–424 | 416–420 | same | def set_lr_scale(self, scale: float): | def set_lr_scale(self, scale: float): |
| chapters/22-the-training-loop.md:513 | anulm/model.py | 23 | 23 | sweep | el.py': 'from __future__ import annotations' 23, |  |
| chapters/22-the-training-loop.md:514 | anulm/model.py | 1059–1089 | 1071–1101 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/22-the-training-loop.md:515 | tests/test_model.py | 437 | 448 | same | def gradient_accumulation_equals_large_batch(): | def gradient_accumulation_equals_large_batch(): |
| chapters/23-precision-memory-speed.md:62 | anulm/training/train.py | 649–654 | 645–650 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/23-precision-memory-speed.md:131 | anulm/model.py | 227–232 | 234–239 | same | def forward(self, x): | def forward(self, x): |
| chapters/23-precision-memory-speed.md:154 | anulm/model.py | 608–615 | 615–622 | same | # fp32 router: top-k is discrete, and bf16 rounding would flip choices | # fp32 router: top-k is discrete, and bf16 rounding would flip choices |
| chapters/23-precision-memory-speed.md:256 | anulm/model.py | 889–897 | 901–909 | same | for i, layer in enumerate(self.layers): | for i, layer in enumerate(self.layers): |
| chapters/23-precision-memory-speed.md:312 | anulm/model.py | 930–944 | 942–956 | same | self.ce_chunk = rows | self.ce_chunk = rows |
| chapters/23-precision-memory-speed.md:357 | anulm/training/train.py | 231–238 | 227–234 | same | total = torch.cuda.get_device_properties(device).total_memory | total = torch.cuda.get_device_properties(device).total_memory |
| chapters/23-precision-memory-speed.md:370 | anulm/model.py | 1268–1276 | 1356–1364 | same | n = sum(p.numel() for p in model.parameters()) | n = sum(p.numel() for p in model.parameters()) |
| chapters/23-precision-memory-speed.md:414 | tests/test_model.py | 480–492 | 491–503 | same | def dense_moe_compiles_without_graph_breaks(): | def dense_moe_compiles_without_graph_breaks(): |
| chapters/23-precision-memory-speed.md:142 | anulm/model.py | 912 | 924 | same | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ |
| chapters/23-precision-memory-speed.md:145 | anulm/model.py | 749 | 756 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/23-precision-memory-speed.md:145 | anulm/model.py | 777 | 784 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/23-precision-memory-speed.md:270 | anulm/model.py | 1052 | 1064 | def | 'enable_gradient_checkpointing' | def enable_gradient_checkpointing(self, enable: bool = True): |
| chapters/23-precision-memory-speed.md:340 | anulm/model.py | 907 | 919 | same | if self.ce_chunk and self.training: | if self.ce_chunk and self.training: |
| chapters/23-precision-memory-speed.md:385 | anulm/training/train.py | 877 | 880 | same | warn_if_spilling(args.device) | warn_if_spilling(args.device) |
| chapters/23-precision-memory-speed.md:431 | tests/test_model.py | 495 | 506 | def | 'sparse_moe_breaks_the_graph_as_documented' | def sparse_moe_breaks_the_graph_as_documented(): |
| chapters/23-precision-memory-speed.md:437 | anulm/model.py | 372–373 | 379–380 | same | if window is not None: | if window is not None: |
| chapters/23-precision-memory-speed.md:517 | anulm/training/train.py | 223–238 | 219–234 | same | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: |
| chapters/23-precision-memory-speed.md:517 | anulm/training/train.py | 649–654 | 645–650 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/23-precision-memory-speed.md:517 | anulm/training/train.py | 705 | ? | UNMAPPED | assert not args.ce_chunk, "--bigram-teacher needs the logits; drop --c |  |
| chapters/23-precision-memory-speed.md:517 | anulm/training/train.py | 877 | 880 | same | warn_if_spilling(args.device) | warn_if_spilling(args.device) |
| chapters/23-precision-memory-speed.md:518 | anulm/multimodal/omni_train.py | 194 | 194 | sweep | y' callers: 'anulm/multimodal/omni_train.py' 194, 'an |  |
| chapters/23-precision-memory-speed.md:518 | anulm/tasks/decide.py | 473 | 475 | same | fit_memory(d.model) | fit_memory(d.model) |
| chapters/23-precision-memory-speed.md:518 | anulm/tasks/decide.py | 473 | 475 | sweep | /omni_train.py' 194, 'anulm/tasks/decide.py' 473 and |  |
| chapters/23-precision-memory-speed.md:518 | anulm/tasks/decide.py | 551 | 553 | same | fit_memory(d.model) | fit_memory(d.model) |
| chapters/23-precision-memory-speed.md:518 | anulm/tasks/decide.py | 551 | 553 | sweep | ain.py' 194, 'anulm/tasks/decide.py' 473 and 551, |  |
| chapters/23-precision-memory-speed.md:519 | anulm/multimodal/speech_encoder.py | 232 | 231 | same | fit_memory(model) | fit_memory(model) |
| chapters/23-precision-memory-speed.md:519 | anulm/multimodal/speech_encoder.py | 232 | 231 | sweep | 'anulm/multimodal/speech_encoder.py' 232, 'an |  |
| chapters/23-precision-memory-speed.md:519 | anulm/multimodal/vision_encoder.py | 457 | 456 | same | fit_memory(model) | fit_memory(model) |
| chapters/23-precision-memory-speed.md:519 | anulm/multimodal/vision_encoder.py | 457 | 456 | sweep | y' 232, 'anulm/multimodal/vision_encoder.py' 457. |  |
| chapters/23-precision-memory-speed.md:520 | anulm/model.py | 227–232 | 234–239 | same | def forward(self, x): | def forward(self, x): |
| chapters/23-precision-memory-speed.md:520 | anulm/model.py | 604–615 | 611–622 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/23-precision-memory-speed.md:521 | anulm/model.py | 372–373 | 379–380 | same | if window is not None: | if window is not None: |
| chapters/23-precision-memory-speed.md:521 | anulm/model.py | 749 | 756 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/23-precision-memory-speed.md:521 | anulm/model.py | 777 | 784 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/23-precision-memory-speed.md:522 | anulm/model.py | 889–897 | 901–909 | same | for i, layer in enumerate(self.layers): | for i, layer in enumerate(self.layers): |
| chapters/23-precision-memory-speed.md:522 | anulm/model.py | 907 | 919 | same | if self.ce_chunk and self.training: | if self.ce_chunk and self.training: |
| chapters/23-precision-memory-speed.md:522 | anulm/model.py | 912 | 924 | same | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ |
| chapters/23-precision-memory-speed.md:522 | anulm/model.py | 918–930 | 930–942 | same | def enable_chunked_loss(self, rows: int = 2048): | def enable_chunked_loss(self, rows: int = 2048): |
| chapters/23-precision-memory-speed.md:523 | anulm/model.py | 932–941 | 944–953 | same | def _chunked_loss(self, x, targets): | def _chunked_loss(self, x, targets): |
| chapters/23-precision-memory-speed.md:523 | anulm/model.py | 943–944 | 955–956 | same | def _ce_sum(self, h, t): | def _ce_sum(self, h, t): |
| chapters/23-precision-memory-speed.md:524 | anulm/model.py | 1052–1056 | 1064–1068 | same | def enable_gradient_checkpointing(self, enable: bool = True): | def enable_gradient_checkpointing(self, enable: bool = True): |
| chapters/23-precision-memory-speed.md:524 | anulm/model.py | 1257–1276 | 1345–1364 | same | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i |
| chapters/23-precision-memory-speed.md:525 | tests/test_model.py | 480 | 491 | same | def dense_moe_compiles_without_graph_breaks(): | def dense_moe_compiles_without_graph_breaks(): |
| chapters/23-precision-memory-speed.md:526 | tests/test_model.py | 495 | 506 | same | def sparse_moe_breaks_the_graph_as_documented(): | def sparse_moe_breaks_the_graph_as_documented(): |
| chapters/23-precision-memory-speed.md:527 | tests/test_model.py | 1531 | 1548 | same | def chunked_loss_matches_the_full_loss_and_its_gradients(): | def chunked_loss_matches_the_full_loss_and_its_gradients(): |
| chapters/24-measuring-progress.md:37 | anulm/training/train.py | 342–352 | 338–348 | same | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, |
| chapters/24-measuring-progress.md:57 | tests/test_model.py | 985–989 | 994–998 | same | data = torch.arange(100_000) | data = torch.arange(100_000) |
| chapters/24-measuring-progress.md:74 | anulm/training/train.py | 481–537 | 477–533 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/24-measuring-progress.md:209 | anulm/training/train.py | 355–376 | 351–372 | same | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i |
| chapters/24-measuring-progress.md:228 | tests/test_model.py | 991–997 | 1000–1006 | same | flat = lambda bs: [int(i) for b in bs for i in b] | flat = lambda bs: [int(i) for b in bs for i in b] |
| chapters/24-measuring-progress.md:291 | anulm/training/train.py | 784–788 | 782–786 | same | eval_spec = ({"mode": "strided", "windows": args.eval_windows, | eval_spec = ({"mode": "strided", "windows": args.eval_windows, |
| chapters/24-measuring-progress.md:315 | anulm/training/train.py | 903–905 | 906–908 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/24-measuring-progress.md:377 | anulm/eval/eval_golden.py | 86–103 | 85–102 | same | # --- multiple choice: one batch, the 4 choices right-padded ------ | # --- multiple choice: one batch, the 4 choices right-padded ------ |
| chapters/24-measuring-progress.md:28 | anulm/training/train.py | 693–699 | 689–695 | same | if memmapped or (split_encoded and val_data is not None): | if memmapped or (split_encoded and val_data is not None): |
| chapters/24-measuring-progress.md:303 | anulm/training/train.py | 693–699 | 689–695 | same | if memmapped or (split_encoded and val_data is not None): | if memmapped or (split_encoded and val_data is not None): |
| chapters/24-measuring-progress.md:477 | anulm/training/train.py | 342–352 | 338–348 | same | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, |
| chapters/24-measuring-progress.md:477 | anulm/training/train.py | 355–376 | 351–372 | same | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i |
| chapters/24-measuring-progress.md:478 | anulm/training/train.py | 481–537 | 477–533 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/24-measuring-progress.md:478 | anulm/training/train.py | 481–537 | 477–533 | sweep | 481–537, 'ma |  |
| chapters/24-measuring-progress.md:478 | anulm/training/train.py | 784–788 | 782–786 | same | eval_spec = ({"mode": "strided", "windows": args.eval_windows, | eval_spec = ({"mode": "strided", "windows": args.eval_windows, |
| chapters/24-measuring-progress.md:478 | anulm/training/train.py | 903–905 | ? | CHECK | bpb = val / math.log(2) / args.bytes_per_token | 'eval_spec' spans [(784, 788, 784)] at the old pin |
| chapters/24-measuring-progress.md:478 | anulm/training/train.py | 903–905 | 903–905 | sweep | 481–537, 'main' 784–788 ('eval_spec') and 903–905 (bit |  |
| chapters/24-measuring-progress.md:480 | tests/test_model.py | 971–1017 | 980–1026 | same | def eval_windows_are_frozen_and_strided_windows_cover_the_split(): | def eval_windows_are_frozen_and_strided_windows_cover_the_split(): |
| chapters/24-measuring-progress.md:480 | tests/test_model.py | 971–1017 | 980–1026 | sweep | 971–1017, 'go |  |
| chapters/24-measuring-progress.md:480 | tests/test_model.py | 1448–1490 | 1465–1507 | same | def golden_items_are_well_formed_and_the_scorer_runs(): | def golden_items_are_well_formed_and_the_scorer_runs(): |
| chapters/24-measuring-progress.md:481 | anulm/eval/eval_golden.py | 59–119 | 58–118 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/24-measuring-progress.md:482 | anulm/eval/eval_code.py | 221–273 | 220–272 | same | def score(ckpt: str, problems: list[dict], device: str, max_new: int,  | def score(ckpt: str, problems: list[dict], device: str, max_new: int,  |
| chapters/24-measuring-progress.md:484 | anulm/training/train.py | 693–699 | 689–695 | same | if memmapped or (split_encoded and val_data is not None): | if memmapped or (split_encoded and val_data is not None): |
| chapters/25-checkpoints-and-resuming.md:27 | anulm/training/train.py | 888–902 | 891–905 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/25-checkpoints-and-resuming.md:183 | tests/test_model.py | 812–822 | 823–833 | same | def window_sampler_resumes_exactly(): | def window_sampler_resumes_exactly(): |
| chapters/25-checkpoints-and-resuming.md:209 | anulm/training/train.py | 790–837 | 788–840 | same | start_step, best_val = 0, float("inf") | start_step, best_val = 0, float("inf") |
| chapters/25-checkpoints-and-resuming.md:209 | anulm/training/train.py | 788–840 |  | EXCERPT | ck = torch.load(last_path, map_location="cpu", weights_only=False) | quoted line not in the new range |
| chapters/25-checkpoints-and-resuming.md:209 | anulm/training/train.py | 811 | 814 | edited-quote | ck = torch.load(args.out, map_location="cpu", weights_only=False) | ck = safe_load(args.out, map_location="cpu") |
| chapters/25-checkpoints-and-resuming.md:209 | anulm/training/train.py | 818 | 821 | edited-quote | assert len(ck["opt"]) == len(opt.opts), ( | if len(ck["opt"]) != len(opt.opts): |
| chapters/25-checkpoints-and-resuming.md:209 | anulm/training/train.py | 819 | 822 | edited-quote | f"{last_path} holds state for {len(ck['opt'])} optimizer(s) but --opti | p.error(f"{last_path} holds state for {len(ck['opt'])} optimizer(s) bu |
| chapters/25-checkpoints-and-resuming.md:267 | anulm/training/train.py | 846–847 | 849–850 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/25-checkpoints-and-resuming.md:341 | anulm/training/train.py | 398–403 | 394–399 | same | # Extending a finished run (--resume with a larger --steps) lands mid- | # Extending a finished run (--resume with a larger --steps) lands mid- |
| chapters/25-checkpoints-and-resuming.md:369 | anulm/model.py | 1205–1238 | 1280–1316 | same | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di |
| chapters/25-checkpoints-and-resuming.md:369 | anulm/model.py | 1280–1316 |  | EXCERPT | if not p.exists() and re.fullmatch(r"[\w.-]+/[\w.-]+", str(path).repla | quoted line not in the new range |
| chapters/25-checkpoints-and-resuming.md:369 | anulm/model.py | 1226 | 1304 | edited-quote | return torch.load(str(p), map_location=map_location, weights_only=Fals | return safe_load(p, map_location) |
| chapters/25-checkpoints-and-resuming.md:166 | anulm/training/train.py | 649 | ? | CHECK | torch.manual_seed(args.seed) | no file named on the line and nothing quoted confirms it |
| chapters/25-checkpoints-and-resuming.md:167 | anulm/training/train.py | 856 | ? | CHECK | if args.sample_with_replacement else sampler.next()) | no file named on the line and nothing quoted confirms it |
| chapters/25-checkpoints-and-resuming.md:201 | tests/test_model.py | 352 | 363 | same | def checkpoint_roundtrip_is_exact(): | def checkpoint_roundtrip_is_exact(): |
| chapters/25-checkpoints-and-resuming.md:355 | anulm/training/train.py | 384–395 | ? | CHECK | if step < args.warmup: | no file named on the line and nothing quoted confirms it |
| chapters/25-checkpoints-and-resuming.md:393 | tests/test_model.py | 914 |  | reattributed | quoted code is in tests/test_model.py, not None |  |
| chapters/25-checkpoints-and-resuming.md:393 | tests/test_model.py | 914 | 924 | same | def an_exported_folder_loads_everywhere_a_pt_does(): | def an_exported_folder_loads_everywhere_a_pt_does(): |
| chapters/25-checkpoints-and-resuming.md:461 | anulm/training/train.py | 152–160 | 152–160 | sweep | 152–160, 'lr |  |
| chapters/25-checkpoints-and-resuming.md:461 | anulm/training/train.py | 379–404 | 375–400 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/25-checkpoints-and-resuming.md:461 | anulm/training/train.py | 790–837 | 788–840 | same | start_step, best_val = 0, float("inf") | start_step, best_val = 0, float("inf") |
| chapters/25-checkpoints-and-resuming.md:461 | anulm/training/train.py | 846–847 | 849–850 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/25-checkpoints-and-resuming.md:461 | anulm/training/train.py | 846–847 | 849–850 | sweep | 0, 'lr_at' 379–404, 'main' 790–837 (resume), 846–847 ('-- |  |
| chapters/25-checkpoints-and-resuming.md:462 | anulm/training/train.py | 888–902 | 891–905 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/25-checkpoints-and-resuming.md:462 | anulm/training/train.py | 888–902 | 891–905 | sweep | 888–902 (sav |  |
| chapters/25-checkpoints-and-resuming.md:463 | anulm/model.py | 1205–1254 | 1280–1342 | same | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di |
| chapters/25-checkpoints-and-resuming.md:464 | tests/test_model.py | 352–369 | 363–380 | same | def checkpoint_roundtrip_is_exact(): | def checkpoint_roundtrip_is_exact(): |
| chapters/25-checkpoints-and-resuming.md:465 | tests/test_model.py | 812–822 | 823–833 | same | def window_sampler_resumes_exactly(): | def window_sampler_resumes_exactly(): |
| chapters/25-checkpoints-and-resuming.md:466 | tests/test_model.py | 310–317 | 321–328 | same | def expert_bias_is_not_trainable(): | def expert_bias_is_not_trainable(): |
| chapters/25-checkpoints-and-resuming.md:466 | tests/test_model.py | 310–317 | 321–328 | sweep | 310–317, 'an |  |
| chapters/25-checkpoints-and-resuming.md:466 | tests/test_model.py | 914–967 | 924–976 | same | def an_exported_folder_loads_everywhere_a_pt_does(): | def an_exported_folder_loads_everywhere_a_pt_does(): |
| chapters/26-debugging-silent-failures.md:30 | tests/test_model.py | 1–10 | 1–15 | same | """ | """ |
| chapters/26-debugging-silent-failures.md:110 | tests/test_model.py | 45–63 | 56–74 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:143 | tests/test_model.py | 462–476 | 473–487 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:173 | tests/test_model.py | 1753–1766 | 1770–1783 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:260 | anulm/model.py | 604–615 | 611–622 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/26-debugging-silent-failures.md:330 | anulm/model.py | 954–972 | 966–984 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/26-debugging-silent-failures.md:363 | anulm/model.py | 974–978 | 986–990 | same | def train(self, mode: bool = True): | def train(self, mode: bool = True): |
| chapters/26-debugging-silent-failures.md:375 | tests/test_model.py | 1671–1681 | 1688–1698 | same | x = torch.randint(0, 259, (2, 8)) | x = torch.randint(0, 259, (2, 8)) |
| chapters/26-debugging-silent-failures.md:399 | anulm/training/train.py | 206–220 | 202–216 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/26-debugging-silent-failures.md:433 | anulm/model.py | 439–442 | 446–449 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/26-debugging-silent-failures.md:447 | tests/test_model.py | 149–153 | 160–164 | same | gqa = tiny(attn="gqa", yarn=True, yarn_factor=40.0, head_dim=8) | gqa = tiny(attn="gqa", yarn=True, yarn_factor=40.0, head_dim=8) |
| chapters/26-debugging-silent-failures.md:504 | tests/test_model.py | 2213–2226 | 2250–2271 | same | def main(): | def main(): |
| chapters/26-debugging-silent-failures.md:504 | tests/test_model.py | 2214 | 2251 | edited-quote | passed, failed = [], [] | passed, skipped, failed = [], [], [] |
| chapters/26-debugging-silent-failures.md:73 | anulm/model.py | 407 | 414 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/26-debugging-silent-failures.md:77 | anulm/model.py | 476–485 | 483–492 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/26-debugging-silent-failures.md:137 | tests/test_model.py | 90 | 101 | def | 'rope_is_relative' | def rope_is_relative(): |
| chapters/26-debugging-silent-failures.md:138 | tests/test_model.py | 642 | 653 | def | 'sliding_window_limits_reach' | def sliding_window_limits_reach(): |
| chapters/26-debugging-silent-failures.md:166 | tests/test_model.py | 593 | 604 | same | def cached_generation_matches_uncached(): | def cached_generation_matches_uncached(): |
| chapters/26-debugging-silent-failures.md:167 | tests/test_model.py | 1531 | 1548 | same | def chunked_loss_matches_the_full_loss_and_its_gradients(): | def chunked_loss_matches_the_full_loss_and_its_gradients(): |
| chapters/26-debugging-silent-failures.md:196 | tests/test_model.py | 787 | 798 | same | def window_sampler_covers_every_window_once_per_epoch(): | def window_sampler_covers_every_window_once_per_epoch(): |
| chapters/26-debugging-silent-failures.md:198 | tests/test_model.py | 1494 | 1511 | same | def qa_packing_masks_prompts_and_keeps_examples_whole(): | def qa_packing_masks_prompts_and_keeps_examples_whole(): |
| chapters/26-debugging-silent-failures.md:469 | tests/test_model.py | 971 | 980 | def | 'eval_windows_are_frozen_and_strided_windows_cover_the_split' | def eval_windows_are_frozen_and_strided_windows_cover_the_split(): |
| chapters/26-debugging-silent-failures.md:479 | tests/test_model.py | 1271 | 1285 | def | 'transformers_wrapper_matches_the_native_model' | def transformers_wrapper_matches_the_native_model(): |
| chapters/26-debugging-silent-failures.md:480 | tests/test_model.py | 1110 | 1119 | def | 'an_exported_qa_checkpoint_still_offers_its_question_mode' | def an_exported_qa_checkpoint_still_offers_its_question_mode(): |
| chapters/26-debugging-silent-failures.md:481 | tests/test_model.py | 1170 | 1177 | def | 'an_export_keeps_what_makes_a_checkpoint_more_than_a_language_model' | def an_export_keeps_what_makes_a_checkpoint_more_than_a_language_model |
| chapters/26-debugging-silent-failures.md:482 | tests/test_model.py | 914 | 924 | def | 'an_exported_folder_loads_everywhere_a_pt_does' | def an_exported_folder_loads_everywhere_a_pt_does(): |
| chapters/26-debugging-silent-failures.md:483 | tests/test_model.py | 1219 | 1225 | def | 'eval_code_sandbox_contains_generated_code' | def eval_code_sandbox_contains_generated_code(): |
| chapters/26-debugging-silent-failures.md:484 | tests/test_model.py | 1448 |  | reattributed | 'golden_items_are_well_formed_and_the_scorer_runs' is defined there |  |
| chapters/26-debugging-silent-failures.md:484 | tests/test_model.py | 1448 | 1465 | def | 'golden_items_are_well_formed_and_the_scorer_runs' | def golden_items_are_well_formed_and_the_scorer_runs(): |
| chapters/26-debugging-silent-failures.md:485 | tests/test_model.py | 1367 |  | reattributed | 'bpe_files_load_under_both_format_ids' is defined there |  |
| chapters/26-debugging-silent-failures.md:485 | tests/test_model.py | 1367 | 1379 | def | 'bpe_files_load_under_both_format_ids' | def bpe_files_load_under_both_format_ids(): |
| chapters/26-debugging-silent-failures.md:486 | tests/test_model.py | 550 |  | reattributed | 'incremental_forward_matches_full_forward' is defined there |  |
| chapters/26-debugging-silent-failures.md:486 | tests/test_model.py | 550 | 561 | def | 'incremental_forward_matches_full_forward' | def incremental_forward_matches_full_forward(): |
| chapters/26-debugging-silent-failures.md:523 | tests/test_model.py | 28 | ? | CHECK | def test(fn): | 'TESTS' spans [(25, 25, 25)] at the old pin |
| chapters/26-debugging-silent-failures.md:546 | tests/test_model.py | 437 | 448 | same | def gradient_accumulation_equals_large_batch(): | def gradient_accumulation_equals_large_batch(): |
| chapters/26-debugging-silent-failures.md:589 | tests/test_model.py | 1–10 | 1–15 | same | """ | """ |
| chapters/26-debugging-silent-failures.md:589 | tests/test_model.py | 1–10 | 1–15 | sweep | - 'tests/test_model.py': module docstring 1–10; 'te |  |
| chapters/26-debugging-silent-failures.md:589 | tests/test_model.py | 28–30 | 34–38 | same | def test(fn): | def test(fn): |
| chapters/26-debugging-silent-failures.md:590 | tests/test_model.py | 45–63 | 56–74 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:591 | tests/test_model.py | 132–161 | 143–172 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:591 | tests/test_model.py | 132–161 | 143–172 | sweep | 132–161; 'de |  |
| chapters/26-debugging-silent-failures.md:591 | tests/test_model.py | 462–476 | 473–487 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:592 | tests/test_model.py | 1658–1681 | 1675–1698 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:593 | tests/test_model.py | 1753–1766 | 1770–1783 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:593 | tests/test_model.py | 2213–2226 | 2250–2271 | same | def main(): | def main(): |
| chapters/26-debugging-silent-failures.md:595 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/26-debugging-silent-failures.md:595 | anulm/model.py | 439–442 | 446–449 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/26-debugging-silent-failures.md:596 | anulm/model.py | 604–615 | 611–622 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/26-debugging-silent-failures.md:596 | anulm/model.py | 954–972 | 966–984 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/26-debugging-silent-failures.md:596 | anulm/model.py | 974–978 | 986–990 | same | def train(self, mode: bool = True): | def train(self, mode: bool = True): |
| chapters/26-debugging-silent-failures.md:597 | anulm/training/train.py | 206–220 | 202–216 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/26-debugging-silent-failures.md:597 | anulm/training/train.py | 223–238 | 219–234 | same | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: |
| chapters/27-decoding-and-the-kv-cache.md:39 | anulm/model.py | 1128–1139 | 1140–1151 | same | B = idx.shape[0] | B = idx.shape[0] |
| chapters/27-decoding-and-the-kv-cache.md:67 | anulm/model.py | 1149–1161 | 1161–1173 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/27-decoding-and-the-kv-cache.md:119 | anulm/model.py | 861–863 | 873–875 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/27-decoding-and-the-kv-cache.md:127 | anulm/model.py | 463–470 | 470–477 | same | if cache is not None: | if cache is not None: |
| chapters/27-decoding-and-the-kv-cache.md:185 | anulm/model.py | 885–887 | 897–899 | edited | assert start + S <= self.cfg.block_size, \ | if start + S > self.cfg.block_size:      # an error, not an assert: py |
| chapters/27-decoding-and-the-kv-cache.md:185 | anulm/model.py | 885 | 897 | edited-quote | assert start + S <= self.cfg.block_size, \ | if start + S > self.cfg.block_size:      # an error, not an assert: py |
| chapters/27-decoding-and-the-kv-cache.md:185 | anulm/model.py | 886 | 898 | edited-quote | f"positions up to {start + S} exceed block_size {self.cfg.block_size}" | raise ValueError(f"positions up to {start + S} exceed block_size {self |
| chapters/27-decoding-and-the-kv-cache.md:193 | anulm/model.py | 332–336 | 339–343 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| chapters/27-decoding-and-the-kv-cache.md:218 | anulm/model.py | 358–374 | 365–381 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/27-decoding-and-the-kv-cache.md:233 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/27-decoding-and-the-kv-cache.md:461 | tests/test_model.py | 550–567 | 561–578 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/27-decoding-and-the-kv-cache.md:64 | anulm/model.py | 1152–1155 | 1164–1167 | same | logits = logits / temperature | logits = logits / temperature |
| chapters/27-decoding-and-the-kv-cache.md:91 | anulm/model.py | 1132–1139 | ? | CHECK | for _ in range(max_new_tokens): | no file named on the line and nothing quoted confirms it |
| chapters/27-decoding-and-the-kv-cache.md:207 | tests/test_model.py | 625 |  | reattributed | quoted code is in tests/test_model.py, not anulm/model.py |  |
| chapters/27-decoding-and-the-kv-cache.md:207 | tests/test_model.py | 625 | 636 | same | def rotary_cache_tracks_dtype_and_positions(): | def rotary_cache_tracks_dtype_and_positions(): |
| chapters/27-decoding-and-the-kv-cache.md:340 | anulm/inference/api.py | 343 | 389 | same | class Generation: | class Generation: |
| chapters/27-decoding-and-the-kv-cache.md:341 | anulm/inference/api.py | 351–352 | 398–399 | same | self.ids = self.eng.encode(prompt)[-(self.eng.cfg.block_size - 1):] | self.ids = self.eng.encode(prompt)[-(self.eng.cfg.block_size - 1):] |
| chapters/27-decoding-and-the-kv-cache.md:393 | anulm/model.py | 749 | 756 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/27-decoding-and-the-kv-cache.md:393 | anulm/model.py | 777 | 784 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/27-decoding-and-the-kv-cache.md:487 | tests/test_model.py | 593 | 604 | same | def cached_generation_matches_uncached(): | def cached_generation_matches_uncached(): |
| chapters/27-decoding-and-the-kv-cache.md:492 | tests/test_model.py | 571 |  | reattributed | 'cache_holds_what_it_should' is defined there |  |
| chapters/27-decoding-and-the-kv-cache.md:492 | tests/test_model.py | 571 | 582 | def | 'cache_holds_what_it_should' | def cache_holds_what_it_should(): |
| chapters/27-decoding-and-the-kv-cache.md:554 | anulm/model.py | 332–336 | 339–343 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| chapters/27-decoding-and-the-kv-cache.md:554 | anulm/model.py | 358–374 | 365–381 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/27-decoding-and-the-kv-cache.md:554 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/27-decoding-and-the-kv-cache.md:555 | anulm/model.py | 463–470 | 470–477 | same | if cache is not None: | if cache is not None: |
| chapters/27-decoding-and-the-kv-cache.md:555 | anulm/model.py | 548–557 | 555–564 | same | if cache is not None: | if cache is not None: |
| chapters/27-decoding-and-the-kv-cache.md:556 | anulm/model.py | 861–863 | 873–875 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/27-decoding-and-the-kv-cache.md:556 | anulm/model.py | 885–887 | 897–899 | edited | assert start + S <= self.cfg.block_size, \ | if start + S > self.cfg.block_size:      # an error, not an assert: py |
| chapters/27-decoding-and-the-kv-cache.md:557 | anulm/model.py | 1128–1161 | 1140–1173 | same | B = idx.shape[0] | B = idx.shape[0] |
| chapters/27-decoding-and-the-kv-cache.md:558 | anulm/inference/api.py | 343 | 389 | same | class Generation: | class Generation: |
| chapters/27-decoding-and-the-kv-cache.md:558 | anulm/inference/api.py | 351–352 | 398–399 | same | self.ids = self.eng.encode(prompt)[-(self.eng.cfg.block_size - 1):] | self.ids = self.eng.encode(prompt)[-(self.eng.cfg.block_size - 1):] |
| chapters/27-decoding-and-the-kv-cache.md:559 | tests/test_model.py | 550 | 561 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/27-decoding-and-the-kv-cache.md:560 | tests/test_model.py | 571 | 582 | same | def cache_holds_what_it_should(): | def cache_holds_what_it_should(): |
| chapters/27-decoding-and-the-kv-cache.md:560 | tests/test_model.py | 593 | 604 | same | def cached_generation_matches_uncached(): | def cached_generation_matches_uncached(): |
| chapters/27-decoding-and-the-kv-cache.md:561 | tests/test_model.py | 625 | 636 | same | def rotary_cache_tracks_dtype_and_positions(): | def rotary_cache_tracks_dtype_and_positions(): |
| chapters/28-choosing-the-next-token.md:19 | anulm/model.py | 1140–1157 | 1152–1169 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/28-choosing-the-next-token.md:216 | anulm/model.py | 1025–1042 | 1037–1054 | same | for _ in range(max_tokens): | for _ in range(max_tokens): |
| chapters/28-choosing-the-next-token.md:254 | anulm/model.py | 1158–1167 | 1170–1179 | same | if eos_id is not None: | if eos_id is not None: |
| chapters/28-choosing-the-next-token.md:282 | anulm/inference/api.py | 192–195 | 218–221 | same | temperature: Optional[float] = Field(None, ge=0, le=2, description="De | temperature: Optional[float] = Field(None, ge=0, le=2, description="De |
| chapters/28-choosing-the-next-token.md:87 | anulm/model.py | 1157 | ? | CHECK | nxt = torch.multinomial(probs, num_samples=1)          # (B, 1) | no file named on the line and nothing quoted confirms it |
| chapters/28-choosing-the-next-token.md:96 | anulm/model.py | 1149-1150 | ? | CHECK | if temperature <= 0: | no file named on the line and nothing quoted confirms it |
| chapters/28-choosing-the-next-token.md:102 | anulm/model.py | 1145-1156 | ? | CHECK | if repetition_penalty != 1.0: | no file named on the line and nothing quoted confirms it |
| chapters/28-choosing-the-next-token.md:136 | tests/test_model.py | 387 |  | reattributed | 'generation_is_deterministic_at_top_k_1' is defined there |  |
| chapters/28-choosing-the-next-token.md:136 | tests/test_model.py | 387 | 398 | def | 'generation_is_deterministic_at_top_k_1' | def generation_is_deterministic_at_top_k_1(): |
| chapters/28-choosing-the-next-token.md:150 | anulm/inference/serve.py | 153 | 158 | same | out = self.model.generate(x, max_tokens, temperature=temperature,   #  | out = self.model.generate(x, max_tokens, temperature=temperature,   #  |
| chapters/28-choosing-the-next-token.md:150 | anulm/inference/api.py | 399 | 446 | same | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/28-choosing-the-next-token.md:152 | tests/test_release_fixes.py | 112 |  | reattributed | 'test_temperature_zero_reaches_the_model_unclamped' is in tests/test_r |  |
| chapters/28-choosing-the-next-token.md:271 | tests/test_model.py | 609 |  | reattributed | 'generation_stops_at_eos' is defined there |  |
| chapters/28-choosing-the-next-token.md:271 | tests/test_model.py | 609 | 620 | def | 'generation_stops_at_eos' | def generation_stops_at_eos(): |
| chapters/28-choosing-the-next-token.md:291 | anulm/inference/api.py | 343 | 389 | def | 'Generation' | class Generation: |
| chapters/28-choosing-the-next-token.md:292 | anulm/inference/api.py | 399–401 | 446–448 | same | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/28-choosing-the-next-token.md:468 | anulm/model.py | 1003–1050 | 1015–1062 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/28-choosing-the-next-token.md:468 | anulm/model.py | 1025–1042 | 1037–1054 | same | for _ in range(max_tokens): | for _ in range(max_tokens): |
| chapters/28-choosing-the-next-token.md:468 | anulm/model.py | 1025–1042 | 1037–1054 | sweep | AnuLM.decode_from_embeds' 1003–1050 (excerpt 1025–1042), |  |
| chapters/28-choosing-the-next-token.md:469 | anulm/model.py | 1140–1167 | 1152–1179 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/28-choosing-the-next-token.md:470 | anulm/inference/api.py | 192–195 | 218–221 | same | temperature: Optional[float] = Field(None, ge=0, le=2, description="De | temperature: Optional[float] = Field(None, ge=0, le=2, description="De |
| chapters/28-choosing-the-next-token.md:470 | anulm/inference/api.py | 343 | 389 | same | class Generation: | class Generation: |
| chapters/28-choosing-the-next-token.md:470 | anulm/inference/api.py | 399–401 | 446–448 | same | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/28-choosing-the-next-token.md:471 | anulm/inference/api.py | 559–561 | 611–613 | same | gen = Generation(chat_prompt(question), req.max_completion_tokens or r | gen = Generation(chat_prompt(question), req.max_completion_tokens or r |
| chapters/28-choosing-the-next-token.md:471 | anulm/inference/api.py | 559–561 | 611–613 | sweep | 559–561, com |  |
| chapters/28-choosing-the-next-token.md:471 | anulm/inference/api.py | 597–598 | 649–650 | same | gen = Generation(prompt, req.max_tokens or 64, 0.7 if req.temperature  | gen = Generation(prompt, req.max_tokens or 64, 0.7 if req.temperature  |
| chapters/28-choosing-the-next-token.md:471 | anulm/inference/api.py | 597–598 | 649–650 | sweep | 559–561, completion defaults 597–598. |  |
| chapters/28-choosing-the-next-token.md:472 | anulm/inference/serve.py | 153–155 | 158–160 | same | out = self.model.generate(x, max_tokens, temperature=temperature,   #  | out = self.model.generate(x, max_tokens, temperature=temperature,   #  |
| chapters/28-choosing-the-next-token.md:472 | anulm/inference/serve.py | 153–155 | 158–160 | sweep | ulm/inference/serve.py': the 'generate' call 153–155 (no |  |
| chapters/28-choosing-the-next-token.md:472 | anulm/inference/serve.py | 210–214 | ? | CHECK | temperature=float(req.get("temperature", 0.8)), | 'generate' spans [(134, 171, 134)] at the old pin |
| chapters/28-choosing-the-next-token.md:472 | anulm/inference/serve.py | 210–214 | 210–214 | sweep | rate' call 153–155 (no clamp), HTTP defaults 210–214. 'an |  |
| chapters/28-choosing-the-next-token.md:472 | anulm/inference/sample.py | 34–35 | 34–35 | sweep | faults 210–214. 'anulm/inference/sample.py': 34–35. |  |
| chapters/28-choosing-the-next-token.md:473 | tests/test_model.py | 387 | 398 | same | def generation_is_deterministic_at_top_k_1(): | def generation_is_deterministic_at_top_k_1(): |
| chapters/28-choosing-the-next-token.md:474 | tests/test_model.py | 609 | 620 | same | def generation_stops_at_eos(): | def generation_stops_at_eos(): |
| chapters/29-long-context.md:59 | anulm/model.py | 253–268 | 260–275 | same | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): |
| chapters/29-long-context.md:84 | anulm/model.py | 298–315 | 305–322 | same | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: |
| chapters/29-long-context.md:139 | tests/test_model.py | 120–129 | 131–140 | same | def yarn_blends_the_right_bands(): | def yarn_blends_the_right_bands(): |
| chapters/29-long-context.md:165 | anulm/model.py | 286–291 | 293–298 | same | if cfg.yarn: | if cfg.yarn: |
| chapters/29-long-context.md:176 | anulm/model.py | 439–442 | 446–449 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/29-long-context.md:291 | anulm/model.py | 805–807 | 812–814 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/29-long-context.md:185 | anulm/model.py | 281–285 | ? | CHECK | # YaRN's temperature, split the way DeepSeek (and so the 105B) splits | no file named on the line and nothing quoted confirms it |
| chapters/29-long-context.md:206 | tests/test_model.py | 133 |  | reattributed | quoted code is in tests/test_model.py, not docs/RESULTS.md |  |
| chapters/29-long-context.md:206 | tests/test_model.py | 133 | 144 | same | def yarn_temperature_reaches_attention(): | def yarn_temperature_reaches_attention(): |
| chapters/29-long-context.md:416 | anulm/model.py | 109–110 | 109–110 | sweep | - 'anulm/model.py': configuration 109–110; '_y |  |
| chapters/29-long-context.md:417 | anulm/model.py | 253–268 | 260–275 | same | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): |
| chapters/29-long-context.md:417 | anulm/model.py | 281–291 | 288–298 | same | # YaRN's temperature, split the way DeepSeek (and so the 105B) splits | # YaRN's temperature, split the way DeepSeek (and so the 105B) splits |
| chapters/29-long-context.md:418 | anulm/model.py | 298–315 | 305–322 | same | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: |
| chapters/29-long-context.md:418 | anulm/model.py | 358–374 | 365–381 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/29-long-context.md:419 | anulm/model.py | 439–442 | 446–449 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/29-long-context.md:419 | anulm/model.py | 805–807 | 812–814 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/29-long-context.md:420 | tests/test_model.py | 120 | 131 | same | def yarn_blends_the_right_bands(): | def yarn_blends_the_right_bands(): |
| chapters/29-long-context.md:421 | tests/test_model.py | 133 | 144 | same | def yarn_temperature_reaches_attention(): | def yarn_temperature_reaches_attention(): |
| chapters/30-instruction-tuning.md:142 | anulm/model.py | 910–916 | 922–928 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/30-instruction-tuning.md:259 | anulm/training/finetune.py | 359–374 | 360–375 | same | for step in range(start_step, end): | for step in range(start_step, end): |
| chapters/30-instruction-tuning.md:415 | anulm/training/finetune.py | 306–319 | 307–320 | same | # A held-out loss over questions the model trained on is optimistic, a | # A held-out loss over questions the model trained on is optimistic, a |
| chapters/30-instruction-tuning.md:493 | tests/test_model.py | 1508–1527 | 1525–1544 | same | for it, (p, a) in zip(items, pairs): | for it, (p, a) in zip(items, pairs): |
| chapters/30-instruction-tuning.md:71 | anulm/training/finetune.py | 385 | 386 | def | 'qa_templates' | "val_loss": val, "qa_template": PROMPT, "qa_templates": PROMPTS, |
| chapters/30-instruction-tuning.md:491 | tests/test_model.py | 1494 |  | reattributed | 'qa_packing_masks_prompts_and_keeps_examples_whole' is defined there |  |
| chapters/30-instruction-tuning.md:491 | tests/test_model.py | 1494 | 1511 | def | 'qa_packing_masks_prompts_and_keeps_examples_whole' | def qa_packing_masks_prompts_and_keeps_examples_whole(): |
| chapters/30-instruction-tuning.md:581 | anulm/training/finetune.py | 222–422 | 222–423 | same | def main(): | def main(): |
| chapters/30-instruction-tuning.md:581 | anulm/training/finetune.py | 306–319 | 307–320 | same | # A held-out loss over questions the model trained on is optimistic, a | # A held-out loss over questions the model trained on is optimistic, a |
| chapters/30-instruction-tuning.md:581 | anulm/training/finetune.py | 306–319 | 307–320 | sweep | 'main' 222–422 (the held-out guard 306–319, the |  |
| chapters/30-instruction-tuning.md:581 | anulm/training/finetune.py | 359–374 | 360–375 | same | for step in range(start_step, end): | for step in range(start_step, end): |
| chapters/30-instruction-tuning.md:581 | anulm/training/finetune.py | 359–374 | 360–375 | sweep | 22–422 (the held-out guard 306–319, the step 359–374, 'qa |  |
| chapters/30-instruction-tuning.md:581 | anulm/training/finetune.py | 385 | 386 | same | "val_loss": val, "qa_template": PROMPT, "qa_templates": PROMPTS, | "val_loss": val, "qa_template": PROMPT, "qa_templates": PROMPTS, |
| chapters/30-instruction-tuning.md:582 | anulm/data/make_chat.py | 176 | 176 | sweep | chat_rows' 119–145, 'main' 148–184 (the call 176). |  |
| chapters/30-instruction-tuning.md:583 | anulm/multimodal/omni_eval.py | 192–195 | ? | CHECK | held = HERE / "data" / "chat_heldout.jsonl" | 'chat_loss' spans [(114, 131, 115)] at the old pin |
| chapters/30-instruction-tuning.md:583 | anulm/multimodal/omni_eval.py | 192–195 | 192–195 | sweep | omni_eval.py': 'chat_loss' 115–131, its call 192–195; |  |
| chapters/30-instruction-tuning.md:585 | anulm/data/make_chat.py | 109–112 | ? | CHECK | return [] | 'main' spans [(148, 184, 148)] at the old pin |
| chapters/30-instruction-tuning.md:587 | anulm/model.py | 910–916 | 922–928 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/30-instruction-tuning.md:587 | anulm/model.py | 914–915 | 926–927 | same | if aux_total is not None: | if aux_total is not None: |
| chapters/30-instruction-tuning.md:587 | anulm/model.py | 914–915 | 926–927 | sweep | 910–916 ('ignore_index=-1'; the regulariser 914–915). |  |
| chapters/30-instruction-tuning.md:588 | tests/test_model.py | 1494 | 1511 | same | def qa_packing_masks_prompts_and_keeps_examples_whole(): | def qa_packing_masks_prompts_and_keeps_examples_whole(): |
| chapters/31-lora.md:192 | anulm/training/finetune.py | 392–399 | 393–400 | same | if args.lora: | if args.lora: |
| chapters/31-lora.md:286 | anulm/lora.py | 196–204 | 201–209 | same | model.load_state_dict(base["model"]) | model.load_state_dict(base["model"]) |
| chapters/31-lora.md:311 | tests/test_model.py | 1047–1061 | 1056–1070 | same | model = AnuLM(cfg).eval() | model = AnuLM(cfg).eval() |
| chapters/31-lora.md:334 | tests/test_model.py | 1102–1106 | 1111–1115 | same | trained_bias = router[sorted(router)[0]] | trained_bias = router[sorted(router)[0]] |
| chapters/31-lora.md:329 | tests/test_model.py | 1021 | 1030 | def | 'lora_starts_as_the_base_model_and_merges_back_exactly' | def lora_starts_as_the_base_model_and_merges_back_exactly(): |
| chapters/31-lora.md:376 | anulm/tasks/decide.py | 247–286 | 249–288 | same | class LoRA(nn.Module): | class LoRA(nn.Module): |
| chapters/31-lora.md:461 | anulm/lora.py | 181–214 | 181–219 | same | def _merge_cli(args) -> int: | def _merge_cli(args) -> int: |
| chapters/31-lora.md:462 | anulm/training/finetune.py | 392–399 | 393–400 | same | if args.lora: | if args.lora: |
| chapters/31-lora.md:462 | anulm/training/finetune.py | 392–399 | 393–400 | sweep | - 'anulm/training/finetune.py': adapter save 392–399; 'up |  |
| chapters/31-lora.md:462 | course/build/anulm.py | 374 |  | reattributed | 'update_expert_biases' is in course/build/anulm.py, not anulm/training |  |
| chapters/31-lora.md:462 | anulm/training/finetune.py | 374 | 374 | sweep | er save 392–399; 'update_expert_biases' call 374. |  |
| chapters/31-lora.md:463 | anulm/model.py | 1059–1089 | 1071–1101 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/31-lora.md:464 | anulm/tasks/decide.py | 247–262 | 249–264 | same | class LoRA(nn.Module): | class LoRA(nn.Module): |
| chapters/31-lora.md:464 | anulm/tasks/decide.py | 265–286 | 267–288 | same | class LoRAQV(nn.Module): | class LoRAQV(nn.Module): |
| chapters/31-lora.md:465 | tests/test_model.py | 1021 | 1030 | same | def lora_starts_as_the_base_model_and_merges_back_exactly(): | def lora_starts_as_the_base_model_and_merges_back_exactly(): |
| chapters/32-classification-and-calibration.md:21 | anulm/tasks/decide.py | 90–99 | 89–101 | same | def encode(tok, text: str, options: list[str], max_len: int = 1024): | def encode(tok, text: str, options: list[str], max_len: int = 1024): |
| chapters/32-classification-and-calibration.md:21 | anulm/tasks/decide.py | 89–101 |  | EXCERPT | option k's name. Pieces are encoded separately so the positions are ex | quoted line not in the new range |
| chapters/32-classification-and-calibration.md:21 | anulm/tasks/decide.py | 89–101 |  | EXCERPT | assert len(ids) <= max_len, f"{len(ids)} tokens; too many options for  | quoted line not in the new range |
| chapters/32-classification-and-calibration.md:60 | anulm/tasks/decide.py | 238–244 | 240–246 | same | def forward(self, idx, pos, mask=None): | def forward(self, idx, pos, mask=None): |
| chapters/32-classification-and-calibration.md:79 | anulm/tasks/decide.py | 289–304 | 291–306 | same | def batchify(examples, tok, pad, with_mask=False): | def batchify(examples, tok, pad, with_mask=False): |
| chapters/32-classification-and-calibration.md:98 | anulm/tasks/decide.py | 198–216 | 200–218 | same | def _hidden(self, idx): | def _hidden(self, idx): |
| chapters/32-classification-and-calibration.md:179 | anulm/model.py | 472–474 | 479–481 | same | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio |
| chapters/32-classification-and-calibration.md:204 | anulm/tasks/decide.py | 366–372 | 368–374 | same | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: |
| chapters/32-classification-and-calibration.md:225 | anulm/tasks/decide.py | 375–395 | 377–397 | same | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d |
| chapters/32-classification-and-calibration.md:320 | tests/test_model.py | 1695–1708 | 1712–1725 | same | ids, pos = D.encode(tok, "where is my card", opts) | ids, pos = D.encode(tok, "where is my card", opts) |
| chapters/32-classification-and-calibration.md:37 | anulm/tasks/decide.py | 84 | 83 | same | def readable(label: str) -> str: | def readable(label: str) -> str: |
| chapters/32-classification-and-calibration.md:74 | anulm/tasks/decide.py | 118 | ? | CHECK | self.head = nn.Linear(hid, 1) | no file named on the line and nothing quoted confirms it |
| chapters/32-classification-and-calibration.md:76 | anulm/tasks/decide.py | 594 | ? | CHECK | loss = F.cross_entropy(logits, y.to(a.device)) | no file named on the line and nothing quoted confirms it |
| chapters/32-classification-and-calibration.md:187 | anulm/bidirectional.py | 56–105 |  | reattributed | 'adapt' is in anulm/bidirectional.py, not anulm/model.py |  |
| chapters/32-classification-and-calibration.md:218 | anulm/tasks/decide.py | 609 | 611 | same | t = fit_temperature(lv, yv) | t = fit_temperature(lv, yv) |
| chapters/32-classification-and-calibration.md:219 | anulm/model.py | 690 |  | reattributed | quoted code is in anulm/model.py, not anulm/tasks/decide.py |  |
| chapters/32-classification-and-calibration.md:219 | anulm/model.py | 690 | 697 | same | """Returns (y, aux): the routed + shared output, and the router's | """Returns (y, aux): the routed + shared output, and the router's |
| chapters/32-classification-and-calibration.md:339 | tests/test_model.py | 1685 | 1702 | def | 'decide_scores_each_option_at_its_last_token_and_calibrates' | def decide_scores_each_option_at_its_last_token_and_calibrates(): |
| chapters/32-classification-and-calibration.md:402 | anulm/tasks/decide.py | 84 | 83 | same | def readable(label: str) -> str: | def readable(label: str) -> str: |
| chapters/32-classification-and-calibration.md:402 | anulm/tasks/decide.py | 90–99 | 89–101 | same | def encode(tok, text: str, options: list[str], max_len: int = 1024): | def encode(tok, text: str, options: list[str], max_len: int = 1024): |
| chapters/32-classification-and-calibration.md:402 | anulm/tasks/decide.py | 103–244 | 105–246 | same | class Decider(nn.Module): | class Decider(nn.Module): |
| chapters/32-classification-and-calibration.md:402 | anulm/tasks/decide.py | 118 | 120 | same | self.head = nn.Linear(hid, 1) | self.head = nn.Linear(hid, 1) |
| chapters/32-classification-and-calibration.md:403 | anulm/tasks/decide.py | 198–216 | 200–218 | same | def _hidden(self, idx): | def _hidden(self, idx): |
| chapters/32-classification-and-calibration.md:403 | anulm/tasks/decide.py | 238–244 | 240–246 | same | def forward(self, idx, pos, mask=None): | def forward(self, idx, pos, mask=None): |
| chapters/32-classification-and-calibration.md:403 | anulm/tasks/decide.py | 247–262 | 249–264 | same | class LoRA(nn.Module): | class LoRA(nn.Module): |
| chapters/32-classification-and-calibration.md:403 | anulm/tasks/decide.py | 289 | 291 | same | def batchify(examples, tok, pad, with_mask=False): | def batchify(examples, tok, pad, with_mask=False): |
| chapters/32-classification-and-calibration.md:404 | anulm/tasks/decide.py | 307 | 309 | same | def make_examples(rows, tok, labs, rng=None): | def make_examples(rows, tok, labs, rng=None): |
| chapters/32-classification-and-calibration.md:404 | anulm/tasks/decide.py | 366–372 | 368–374 | same | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: |
| chapters/32-classification-and-calibration.md:404 | anulm/tasks/decide.py | 375–395 | 377–397 | same | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d |
| chapters/32-classification-and-calibration.md:405 | anulm/tasks/decide.py | 543–614 | 545–616 | same | def train(a): | def train(a): |
| chapters/32-classification-and-calibration.md:405 | anulm/tasks/decide.py | 543–614 | 545–616 | sweep | 543–614 (los |  |
| chapters/32-classification-and-calibration.md:405 | anulm/tasks/decide.py | 594 | 596 | same | loss = F.cross_entropy(logits, y.to(a.device)) | loss = F.cross_entropy(logits, y.to(a.device)) |
| chapters/32-classification-and-calibration.md:405 | anulm/tasks/decide.py | 594 | 596 | sweep | 543–614 (loss 594, tem |  |
| chapters/32-classification-and-calibration.md:405 | anulm/tasks/decide.py | 609 | 611 | same | t = fit_temperature(lv, yv) | t = fit_temperature(lv, yv) |
| chapters/32-classification-and-calibration.md:405 | anulm/tasks/decide.py | 609 | 611 | sweep | 543–614 (loss 594, temperature 609), 'D |  |
| chapters/32-classification-and-calibration.md:405 | anulm/tasks/decide.py | 687–697 | 691–704 | same | def __call__(self, message: str, options: list[str] \| None = None, top | def __call__(self, message: str, options: list[str] \| None = None, top |
| chapters/32-classification-and-calibration.md:407 | anulm/model.py | 472–474 | 479–481 | same | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio |
| chapters/32-classification-and-calibration.md:408 | tests/test_model.py | 1685 | 1702 | same | def decide_scores_each_option_at_its_last_token_and_calibrates(): | def decide_scores_each_option_at_its_last_token_and_calibrates(): |
| chapters/33-retrieval-and-readers.md:94 | anulm/inference/rag.py | 284–304 | 288–308 | same | scores = np.zeros(self.n, dtype=np.float32) | scores = np.zeros(self.n, dtype=np.float32) |
| chapters/33-retrieval-and-readers.md:185 | anulm/inference/rag.py | 263–277 | 267–281 | same | mode = mode or self.mode | mode = mode or self.mode |
| chapters/33-retrieval-and-readers.md:185 | anulm/inference/rag.py | 267 | 271 | edited-quote | self.dense = Dense() | self.dense = Dense(self.device) |
| chapters/33-retrieval-and-readers.md:249 | anulm/inference/rag.py | 340–347 | 344–351 | same | def prompt(self, question: str, passages: list[str]) -> tuple[str, str | def prompt(self, question: str, passages: list[str]) -> tuple[str, str |
| chapters/33-retrieval-and-readers.md:285 | anulm/tasks/rc_pointer.py | 61–73 | 60–72 | same | def forward(self, idx): | def forward(self, idx): |
| chapters/33-retrieval-and-readers.md:305 | anulm/tasks/rc_pointer.py | 188–193 | 185–190 | same | def loss_of(items): | def loss_of(items): |
| chapters/33-retrieval-and-readers.md:319 | anulm/tasks/rc_pointer.py | 83–103 | 82–102 | edited | def encode(tok, qword, q, body, answer=None): | def encode(tok, qword, q, body, answer=None, max_len: int = MAX_LEN): |
| chapters/33-retrieval-and-readers.md:319 | anulm/tasks/rc_pointer.py | 83 | 82 | edited-quote | def encode(tok, qword, q, body, answer=None): | def encode(tok, qword, q, body, answer=None, max_len: int = MAX_LEN): |
| chapters/33-retrieval-and-readers.md:319 | anulm/tasks/rc_pointer.py | 99 | 98 | edited-quote | if en < MAX_LEN and span: | if en < max_len and span: |
| chapters/33-retrieval-and-readers.md:319 | anulm/tasks/rc_pointer.py | 100 | 99 | edited-quote | return ids[:MAX_LEN], st, en | return ids[:max_len], st, en |
| chapters/33-retrieval-and-readers.md:319 | anulm/tasks/rc_pointer.py | 103 | 102 | edited-quote | return (head + tok.encode(body))[:MAX_LEN], 0, 0 | return (head + tok.encode(body))[:max_len], 0, 0 |
| chapters/33-retrieval-and-readers.md:389 | anulm/inference/rag.py | 374–396 | 378–400 | same | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") |
| chapters/33-retrieval-and-readers.md:181 | anulm/inference/rag.py | 213 |  | reattributed | quoted code is in anulm/inference/rag.py, not None |  |
| chapters/33-retrieval-and-readers.md:236 | anulm/inference/rag.py | 252 | 256 | same | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" |
| chapters/33-retrieval-and-readers.md:237 | anulm/inference/rag.py | 250 | ? | CHECK | # the top 3 every time (BM25: 15/20) and took our own reader from 0/20 | no file named on the line and nothing quoted confirms it |
| chapters/33-retrieval-and-readers.md:272 | anulm/inference/rag.py | 350–360 | ? | CHECK | def read(self, prompt: str, max_tokens: int = 32) -> str: | no file named on the line and nothing quoted confirms it |
| chapters/33-retrieval-and-readers.md:497 | anulm/inference/rag.py | 42–43 | 42–43 | sweep | - 'anulm/inference/rag.py': constants 42–43, 'wo |  |
| chapters/33-retrieval-and-readers.md:497 | anulm/inference/rag.py | 91 | ? | CHECK | ids = term_ids(title + " " + chunk) | 'term_ids' spans [(57, 60, 57)] at the old pin |
| chapters/33-retrieval-and-readers.md:497 | anulm/inference/rag.py | 91 | 91 | sweep | rds' 51, 'term_ids' 57, title + text indexed 91 (in |  |
| chapters/33-retrieval-and-readers.md:498 | anulm/inference/rag.py | 374–396 | 378–400 | same | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") |
| chapters/33-retrieval-and-readers.md:499 | anulm/inference/rag.py | 136–145 | 136–145 | sweep | 136–145), 'D |  |
| chapters/33-retrieval-and-readers.md:499 | anulm/inference/rag.py | 233–304 | 233–308 | same | class Retriever: | class Retriever: |
| chapters/33-retrieval-and-readers.md:500 | anulm/inference/rag.py | 252 | 256 | same | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" |
| chapters/33-retrieval-and-readers.md:500 | anulm/inference/rag.py | 252 | 256 | sweep | mode 252, 'se |  |
| chapters/33-retrieval-and-readers.md:500 | anulm/inference/rag.py | 257–277 | 261–281 | same | def search(self, query: str, k: int = 3, lang: str \| None = None, | def search(self, query: str, k: int = 3, lang: str \| None = None, |
| chapters/33-retrieval-and-readers.md:500 | anulm/inference/rag.py | 279–304 | 283–308 | same | def bm25(self, query: str, k: int = 3, lang: str \| None = None, | def bm25(self, query: str, k: int = 3, lang: str \| None = None, |
| chapters/33-retrieval-and-readers.md:500 | anulm/inference/rag.py | 320–371 | 324–375 | same | class RagQA: | class RagQA: |
| chapters/33-retrieval-and-readers.md:501 | anulm/tasks/rc_pointer.py | 51–73 | 49–72 | same | class Pointer(nn.Module): | class Pointer(nn.Module): |
| chapters/33-retrieval-and-readers.md:501 | anulm/tasks/rc_pointer.py | 83–103 | 82–102 | edited | def encode(tok, qword, q, body, answer=None): | def encode(tok, qword, q, body, answer=None, max_len: int = MAX_LEN): |
| chapters/33-retrieval-and-readers.md:502 | anulm/tasks/rc_pointer.py | 157–232 | 155–229 | same | def train(a): | def train(a): |
| chapters/33-retrieval-and-readers.md:502 | anulm/tasks/rc_pointer.py | 188–193 | 185–190 | same | def loss_of(items): | def loss_of(items): |
| chapters/33-retrieval-and-readers.md:502 | anulm/tasks/rc_pointer.py | 188–193 | 185–190 | sweep | 'train' 157–232 (loss 188–193). |  |
| chapters/34-preference-optimisation-and-rl.md:65 | anulm/model.py | 903–909 | 915–921 | same | if targets is None: | if targets is None: |
| chapters/34-preference-optimisation-and-rl.md:319 | anulm/model.py | 1149–1157 | 1161–1169 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/34-preference-optimisation-and-rl.md:407 | anulm/model.py | 903–909 | 915–921 | same | if targets is None: | if targets is None: |
| chapters/34-preference-optimisation-and-rl.md:407 | anulm/model.py | 1149–1157 | 1161–1169 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/35-wider-vocabulary.md:42 | anulm/multimodal/speech_vocab.py | 62–87 | 60–85 | same | CONTROL = ["<\|speech\|>", "<\|/speech\|>", "<\|text\|>", "<\|image\|>", "<\|/i | CONTROL = ["<\|speech\|>", "<\|/speech\|>", "<\|text\|>", "<\|image\|>", "<\|/i |
| chapters/35-wider-vocabulary.md:126 | anulm/multimodal/speech_vocab.py | 110–116 | 108–114 | same | def _grow(weight: torch.Tensor, n_new: int, gen: torch.Generator) -> t | def _grow(weight: torch.Tensor, n_new: int, gen: torch.Generator) -> t |
| chapters/35-wider-vocabulary.md:143 | anulm/multimodal/speech_vocab.py | 142–158 | 140–156 | same | gen = torch.Generator().manual_seed(seed) | gen = torch.Generator().manual_seed(seed) |
| chapters/35-wider-vocabulary.md:237 | anulm/model.py | 875–884 | 887–896 | same | # 'embeds' bypasses the embedding table: (B, S, D) straight in. Audio | # 'embeds' bypasses the embedding table: (B, S, D) straight in. Audio |
| chapters/35-wider-vocabulary.md:264 | anulm/multimodal/speech_encoder.py | 106–135 | 105–134 | same | def build(self, states: torch.Tensor, text_ids: list[int], answer_ids: | def build(self, states: torch.Tensor, text_ids: list[int], answer_ids: |
| chapters/35-wider-vocabulary.md:343 | anulm/model.py | 1140–1144 | 1152–1156 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/35-wider-vocabulary.md:357 | anulm/inference/serve.py | 61–63 | 66–68 | same | self.model = AnuLM(cfg).to(device).eval() | self.model = AnuLM(cfg).to(device).eval() |
| chapters/35-wider-vocabulary.md:396 | anulm/model.py | 1021–1045 | 1033–1057 | same | dev = prefix.device | dev = prefix.device |
| chapters/35-wider-vocabulary.md:422 | anulm/multimodal/speech_encoder.py | 384–405 | 383–404 | same | texts = [] | texts = [] |
| chapters/35-wider-vocabulary.md:62 | anulm/multimodal/speech_vocab.py | 58–61 |  | reattributed | quoted code is in anulm/multimodal/speech_vocab.py, not anulm/multimod |  |
| chapters/35-wider-vocabulary.md:62 | anulm/multimodal/speech_vocab.py | 58–61 | 56–59 | same | # Image tokens are here, in the speech file, because the vocabulary is | # Image tokens are here, in the speech file, because the vocabulary is |
| chapters/35-wider-vocabulary.md:253 | anulm/model.py | 884 | ? | CHECK | x = F.embedding(idx, self._table("in"))               # (B, S, D) | no file named on the line and nothing quoted confirms it |
| chapters/35-wider-vocabulary.md:333 | anulm/multimodal/speech_encoder.py | 288 | 287 | same | _, loss = model(None, tb, embeds=eb) | _, loss = model(None, tb, embeds=eb) |
| chapters/35-wider-vocabulary.md:458 | tests/test_model.py | 1560 | 1577 | def | 'cached_embeds_decoding_matches_prefix_replay' | def cached_embeds_decoding_matches_prefix_replay(): |
| chapters/35-wider-vocabulary.md:516 | anulm/multimodal/speech_vocab.py | 62–63 | 60–61 | same | CONTROL = ["<\|speech\|>", "<\|/speech\|>", "<\|text\|>", "<\|image\|>", "<\|/i | CONTROL = ["<\|speech\|>", "<\|/speech\|>", "<\|text\|>", "<\|image\|>", "<\|/i |
| chapters/35-wider-vocabulary.md:516 | anulm/multimodal/speech_vocab.py | 71–107 | 69–105 | same | class SpeechVocab: | class SpeechVocab: |
| chapters/35-wider-vocabulary.md:516 | anulm/multimodal/speech_vocab.py | 110–116 | 108–114 | same | def _grow(weight: torch.Tensor, n_new: int, gen: torch.Generator) -> t | def _grow(weight: torch.Tensor, n_new: int, gen: torch.Generator) -> t |
| chapters/35-wider-vocabulary.md:517 | anulm/multimodal/speech_vocab.py | 119–161 | 117–159 | same | def grow_checkpoint(ckpt: str, out: str, seed: int = 0, device: str =  | def grow_checkpoint(ckpt: str, out: str, seed: int = 0, device: str =  |
| chapters/35-wider-vocabulary.md:520 | anulm/model.py | 865–916 | 877–928 | same | def forward(self, idx, targets=None, cache: Optional[list] = None, sta | def forward(self, idx, targets=None, cache: Optional[list] = None, sta |
| chapters/35-wider-vocabulary.md:520 | anulm/model.py | 1003–1050 | 1015–1062 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/35-wider-vocabulary.md:521 | anulm/model.py | 1103–1168 | 1115–1180 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/35-wider-vocabulary.md:522 | anulm/multimodal/speech_encoder.py | 106–135 | 105–134 | same | def build(self, states: torch.Tensor, text_ids: list[int], answer_ids: | def build(self, states: torch.Tensor, text_ids: list[int], answer_ids: |
| chapters/35-wider-vocabulary.md:523 | anulm/multimodal/speech_encoder.py | 288 | 287 | same | _, loss = model(None, tb, embeds=eb) | _, loss = model(None, tb, embeds=eb) |
| chapters/35-wider-vocabulary.md:523 | anulm/multimodal/speech_encoder.py | 288 | 287 | sweep | 288, 'tr |  |
| chapters/35-wider-vocabulary.md:523 | anulm/multimodal/speech_encoder.py | 361–405 | 360–404 | same | def transcribe(sm, enc, tok, wav_path: str, max_tokens: int = 64, devi | def transcribe(sm, enc, tok, wav_path: str, max_tokens: int = 64, devi |
| chapters/35-wider-vocabulary.md:524 | anulm/inference/serve.py | 53–107 | 58–112 | same | def __init__(self, ckpt: str, device: str): | def __init__(self, ckpt: str, device: str): |
| chapters/35-wider-vocabulary.md:525 | tests/test_model.py | 1560 | 1577 | same | def cached_embeds_decoding_matches_prefix_replay(): | def cached_embeds_decoding_matches_prefix_replay(): |
| chapters/36-frozen-encoders.md:117 | anulm/multimodal/speech_encoder.py | 76–93 | 75–92 | same | def __init__(self, in_dim: int, out_dim: int, stack: int = STACK, hidd | def __init__(self, in_dim: int, out_dim: int, stack: int = STACK, hidd |
| chapters/36-frozen-encoders.md:359 | anulm/multimodal/vision_encoder.py | 486–489 | 485–488 | same | opt = torch.optim.AdamW( | opt = torch.optim.AdamW( |
| chapters/36-frozen-encoders.md:463 | anulm/multimodal/speech_encoder.py | 198–204 | 197–203 | same | for r in chunk: | for r in chunk: |
| chapters/36-frozen-encoders.md:477 | anulm/multimodal/speech_encoder.py | 56–57 | 55–56 | same | STACK = 4                      # 50 Hz -> 12.5 Hz | STACK = 4                      # 50 Hz -> 12.5 Hz |
| chapters/36-frozen-encoders.md:489 | anulm/multimodal/speech_encoder.py | 160–183 | 159–182 | same | WHISPER_SECONDS = 30.0      # the encoder's window; WhisperEncoder.enc | WHISPER_SECONDS = 30.0      # the encoder's window; WhisperEncoder.enc |
| chapters/36-frozen-encoders.md:138 | anulm/multimodal/speech_encoder.py | 56 | 55 | same | STACK = 4                      # 50 Hz -> 12.5 Hz | STACK = 4                      # 50 Hz -> 12.5 Hz |
| chapters/36-frozen-encoders.md:344 | anulm/multimodal/omni_train.py | 501 | 502 | same | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", |
| chapters/36-frozen-encoders.md:369 | anulm/multimodal/vision_encoder.py | 654–655 | 653–654 | same | t.add_argument("--lr", type=float, default=5e-5) | t.add_argument("--lr", type=float, default=5e-5) |
| chapters/36-frozen-encoders.md:515 | anulm/multimodal/speech_encoder.py | 250 | 216 | def | 'train' | def train(args) -> None: |
| chapters/36-frozen-encoders.md:590 | anulm/multimodal/vision_encoder.py | 443–535 | 443–534 | same | def train(args) -> None: | def train(args) -> None: |
| chapters/36-frozen-encoders.md:591 | anulm/multimodal/vision_encoder.py | 486–489 | 485–488 | same | opt = torch.optim.AdamW( | opt = torch.optim.AdamW( |
| chapters/36-frozen-encoders.md:591 | anulm/multimodal/vision_encoder.py | 486–489 | 485–488 | sweep | (optimiser 486–489, def |  |
| chapters/36-frozen-encoders.md:591 | anulm/multimodal/vision_encoder.py | 654–655 | 653–654 | same | t.add_argument("--lr", type=float, default=5e-5) | t.add_argument("--lr", type=float, default=5e-5) |
| chapters/36-frozen-encoders.md:591 | anulm/multimodal/vision_encoder.py | 654–655 | 653–654 | sweep | (optimiser 486–489, defaults 654–655). |  |
| chapters/36-frozen-encoders.md:592 | anulm/multimodal/speech_encoder.py | 56 | 55 | same | STACK = 4                      # 50 Hz -> 12.5 Hz | STACK = 4                      # 50 Hz -> 12.5 Hz |
| chapters/36-frozen-encoders.md:592 | anulm/multimodal/speech_encoder.py | 65–93 | 64–92 | same | class AudioProjector(nn.Module): | class AudioProjector(nn.Module): |
| chapters/36-frozen-encoders.md:592 | anulm/multimodal/speech_encoder.py | 160 | 159 | same | WHISPER_SECONDS = 30.0      # the encoder's window; WhisperEncoder.enc | WHISPER_SECONDS = 30.0      # the encoder's window; WhisperEncoder.enc |
| chapters/36-frozen-encoders.md:593 | anulm/multimodal/speech_encoder.py | 163–169 | 162–168 | same | def clip_seconds(row: dict) -> float: | def clip_seconds(row: dict) -> float: |
| chapters/36-frozen-encoders.md:593 | anulm/multimodal/speech_encoder.py | 172–183 | 171–182 | same | def whisper_window_rows(rows: list[dict], name: str = "asr") -> list[d | def whisper_window_rows(rows: list[dict], name: str = "asr") -> list[d |
| chapters/36-frozen-encoders.md:593 | anulm/multimodal/speech_encoder.py | 186–214 | 185–213 | same | def batches(rows, enc: WhisperEncoder, tok, sm: ContinuousSpeech, devi | def batches(rows, enc: WhisperEncoder, tok, sm: ContinuousSpeech, devi |
| chapters/36-frozen-encoders.md:593 | anulm/multimodal/speech_encoder.py | 250 | 249 | same | rows = whisper_window_rows(load_manifest(args.prefix), args.prefix) | rows = whisper_window_rows(load_manifest(args.prefix), args.prefix) |
| chapters/36-frozen-encoders.md:593 | anulm/multimodal/speech_encoder.py | 250 | 249 | sweep | ' 172–183, 'batches' 186–214, 'train''s call 250. |  |
| chapters/37-one-model-many-tasks.md:235 | anulm/multimodal/omni_train.py | 422–451 | 423–452 | same | w = np.array([t.weight for t in om.tasks], dtype=float) | w = np.array([t.weight for t in om.tasks], dtype=float) |
| chapters/37-one-model-many-tasks.md:299 | anulm/multimodal/omni_train.py | 365–402 | 365–403 | same | # Fresh projectors (vision always, audio without --audio-proj-from) ar | # Fresh projectors (vision always, audio without --audio-proj-from) ar |
| chapters/37-one-model-many-tasks.md:299 | anulm/multimodal/omni_train.py | 365–403 |  | EXCERPT | st = torch.load(last, map_location=dev, weights_only=False) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:353 | anulm/multimodal/omni_train.py | 457–464 | 458–465 | same | if step % args.eval_every == 0 or step == args.steps: | if step % args.eval_every == 0 or step == args.steps: |
| chapters/37-one-model-many-tasks.md:77 | anulm/multimodal/omni_train.py | 104–111 and 161–174 |  | reattributed | quoted code is in anulm/multimodal/omni_train.py, not None |  |
| chapters/37-one-model-many-tasks.md:202 | anulm/multimodal/omni_train.py | 33–37 | ? | CHECK | **Held out.** The first 2% of distinct input groups (normalized media  | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:206 | experiments/omni_one.py | 77–87 | ? | CHECK | TASKS = [ | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:290 | anulm/multimodal/omni_train.py | 504 | 505 | same | p.add_argument("--accum", type=int, default=1, help="batches summed in | p.add_argument("--accum", type=int, default=1, help="batches summed in |
| chapters/37-one-model-many-tasks.md:290 | anulm/multimodal/omni_train.py | 516–517 | 517–518 | same | if args.accum < 1: | if args.accum < 1: |
| chapters/37-one-model-many-tasks.md:293 | anulm/multimodal/omni_train.py | 493–494 | 494–495 | same | p.add_argument("--lr", type=float, default=5e-5) | p.add_argument("--lr", type=float, default=5e-5) |
| chapters/37-one-model-many-tasks.md:334 | anulm/multimodal/omni_train.py | 474 | ? | CHECK | st.update(opt=opt.state_dict(), first=first, best=best, seen=seen, rec | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:339 | anulm/multimodal/omni_train.py | 403–413 | ? | CHECK | model.load_state_dict(st["model"]) | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:341 | anulm/multimodal/omni_train.py | 505 | ? | CHECK | p.add_argument("--stop-after", type=int, default=0, | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:343 | anulm/multimodal/omni_train.py | 470–471 | ? | CHECK | if args.stop_after and step >= args.stop_after: | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:366 | anulm/multimodal/omni_train.py | 415 | 416 | same | first = evaluate() | first = evaluate() |
| chapters/37-one-model-many-tasks.md:367 | anulm/multimodal/omni_train.py | 416–417 | 417–418 | same | if any(not math.isfinite(value) or value <= 0 for value in first.value | if any(not math.isfinite(value) or value <= 0 for value in first.value |
| chapters/37-one-model-many-tasks.md:513 | experiments/omni_one.py | 94–97 | ? | CHECK | # Hindi: the real-speaker data (Kathbath, MUCS) trades FLEURS for ever | no file named on the line and nothing quoted confirms it |
| chapters/37-one-model-many-tasks.md:722 | anulm/multimodal/omni_train.py | 85–101 | 85–101 | sweep | 85–101, 'pa |  |
| chapters/37-one-model-many-tasks.md:726 | anulm/multimodal/omni_train.py | 362–478 | 362–479 | same | def train(args) -> None: | def train(args) -> None: |
| chapters/37-one-model-many-tasks.md:726 | anulm/multimodal/omni_train.py | 481–518 | 482–519 | same | def main() -> None: | def main() -> None: |
| chapters/37-one-model-many-tasks.md:730 | experiments/omni_one.py | 460–469 | 460–469 | sweep | command 460–469. |  |
| chapters/38-serving.md:37 | anulm/inference/serve.py | 53–63 | 58–68 | same | def __init__(self, ckpt: str, device: str): | def __init__(self, ckpt: str, device: str): |
| chapters/38-serving.md:59 | anulm/inference/serve.py | 105–107 | 110–112 | same | self.lock = threading.Lock() | self.lock = threading.Lock() |
| chapters/38-serving.md:74 | anulm/inference/serve.py | 147–161 | 152–166 | same | ids = self.encode(prompt)[-(self.cfg.block_size - 1):] | ids = self.encode(prompt)[-(self.cfg.block_size - 1):] |
| chapters/38-serving.md:116 | anulm/inference/serve.py | 198–218 | ? | UNMAPPED | def do_POST(self): |  |
| chapters/38-serving.md:116 | anulm/inference/serve.py | 198 | ? | EXCERPT | header range unresolved: make_handler |  |
| chapters/38-serving.md:154 | anulm/inference/api.py | 192–198 | 218–224 | same | temperature: Optional[float] = Field(None, ge=0, le=2, description="De | temperature: Optional[float] = Field(None, ge=0, le=2, description="De |
| chapters/38-serving.md:170 | anulm/inference/api.py | 518–523 | 570–575 | same | users = [m for m in req.messages if m.role == "user"] | users = [m for m in req.messages if m.role == "user"] |
| chapters/38-serving.md:186 | anulm/inference/api.py | 264–265 | 295–296 | same | models: A.Models \| None = None | models: A.Models \| None = None |
| chapters/38-serving.md:208 | anulm/inference/api.py | 383–401 | 430–448 | same | def on_token(nxt): | def on_token(nxt): |
| chapters/38-serving.md:236 | anulm/inference/api.py | 360–376 | 407–423 | same | def _release(self, final: bool) -> str: | def _release(self, final: bool) -> str: |
| chapters/38-serving.md:286 | anulm/inference/serve.py | 124–132 | 129–137 | same | def decode_complete(self, ids) -> str: | def decode_complete(self, ids) -> str: |
| chapters/38-serving.md:335 | anulm/inference/api.py | 402–409 | 449–456 | same | # Release the held-back tail unless a stop string ended the reply. End | # Release the held-back tail unless a stop string ended the reply. End |
| chapters/38-serving.md:369 | anulm/inference/api.py | 335–340 | 381–386 | same | def stops_of(stop) -> list[str]: | def stops_of(stop) -> list[str]: |
| chapters/38-serving.md:410 | anulm/inference/api.py | 428–462 | 476–510 | same | q: queue.Queue = queue.Queue() | q: queue.Queue = queue.Queue() |
| chapters/38-serving.md:410 | anulm/inference/api.py | 435 | 483 | edited-quote | q.put(("error", f"{type(e).__name__}: {e}")) | q.put(("error", internal_error_message(log_error("streaming generation |
| chapters/38-serving.md:109 | anulm/inference/serve.py | 147 | ? | CHECK | ids = self.encode(prompt)[-(self.cfg.block_size - 1):] | no file named on the line and nothing quoted confirms it |
| chapters/38-serving.md:110 | anulm/inference/serve.py | 141–146 | ? | CHECK | if mode == "translate" and self.qa_templates and "en-hi" in self.qa_te | no file named on the line and nothing quoted confirms it |
| chapters/38-serving.md:144 | anulm/inference/serve.py | 242 | 264 | def | 'ThreadingHTTPServer' | server = ThreadingHTTPServer((args.host, args.port), make_handler(engi |
| chapters/38-serving.md:206 | anulm/model.py | 1162–1163 |  | reattributed | quoted code is in anulm/model.py, not None |  |
| chapters/38-serving.md:206 | anulm/model.py | 1162–1163 | 1174–1175 | same | if on_token is not None and on_token(nxt): | if on_token is not None and on_token(nxt): |
| chapters/38-serving.md:323 | anulm/inference/api.py | 488–505 |  | reattributed | 'one_shot_reply' is in anulm/inference/api.py, not None |  |
| chapters/38-serving.md:323 | anulm/inference/api.py | 488–505 | 540–557 | same | def one_shot_reply(req: ChatRequest, content: str, extra: dict \| None  | def one_shot_reply(req: ChatRequest, content: str, extra: dict \| None  |
| chapters/38-serving.md:349 | anulm/inference/api.py | 358 | 405 | same | self.matched_stop = False          # a 'stop' string was found (EOS al | self.matched_stop = False          # a 'stop' string was found (EOS al |
| chapters/38-serving.md:474 | anulm/inference/api.py | 515 |  | reattributed | quoted code is in anulm/inference/api.py, not docs/RESULTS.md |  |
| chapters/38-serving.md:474 | anulm/inference/api.py | 515 | 567 | same | raise APIError(400, "Tool calling is not supported: the model was not  | raise APIError(400, "Tool calling is not supported: the model was not  |
| chapters/38-serving.md:493 | anulm/inference/api.py | 352 | 399 | same | self.max_tokens = max(1, min(max_tokens, self.eng.cfg.block_size - len | self.max_tokens = max(1, min(max_tokens, self.eng.cfg.block_size - len |
| chapters/38-serving.md:533 | anulm/inference/serve.py | 50–171 | 55–176 | same | class Engine: | class Engine: |
| chapters/38-serving.md:533 | anulm/inference/serve.py | 53–107 | 58–112 | same | def __init__(self, ckpt: str, device: str): | def __init__(self, ckpt: str, device: str): |
| chapters/38-serving.md:533 | anulm/inference/serve.py | 124–132 | 129–137 | same | def decode_complete(self, ids) -> str: | def decode_complete(self, ids) -> str: |
| chapters/38-serving.md:534 | anulm/inference/serve.py | 134–171 | 139–176 | same | def generate(self, prompt: str, max_tokens: int, temperature: float, t | def generate(self, prompt: str, max_tokens: int, temperature: float, t |
| chapters/38-serving.md:535 | anulm/inference/serve.py | 174–223 | 179–245 | same | def make_handler(engine: Engine): | def make_handler(engine: Engine): |
| chapters/38-serving.md:535 | anulm/inference/serve.py | 226–247 | 248–269 | same | def main(): | def main(): |
| chapters/38-serving.md:536 | anulm/inference/api.py | 186–203 | 212–229 | same | class ChatRequest(BaseModel): | class ChatRequest(BaseModel): |
| chapters/38-serving.md:536 | anulm/inference/api.py | 265 | 296 | same | GPU = threading.Lock()                # one request on the device at a | GPU = threading.Lock()                # one request on the device at a |
| chapters/38-serving.md:536 | anulm/inference/api.py | 335–340 | 381–386 | same | def stops_of(stop) -> list[str]: | def stops_of(stop) -> list[str]: |
| chapters/38-serving.md:537 | anulm/inference/api.py | 343–409 | 389–456 | same | class Generation: | class Generation: |
| chapters/38-serving.md:537 | anulm/inference/api.py | 360–376 | 407–423 | same | def _release(self, final: bool) -> str: | def _release(self, final: bool) -> str: |
| chapters/38-serving.md:537 | anulm/inference/api.py | 378–409 | 425–456 | same | def run(self, emit=None): | def run(self, emit=None): |
| chapters/38-serving.md:538 | anulm/inference/api.py | 488–505 | 540–557 | same | def one_shot_reply(req: ChatRequest, content: str, extra: dict \| None  | def one_shot_reply(req: ChatRequest, content: str, extra: dict \| None  |
| chapters/38-serving.md:539 | anulm/inference/api.py | 423–464 | 471–512 | same | def stream_generation(gen: Generation, chunk, final_chunk, usage_chunk | def stream_generation(gen: Generation, chunk, final_chunk, usage_chunk |
| chapters/38-serving.md:539 | anulm/inference/api.py | 508–586 | 560–638 | same | @app.post("/v1/chat/completions", tags=["OpenAI-compatible"], dependen | @app.post("/v1/chat/completions", tags=["OpenAI-compatible"], dependen |
| chapters/38-serving.md:540 | anulm/model.py | 1103–1168 | 1115–1180 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/38-serving.md:540 | anulm/model.py | 1162–1163 | 1174–1175 | same | if on_token is not None and on_token(nxt): | if on_token is not None and on_token(nxt): |
| chapters/39-exporting.md:58 | anulm/hf/export_hf.py | 51–60 | 52–61 | same | def to_bf16(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: | def to_bf16(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: |
| chapters/39-exporting.md:78 | anulm/hf/export_hf.py | 265–270 | 271–276 | same | n_params, sd, flat_groups = 0, None, {} | n_params, sd, flat_groups = 0, None, {} |
| chapters/39-exporting.md:193 | anulm/hf/export_hf.py | 307–356 | 313–362 | same | if cfg is not None: | if cfg is not None: |
| chapters/39-exporting.md:248 | anulm/model.py | 1221–1241 | 1296–1319 | same | p = _P(path) | p = _P(path) |
| chapters/39-exporting.md:248 | anulm/model.py | 1296–1319 |  | EXCERPT | if not p.exists() and re.fullmatch(r"[\w.-]+/[\w.-]+", str(path).repla | quoted line not in the new range |
| chapters/39-exporting.md:248 | anulm/model.py | 1226 | 1304 | edited-quote | return torch.load(str(p), map_location=map_location, weights_only=Fals | return safe_load(p, map_location) |
| chapters/39-exporting.md:248 | anulm/model.py | 1234 | 1312 | edited-quote | cfg_d["tokenizer_path"] = str(p / cfg_d["tokenizer_path"]) | cfg_d["tokenizer_path"] = _inside(p, cfg_d["tokenizer_path"], "tokeniz |
| chapters/39-exporting.md:289 | anulm/hf/export_hf.py | 114–139 | 115–140 | same | def check_parity(out: Path, weights: dict \| None, groups: dict, tok_sr | def check_parity(out: Path, weights: dict \| None, groups: dict, tok_sr |
| chapters/39-exporting.md:89 | anulm/hf/export_hf.py | 230–232 | 231–233 | same | p.add_argument("--weights-dtype", choices=("mixed", "float32"), defaul | p.add_argument("--weights-dtype", choices=("mixed", "float32"), defaul |
| chapters/39-exporting.md:240 | anulm/hf/export_hf.py | 281–287 | 287–293 | same | if groups: | if groups: |
| chapters/39-exporting.md:280 | tests/test_model.py | 914 | 924 | def | 'an_exported_folder_loads_everywhere_a_pt_does' | def an_exported_folder_loads_everywhere_a_pt_does(): |
| chapters/39-exporting.md:281 | tests/test_model.py | 961–963 | ? | CHECK | for name in ("eval_bench.py", "eval_context.py", "eval_golden.py", "ev | no file named on the line and nothing quoted confirms it |
| chapters/39-exporting.md:286 | anulm/hf/export_hf.py | 372–373 |  | reattributed | 'main' is in anulm/hf/export_hf.py, not tests/test_model.py |  |
| chapters/39-exporting.md:286 | anulm/hf/export_hf.py | 372–373 | 378–379 | same | if not args.skip_parity: | if not args.skip_parity: |
| chapters/39-exporting.md:323 | anulm/hf/export_hf.py | 111 | 112 | same | PARITY_TEXTS = ("Hello, world.", "भारत की राजधानी नई दिल्ली है।", "def | PARITY_TEXTS = ("Hello, world.", "भारत की राजधानी नई दिल्ली है।", "def |
| chapters/39-exporting.md:430 | tests/test_model.py | 1271 | 1285 | def | 'transformers_wrapper_matches_the_native_model' | def transformers_wrapper_matches_the_native_model(): |
| chapters/39-exporting.md:437 | anulm/hf/modeling_anulm.py | 91–109 |  | reattributed | 'rebuild_rotary' is in anulm/hf/modeling_anulm.py, not tests/test_mode |  |
| chapters/40-the-production-gap.md:24 | anulm/model.py | 861–863 | 873–875 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/40-the-production-gap.md:32 | anulm/model.py | 463–469 | 470–476 | same | if cache is not None: | if cache is not None: |
| chapters/40-the-production-gap.md:48 | anulm/model.py | 1132–1139 | 1144–1151 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/40-the-production-gap.md:95 | anulm/inference/api.py | 394–401 | 441–448 | same | with GPU: | with GPU: |
| chapters/40-the-production-gap.md:174 | anulm/model.py | 745–752 | 752–759 | same | key = sum(p._version for e in self.experts for p in e.parameters()) | key = sum(p._version for e in self.experts for p in e.parameters()) |
| chapters/40-the-production-gap.md:261 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/40-the-production-gap.md:295 | anulm/training/train.py | 467–475 | 463–471 | same | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam |
| chapters/40-the-production-gap.md:343 | anulm/inference/api.py | 170–174 | 173–180 | same | def check_key(authorization: Optional[str] = Header(default=None)): | def check_key(authorization: Optional[str] = Header(default=None)): |
| chapters/40-the-production-gap.md:343 | anulm/inference/api.py | 173–180 |  | EXCERPT | if key and authorization != f"Bearer {key}": | quoted line not in the new range |
| chapters/40-the-production-gap.md:434 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/40-the-production-gap.md:434 | anulm/model.py | 463–469 | 470–476 | same | if cache is not None: | if cache is not None: |
| chapters/40-the-production-gap.md:435 | anulm/model.py | 861–863 | 873–875 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/40-the-production-gap.md:436 | anulm/training/train.py | 451–478 | 447–474 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/40-the-production-gap.md:438 | anulm/inference/api.py | 170–174 | 173–180 | same | def check_key(authorization: Optional[str] = Header(default=None)): | def check_key(authorization: Optional[str] = Header(default=None)): |
| chapters/40-the-production-gap.md:438 | anulm/inference/api.py | 265 | 296 | same | GPU = threading.Lock()                # one request on the device at a | GPU = threading.Lock()                # one request on the device at a |
| chapters/40-the-production-gap.md:438 | anulm/inference/api.py | 378–409 | 425–456 | same | def run(self, emit=None): | def run(self, emit=None): |
| chapters/41-capstone.md:65 | anulm/training/train.py | 379–385 | 375–381 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/41-capstone.md:77 | anulm/training/train.py | 715–733 | 712–731 | same | builder = {"30b": AnuLMConfig.nano_30b, | builder = {"30b": AnuLMConfig.nano_30b, |
| chapters/41-capstone.md:77 | anulm/training/train.py | 712–731 |  | EXCERPT | assert field is not None and _, f"--cfg: unknown field {kv!r}" | quoted line not in the new range |
| chapters/41-capstone.md:157 | tests/test_model.py | 338–348 | 349–359 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/41-capstone.md:184 | anulm/training/train.py | 852–875 | 855–878 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/41-capstone.md:223 | anulm/training/train.py | 888–905 | 891–908 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/41-capstone.md:365 | anulm/model.py | 1149–1157 | 1161–1169 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/41-capstone.md:593 | tests/test_model.py | 387–395 | 398–406 | same | def generation_is_deterministic_at_top_k_1(): | def generation_is_deterministic_at_top_k_1(): |
| chapters/41-capstone.md:618 | tests/test_model.py | 2213–2226 | 2250–2271 | same | def main(): | def main(): |
| chapters/41-capstone.md:618 | tests/test_model.py | 2214 | 2251 | edited-quote | passed, failed = [], [] | passed, skipped, failed = [], [], [] |
| chapters/41-capstone.md:696 | anulm/model.py | 777–780 | 784–787 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/41-capstone.md:50 | docs/TUTORIAL.md | 579–580 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/41-capstone.md:55 | .github/workflows/tests.yml | 84 | 131 | same | run: \| | run: \| |
| chapters/41-capstone.md:62 | anulm/training/train.py | 558 |  | reattributed | quoted code is in anulm/training/train.py, not .github/workflows/tests |  |
| chapters/41-capstone.md:62 | anulm/training/train.py | 558 | 554 | same | p.add_argument("--warmup", type=int, default=100) | p.add_argument("--warmup", type=int, default=100) |
| chapters/41-capstone.md:140 | anulm/training/train.py | 728–732 | ? | CHECK | kw = dict(block_size=args.block_size, yarn=args.yarn, | no file named on the line and nothing quoted confirms it |
| chapters/41-capstone.md:384 | anulm/model.py | 1156 | ? | CHECK | probs = F.softmax(logits, dim=-1) | no file named on the line and nothing quoted confirms it |
| chapters/41-capstone.md:638 | tests/test_model.py | 2230 | ? | CHECK | sys.exit(main()) | no file named on the line and nothing quoted confirms it |
| chapters/41-capstone.md:654 | tests/test_model.py | 688, 709, 727 |  | reattributed | quoted code is in tests/test_model.py, not docs/TUTORIAL.md |  |
| chapters/41-capstone.md:654 | tests/test_model.py | 688 | 699 | edited | return | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/41-capstone.md:654 | tests/test_model.py | 709 | 720 | edited | return | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/41-capstone.md:654 | tests/test_model.py | 727 | 738 | edited | return | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/41-capstone.md:663 | anulm/hf/export_hf.py | 322–332 | 328–338 | same | # Small tensors stay float32 on disk AND must be loaded that way: the | # Small tensors stay float32 on disk AND must be loaded that way: the |
| chapters/41-capstone.md:693 | data/multi32k.json | 749 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| chapters/41-capstone.md:883 | anulm/training/train.py | 540–921 | 536–924 | same | def main(): | def main(): |
| chapters/41-capstone.md:883 | anulm/training/train.py | 715–725 | 712–723 | same | builder = {"30b": AnuLMConfig.nano_30b, | builder = {"30b": AnuLMConfig.nano_30b, |
| chapters/41-capstone.md:883 | anulm/training/train.py | 715–725 | 712–723 | sweep | ng/train.py': 'main' 540–921 (preset builder 715–725, key |  |
| chapters/41-capstone.md:884 | anulm/training/train.py | 728–733 | 726–731 | same | kw = dict(block_size=args.block_size, yarn=args.yarn, | kw = dict(block_size=args.block_size, yarn=args.yarn, |
| chapters/41-capstone.md:884 | anulm/training/train.py | 728–733 | 726–731 | sweep | 728–733, the |  |
| chapters/41-capstone.md:884 | anulm/training/train.py | 852–875 | 855–878 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/41-capstone.md:884 | anulm/training/train.py | 852–875 | 855–878 | sweep | 728–733, the step 852–875, eva |  |
| chapters/41-capstone.md:884 | anulm/training/train.py | 888–905 | 891–908 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/41-capstone.md:884 | anulm/training/train.py | 888–905 | 891–908 | sweep | –733, the step 852–875, evaluation and saves 888–905), 'l |  |
| chapters/41-capstone.md:885 | anulm/training/train.py | 379–404 | 375–400 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/41-capstone.md:885 | anulm/training/train.py | 481–537 | 477–533 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/41-capstone.md:886 | anulm/training/train.py | 552 | 548 | same | p.add_argument("--steps", type=int, default=2000) | p.add_argument("--steps", type=int, default=2000) |
| chapters/41-capstone.md:886 | anulm/training/train.py | 558 | 554 | same | p.add_argument("--warmup", type=int, default=100) | p.add_argument("--warmup", type=int, default=100) |
| chapters/41-capstone.md:886 | anulm/training/train.py | 558 | 554 | sweep | '--steps' 2000 (line 552), '--warmup' 100 (558), '- |  |
| chapters/41-capstone.md:886 | anulm/training/train.py | 579–580 | 575–576 | same | p.add_argument("--device", type=str, | p.add_argument("--device", type=str, |
| chapters/41-capstone.md:886 | anulm/training/train.py | 579–580 | 575–576 | sweep | line 552), '--warmup' 100 (558), '--device' (579–580). |  |
| chapters/41-capstone.md:887 | .github/workflows/tests.yml | 84 | 131 | same | run: \| | run: \| |
| chapters/41-capstone.md:888 | anulm/model.py | 1104–1168 | 1116–1180 | same | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/41-capstone.md:888 | anulm/model.py | 1149–1157 | 1161–1169 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/41-capstone.md:888 | anulm/model.py | 1149–1157 | 1161–1169 | sweep | uLM.generate' 1104–1168 (greedy and sampling 1149–1157), |  |
| chapters/41-capstone.md:889 | anulm/model.py | 1092–1101 | 1104–1113 | same | def num_params(self) -> tuple[int, int]: | def num_params(self) -> tuple[int, int]: |
| chapters/41-capstone.md:889 | anulm/model.py | 759–783 | 766–790 | same | def _route_grouped(self, x_flat, topk_idx, topk_w): | def _route_grouped(self, x_flat, topk_idx, topk_w): |
| chapters/41-capstone.md:890 | anulm/model.py | 749 | 756 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/41-capstone.md:890 | anulm/model.py | 749 | 756 | sweep | 749 and |  |
| chapters/41-capstone.md:890 | anulm/model.py | 777 | 784 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/41-capstone.md:890 | anulm/model.py | 777 | 784 | sweep | 749 and 777). |  |
| chapters/41-capstone.md:891 | tests/test_model.py | 338 | 349 | same | def initial_loss_is_ln_vocab(): | def initial_loss_is_ln_vocab(): |
| chapters/41-capstone.md:892 | tests/test_model.py | 387 | 398 | same | def generation_is_deterministic_at_top_k_1(): | def generation_is_deterministic_at_top_k_1(): |
| chapters/41-capstone.md:893 | tests/test_model.py | 399 | 410 | same | def param_count_matches_analysis(): | def param_count_matches_analysis(): |
| chapters/41-capstone.md:893 | tests/test_model.py | 399 | 410 | sweep | 399, 'gr |  |
| chapters/41-capstone.md:893 | tests/test_model.py | 437 | 448 | same | def gradient_accumulation_equals_large_batch(): | def gradient_accumulation_equals_large_batch(): |
| chapters/41-capstone.md:894 | tests/test_model.py | 550 | 561 | same | def incremental_forward_matches_full_forward(): | def incremental_forward_matches_full_forward(): |
| chapters/41-capstone.md:895 | tests/test_model.py | 593 | 604 | same | def cached_generation_matches_uncached(): | def cached_generation_matches_uncached(): |
| chapters/41-capstone.md:895 | tests/test_model.py | 2213–2226 | 2250–2271 | same | def main(): | def main(): |
| chapters/41-capstone.md:896 | tests/test_model.py | 688 | 699 | edited | return | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/41-capstone.md:896 | tests/test_model.py | 688 | 699 | sweep | returns at 688, 709 |  |
| chapters/41-capstone.md:896 | tests/test_model.py | 709 | 720 | edited | return | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/41-capstone.md:896 | tests/test_model.py | 709 | 720 | sweep | returns at 688, 709, 727 |  |
| chapters/41-capstone.md:896 | tests/test_model.py | 727 | 738 | edited | return | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/41-capstone.md:896 | tests/test_model.py | 727 | 738 | sweep | returns at 688, 709, 727 and |  |
| chapters/41-capstone.md:896 | tests/test_model.py | 888–891 | 898–901 | edited | try: | try: |
| chapters/41-capstone.md:896 | tests/test_model.py | 888–891 | 898–901 | sweep | returns at 688, 709, 727 and 888–891; 76 |  |
| chapters/41-capstone.md:896 | tests/test_model.py | 76 | 87 | same | h, _ = layer(h, cos, sin) | h, _ = layer(h, cos, sin) |
| chapters/41-capstone.md:896 | tests/test_model.py | 76 | 87 | sweep | returns at 688, 709, 727 and 888–891; 76 test |  |
| chapters/41-capstone.md:897 | anulm/hf/export_hf.py | 51–60 | 52–61 | same | def to_bf16(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: | def to_bf16(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: |
| chapters/41-capstone.md:898 | anulm/hf/export_hf.py | 322–332 | 328–338 | same | # Small tensors stay float32 on disk AND must be loaded that way: the | # Small tensors stay float32 on disk AND must be loaded that way: the |
| chapters/41-capstone.md:898 | anulm/hf/export_hf.py | 322–332 | 328–338 | sweep | float32 config and its comment 322–332: by |  |
| chapters/41-capstone.md:906 | anulm/bpe.py | 11 | 11 | sweep | 11 and |  |
| chapters/41-capstone.md:906 | anulm/bpe.py | 15 | 15 | sweep | 11 and 15, 'co |  |
| chapters/A1-shape-trace.md:64 | anulm/model.py | 884 | 896 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| chapters/A1-shape-trace.md:76 | anulm/model.py | 907 | 919 | same | if self.ce_chunk and self.training: | if self.ce_chunk and self.training: |
| chapters/A2-parameter-arithmetic.md:34 | anulm/model.py | 39–211 | 39–218 | same | @dataclass | @dataclass |
| chapters/A2-parameter-arithmetic.md:85 | anulm/model.py | 126–149 |  | reattributed | '__post_init__' is in anulm/model.py, not data/multi32k.json |  |
| chapters/A2-parameter-arithmetic.md:85 | anulm/model.py | 126–149 | 126–156 | same | def __post_init__(self): | def __post_init__(self): |
| chapters/A2-parameter-arithmetic.md:173 | anulm/model.py | 180–181 | 187–188 | same | Requires '--grad-ckpt'. Without it the same model peaks at 15.75 GB an | Requires '--grad-ckpt'. Without it the same model peaks at 15.75 GB an |
| chapters/A3-test-index.md:11 | tests/test_model.py | 28 | 34 | same | def test(fn): | def test(fn): |
| chapters/A3-test-index.md:12 | tests/test_model.py | 2213 | 2250 | same | def main(): | def main(): |
| chapters/A3-test-index.md:13 | tests/test_model.py | 33 | ? | CHECK | def tiny(**kw) -> AnuLMConfig: | no file named on the line and nothing quoted confirms it |
| chapters/A3-test-index.md:133 | tests/test_model.py | 1590 |  | reattributed | 'native_gqa_matches_repeat_interleave' is in tests/test_model.py, not  |  |
| chapters/A3-test-index.md:133 | tests/test_model.py | 1590 | 1607 | same | m = AnuLM(tiny(n_kv_head=1, sliding_window=4, max_window_layers=1)).ev | m = AnuLM(tiny(n_kv_head=1, sliding_window=4, max_window_layers=1)).ev |
| chapters/A3-test-index.md:134 | tests/test_model.py | 1102–1106 |  | reattributed | 'lora_starts_as_the_base_model_and_merges_back_exactly' is in tests/te |  |
| chapters/A3-test-index.md:134 | tests/test_model.py | 1102–1106 | 1111–1115 | same | trained_bias = router[sorted(router)[0]] | trained_bias = router[sorted(router)[0]] |
| chapters/A3-test-index.md:135 | tests/test_model.py | 688, 709, 727 |  | reattributed | quoted code is in tests/test_model.py, not None |  |
| chapters/A3-test-index.md:135 | tests/test_model.py | 688 | 699 | edited | return | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/A3-test-index.md:135 | tests/test_model.py | 709 | 720 | edited | return | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/A3-test-index.md:135 | tests/test_model.py | 727 | 738 | edited | return | raise unittest.SkipTest("this torch has no F.grouped_mm") |
| chapters/A3-test-index.md:136 | tests/test_model.py | 1282–1286, 1342–1346 |  | reattributed | quoted code is in tests/test_model.py, not None |  |
| chapters/A3-test-index.md:136 | tests/test_model.py | 1282–1286 | ? | UNMAPPED | try: |  |
| chapters/A3-test-index.md:136 | tests/test_model.py | 1342–1346 | ? | UNMAPPED | try: |  |
| chapters/A3-test-index.md:137 | tests/test_model.py | 888–891 |  | reattributed | quoted code is in tests/test_model.py, not None |  |
| chapters/A3-test-index.md:137 | tests/test_model.py | 888–891 | 898–901 | edited | try: | try: |
| chapters/A4-glossary.md:52 | anulm/hf/export_hf.py | 51–60 | 52–61 | same | def to_bf16(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: | def to_bf16(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: |
| chapters/A4-glossary.md:52 | anulm/hf/export_hf.py | 230–232 | 231–233 | same | p.add_argument("--weights-dtype", choices=("mixed", "float32"), defaul | p.add_argument("--weights-dtype", choices=("mixed", "float32"), defaul |
| chapters/A4-glossary.md:159 | tests/test_model.py | 33 | 44 | same | def tiny(**kw) -> AnuLMConfig: | def tiny(**kw) -> AnuLMConfig: |
| REPOSITORY_SOURCE_MAP.md:92 | anulm/training/train.py | 101 | ? | CHECK | BYTE_EOS = 257          # byte mode: 256 BOS, 257 EOS, 258 PAD | 'load_bytes' spans [(62, 98, 62)] at the old pin |
| REPOSITORY_SOURCE_MAP.md:154 | anulm/multimodal/vision_encoder.py | 504 |  | reattributed | 'train' is in anulm/multimodal/vision_encoder.py, not anulm/multimodal |  |
| REPOSITORY_SOURCE_MAP.md:154 | anulm/multimodal/vision_encoder.py | 504 | 503 | same | torch.nn.utils.clip_grad_norm_(vl.parameters(), 1.0) | torch.nn.utils.clip_grad_norm_(vl.parameters(), 1.0) |
| REPOSITORY_SOURCE_MAP.md:155 | anulm/multimodal/omni_train.py | 501 | ? | CHECK | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", | 'Omni._match_scale' spans [(249, 263, 250)] at the old pin |
| REPOSITORY_ARCHITECTURE.md:118 | anulm/training/train.py | 399–453 | 395–449 | same | # several times the lr the weights last saw. --rewarmup ramps back up  | # several times the lr the weights last saw. --rewarmup ramps back up  |
| REPOSITORY_ARCHITECTURE.md:120 | anulm/training/train.py | 241–254 | 237–250 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| REPOSITORY_ARCHITECTURE.md:121 | anulm/model.py | 884 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:121 | anulm/model.py | 884 | 896 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| REPOSITORY_ARCHITECTURE.md:121 | anulm/model.py | 954–972 |  | reattributed | '_table' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:121 | anulm/model.py | 954–972 | 966–984 | same | def _table(self, side: str): | def _table(self, side: str): |
| REPOSITORY_ARCHITECTURE.md:122 | anulm/model.py | 332–336 |  | reattributed | 'Rotary.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:122 | anulm/model.py | 332–336 | 339–343 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| REPOSITORY_ARCHITECTURE.md:123 | anulm/model.py | 809–817 |  | reattributed | 'Block.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:123 | anulm/model.py | 809–817 | 816–824 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| REPOSITORY_ARCHITECTURE.md:124 | anulm/model.py | 444–487 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not anulm/training/train.p |  |
| REPOSITORY_ARCHITECTURE.md:124 | anulm/model.py | 444–487 | 451–494 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| REPOSITORY_ARCHITECTURE.md:125 | anulm/model.py | 358–374 |  | reattributed | '_attn_mask' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:125 | anulm/model.py | 358–374 | 365–381 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| REPOSITORY_ARCHITECTURE.md:125 | anulm/model.py | 407–411 |  | reattributed | '_sdpa' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:125 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| REPOSITORY_ARCHITECTURE.md:126 | anulm/model.py | 246 |  | reattributed | quoted code is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:126 | anulm/model.py | 246 | 253 | same | return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) | return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) |
| REPOSITORY_ARCHITECTURE.md:126 | anulm/model.py | 689–700 |  | reattributed | 'MoE.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:126 | anulm/model.py | 689–700 | 696–707 | same | def forward(self, x):                                     # (B, S, D) | def forward(self, x):                                     # (B, S, D) |
| REPOSITORY_ARCHITECTURE.md:127 | anulm/model.py | 617–665 |  | reattributed | 'Router._gate' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:127 | anulm/model.py | 617–665 | 624–672 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| REPOSITORY_ARCHITECTURE.md:128 | anulm/model.py | 900–916 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:128 | anulm/model.py | 900–916 | 912–928 | same | x = self.norm(x) | x = self.norm(x) |
| REPOSITORY_ARCHITECTURE.md:129 | anulm/training/train.py | 451–478 | 447–474 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| REPOSITORY_ARCHITECTURE.md:130 | anulm/training/train.py | 379–404 |  | reattributed | 'lr_at' is in anulm/training/train.py, not anulm/muon.py |  |
| REPOSITORY_ARCHITECTURE.md:130 | anulm/training/train.py | 379–404 | 375–400 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| REPOSITORY_ARCHITECTURE.md:131 | anulm/training/train.py | 847–905 |  | reattributed | quoted code is in anulm/training/train.py, not anulm/muon.py |  |
| REPOSITORY_ARCHITECTURE.md:131 | anulm/training/train.py | 847–905 | 850–908 | same | for step in range(start_step, end): | for step in range(start_step, end): |
| REPOSITORY_ARCHITECTURE.md:146 | anulm/model.py | 861 |  | reattributed | 'new_cache' is in anulm/model.py, not anulm/inference/sample.py |  |
| REPOSITORY_ARCHITECTURE.md:146 | anulm/model.py | 861 | 873 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| REPOSITORY_ARCHITECTURE.md:148 | anulm/inference/sample.py | 463–469 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| REPOSITORY_ARCHITECTURE.md:149 | anulm/inference/sample.py | 548–556 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| REPOSITORY_ARCHITECTURE.md:154 | anulm/model.py | 358–374 |  | reattributed | quoted code is in anulm/model.py, not anulm/inference/sample.py |  |
| REPOSITORY_ARCHITECTURE.md:154 | anulm/model.py | 358–374 | 365–381 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| REPOSITORY_ARCHITECTURE.md:156 | anulm/inference/sample.py | 1136–1139 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| REPOSITORY_ARCHITECTURE.md:161 | anulm/inference/api.py | 193 | 219 | same | top_p: Optional[float] = Field(None, description="Accepted for compati | top_p: Optional[float] = Field(None, description="Accepted for compati |
| REPOSITORY_ARCHITECTURE.md:164 | anulm/hf/modeling_anulm.py | 879–882 | ? | CHECK |  | no file named on the line and nothing quoted confirms it |
| REPOSITORY_ARCHITECTURE.md:164 | anulm/model.py | 1004–1050 |  | reattributed | 'decode_from_embeds' is in anulm/model.py, not anulm/hf/modeling_anulm |  |
| REPOSITORY_ARCHITECTURE.md:164 | anulm/model.py | 1004–1050 | 1016–1062 | same | def decode_from_embeds(self, prefix, max_tokens: int, stop, no_repeat_ | def decode_from_embeds(self, prefix, max_tokens: int, stop, no_repeat_ |
| REPOSITORY_ARCHITECTURE.md:173 | anulm/model.py | 702–783 |  | reattributed | '_route_grouped' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:173 | anulm/model.py | 702–783 | 709–790 | same | def _route_sparse(self, x_flat, topk_idx, topk_w): | def _route_sparse(self, x_flat, topk_idx, topk_w): |
| REPOSITORY_ARCHITECTURE.md:174 | anulm/model.py | 604–615 |  | reattributed | 'Router.forward' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:174 | anulm/model.py | 604–615 | 611–622 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| REPOSITORY_ARCHITECTURE.md:175 | anulm/model.py | 640–663 |  | reattributed | 'Router._gate' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:175 | anulm/model.py | 640–663 | 647–670 | same | aux = None | aux = None |
| REPOSITORY_ARCHITECTURE.md:175 | anulm/model.py | 1059–1089 |  | reattributed | 'AnuLM.update_expert_biases' is in anulm/model.py, not anulm/hf/modeli |  |
| REPOSITORY_ARCHITECTURE.md:175 | anulm/model.py | 1059–1089 | 1071–1101 | same | @torch.no_grad() | @torch.no_grad() |
| REPOSITORY_ARCHITECTURE.md:176 | anulm/model.py | 890–895 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:176 | anulm/model.py | 890–895 | 902–907 | same | if self.grad_ckpt and self.training: | if self.grad_ckpt and self.training: |
| REPOSITORY_ARCHITECTURE.md:177 | anulm/model.py | 918–944 |  | reattributed | '_chunked_loss' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:177 | anulm/model.py | 918–944 | 930–956 | same | def enable_chunked_loss(self, rows: int = 2048): | def enable_chunked_loss(self, rows: int = 2048): |
| REPOSITORY_ARCHITECTURE.md:178 | anulm/model.py | 383–404 |  | reattributed | 'merge_tree_matrix' is in anulm/model.py, not anulm/hf/modeling_anulm. |  |
| REPOSITORY_ARCHITECTURE.md:178 | anulm/model.py | 383–404 | 390–411 | same | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: |
| REPOSITORY_ARCHITECTURE.md:179 | anulm/model.py | 848–851 |  | reattributed | 'AnuLM.__init__' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:179 | anulm/model.py | 848–851 | 860–863 | same | # nanoGPT's scaled init for the residual output projections. | # nanoGPT's scaled init for the residual output projections. |
| REPOSITORY_ARCHITECTURE.md:180 | anulm/training/train.py | 649–654 | 645–650 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| REPOSITORY_ARCHITECTURE.md:180 | anulm/training/train.py | 467 | 463 | same | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam |
| REPOSITORY_ARCHITECTURE.md:180 | anulm/training/train.py | 752–772 | 750–770 | same | if args.compile: | if args.compile: |
| REPOSITORY_ARCHITECTURE.md:181 | anulm/training/train.py | 854–870 | 857–873 | same | for _ in range(args.grad_accum): | for _ in range(args.grad_accum): |
| REPOSITORY_ARCHITECTURE.md:182 | anulm/training/train.py | 790–837 | ? | CHECK | start_step, best_val = 0, float("inf") | 'atomic_save' spans [(32, 46, 32)] at the old pin |
| REPOSITORY_ARCHITECTURE.md:183 | anulm/training/train.py | 481–537 | 477–533 | same | @torch.no_grad() | @torch.no_grad() |
| REPOSITORY_ARCHITECTURE.md:184 | anulm/training/train.py | 206–220 | 202–216 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| REPOSITORY_ARCHITECTURE.md:187 | anulm/model.py | 1257–1276 |  | reattributed | 'fit_memory' is in anulm/model.py, not anulm/bigram_teacher.py |  |
| REPOSITORY_ARCHITECTURE.md:187 | anulm/model.py | 1257–1276 | 1345–1364 | same | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i |
| FRONT_MATTER.md:33 | anulm/model.py | 407–411 | 414–418 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| README.md:75 | anulm/model.py | 608–614 | 615–621 | same | # fp32 router: top-k is discrete, and bf16 rounding would flip choices | # fp32 router: top-k is discrete, and bf16 rounding would flip choices |
| BOOK_PLAN.md:87 | anulm/model.py | 910–916 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/bidirectional.py |  |
| BOOK_PLAN.md:87 | anulm/model.py | 910–916 | 922–928 | same | logits = self._head(x) | logits = self._head(x) |
| BOOK_PLAN.md:87 | anulm/training/train.py | 903 | 906 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| BOOK_PLAN.md:94 | anulm/model.py | 450–456 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not anulm/training/train.p |  |
| BOOK_PLAN.md:94 | anulm/model.py | 450–456 | 457–463 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| BOOK_PLAN.md:102 | anulm/training/train.py | 865 | 868 | same | (loss / args.grad_accum).backward() | (loss / args.grad_accum).backward() |
| STYLE.md:111 | anulm/model.py | 444–487 | 451–494 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| RESEARCH_REFERENCES.md:203 | anulm/model.py | 131 | 137 | edited | "moe_impl='grouped' needs torch.nn.functional.grouped_mm (present in 2 | "moe_impl='grouped' needs torch.nn.functional.grouped_mm (present in 2 |
| RESEARCH_REFERENCES.md:270 | anulm/model.py | 109–110 | ? | CHECK | yarn_beta_fast: float = 32.0 | 'yarn_mscale' spans [(263, 268, 263)] at the old pin |
| RESEARCH_REFERENCES.md:270 | anulm/model.py | 442 | ? | CHECK | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) | 'yarn_mscale' spans [(263, 268, 263)] at the old pin |
| RESEARCH_REFERENCES.md:273 | anulm/model.py | 436–438 | ? | CHECK | if cfg.use_qk_norm: | 'RMSNorm' spans [(218, 232, 218)] at the old pin |
