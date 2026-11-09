# Re-pin report: @b42998d -> @2e80b3e

Written by `book/tools/repin.py`. Every line citation it moved, with the code at
the old line and at the new one. Rows marked UNMAPPED, CHECK, EXCERPT or OUT OF
FILE were **not** resolved automatically and need a person.

| kind | count |
|---|---|
| EXCERPT | 12 |
| STALE OVERRIDE | 1 |
| UNMAPPED | 15 |
| def | 16 |
| edited | 18 |
| edited-quote | 5 |
| other-pin | 37 |
| override | 81 |
| reattributed | 96 |
| same | 763 |
| sweep | 123 |

## Needs a person (28)

| where | file | old | new | kind | old code | new code |
|---|---|---|---|---|---|---|
| chapters/09-byte-level-bpe.md:604 | anulm/bpe.py | 221–244 | ? | UNMAPPED | # The format id the project wrote before it was renamed on 2026-09-18. |  |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 712–732 | ? | UNMAPPED | builder = {"30b": AnuLMConfig.nano_30b, |  |
| chapters/26-debugging-silent-failures.md:493 | anulm/eval/eval_code.py | 1544 | ? | UNMAPPED | 'bpe_files_load_and_foreign_ones_are_refused' (line 1544) |  |
| chapters/33-retrieval-and-readers.md:191 | anulm/inference/rag.py | 395–410 |  | EXCERPT | self.dense = Dense(self.device) | quoted line not in the new range |
| chapters/33-retrieval-and-readers.md:191 | anulm/inference/rag.py | 395–410 |  | EXCERPT | di, dv = self.dense.search(query, 100 if mode == "hybrid" else k) | quoted line not in the new range |
| chapters/33-retrieval-and-readers.md:536 | anulm/inference/rag.py | 103–168 | ? | UNMAPPED | def build(langs=("simple", "hi"), workers: int = 12) -> None: |  |
| chapters/37-one-model-many-tasks.md:33 | anulm/multimodal/omni_train.py | 104–174 | ? | UNMAPPED | def pad(built, device): |  |
| chapters/37-one-model-many-tasks.md:33 | anulm/multimodal/omni_train.py | 104 | ? | EXCERPT | header range unresolved: text_batches |  |
| chapters/37-one-model-many-tasks.md:233 | anulm/multimodal/omni_train.py | 555–591 |  | EXCERPT | eb, tb = om.next_batch(t) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:233 | anulm/multimodal/omni_train.py | 555–591 |  | EXCERPT | _, loss = model(None, tb, embeds=eb) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:233 | anulm/multimodal/omni_train.py | 555–591 |  | EXCERPT | run_loss[t.name].append(loss.item()) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:357 | anulm/multimodal/omni_train.py | 460–467 | ? | UNMAPPED | if step % args.eval_every == 0 or step == args.steps or step == args.s |  |
| chapters/37-one-model-many-tasks.md:357 | anulm/multimodal/omni_train.py | 460 | ? | EXCERPT | header range unresolved: train |  |
| chapters/37-one-model-many-tasks.md:475 | anulm/multimodal/omni.py | 47–62 |  | EXCERPT | enc = WhisperEncoder(self.meta.get("whisper", "small"), self.device) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:603 | experiments/omni_one.py | 65–100 |  | EXCERPT | STEPS = 12_500                  # x ACCUM = the first run's 100,000 ba | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:603 | experiments/omni_one.py | 65–100 |  | EXCERPT | ("chat", "text", "data/chat_train.jsonl", 0.25, 8), | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:603 | experiments/omni_one.py | 65–100 |  | EXCERPT | ("intent", "text", "data/omni/intent_train.jsonl", 0.06, 8), | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:328 | anulm/multimodal/omni_train.py | 353–359 | ? | UNMAPPED | def recipe_of(args, tasks) -> dict: |  |
| chapters/37-one-model-many-tasks.md:857 | anulm/multimodal/omni_train.py | 161–174 | ? | UNMAPPED | def text_batches(rows, model, device, batch_size, shuffle=True, seed=0 |  |
| chapters/37-one-model-many-tasks.md:860 | anulm/multimodal/omni_train.py | 353–359 | ? | UNMAPPED | def recipe_of(args, tasks) -> dict: |  |
| chapters/38-serving.md:634 | anulm/apps/app_omni.py | 302–355 | ? | UNMAPPED | def chat_text(models: Models, text: str, max_tokens: int = 200) -> tup |  |
| chapters/38-serving.md:739 | anulm/apps/app_omni.py | 302–355 | ? | UNMAPPED | def chat_text(models: Models, text: str, max_tokens: int = 200) -> tup |  |
| chapters/40-the-production-gap.md:382 | anulm/multimodal/omni.py | 106–110 | ? | UNMAPPED | body = "\n\n".join(f"[{i + 1}] {' '.join(h['text'].split()[:120])}" fo |  |
| chapters/40-the-production-gap.md:382 | anulm/multimodal/omni.py | 106 | ? | EXCERPT | header range unresolved: Reader.ask |  |
| chapters/41-capstone.md:875 | anulm/training/train.py | 712–723 | ? | UNMAPPED | builder = {"30b": AnuLMConfig.nano_30b, |  |
| REPOSITORY_ARCHITECTURE.md:18 | anulm/model.py | 1–17 | ? | UNMAPPED | """ |  |
| REPOSITORY_ARCHITECTURE.md:203 | anulm/model.py | 19 | ? | UNMAPPED | * no expert parallelism, no vLLM branch, no FlashAttention subclass: |  |
| chapters/26-debugging-silent-failures.md:493 | tests/test_model.py | 1544 |  | STALE OVERRIDE | no citation matches this row |  |

## Every change

| where | file | old | new | kind | old code | new code |
|---|---|---|---|---|---|---|
| chapters/01-what-this-book-builds.md:133 | anulm/model.py | 1144–1173 | 1154–1183 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/01-what-this-book-builds.md:178 | anulm/model.py | 915–918 | 925–928 | same | if targets is None: | if targets is None: |
| chapters/01-what-this-book-builds.md:285 | anulm/model.py | 827–843 | 837–853 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/01-what-this-book-builds.md:355 | anulm/quickstart.py | 185–192 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/01-what-this-book-builds.md:373 | anulm/model.py | 154–167 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/01-what-this-book-builds.md:130 | anulm/model.py | 1116 |  | reattributed | 'generate' is defined there |  |
| chapters/01-what-this-book-builds.md:130 | anulm/model.py | 1116 | 1126 | def | 'generate' | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/01-what-this-book-builds.md:215 | anulm/inference/sample.py | 34–35 |  | override | decided in repin-overrides.tsv |  |
| chapters/01-what-this-book-builds.md:369 | anulm/quickstart.py | 226 |  | reattributed | quoted code is in anulm/quickstart.py, not anulm/inference/sample.py |  |
| chapters/01-what-this-book-builds.md:541 | anulm/model.py | 36 | 32 | same | # CONFIG | # CONFIG |
| chapters/01-what-this-book-builds.md:541 | anulm/model.py | 222 | 232 | same | # NORM / MLP | # NORM / MLP |
| chapters/01-what-this-book-builds.md:541 | anulm/model.py | 257 | 267 | edited | # ROPE  (see sarvam/yarn_rotary_annotated.py) | # ROPE AND YaRN |
| chapters/01-what-this-book-builds.md:541 | anulm/model.py | 422 | 432 | edited | # ATTENTION -- GQA  (see sarvam/gqa_forward_annotated.py) | # ATTENTION -- GQA |
| chapters/01-what-this-book-builds.md:541 | anulm/model.py | 498 | 508 | edited | # ATTENTION -- MLA  (see sarvam/mla_forward_annotated.py) | # ATTENTION -- MLA |
| chapters/01-what-this-book-builds.md:541 | anulm/model.py | 582 | 592 | edited | # MoE  (see sarvam/moe_routing_annotated.py) | # MIXTURE OF EXPERTS |
| chapters/01-what-this-book-builds.md:541 | anulm/model.py | 794 | 804 | same | # BLOCK / MODEL | # BLOCK / MODEL |
| chapters/01-what-this-book-builds.md:542 | anulm/model.py |  |  | other-pin | 154–167 (@c4a39f8) |  |
| chapters/01-what-this-book-builds.md:542 | anulm/model.py | 797–824 | 807–834 | same | class Block(nn.Module): | class Block(nn.Module): |
| chapters/01-what-this-book-builds.md:543 | anulm/model.py | 0 | 0 | sweep | layer 0 dens |  |
| chapters/01-what-this-book-builds.md:543 | anulm/model.py | 811 |  | override | decided in repin-overrides.tsv |  |
| chapters/01-what-this-book-builds.md:543 | anulm/model.py | 811 | 821 | same | self.mlp = MoE(cfg) if self.is_moe else SwiGLU(cfg.hidden_size, cfg.in | self.mlp = MoE(cfg) if self.is_moe else SwiGLU(cfg.hidden_size, cfg.in |
| chapters/01-what-this-book-builds.md:543 | anulm/model.py | 827–843 | 837–853 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/01-what-this-book-builds.md:544 | anulm/model.py | 915–918 | 925–928 | same | if targets is None: | if targets is None: |
| chapters/01-what-this-book-builds.md:544 | anulm/model.py | 1115–1180 | 1125–1190 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/01-what-this-book-builds.md:544 | anulm/model.py | 1144–1173 | 1154–1183 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/01-what-this-book-builds.md:544 | anulm/model.py | 1144–1173 | 1154–1183 | sweep | d' 915–918; 'AnuLM.generate' 1115–1180 (loop 1144–1173; |  |
| chapters/01-what-this-book-builds.md:545 | anulm/model.py | 1161–1162 | 1171–1172 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/01-what-this-book-builds.md:545 | anulm/model.py | 1161–1162 | 1171–1172 | sweep | the 'temperature <= 0' greedy branch 1161–1162). |  |
| chapters/01-what-this-book-builds.md:547 | anulm/inference/sample.py | 34–35 | 34–35 | sweep | inference/sample.py': 'main' 26–93 (defaults 34–35, gen |  |
| chapters/01-what-this-book-builds.md:547 | anulm/inference/sample.py | 69–77 | 69–77 | sweep | y': 'main' 26–93 (defaults 34–35, generation 69–77). |  |
| chapters/01-what-this-book-builds.md:548 | anulm/quickstart.py | 185–192 | 185–192 | edited | rc = run([sys.executable, "-u", "-m", "anulm", "train", "--preset", "3 | rc = run([sys.executable, "-u", "-m", "anulm", "train", "--preset", "t |
| chapters/01-what-this-book-builds.md:549 | README.md | 29 | 28 | same | \| base v2: Hindi, English, Python, 11.3B tokens \| [toonist/AnuLM-Base- | \| base v2: Hindi, English, Python, 11.3B tokens \| [toonist/AnuLM-Base- |
| chapters/02-from-counts-to-transformers.md:85 | anulm/model.py | 836–841 | 846–851 | same | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) |
| chapters/02-from-counts-to-transformers.md:98 | anulm/model.py | 900–912 | 910–922 | same | aux_total = None | aux_total = None |
| chapters/02-from-counts-to-transformers.md:272 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/02-from-counts-to-transformers.md:287 | anulm/model.py | 480–486 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/02-from-counts-to-transformers.md:327 | anulm/model.py | 481 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/02-from-counts-to-transformers.md:327 | anulm/model.py | 481 | 491 | same | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star |
| chapters/02-from-counts-to-transformers.md:409 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/02-from-counts-to-transformers.md:409 | anulm/model.py | 470–476 | 480–486 | same | if cache is not None: | if cache is not None: |
| chapters/02-from-counts-to-transformers.md:409 | anulm/model.py | 481 | 491 | same | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star |
| chapters/02-from-counts-to-transformers.md:410 | anulm/model.py | 836–841 | 846–851 | same | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) |
| chapters/02-from-counts-to-transformers.md:410 | anulm/model.py | 900–912 | 910–922 | same | aux_total = None | aux_total = None |
| chapters/02-from-counts-to-transformers.md:411 | anulm/bigram_teacher.py | 42–51 | 42–51 | sweep | er.py': 'build_bigram_teacher' 34–62 (counts 42–51), |  |
| chapters/02-from-counts-to-transformers.md:413 | anulm/bidirectional.py | 82–93 | 82–93 | sweep | : 'bidir_hidden' 40–53; 'adapt' 56–105 (MNTP 82–93). |  |
| chapters/02-from-counts-to-transformers.md:415 | anulm/training/train.py | 237–250 | 240–253 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/02-from-counts-to-transformers.md:415 | anulm/training/train.py | 447–474 | 450–482 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/03-the-one-job.md:22 | anulm/training/train.py | 237–247 | 240–250 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/03-the-one-job.md:47 | anulm/model.py | 922–928 | 932–938 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/03-the-one-job.md:175 | anulm/model.py | 859–863 | 869–873 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/03-the-one-job.md:185 | anulm/model.py | 865–871 | 875–881 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/03-the-one-job.md:332 | anulm/training/train.py | 906–908 | 915–917 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/03-the-one-job.md:300 | anulm/model.py | 1196–1205 | 1206–1215 | same | x = torch.randint(0, cfg.vocab_size, (2, 64)) | x = torch.randint(0, cfg.vocab_size, (2, 64)) |
| chapters/03-the-one-job.md:347 | anulm/training/train.py | 656–671 | 667–682 | same | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 |
| chapters/03-the-one-job.md:373 | anulm/training/train.py | 906 | 915 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/03-the-one-job.md:453 | anulm/model.py | 859–863 | 869–873 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/03-the-one-job.md:453 | anulm/model.py | 865–871 | 875–881 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/03-the-one-job.md:454 | anulm/model.py | 922–928 | 932–938 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/03-the-one-job.md:454 | anulm/model.py | 1183–1208 | 1193–1218 | same | if __name__ == "__main__": | if __name__ == "__main__": |
| chapters/03-the-one-job.md:455 | anulm/training/train.py | 237–250 | 240–253 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/03-the-one-job.md:455 | anulm/training/train.py | 656–671 | 667–682 | same | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 |
| chapters/03-the-one-job.md:456 | anulm/training/train.py | 906–908 | 915–917 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/03-the-one-job.md:456 | anulm/training/train.py | 906–908 | 915–917 | sweep | 906–908 (bit |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:75 | anulm/model.py | 447–451 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:112 | anulm/model.py | 457–463 | 467–473 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:177 | anulm/model.py | 483–494 | 493–504 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:237 | anulm/model.py | 353–362 | 363–372 | same | def apply_rope(q, k, cos, sin): | def apply_rope(q, k, cos, sin): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:255 | anulm/model.py | 234–239 | 244–249 | same | def forward(self, x): | def forward(self, x): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:64 | anulm/model.py | 461–463 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not book/code-examples/ch0 |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:64 | anulm/model.py | 461–463 | 471–473 | same | q = q.transpose(1, 2)                         # (B, H,   S, Dh) | q = q.transpose(1, 2)                         # (B, H,   S, Dh) |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:214 | anulm/model.py | 493 |  | reattributed | quoted code is in anulm/model.py, not book/code-examples/ch04_shapes.p |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:214 | anulm/model.py | 493 | 503 | same | y = y.transpose(1, 2).contiguous().view(B, S, -1) | y = y.transpose(1, 2).contiguous().view(B, S, -1) |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:296 | anulm/model.py | 470–477 |  | reattributed | quoted code is in anulm/model.py, not book/code-examples/ch04_shapes.p |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:296 | anulm/model.py | 470–477 | 480–487 | same | if cache is not None: | if cache is not None: |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:361 | anulm/model.py | 461 |  | override | decided in repin-overrides.tsv |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:361 | anulm/model.py | 461 | 471 | same | q = q.transpose(1, 2)                         # (B, H,   S, Dh) | q = q.transpose(1, 2)                         # (B, H,   S, Dh) |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:405 | anulm/model.py | 234–239 | 244–249 | same | def forward(self, x): | def forward(self, x): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:405 | anulm/model.py | 346–350 | 356–360 | same | def rotate_half(x): | def rotate_half(x): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:406 | anulm/model.py | 353–362 | 363–372 | same | def apply_rope(q, k, cos, sin): | def apply_rope(q, k, cos, sin): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:406 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:406 | anulm/model.py | 437–441 | 447–451 | edited | # One fused projection: (H + 2*Hkv) * Dh, exactly as the 30B does. | # One fused projection: (H + 2*Hkv) * Dh, split on the head axis. |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:407 | anulm/model.py | 457–463 | 467–473 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:407 | anulm/model.py | 470–477 | 480–487 | same | if cache is not None: | if cache is not None: |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:407 | anulm/model.py | 483–494 | 493–504 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/05-softmax-temperature-and-cross-entropy.md:78 | anulm/model.py | 624–626 | 634–636 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/05-softmax-temperature-and-cross-entropy.md:95 | anulm/model.py | 923–925 | 933–935 | same | loss = F.cross_entropy( | loss = F.cross_entropy( |
| chapters/05-softmax-temperature-and-cross-entropy.md:169 | anulm/model.py | 1152–1169 | 1162–1179 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:404 | anulm/model.py | 944–956 | 954–966 | same | def _chunked_loss(self, x, targets): | def _chunked_loss(self, x, targets): |
| chapters/05-softmax-temperature-and-cross-entropy.md:111 | anulm/training/train.py | 645–650 |  | override | decided in repin-overrides.tsv |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:111 | anulm/training/train.py | 645–650 | 656–661 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/05-softmax-temperature-and-cross-entropy.md:158 | anulm/model.py | 924 |  | reattributed | quoted code is in anulm/model.py, not book/code-examples/ch05_softmax. |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:158 | anulm/model.py | 924 | 934 | same | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ |
| chapters/05-softmax-temperature-and-cross-entropy.md:160 | anulm/model.py | 1152 |  | override | decided in repin-overrides.tsv |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:160 | anulm/model.py | 1152 | 1162 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:224 | anulm/inference/sample.py | 34–35 |  | override | decided in repin-overrides.tsv |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:249 | anulm/model.py | 1166–1167 |  | override | decided in repin-overrides.tsv |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:249 | anulm/model.py | 1166–1167 | 1176–1177 | same | v, _ = torch.topk(logits, min(top_k, logits.size(-1))) | v, _ = torch.topk(logits, min(top_k, logits.size(-1))) |
| chapters/05-softmax-temperature-and-cross-entropy.md:263 | anulm/model.py | 1038–1049 | 1048–1059 | same | scores = logits[0, -1].float() | scores = logits[0, -1].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:499 | anulm/model.py | 624–626 | 634–636 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/05-softmax-temperature-and-cross-entropy.md:499 | anulm/model.py | 923–925 | 933–935 | same | loss = F.cross_entropy( | loss = F.cross_entropy( |
| chapters/05-softmax-temperature-and-cross-entropy.md:500 | anulm/model.py | 944–953 | 954–963 | same | def _chunked_loss(self, x, targets): | def _chunked_loss(self, x, targets): |
| chapters/05-softmax-temperature-and-cross-entropy.md:500 | anulm/model.py | 955–956 | 965–966 | same | def _ce_sum(self, h, t): | def _ce_sum(self, h, t): |
| chapters/05-softmax-temperature-and-cross-entropy.md:501 | anulm/model.py | 1038–1049 | 1048–1059 | same | scores = logits[0, -1].float() | scores = logits[0, -1].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:501 | anulm/model.py | 1152–1169 | 1162–1179 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:505 | anulm/training/train.py | 645–650 | 656–661 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/06-gradients-and-backpropagation.md:84 | anulm/model.py | 242–253 | 252–263 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/06-gradients-and-backpropagation.md:168 | anulm/model.py | 820–824 | 830–834 | same | x = x + self.attn(self.input_layernorm(x), cos, sin, | x = x + self.attn(self.input_layernorm(x), cos, sin, |
| chapters/06-gradients-and-backpropagation.md:197 | anulm/training/train.py | 855–873 | 864–882 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/06-gradients-and-backpropagation.md:237 | anulm/training/train.py | 478 | 486 | def | 'evaluate' | def evaluate(model, data, args, iters: int \| None = None) -> tuple[flo |
| chapters/06-gradients-and-backpropagation.md:237 | anulm/model.py | 1116 |  | reattributed | 'AnuLM.generate' is in anulm/model.py, not anulm/training/train.py |  |
| chapters/06-gradients-and-backpropagation.md:237 | anulm/model.py | 1116 | 1126 | same | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/06-gradients-and-backpropagation.md:239 | anulm/model.py | 913 |  | reattributed | quoted code is in anulm/model.py, not anulm/training/train.py |  |
| chapters/06-gradients-and-backpropagation.md:239 | anulm/model.py | 913 | 923 | same | self.aux_loss = aux_total.detach() if aux_total is not None else None | self.aux_loss = aux_total.detach() if aux_total is not None else None |
| chapters/06-gradients-and-backpropagation.md:439 | anulm/model.py | 242–253 | 252–263 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/06-gradients-and-backpropagation.md:439 | anulm/model.py | 820–824 | 830–834 | same | x = x + self.attn(self.input_layernorm(x), cos, sin, | x = x + self.attn(self.input_layernorm(x), cos, sin, |
| chapters/06-gradients-and-backpropagation.md:440 | anulm/model.py | 913 | 923 | same | self.aux_loss = aux_total.detach() if aux_total is not None else None | self.aux_loss = aux_total.detach() if aux_total is not None else None |
| chapters/06-gradients-and-backpropagation.md:440 | anulm/model.py | 1115–1116 | 1125–1126 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/06-gradients-and-backpropagation.md:441 | anulm/training/train.py | 477–478 | 485–486 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/06-gradients-and-backpropagation.md:441 | anulm/training/train.py | 855–873 | 864–882 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:77 | anulm/training/train.py | 447–474 | 450–482 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:204 | anulm/training/train.py | 375–400 | 378–403 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:251 | anulm/training/train.py | 852–853 | 861–862 | same | lr = lr_at(step, args, start_step) | lr = lr_at(step, args, start_step) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:258 | anulm/training/train.py | 416–420 | 419–423 | same | def set_lr_scale(self, scale: float): | def set_lr_scale(self, scale: float): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:310 | tests/test_model.py | 1800–1809 | 1798–1807 | same | def wsd_schedule_is_flat_then_decays_to_min_lr(): | def wsd_schedule_is_flat_then_decays_to_min_lr(): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:336 | anulm/training/train.py | 874–878 | 883–887 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:21 | anulm/training/train.py | 555–556 | 564–565 | same | p.add_argument("--weight-decay", type=float, default=0.1) | p.add_argument("--weight-decay", type=float, default=0.1) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:465 | anulm/muon.py | 59 |  | reattributed | 'zeropower_batched' is defined there |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:489 | tests/test_model.py | 1902 | 1900 | def | 'batched_muon_matches_per_matrix_muon' | def batched_muon_matches_per_matrix_muon(): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:571 | anulm/training/train.py | 375–400 | 378–403 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:571 | anulm/training/train.py | 403–444 | 406–447 | same | class OptimizerSet: | class OptimizerSet: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:572 | anulm/training/train.py | 416–420 | 419–423 | same | def set_lr_scale(self, scale: float): | def set_lr_scale(self, scale: float): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:572 | anulm/training/train.py | 416–420 | 419–423 | sweep | 416–420); 'm |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:572 | anulm/training/train.py | 447–474 | 450–482 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:572 | anulm/training/train.py | 555–556 | 564–565 | same | p.add_argument("--weight-decay", type=float, default=0.1) | p.add_argument("--weight-decay", type=float, default=0.1) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:572 | anulm/training/train.py | 555–556 | 564–565 | sweep | ); 'make_optimizer' 447–474; 'main' defaults 555–556, sch |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:573 | anulm/training/train.py | 852–853 | 861–862 | same | lr = lr_at(step, args, start_step) | lr = lr_at(step, args, start_step) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:573 | anulm/training/train.py | 852–853 | 861–862 | sweep | 852–853, cli |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:573 | anulm/training/train.py | 874–878 | 883–887 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:573 | anulm/training/train.py | 874–878 | 883–887 | sweep | 852–853, clip and step 874–878. |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:576 | anulm/muon.py | 152–165 | 152–165 | sweep | 'build_optimizers' 144–182 (split 152–165). |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:577 | tests/test_model.py | 1800–1809 | 1798–1807 | same | def wsd_schedule_is_flat_then_decays_to_min_lr(): | def wsd_schedule_is_flat_then_decays_to_min_lr(): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:578 | tests/test_model.py | 1902 | 1900 | same | def batched_muon_matches_per_matrix_muon(): | def batched_muon_matches_per_matrix_muon(): |
| chapters/08-bytes-as-tokens.md:98 | anulm/model.py | 47 | 43 | same | vocab_size: int = 259           # 256 bytes + BOS + EOS + PAD (byte mo | vocab_size: int = 259           # 256 bytes + BOS + EOS + PAD (byte mo |
| chapters/08-bytes-as-tokens.md:104 | anulm/training/train.py | 101 | 104 | same | BYTE_EOS = 257          # byte mode: 256 BOS, 257 EOS, 258 PAD | BYTE_EOS = 257          # byte mode: 256 BOS, 257 EOS, 258 PAD |
| chapters/08-bytes-as-tokens.md:154 | anulm/training/train.py | 62–85 | 65–88 | same | def load_bytes(path: str \| None) -> torch.Tensor: | def load_bytes(path: str \| None) -> torch.Tensor: |
| chapters/08-bytes-as-tokens.md:182 | anulm/training/train.py | 86–98 | 89–101 | same | # Text files written on Windows carry CRLF; the tokenizer path reads t | # Text files written on Windows carry CRLF; the tokenizer path reads t |
| chapters/08-bytes-as-tokens.md:261 | tests/test_model.py | 1595–1613 | 1593–1611 | same | def byte_loader_inserts_eos_at_document_boundaries(): | def byte_loader_inserts_eos_at_document_boundaries(): |
| chapters/08-bytes-as-tokens.md:137 | anulm/training/train.py | 28 | 27 | same | from anulm.bpe import DOC_SEP | from anulm.bpe import DOC_SEP |
| chapters/08-bytes-as-tokens.md:174 | anulm/training/train.py | 68–85 |  | override | decided in repin-overrides.tsv |  |
| chapters/08-bytes-as-tokens.md:174 | anulm/training/train.py | 68–85 | 71–88 | same | else: | else: |
| chapters/08-bytes-as-tokens.md:208 | anulm/training/train.py | 101 | 104 | same | BYTE_EOS = 257          # byte mode: 256 BOS, 257 EOS, 258 PAD | BYTE_EOS = 257          # byte mode: 256 BOS, 257 EOS, 258 PAD |
| chapters/08-bytes-as-tokens.md:329 | anulm/training/train.py | 656 |  | override | decided in repin-overrides.tsv |  |
| chapters/08-bytes-as-tokens.md:329 | anulm/training/train.py | 656 | 667 | same | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 |
| chapters/08-bytes-as-tokens.md:331 | anulm/training/train.py | 652–655 |  | override | decided in repin-overrides.tsv |  |
| chapters/08-bytes-as-tokens.md:331 | anulm/training/train.py | 652–655 | 663–666 | same | # --- data ----------------------------------------------------------- | # --- data ----------------------------------------------------------- |
| chapters/08-bytes-as-tokens.md:383 | anulm/model.py | 1171–1179 | 1181–1189 | same | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) |
| chapters/08-bytes-as-tokens.md:409 | anulm/training/train.py | 688 | 699 | same | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") |
| chapters/08-bytes-as-tokens.md:470 | anulm/model.py | 47 | 43 | same | vocab_size: int = 259           # 256 bytes + BOS + EOS + PAD (byte mo | vocab_size: int = 259           # 256 bytes + BOS + EOS + PAD (byte mo |
| chapters/08-bytes-as-tokens.md:470 | anulm/model.py | 1171–1179 | 1181–1189 | same | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) |
| chapters/08-bytes-as-tokens.md:472 | anulm/training/train.py | 28 | 27 | same | from anulm.bpe import DOC_SEP | from anulm.bpe import DOC_SEP |
| chapters/08-bytes-as-tokens.md:472 | anulm/training/train.py | 51–59 | 54–62 | same | FALLBACK = """भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं। | FALLBACK = """भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं। |
| chapters/08-bytes-as-tokens.md:472 | anulm/training/train.py | 62–98 | 65–101 | same | def load_bytes(path: str \| None) -> torch.Tensor: | def load_bytes(path: str \| None) -> torch.Tensor: |
| chapters/08-bytes-as-tokens.md:473 | anulm/training/train.py | 101 | 104 | same | BYTE_EOS = 257          # byte mode: 256 BOS, 257 EOS, 258 PAD | BYTE_EOS = 257          # byte mode: 256 BOS, 257 EOS, 258 PAD |
| chapters/08-bytes-as-tokens.md:474 | anulm/training/train.py | 652–656 | 663–667 | same | # --- data ----------------------------------------------------------- | # --- data ----------------------------------------------------------- |
| chapters/08-bytes-as-tokens.md:474 | anulm/training/train.py | 688 | 699 | same | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") |
| chapters/08-bytes-as-tokens.md:474 | anulm/training/train.py | 906–908 | 915–917 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/08-bytes-as-tokens.md:478 | tests/test_model.py | 1595–1613 | 1593–1611 | same | def byte_loader_inserts_eos_at_document_boundaries(): | def byte_loader_inserts_eos_at_document_boundaries(): |
| chapters/08-bytes-as-tokens.md:478 | tests/test_model.py | 1595–1613 | 1593–1611 | sweep | 1595–1613. |  |
| chapters/09-byte-level-bpe.md:478 | anulm/bpe.py | 344–352 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/09-byte-level-bpe.md:214 | anulm/bpe.py | 120–121 |  | override | decided in repin-overrides.tsv |  |
| chapters/09-byte-level-bpe.md:427 | tests/test_model.py | 1520–1523 |  | reattributed | quoted code is in tests/test_model.py, not anulm/bpe.py |  |
| chapters/09-byte-level-bpe.md:463 | ? |  |  | other-pin | lines 220–239 @c4a39f8 |  |
| chapters/09-byte-level-bpe.md:465 | ? |  |  | other-pin | lines 236–238 @c4a39f8) |  |
| chapters/09-byte-level-bpe.md:467 | anulm/bpe.py | 297–345 |  | reattributed | 'cmd_encode' is in anulm/bpe.py, not None |  |
| chapters/09-byte-level-bpe.md:467 | anulm/bpe.py | 297–345 | 293–341 | same | def cmd_encode(args): | def cmd_encode(args): |
| chapters/09-byte-level-bpe.md:541 | anulm/bpe.py | 120–121 |  | override | decided in repin-overrides.tsv |  |
| chapters/09-byte-level-bpe.md:603 | anulm/bpe.py | 64–83 | 64–83 | sweep | 64–83, 'BP |  |
| chapters/09-byte-level-bpe.md:604 | anulm/bpe.py | 221–244 | ? | UNMAPPED | # The format id the project wrote before it was renamed on 2026-09-18. |  |
| chapters/09-byte-level-bpe.md:605 | anulm/bpe.py |  |  | other-pin | 344–352 @c4a39f8) |  |
| chapters/09-byte-level-bpe.md:605 | anulm/bpe.py | 297–345 | 293–341 | same | def cmd_encode(args): | def cmd_encode(args): |
| chapters/09-byte-level-bpe.md:605 | anulm/bpe.py | 297–345 | 293–341 | sweep | 297–345, '_s |  |
| chapters/09-byte-level-bpe.md:605 | anulm/bpe.py | 348–356 | 344–352 | same | def _stats(tok, text): | def _stats(tok, text): |
| chapters/09-byte-level-bpe.md:608 | tests/test_model.py |  |  | other-pin | 1544 (@c4a39f8) |  |
| chapters/10-embeddings-and-the-output-table.md:58 | anulm/model.py | 827–843 | 837–853 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/10-embeddings-and-the-output-table.md:166 | anulm/model.py | 896 | 906 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| chapters/10-embeddings-and-the-output-table.md:172 | anulm/model.py | 966–970 | 976–980 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/10-embeddings-and-the-output-table.md:182 | anulm/model.py | 992–995 | 1002–1005 | same | def _head(self, h): | def _head(self, h): |
| chapters/10-embeddings-and-the-output-table.md:81 | anulm/model.py | 843 | 853 | same | self.lm_head.weight = self.embed_tokens.weight | self.lm_head.weight = self.embed_tokens.weight |
| chapters/10-embeddings-and-the-output-table.md:83 | anulm/model.py |  |  | other-pin | line 113 @c4a39f8 |  |
| chapters/10-embeddings-and-the-output-table.md:101 | anulm/model.py | 118 |  | reattributed | quoted code is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/10-embeddings-and-the-output-table.md:101 | anulm/model.py | 118 | 113 | same | initializer_range: float = 0.02 | initializer_range: float = 0.02 |
| chapters/10-embeddings-and-the-output-table.md:224 | anulm/training/train.py | 447–474 | 450–482 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/10-embeddings-and-the-output-table.md:226 | anulm/training/train.py | 555 |  | reattributed | quoted code is in anulm/training/train.py, not tools/windows/_v2_run.c |  |
| chapters/10-embeddings-and-the-output-table.md:226 | anulm/training/train.py | 555 | 564 | same | p.add_argument("--weight-decay", type=float, default=0.1) | p.add_argument("--weight-decay", type=float, default=0.1) |
| chapters/10-embeddings-and-the-output-table.md:287 | anulm/model.py | 390 |  | reattributed | 'merge_tree_matrix' is in anulm/model.py, not tools/windows/_v2_run.cm |  |
| chapters/10-embeddings-and-the-output-table.md:287 | anulm/model.py | 390 | 400 | same | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: |
| chapters/10-embeddings-and-the-output-table.md:290 | anulm/model.py | 997–1013 |  | reattributed | 'bake_merge_tree' is in anulm/model.py, not tools/windows/_v2_run.cmd |  |
| chapters/10-embeddings-and-the-output-table.md:290 | anulm/model.py | 997–1013 | 1007–1023 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/10-embeddings-and-the-output-table.md:299 | anulm/model.py | 390–411 | 400–421 | same | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: |
| chapters/10-embeddings-and-the-output-table.md:327 | anulm/model.py | 397 |  | override | decided in repin-overrides.tsv |  |
| chapters/10-embeddings-and-the-output-table.md:327 | anulm/model.py | 397 | 407 | same | coef = [{i: 1.0} for i in range(256)] | coef = [{i: 1.0} for i in range(256)] |
| chapters/10-embeddings-and-the-output-table.md:343 | tests/test_model.py | 1819–1826 | 1817–1824 | same | merges = [[97, 98], [256, 99], [257, 257], [100, 101]]      # ab, abc, | merges = [[97, 98], [256, 99], [257, 257], [100, 101]]      # ab, abc, |
| chapters/10-embeddings-and-the-output-table.md:359 | anulm/model.py | 959–964 | 969–974 | same | def _compose(self, W): | def _compose(self, W): |
| chapters/10-embeddings-and-the-output-table.md:372 | anulm/model.py | 971–990 | 981–1000 | same | if self.training: | if self.training: |
| chapters/10-embeddings-and-the-output-table.md:400 | tests/test_model.py | 1849 | 1847 | def | 'merge_tree_eval_cache_sees_optimizer_updates' | def merge_tree_eval_cache_sees_optimizer_updates(): |
| chapters/10-embeddings-and-the-output-table.md:406 | anulm/model.py | 997–1013 | 1007–1023 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/10-embeddings-and-the-output-table.md:505 | anulm/model.py | 117–118 | 112–113 | edited | tie_word_embeddings: bool = False   # Sarvam unties both models | tie_word_embeddings: bool = False   # untied input and output tables |
| chapters/10-embeddings-and-the-output-table.md:505 | anulm/model.py | 119–124 | 114–119 | same | # Merge-tree reparameterisation of the token tables ("in", "out", "bot | # Merge-tree reparameterisation of the token tables ("in", "out", "bot |
| chapters/10-embeddings-and-the-output-table.md:506 | anulm/model.py | 390–411 | 400–421 | same | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: |
| chapters/10-embeddings-and-the-output-table.md:507 | anulm/model.py | 827–863 | 837–873 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/10-embeddings-and-the-output-table.md:507 | anulm/model.py | 865–871 | 875–881 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/10-embeddings-and-the-output-table.md:508 | anulm/model.py | 896 | 906 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| chapters/10-embeddings-and-the-output-table.md:508 | anulm/model.py | 959–964 | 969–974 | same | def _compose(self, W): | def _compose(self, W): |
| chapters/10-embeddings-and-the-output-table.md:508 | anulm/model.py | 966–984 | 976–994 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/10-embeddings-and-the-output-table.md:509 | anulm/model.py | 986–990 | 996–1000 | same | def train(self, mode: bool = True): | def train(self, mode: bool = True): |
| chapters/10-embeddings-and-the-output-table.md:509 | anulm/model.py | 992–995 | 1002–1005 | same | def _head(self, h): | def _head(self, h): |
| chapters/10-embeddings-and-the-output-table.md:509 | anulm/model.py | 997–1013 | 1007–1023 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/10-embeddings-and-the-output-table.md:510 | anulm/training/train.py | 447–474 | 450–482 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/10-embeddings-and-the-output-table.md:512 | tests/test_model.py | 1813 | 1811 | same | def merge_tree_composes_trains_and_bakes_losslessly(): | def merge_tree_composes_trains_and_bakes_losslessly(): |
| chapters/10-embeddings-and-the-output-table.md:513 | tests/test_model.py | 1849 | 1847 | same | def merge_tree_eval_cache_sees_optimizer_updates(): | def merge_tree_eval_cache_sees_optimizer_updates(): |
| chapters/11-attention.md:278 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/11-attention.md:295 | anulm/model.py | 440–459 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/11-attention.md:332 | anulm/model.py | 461–486 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/11-attention.md:365 | anulm/model.py | 477–494 | 487–504 | same | T = k.shape[2] | T = k.shape[2] |
| chapters/11-attention.md:463 | anulm/model.py | 225–239 | 235–249 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/11-attention.md:326 | anulm/model.py | 308 |  | override | decided in repin-overrides.tsv |  |
| chapters/11-attention.md:326 | anulm/model.py | 308 | 318 | same | if not cfg.yarn: | if not cfg.yarn: |
| chapters/11-attention.md:394 | anulm/model.py | 387 | 397 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/11-attention.md:405 | anulm/model.py | 374 | 384 | same | if window is None and (start == 0 or S == 1): | if window is None and (start == 0 or S == 1): |
| chapters/11-attention.md:406 | anulm/model.py | 416 | 426 | def | 'is_causal' | q, k, v, attn_mask=mask, is_causal=(mask is None and S > 1), |
| chapters/11-attention.md:412 | tests/test_model.py | 473–477 |  | reattributed | quoted code is in tests/test_model.py, not anulm/model.py |  |
| chapters/11-attention.md:416 | tests/test_model.py | 750 |  | reattributed | 'sliding_window_limits_reach' is in tests/test_model.py, not anulm/mod |  |
| chapters/11-attention.md:422 | anulm/model.py | 446–448 |  | override | decided in repin-overrides.tsv |  |
| chapters/11-attention.md:422 | anulm/model.py | 446–448 | 456–458 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/11-attention.md:459 | anulm/model.py | 465–467 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not tests/test_model.py |  |
| chapters/11-attention.md:459 | anulm/model.py | 465–467 | 475–477 | edited | if self.use_qk_norm:                          # before RoPE, as in the | if self.use_qk_norm:                          # before RoPE |
| chapters/11-attention.md:482 | anulm/model.py | 231 |  | override | decided in repin-overrides.tsv |  |
| chapters/11-attention.md:482 | anulm/model.py | 231 | 241 | same | self.weight = nn.Parameter(torch.ones(dim)) | self.weight = nn.Parameter(torch.ones(dim)) |
| chapters/11-attention.md:529 | book/code-examples/attention_by_hand.py |  |  | other-pin | line 475 (@c4a39f8) |  |
| chapters/11-attention.md:667 | anulm/model.py | 482 |  | override | decided in repin-overrides.tsv |  |
| chapters/11-attention.md:667 | anulm/model.py | 482 | 492 | same | drop = self.dropout if self.training else 0.0 | drop = self.dropout if self.training else 0.0 |
| chapters/11-attention.md:743 | anulm/model.py | 225–239 | 235–249 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/11-attention.md:743 | anulm/model.py | 298 | 308 | same | self.attn_mscale = self.table_mscale = 1.0 | self.attn_mscale = self.table_mscale = 1.0 |
| chapters/11-attention.md:744 | anulm/model.py | 365–381 | 375–391 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/11-attention.md:744 | anulm/model.py | 387 | 397 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/11-attention.md:744 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/11-attention.md:745 | anulm/model.py | 430–449 | 440–459 | same | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): |
| chapters/11-attention.md:745 | anulm/model.py | 451–494 | 461–504 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/12-causal-masking.md:71 | anulm/model.py | 138–140 | 133–135 | same | if self.sliding_window is not None: | if self.sliding_window is not None: |
| chapters/12-causal-masking.md:88 | anulm/model.py | 365–381 | 375–391 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/12-causal-masking.md:146 | anulm/model.py | 477–482 | 487–492 | same | T = k.shape[2] | T = k.shape[2] |
| chapters/12-causal-masking.md:190 | anulm/model.py | 1144–1151 | 1154–1161 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/12-causal-masking.md:229 | anulm/model.py | 812–814 | 822–824 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/12-causal-masking.md:79 | anulm/model.py | 129–131 | 124–126 | same | def check(ok, msg): | def check(ok, msg): |
| chapters/12-causal-masking.md:158 | anulm/model.py | 575 |  | override | decided in repin-overrides.tsv |  |
| chapters/12-causal-masking.md:158 | anulm/model.py | 575 | 585 | same | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star |
| chapters/12-causal-masking.md:204 | anulm/model.py | 1173 | 1183 | same | pending = nxt | pending = nxt |
| chapters/12-causal-masking.md:538 | anulm/model.py | 138–140 | 133–135 | same | if self.sliding_window is not None: | if self.sliding_window is not None: |
| chapters/12-causal-masking.md:538 | anulm/model.py | 365–381 | 375–391 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/12-causal-masking.md:539 | anulm/model.py | 477–482 | 487–492 | same | T = k.shape[2] | T = k.shape[2] |
| chapters/12-causal-masking.md:539 | anulm/model.py | 575 | 585 | same | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star |
| chapters/12-causal-masking.md:540 | anulm/model.py | 812–814 | 822–824 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/12-causal-masking.md:540 | anulm/model.py | 812–814 | 822–824 | sweep | 812–814, 'An |  |
| chapters/12-causal-masking.md:540 | anulm/model.py | 1144–1151 | 1154–1161 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/12-causal-masking.md:540 | anulm/model.py | 1173 | 1183 | same | pending = nxt | pending = nxt |
| chapters/13-multi-head-and-gqa.md:31 | anulm/model.py | 150–154 | 145–149 | same | if self.attn == "gqa": | if self.attn == "gqa": |
| chapters/13-multi-head-and-gqa.md:156 | anulm/model.py | 483–494 | 493–504 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/13-multi-head-and-gqa.md:42 | anulm/model.py |  |  | other-pin | 188 @c4a39f8) |  |
| chapters/13-multi-head-and-gqa.md:44 | anulm/model.py | 129–131 | 124–126 | same | def check(ok, msg): | def check(ok, msg): |
| chapters/13-multi-head-and-gqa.md:51 | anulm/model.py | 430–449 |  | reattributed | 'GQAttention.__init__' is in anulm/model.py, not None |  |
| chapters/13-multi-head-and-gqa.md:51 | anulm/model.py | 430–449 | 440–459 | same | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): |
| chapters/13-multi-head-and-gqa.md:95 | anulm/model.py | 451–468 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not None |  |
| chapters/13-multi-head-and-gqa.md:95 | anulm/model.py | 451–468 | 461–478 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/13-multi-head-and-gqa.md:195 | anulm/model.py | 387 | 397 | def | 'NATIVE_GQA' | NATIVE_GQA = False |
| chapters/13-multi-head-and-gqa.md:201 | anulm/model.py | 384–387 | 394–397 | same | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa |
| chapters/13-multi-head-and-gqa.md:210 | anulm/training/train.py | 637–643 |  | override | decided in repin-overrides.tsv |  |
| chapters/13-multi-head-and-gqa.md:210 | anulm/training/train.py | 637–643 | 648–654 | same | p.add_argument("--native-gqa", action="store_true", | p.add_argument("--native-gqa", action="store_true", |
| chapters/13-multi-head-and-gqa.md:212 | tests/test_model.py | 1767–1796 | 1765–1794 | same | def native_gqa_matches_repeat_interleave(): | def native_gqa_matches_repeat_interleave(): |
| chapters/13-multi-head-and-gqa.md:286 | anulm/model.py | 756 | 766 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/13-multi-head-and-gqa.md:286 | anulm/model.py | 784 | 794 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/13-multi-head-and-gqa.md:370 | anulm/model.py | 129–131 | 124–126 | same | def check(ok, msg): | def check(ok, msg): |
| chapters/13-multi-head-and-gqa.md:370 | anulm/model.py | 150–154 | 145–149 | same | if self.attn == "gqa": | if self.attn == "gqa": |
| chapters/13-multi-head-and-gqa.md:370 | anulm/model.py | 384–387 | 394–397 | same | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa |
| chapters/13-multi-head-and-gqa.md:371 | anulm/model.py | 430–445 | 440–455 | same | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): |
| chapters/13-multi-head-and-gqa.md:371 | anulm/model.py | 449 | 459 | same | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) |
| chapters/13-multi-head-and-gqa.md:371 | anulm/model.py | 451–494 | 461–504 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/13-multi-head-and-gqa.md:372 | anulm/training/train.py | 637–643 | 648–654 | same | p.add_argument("--native-gqa", action="store_true", | p.add_argument("--native-gqa", action="store_true", |
| chapters/13-multi-head-and-gqa.md:373 | tests/test_model.py | 1767 | 1765 | same | def native_gqa_matches_repeat_interleave(): | def native_gqa_matches_repeat_interleave(): |
| chapters/14-rotary-positions.md:116 | anulm/model.py | 304–309 | 314–319 | same | @staticmethod | @staticmethod |
| chapters/14-rotary-positions.md:162 | anulm/model.py | 324–337 | 334–347 | same | def _grow(self, seq_len: int, device, dtype): | def _grow(self, seq_len: int, device, dtype): |
| chapters/14-rotary-positions.md:183 | anulm/model.py | 339–343 | 349–353 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| chapters/14-rotary-positions.md:255 | anulm/model.py | 346–350 | 356–360 | same | def rotate_half(x): | def rotate_half(x): |
| chapters/14-rotary-positions.md:263 | anulm/model.py | 363–372 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/14-rotary-positions.md:158 | anulm/model.py | 837–839 |  | reattributed | 'AnuLM.__init__' is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/14-rotary-positions.md:158 | anulm/model.py | 837–839 | 847–849 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/14-rotary-positions.md:159 | anulm/model.py | 899 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/14-rotary-positions.md:159 | anulm/model.py | 899 | 909 | same | cos, sin = self.rotary(S, x.device, x.dtype, start=start) | cos, sin = self.rotary(S, x.device, x.dtype, start=start) |
| chapters/14-rotary-positions.md:197 | anulm/model.py | 301–302 | 311–312 | same | self.register_buffer("cos_cached", torch.empty(0), persistent=False) | self.register_buffer("cos_cached", torch.empty(0), persistent=False) |
| chapters/14-rotary-positions.md:200 | anulm/model.py | 282–287 | 292–297 | same | def __init__(self, cfg: AnuLMConfig, dim: int): | def __init__(self, cfg: AnuLMConfig, dim: int): |
| chapters/14-rotary-positions.md:213 | anulm/model.py | 325–329 | 335–339 | same | # Keyed on device and dtype as well as length: a model moved with .to( | # Keyed on device and dtype as well as length: a model moved with .to( |
| chapters/14-rotary-positions.md:215 | anulm/model.py | 324–337 | 334–347 | same | def _grow(self, seq_len: int, device, dtype): | def _grow(self, seq_len: int, device, dtype): |
| chapters/14-rotary-positions.md:366 | anulm/model.py | 756 | 766 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/14-rotary-positions.md:366 | anulm/model.py | 784 | 794 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/14-rotary-positions.md:475 | ? |  |  | other-pin | 363–372 @c4a39f8) |  |
| chapters/14-rotary-positions.md:475 | anulm/model.py | 278–343 | 288–353 | same | class Rotary(nn.Module): | class Rotary(nn.Module): |
| chapters/14-rotary-positions.md:475 | anulm/model.py | 346–350 | 356–360 | same | def rotate_half(x): | def rotate_half(x): |
| chapters/14-rotary-positions.md:475 | anulm/model.py | 353–362 | 363–372 | same | def apply_rope(q, k, cos, sin): | def apply_rope(q, k, cos, sin): |
| chapters/14-rotary-positions.md:476 | anulm/model.py | 756 | 766 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/14-rotary-positions.md:476 | anulm/model.py | 784 | 794 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/14-rotary-positions.md:476 | anulm/model.py | 837–839 | 847–849 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/14-rotary-positions.md:476 | anulm/model.py | 899 | 909 | same | cos, sin = self.rotary(S, x.device, x.dtype, start=start) | cos, sin = self.rotary(S, x.device, x.dtype, start=start) |
| chapters/15-normalisation-and-the-residual-stream.md:73 | anulm/model.py | 225–239 | 235–249 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/15-normalisation-and-the-residual-stream.md:142 | anulm/model.py | 802–811 | 812–821 | same | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): |
| chapters/15-normalisation-and-the-residual-stream.md:159 | anulm/model.py | 816–824 | 826–834 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/15-normalisation-and-the-residual-stream.md:181 | anulm/model.py | 839–841 | 849–851 | same | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range |
| chapters/15-normalisation-and-the-residual-stream.md:189 | anulm/model.py | 912 | 922 | same | x = self.norm(x) | x = self.norm(x) |
| chapters/15-normalisation-and-the-residual-stream.md:272 | anulm/model.py | 859–863 | 869–873 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/15-normalisation-and-the-residual-stream.md:282 | anulm/model.py | 865–871 | 875–881 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/15-normalisation-and-the-residual-stream.md:436 | anulm/model.py | 225–239 | 235–249 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/15-normalisation-and-the-residual-stream.md:436 | anulm/model.py | 802–811 | 812–821 | same | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): |
| chapters/15-normalisation-and-the-residual-stream.md:437 | anulm/model.py | 816–824 | 826–834 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/15-normalisation-and-the-residual-stream.md:437 | anulm/model.py | 816–824 | 826–834 | sweep | 816–824, 'An |  |
| chapters/15-normalisation-and-the-residual-stream.md:437 | anulm/model.py | 839–841 | 849–851 | same | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range |
| chapters/15-normalisation-and-the-residual-stream.md:437 | anulm/model.py | 859–863 | 869–873 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/15-normalisation-and-the-residual-stream.md:437 | anulm/model.py | 865–871 | 875–881 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/15-normalisation-and-the-residual-stream.md:438 | anulm/model.py | 912 | 922 | same | x = self.norm(x) | x = self.norm(x) |
| chapters/15-normalisation-and-the-residual-stream.md:439 | anulm/multimodal/speech_encoder.py | 79–82 | 79–82 | sweep | - 'anulm/multimodal/speech_encoder.py' 79–82 and |  |
| chapters/15-normalisation-and-the-residual-stream.md:439 | anulm/multimodal/vision_encoder.py | 124 | 124 | sweep | –82 and 'anulm/multimodal/vision_encoder.py' 124 (Lay |  |
| chapters/16-swiglu.md:90 | anulm/model.py | 242–253 | 252–263 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/16-swiglu.md:158 | anulm/model.py | 77–79 | 72–74 | same | moe_intermediate_size: int = 96 | moe_intermediate_size: int = 96 |
| chapters/16-swiglu.md:169 | anulm/model.py | 187–196 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/16-swiglu.md:231 | anulm/model.py | 809–811 | 819–821 | edited | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in both mod | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in every pr |
| chapters/16-swiglu.md:243 | anulm/model.py | 688–694 | 698–704 | same | self.experts = nn.ModuleList( | self.experts = nn.ModuleList( |
| chapters/16-swiglu.md:260 | anulm/model.py | 860–863 | 870–873 | same | # nanoGPT's scaled init for the residual output projections. | # nanoGPT's scaled init for the residual output projections. |
| chapters/16-swiglu.md:426 | anulm/model.py | 77–79 | 72–74 | same | moe_intermediate_size: int = 96 | moe_intermediate_size: int = 96 |
| chapters/16-swiglu.md:426 | anulm/model.py | 242–253 | 252–263 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/16-swiglu.md:426 | anulm/model.py | 688–694 | 698–704 | same | self.experts = nn.ModuleList( | self.experts = nn.ModuleList( |
| chapters/16-swiglu.md:427 | anulm/model.py |  |  | other-pin | at @c4a39f8, 'tiny_gqa' |  |
| chapters/16-swiglu.md:427 | anulm/model.py | 809–811 | 819–821 | edited | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in both mod | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in every pr |
| chapters/16-swiglu.md:427 | anulm/model.py | 860–863 | 870–873 | same | # nanoGPT's scaled init for the residual output projections. | # nanoGPT's scaled init for the residual output projections. |
| chapters/16-swiglu.md:428 | anulm/model.py |  |  | other-pin | line 163, 'base_350m' 187–196, 'tiny_mla' line 223. |  |
| chapters/17-router.md:91 | anulm/model.py | 594–609 | 604–619 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/17-router.md:133 | anulm/model.py | 611–622 | 621–632 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/17-router.md:182 | anulm/model.py | 624–645 | 634–655 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/17-router.md:258 | anulm/model.py | 629–638 | 639–648 | same | if self.n_group > 1: | if self.n_group > 1: |
| chapters/17-router.md:326 | anulm/model.py | 144–148 | 139–143 | same | per_group = self.num_experts // self.n_group | per_group = self.num_experts // self.n_group |
| chapters/17-router.md:14 | anulm/model.py | 585–672 | 595–682 | same | class Router(nn.Module): | class Router(nn.Module): |
| chapters/17-router.md:336 | anulm/model.py | 129–131 | 124–126 | same | def check(ok, msg): | def check(ok, msg): |
| chapters/17-router.md:339 | anulm/model.py | 629–638 |  | override | decided in repin-overrides.tsv |  |
| chapters/17-router.md:339 | anulm/model.py | 629–638 | 639–648 | same | if self.n_group > 1: | if self.n_group > 1: |
| chapters/17-router.md:477 | anulm/model.py | 144–148 | 139–143 | same | per_group = self.num_experts // self.n_group | per_group = self.num_experts // self.n_group |
| chapters/17-router.md:477 | anulm/model.py | 594–609 | 604–619 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/17-router.md:478 | anulm/model.py | 611–622 | 621–632 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/17-router.md:478 | anulm/model.py | 624–645 | 634–655 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/17-router.md:478 | anulm/model.py | 629–638 | 639–648 | same | if self.n_group > 1: | if self.n_group > 1: |
| chapters/17-router.md:478 | anulm/model.py | 629–638 | 639–648 | sweep | ard' 611–622, 'Router._gate' 624–645 (groups 629–638). |  |
| chapters/18-dispatch-and-balance.md:68 | anulm/model.py | 719–737 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/18-dispatch-and-balance.md:125 | anulm/model.py | 729–738 | 739–748 | same | def _route_dense(self, x_flat, topk_idx, topk_w): | def _route_dense(self, x_flat, topk_idx, topk_w): |
| chapters/18-dispatch-and-balance.md:144 | anulm/model.py | 775–790 | 785–800 | same | N, D = x_flat.shape | N, D = x_flat.shape |
| chapters/18-dispatch-and-balance.md:175 | anulm/model.py | 750–759 | 760–769 | same | # Keyed on the parameters' in-place version counters, so a | # Keyed on the parameters' in-place version counters, so a |
| chapters/18-dispatch-and-balance.md:276 | anulm/model.py | 647–652 | 657–662 | same | aux = None | aux = None |
| chapters/18-dispatch-and-balance.md:289 | anulm/training/train.py | 876–878 | 885–887 | same | opt.step() | opt.step() |
| chapters/18-dispatch-and-balance.md:297 | anulm/model.py | 1081–1101 | 1091–1111 | same | ratios = [] | ratios = [] |
| chapters/18-dispatch-and-balance.md:412 | anulm/model.py | 653–670 | 663–680 | same | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: |
| chapters/18-dispatch-and-balance.md:268 | anulm/model.py | 753–754 |  | override | decided in repin-overrides.tsv |  |
| chapters/18-dispatch-and-balance.md:268 | anulm/model.py | 753–754 | 763–764 | same | if not self.training and self._stacked is not None and self._stacked[0 | if not self.training and self._stacked is not None and self._stacked[0 |
| chapters/18-dispatch-and-balance.md:569 | anulm/model.py | 647–652 | 657–662 | same | aux = None | aux = None |
| chapters/18-dispatch-and-balance.md:569 | anulm/model.py | 653–670 | 663–680 | same | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: |
| chapters/18-dispatch-and-balance.md:569 | anulm/model.py | 709–727 | 719–737 | same | def _route_sparse(self, x_flat, topk_idx, topk_w): | def _route_sparse(self, x_flat, topk_idx, topk_w): |
| chapters/18-dispatch-and-balance.md:570 | anulm/model.py | 729–738 | 739–748 | same | def _route_dense(self, x_flat, topk_idx, topk_w): | def _route_dense(self, x_flat, topk_idx, topk_w): |
| chapters/18-dispatch-and-balance.md:570 | anulm/model.py | 750–759 | 760–769 | same | # Keyed on the parameters' in-place version counters, so a | # Keyed on the parameters' in-place version counters, so a |
| chapters/18-dispatch-and-balance.md:571 | anulm/model.py | 775–790 | 785–800 | same | N, D = x_flat.shape | N, D = x_flat.shape |
| chapters/18-dispatch-and-balance.md:571 | anulm/model.py | 775–790 | 785–800 | sweep | 775–790; 'An |  |
| chapters/18-dispatch-and-balance.md:571 | anulm/model.py | 1071–1101 | 1081–1111 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/18-dispatch-and-balance.md:572 | anulm/training/train.py | 876–878 | 885–887 | same | opt.step() | opt.step() |
| chapters/18-dispatch-and-balance.md:576 | tests/test_model.py | 603 | 603 | sweep | 603, 'gr |  |
| chapters/19-latent-attention.md:73 | anulm/model.py | 516–533 | 526–543 | same | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, |
| chapters/19-latent-attention.md:111 | anulm/model.py | 150–156 | 145–151 | same | if self.attn == "gqa": | if self.attn == "gqa": |
| chapters/19-latent-attention.md:131 | anulm/model.py | 539–553 | 549–563 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/19-latent-attention.md:158 | anulm/model.py | 564–578 | 574–588 | same | T = compressed.shape[1] | T = compressed.shape[1] |
| chapters/19-latent-attention.md:230 | anulm/model.py | 565–573 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/19-latent-attention.md:65 | anulm/model.py | 550–551 |  | override | decided in repin-overrides.tsv |  |
| chapters/19-latent-attention.md:65 | anulm/model.py | 550–551 | 560–561 | same | # RoPE touches only the rope slice; the latent stays position-free -- | # RoPE touches only the rope slice; the latent stays position-free -- |
| chapters/19-latent-attention.md:125 | anulm/model.py | 837 | 847 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/19-latent-attention.md:415 | anulm/model.py | 837 |  | reattributed | quoted code is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/19-latent-attention.md:415 | anulm/model.py | 837 | 847 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/19-latent-attention.md:470 | anulm/model.py | 150–156 | 145–151 | same | if self.attn == "gqa": | if self.attn == "gqa": |
| chapters/19-latent-attention.md:471 | anulm/model.py |  |  | other-pin | 565–573 @c4a39f8) |  |
| chapters/19-latent-attention.md:471 | anulm/model.py | 516–533 | 526–543 | same | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, |
| chapters/19-latent-attention.md:471 | anulm/model.py | 516–533 | 526–543 | sweep | 516–533, 'ML |  |
| chapters/19-latent-attention.md:471 | anulm/model.py | 539–553 | 549–563 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/19-latent-attention.md:471 | anulm/model.py | 564–578 | 574–588 | same | T = compressed.shape[1] | T = compressed.shape[1] |
| chapters/19-latent-attention.md:472 | anulm/model.py | 837 |  | override | decided in repin-overrides.tsv |  |
| chapters/19-latent-attention.md:472 | anulm/model.py | 837 | 847 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/20-the-whole-model.md:121 | anulm/model.py | 891–899 | 901–909 | same | if embeds is not None: | if embeds is not None: |
| chapters/20-the-whole-model.md:157 | anulm/model.py | 900–911 | 910–921 | same | aux_total = None | aux_total = None |
| chapters/20-the-whole-model.md:218 | anulm/model.py | 696–707 | 706–717 | same | def forward(self, x):                                     # (B, S, D) | def forward(self, x):                                     # (B, S, D) |
| chapters/20-the-whole-model.md:274 | anulm/model.py | 912–928 | 922–938 | same | x = self.norm(x) | x = self.norm(x) |
| chapters/20-the-whole-model.md:383 | anulm/model.py | 1104–1113 | 1114–1123 | same | def num_params(self) -> tuple[int, int]: | def num_params(self) -> tuple[int, int]: |
| chapters/20-the-whole-model.md:107 | anulm/model.py | 833–863 | 843–873 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/20-the-whole-model.md:138 | anulm/model.py | 324–343 | 334–353 | same | def _grow(self, seq_len: int, device, dtype): | def _grow(self, seq_len: int, device, dtype): |
| chapters/20-the-whole-model.md:177 | anulm/model.py | 816–824 | 826–834 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/20-the-whole-model.md:187 | anulm/model.py | 451–494 | 461–504 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/20-the-whole-model.md:195 | anulm/model.py | 387 | 397 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/20-the-whole-model.md:213 | anulm/model.py | 242–253 |  | reattributed | 'SwiGLU' is in anulm/model.py, not book/code-examples/trace_forward.py |  |
| chapters/20-the-whole-model.md:213 | anulm/model.py | 242–253 | 252–263 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/20-the-whole-model.md:236 | anulm/model.py | 611–622 | 621–632 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/20-the-whole-model.md:238 | anulm/model.py | 624–672 | 634–682 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/20-the-whole-model.md:300 | anulm/model.py | 841–843 | 851–853 | same | self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False) | self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False) |
| chapters/20-the-whole-model.md:300 | anulm/model.py | 117 | 112 | edited | tie_word_embeddings: bool = False   # Sarvam unties both models | tie_word_embeddings: bool = False   # untied input and output tables |
| chapters/20-the-whole-model.md:304 | anulm/training/train.py | 237–250 | 240–253 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/20-the-whole-model.md:333 | anulm/model.py | 1152–1169 |  | reattributed | 'AnuLM.generate' is in anulm/model.py, not docs/RESULTS.md |  |
| chapters/20-the-whole-model.md:333 | anulm/model.py | 1152–1169 | 1162–1179 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/20-the-whole-model.md:554 | anulm/model.py | 225–239 | 235–249 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/20-the-whole-model.md:554 | anulm/model.py | 242–253 | 252–263 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/20-the-whole-model.md:554 | anulm/model.py | 324–343 | 334–353 | same | def _grow(self, seq_len: int, device, dtype): | def _grow(self, seq_len: int, device, dtype): |
| chapters/20-the-whole-model.md:555 | anulm/model.py | 387 | 397 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/20-the-whole-model.md:556 | anulm/model.py | 451–494 | 461–504 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/20-the-whole-model.md:556 | anulm/model.py | 611–622 | 621–632 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/20-the-whole-model.md:557 | anulm/model.py | 624–672 | 634–682 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/20-the-whole-model.md:557 | anulm/model.py | 696–707 | 706–717 | same | def forward(self, x):                                     # (B, S, D) | def forward(self, x):                                     # (B, S, D) |
| chapters/20-the-whole-model.md:557 | anulm/model.py | 816–824 | 826–834 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/20-the-whole-model.md:558 | anulm/model.py | 833–863 | 843–873 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/20-the-whole-model.md:558 | anulm/model.py | 877–928 | 887–938 | same | def forward(self, idx, targets=None, cache: Optional[list] = None, sta | def forward(self, idx, targets=None, cache: Optional[list] = None, sta |
| chapters/20-the-whole-model.md:559 | anulm/model.py | 1104–1113 | 1114–1123 | same | def num_params(self) -> tuple[int, int]: | def num_params(self) -> tuple[int, int]: |
| chapters/20-the-whole-model.md:559 | anulm/model.py | 1115–1180 | 1125–1190 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/20-the-whole-model.md:560 | anulm/training/train.py | 237–250 | 240–253 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/21-data.md:138 | anulm/training/train.py | 659–672 | 670–683 | same | memmapped = bool(args.data and args.data.endswith("meta.json")) | memmapped = bool(args.data and args.data.endswith("meta.json")) |
| chapters/21-data.md:158 | anulm/training/train.py | 163–173 | 166–176 | same | def load_memmap(meta_path: str): | def load_memmap(meta_path: str): |
| chapters/21-data.md:176 | anulm/training/train.py | 237–250 | 240–253 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/21-data.md:273 | anulm/training/train.py | 118–160 | 121–163 | same | def __init__(self, n_tokens: int, block_size: int, batch_size: int, se | def __init__(self, n_tokens: int, block_size: int, batch_size: int, se |
| chapters/21-data.md:394 | anulm/training/train.py | 689–695 | 700–706 | same | if memmapped or (split_encoded and val_data is not None): | if memmapped or (split_encoded and val_data is not None): |
| chapters/21-data.md:412 | anulm/training/train.py | 202–216 | 205–219 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/21-data.md:59 | experiments/build_v2.py | 121 |  | override | decided in repin-overrides.tsv |  |
| chapters/21-data.md:82 | experiments/build_v2.py | 44 |  | reattributed | quoted code is in experiments/build_v2.py, not None |  |
| chapters/21-data.md:313 | anulm/training/train.py | 147–150 | 150–153 | same | @property | @property |
| chapters/21-data.md:353 | anulm/training/train.py | 590 |  | reattributed | quoted code is in anulm/training/train.py, not book/code-examples/ch21 |  |
| chapters/21-data.md:353 | anulm/training/train.py | 590 | 599 | same | p.add_argument("--sample-with-replacement", action="store_true", | p.add_argument("--sample-with-replacement", action="store_true", |
| chapters/21-data.md:354 | anulm/training/train.py | 692–693 |  | override | decided in repin-overrides.tsv |  |
| chapters/21-data.md:354 | anulm/training/train.py | 692–693 | 703–704 | same | train_data, val_data = interleaved_split(data) | train_data, val_data = interleaved_split(data) |
| chapters/21-data.md:384 | anulm/model.py | 365 |  | reattributed | '_attn_mask' is defined there |  |
| chapters/21-data.md:384 | anulm/model.py | 365 | 375 | def | '_attn_mask' | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/21-data.md:471 | tests/test_model.py | 1922 |  | reattributed | 'memmap_batches_match_in_memory_batches' is defined there |  |
| chapters/21-data.md:471 | tests/test_model.py | 1922 | 1920 | def | 'memmap_batches_match_in_memory_batches' | def memmap_batches_match_in_memory_batches(): |
| chapters/21-data.md:473 | tests/test_model.py | 931 |  | reattributed | 'window_sampler_covers_every_window_once_per_epoch' is defined there |  |
| chapters/21-data.md:474 | tests/test_model.py | 956 |  | reattributed | 'window_sampler_resumes_exactly' is defined there |  |
| chapters/21-data.md:475 | tests/test_model.py | 1944 |  | reattributed | 'interleaved_split_samples_the_whole_corpus' is defined there |  |
| chapters/21-data.md:475 | tests/test_model.py | 1944 | 1942 | def | 'interleaved_split_samples_the_whole_corpus' | def interleaved_split_samples_the_whole_corpus(): |
| chapters/21-data.md:526 | anulm/training/finetune.py | 103–114 |  | reattributed | 'flush' is in anulm/training/finetune.py, not experiments/build_v2.py |  |
| chapters/21-data.md:526 | experiments/build_v2.py | 121 | 121 | sweep | 'main' 86–142 ('flush' 103–114, budget check 121, 'me |  |
| chapters/21-data.md:528 | anulm/training/train.py | 104–160 | 107–163 | same | class WindowSampler: | class WindowSampler: |
| chapters/21-data.md:528 | anulm/training/train.py | 147–150 | 150–153 | same | @property | @property |
| chapters/21-data.md:528 | anulm/training/train.py | 163–173 | 166–176 | same | def load_memmap(meta_path: str): | def load_memmap(meta_path: str): |
| chapters/21-data.md:529 | anulm/training/train.py | 202–216 | 205–219 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/21-data.md:529 | anulm/training/train.py | 237–250 | 240–253 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/21-data.md:530 | anulm/training/train.py | 351–372 | 354–375 | same | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i |
| chapters/21-data.md:530 | anulm/training/train.py | 351–372 | 354–375 | sweep | 351–372, 'ma |  |
| chapters/21-data.md:530 | anulm/training/train.py | 590 | 599 | same | p.add_argument("--sample-with-replacement", action="store_true", | p.add_argument("--sample-with-replacement", action="store_true", |
| chapters/21-data.md:530 | anulm/training/train.py | 659–672 | 670–683 | same | memmapped = bool(args.data and args.data.endswith("meta.json")) | memmapped = bool(args.data and args.data.endswith("meta.json")) |
| chapters/21-data.md:530 | anulm/training/train.py | 689–695 | 700–706 | same | if memmapped or (split_encoded and val_data is not None): | if memmapped or (split_encoded and val_data is not None): |
| chapters/21-data.md:530 | anulm/training/train.py | 858–859 | 867–868 | same | ix = (torch.randint(len(train_data) - args.block_size - 1, (args.batch | ix = (torch.randint(len(train_data) - args.block_size - 1, (args.batch |
| chapters/21-data.md:531 | anulm/model.py | 365–381 | 375–391 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/21-data.md:534 | tests/test_model.py | 1922 | 1920 | same | def memmap_batches_match_in_memory_batches(): | def memmap_batches_match_in_memory_batches(): |
| chapters/21-data.md:535 | tests/test_model.py | 1944 | 1942 | same | def interleaved_split_samples_the_whole_corpus(): | def interleaved_split_samples_the_whole_corpus(): |
| chapters/22-the-training-loop.md:43 | anulm/training/train.py | 30–32 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/22-the-training-loop.md:51 | anulm/training/train.py | 723–741 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/22-the-training-loop.md:127 | anulm/training/train.py | 749–772 | 758–781 | same | raw_model = model | raw_model = model |
| chapters/22-the-training-loop.md:169 | anulm/training/train.py | 849–873 | 858–882 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/22-the-training-loop.md:227 | anulm/training/train.py | 874–880 | 883–889 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/22-the-training-loop.md:294 | anulm/training/train.py | 882–889 | 891–898 | same | if step % args.log_every == 0 or step == args.steps - 1: | if step % args.log_every == 0 or step == args.steps - 1: |
| chapters/22-the-training-loop.md:331 | anulm/training/train.py | 906–908 | 915–917 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/22-the-training-loop.md:363 | anulm/training/train.py | 375–391 | 378–394 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/22-the-training-loop.md:10 | anulm/training/train.py | 536–924 | 544–933 | same | def main(): | def main(): |
| chapters/22-the-training-loop.md:88 | anulm/model.py | 23 | 19 | same | from __future__ import annotations | from __future__ import annotations |
| chapters/22-the-training-loop.md:120 | tests/test_model.py | 736 |  | reattributed | quoted code is in tests/test_model.py, not book/code-examples/ch22_one |  |
| chapters/22-the-training-loop.md:152 | anulm/training/train.py | 757–759 | 766–768 | same | # Inductor generates C++ and needs a host compiler (MSVC 'cl' on Windo | # Inductor generates C++ and needs a host compiler (MSVC 'cl' on Windo |
| chapters/22-the-training-loop.md:251 | anulm/training/train.py | 852–878 | 861–887 | same | lr = lr_at(step, args, start_step) | lr = lr_at(step, args, start_step) |
| chapters/22-the-training-loop.md:284 | anulm/training/train.py | 869 |  | override | decided in repin-overrides.tsv |  |
| chapters/22-the-training-loop.md:284 | anulm/training/train.py | 869 | 878 | same | loss_acc += lm_loss.item() / args.grad_accum | loss_acc += lm_loss.item() / args.grad_accum |
| chapters/22-the-training-loop.md:328 | anulm/training/train.py | 891–905 |  | reattributed | 'main' is in anulm/training/train.py, not book/code-examples/ch22_read |  |
| chapters/22-the-training-loop.md:328 | anulm/training/train.py | 891–905 | 900–914 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/22-the-training-loop.md:346 | anulm/training/train.py | 796–797 | 805–806 | same | try: | try: |
| chapters/22-the-training-loop.md:348 | anulm/training/train.py | 798–802 | 807–811 | same | except UnsafeCheckpointError: | except UnsafeCheckpointError: |
| chapters/22-the-training-loop.md:462 | tests/test_model.py | 545 |  | reattributed | 'gradient_accumulation_equals_large_batch' is defined there |  |
| chapters/22-the-training-loop.md:517 | ? |  |  | other-pin | 723–741 @c4a39f8) |  |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 536–924 | 544–933 | same | def main(): | def main(): |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 646 | 657 | same | dtype = (torch.bfloat16 if args.device.startswith("cuda") | dtype = (torch.bfloat16 if args.device.startswith("cuda") |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 646 | 657 | sweep | - 'anulm/training/train.py': 'main' 536–924 (646, 712 |  |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 712–732 | ? | UNMAPPED | builder = {"30b": AnuLMConfig.nano_30b, |  |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 712–732 | 712–732 | sweep | ulm/training/train.py': 'main' 536–924 (646, 712–732 (x |  |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 736 | 745 | same | model = AnuLM(cfg).to(args.device) | model = AnuLM(cfg).to(args.device) |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 736 | 745 | sweep | rain.py': 'main' 536–924 (646, 712–732 (x, 736, 747 |  |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 747 | 756 | same | f"({100*active/total:.1f}%)  \|  device={args.device} dtype={dtype}") | f"({100*active/total:.1f}%)  \|  device={args.device} dtype={dtype}") |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 747 | 756 | sweep | py': 'main' 536–924 (646, 712–732 (x, 736, 747, 749 |  |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 749–772 | 758–781 | same | raw_model = model | raw_model = model |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 749–772 | 758–781 | sweep | 'main' 536–924 (646, 712–732 (x, 736, 747, 749–772, 849 |  |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 849–873 | 858–882 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 849–873 | 858–882 | sweep | 6–924 (646, 712–732 (x, 736, 747, 749–772, 849–873, 869 |  |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 869 | 878 | same | loss_acc += lm_loss.item() / args.grad_accum | loss_acc += lm_loss.item() / args.grad_accum |
| chapters/22-the-training-loop.md:517 | anulm/training/train.py | 869 | 878 | sweep | 6, 712–732 (x, 736, 747, 749–772, 849–873, 869, |  |
| chapters/22-the-training-loop.md:518 | anulm/training/train.py | 874–880 | 883–889 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/22-the-training-loop.md:518 | anulm/training/train.py | 874–880 | 883–889 | sweep | 874–880, 882 |  |
| chapters/22-the-training-loop.md:518 | anulm/training/train.py | 882–889 | 891–898 | same | if step % args.log_every == 0 or step == args.steps - 1: | if step % args.log_every == 0 or step == args.steps - 1: |
| chapters/22-the-training-loop.md:518 | anulm/training/train.py | 882–889 | 891–898 | sweep | 874–880, 882–889, 891 |  |
| chapters/22-the-training-loop.md:518 | anulm/training/train.py | 891–908 | 900–917 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/22-the-training-loop.md:518 | anulm/training/train.py | 891–908 | 900–917 | sweep | 874–880, 882–889, 891–908), 'l |  |
| chapters/22-the-training-loop.md:518 | anulm/training/train.py | 375–400 | 378–403 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/22-the-training-loop.md:518 | anulm/training/train.py | 416–420 | 419–423 | same | def set_lr_scale(self, scale: float): | def set_lr_scale(self, scale: float): |
| chapters/22-the-training-loop.md:519 | anulm/model.py | 23 | 19 | same | from __future__ import annotations | from __future__ import annotations |
| chapters/22-the-training-loop.md:519 | anulm/model.py | 23 | 19 | sweep | el.py': 'from __future__ import annotations' 23, |  |
| chapters/22-the-training-loop.md:520 | anulm/model.py | 1071–1101 | 1081–1111 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/23-precision-memory-speed.md:62 | anulm/training/train.py | 645–650 | 656–661 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/23-precision-memory-speed.md:134 | anulm/model.py | 234–239 | 244–249 | same | def forward(self, x): | def forward(self, x): |
| chapters/23-precision-memory-speed.md:157 | anulm/model.py | 615–622 | 625–632 | same | # fp32 router: top-k is discrete, and bf16 rounding would flip choices | # fp32 router: top-k is discrete, and bf16 rounding would flip choices |
| chapters/23-precision-memory-speed.md:263 | anulm/model.py | 901–909 | 911–919 | same | for i, layer in enumerate(self.layers): | for i, layer in enumerate(self.layers): |
| chapters/23-precision-memory-speed.md:319 | anulm/model.py | 942–956 | 952–966 | same | self.ce_chunk = rows | self.ce_chunk = rows |
| chapters/23-precision-memory-speed.md:364 | anulm/training/train.py | 227–234 | 230–237 | same | total = torch.cuda.get_device_properties(device).total_memory | total = torch.cuda.get_device_properties(device).total_memory |
| chapters/23-precision-memory-speed.md:377 | anulm/model.py | 1356–1364 | 1359–1367 | same | n = sum(p.numel() for p in model.parameters()) | n = sum(p.numel() for p in model.parameters()) |
| chapters/23-precision-memory-speed.md:145 | anulm/model.py | 924 | 934 | same | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ |
| chapters/23-precision-memory-speed.md:148 | anulm/model.py | 756 | 766 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/23-precision-memory-speed.md:148 | anulm/model.py | 784 | 794 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/23-precision-memory-speed.md:277 | anulm/model.py | 1064 | 1074 | def | 'enable_gradient_checkpointing' | def enable_gradient_checkpointing(self, enable: bool = True): |
| chapters/23-precision-memory-speed.md:347 | anulm/model.py | 919 | 929 | same | if self.ce_chunk and self.training: | if self.ce_chunk and self.training: |
| chapters/23-precision-memory-speed.md:392 | anulm/training/train.py | 880 | 889 | same | warn_if_spilling(args.device) | warn_if_spilling(args.device) |
| chapters/23-precision-memory-speed.md:398 | anulm/multimodal/omni_train.py | 194 | 306 | same | fit_memory(self.model) | fit_memory(self.model) |
| chapters/23-precision-memory-speed.md:444 | anulm/model.py | 379–380 | 389–390 | same | if window is not None: | if window is not None: |
| chapters/23-precision-memory-speed.md:527 | anulm/training/train.py | 219–234 | 222–237 | same | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: |
| chapters/23-precision-memory-speed.md:527 | anulm/training/train.py | 645–650 | 656–661 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/23-precision-memory-speed.md:527 | anulm/training/train.py | 701–702 | 712–713 | same | if args.ce_chunk: | if args.ce_chunk: |
| chapters/23-precision-memory-speed.md:527 | anulm/training/train.py | 880 | 889 | same | warn_if_spilling(args.device) | warn_if_spilling(args.device) |
| chapters/23-precision-memory-speed.md:528 | anulm/multimodal/omni_train.py | 194 | 306 | same | fit_memory(self.model) | fit_memory(self.model) |
| chapters/23-precision-memory-speed.md:528 | anulm/multimodal/omni_train.py | 194 | 306 | sweep | y' callers: 'anulm/multimodal/omni_train.py' 194, 'an |  |
| chapters/23-precision-memory-speed.md:528 | anulm/tasks/decide.py | 475 | 485 | same | fit_memory(d.model) | fit_memory(d.model) |
| chapters/23-precision-memory-speed.md:528 | anulm/tasks/decide.py | 475 | 485 | sweep | /omni_train.py' 194, 'anulm/tasks/decide.py' 475 and |  |
| chapters/23-precision-memory-speed.md:528 | anulm/tasks/decide.py | 553 | 563 | same | fit_memory(d.model) | fit_memory(d.model) |
| chapters/23-precision-memory-speed.md:528 | anulm/tasks/decide.py | 553 | 563 | sweep | ain.py' 194, 'anulm/tasks/decide.py' 475 and 553, |  |
| chapters/23-precision-memory-speed.md:529 | anulm/multimodal/speech_encoder.py | 231 | 231 | sweep | 'anulm/multimodal/speech_encoder.py' 231, 'an |  |
| chapters/23-precision-memory-speed.md:529 | anulm/multimodal/vision_encoder.py | 456 | 456 | sweep | y' 231, 'anulm/multimodal/vision_encoder.py' 456. |  |
| chapters/23-precision-memory-speed.md:530 | anulm/model.py | 234–239 | 244–249 | same | def forward(self, x): | def forward(self, x): |
| chapters/23-precision-memory-speed.md:530 | anulm/model.py | 611–622 | 621–632 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/23-precision-memory-speed.md:531 | anulm/model.py | 379–380 | 389–390 | same | if window is not None: | if window is not None: |
| chapters/23-precision-memory-speed.md:531 | anulm/model.py | 756 | 766 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/23-precision-memory-speed.md:531 | anulm/model.py | 784 | 794 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/23-precision-memory-speed.md:532 | anulm/model.py | 901–909 | 911–919 | same | for i, layer in enumerate(self.layers): | for i, layer in enumerate(self.layers): |
| chapters/23-precision-memory-speed.md:532 | anulm/model.py | 919 | 929 | same | if self.ce_chunk and self.training: | if self.ce_chunk and self.training: |
| chapters/23-precision-memory-speed.md:532 | anulm/model.py | 924 | 934 | same | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ |
| chapters/23-precision-memory-speed.md:532 | anulm/model.py | 930–942 | 940–952 | same | def enable_chunked_loss(self, rows: int = 2048): | def enable_chunked_loss(self, rows: int = 2048): |
| chapters/23-precision-memory-speed.md:533 | anulm/model.py | 944–953 | 954–963 | same | def _chunked_loss(self, x, targets): | def _chunked_loss(self, x, targets): |
| chapters/23-precision-memory-speed.md:533 | anulm/model.py | 955–956 | 965–966 | same | def _ce_sum(self, h, t): | def _ce_sum(self, h, t): |
| chapters/23-precision-memory-speed.md:534 | anulm/model.py | 1064–1068 | 1074–1078 | same | def enable_gradient_checkpointing(self, enable: bool = True): | def enable_gradient_checkpointing(self, enable: bool = True): |
| chapters/23-precision-memory-speed.md:534 | anulm/model.py | 1345–1364 | 1348–1367 | same | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i |
| chapters/23-precision-memory-speed.md:537 | tests/test_model.py | 1713 | 1711 | same | def chunked_loss_matches_the_full_loss_and_its_gradients(): | def chunked_loss_matches_the_full_loss_and_its_gradients(): |
| chapters/24-measuring-progress.md:37 | anulm/training/train.py | 338–348 | 341–351 | same | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, |
| chapters/24-measuring-progress.md:74 | anulm/training/train.py | 477–533 | 485–541 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/24-measuring-progress.md:187 | anulm/training/train.py | 351–372 | 354–375 | same | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i |
| chapters/24-measuring-progress.md:269 | anulm/training/train.py | 782–786 | 791–795 | same | eval_spec = ({"mode": "strided", "windows": args.eval_windows, | eval_spec = ({"mode": "strided", "windows": args.eval_windows, |
| chapters/24-measuring-progress.md:28 | anulm/training/train.py | 689–695 | 700–706 | same | if memmapped or (split_encoded and val_data is not None): | if memmapped or (split_encoded and val_data is not None): |
| chapters/24-measuring-progress.md:281 | anulm/training/train.py | 689–695 | 700–706 | same | if memmapped or (split_encoded and val_data is not None): | if memmapped or (split_encoded and val_data is not None): |
| chapters/24-measuring-progress.md:294 | anulm/training/train.py | 906–908 | 915–917 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/24-measuring-progress.md:449 | anulm/training/train.py | 338–348 | 341–351 | same | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, |
| chapters/24-measuring-progress.md:449 | anulm/training/train.py | 351–372 | 354–375 | same | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i |
| chapters/24-measuring-progress.md:450 | anulm/training/train.py | 477–533 | 485–541 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/24-measuring-progress.md:450 | anulm/training/train.py | 477–533 | 485–541 | sweep | 477–533, 'ma |  |
| chapters/24-measuring-progress.md:450 | anulm/training/train.py | 782–786 | 791–795 | same | eval_spec = ({"mode": "strided", "windows": args.eval_windows, | eval_spec = ({"mode": "strided", "windows": args.eval_windows, |
| chapters/24-measuring-progress.md:450 | anulm/training/train.py | 906–908 |  | override | decided in repin-overrides.tsv |  |
| chapters/24-measuring-progress.md:450 | anulm/training/train.py | 906–908 | 915–917 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/24-measuring-progress.md:450 | anulm/training/train.py | 906–908 | 915–917 | sweep | 477–533, 'main' 782–786 ('eval_spec') and 906–908 (bit |  |
| chapters/24-measuring-progress.md:452 | tests/test_model.py | 1113–1159 | 1113–1159 | sweep | 1113–1159, 'go |  |
| chapters/24-measuring-progress.md:452 | tests/test_model.py | 1630–1672 | 1628–1670 | same | def golden_items_are_well_formed_and_the_scorer_runs(): | def golden_items_are_well_formed_and_the_scorer_runs(): |
| chapters/24-measuring-progress.md:456 | anulm/training/train.py | 689–695 | 700–706 | same | if memmapped or (split_encoded and val_data is not None): | if memmapped or (split_encoded and val_data is not None): |
| chapters/25-checkpoints-and-resuming.md:28 | anulm/training/train.py | 891–905 | 900–914 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/25-checkpoints-and-resuming.md:85 | anulm/training/train.py | 32–46 | 35–49 | same | def atomic_save(obj, path) -> None: | def atomic_save(obj, path) -> None: |
| chapters/25-checkpoints-and-resuming.md:175 | anulm/training/train.py | 152–160 | 155–163 | same | def state(self) -> dict: | def state(self) -> dict: |
| chapters/25-checkpoints-and-resuming.md:215 | anulm/training/train.py | 788–840 | 797–849 | same | start_step, best_val = 0, float("inf") | start_step, best_val = 0, float("inf") |
| chapters/25-checkpoints-and-resuming.md:283 | anulm/training/train.py | 849–850 | 858–859 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/25-checkpoints-and-resuming.md:362 | anulm/training/train.py | 394–399 | 397–402 | same | # Extending a finished run (--resume with a larger --steps) lands mid- | # Extending a finished run (--resume with a larger --steps) lands mid- |
| chapters/25-checkpoints-and-resuming.md:397 | anulm/model.py | 1231–1261 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/25-checkpoints-and-resuming.md:501 | anulm/model.py | 1280–1316 | 1283–1319 | same | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di |
| chapters/25-checkpoints-and-resuming.md:172 | anulm/training/train.py | 645 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:172 | anulm/training/train.py | 645 | 656 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/25-checkpoints-and-resuming.md:173 | anulm/training/train.py | 859 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:173 | anulm/training/train.py | 859 | 868 | same | if args.sample_with_replacement else sampler.next()) | if args.sample_with_replacement else sampler.next()) |
| chapters/25-checkpoints-and-resuming.md:349 | anulm/training/train.py | 857–876 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:349 | anulm/training/train.py | 857–876 | 866–885 | same | for _ in range(args.grad_accum): | for _ in range(args.grad_accum): |
| chapters/25-checkpoints-and-resuming.md:376 | anulm/training/train.py | 380–391 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:376 | anulm/training/train.py | 380–391 | 383–394 | same | if step < args.warmup: | if step < args.warmup: |
| chapters/25-checkpoints-and-resuming.md:491 | book/code-examples/ch25_resume.py | 96 |  | reattributed | quoted code is in book/code-examples/ch25_resume.py, not tests/test_qu |  |
| chapters/25-checkpoints-and-resuming.md:492 | tests/test_quality_loading.py | 124 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:526 | anulm/model.py | 1261 | 1264 | same | _HUB_ID = r"[\w.-]+/[\w.-]+" | _HUB_ID = r"[\w.-]+/[\w.-]+" |
| chapters/25-checkpoints-and-resuming.md:533 | tests/test_model.py | 1057 |  | reattributed | 'an_exported_folder_loads_everywhere_a_pt_does' is defined there |  |
| chapters/25-checkpoints-and-resuming.md:612 | anulm/training/train.py | 32–46 | 35–49 | same | def atomic_save(obj, path) -> None: | def atomic_save(obj, path) -> None: |
| chapters/25-checkpoints-and-resuming.md:613 | anulm/training/train.py | 152–160 | 155–163 | same | def state(self) -> dict: | def state(self) -> dict: |
| chapters/25-checkpoints-and-resuming.md:613 | anulm/training/train.py | 152–160 | 155–163 | sweep | 152–160, 'lr |  |
| chapters/25-checkpoints-and-resuming.md:613 | anulm/training/train.py | 375–400 | 378–403 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/25-checkpoints-and-resuming.md:613 | anulm/training/train.py | 788–840 | 797–849 | same | start_step, best_val = 0, float("inf") | start_step, best_val = 0, float("inf") |
| chapters/25-checkpoints-and-resuming.md:613 | anulm/training/train.py | 849–850 | 858–859 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/25-checkpoints-and-resuming.md:613 | anulm/training/train.py | 849–850 | 858–859 | sweep | 0, 'lr_at' 375–400, 'main' 788–840 (resume), 849–850 ('-- |  |
| chapters/25-checkpoints-and-resuming.md:614 | anulm/training/train.py | 857–876 | 866–885 | same | for _ in range(args.grad_accum): | for _ in range(args.grad_accum): |
| chapters/25-checkpoints-and-resuming.md:614 | anulm/training/train.py | 857–876 | 866–885 | sweep | 857–876 (one |  |
| chapters/25-checkpoints-and-resuming.md:614 | anulm/training/train.py | 891–905 | 900–914 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/25-checkpoints-and-resuming.md:614 | anulm/training/train.py | 891–905 | 900–914 | sweep | 857–876 (one step), 891–905 (sav |  |
| chapters/25-checkpoints-and-resuming.md:615 | anulm/model.py | 1219 | 1222 | same | TRUST_PICKLE_ENV = "ANULM_TRUST_PICKLE" | TRUST_PICKLE_ENV = "ANULM_TRUST_PICKLE" |
| chapters/25-checkpoints-and-resuming.md:616 | anulm/model.py | 1222–1225 | 1225–1228 | same | class UnsafeCheckpointError(Exception): | class UnsafeCheckpointError(Exception): |
| chapters/25-checkpoints-and-resuming.md:616 | anulm/model.py | 1228–1258 | 1231–1261 | same | def safe_load(path, map_location="cpu", trust_pickle: Optional[bool] = | def safe_load(path, map_location="cpu", trust_pickle: Optional[bool] = |
| chapters/25-checkpoints-and-resuming.md:616 | anulm/model.py | 1261 | 1264 | same | _HUB_ID = r"[\w.-]+/[\w.-]+" | _HUB_ID = r"[\w.-]+/[\w.-]+" |
| chapters/25-checkpoints-and-resuming.md:617 | anulm/model.py |  |  | other-pin | 1231–1261 @c4a39f8 |  |
| chapters/25-checkpoints-and-resuming.md:617 | anulm/model.py | 1280–1342 | 1283–1345 | same | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di |
| chapters/25-checkpoints-and-resuming.md:621 | tests/test_model.py | 400–407 | 400–407 | sweep | 400–407, 'an |  |
| chapters/26-debugging-silent-failures.md:178 | tests/test_model.py | 1943–1956 | 1941–1954 | same | @test | @test |
| chapters/26-debugging-silent-failures.md:342 | anulm/model.py | 966–984 | 976–994 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/26-debugging-silent-failures.md:375 | anulm/model.py | 986–990 | 996–1000 | same | def train(self, mode: bool = True): | def train(self, mode: bool = True): |
| chapters/26-debugging-silent-failures.md:387 | tests/test_model.py | 1861–1871 | 1859–1869 | same | x = torch.randint(0, 259, (2, 8)) | x = torch.randint(0, 259, (2, 8)) |
| chapters/26-debugging-silent-failures.md:411 | anulm/training/train.py | 202–216 | 205–219 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/26-debugging-silent-failures.md:441 | anulm/model.py | 446–449 | 456–459 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/26-debugging-silent-failures.md:514 | tests/test_model.py | 2434–2455 | 2432–2453 | same | def main(): | def main(): |
| chapters/26-debugging-silent-failures.md:78 | anulm/model.py | 414 | 424 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/26-debugging-silent-failures.md:82 | anulm/model.py | 483–492 | 493–502 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/26-debugging-silent-failures.md:172 | tests/test_model.py | 1713 | 1711 | same | def chunked_loss_matches_the_full_loss_and_its_gradients(): | def chunked_loss_matches_the_full_loss_and_its_gradients(): |
| chapters/26-debugging-silent-failures.md:203 | tests/test_model.py | 1676 | 1674 | same | def qa_packing_masks_prompts_and_keeps_examples_whole(): | def qa_packing_masks_prompts_and_keeps_examples_whole(): |
| chapters/26-debugging-silent-failures.md:492 | tests/test_model.py | 1630 |  | reattributed | 'golden_items_are_well_formed_and_the_scorer_runs' is defined there |  |
| chapters/26-debugging-silent-failures.md:492 | tests/test_model.py | 1630 | 1628 | def | 'golden_items_are_well_formed_and_the_scorer_runs' | def golden_items_are_well_formed_and_the_scorer_runs(): |
| chapters/26-debugging-silent-failures.md:493 | anulm/eval/eval_code.py | 1544 | ? | UNMAPPED | 'bpe_files_load_and_foreign_ones_are_refused' (line 1544) |  |
| chapters/26-debugging-silent-failures.md:494 | tests/test_model.py | 658 |  | reattributed | 'incremental_forward_matches_full_forward' is defined there |  |
| chapters/26-debugging-silent-failures.md:541 | tests/test_model.py | 34 |  | override | decided in repin-overrides.tsv |  |
| chapters/26-debugging-silent-failures.md:659 | tests/test_model.py | 1–15 | 1–15 | sweep | - 'tests/test_model.py': module docstring 1–15; 'te |  |
| chapters/26-debugging-silent-failures.md:660 | tests/test_model.py | 831–832 | 831–832 | sweep | matches_sparse_under_bf16' 826–844 (the skip 831–832); |  |
| chapters/26-debugging-silent-failures.md:667 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/26-debugging-silent-failures.md:667 | anulm/model.py | 446–449 | 456–459 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/26-debugging-silent-failures.md:668 | anulm/model.py | 611–622 | 621–632 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/26-debugging-silent-failures.md:668 | anulm/model.py | 966–984 | 976–994 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/26-debugging-silent-failures.md:668 | anulm/model.py | 986–990 | 996–1000 | same | def train(self, mode: bool = True): | def train(self, mode: bool = True): |
| chapters/26-debugging-silent-failures.md:669 | anulm/training/train.py | 202–216 | 205–219 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/26-debugging-silent-failures.md:669 | anulm/training/train.py | 219–234 | 222–237 | same | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: |
| chapters/27-decoding-and-the-kv-cache.md:39 | anulm/model.py | 1140–1151 | 1150–1161 | same | B = idx.shape[0] | B = idx.shape[0] |
| chapters/27-decoding-and-the-kv-cache.md:67 | anulm/model.py | 1161–1173 | 1171–1183 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/27-decoding-and-the-kv-cache.md:119 | anulm/model.py | 873–875 | 883–885 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/27-decoding-and-the-kv-cache.md:127 | anulm/model.py | 480–487 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/27-decoding-and-the-kv-cache.md:185 | anulm/model.py | 897–899 | 907–909 | same | if start + S > self.cfg.block_size:      # an error, not an assert: py | if start + S > self.cfg.block_size:      # an error, not an assert: py |
| chapters/27-decoding-and-the-kv-cache.md:193 | anulm/model.py | 339–343 | 349–353 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| chapters/27-decoding-and-the-kv-cache.md:219 | anulm/model.py | 365–381 | 375–391 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/27-decoding-and-the-kv-cache.md:234 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/27-decoding-and-the-kv-cache.md:64 | anulm/model.py | 1164–1167 | 1174–1177 | same | logits = logits / temperature | logits = logits / temperature |
| chapters/27-decoding-and-the-kv-cache.md:91 | anulm/model.py | 1144–1151 |  | override | decided in repin-overrides.tsv |  |
| chapters/27-decoding-and-the-kv-cache.md:91 | anulm/model.py | 1144–1151 | 1154–1161 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/27-decoding-and-the-kv-cache.md:208 | tests/test_model.py | 733 |  | reattributed | quoted code is in tests/test_model.py, not anulm/model.py |  |
| chapters/27-decoding-and-the-kv-cache.md:494 | tests/test_model.py | 679 |  | reattributed | 'cache_holds_what_it_should' is defined there |  |
| chapters/27-decoding-and-the-kv-cache.md:556 | anulm/model.py | 339–343 | 349–353 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| chapters/27-decoding-and-the-kv-cache.md:556 | anulm/model.py | 365–381 | 375–391 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/27-decoding-and-the-kv-cache.md:556 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/27-decoding-and-the-kv-cache.md:557 | anulm/model.py | 470–477 | 480–487 | same | if cache is not None: | if cache is not None: |
| chapters/27-decoding-and-the-kv-cache.md:557 | anulm/model.py | 555–564 | 565–574 | same | if cache is not None: | if cache is not None: |
| chapters/27-decoding-and-the-kv-cache.md:558 | anulm/model.py | 873–875 | 883–885 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/27-decoding-and-the-kv-cache.md:558 | anulm/model.py | 897–899 | 907–909 | same | if start + S > self.cfg.block_size:      # an error, not an assert: py | if start + S > self.cfg.block_size:      # an error, not an assert: py |
| chapters/27-decoding-and-the-kv-cache.md:559 | anulm/model.py | 1140–1173 | 1150–1183 | same | B = idx.shape[0] | B = idx.shape[0] |
| chapters/28-choosing-the-next-token.md:19 | anulm/model.py | 1152–1169 | 1162–1179 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/28-choosing-the-next-token.md:219 | anulm/model.py | 1037–1054 | 1047–1064 | same | for _ in range(max_tokens): | for _ in range(max_tokens): |
| chapters/28-choosing-the-next-token.md:257 | anulm/model.py | 1170–1179 | 1180–1189 | same | if eos_id is not None: | if eos_id is not None: |
| chapters/28-choosing-the-next-token.md:90 | anulm/model.py | 1169 |  | override | decided in repin-overrides.tsv |  |
| chapters/28-choosing-the-next-token.md:90 | anulm/model.py | 1169 | 1179 | same | nxt = torch.multinomial(probs, num_samples=1)          # (B, 1) | nxt = torch.multinomial(probs, num_samples=1)          # (B, 1) |
| chapters/28-choosing-the-next-token.md:99 | anulm/model.py | 1161–1162 |  | override | decided in repin-overrides.tsv |  |
| chapters/28-choosing-the-next-token.md:99 | anulm/model.py | 1161–1162 | 1171–1172 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/28-choosing-the-next-token.md:105 | anulm/model.py | 1157–1168 |  | override | decided in repin-overrides.tsv |  |
| chapters/28-choosing-the-next-token.md:105 | anulm/model.py | 1157–1168 | 1167–1178 | same | if repetition_penalty != 1.0: | if repetition_penalty != 1.0: |
| chapters/28-choosing-the-next-token.md:139 | tests/test_model.py | 495 |  | reattributed | 'generation_is_deterministic_at_top_k_1' is defined there |  |
| chapters/28-choosing-the-next-token.md:155 | tests/test_release_fixes.py | 112 |  | reattributed | 'test_temperature_zero_reaches_the_model_unclamped' is in tests/test_r |  |
| chapters/28-choosing-the-next-token.md:274 | tests/test_model.py | 717 |  | reattributed | 'generation_stops_at_eos' is defined there |  |
| chapters/28-choosing-the-next-token.md:487 | anulm/model.py | 1015–1062 | 1025–1072 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/28-choosing-the-next-token.md:487 | anulm/model.py | 1037–1054 | 1047–1064 | same | for _ in range(max_tokens): | for _ in range(max_tokens): |
| chapters/28-choosing-the-next-token.md:487 | anulm/model.py | 1037–1054 | 1047–1064 | sweep | AnuLM.decode_from_embeds' 1015–1062 (excerpt 1037–1054), |  |
| chapters/28-choosing-the-next-token.md:488 | anulm/model.py | 1152–1179 | 1162–1189 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/28-choosing-the-next-token.md:490 | anulm/inference/api.py | 611–613 | 611–613 | sweep | 611–613, com |  |
| chapters/28-choosing-the-next-token.md:490 | anulm/inference/api.py | 649–650 | 649–650 | sweep | 611–613, completion defaults 649–650. |  |
| chapters/28-choosing-the-next-token.md:491 | anulm/inference/serve.py | 158–160 | 158–160 | sweep | ulm/inference/serve.py': the 'generate' call 158–160 (no |  |
| chapters/28-choosing-the-next-token.md:491 | course/checks/ch08.py | 224–229 |  | reattributed | 'generate' is in course/checks/ch08.py, not anulm/inference/serve.py |  |
| chapters/28-choosing-the-next-token.md:491 | anulm/inference/serve.py | 224–229 | 224–229 | sweep | rate' call 158–160 (no clamp), HTTP defaults 224–229. 'an |  |
| chapters/28-choosing-the-next-token.md:491 | anulm/inference/sample.py | 34–35 | 34–35 | sweep | faults 224–229. 'anulm/inference/sample.py': 34–35. |  |
| chapters/29-long-context.md:57 | anulm/model.py | 260–275 | 270–285 | same | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): |
| chapters/29-long-context.md:82 | anulm/model.py | 305–322 | 315–332 | same | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: |
| chapters/29-long-context.md:163 | anulm/model.py | 293–298 | 303–308 | same | if cfg.yarn: | if cfg.yarn: |
| chapters/29-long-context.md:174 | anulm/model.py | 446–449 | 456–459 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/29-long-context.md:291 | anulm/model.py | 812–814 | 822–824 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/29-long-context.md:80 | anulm/model.py | 109–110 | 104–105 | same | yarn_beta_fast: float = 32.0 | yarn_beta_fast: float = 32.0 |
| chapters/29-long-context.md:183 | anulm/model.py | 288–292 |  | override | decided in repin-overrides.tsv |  |
| chapters/29-long-context.md:183 | anulm/model.py | 288–292 | 298–302 | edited | # YaRN's temperature, split the way DeepSeek (and so the 105B) splits | # YaRN's temperature, split the way DeepSeek-V2 splits |
| chapters/29-long-context.md:204 | tests/test_model.py | 144 |  | reattributed | quoted code is in tests/test_model.py, not docs/RESULTS.md |  |
| chapters/29-long-context.md:215 | anulm/model.py | 897–898 | 907–908 | same | if start + S > self.cfg.block_size:      # an error, not an assert: py | if start + S > self.cfg.block_size:      # an error, not an assert: py |
| chapters/29-long-context.md:416 | anulm/model.py | 109–110 | 104–105 | same | yarn_beta_fast: float = 32.0 | yarn_beta_fast: float = 32.0 |
| chapters/29-long-context.md:416 | anulm/model.py | 109–110 | 104–105 | sweep | - 'anulm/model.py': configuration 109–110; '_y |  |
| chapters/29-long-context.md:417 | anulm/model.py | 260–275 | 270–285 | same | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): |
| chapters/29-long-context.md:417 | anulm/model.py | 288–298 | 298–308 | edited | # YaRN's temperature, split the way DeepSeek (and so the 105B) splits | # YaRN's temperature, split the way DeepSeek-V2 splits |
| chapters/29-long-context.md:418 | anulm/model.py | 305–322 | 315–332 | same | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: |
| chapters/29-long-context.md:418 | anulm/model.py | 365–381 | 375–391 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/29-long-context.md:419 | anulm/model.py | 446–449 | 456–459 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/29-long-context.md:419 | anulm/model.py | 812–814 | 822–824 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/30-instruction-tuning.md:142 | anulm/model.py | 922–928 | 932–938 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/30-instruction-tuning.md:500 | tests/test_model.py | 1690–1709 | 1688–1707 | same | for it, (p, a) in zip(items, pairs): | for it, (p, a) in zip(items, pairs): |
| chapters/30-instruction-tuning.md:338 | book/code-examples/ch30_measure_chat_split.py |  |  | other-pin | at @773b6a8, lines 109–112, unchanged: the split that built data/chat_ |  |
| chapters/30-instruction-tuning.md:454 | experiments/omni_one.py | 100 | 119 | same | "chat_heldout_loss": ("<=", 3.30),     # chat_v3: 3.146 by omni_eval.p | "chat_heldout_loss": ("<=", 3.30),     # chat_v3: 3.146 by omni_eval.p |
| chapters/30-instruction-tuning.md:498 | tests/test_model.py | 1676 |  | reattributed | 'qa_packing_masks_prompts_and_keeps_examples_whole' is defined there |  |
| chapters/30-instruction-tuning.md:498 | tests/test_model.py | 1676 | 1674 | def | 'qa_packing_masks_prompts_and_keeps_examples_whole' | def qa_packing_masks_prompts_and_keeps_examples_whole(): |
| chapters/30-instruction-tuning.md:588 | anulm/training/finetune.py | 307–320 | 307–320 | sweep | 'main' 222–423 (the held-out guard 307–320, the |  |
| chapters/30-instruction-tuning.md:588 | anulm/training/finetune.py | 360–375 | 360–375 | sweep | 22–423 (the held-out guard 307–320, the step 360–375, 'qa |  |
| chapters/30-instruction-tuning.md:589 | anulm/data/make_chat.py | 176 | 176 | sweep | chat_rows' 119–145, 'main' 148–184 (the call 176). |  |
| chapters/30-instruction-tuning.md:590 | anulm/multimodal/omni_eval.py | 192–195 |  | override | decided in repin-overrides.tsv |  |
| chapters/30-instruction-tuning.md:590 | anulm/multimodal/omni_eval.py | 192–195 | 192–195 | sweep | omni_eval.py': 'chat_loss' 115–131, its call 192–195; |  |
| chapters/30-instruction-tuning.md:591 | experiments/omni_one.py | 92–104 | 111–123 | same | GATES = { | GATES = { |
| chapters/30-instruction-tuning.md:591 | experiments/omni_one.py | 100 | 119 | same | "chat_heldout_loss": ("<=", 3.30),     # chat_v3: 3.146 by omni_eval.p | "chat_heldout_loss": ("<=", 3.30),     # chat_v3: 3.146 by omni_eval.p |
| chapters/30-instruction-tuning.md:593 | anulm/model.py | 922–928 | 932–938 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/30-instruction-tuning.md:593 | anulm/model.py | 926–927 | 936–937 | same | if aux_total is not None: | if aux_total is not None: |
| chapters/30-instruction-tuning.md:593 | anulm/model.py | 926–927 | 936–937 | sweep | 922–928 ('ignore_index=-1'; the regulariser 926–927). |  |
| chapters/30-instruction-tuning.md:594 | tests/test_model.py | 1676 | 1674 | same | def qa_packing_masks_prompts_and_keeps_examples_whole(): | def qa_packing_masks_prompts_and_keeps_examples_whole(): |
| chapters/31-lora.md:371 | tests/test_model.py | 1181–1182 |  | override | decided in repin-overrides.tsv |  |
| chapters/31-lora.md:429 | anulm/tasks/decide.py | 249–288 | 259–298 | same | class LoRA(nn.Module): | class LoRA(nn.Module): |
| chapters/31-lora.md:515 | anulm/training/finetune.py | 393–400 | 393–400 | sweep | - 'anulm/training/finetune.py': adapter save 393–400; 'up |  |
| chapters/31-lora.md:515 | anulm/training/finetune.py | 375 | 375 | sweep | er save 393–400; 'update_expert_biases' call 375. |  |
| chapters/31-lora.md:516 | anulm/model.py | 1071–1101 | 1081–1111 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/31-lora.md:517 | anulm/tasks/decide.py | 249–264 | 259–274 | same | class LoRA(nn.Module): | class LoRA(nn.Module): |
| chapters/31-lora.md:517 | anulm/tasks/decide.py | 267–288 | 277–298 | same | class LoRAQV(nn.Module): | class LoRAQV(nn.Module): |
| chapters/31-lora.md:519 | tests/test_model.py | 1207–1218 | 1207–1218 | sweep | (the drift 1207–1218, the |  |
| chapters/31-lora.md:519 | tests/test_model.py | 1247–1258 | 1247–1258 | sweep | (the drift 1207–1218, the naive merge 1247–1258), 'l |  |
| chapters/32-classification-and-calibration.md:21 | anulm/tasks/decide.py | 89–101 | 99–111 | same | def encode(tok, text: str, options: list[str], max_len: int = 1024): | def encode(tok, text: str, options: list[str], max_len: int = 1024): |
| chapters/32-classification-and-calibration.md:66 | anulm/tasks/decide.py | 240–246 | 250–256 | same | def forward(self, idx, pos, mask=None): | def forward(self, idx, pos, mask=None): |
| chapters/32-classification-and-calibration.md:85 | anulm/tasks/decide.py | 291–306 | 301–316 | same | def batchify(examples, tok, pad, with_mask=False): | def batchify(examples, tok, pad, with_mask=False): |
| chapters/32-classification-and-calibration.md:104 | anulm/tasks/decide.py | 200–218 | 210–228 | same | def _hidden(self, idx): | def _hidden(self, idx): |
| chapters/32-classification-and-calibration.md:185 | anulm/model.py | 479–481 | 489–491 | same | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio |
| chapters/32-classification-and-calibration.md:210 | anulm/tasks/decide.py | 368–374 | 378–384 | same | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: |
| chapters/32-classification-and-calibration.md:231 | anulm/tasks/decide.py | 377–397 | 387–407 | same | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d |
| chapters/32-classification-and-calibration.md:335 | tests/test_model.py | 1885–1898 | 1883–1896 | same | ids, pos = D.encode(tok, "where is my card", opts) | ids, pos = D.encode(tok, "where is my card", opts) |
| chapters/32-classification-and-calibration.md:40 | anulm/tasks/decide.py | 83 | 93 | same | def readable(label: str) -> str: | def readable(label: str) -> str: |
| chapters/32-classification-and-calibration.md:44 | anulm/tasks/decide.py | 696 | 706 | def | 'block_size' | ids, pos = encode(self.tok, message, options, max_len=getattr(cfg, "bl |
| chapters/32-classification-and-calibration.md:80 | anulm/tasks/decide.py | 120 |  | override | decided in repin-overrides.tsv |  |
| chapters/32-classification-and-calibration.md:80 | anulm/tasks/decide.py | 120 | 130 | same | self.head = nn.Linear(hid, 1) | self.head = nn.Linear(hid, 1) |
| chapters/32-classification-and-calibration.md:82 | anulm/tasks/decide.py | 596 |  | override | decided in repin-overrides.tsv |  |
| chapters/32-classification-and-calibration.md:82 | anulm/tasks/decide.py | 596 | 606 | same | loss = F.cross_entropy(logits, y.to(a.device)) | loss = F.cross_entropy(logits, y.to(a.device)) |
| chapters/32-classification-and-calibration.md:193 | anulm/bidirectional.py | 56–105 |  | reattributed | 'adapt' is in anulm/bidirectional.py, not anulm/model.py |  |
| chapters/32-classification-and-calibration.md:224 | anulm/tasks/decide.py | 611 | 621 | same | t = fit_temperature(lv, yv) | t = fit_temperature(lv, yv) |
| chapters/32-classification-and-calibration.md:225 | anulm/model.py | 697 |  | reattributed | quoted code is in anulm/model.py, not anulm/tasks/decide.py |  |
| chapters/32-classification-and-calibration.md:225 | anulm/model.py | 697 | 707 | same | """Returns (y, aux): the routed + shared output, and the router's | """Returns (y, aux): the routed + shared output, and the router's |
| chapters/32-classification-and-calibration.md:354 | tests/test_model.py | 1875 | 1873 | def | 'decide_scores_each_option_at_its_last_token_and_calibrates' | def decide_scores_each_option_at_its_last_token_and_calibrates(): |
| chapters/32-classification-and-calibration.md:417 | anulm/tasks/decide.py | 83 | 93 | same | def readable(label: str) -> str: | def readable(label: str) -> str: |
| chapters/32-classification-and-calibration.md:417 | anulm/tasks/decide.py | 89–101 | 99–111 | same | def encode(tok, text: str, options: list[str], max_len: int = 1024): | def encode(tok, text: str, options: list[str], max_len: int = 1024): |
| chapters/32-classification-and-calibration.md:417 | anulm/tasks/decide.py | 105–246 | 115–256 | same | class Decider(nn.Module): | class Decider(nn.Module): |
| chapters/32-classification-and-calibration.md:417 | anulm/tasks/decide.py | 120 | 130 | same | self.head = nn.Linear(hid, 1) | self.head = nn.Linear(hid, 1) |
| chapters/32-classification-and-calibration.md:418 | anulm/tasks/decide.py | 200–218 | 210–228 | same | def _hidden(self, idx): | def _hidden(self, idx): |
| chapters/32-classification-and-calibration.md:418 | anulm/tasks/decide.py | 240–246 | 250–256 | same | def forward(self, idx, pos, mask=None): | def forward(self, idx, pos, mask=None): |
| chapters/32-classification-and-calibration.md:418 | anulm/tasks/decide.py | 249–264 | 259–274 | same | class LoRA(nn.Module): | class LoRA(nn.Module): |
| chapters/32-classification-and-calibration.md:418 | anulm/tasks/decide.py | 291 | 301 | same | def batchify(examples, tok, pad, with_mask=False): | def batchify(examples, tok, pad, with_mask=False): |
| chapters/32-classification-and-calibration.md:419 | anulm/tasks/decide.py | 309 | 319 | same | def make_examples(rows, tok, labs, rng=None): | def make_examples(rows, tok, labs, rng=None): |
| chapters/32-classification-and-calibration.md:419 | anulm/tasks/decide.py | 368–374 | 378–384 | same | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: |
| chapters/32-classification-and-calibration.md:419 | anulm/tasks/decide.py | 377–397 | 387–407 | same | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d |
| chapters/32-classification-and-calibration.md:420 | anulm/tasks/decide.py | 545–616 | 555–626 | same | def train(a): | def train(a): |
| chapters/32-classification-and-calibration.md:420 | anulm/tasks/decide.py | 545–616 | 555–626 | sweep | 545–616 (los |  |
| chapters/32-classification-and-calibration.md:420 | anulm/tasks/decide.py | 596 | 606 | same | loss = F.cross_entropy(logits, y.to(a.device)) | loss = F.cross_entropy(logits, y.to(a.device)) |
| chapters/32-classification-and-calibration.md:420 | anulm/tasks/decide.py | 596 | 606 | sweep | 545–616 (loss 596, tem |  |
| chapters/32-classification-and-calibration.md:420 | anulm/tasks/decide.py | 611 | 621 | same | t = fit_temperature(lv, yv) | t = fit_temperature(lv, yv) |
| chapters/32-classification-and-calibration.md:420 | anulm/tasks/decide.py | 611 | 621 | sweep | 545–616 (loss 596, temperature 611), 'D |  |
| chapters/32-classification-and-calibration.md:420 | anulm/tasks/decide.py | 691–704 | 701–714 | same | def __call__(self, message: str, options: list[str] \| None = None, top | def __call__(self, message: str, options: list[str] \| None = None, top |
| chapters/32-classification-and-calibration.md:423 | anulm/model.py | 479–481 | 489–491 | same | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio |
| chapters/32-classification-and-calibration.md:424 | tests/test_model.py | 1875 | 1873 | same | def decide_scores_each_option_at_its_last_token_and_calibrates(): | def decide_scores_each_option_at_its_last_token_and_calibrates(): |
| chapters/33-retrieval-and-readers.md:28 | anulm/inference/rag.py | 42–60 | 44–62 | same | BUCKETS = 1 << 24 | BUCKETS = 1 << 24 |
| chapters/33-retrieval-and-readers.md:66 | anulm/inference/rag.py | 136–145 | 151–160 | same | N, avgdl = len(pairs), float(lens.mean()) | N, avgdl = len(pairs), float(lens.mean()) |
| chapters/33-retrieval-and-readers.md:94 | anulm/inference/rag.py | 288–308 | 417–437 | same | scores = np.zeros(self.n, dtype=np.float32) | scores = np.zeros(self.n, dtype=np.float32) |
| chapters/33-retrieval-and-readers.md:159 | anulm/inference/rag.py | 195–205 | 210–220 | same | def embed(self, texts: list[str], max_len: int = 256): | def embed(self, texts: list[str], max_len: int = 256): |
| chapters/33-retrieval-and-readers.md:191 | anulm/inference/rag.py | 267–281 | 395–410 | same | mode = mode or self.mode | mode = mode or self.mode |
| chapters/33-retrieval-and-readers.md:191 | anulm/inference/rag.py | 395–410 |  | EXCERPT | self.dense = Dense(self.device) | quoted line not in the new range |
| chapters/33-retrieval-and-readers.md:191 | anulm/inference/rag.py | 395–410 |  | EXCERPT | di, dv = self.dense.search(query, 100 if mode == "hybrid" else k) | quoted line not in the new range |
| chapters/33-retrieval-and-readers.md:191 | anulm/inference/rag.py | 274 | 403 | edited-quote | return [{**self.chunks[i], "score": v} for i, v in zip(di, dv)] | return rerank(query, [{**self.chunks[i], "score": v, "_i": i} for i, v |
| chapters/33-retrieval-and-readers.md:260 | anulm/inference/rag.py | 344–351 | 473–480 | same | def prompt(self, question: str, passages: list[str]) -> tuple[str, str | def prompt(self, question: str, passages: list[str]) -> tuple[str, str |
| chapters/33-retrieval-and-readers.md:427 | anulm/inference/rag.py | 378–400 | 507–529 | same | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") |
| chapters/33-retrieval-and-readers.md:49 | anulm/inference/rag.py | 91 | 93 | same | ids = term_ids(title + " " + chunk) | ids = term_ids(title + " " + chunk) |
| chapters/33-retrieval-and-readers.md:181 | anulm/inference/rag.py | 213 |  | override | decided in repin-overrides.tsv |  |
| chapters/33-retrieval-and-readers.md:181 | anulm/inference/rag.py | 213 | 230 | same | out = torch.empty(len(chunks), self.m.config.hidden_size, dtype=torch. | out = torch.empty(len(chunks), self.m.config.hidden_size, dtype=torch. |
| chapters/33-retrieval-and-readers.md:186 | anulm/inference/rag.py | 342 |  | reattributed | 'RagQA' is in anulm/inference/rag.py, not None |  |
| chapters/33-retrieval-and-readers.md:186 | anulm/inference/rag.py | 342 | 471 | same | self.r = retriever or Retriever(device=device) | self.r = retriever or Retriever(device=device) |
| chapters/33-retrieval-and-readers.md:243 | anulm/inference/rag.py | 256 | 378 | same | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" |
| chapters/33-retrieval-and-readers.md:247 | anulm/inference/rag.py | 253–254 |  | override | decided in repin-overrides.tsv |  |
| chapters/33-retrieval-and-readers.md:247 | anulm/inference/rag.py | 253–254 | 375–376 | same | # Dense when its index exists: on the 20 demo questions it put the ans | # Dense when its index exists: on the 20 demo questions it put the ans |
| chapters/33-retrieval-and-readers.md:283 | anulm/inference/rag.py | 354–364 |  | override | decided in repin-overrides.tsv |  |
| chapters/33-retrieval-and-readers.md:283 | anulm/inference/rag.py | 354–364 | 483–493 | same | def read(self, prompt: str, max_tokens: int = 32) -> str: | def read(self, prompt: str, max_tokens: int = 32) -> str: |
| chapters/33-retrieval-and-readers.md:535 | anulm/inference/rag.py | 42–43 | 44–45 | same | BUCKETS = 1 << 24 | BUCKETS = 1 << 24 |
| chapters/33-retrieval-and-readers.md:535 | anulm/inference/rag.py | 42–43 | 44–45 | sweep | - 'anulm/inference/rag.py': constants 42–43, 'wo |  |
| chapters/33-retrieval-and-readers.md:535 | anulm/inference/rag.py | 51 | 53 | same | def words(text: str) -> list[str]: | def words(text: str) -> list[str]: |
| chapters/33-retrieval-and-readers.md:535 | anulm/inference/rag.py | 57 | 59 | same | def term_ids(text: str) -> np.ndarray: | def term_ids(text: str) -> np.ndarray: |
| chapters/33-retrieval-and-readers.md:535 | anulm/inference/rag.py | 91 |  | override | decided in repin-overrides.tsv |  |
| chapters/33-retrieval-and-readers.md:535 | anulm/inference/rag.py | 91 | 93 | same | ids = term_ids(title + " " + chunk) | ids = term_ids(title + " " + chunk) |
| chapters/33-retrieval-and-readers.md:535 | anulm/inference/rag.py | 91 | 93 | sweep | rds' 51, 'term_ids' 57, title + text indexed 91 (in |  |
| chapters/33-retrieval-and-readers.md:536 | anulm/inference/rag.py | 378–400 | 507–529 | same | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") |
| chapters/33-retrieval-and-readers.md:536 | anulm/inference/rag.py | 103–168 | ? | UNMAPPED | def build(langs=("simple", "hi"), workers: int = 12) -> None: |  |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 136–145 | 151–160 | same | N, avgdl = len(pairs), float(lens.mean()) | N, avgdl = len(pairs), float(lens.mean()) |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 136–145 | 151–160 | sweep | 136–145), 'D |  |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 174–230 | 189–247 | same | class Dense: | class Dense: |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 184–193 | 199–208 | edited | def __init__(self, device: str \| None = None): | def __init__(self, device: str \| None = None, index: Path = RAG / "den |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 184–193 | 199–208 | sweep | 136–145), 'Dense' 174–230 (device 184–193, 'em |  |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 195–205 | 210–220 | same | def embed(self, texts: list[str], max_len: int = 256): | def embed(self, texts: list[str], max_len: int = 256): |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 233–308 | 354–437 | same | class Retriever: | class Retriever: |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 241 | 362 | edited | self.device = device | self.device, self.root = device, Path(root) |
| chapters/33-retrieval-and-readers.md:537 | anulm/inference/rag.py | 241 | 362 | sweep | embed' 195–205), 'Retriever' 233–308 (device 241, def |  |
| chapters/33-retrieval-and-readers.md:538 | anulm/inference/rag.py | 256 | 378 | same | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" |
| chapters/33-retrieval-and-readers.md:538 | anulm/inference/rag.py | 256 | 378 | sweep | mode 256, 'se |  |
| chapters/33-retrieval-and-readers.md:538 | anulm/inference/rag.py | 261–281 | 388–410 | same | def search(self, query: str, k: int = 3, lang: str \| None = None, | def search(self, query: str, k: int = 3, lang: str \| None = None, |
| chapters/33-retrieval-and-readers.md:538 | anulm/inference/rag.py | 283–308 | 412–437 | same | def bm25(self, query: str, k: int = 3, lang: str \| None = None, | def bm25(self, query: str, k: int = 3, lang: str \| None = None, |
| chapters/33-retrieval-and-readers.md:538 | anulm/inference/rag.py | 324–375 | 453–504 | same | class RagQA: | class RagQA: |
| chapters/33-retrieval-and-readers.md:540 | anulm/tasks/rc_pointer.py | 185–190 | 185–190 | sweep | 'train' 155–229 (loss 185–190). |  |
| chapters/34-preference-optimisation-and-rl.md:65 | anulm/model.py | 915–921 | 925–931 | same | if targets is None: | if targets is None: |
| chapters/34-preference-optimisation-and-rl.md:323 | anulm/model.py | 1161–1169 | 1171–1179 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/34-preference-optimisation-and-rl.md:408 | anulm/model.py | 915–921 | 925–931 | same | if targets is None: | if targets is None: |
| chapters/34-preference-optimisation-and-rl.md:408 | anulm/model.py | 1161–1169 | 1171–1179 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/35-wider-vocabulary.md:231 | anulm/model.py | 887–896 | 897–906 | same | # 'embeds' bypasses the embedding table: (B, S, D) straight in. Audio | # 'embeds' bypasses the embedding table: (B, S, D) straight in. Audio |
| chapters/35-wider-vocabulary.md:337 | anulm/model.py | 1152–1156 | 1162–1166 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/35-wider-vocabulary.md:390 | anulm/model.py | 1033–1057 | 1043–1067 | same | dev = prefix.device | dev = prefix.device |
| chapters/35-wider-vocabulary.md:247 | anulm/model.py | 896 |  | override | decided in repin-overrides.tsv |  |
| chapters/35-wider-vocabulary.md:247 | anulm/model.py | 896 | 906 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| chapters/35-wider-vocabulary.md:448 | anulm/multimodal/speech_encoder.py | 105–112 |  | reattributed | quoted code is in anulm/multimodal/speech_encoder.py, not anulm/multim |  |
| chapters/35-wider-vocabulary.md:455 | tests/test_model.py | 1742 | 1740 | def | 'cached_embeds_decoding_matches_prefix_replay' | def cached_embeds_decoding_matches_prefix_replay(): |
| chapters/35-wider-vocabulary.md:517 | anulm/model.py | 877–928 | 887–938 | same | def forward(self, idx, targets=None, cache: Optional[list] = None, sta | def forward(self, idx, targets=None, cache: Optional[list] = None, sta |
| chapters/35-wider-vocabulary.md:517 | anulm/model.py | 1015–1062 | 1025–1072 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/35-wider-vocabulary.md:518 | anulm/model.py | 1115–1180 | 1125–1190 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/35-wider-vocabulary.md:520 | anulm/multimodal/speech_encoder.py | 287 | 287 | sweep | 287, 'tr |  |
| chapters/35-wider-vocabulary.md:522 | tests/test_model.py | 1742 | 1740 | same | def cached_embeds_decoding_matches_prefix_replay(): | def cached_embeds_decoding_matches_prefix_replay(): |
| chapters/36-frozen-encoders.md:356 | anulm/multimodal/omni_train.py | 504 | 647 | same | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", |
| chapters/36-frozen-encoders.md:357 | anulm/multimodal/omni_train.py | 249–263 | 371–385 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/36-frozen-encoders.md:445 | anulm/multimodal/speech_encoder.py | 197–203 |  | reattributed | 'batches' is in anulm/multimodal/speech_encoder.py, not None |  |
| chapters/36-frozen-encoders.md:475 | anulm/multimodal/omni_train.py | 99 | 107 | same | return whisper_window_rows(load_manifest(task.path), task.name) | return whisper_window_rows(load_manifest(task.path), task.name) |
| chapters/36-frozen-encoders.md:550 | anulm/multimodal/vision_encoder.py | 485–488 | 485–488 | sweep | (optimiser 485–488, def |  |
| chapters/36-frozen-encoders.md:550 | anulm/multimodal/vision_encoder.py | 653–654 | 653–654 | sweep | (optimiser 485–488, defaults 653–654). |  |
| chapters/36-frozen-encoders.md:552 | anulm/multimodal/speech_encoder.py | 249 | 249 | sweep | ' 171–182, 'batches' 185–213, 'train''s call 249. |  |
| chapters/36-frozen-encoders.md:553 | anulm/multimodal/omni_train.py | 99 | 107 | same | return whisper_window_rows(load_manifest(task.path), task.name) | return whisper_window_rows(load_manifest(task.path), task.name) |
| chapters/37-one-model-many-tasks.md:33 | anulm/multimodal/omni_train.py | 104–174 | ? | UNMAPPED | def pad(built, device): |  |
| chapters/37-one-model-many-tasks.md:33 | anulm/multimodal/omni_train.py | 104 | ? | EXCERPT | header range unresolved: text_batches |  |
| chapters/37-one-model-many-tasks.md:83 | anulm/multimodal/omni_train.py | 69–82 | 77–90 | same | def __init__(self, spec: str): | def __init__(self, spec: str): |
| chapters/37-one-model-many-tasks.md:115 | anulm/multimodal/omni_train.py | 216–235 | 338–357 | same | self.caches = {} | self.caches = {} |
| chapters/37-one-model-many-tasks.md:159 | anulm/multimodal/omni_train.py | 123–136 | 131–144 | same | def split_rows(rows, kind, held_fraction=0.02): | def split_rows(rows, kind, held_fraction=0.02): |
| chapters/37-one-model-many-tasks.md:181 | anulm/multimodal/omni_train.py | 148–159 | 156–167 | same | def disjoint_held(training, held, kind, name): | def disjoint_held(training, held, kind, name): |
| chapters/37-one-model-many-tasks.md:233 | anulm/multimodal/omni_train.py | 423–452 | 555–591 | same | w = np.array([t.weight for t in om.tasks], dtype=float) | w = np.array([t.weight for t in om.tasks], dtype=float) |
| chapters/37-one-model-many-tasks.md:233 | anulm/multimodal/omni_train.py | 555–591 |  | EXCERPT | eb, tb = om.next_batch(t) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:233 | anulm/multimodal/omni_train.py | 555–591 |  | EXCERPT | _, loss = model(None, tb, embeds=eb) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:233 | anulm/multimodal/omni_train.py | 555–591 |  | EXCERPT | run_loss[t.name].append(loss.item()) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:357 | anulm/multimodal/omni_train.py | 460–467 | ? | UNMAPPED | if step % args.eval_every == 0 or step == args.steps or step == args.s |  |
| chapters/37-one-model-many-tasks.md:357 | anulm/multimodal/omni_train.py | 460 | ? | EXCERPT | header range unresolved: train |  |
| chapters/37-one-model-many-tasks.md:450 | anulm/multimodal/omni_train.py | 336–350 | 459–473 | same | def state(self, step, score, losses) -> dict: | def state(self, step, score, losses) -> dict: |
| chapters/37-one-model-many-tasks.md:475 | anulm/multimodal/omni.py | 47–57 | 47–62 | same | def ear(self): | def ear(self): |
| chapters/37-one-model-many-tasks.md:475 | anulm/multimodal/omni.py | 47–62 |  | EXCERPT | enc = WhisperEncoder(self.meta.get("whisper", "small"), self.device) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:501 | experiments/omni_one.py | 384–392 | 403–411 | same | def gates(res) -> dict: | def gates(res) -> dict: |
| chapters/37-one-model-many-tasks.md:543 | anulm/multimodal/omni_train.py | 322–334 | 445–457 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/37-one-model-many-tasks.md:543 | anulm/multimodal/omni_train.py | 326 | 449 | edited-quote | for eb, tb in self.batches(t, t.held[: self.args.eval_n], 1, shuffle=F | for eb, tb, *_ in self.batches(t, t.held[: self.args.eval_n], 1, shuff |
| chapters/37-one-model-many-tasks.md:603 | experiments/omni_one.py | 65–87 | 65–100 | same | # Each optimizer step sums ACCUM batches. One batch per step (the firs | # Each optimizer step sums ACCUM batches. One batch per step (the firs |
| chapters/37-one-model-many-tasks.md:603 | experiments/omni_one.py | 65–100 |  | EXCERPT | STEPS = 12_500                  # x ACCUM = the first run's 100,000 ba | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:603 | experiments/omni_one.py | 65–100 |  | EXCERPT | ("chat", "text", "data/chat_train.jsonl", 0.25, 8), | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:603 | experiments/omni_one.py | 65–100 |  | EXCERPT | ("intent", "text", "data/omni/intent_train.jsonl", 0.06, 8), | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:603 | experiments/omni_one.py | 84 | 97 | edited-quote | ("asr_hi", "asr", "data/omni/hindi_train", 0.32, 4), | ("asr_hi", "asr", "data/omni/hindi_train", 0.26, 4), |
| chapters/37-one-model-many-tasks.md:603 | experiments/omni_one.py | 85 | 98 | edited-quote | ("caption", "image", "data/omni/caption_train.jsonl", 0.17, 8), | ("caption", "image", "data/omni/caption_train.jsonl", 0.13, 8),   # no |
| chapters/37-one-model-many-tasks.md:644 | experiments/omni_one.py | 471–480 | 493–502 | same | args = ["-m", "anulm.multimodal.omni_train", "--ckpt", "ckpt_chat_mm.p | args = ["-m", "anulm.multimodal.omni_train", "--ckpt", "ckpt_chat_mm.p |
| chapters/37-one-model-many-tasks.md:644 | experiments/omni_one.py | 474 | 496 | edited-quote | "--grad-ckpt"]     # unattended: an out-of-memory at 3 a.m. would stop | "--grad-ckpt", "--max-ratio", MAX_RATIO]     # unattended: an out-of-m |
| chapters/37-one-model-many-tasks.md:62 | anulm/multimodal/omni_train.py | 85–96 | 93–104 | same | def load_rows(task: Task, tok, max_len: int) -> list[dict]: | def load_rows(task: Task, tok, max_len: int) -> list[dict]: |
| chapters/37-one-model-many-tasks.md:102 | anulm/multimodal/omni_train.py | 64 | 72 | def | 'image' | KINDS = ("text", "asr", "image") |
| chapters/37-one-model-many-tasks.md:103 | anulm/multimodal/omni_train.py | 65 | 73 | same | DEFAULT_BATCH = {"text": 8, "asr": 4, "image": 8} | DEFAULT_BATCH = {"text": 8, "asr": 4, "image": 8} |
| chapters/37-one-model-many-tasks.md:142 | anulm/multimodal/omni_train.py | 85–101 | 93–109 | same | def load_rows(task: Task, tok, max_len: int) -> list[dict]: | def load_rows(task: Task, tok, max_len: int) -> list[dict]: |
| chapters/37-one-model-many-tasks.md:148 | anulm/multimodal/omni_train.py | 139–144 | 147–152 | same | def check_split(training, held, kind, name): | def check_split(training, held, kind, name): |
| chapters/37-one-model-many-tasks.md:150 | anulm/multimodal/omni_train.py | 265–296 | 387–418 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/37-one-model-many-tasks.md:174 | anulm/multimodal/omni_train.py | 114–120 | 122–128 | same | def _row_group(row, kind): | def _row_group(row, kind): |
| chapters/37-one-model-many-tasks.md:200 | anulm/multimodal/omni_train.py | 33–37 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:204 | experiments/omni_one.py | 77–87 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:204 | experiments/omni_one.py | 77–87 | 85–100 | same | TASKS = [ | TASKS = [ |
| chapters/37-one-model-many-tasks.md:274 | anulm/multimodal/omni_train.py | 309–320 | 432–443 | same | def next_batch(self, t: Task): | def next_batch(self, t: Task): |
| chapters/37-one-model-many-tasks.md:288 | anulm/multimodal/omni_train.py | 507 | 650 | same | p.add_argument("--accum", type=int, default=1, help="batches summed in | p.add_argument("--accum", type=int, default=1, help="batches summed in |
| chapters/37-one-model-many-tasks.md:288 | anulm/multimodal/omni_train.py | 519–520 | 669–670 | same | if args.accum < 1: | if args.accum < 1: |
| chapters/37-one-model-many-tasks.md:291 | anulm/multimodal/omni_train.py | 372–376 | 498–502 | same | groups = [{"params": model.parameters(), "lr": args.lr}] | groups = [{"params": model.parameters(), "lr": args.lr}] |
| chapters/37-one-model-many-tasks.md:291 | anulm/multimodal/omni_train.py | 496–497 | 639–640 | same | p.add_argument("--lr", type=float, default=5e-5) | p.add_argument("--lr", type=float, default=5e-5) |
| chapters/37-one-model-many-tasks.md:301 | anulm/multimodal/omni_train.py | 365–404 | 491–535 | same | # Fresh projectors (vision always, audio without --audio-proj-from) ar | # Fresh projectors (vision always, audio without --audio-proj-from) ar |
| chapters/37-one-model-many-tasks.md:328 | anulm/multimodal/omni_train.py | 353–359 | ? | UNMAPPED | def recipe_of(args, tasks) -> dict: |  |
| chapters/37-one-model-many-tasks.md:338 | docs/RESULTS.md | 476–478 |  | reattributed | quoted code is in docs/RESULTS.md, not anulm/multimodal/omni_train.py |  |
| chapters/37-one-model-many-tasks.md:338 | docs/RESULTS.md | 476–478 | 473–475 | same | also truncated the best-val file once; the '.last' file restored it. B | also truncated the best-val file once; the '.last' file restored it. B |
| chapters/37-one-model-many-tasks.md:341 | anulm/multimodal/omni_train.py | 404–414 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:341 | anulm/multimodal/omni_train.py | 404–414 | 535–546 | same | model.load_state_dict(st["model"]) | model.load_state_dict(st["model"]) |
| chapters/37-one-model-many-tasks.md:343 | anulm/multimodal/omni_train.py | 508 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:343 | anulm/multimodal/omni_train.py | 508 | 651 | same | p.add_argument("--stop-after", type=int, default=0, | p.add_argument("--stop-after", type=int, default=0, |
| chapters/37-one-model-many-tasks.md:345 | anulm/multimodal/omni_train.py | 473–474 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:345 | anulm/multimodal/omni_train.py | 473–474 | 615–616 | same | if args.stop_after and step >= args.stop_after: | if args.stop_after and step >= args.stop_after: |
| chapters/37-one-model-many-tasks.md:370 | anulm/multimodal/omni_train.py | 416 | 548 | same | first = evaluate() | first = evaluate() |
| chapters/37-one-model-many-tasks.md:371 | anulm/multimodal/omni_train.py | 417–418 | 549–550 | same | if any(not math.isfinite(value) or value <= 0 for value in first.value | if any(not math.isfinite(value) or value <= 0 for value in first.value |
| chapters/37-one-model-many-tasks.md:432 | experiments/omni_one.py | 467 | 489 | same | run("3_grow", ["-m", "anulm.multimodal.speech_vocab", "--ckpt", "ckpt_ | run("3_grow", ["-m", "anulm.multimodal.speech_vocab", "--ckpt", "ckpt_ |
| chapters/37-one-model-many-tasks.md:517 | experiments/omni_one.py | 94–97 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:517 | experiments/omni_one.py | 94–97 | 113–116 | same | # Hindi: the real-speaker data (Kathbath, MUCS) trades FLEURS for ever | # Hindi: the real-speaker data (Kathbath, MUCS) trades FLEURS for ever |
| chapters/37-one-model-many-tasks.md:591 | experiments/omni_one.py | 100 | 119 | same | "chat_heldout_loss": ("<=", 3.30),     # chat_v3: 3.146 by omni_eval.p | "chat_heldout_loss": ("<=", 3.30),     # chat_v3: 3.146 by omni_eval.p |
| chapters/37-one-model-many-tasks.md:661 | experiments/omni_one.py | 88 | 107 | same | HELD = {"chat": "data/chat_heldout.jsonl", "read": "data/rc_heldout.js | HELD = {"chat": "data/chat_heldout.jsonl", "read": "data/rc_heldout.js |
| chapters/37-one-model-many-tasks.md:663 | experiments/omni_one.py | 222–247 | 241–266 | same | def prepare_training(args, retrain: bool = False): | def prepare_training(args, retrain: bool = False): |
| chapters/37-one-model-many-tasks.md:667 | experiments/omni_one.py | 234–237 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:667 | experiments/omni_one.py | 234–237 | 253–256 | same | if previous is not None and "6_train" in s and not retrain: | if previous is not None and "6_train" in s and not retrain: |
| chapters/37-one-model-many-tasks.md:725 | docs/TRAIN_FROM_SCRATCH.md | 491 | 494 | same | python -m anulm finetune --ckpt ckpt_base_v2.pt --qa data/chat_train.j | python -m anulm finetune --ckpt ckpt_base_v2.pt --qa data/chat_train.j |
| chapters/37-one-model-many-tasks.md:745 | ? |  |  | other-pin | at @e0302d7: |  |
| chapters/37-one-model-many-tasks.md:855 | anulm/multimodal/omni_train.py | 64–65 | 72–73 | same | KINDS = ("text", "asr", "image") | KINDS = ("text", "asr", "image") |
| chapters/37-one-model-many-tasks.md:855 | anulm/multimodal/omni_train.py | 68–82 | 76–90 | same | class Task: | class Task: |
| chapters/37-one-model-many-tasks.md:856 | anulm/multimodal/omni_train.py | 85–101 | 93–109 | same | def load_rows(task: Task, tok, max_len: int) -> list[dict]: | def load_rows(task: Task, tok, max_len: int) -> list[dict]: |
| chapters/37-one-model-many-tasks.md:856 | anulm/multimodal/omni_train.py | 85–101 | 93–109 | sweep | 85–101, 'pa |  |
| chapters/37-one-model-many-tasks.md:856 | anulm/multimodal/omni_train.py | 104–111 | 112–119 | same | def pad(built, device): | def pad(built, device): |
| chapters/37-one-model-many-tasks.md:856 | anulm/multimodal/omni_train.py | 114–120 | 122–128 | same | def _row_group(row, kind): | def _row_group(row, kind): |
| chapters/37-one-model-many-tasks.md:856 | anulm/multimodal/omni_train.py | 123–136 | 131–144 | same | def split_rows(rows, kind, held_fraction=0.02): | def split_rows(rows, kind, held_fraction=0.02): |
| chapters/37-one-model-many-tasks.md:857 | anulm/multimodal/omni_train.py | 139–144 | 147–152 | same | def check_split(training, held, kind, name): | def check_split(training, held, kind, name): |
| chapters/37-one-model-many-tasks.md:857 | anulm/multimodal/omni_train.py | 148–159 | 156–167 | same | def disjoint_held(training, held, kind, name): | def disjoint_held(training, held, kind, name): |
| chapters/37-one-model-many-tasks.md:857 | anulm/multimodal/omni_train.py | 161–174 | ? | UNMAPPED | def text_batches(rows, model, device, batch_size, shuffle=True, seed=0 |  |
| chapters/37-one-model-many-tasks.md:858 | anulm/multimodal/omni_train.py | 181–235 | 293–357 | same | def __init__(self, args): | def __init__(self, args): |
| chapters/37-one-model-many-tasks.md:858 | anulm/multimodal/omni_train.py | 265–296 | 387–418 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/37-one-model-many-tasks.md:859 | anulm/multimodal/omni_train.py | 309–320 | 432–443 | same | def next_batch(self, t: Task): | def next_batch(self, t: Task): |
| chapters/37-one-model-many-tasks.md:859 | anulm/multimodal/omni_train.py | 322–334 | 445–457 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/37-one-model-many-tasks.md:859 | anulm/multimodal/omni_train.py | 336–350 | 459–473 | same | def state(self, step, score, losses) -> dict: | def state(self, step, score, losses) -> dict: |
| chapters/37-one-model-many-tasks.md:860 | anulm/multimodal/omni_train.py | 353–359 | ? | UNMAPPED | def recipe_of(args, tasks) -> dict: |  |
| chapters/37-one-model-many-tasks.md:860 | anulm/multimodal/omni_train.py | 362–481 | 488–624 | same | def train(args) -> None: | def train(args) -> None: |
| chapters/37-one-model-many-tasks.md:860 | anulm/multimodal/omni_train.py | 484–521 | 627–671 | same | def main() -> None: | def main() -> None: |
| chapters/37-one-model-many-tasks.md:861 | anulm/multimodal/omni.py | 33–86 | 33–91 | same | class Omni: | class Omni: |
| chapters/37-one-model-many-tasks.md:862 | experiments/omni_one.py | 65–74 | 65–82 | same | # Each optimizer step sums ACCUM batches. One batch per step (the firs | # Each optimizer step sums ACCUM batches. One batch per step (the firs |
| chapters/37-one-model-many-tasks.md:862 | experiments/omni_one.py | 77–87 | 85–100 | same | TASKS = [ | TASKS = [ |
| chapters/37-one-model-many-tasks.md:862 | experiments/omni_one.py | 88 | 107 | same | HELD = {"chat": "data/chat_heldout.jsonl", "read": "data/rc_heldout.js | HELD = {"chat": "data/chat_heldout.jsonl", "read": "data/rc_heldout.js |
| chapters/37-one-model-many-tasks.md:863 | experiments/omni_one.py | 92–104 | 111–123 | same | GATES = { | GATES = { |
| chapters/37-one-model-many-tasks.md:863 | experiments/omni_one.py | 222–247 | 241–266 | same | def prepare_training(args, retrain: bool = False): | def prepare_training(args, retrain: bool = False): |
| chapters/37-one-model-many-tasks.md:863 | experiments/omni_one.py | 234–237 | 253–256 | same | if previous is not None and "6_train" in s and not retrain: | if previous is not None and "6_train" in s and not retrain: |
| chapters/37-one-model-many-tasks.md:863 | experiments/omni_one.py | 234–237 | 253–256 | sweep | e_training' 222–247 (the '--retrain' refusal 234–237; the |  |
| chapters/37-one-model-many-tasks.md:863 | experiments/omni_one.py | 450 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:863 | experiments/omni_one.py | 450 | 469 | same | ap.add_argument("--retrain", action="store_true", | ap.add_argument("--retrain", action="store_true", |
| chapters/37-one-model-many-tasks.md:863 | experiments/omni_one.py | 450 | 469 | sweep | 7 (the '--retrain' refusal 234–237; the flag 450), 'g |  |
| chapters/37-one-model-many-tasks.md:863 | experiments/omni_one.py | 384–392 | 403–411 | same | def gates(res) -> dict: | def gates(res) -> dict: |
| chapters/37-one-model-many-tasks.md:864 | experiments/omni_one.py | 471–480 | 493–502 | same | args = ["-m", "anulm.multimodal.omni_train", "--ckpt", "ckpt_chat_mm.p | args = ["-m", "anulm.multimodal.omni_train", "--ckpt", "ckpt_chat_mm.p |
| chapters/37-one-model-many-tasks.md:864 | experiments/omni_one.py | 471–480 | 493–502 | sweep | command 471–480. |  |
| chapters/37-one-model-many-tasks.md:866 | anulm/multimodal/omni_eval.py | 178–182 | 178–182 | sweep | skipped when missing 178–182. |  |
| chapters/37-one-model-many-tasks.md:867 | docs/TRAIN_FROM_SCRATCH.md | 491 | 494 | same | python -m anulm finetune --ckpt ckpt_base_v2.pt --qa data/chat_train.j | python -m anulm finetune --ckpt ckpt_base_v2.pt --qa data/chat_train.j |
| chapters/37-one-model-many-tasks.md:874 | experiments/omni_one.py | 2026-11 | 2026-11 | sweep | - The November run (2026-11-03 t |  |
| chapters/37-one-model-many-tasks.md:874 | experiments/omni_one.py | 2026-11 | 2026-11 | sweep | - The November run (2026-11-03 to 2026-11-04): |  |
| chapters/38-serving.md:615 | anulm/apps/app_omni.py | 181–189 | 194–202 | same | self.shared = threading.RLock() | self.shared = threading.RLock() |
| chapters/38-serving.md:109 | anulm/inference/serve.py | 152 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:110 | anulm/inference/serve.py | 146–151 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:235 | anulm/model.py | 1174–1175 |  | reattributed | quoted code is in anulm/model.py, not book/code-examples/ch38_streamin |  |
| chapters/38-serving.md:235 | anulm/model.py | 1174–1175 | 1184–1185 | same | if on_token is not None and on_token(nxt): | if on_token is not None and on_token(nxt): |
| chapters/38-serving.md:352 | anulm/inference/api.py | 540–557 |  | reattributed | 'one_shot_reply' is in anulm/inference/api.py, not book/code-examples/ |  |
| chapters/38-serving.md:535 | anulm/inference/api.py | 748–751 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:575 | anulm/inference/api.py | 483 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:603 | anulm/multimodal/omni.py | 158 | 181 | same | prob = F.softmax(score * 10, 0)        # mean log-probs are close toge | prob = F.softmax(score * 10, 0)        # mean log-probs are close toge |
| chapters/38-serving.md:609 | anulm/multimodal/omni.py | 131–139 | 154–162 | same | if options is None: | if options is None: |
| chapters/38-serving.md:634 | anulm/apps/app_omni.py | 302–355 | ? | UNMAPPED | def chat_text(models: Models, text: str, max_tokens: int = 200) -> tup |  |
| chapters/38-serving.md:665 | anulm/inference/api.py | 567 |  | reattributed | quoted code is in anulm/inference/api.py, not docs/RESULTS.md |  |
| chapters/38-serving.md:730 | anulm/inference/api.py | 195–200 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:730 | anulm/inference/api.py | 195–200 | 195–200 | sweep | y' 173–180, 'exposure_warning' 183–188, caps 195–200, |  |
| chapters/38-serving.md:737 | anulm/multimodal/omni.py | 124–139 | 147–162 | same | def choices(self, options: list[str] \| None) -> list[str]: | def choices(self, options: list[str] \| None) -> list[str]: |
| chapters/38-serving.md:737 | anulm/multimodal/omni.py | 142 | 165 | same | def __call__(self, message: str, options: list[str] \| None = None, top | def __call__(self, message: str, options: list[str] \| None = None, top |
| chapters/38-serving.md:738 | anulm/apps/app_omni.py | 169–181 | 182–194 | same | def __init__(self, device: str, brain: str): | def __init__(self, device: str, brain: str): |
| chapters/38-serving.md:738 | anulm/apps/app_omni.py | 183–189 | 196–202 | same | def _get(self, key, make): | def _get(self, key, make): |
| chapters/38-serving.md:739 | anulm/apps/app_omni.py | 302–355 | ? | UNMAPPED | def chat_text(models: Models, text: str, max_tokens: int = 200) -> tup |  |
| chapters/38-serving.md:740 | anulm/model.py | 1115–1180 | 1125–1190 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/38-serving.md:740 | anulm/model.py | 1174–1175 | 1184–1185 | same | if on_token is not None and on_token(nxt): | if on_token is not None and on_token(nxt): |
| chapters/39-exporting.md:255 | anulm/model.py | 1296–1319 | 1299–1322 | same | p = _P(path) | p = _P(path) |
| chapters/39-exporting.md:284 | anulm/model.py | 1261 | 1264 | same | _HUB_ID = r"[\w.-]+/[\w.-]+" | _HUB_ID = r"[\w.-]+/[\w.-]+" |
| chapters/39-exporting.md:293 | tests/test_model.py | 1103–1105 |  | override | decided in repin-overrides.tsv |  |
| chapters/39-exporting.md:309 | anulm/model.py | 1264–1277 | 1267–1280 | same | def _inside(root, value: str, what: str) -> str: | def _inside(root, value: str, what: str) -> str: |
| chapters/39-exporting.md:333 | anulm/model.py | 1328–1340 | 1331–1343 | same | if "model" not in ck and base and extras.get("base_model"): | if "model" not in ck and base and extras.get("base_model"): |
| chapters/39-exporting.md:377 | anulm/hf/export_hf.py | 378–379 |  | reattributed | 'main' is in anulm/hf/export_hf.py, not None |  |
| chapters/39-exporting.md:528 | anulm/hf/modeling_anulm.py | 91–109 |  | reattributed | 'rebuild_rotary' is in anulm/hf/modeling_anulm.py, not tests/test_mode |  |
| chapters/40-the-production-gap.md:24 | anulm/model.py | 873–875 | 883–885 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/40-the-production-gap.md:32 | anulm/model.py | 480–486 |  | other-pin | excerpt pinned to @c4a39f8, left alone |  |
| chapters/40-the-production-gap.md:48 | anulm/model.py | 1144–1151 | 1154–1161 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/40-the-production-gap.md:177 | anulm/model.py | 752–759 | 762–769 | same | key = sum(p._version for e in self.experts for p in e.parameters()) | key = sum(p._version for e in self.experts for p in e.parameters()) |
| chapters/40-the-production-gap.md:268 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/40-the-production-gap.md:302 | anulm/training/train.py | 463–471 | 466–479 | same | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam |
| chapters/40-the-production-gap.md:382 | anulm/multimodal/omni.py | 106–110 | ? | UNMAPPED | body = "\n\n".join(f"[{i + 1}] {' '.join(h['text'].split()[:120])}" fo |  |
| chapters/40-the-production-gap.md:382 | anulm/multimodal/omni.py | 106 | ? | EXCERPT | header range unresolved: Reader.ask |  |
| chapters/40-the-production-gap.md:358 | anulm/inference/api.py | 195–200 |  | override | decided in repin-overrides.tsv |  |
| chapters/40-the-production-gap.md:364 | anulm/inference/api.py | 305–323 |  | reattributed | '_any_error' is in anulm/inference/api.py, not anulm/inference/serve.p |  |
| chapters/40-the-production-gap.md:368 | anulm/apps/app_omni.py | 181 | 194 | same | self.shared = threading.RLock() | self.shared = threading.RLock() |
| chapters/40-the-production-gap.md:377 | anulm/inference/api.py | 296 |  | reattributed | 'GPU' is in anulm/inference/api.py, not anulm/apps/app_omni.py |  |
| chapters/40-the-production-gap.md:467 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/40-the-production-gap.md:467 | anulm/model.py | 470–476 | 480–486 | same | if cache is not None: | if cache is not None: |
| chapters/40-the-production-gap.md:468 | anulm/model.py | 873–875 | 883–885 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/40-the-production-gap.md:469 | anulm/training/train.py | 447–474 | 450–482 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/40-the-production-gap.md:472 | anulm/inference/api.py | 195–200 | 195–200 | sweep | limits 195–200, 'GP |  |
| chapters/40-the-production-gap.md:474 | anulm/inference/serve.py | 212–213 |  | override | decided in repin-overrides.tsv |  |
| chapters/40-the-production-gap.md:474 | anulm/inference/serve.py | 212–213 | 212–213 | sweep | ence/serve.py': 'MAX_BODY_BYTES' 47, checked 212–213. |  |
| chapters/40-the-production-gap.md:475 | anulm/apps/app_omni.py | 173–181 | 186–194 | same | # One lock for everything on the shared decoder. Chat and Decide share | # One lock for everything on the shared decoder. Chat and Decide share |
| chapters/40-the-production-gap.md:475 | anulm/apps/app_omni.py | 173–181 | 186–194 | sweep | pp_omni.py': 'Models.__init__''s shared lock 173–181. |  |
| chapters/40-the-production-gap.md:476 | anulm/multimodal/omni.py | 97–111 | 111–134 | same | def ask(self, question: str, k: int = 3) -> dict: | def ask(self, question: str, k: int = 3) -> dict: |
| chapters/41-capstone.md:66 | anulm/training/train.py | 375–381 | 378–384 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/41-capstone.md:193 | anulm/training/train.py | 855–878 | 864–887 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/41-capstone.md:232 | anulm/training/train.py | 891–908 | 900–917 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/41-capstone.md:374 | anulm/model.py | 1161–1169 | 1171–1179 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/41-capstone.md:627 | tests/test_model.py | 2434–2455 | 2432–2453 | same | def main(): | def main(): |
| chapters/41-capstone.md:27 | ? |  |  | other-pin | at @fdbb09a, with 76 tests; at this pin the suite holds 80 (41.5). |  |
| chapters/41-capstone.md:51 | anulm/training/train.py | 575–576 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:51 | anulm/training/train.py | 575–576 | 584–585 | same | p.add_argument("--device", type=str, | p.add_argument("--device", type=str, |
| chapters/41-capstone.md:56 | .github/workflows/tests.yml | 132 | 132 | edited | python -u -m anulm train --preset 30b --steps 50 --eval-every 25 --out | python -u -m anulm train --preset tiny-gqa --steps 50 --eval-every 25  |
| chapters/41-capstone.md:63 | anulm/training/train.py | 554 |  | reattributed | quoted code is in anulm/training/train.py, not .github/workflows/tests |  |
| chapters/41-capstone.md:63 | anulm/training/train.py | 554 | 563 | same | p.add_argument("--warmup", type=int, default=100) | p.add_argument("--warmup", type=int, default=100) |
| chapters/41-capstone.md:81 | anulm/training/train.py | 723–740 | 732–749 | same | overrides[key] = None if val.lower() == "none" else cast(val) | overrides[key] = None if val.lower() == "none" else cast(val) |
| chapters/41-capstone.md:105 | anulm/training/train.py | 30–32 | 33–35 | same |  |  |
| chapters/41-capstone.md:146 | anulm/training/train.py |  |  | other-pin | lines 735–739 (@c4a39f8) |  |
| chapters/41-capstone.md:183 | tests/test_model.py | 447–456 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:261 | anulm/training/train.py | 104–160 | 107–163 | same | class WindowSampler: | class WindowSampler: |
| chapters/41-capstone.md:393 | anulm/model.py | 1168 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:393 | anulm/model.py | 1168 | 1178 | same | probs = F.softmax(logits, dim=-1) | probs = F.softmax(logits, dim=-1) |
| chapters/41-capstone.md:656 | tests/test_model.py | 2480 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:656 | tests/test_model.py | 2480 | 2520 | same | sys.exit(main()) | sys.exit(main()) |
| chapters/41-capstone.md:671 | README.md | 115 |  | reattributed | quoted code is in README.md, not book/measurements/part_x_tests.json |  |
| chapters/41-capstone.md:671 | README.md | 115 | 107 | edited | python -m tests.test_model                 # 80 tests, under a minute, | python -m tests.test_model                 # 82 tests, under a minute, |
| chapters/41-capstone.md:875 | anulm/training/train.py | 536–924 | 544–933 | same | def main(): | def main(): |
| chapters/41-capstone.md:875 | anulm/training/train.py | 712–723 | ? | UNMAPPED | builder = {"30b": AnuLMConfig.nano_30b, |  |
| chapters/41-capstone.md:875 | anulm/training/train.py | 712–723 | 712–723 | sweep | ng/train.py': 'main' 536–924 (preset builder 712–723, key |  |
| chapters/41-capstone.md:876 | anulm/training/train.py | 726–731 | 735–740 | same | kw = dict(block_size=args.block_size, yarn=args.yarn, | kw = dict(block_size=args.block_size, yarn=args.yarn, |
| chapters/41-capstone.md:876 | anulm/training/train.py | 726–731 | 735–740 | sweep | 726–731, the |  |
| chapters/41-capstone.md:876 | anulm/training/train.py | 855–878 | 864–887 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/41-capstone.md:876 | anulm/training/train.py | 855–878 | 864–887 | sweep | 726–731, the step 855–878, eva |  |
| chapters/41-capstone.md:876 | anulm/training/train.py | 891–908 | 900–917 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/41-capstone.md:876 | anulm/training/train.py | 891–908 | 900–917 | sweep | –731, the step 855–878, evaluation and saves 891–908), 'l |  |
| chapters/41-capstone.md:876 | anulm/training/train.py | 62–98 | 65–101 | same | def load_bytes(path: str \| None) -> torch.Tensor: | def load_bytes(path: str \| None) -> torch.Tensor: |
| chapters/41-capstone.md:877 | anulm/training/train.py | 104–160 | 107–163 | same | class WindowSampler: | class WindowSampler: |
| chapters/41-capstone.md:877 | anulm/training/train.py | 375–400 | 378–403 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/41-capstone.md:877 | anulm/training/train.py | 477–533 | 485–541 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/41-capstone.md:878 | anulm/training/train.py | 548 | 557 | same | p.add_argument("--steps", type=int, default=2000) | p.add_argument("--steps", type=int, default=2000) |
| chapters/41-capstone.md:878 | anulm/training/train.py | 554 | 563 | same | p.add_argument("--warmup", type=int, default=100) | p.add_argument("--warmup", type=int, default=100) |
| chapters/41-capstone.md:878 | anulm/training/train.py | 554 | 563 | sweep | '--steps' 2000 (line 548), '--warmup' 100 (554), '- |  |
| chapters/41-capstone.md:878 | anulm/training/train.py | 575–576 | 584–585 | same | p.add_argument("--device", type=str, | p.add_argument("--device", type=str, |
| chapters/41-capstone.md:878 | anulm/training/train.py | 575–576 | 584–585 | sweep | line 548), '--warmup' 100 (554), '--device' (575–576). |  |
| chapters/41-capstone.md:879 | ? |  |  | other-pin | at @c4a39f8: 'PRESETS' 30–32; in 'main', the |  |
| chapters/41-capstone.md:880 | anulm/training/train.py |  |  | other-pin | preset builder 723–732 and the keyword arguments 735–740. |  |
| chapters/41-capstone.md:881 | .github/workflows/tests.yml | 132 | 132 | edited | python -u -m anulm train --preset 30b --steps 50 --eval-every 25 --out | python -u -m anulm train --preset tiny-gqa --steps 50 --eval-every 25  |
| chapters/41-capstone.md:882 | anulm/model.py | 1115–1180 | 1125–1190 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/41-capstone.md:882 | anulm/model.py | 1161–1169 | 1171–1179 | same | if temperature <= 0: | if temperature <= 0: |
| chapters/41-capstone.md:882 | anulm/model.py | 1161–1169 | 1171–1179 | sweep | uLM.generate' 1115–1180 (greedy and sampling 1161–1169), |  |
| chapters/41-capstone.md:883 | anulm/model.py | 1104–1113 | 1114–1123 | same | def num_params(self) -> tuple[int, int]: | def num_params(self) -> tuple[int, int]: |
| chapters/41-capstone.md:886 | tests/test_model.py | 507 | 507 | sweep | 507, 'gr |  |
| chapters/41-capstone.md:888 | tests/test_model.py | 2434–2455 | 2432–2453 | same | def main(): | def main(): |
| chapters/41-capstone.md:889 | tests/test_model.py | 832 | 832 | sweep | 'unittest.SkipTest' at 832, 853 |  |
| chapters/41-capstone.md:889 | tests/test_model.py | 853 | 853 | sweep | 'unittest.SkipTest' at 832, 853, 871 |  |
| chapters/41-capstone.md:889 | tests/test_model.py | 871 | 871 | sweep | 'unittest.SkipTest' at 832, 853, 871, 997 |  |
| chapters/41-capstone.md:889 | tests/test_model.py | 997 | 997 | sweep | 'unittest.SkipTest' at 832, 853, 871, 997, 103 |  |
| chapters/41-capstone.md:889 | tests/test_model.py | 1034 | 1034 | sweep | 'unittest.SkipTest' at 832, 853, 871, 997, 1034, 107 |  |
| chapters/41-capstone.md:889 | tests/test_model.py | 1073 | 1073 | sweep | ttest.SkipTest' at 832, 853, 871, 997, 1034, 1073, 129 |  |
| chapters/41-capstone.md:889 | tests/test_model.py | 1299 | 1299 | sweep | SkipTest' at 832, 853, 871, 997, 1034, 1073, 1299, 130 |  |
| chapters/41-capstone.md:889 | tests/test_model.py | 1303 | 1303 | sweep | st' at 832, 853, 871, 997, 1034, 1073, 1299, 1303, 135 |  |
| chapters/41-capstone.md:889 | tests/test_model.py | 1351 | 1351 | sweep | 832, 853, 871, 997, 1034, 1073, 1299, 1303, 1351, |  |
| chapters/41-capstone.md:890 | tests/test_model.py | 1464 | 1464 | sweep | 1464, 152 |  |
| chapters/41-capstone.md:890 | tests/test_model.py | 1523 | 1523 | sweep | 1464, 1523, 215 |  |
| chapters/41-capstone.md:890 | tests/test_model.py | 2152 | 2150 | same | raise unittest.SkipTest(f"API dependency unavailable: {e}") | raise unittest.SkipTest(f"API dependency unavailable: {e}") |
| chapters/41-capstone.md:890 | tests/test_model.py | 2152 | 2150 | sweep | 1464, 1523, 2152 and |  |
| chapters/41-capstone.md:890 | tests/test_model.py | 2203 | 2201 | same | raise unittest.SkipTest(f"app dependency unavailable: {e}") | raise unittest.SkipTest(f"app dependency unavailable: {e}") |
| chapters/41-capstone.md:890 | tests/test_model.py | 2203 | 2201 | sweep | 1464, 1523, 2152 and 2203; 'sy |  |
| chapters/41-capstone.md:890 | tests/test_model.py | 2480 | 2520 | same | sys.exit(main()) | sys.exit(main()) |
| chapters/41-capstone.md:890 | tests/test_model.py | 2480 | 2520 | sweep | 464, 1523, 2152 and 2203; 'sys.exit(main())' 2480; 80 |  |
| chapters/41-capstone.md:890 | tests/test_model.py | 80 | 80 | sweep | 523, 2152 and 2203; 'sys.exit(main())' 2480; 80 test |  |
| chapters/41-capstone.md:892 | tests/test_quality_serving.py | 63 | 63 | sweep | 'tests/test_quality_serving.py': 63 'uni |  |
| chapters/41-capstone.md:892 | tests/test_quality_serving.py | 18 | 18 | sweep | st_quality_serving.py': 63 'unittest' tests (18, 19 |  |
| chapters/41-capstone.md:892 | tests/test_quality_serving.py | 19 | 19 | sweep | uality_serving.py': 63 'unittest' tests (18, 19 and |  |
| chapters/41-capstone.md:892 | tests/test_quality_serving.py | 26 | 26 | sweep | serving.py': 63 'unittest' tests (18, 19 and 26). |  |
| chapters/41-capstone.md:894 | anulm/hf/export_hf.py | 328–339 | 328–339 | sweep | float32 config and its comment 328–339: by |  |
| chapters/41-capstone.md:902 | anulm/bpe.py | 11 | 11 | sweep | 11 and |  |
| chapters/41-capstone.md:902 | anulm/bpe.py | 15 | 15 | sweep | 11 and 15, 'co |  |
| chapters/A1-shape-trace.md:64 | anulm/model.py | 896 | 906 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| chapters/A1-shape-trace.md:76 | anulm/model.py | 919 | 929 | same | if self.ce_chunk and self.training: | if self.ce_chunk and self.training: |
| chapters/A2-parameter-arithmetic.md:34 | anulm/model.py | 39–218 | 35–228 | same | @dataclass | @dataclass |
| chapters/A2-parameter-arithmetic.md:85 | anulm/model.py | 126–156 |  | reattributed | '__post_init__' is in anulm/model.py, not data/multi32k.json |  |
| chapters/A2-parameter-arithmetic.md:85 | anulm/model.py | 126–156 | 121–151 | same | def __post_init__(self): | def __post_init__(self): |
| chapters/A3-test-index.md:11 | tests/test_model.py | 34 |  | override | decided in repin-overrides.tsv |  |
| chapters/A3-test-index.md:12 | tests/test_model.py | 2434 | 2432 | same | def main(): | def main(): |
| chapters/A3-test-index.md:15 | tests/test_model.py | 44 |  | override | decided in repin-overrides.tsv |  |
| chapters/A3-test-index.md:141 | tests/test_model.py | 832, 853, 871 |  | reattributed | quoted code is in tests/test_model.py, not book/code-examples/ch41_sha |  |
| chapters/A3-test-index.md:142 | tests/test_model.py | 1034 |  | reattributed | quoted code is in tests/test_model.py, not book/code-examples/ch41_sha |  |
| chapters/A6-number-audit.md:46 | ? |  |  | other-pin | at @fdbb09a, when it held 76 tests), the second in about 5 s. '-1' |  |
| chapters/A6-number-audit.md:187 | anulm/model.py | 756 |  | override | decided in repin-overrides.tsv |  |
| chapters/A6-number-audit.md:187 | anulm/model.py | 756 | 766 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/A6-number-audit.md:187 | anulm/model.py | 784 |  | override | decided in repin-overrides.tsv |  |
| chapters/A6-number-audit.md:187 | anulm/model.py | 784 | 794 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/A6-number-audit.md:196 | anulm/model.py | 756 and 784 |  | override | decided in repin-overrides.tsv |  |
| chapters/A6-number-audit.md:196 | anulm/model.py | 756 | 766 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/A6-number-audit.md:196 | anulm/model.py | 784 | 794 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| REPOSITORY_SOURCE_MAP.md:92 | anulm/training/train.py | 62–98 | 65–101 | same | def load_bytes(path: str \| None) -> torch.Tensor: | def load_bytes(path: str \| None) -> torch.Tensor: |
| REPOSITORY_SOURCE_MAP.md:92 | anulm/training/train.py | 101 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_SOURCE_MAP.md:92 | anulm/training/train.py | 101 | 104 | same | BYTE_EOS = 257          # byte mode: 256 BOS, 257 EOS, 258 PAD | BYTE_EOS = 257          # byte mode: 256 BOS, 257 EOS, 258 PAD |
| REPOSITORY_SOURCE_MAP.md:154 | anulm/multimodal/vision_encoder.py | 507 |  | reattributed | 'train' is in anulm/multimodal/vision_encoder.py, not anulm/multimodal |  |
| REPOSITORY_SOURCE_MAP.md:155 | anulm/multimodal/omni_train.py | 504 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_SOURCE_MAP.md:155 | anulm/multimodal/omni_train.py | 504 | 647 | same | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", | p.add_argument("--proj-scale", choices=("ln", "embed"), default="ln", |
| REPOSITORY_ARCHITECTURE.md:18 | anulm/model.py | 1–17 | ? | UNMAPPED | """ |  |
| REPOSITORY_ARCHITECTURE.md:117 | anulm/training/train.py | 62–98 | 65–101 | same | def load_bytes(path: str \| None) -> torch.Tensor: | def load_bytes(path: str \| None) -> torch.Tensor: |
| REPOSITORY_ARCHITECTURE.md:117 | anulm/training/train.py | 395–449 | 398–452 | same | # several times the lr the weights last saw. --rewarmup ramps back up  | # several times the lr the weights last saw. --rewarmup ramps back up  |
| REPOSITORY_ARCHITECTURE.md:117 | anulm/training/train.py | 163–173 | 166–176 | same | def load_memmap(meta_path: str): | def load_memmap(meta_path: str): |
| REPOSITORY_ARCHITECTURE.md:118 | anulm/training/train.py | 104–160 | 107–163 | same | class WindowSampler: | class WindowSampler: |
| REPOSITORY_ARCHITECTURE.md:119 | anulm/training/train.py | 237–250 | 240–253 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| REPOSITORY_ARCHITECTURE.md:120 | anulm/model.py | 896 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:120 | anulm/model.py | 896 | 906 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| REPOSITORY_ARCHITECTURE.md:120 | anulm/model.py | 966–984 |  | reattributed | '_table' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:120 | anulm/model.py | 966–984 | 976–994 | same | def _table(self, side: str): | def _table(self, side: str): |
| REPOSITORY_ARCHITECTURE.md:121 | anulm/model.py | 339–343 |  | reattributed | 'Rotary.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:121 | anulm/model.py | 339–343 | 349–353 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| REPOSITORY_ARCHITECTURE.md:122 | anulm/model.py | 816–824 |  | reattributed | 'Block.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:122 | anulm/model.py | 816–824 | 826–834 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| REPOSITORY_ARCHITECTURE.md:123 | anulm/model.py | 451–494 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not anulm/training/train.p |  |
| REPOSITORY_ARCHITECTURE.md:123 | anulm/model.py | 451–494 | 461–504 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| REPOSITORY_ARCHITECTURE.md:124 | anulm/model.py | 365–381 |  | reattributed | '_attn_mask' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:124 | anulm/model.py | 365–381 | 375–391 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| REPOSITORY_ARCHITECTURE.md:124 | anulm/model.py | 414–418 |  | reattributed | '_sdpa' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:124 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| REPOSITORY_ARCHITECTURE.md:125 | anulm/model.py | 253 |  | reattributed | quoted code is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:125 | anulm/model.py | 253 | 263 | same | return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) | return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) |
| REPOSITORY_ARCHITECTURE.md:125 | anulm/model.py | 696–707 |  | reattributed | 'MoE.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:125 | anulm/model.py | 696–707 | 706–717 | same | def forward(self, x):                                     # (B, S, D) | def forward(self, x):                                     # (B, S, D) |
| REPOSITORY_ARCHITECTURE.md:126 | anulm/model.py | 624–672 |  | reattributed | 'Router._gate' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:126 | anulm/model.py | 624–672 | 634–682 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| REPOSITORY_ARCHITECTURE.md:127 | anulm/model.py | 912–928 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/training/train.py |  |
| REPOSITORY_ARCHITECTURE.md:127 | anulm/model.py | 912–928 | 922–938 | same | x = self.norm(x) | x = self.norm(x) |
| REPOSITORY_ARCHITECTURE.md:128 | anulm/training/train.py | 447–474 | 450–482 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| REPOSITORY_ARCHITECTURE.md:129 | anulm/training/train.py | 375–400 |  | reattributed | 'lr_at' is in anulm/training/train.py, not anulm/muon.py |  |
| REPOSITORY_ARCHITECTURE.md:129 | anulm/training/train.py | 375–400 | 378–403 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| REPOSITORY_ARCHITECTURE.md:130 | anulm/training/train.py | 850–908 |  | reattributed | 'main' is in anulm/training/train.py, not anulm/muon.py |  |
| REPOSITORY_ARCHITECTURE.md:130 | anulm/training/train.py | 850–908 | 859–917 | same | for step in range(start_step, end): | for step in range(start_step, end): |
| REPOSITORY_ARCHITECTURE.md:145 | anulm/model.py | 873 |  | reattributed | 'new_cache' is in anulm/model.py, not anulm/inference/sample.py |  |
| REPOSITORY_ARCHITECTURE.md:145 | anulm/model.py | 873 | 883 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| REPOSITORY_ARCHITECTURE.md:147 | anulm/model.py | 470–476 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:147 | anulm/model.py | 470–476 | 480–486 | same | if cache is not None: | if cache is not None: |
| REPOSITORY_ARCHITECTURE.md:148 | anulm/model.py | 555–563 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:148 | anulm/model.py | 555–563 | 565–573 | same | if cache is not None: | if cache is not None: |
| REPOSITORY_ARCHITECTURE.md:153 | anulm/model.py | 365–381 |  | reattributed | quoted code is in anulm/model.py, not anulm/inference/sample.py |  |
| REPOSITORY_ARCHITECTURE.md:153 | anulm/model.py | 365–381 | 375–391 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| REPOSITORY_ARCHITECTURE.md:155 | tests/test_model.py | 1148–1151 |  | reattributed | quoted code is in tests/test_model.py, not anulm/inference/sample.py |  |
| REPOSITORY_ARCHITECTURE.md:163 | anulm/model.py | 891–894 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:163 | anulm/model.py | 891–894 | 901–904 | same | if embeds is not None: | if embeds is not None: |
| REPOSITORY_ARCHITECTURE.md:163 | anulm/model.py | 1016–1062 |  | reattributed | 'decode_from_embeds' is in anulm/model.py, not anulm/hf/modeling_anulm |  |
| REPOSITORY_ARCHITECTURE.md:163 | anulm/model.py | 1016–1062 | 1026–1072 | same | def decode_from_embeds(self, prefix, max_tokens: int, stop, no_repeat_ | def decode_from_embeds(self, prefix, max_tokens: int, stop, no_repeat_ |
| REPOSITORY_ARCHITECTURE.md:172 | anulm/model.py | 709–790 |  | reattributed | '_route_grouped' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:172 | anulm/model.py | 709–790 | 719–800 | same | def _route_sparse(self, x_flat, topk_idx, topk_w): | def _route_sparse(self, x_flat, topk_idx, topk_w): |
| REPOSITORY_ARCHITECTURE.md:173 | anulm/model.py | 611–622 |  | reattributed | 'Router.forward' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:173 | anulm/model.py | 611–622 | 621–632 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| REPOSITORY_ARCHITECTURE.md:174 | anulm/model.py | 647–670 |  | reattributed | 'Router._gate' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:174 | anulm/model.py | 647–670 | 657–680 | same | aux = None | aux = None |
| REPOSITORY_ARCHITECTURE.md:174 | anulm/model.py | 1071–1101 |  | reattributed | 'AnuLM.update_expert_biases' is in anulm/model.py, not anulm/hf/modeli |  |
| REPOSITORY_ARCHITECTURE.md:174 | anulm/model.py | 1071–1101 | 1081–1111 | same | @torch.no_grad() | @torch.no_grad() |
| REPOSITORY_ARCHITECTURE.md:175 | anulm/model.py | 902–907 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:175 | anulm/model.py | 902–907 | 912–917 | same | if self.grad_ckpt and self.training: | if self.grad_ckpt and self.training: |
| REPOSITORY_ARCHITECTURE.md:176 | anulm/model.py | 930–956 |  | reattributed | '_chunked_loss' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:176 | anulm/model.py | 930–956 | 940–966 | same | def enable_chunked_loss(self, rows: int = 2048): | def enable_chunked_loss(self, rows: int = 2048): |
| REPOSITORY_ARCHITECTURE.md:177 | anulm/model.py | 390–411 |  | reattributed | 'merge_tree_matrix' is in anulm/model.py, not anulm/hf/modeling_anulm. |  |
| REPOSITORY_ARCHITECTURE.md:177 | anulm/model.py | 390–411 | 400–421 | same | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: |
| REPOSITORY_ARCHITECTURE.md:178 | anulm/model.py | 860–863 |  | reattributed | 'AnuLM.__init__' is in anulm/model.py, not anulm/hf/modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:178 | anulm/model.py | 860–863 | 870–873 | same | # nanoGPT's scaled init for the residual output projections. | # nanoGPT's scaled init for the residual output projections. |
| REPOSITORY_ARCHITECTURE.md:179 | anulm/training/train.py | 645–650 | 656–661 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| REPOSITORY_ARCHITECTURE.md:179 | anulm/training/train.py | 463 | 466 | same | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam |
| REPOSITORY_ARCHITECTURE.md:179 | anulm/training/train.py | 750–770 | 759–779 | same | if args.compile: | if args.compile: |
| REPOSITORY_ARCHITECTURE.md:180 | anulm/training/train.py | 857–873 | 866–882 | same | for _ in range(args.grad_accum): | for _ in range(args.grad_accum): |
| REPOSITORY_ARCHITECTURE.md:181 | anulm/training/train.py | 32–46 | 35–49 | same | def atomic_save(obj, path) -> None: | def atomic_save(obj, path) -> None: |
| REPOSITORY_ARCHITECTURE.md:181 | anulm/training/train.py | 788–840 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:181 | anulm/training/train.py | 788–840 | 797–849 | same | start_step, best_val = 0, float("inf") | start_step, best_val = 0, float("inf") |
| REPOSITORY_ARCHITECTURE.md:182 | anulm/training/train.py | 477–533 | 485–541 | same | @torch.no_grad() | @torch.no_grad() |
| REPOSITORY_ARCHITECTURE.md:183 | anulm/training/train.py | 202–216 | 205–219 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| REPOSITORY_ARCHITECTURE.md:186 | anulm/model.py | 1345–1364 |  | reattributed | 'fit_memory' is in anulm/model.py, not anulm/bigram_teacher.py |  |
| REPOSITORY_ARCHITECTURE.md:186 | anulm/model.py | 1345–1364 | 1348–1367 | same | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i |
| REPOSITORY_ARCHITECTURE.md:203 | anulm/model.py | 19 | ? | UNMAPPED | * no expert parallelism, no vLLM branch, no FlashAttention subclass: |  |
| FRONT_MATTER.md:73 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| FRONT_MATTER.md:96 | anulm/model.py | 414–418 | 424–428 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| README.md:76 | anulm/model.py | 615–621 | 625–631 | same | # fp32 router: top-k is discrete, and bf16 rounding would flip choices | # fp32 router: top-k is discrete, and bf16 rounding would flip choices |
| README.md:77 | anulm/training/train.py | 32–46 | 35–49 | same | def atomic_save(obj, path) -> None: | def atomic_save(obj, path) -> None: |
| BOOK_PLAN.md:87 | anulm/model.py | 922–928 |  | reattributed | 'AnuLM.forward' is in anulm/model.py, not anulm/bidirectional.py |  |
| BOOK_PLAN.md:87 | anulm/model.py | 922–928 | 932–938 | same | logits = self._head(x) | logits = self._head(x) |
| BOOK_PLAN.md:87 | anulm/training/train.py | 906 | 915 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| BOOK_PLAN.md:94 | anulm/model.py | 457–463 |  | reattributed | 'GQAttention.forward' is in anulm/model.py, not anulm/training/train.p |  |
| BOOK_PLAN.md:94 | anulm/model.py | 457–463 | 467–473 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| BOOK_PLAN.md:102 | anulm/training/train.py | 868 | 877 | same | (loss / args.grad_accum).backward() | (loss / args.grad_accum).backward() |
| STYLE.md:111 | anulm/model.py | 451–494 | 461–504 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| RESEARCH_REFERENCES.md:204 | anulm/model.py | 137 | 132 | same | "moe_impl='grouped' needs torch.nn.functional.grouped_mm (present in 2 | "moe_impl='grouped' needs torch.nn.functional.grouped_mm (present in 2 |
| RESEARCH_REFERENCES.md:301 | anulm/model.py | 109–110 |  | override | decided in repin-overrides.tsv |  |
| RESEARCH_REFERENCES.md:301 | anulm/model.py | 109–110 | 104–105 | same | yarn_beta_fast: float = 32.0 | yarn_beta_fast: float = 32.0 |
| RESEARCH_REFERENCES.md:301 | anulm/model.py | 449 |  | override | decided in repin-overrides.tsv |  |
| RESEARCH_REFERENCES.md:301 | anulm/model.py | 449 | 459 | same | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) |
| RESEARCH_REFERENCES.md:304 | anulm/model.py | 443–445 |  | override | decided in repin-overrides.tsv |  |
| RESEARCH_REFERENCES.md:304 | anulm/model.py | 443–445 | 453–455 | same | if cfg.use_qk_norm: | if cfg.use_qk_norm: |
| chapters/26-debugging-silent-failures.md:493 | tests/test_model.py | 1544 |  | STALE OVERRIDE | no citation matches this row |  |
