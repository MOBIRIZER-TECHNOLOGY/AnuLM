# Re-pin report: @773b6a8 -> @084cbdd

Written by `book/tools/repin.py`. Every line citation it moved, with the code at
the old line and at the new one. Rows marked UNMAPPED, CHECK, EXCERPT or OUT OF
FILE were **not** resolved automatically and need a person.

| kind | count |
|---|---|
| CHECK | 2 |
| EXCERPT | 27 |
| UNMAPPED | 28 |
| def | 8 |
| edited | 26 |
| edited-quote | 31 |
| override | 58 |
| reattributed | 78 |
| reindented | 4 |
| same | 818 |
| sweep | 96 |

## Needs a person (57)

| where | file | old | new | kind | old code | new code |
|---|---|---|---|---|---|---|
| chapters/01-what-this-book-builds.md:114 | model.py | 1132–1161 |  | EXCERPT | logits = logits / max(temperature, 1e-5) | quoted line not in the new range |
| chapters/01-what-this-book-builds.md:114 | model.py | 1132–1161 |  | EXCERPT | nxt = torch.multinomial(probs, num_samples=1)              # (B, 1) | quoted line not in the new range |
| chapters/03-the-one-job.md:322 | train.py | 660–675 |  | EXCERPT | if memmapped: | quoted line not in the new range |
| chapters/05-softmax-temperature-and-cross-entropy.md:172 | model.py | 1091–1105 | ? | UNMAPPED | logits = logits[:, -1, :].float() |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:172 | model.py | 1091 | ? | EXCERPT | header range unresolved: AnuLM.generate |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:472 | model.py | 1091–1105 | ? | UNMAPPED | logits = logits[:, -1, :].float() |  |
| chapters/18-dispatch-and-balance.md:408 | export_hf.py | 271–279 | ? | UNMAPPED | # Small tensors stay float32 on disk AND must be loaded that way: the |  |
| chapters/18-dispatch-and-balance.md:408 | export_hf.py | 271 | ? | EXCERPT | header range unresolved: main |  |
| chapters/18-dispatch-and-balance.md:587 | export_hf.py | 271–279 | ? | UNMAPPED | # Small tensors stay float32 on disk AND must be loaded that way: the |  |
| chapters/20-the-whole-model.md:489 | model.py | 1091–1105 | ? | UNMAPPED | logits = logits[:, -1, :].float() |  |
| chapters/20-the-whole-model.md:489 | model.py | 1091 | ? | EXCERPT | header range unresolved: AnuLM.generate |  |
| chapters/21-data.md:138 | train.py | 663–676 |  | EXCERPT | if memmapped: | quoted line not in the new range |
| chapters/24-measuring-progress.md:478 | train.py | 740–742 | ? | CHECK | bpb = val / math.log(2) / args.bytes_per_token | 'eval_spec' spans [(622, 626, 622)] at the old pin |
| chapters/24-measuring-progress.md:482 | eval_code.py | 185–230 | ? | UNMAPPED | def score(ckpt: str, problems: list[dict], device: str, max_new: int,  |  |
| chapters/27-decoding-and-the-kv-cache.md:66 | model.py | 1156–1161 |  | EXCERPT | nxt = torch.multinomial(probs, num_samples=1)              # (B, 1) | quoted line not in the new range |
| chapters/28-choosing-the-next-token.md:19 | model.py | 1091–1105 | ? | UNMAPPED | logits = logits[:, -1, :].float() |  |
| chapters/28-choosing-the-next-token.md:19 | model.py | 1091 | ? | EXCERPT | header range unresolved: AnuLM.generate |  |
| chapters/28-choosing-the-next-token.md:77 | model.py | 1105 | ? | UNMAPPED | nxt = torch.multinomial(probs, num_samples=1)              # (B, 1) |  |
| chapters/30-instruction-tuning.md:318 | make_chat.py | 105–112 | ? | UNMAPPED | for name, items in parts.items(): |  |
| chapters/30-instruction-tuning.md:318 | make_chat.py | 105 | ? | EXCERPT | header range unresolved: main |  |
| chapters/30-instruction-tuning.md:482 | make_chat.py | 97–119 | ? | UNMAPPED | def main(): |  |
| chapters/30-instruction-tuning.md:482 | make_chat.py | 109–112 | ? | UNMAPPED | rng.shuffle(rows) |  |
| chapters/33-retrieval-and-readers.md:497 | rag.py | 87 | ? | CHECK | ids = term_ids(title + " " + chunk) | 'term_ids' spans [(55, 56, 55)] at the old pin |
| chapters/34-preference-optimisation-and-rl.md:319 | model.py | 1101–1105 | ? | UNMAPPED | if top_k is not None: |  |
| chapters/34-preference-optimisation-and-rl.md:319 | model.py | 1101 | ? | EXCERPT | header range unresolved: AnuLM.generate |  |
| chapters/34-preference-optimisation-and-rl.md:400 | model.py | 1101–1105 | ? | UNMAPPED | if top_k is not None: |  |
| chapters/35-wider-vocabulary.md:264 | speech_encoder.py | 106–126 | ? | UNMAPPED | def build(self, states: torch.Tensor, text_ids: list[int], answer_ids: |  |
| chapters/35-wider-vocabulary.md:264 | speech_encoder.py | 106 | ? | EXCERPT | header range unresolved: ContinuousSpeech.build |  |
| chapters/35-wider-vocabulary.md:410 | speech_encoder.py | 342–346 | ? | UNMAPPED | with autocast: |  |
| chapters/35-wider-vocabulary.md:410 | speech_encoder.py | 342 | ? | EXCERPT | header range unresolved: transcribe |  |
| chapters/35-wider-vocabulary.md:485 | speech_encoder.py | 106–126 | ? | UNMAPPED | def build(self, states: torch.Tensor, text_ids: list[int], answer_ids: |  |
| chapters/35-wider-vocabulary.md:486 | speech_encoder.py | 342–346 | ? | UNMAPPED | with autocast: |  |
| chapters/36-frozen-encoders.md:353 | vision_encoder.py | 312–361 |  | EXCERPT | if path.exists() and path.stat().st_size == int(np.prod(shape)) * 2: | quoted line not in the new range |
| chapters/36-frozen-encoders.md:353 | vision_encoder.py | 312–361 |  | EXCERPT | mm = np.memmap(path, dtype=dtype, mode="w+", shape=shape) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:113 | omni_train.py | 217–235 |  | EXCERPT | cut = max(1, int(len(rows) * 0.02)) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:113 | omni_train.py | 217–235 |  | EXCERPT | t.held, t.rows = rows[:cut], rows[cut:] | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:113 | omni_train.py | 217–235 |  | EXCERPT | from vision_encoder import build_cache | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:113 | omni_train.py | 217–235 |  | EXCERPT | self.caches[t.name] = build_cache(rows, self.tower, Path(t.path).with_ | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:162 | omni_train.py | 282–305 | ? | UNMAPPED | w = np.array([t.weight for t in om.tasks], dtype=float) |  |
| chapters/37-one-model-many-tasks.md:162 | omni_train.py | 282 | ? | EXCERPT | header range unresolved: train |  |
| chapters/37-one-model-many-tasks.md:384 | omni_train.py | 211–221 | ? | UNMAPPED | @torch.no_grad() |  |
| chapters/37-one-model-many-tasks.md:384 | omni_train.py | 211 | ? | EXCERPT | header range unresolved: Omni.held_loss |  |
| chapters/37-one-model-many-tasks.md:494 | omni_train.py | 333–361 | ? | UNMAPPED | def main() -> None: |  |
| chapters/38-serving.md:74 | serve.py | 129–141 | ? | UNMAPPED | ids = self.encode(prompt)[-(self.cfg.block_size - 1):] |  |
| chapters/38-serving.md:74 | serve.py | 129 | ? | EXCERPT | header range unresolved: Engine.generate |  |
| chapters/38-serving.md:232 | api.py | 360–376 |  | EXCERPT | text = self.eng.decode(self.new) | quoted line not in the new range |
| chapters/38-serving.md:232 | api.py | 360–376 |  | EXCERPT | if self.finish != "stop" and not final: | quoted line not in the new range |
| chapters/38-serving.md:232 | api.py | 360–376 |  | EXCERPT | text = text.rstrip("�") | quoted line not in the new range |
| chapters/38-serving.md:294 | api.py | 390–394 | ? | UNMAPPED | if self.finish != "stop": |  |
| chapters/38-serving.md:294 | api.py | 390 | ? | EXCERPT | header range unresolved: Generation.run |  |
| chapters/41-capstone.md:365 | model.py | 1100–1105 | ? | UNMAPPED | logits = logits / max(temperature, 1e-5) |  |
| chapters/41-capstone.md:365 | model.py | 1100 | ? | EXCERPT | header range unresolved: AnuLM.generate |  |
| chapters/41-capstone.md:653 | export_hf.py | 271–279 | ? | UNMAPPED | # Small tensors stay float32 on disk AND must be loaded that way: the |  |
| chapters/41-capstone.md:877 | model.py | 1100–1105 | ? | UNMAPPED | logits = logits / max(temperature, 1e-5) |  |
| chapters/41-capstone.md:886 | export_hf.py | 271–279 | ? | UNMAPPED | # Small tensors stay float32 on disk AND must be loaded that way: the |  |
| chapters/A4-glossary.md:52 | export_hf.py | 271–279 | ? | UNMAPPED | # Small tensors stay float32 on disk AND must be loaded that way: the |  |
| REPOSITORY_SOURCE_MAP.md:150 | omni_train.py | 196–211 | ? | UNMAPPED | from speech_encoder import batches |  |

## Every change

| where | file | old | new | kind | old code | new code |
|---|---|---|---|---|---|---|
| chapters/01-what-this-book-builds.md:114 | model.py | 1083–1109 | 1132–1161 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/01-what-this-book-builds.md:114 | model.py | 1132–1161 |  | EXCERPT | logits = logits / max(temperature, 1e-5) | quoted line not in the new range |
| chapters/01-what-this-book-builds.md:114 | model.py | 1132–1161 |  | EXCERPT | nxt = torch.multinomial(probs, num_samples=1)              # (B, 1) | quoted line not in the new range |
| chapters/01-what-this-book-builds.md:154 | model.py | 860–863 | 903–906 | same | if targets is None: | if targets is None: |
| chapters/01-what-this-book-builds.md:172 | sample.py | 66–74 | 69–77 | same | else: | else: |
| chapters/01-what-this-book-builds.md:230 | model.py | 782–793 | 820–836 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/01-what-this-book-builds.md:337 | quickstart.py | 182–189 | 185–192 | edited | rc = run([sys.executable, "-u", "train.py", "--preset", "30b", "--step | rc = run([sys.executable, "-u", "-m", "anulm", "train", "--preset", "3 |
| chapters/01-what-this-book-builds.md:337 | quickstart.py | 182 | 185 | edited-quote | rc = run([sys.executable, "-u", "train.py", "--preset", "30b", "--step | rc = run([sys.executable, "-u", "-m", "anulm", "train", "--preset", "3 |
| chapters/01-what-this-book-builds.md:337 | quickstart.py | 188 | 191 | edited-quote | rc = run([sys.executable, "sample.py", "--ckpt", "ckpt_quickstart.pt", | rc = run([sys.executable, "-m", "anulm", "sample", "--ckpt", "ckpt_qui |
| chapters/01-what-this-book-builds.md:355 | model.py | 147–159 | 152–164 | same | @staticmethod | @staticmethod |
| chapters/01-what-this-book-builds.md:111 | model.py | 1059 |  | reattributed | 'generate' is defined there |  |
| chapters/01-what-this-book-builds.md:111 | model.py | 1059 | 1104 | def | 'generate' | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/01-what-this-book-builds.md:191 | sample.py | 31–32 |  | override | decided in repin-overrides.tsv |  |
| chapters/01-what-this-book-builds.md:191 | sample.py | 31–32 | 34–35 | same | p.add_argument("--temperature", type=float, default=0.8) | p.add_argument("--temperature", type=float, default=0.8) |
| chapters/01-what-this-book-builds.md:351 | quickstart.py | 220 |  | reattributed | quoted code is in quickstart.py, not sample.py |  |
| chapters/01-what-this-book-builds.md:351 | quickstart.py | 220 | 226 | same | p.add_argument("--demo", action="store_true", help="download a release | p.add_argument("--demo", action="store_true", help="download a release |
| chapters/01-what-this-book-builds.md:518 | model.py | 210 | 215 | same | # NORM / MLP | # NORM / MLP |
| chapters/01-what-this-book-builds.md:518 | model.py | 240 | 250 | same | # ROPE  (see sarvam/yarn_rotary_annotated.py) | # ROPE  (see sarvam/yarn_rotary_annotated.py) |
| chapters/01-what-this-book-builds.md:518 | model.py | 401 | 415 | same | # ATTENTION -- GQA  (see sarvam/gqa_forward_annotated.py) | # ATTENTION -- GQA  (see sarvam/gqa_forward_annotated.py) |
| chapters/01-what-this-book-builds.md:518 | model.py | 469 | 491 | same | # ATTENTION -- MLA  (see sarvam/mla_forward_annotated.py) | # ATTENTION -- MLA  (see sarvam/mla_forward_annotated.py) |
| chapters/01-what-this-book-builds.md:518 | model.py | 547 | 575 | same | # MoE  (see sarvam/moe_routing_annotated.py) | # MoE  (see sarvam/moe_routing_annotated.py) |
| chapters/01-what-this-book-builds.md:518 | model.py | 753 | 787 | same | # BLOCK / MODEL | # BLOCK / MODEL |
| chapters/01-what-this-book-builds.md:519 | model.py | 147–159 | 152–164 | same | @staticmethod | @staticmethod |
| chapters/01-what-this-book-builds.md:519 | model.py | 782–793 | 820–836 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/01-what-this-book-builds.md:520 | model.py | 860–863 | 903–906 | same | if targets is None: | if targets is None: |
| chapters/01-what-this-book-builds.md:520 | model.py | 1059–1116 | 1104–1168 | same | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/01-what-this-book-builds.md:520 | model.py | 1083–1109 | 1132–1161 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/01-what-this-book-builds.md:520 | model.py | 1083–1109 | 1132–1161 | sweep | d' 860–863; 'AnuLM.generate' 1059–1116 (loop 1083–1109). |  |
| chapters/01-what-this-book-builds.md:522 | sample.py | 26–90 | 26–93 | same | def main(): | def main(): |
| chapters/01-what-this-book-builds.md:522 | sample.py | 31–32 | 34–35 | same | p.add_argument("--temperature", type=float, default=0.8) | p.add_argument("--temperature", type=float, default=0.8) |
| chapters/01-what-this-book-builds.md:522 | sample.py | 31–32 | 34–35 | sweep | - 'sample.py': 'main' 26–90 (defaults 31–32, gen |  |
| chapters/01-what-this-book-builds.md:522 | sample.py | 66–74 | 69–77 | same | else: | else: |
| chapters/01-what-this-book-builds.md:522 | sample.py | 66–74 | 69–77 | sweep | y': 'main' 26–90 (defaults 31–32, generation 66–74). |  |
| chapters/01-what-this-book-builds.md:523 | quickstart.py | 170–195 | 173–198 | same | def train(state: dict) -> int: | def train(state: dict) -> int: |
| chapters/01-what-this-book-builds.md:523 | quickstart.py | 182–189 | 185–192 | edited | rc = run([sys.executable, "-u", "train.py", "--preset", "30b", "--step | rc = run([sys.executable, "-u", "-m", "anulm", "train", "--preset", "3 |
| chapters/02-from-counts-to-transformers.md:82 | model.py | 786–791 | 829–834 | same | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) |
| chapters/02-from-counts-to-transformers.md:96 | model.py | 845–857 | 888–900 | same | aux_total = None | aux_total = None |
| chapters/02-from-counts-to-transformers.md:265 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/02-from-counts-to-transformers.md:280 | model.py | 441–447 | 463–469 | same | if cache is not None: | if cache is not None: |
| chapters/02-from-counts-to-transformers.md:340 | bidirectional.py | 78–89 | 82–93 | same | masked = torch.rand(x.shape, device=x.device) < a.mask_frac | masked = torch.rand(x.shape, device=x.device) < a.mask_frac |
| chapters/02-from-counts-to-transformers.md:67 | bigram_teacher.py | 65–70 |  | reattributed | 'teacher_kl' is in bigram_teacher.py, not docs/RESULTS.md |  |
| chapters/02-from-counts-to-transformers.md:334 | model.py | 452 |  | reattributed | 'GQAttention.forward' is in model.py, not bidirectional.py |  |
| chapters/02-from-counts-to-transformers.md:334 | model.py | 452 | 474 | same | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star |
| chapters/02-from-counts-to-transformers.md:416 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/02-from-counts-to-transformers.md:416 | model.py | 441–447 | 463–469 | same | if cache is not None: | if cache is not None: |
| chapters/02-from-counts-to-transformers.md:416 | model.py | 452 | 474 | same | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star |
| chapters/02-from-counts-to-transformers.md:417 | model.py | 786–791 | 829–834 | same | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) | self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size) |
| chapters/02-from-counts-to-transformers.md:417 | model.py | 845–857 | 888–900 | same | aux_total = None | aux_total = None |
| chapters/02-from-counts-to-transformers.md:418 | bigram_teacher.py | 42–51 | 42–51 | sweep | er.py': 'build_bigram_teacher' 34–62 (counts 42–51), |  |
| chapters/02-from-counts-to-transformers.md:420 | bidirectional.py | 56–101 | 56–105 | same | def adapt(a): | def adapt(a): |
| chapters/02-from-counts-to-transformers.md:420 | bidirectional.py | 78–89 | 82–93 | same | masked = torch.rand(x.shape, device=x.device) < a.mask_frac | masked = torch.rand(x.shape, device=x.device) < a.mask_frac |
| chapters/02-from-counts-to-transformers.md:420 | bidirectional.py | 78–89 | 82–93 | sweep | : 'bidir_hidden' 40–53; 'adapt' 56–101 (MNTP 78–89). |  |
| chapters/02-from-counts-to-transformers.md:422 | train.py | 199–208 | 241–254 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/02-from-counts-to-transformers.md:422 | train.py | 310–337 | 451–478 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/03-the-one-job.md:22 | train.py | 199–205 | 241–251 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/03-the-one-job.md:47 | model.py | 867–873 | 910–916 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/03-the-one-job.md:173 | model.py | 804–808 | 847–851 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/03-the-one-job.md:183 | model.py | 810–816 | 853–859 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/03-the-one-job.md:279 | model.py | 1132–1141 | 1184–1193 | same | x = torch.randint(0, cfg.vocab_size, (2, 64)) | x = torch.randint(0, cfg.vocab_size, (2, 64)) |
| chapters/03-the-one-job.md:311 | train.py | 740–742 | 903–905 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/03-the-one-job.md:322 | train.py | 509–514 | 660–675 | same | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 |
| chapters/03-the-one-job.md:322 | train.py | 660–675 |  | EXCERPT | if memmapped: | quoted line not in the new range |
| chapters/03-the-one-job.md:339 | train.py | 740 | 903 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/03-the-one-job.md:418 | model.py | 804–808 | 847–851 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/03-the-one-job.md:418 | model.py | 810–816 | 853–859 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/03-the-one-job.md:419 | model.py | 867–873 | 910–916 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/03-the-one-job.md:419 | model.py | 1119–1144 | 1171–1196 | same | if __name__ == "__main__": | if __name__ == "__main__": |
| chapters/03-the-one-job.md:420 | train.py | 199–208 | 241–254 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/03-the-one-job.md:420 | train.py | 509–514 | 660–675 | same | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 |
| chapters/03-the-one-job.md:421 | train.py | 740–742 | 903–905 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/03-the-one-job.md:421 | train.py | 740–742 | 903–905 | sweep | 740–742 (bit |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:73 | model.py | 412–416 | 430–434 | same | # One fused projection: (H + 2*Hkv) * Dh, exactly as the 30B does. | # One fused projection: (H + 2*Hkv) * Dh, exactly as the 30B does. |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:110 | model.py | 428–434 | 450–456 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:175 | model.py | 454–465 | 476–487 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:237 | model.py | 327–341 | 339–355 | same | def rotate_half(x): | def rotate_half(x): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:265 | model.py | 219–223 | 227–232 | same | def forward(self, x): | def forward(self, x): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:307 | model.py | 441–448 | 463–470 | same | if cache is not None: | if cache is not None: |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:62 | model.py | 432–434 |  | reattributed | 'GQAttention.forward' is in model.py, not None |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:62 | model.py | 432–434 | 454–456 | same | q = q.transpose(1, 2)                         # (B, H,   S, Dh) | q = q.transpose(1, 2)                         # (B, H,   S, Dh) |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:192 | model.py | 366 | 380 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:228 | model.py | 464 |  | reattributed | quoted code is in model.py, not None |  |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:228 | model.py | 464 | 486 | same | y = y.transpose(1, 2).contiguous().view(B, S, -1) | y = y.transpose(1, 2).contiguous().view(B, S, -1) |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:428 | model.py | 219–223 | 227–232 | same | def forward(self, x): | def forward(self, x): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:428 | model.py | 327–329 | 339–343 | same | def rotate_half(x): | def rotate_half(x): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:429 | model.py | 332–341 | 346–355 | same | def apply_rope(q, k, cos, sin): | def apply_rope(q, k, cos, sin): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:429 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:429 | model.py | 412–416 | 430–434 | same | # One fused projection: (H + 2*Hkv) * Dh, exactly as the 30B does. | # One fused projection: (H + 2*Hkv) * Dh, exactly as the 30B does. |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:430 | model.py | 366 | 380 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:430 | model.py | 428–434 | 450–456 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:430 | model.py | 441–448 | 463–470 | same | if cache is not None: | if cache is not None: |
| chapters/04-tensors-shapes-and-the-matrix-multiply.md:430 | model.py | 454–465 | 476–487 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/05-softmax-temperature-and-cross-entropy.md:77 | model.py | 589–591 | 617–619 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/05-softmax-temperature-and-cross-entropy.md:94 | model.py | 868–870 | 911–913 | same | loss = F.cross_entropy( | loss = F.cross_entropy( |
| chapters/05-softmax-temperature-and-cross-entropy.md:150 | train.py | 498–503 | 649–654 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/05-softmax-temperature-and-cross-entropy.md:172 | model.py | 1091–1105 | ? | UNMAPPED | logits = logits[:, -1, :].float() |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:172 | model.py | 1091 | ? | EXCERPT | header range unresolved: AnuLM.generate |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:218 | api.py | 387–389 | 399–401 | edited | self.eng.model.generate(x, self.max_tokens, temperature=max(self.tempe | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/05-softmax-temperature-and-cross-entropy.md:218 | api.py | 387 | 399 | edited-quote | self.eng.model.generate(x, self.max_tokens, temperature=max(self.tempe | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/05-softmax-temperature-and-cross-entropy.md:249 | model.py | 981–992 | 1026–1037 | same | scores = logits[0, -1].float() | scores = logits[0, -1].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:379 | model.py | 889–901 | 932–944 | same | def _chunked_loss(self, x, targets): | def _chunked_loss(self, x, targets): |
| chapters/05-softmax-temperature-and-cross-entropy.md:161 | model.py | 869 |  | reattributed | quoted code is in model.py, not train.py |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:161 | model.py | 869 | 912 | same | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ |
| chapters/05-softmax-temperature-and-cross-entropy.md:163 | model.py | 1091 |  | override | decided in repin-overrides.tsv |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:163 | model.py | 1091 | 1140 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:226 | api.py | 187 | 192 | same | temperature: Optional[float] = Field(None, ge=0, le=2, description="De | temperature: Optional[float] = Field(None, ge=0, le=2, description="De |
| chapters/05-softmax-temperature-and-cross-entropy.md:229 | sample.py | 31–32 |  | override | decided in repin-overrides.tsv |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:229 | sample.py | 31–32 | 34–35 | same | p.add_argument("--temperature", type=float, default=0.8) | p.add_argument("--temperature", type=float, default=0.8) |
| chapters/05-softmax-temperature-and-cross-entropy.md:238 | model.py | 1102-1103 |  | override | decided in repin-overrides.tsv |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:238 | model.py | 1102–1103 | 1154–1155 | reindented | v, _ = torch.topk(logits, min(top_k, logits.size(-1))) | v, _ = torch.topk(logits, min(top_k, logits.size(-1))) |
| chapters/05-softmax-temperature-and-cross-entropy.md:470 | model.py | 589–591 | 617–619 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/05-softmax-temperature-and-cross-entropy.md:470 | model.py | 868–870 | 911–913 | same | loss = F.cross_entropy( | loss = F.cross_entropy( |
| chapters/05-softmax-temperature-and-cross-entropy.md:471 | model.py | 889–898 | 932–941 | same | def _chunked_loss(self, x, targets): | def _chunked_loss(self, x, targets): |
| chapters/05-softmax-temperature-and-cross-entropy.md:471 | model.py | 900–901 | 943–944 | same | def _ce_sum(self, h, t): | def _ce_sum(self, h, t): |
| chapters/05-softmax-temperature-and-cross-entropy.md:472 | model.py | 981–992 | 1026–1037 | same | scores = logits[0, -1].float() | scores = logits[0, -1].float() |
| chapters/05-softmax-temperature-and-cross-entropy.md:472 | model.py | 1091–1105 | ? | UNMAPPED | logits = logits[:, -1, :].float() |  |
| chapters/05-softmax-temperature-and-cross-entropy.md:473 | api.py | 387–389 | 399–401 | edited | self.eng.model.generate(x, self.max_tokens, temperature=max(self.tempe | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/05-softmax-temperature-and-cross-entropy.md:474 | api.py | 187 | 192 | same | temperature: Optional[float] = Field(None, ge=0, le=2, description="De | temperature: Optional[float] = Field(None, ge=0, le=2, description="De |
| chapters/05-softmax-temperature-and-cross-entropy.md:475 | sample.py | 31–32 | 34–35 | same | p.add_argument("--temperature", type=float, default=0.8) | p.add_argument("--temperature", type=float, default=0.8) |
| chapters/05-softmax-temperature-and-cross-entropy.md:476 | train.py | 498–503 | 649–654 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/06-gradients-and-backpropagation.md:84 | model.py | 226–236 | 235–246 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/06-gradients-and-backpropagation.md:167 | model.py | 775–779 | 813–817 | same | x = x + self.attn(self.input_layernorm(x), cos, sin, | x = x + self.attn(self.input_layernorm(x), cos, sin, |
| chapters/06-gradients-and-backpropagation.md:196 | train.py | 689–707 | 852–870 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/06-gradients-and-backpropagation.md:236 | train.py | 341 | 482 | def | 'evaluate' | def evaluate(model, data, args, iters: int \| None = None) -> tuple[flo |
| chapters/06-gradients-and-backpropagation.md:236 | model.py | 1059 |  | reattributed | 'AnuLM.generate' is in model.py, not train.py |  |
| chapters/06-gradients-and-backpropagation.md:236 | model.py | 1059 | 1104 | same | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/06-gradients-and-backpropagation.md:238 | model.py | 858 |  | reattributed | quoted code is in model.py, not train.py |  |
| chapters/06-gradients-and-backpropagation.md:238 | model.py | 858 | 901 | same | self.aux_loss = aux_total.detach() if aux_total is not None else None | self.aux_loss = aux_total.detach() if aux_total is not None else None |
| chapters/06-gradients-and-backpropagation.md:438 | model.py | 226–236 | 235–246 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/06-gradients-and-backpropagation.md:438 | model.py | 775–779 | 813–817 | same | x = x + self.attn(self.input_layernorm(x), cos, sin, | x = x + self.attn(self.input_layernorm(x), cos, sin, |
| chapters/06-gradients-and-backpropagation.md:439 | model.py | 858 | 901 | same | self.aux_loss = aux_total.detach() if aux_total is not None else None | self.aux_loss = aux_total.detach() if aux_total is not None else None |
| chapters/06-gradients-and-backpropagation.md:439 | model.py | 1058–1059 | 1103–1104 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/06-gradients-and-backpropagation.md:440 | train.py | 340–341 | 481–482 | edited | @torch.no_grad() | @torch.no_grad() |
| chapters/06-gradients-and-backpropagation.md:440 | train.py | 689–707 | 852–870 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:76 | train.py | 310–337 | 451–478 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:76 | train.py | 328 | 469 | edited-quote | from muon import build_optimizers | from anulm.muon import build_optimizers |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:204 | train.py | 248–269 | 379–404 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:249 | train.py | 686–687 | 849–850 | same | lr = lr_at(step, args, start_step) | lr = lr_at(step, args, start_step) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:256 | train.py | 285–288 | 420–424 | same | def set_lr_scale(self, scale: float): | def set_lr_scale(self, scale: float): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:308 | test_model.py | 1612 | 1612 | edited-quote | from train import lr_at | from anulm.training.train import lr_at |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:334 | train.py | 708–712 | 871–875 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:459 | muon.py | 121–129 | 125–133 | same | by_shape: dict = {} | by_shape: dict = {} |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:21 | train.py | 412–413 | 559–560 | same | p.add_argument("--weight-decay", type=float, default=0.1) | p.add_argument("--weight-decay", type=float, default=0.1) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:191 | muon.py | 148–161 | 152–165 | same | muon_params, decay, no_decay = [], [], [] | muon_params, decay, no_decay = [], [], [] |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:193 | muon.py | 13–14 |  | override | decided in repin-overrides.tsv |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:456 | muon.py | 59 |  | reattributed | 'zeropower_batched' is defined there |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:563 | train.py | 248–269 | 379–404 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:563 | train.py | 272–307 | 407–448 | same | class OptimizerSet: | class OptimizerSet: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:564 | train.py | 285–288 | 420–424 | same | def set_lr_scale(self, scale: float): | def set_lr_scale(self, scale: float): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:564 | train.py | 285–288 | 420–424 | sweep | 285–288); 'm |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:564 | train.py | 310–337 | 451–478 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:564 | train.py | 412–413 | 559–560 | same | p.add_argument("--weight-decay", type=float, default=0.1) | p.add_argument("--weight-decay", type=float, default=0.1) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:564 | train.py | 412–413 | 559–560 | sweep | ); 'make_optimizer' 310–337; 'main' defaults 412–413, sch |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:565 | train.py | 686–687 | 849–850 | same | lr = lr_at(step, args, start_step) | lr = lr_at(step, args, start_step) |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:565 | train.py | 686–687 | 849–850 | sweep | 686–687, cli |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:565 | train.py | 708–712 | 871–875 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:565 | train.py | 708–712 | 871–875 | sweep | 686–687, clip and step 708–712. |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:566 | muon.py | 13–14 | 13–14 | sweep | - 'muon.py': module docstring 13–14; 'ze |  |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:567 | muon.py | 85–137 | 85–141 | same | class Muon(torch.optim.Optimizer): | class Muon(torch.optim.Optimizer): |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:567 | muon.py | 121–129 | 125–133 | same | by_shape: dict = {} | by_shape: dict = {} |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:568 | muon.py | 140–178 | 144–182 | same | def build_optimizers(model, muon_lr: float = 0.02, adamw_lr: float = 3 | def build_optimizers(model, muon_lr: float = 0.02, adamw_lr: float = 3 |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:568 | muon.py | 148–161 | 152–165 | same | muon_params, decay, no_decay = [], [], [] | muon_params, decay, no_decay = [], [], [] |
| chapters/07-optimisation-adamw-schedules-and-clipping.md:568 | muon.py | 148–161 | 152–165 | sweep | 'build_optimizers' 140–178 (split 148–161). |  |
| chapters/08-bytes-as-tokens.md:98 | model.py | 42 | 47 | same | vocab_size: int = 259           # 256 bytes + BOS + EOS + PAD (byte mo | vocab_size: int = 259           # 256 bytes + BOS + EOS + PAD (byte mo |
| chapters/08-bytes-as-tokens.md:119 | sample.py | 67–69 | 70–72 | same | # Byte-level: the prompt is its own UTF-8 encoding. 257 is EOS. | # Byte-level: the prompt is its own UTF-8 encoding. 257 is EOS. |
| chapters/08-bytes-as-tokens.md:270 | test_model.py | 1415 | 1415 | edited-quote | from bpe import DOC_SEP | from anulm.bpe import DOC_SEP |
| chapters/08-bytes-as-tokens.md:270 | test_model.py | 1416 | 1416 | edited-quote | from train import BYTE_EOS, load_bytes | from anulm.training.train import BYTE_EOS, load_bytes |
| chapters/08-bytes-as-tokens.md:338 | train.py | 505–509 | 656–660 | same | # --- data ----------------------------------------------------------- | # --- data ----------------------------------------------------------- |
| chapters/08-bytes-as-tokens.md:348 | train.py | 740–742 | 903–905 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/08-bytes-as-tokens.md:393 | sample.py | 82–90 | 85–93 | same | # Two ways a byte model produces something that is not a byte string: | # Two ways a byte model produces something that is not a byte string: |
| chapters/08-bytes-as-tokens.md:137 | train.py | 28 | 28 | edited | from bpe import DOC_SEP | from anulm.bpe import DOC_SEP |
| chapters/08-bytes-as-tokens.md:357 | train.py | 509 |  | override | decided in repin-overrides.tsv |  |
| chapters/08-bytes-as-tokens.md:357 | train.py | 509 | 660 | same | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 | tokenizer_path, vocab_size, args.bytes_per_token = None, 259, 1.0 |
| chapters/08-bytes-as-tokens.md:409 | model.py | 1107–1115 | 1159–1167 | same | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) |
| chapters/08-bytes-as-tokens.md:435 | train.py | 531 | 692 | same | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") |
| chapters/08-bytes-as-tokens.md:492 | model.py | 42 | 47 | same | vocab_size: int = 259           # 256 bytes + BOS + EOS + PAD (byte mo | vocab_size: int = 259           # 256 bytes + BOS + EOS + PAD (byte mo |
| chapters/08-bytes-as-tokens.md:492 | model.py | 1107–1115 | 1159–1167 | same | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) | nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt) |
| chapters/08-bytes-as-tokens.md:494 | train.py | 28 | 28 | edited | from bpe import DOC_SEP | from anulm.bpe import DOC_SEP |
| chapters/08-bytes-as-tokens.md:496 | train.py | 505–509 | 656–660 | same | # --- data ----------------------------------------------------------- | # --- data ----------------------------------------------------------- |
| chapters/08-bytes-as-tokens.md:496 | train.py | 531 | 692 | same | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") | print(f"byte data: {n_docs} documents, EOS ({BYTE_EOS}) after each") |
| chapters/08-bytes-as-tokens.md:496 | train.py | 740–742 | 903–905 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/08-bytes-as-tokens.md:498 | sample.py | 67–69 | 70–72 | same | # Byte-level: the prompt is its own UTF-8 encoding. 257 is EOS. | # Byte-level: the prompt is its own UTF-8 encoding. 257 is EOS. |
| chapters/08-bytes-as-tokens.md:498 | sample.py | 82–90 | 85–93 | same | # Two ways a byte model produces something that is not a byte string: | # Two ways a byte model produces something that is not a byte string: |
| chapters/08-bytes-as-tokens.md:500 | test_model.py | 1413–1431 | 1413–1431 | sweep | 1413–1431. |  |
| chapters/09-byte-level-bpe.md:83 | bpe.py | 51–67 | 64–83 | same | class BPE: | class BPE: |
| chapters/09-byte-level-bpe.md:157 | bpe.py | 71–92 | 87–111 | same | @classmethod | @classmethod |
| chapters/09-byte-level-bpe.md:189 | bpe.py | 94–107 | 113–126 | same | heap = [(-c, p) for p, c in pair_counts.items()] | heap = [(-c, p) for p, c in pair_counts.items()] |
| chapters/09-byte-level-bpe.md:219 | bpe.py | 109–143 | 128–162 | same | touched: set[tuple[int, int]] = set() | touched: set[tuple[int, int]] = set() |
| chapters/09-byte-level-bpe.md:277 | bpe.py | 147–170 | 166–189 | same | def _encode_chunk(self, chunk: bytes) -> list[int]: | def _encode_chunk(self, chunk: bytes) -> list[int]: |
| chapters/09-byte-level-bpe.md:341 | bpe.py | 172–190 | 191–212 | same | def encode(self, text: str) -> list[int]: | def encode(self, text: str) -> list[int]: |
| chapters/09-byte-level-bpe.md:381 | test_model.py | 827 | 827 | edited-quote | from bpe import BPE, DOC_SEP | from anulm.bpe import BPE, DOC_SEP |
| chapters/09-byte-level-bpe.md:381 | test_model.py | 828 | 828 | edited-quote | from train import FALLBACK | from anulm.training.train import FALLBACK |
| chapters/09-byte-level-bpe.md:454 | bpe.py | 315–323 | 347–355 | same | def _stats(tok, text): | def _stats(tok, text): |
| chapters/09-byte-level-bpe.md:213 | bpe.py | 101–102 |  | override | decided in repin-overrides.tsv |  |
| chapters/09-byte-level-bpe.md:213 | bpe.py | 101–102 | 120–121 | same | if -neg != pair_counts.get(pair, 0): | if -neg != pair_counts.get(pair, 0): |
| chapters/09-byte-level-bpe.md:405 | test_model.py | 1342–1346 |  | override | decided in repin-overrides.tsv |  |
| chapters/09-byte-level-bpe.md:442 | bpe.py | 194–213 |  | reattributed | quoted code is in bpe.py, not None |  |
| chapters/09-byte-level-bpe.md:442 | bpe.py | 194–213 | 221–243 | same | # The format id the project wrote before it was renamed on 2026-09-18. | # The format id the project wrote before it was renamed on 2026-09-18. |
| chapters/09-byte-level-bpe.md:443 | bpe.py | 264–312 |  | reattributed | 'cmd_encode' is in bpe.py, not None |  |
| chapters/09-byte-level-bpe.md:443 | bpe.py | 264–312 | 296–344 | same | def cmd_encode(args): | def cmd_encode(args): |
| chapters/09-byte-level-bpe.md:517 | docs/RESULTS.md | 101–102 |  | override | decided in repin-overrides.tsv |  |
| chapters/09-byte-level-bpe.md:579 | bpe.py | 51–67 | 64–83 | same | class BPE: | class BPE: |
| chapters/09-byte-level-bpe.md:579 | bpe.py | 51–67 | 64–83 | sweep | 51–67, 'BP |  |
| chapters/09-byte-level-bpe.md:579 | bpe.py | 71–143 | 87–162 | same | @classmethod | @classmethod |
| chapters/09-byte-level-bpe.md:579 | bpe.py | 147–170 | 166–189 | same | def _encode_chunk(self, chunk: bytes) -> list[int]: | def _encode_chunk(self, chunk: bytes) -> list[int]: |
| chapters/09-byte-level-bpe.md:580 | bpe.py | 172–190 | 191–212 | same | def encode(self, text: str) -> list[int]: | def encode(self, text: str) -> list[int]: |
| chapters/09-byte-level-bpe.md:580 | bpe.py | 194–213 | 221–243 | same | # The format id the project wrote before it was renamed on 2026-09-18. | # The format id the project wrote before it was renamed on 2026-09-18. |
| chapters/09-byte-level-bpe.md:581 | bpe.py | 264–312 | 296–344 | same | def cmd_encode(args): | def cmd_encode(args): |
| chapters/09-byte-level-bpe.md:581 | bpe.py | 264–312 | 296–344 | sweep | 264–312, '_s |  |
| chapters/09-byte-level-bpe.md:581 | bpe.py | 315–323 | 347–355 | same | def _stats(tok, text): | def _stats(tok, text): |
| chapters/10-embeddings-and-the-output-table.md:57 | model.py | 782–794 | 820–837 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/10-embeddings-and-the-output-table.md:98 | model.py | 804–816 | 847–859 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/10-embeddings-and-the-output-table.md:190 | model.py | 841 | 884 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| chapters/10-embeddings-and-the-output-table.md:196 | model.py | 911–915 | 954–958 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/10-embeddings-and-the-output-table.md:206 | model.py | 935–938 | 980–983 | same | def _head(self, h): | def _head(self, h): |
| chapters/10-embeddings-and-the-output-table.md:310 | model.py | 369–390 | 383–404 | same | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: |
| chapters/10-embeddings-and-the-output-table.md:370 | model.py | 904–909 | 947–952 | same | def _compose(self, W): | def _compose(self, W): |
| chapters/10-embeddings-and-the-output-table.md:383 | model.py | 916–933 | 959–978 | same | if self.training: | if self.training: |
| chapters/10-embeddings-and-the-output-table.md:417 | model.py | 940–956 | 985–1001 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/10-embeddings-and-the-output-table.md:79 | model.py | 793 | 836 | same | self.lm_head.weight = self.embed_tokens.weight | self.lm_head.weight = self.embed_tokens.weight |
| chapters/10-embeddings-and-the-output-table.md:80 | model.py | 112 | 117 | same | tie_word_embeddings: bool = False   # Sarvam unties both models | tie_word_embeddings: bool = False   # Sarvam unties both models |
| chapters/10-embeddings-and-the-output-table.md:117 | model.py | 113 | 118 | same | initializer_range: float = 0.02 | initializer_range: float = 0.02 |
| chapters/10-embeddings-and-the-output-table.md:248 | train.py | 310–337 | 451–478 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/10-embeddings-and-the-output-table.md:250 | train.py | 412 |  | reattributed | quoted code is in train.py, not _v2_run.cmd |  |
| chapters/10-embeddings-and-the-output-table.md:250 | train.py | 412 | 559 | same | p.add_argument("--weight-decay", type=float, default=0.1) | p.add_argument("--weight-decay", type=float, default=0.1) |
| chapters/10-embeddings-and-the-output-table.md:338 | model.py | 376 |  | override | decided in repin-overrides.tsv |  |
| chapters/10-embeddings-and-the-output-table.md:338 | model.py | 376 | 390 | same | coef = [{i: 1.0} for i in range(256)] | coef = [{i: 1.0} for i in range(256)] |
| chapters/10-embeddings-and-the-output-table.md:516 | model.py | 112–113 | 117–118 | same | tie_word_embeddings: bool = False   # Sarvam unties both models | tie_word_embeddings: bool = False   # Sarvam unties both models |
| chapters/10-embeddings-and-the-output-table.md:516 | model.py | 114–119 | 119–124 | same | # Merge-tree reparameterisation of the token tables ("in", "out", "bot | # Merge-tree reparameterisation of the token tables ("in", "out", "bot |
| chapters/10-embeddings-and-the-output-table.md:517 | model.py | 369–390 | 383–404 | same | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: |
| chapters/10-embeddings-and-the-output-table.md:518 | model.py | 782–808 | 820–851 | same | class AnuLM(nn.Module): | class AnuLM(nn.Module): |
| chapters/10-embeddings-and-the-output-table.md:518 | model.py | 810–816 | 853–859 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/10-embeddings-and-the-output-table.md:519 | model.py | 841 | 884 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| chapters/10-embeddings-and-the-output-table.md:519 | model.py | 904–909 | 947–952 | same | def _compose(self, W): | def _compose(self, W): |
| chapters/10-embeddings-and-the-output-table.md:519 | model.py | 911–929 | 954–972 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/10-embeddings-and-the-output-table.md:520 | model.py | 931–933 | 974–978 | same | def train(self, mode: bool = True): | def train(self, mode: bool = True): |
| chapters/10-embeddings-and-the-output-table.md:520 | model.py | 935–938 | 980–983 | same | def _head(self, h): | def _head(self, h): |
| chapters/10-embeddings-and-the-output-table.md:520 | model.py | 940–956 | 985–1001 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/10-embeddings-and-the-output-table.md:521 | train.py | 310–337 | 451–478 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/11-attention.md:275 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/11-attention.md:292 | model.py | 405–424 | 423–442 | same | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): |
| chapters/11-attention.md:328 | model.py | 426–447 | 444–469 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/11-attention.md:360 | model.py | 448–465 | 470–487 | same | T = k.shape[2] | T = k.shape[2] |
| chapters/11-attention.md:396 | model.py | 344–360 | 358–374 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/11-attention.md:506 | model.py | 213–223 | 218–232 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/11-attention.md:323 | model.py | 279 | 291 | same | self.attn_mscale = self.table_mscale = 1.0 | self.attn_mscale = self.table_mscale = 1.0 |
| chapters/11-attention.md:389 | model.py | 366 | 380 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/11-attention.md:418 | model.py | 353 |  | override | decided in repin-overrides.tsv |  |
| chapters/11-attention.md:418 | model.py | 353 | 367 | same | if window is None and (start == 0 or S == 1): | if window is None and (start == 0 or S == 1): |
| chapters/11-attention.md:424 | model.py | 355–359 |  | override | decided in repin-overrides.tsv |  |
| chapters/11-attention.md:424 | model.py | 355–359 | 369–373 | same | q_pos = torch.arange(start, start + S, device=device)[:, None] | q_pos = torch.arange(start, start + S, device=device)[:, None] |
| chapters/11-attention.md:465 | test_model.py | 421–423 |  | override | decided in repin-overrides.tsv |  |
| chapters/11-attention.md:502 | model.py | 436–438 |  | reattributed | 'GQAttention.forward' is in model.py, not test_model.py |  |
| chapters/11-attention.md:502 | model.py | 436–438 | 458–460 | same | if self.use_qk_norm:                          # before RoPE, as in the | if self.use_qk_norm:                          # before RoPE, as in the |
| chapters/11-attention.md:523 | model.py | 216 |  | override | decided in repin-overrides.tsv |  |
| chapters/11-attention.md:523 | model.py | 216 | 224 | same | self.weight = nn.Parameter(torch.ones(dim)) | self.weight = nn.Parameter(torch.ones(dim)) |
| chapters/11-attention.md:571 | model.py | 436 |  | reattributed | quoted code is in model.py, not None |  |
| chapters/11-attention.md:571 | model.py | 436 | 458 | same | if self.use_qk_norm:                          # before RoPE, as in the | if self.use_qk_norm:                          # before RoPE, as in the |
| chapters/11-attention.md:770 | model.py | 213–223 | 218–232 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/11-attention.md:770 | model.py | 279 | 291 | same | self.attn_mscale = self.table_mscale = 1.0 | self.attn_mscale = self.table_mscale = 1.0 |
| chapters/11-attention.md:771 | model.py | 344–360 | 358–374 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/11-attention.md:771 | model.py | 366 | 380 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/11-attention.md:771 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/11-attention.md:772 | model.py | 405–424 | 423–442 | same | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): |
| chapters/11-attention.md:772 | model.py | 436–438 | 458–460 | same | if self.use_qk_norm:                          # before RoPE, as in the | if self.use_qk_norm:                          # before RoPE, as in the |
| chapters/11-attention.md:772 | model.py | 448–465 | 470–487 | same | T = k.shape[2] | T = k.shape[2] |
| chapters/12-causal-masking.md:71 | model.py | 127–129 | 132–134 | same | if self.sliding_window is not None: | if self.sliding_window is not None: |
| chapters/12-causal-masking.md:85 | model.py | 344–360 | 358–374 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/12-causal-masking.md:143 | model.py | 448–453 | 470–475 | same | T = k.shape[2] | T = k.shape[2] |
| chapters/12-causal-masking.md:187 | model.py | 1083–1090 | 1132–1139 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/12-causal-masking.md:204 | modeling_anulm.py | 249–255 | 274–282 | same | def prepare_inputs_for_generation(self, input_ids, past_key_values=Non | def prepare_inputs_for_generation(self, input_ids, past_key_values=Non |
| chapters/12-causal-masking.md:225 | model.py | 767–769 | 805–807 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/12-causal-masking.md:155 | model.py | 540 |  | override | decided in repin-overrides.tsv |  |
| chapters/12-causal-masking.md:155 | model.py | 540 | 568 | same | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star |
| chapters/12-causal-masking.md:201 | model.py | 1109 | 1161 | same | pending = nxt | pending = nxt |
| chapters/12-causal-masking.md:262 | model.py | 716 | 749 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/12-causal-masking.md:262 | model.py | 743 | 777 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/12-causal-masking.md:509 | model.py | 127–129 | 132–134 | same | if self.sliding_window is not None: | if self.sliding_window is not None: |
| chapters/12-causal-masking.md:509 | model.py | 344–360 | 358–374 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/12-causal-masking.md:510 | model.py | 448–453 | 470–475 | same | T = k.shape[2] | T = k.shape[2] |
| chapters/12-causal-masking.md:510 | model.py | 540 | 568 | same | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star | mask = bidir_mask if bidir_mask is not None else _attn_mask(S, T, star |
| chapters/12-causal-masking.md:511 | model.py | 767–769 | 805–807 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/12-causal-masking.md:511 | model.py | 767–769 | 805–807 | sweep | 767–769, 'An |  |
| chapters/12-causal-masking.md:511 | model.py | 1083–1090 | 1132–1139 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/12-causal-masking.md:511 | model.py | 1109 | 1161 | same | pending = nxt | pending = nxt |
| chapters/12-causal-masking.md:512 | modeling_anulm.py | 249–255 | 274–282 | same | def prepare_inputs_for_generation(self, input_ids, past_key_values=Non | def prepare_inputs_for_generation(self, input_ids, past_key_values=Non |
| chapters/13-multi-head-and-gqa.md:31 | model.py | 139–142 | 144–147 | same | if self.attn == "gqa": | if self.attn == "gqa": |
| chapters/13-multi-head-and-gqa.md:49 | model.py | 405–420 | 423–438 | same | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): |
| chapters/13-multi-head-and-gqa.md:111 | model.py | 426–439 | 444–461 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/13-multi-head-and-gqa.md:148 | model.py | 441–448 | 463–470 | same | if cache is not None: | if cache is not None: |
| chapters/13-multi-head-and-gqa.md:197 | model.py | 454–465 | 476–487 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/13-multi-head-and-gqa.md:235 | model.py | 363–366 | 377–380 | same | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa |
| chapters/13-multi-head-and-gqa.md:248 | test_model.py | 1588 | 1588 | edited-quote | import model as M | from anulm import model as M |
| chapters/13-multi-head-and-gqa.md:75 | model.py | 424 |  | override | decided in repin-overrides.tsv |  |
| chapters/13-multi-head-and-gqa.md:75 | model.py | 424 | 442 | same | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) |
| chapters/13-multi-head-and-gqa.md:244 | model.py | 490–496 |  | override | decided in repin-overrides.tsv |  |
| chapters/13-multi-head-and-gqa.md:244 | model.py | 490–496 | 516–522 | same | self.kv_b_proj = nn.Linear( | self.kv_b_proj = nn.Linear( |
| chapters/13-multi-head-and-gqa.md:311 | model.py | 716 | 749 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/13-multi-head-and-gqa.md:311 | model.py | 743 | 777 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/13-multi-head-and-gqa.md:394 | model.py | 139–142 | 144–147 | same | if self.attn == "gqa": | if self.attn == "gqa": |
| chapters/13-multi-head-and-gqa.md:394 | model.py | 363–366 | 377–380 | same | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa | # GQA through SDPA's own 'enable_gqa' instead of repeat_interleave. Sa |
| chapters/13-multi-head-and-gqa.md:395 | model.py | 405–420 | 423–438 | same | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, rotary: Rotary): |
| chapters/13-multi-head-and-gqa.md:395 | model.py | 424 | 442 | same | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) |
| chapters/13-multi-head-and-gqa.md:395 | model.py | 426–465 | 444–487 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/13-multi-head-and-gqa.md:396 | train.py | 490–496 | 641–647 | same | p.add_argument("--native-gqa", action="store_true", | p.add_argument("--native-gqa", action="store_true", |
| chapters/14-rotary-positions.md:115 | model.py | 285–290 | 297–302 | same | @staticmethod | @staticmethod |
| chapters/14-rotary-positions.md:162 | model.py | 263–268 | 275–280 | same | def __init__(self, cfg: AnuLMConfig, dim: int): | def __init__(self, cfg: AnuLMConfig, dim: int): |
| chapters/14-rotary-positions.md:176 | model.py | 305–318 | 317–330 | same | def _grow(self, seq_len: int, device, dtype): | def _grow(self, seq_len: int, device, dtype): |
| chapters/14-rotary-positions.md:201 | model.py | 320–324 | 332–336 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| chapters/14-rotary-positions.md:243 | model.py | 327–329 | 339–343 | same | def rotate_half(x): | def rotate_half(x): |
| chapters/14-rotary-positions.md:251 | model.py | 332–341 | 346–355 | same | def apply_rope(q, k, cos, sin): | def apply_rope(q, k, cos, sin): |
| chapters/14-rotary-positions.md:157 | model.py | 787–789 |  | reattributed | 'AnuLM.__init__' is in model.py, not docs/RESULTS.md |  |
| chapters/14-rotary-positions.md:157 | model.py | 787–789 | 830–832 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/14-rotary-positions.md:158 | model.py | 844 |  | reattributed | 'AnuLM.forward' is in model.py, not docs/RESULTS.md |  |
| chapters/14-rotary-positions.md:158 | model.py | 844 | 887 | same | cos, sin = self.rotary(S, x.device, x.dtype, start=start) | cos, sin = self.rotary(S, x.device, x.dtype, start=start) |
| chapters/14-rotary-positions.md:159 | docs/RESULTS.md | 282–283 |  | override | decided in repin-overrides.tsv |  |
| chapters/14-rotary-positions.md:354 | model.py | 716 | 749 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/14-rotary-positions.md:354 | model.py | 743 | 777 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/14-rotary-positions.md:457 | model.py | 259–324 | 271–336 | same | class Rotary(nn.Module): | class Rotary(nn.Module): |
| chapters/14-rotary-positions.md:457 | model.py | 327–329 | 339–343 | same | def rotate_half(x): | def rotate_half(x): |
| chapters/14-rotary-positions.md:457 | model.py | 332–341 | 346–355 | same | def apply_rope(q, k, cos, sin): | def apply_rope(q, k, cos, sin): |
| chapters/14-rotary-positions.md:458 | model.py | 716 | 749 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/14-rotary-positions.md:458 | model.py | 743 | 777 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/14-rotary-positions.md:458 | model.py | 787–789 | 830–832 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/14-rotary-positions.md:458 | model.py | 844 | 887 | same | cos, sin = self.rotary(S, x.device, x.dtype, start=start) | cos, sin = self.rotary(S, x.device, x.dtype, start=start) |
| chapters/15-normalisation-and-the-residual-stream.md:73 | model.py | 213–223 | 218–232 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/15-normalisation-and-the-residual-stream.md:135 | model.py | 757–766 | 795–804 | same | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): |
| chapters/15-normalisation-and-the-residual-stream.md:152 | model.py | 771–779 | 809–817 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/15-normalisation-and-the-residual-stream.md:174 | model.py | 789–791 | 832–834 | same | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range |
| chapters/15-normalisation-and-the-residual-stream.md:182 | model.py | 857 | 900 | same | x = self.norm(x) | x = self.norm(x) |
| chapters/15-normalisation-and-the-residual-stream.md:269 | model.py | 804–808 | 847–851 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/15-normalisation-and-the-residual-stream.md:279 | model.py | 810–816 | 853–859 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/15-normalisation-and-the-residual-stream.md:67 | vision_encoder.py | 115 | 124 | same | nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim), nn.LayerN | nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim), nn.LayerN |
| chapters/15-normalisation-and-the-residual-stream.md:421 | model.py | 213–223 | 218–232 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/15-normalisation-and-the-residual-stream.md:421 | model.py | 757–766 | 795–804 | same | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): | def __init__(self, cfg: AnuLMConfig, layer_idx: int, rotary: Rotary): |
| chapters/15-normalisation-and-the-residual-stream.md:422 | model.py | 771–779 | 809–817 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/15-normalisation-and-the-residual-stream.md:422 | model.py | 771–779 | 809–817 | sweep | 771–779, 'An |  |
| chapters/15-normalisation-and-the-residual-stream.md:422 | model.py | 789–791 | 832–834 | same | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range | self.layers = nn.ModuleList([Block(cfg, i, self.rotary) for i in range |
| chapters/15-normalisation-and-the-residual-stream.md:422 | model.py | 804–808 | 847–851 | same | self.apply(self._init_weights) | self.apply(self._init_weights) |
| chapters/15-normalisation-and-the-residual-stream.md:422 | model.py | 810–816 | 853–859 | same | def _init_weights(self, m): | def _init_weights(self, m): |
| chapters/15-normalisation-and-the-residual-stream.md:423 | model.py | 857 | 900 | same | x = self.norm(x) | x = self.norm(x) |
| chapters/15-normalisation-and-the-residual-stream.md:424 | speech_encoder.py | 80–83 | 80–83 | sweep | - 'speech_encoder.py' 80–83 and |  |
| chapters/15-normalisation-and-the-residual-stream.md:424 | vision_encoder.py | 115 | 124 | same | nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim), nn.LayerN | nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim), nn.LayerN |
| chapters/15-normalisation-and-the-residual-stream.md:424 | vision_encoder.py | 115 | 124 | sweep | ch_encoder.py' 80–83 and 'vision_encoder.py' 115 (Lay |  |
| chapters/16-swiglu.md:90 | model.py | 226–236 | 235–246 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/16-swiglu.md:157 | model.py | 72–74 | 77–79 | same | moe_intermediate_size: int = 96 | moe_intermediate_size: int = 96 |
| chapters/16-swiglu.md:168 | model.py | 179–188 | 184–193 | same | cfg = dict( | cfg = dict( |
| chapters/16-swiglu.md:230 | model.py | 764–766 | 802–804 | same | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in both mod | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in both mod |
| chapters/16-swiglu.md:242 | model.py | 648–654 | 681–687 | same | self.experts = nn.ModuleList( | self.experts = nn.ModuleList( |
| chapters/16-swiglu.md:260 | model.py | 805–808 | 848–851 | same | # nanoGPT's scaled init for the residual output projections. | # nanoGPT's scaled init for the residual output projections. |
| chapters/16-swiglu.md:222 | model.py | 154 | 159 | same | intermediate_size=768,         # hidden*2 | intermediate_size=768,         # hidden*2 |
| chapters/16-swiglu.md:223 | model.py | 201 | 206 | same | intermediate_size=1536,        # hidden*4 | intermediate_size=1536,        # hidden*4 |
| chapters/16-swiglu.md:431 | model.py | 72–74 | 77–79 | same | moe_intermediate_size: int = 96 | moe_intermediate_size: int = 96 |
| chapters/16-swiglu.md:431 | model.py | 154 | 159 | same | intermediate_size=768,         # hidden*2 | intermediate_size=768,         # hidden*2 |
| chapters/16-swiglu.md:431 | model.py | 179–188 | 184–193 | same | cfg = dict( | cfg = dict( |
| chapters/16-swiglu.md:432 | model.py | 201 | 206 | same | intermediate_size=1536,        # hidden*4 | intermediate_size=1536,        # hidden*4 |
| chapters/16-swiglu.md:432 | model.py | 226–236 | 235–246 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/16-swiglu.md:432 | model.py | 648–654 | 681–687 | same | self.experts = nn.ModuleList( | self.experts = nn.ModuleList( |
| chapters/16-swiglu.md:433 | model.py | 764–766 | 802–804 | same | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in both mod | # Layer 0 dense, the rest MoE -- first_k_dense_replace = 1 in both mod |
| chapters/16-swiglu.md:433 | model.py | 805–808 | 848–851 | same | # nanoGPT's scaled init for the residual output projections. | # nanoGPT's scaled init for the residual output projections. |
| chapters/17-router.md:91 | model.py | 559–574 | 587–602 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/17-router.md:133 | model.py | 576–587 | 604–615 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/17-router.md:183 | model.py | 589–610 | 617–638 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/17-router.md:259 | model.py | 594–603 | 622–631 | same | if self.n_group > 1: | if self.n_group > 1: |
| chapters/17-router.md:325 | model.py | 133–137 | 138–142 | same | per_group = self.num_experts // self.n_group | per_group = self.num_experts // self.n_group |
| chapters/17-router.md:14 | model.py | 550–637 | 578–665 | same | class Router(nn.Module): | class Router(nn.Module): |
| chapters/17-router.md:336 | test_model.py | 594–603 |  | override | decided in repin-overrides.tsv |  |
| chapters/17-router.md:475 | model.py | 133–137 | 138–142 | same | per_group = self.num_experts // self.n_group | per_group = self.num_experts // self.n_group |
| chapters/17-router.md:475 | model.py | 559–574 | 587–602 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/17-router.md:476 | model.py | 576–587 | 604–615 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/17-router.md:476 | model.py | 589–610 | 617–638 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/17-router.md:476 | model.py | 594–603 | 622–631 | same | if self.n_group > 1: | if self.n_group > 1: |
| chapters/17-router.md:476 | model.py | 594–603 | 622–631 | sweep | ard' 576–587, 'Router._gate' 589–610 (groups 594–603). |  |
| chapters/17-router.md:481 | sarvam/sarvam-105b/modeling_sarvam_moe.py | 276–277 | 276–277 | sweep | 'sarvam/sarvam-105b/modeling_sarvam_moe.py' 276–277, |  |
| chapters/17-router.md:482 | sarvam/sarvam-105b/modeling_sarvam_moe.py | 305–313 | 305–313 | sweep | 305–313: the |  |
| chapters/18-dispatch-and-balance.md:67 | model.py | 669–687 | 702–720 | same | def _route_sparse(self, x_flat, topk_idx, topk_w): | def _route_sparse(self, x_flat, topk_idx, topk_w): |
| chapters/18-dispatch-and-balance.md:125 | model.py | 689–698 | 722–731 | same | def _route_dense(self, x_flat, topk_idx, topk_w): | def _route_dense(self, x_flat, topk_idx, topk_w): |
| chapters/18-dispatch-and-balance.md:144 | model.py | 734–749 | 768–783 | same | N, D = x_flat.shape | N, D = x_flat.shape |
| chapters/18-dispatch-and-balance.md:175 | model.py | 710–719 | 743–752 | same | # Keyed on the parameters' in-place version counters, so a | # Keyed on the parameters' in-place version counters, so a |
| chapters/18-dispatch-and-balance.md:267 | model.py | 612–617 | 640–645 | same | aux = None | aux = None |
| chapters/18-dispatch-and-balance.md:280 | train.py | 710–712 | 873–875 | same | opt.step() | opt.step() |
| chapters/18-dispatch-and-balance.md:288 | model.py | 1024–1044 | 1069–1089 | same | ratios = [] | ratios = [] |
| chapters/18-dispatch-and-balance.md:408 | export_hf.py | 271–279 | ? | UNMAPPED | # Small tensors stay float32 on disk AND must be loaded that way: the |  |
| chapters/18-dispatch-and-balance.md:408 | export_hf.py | 271 | ? | EXCERPT | header range unresolved: main |  |
| chapters/18-dispatch-and-balance.md:426 | model.py | 618–635 | 646–663 | same | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: |
| chapters/18-dispatch-and-balance.md:259 | docs/RESULTS.md | 713–714 |  | override | decided in repin-overrides.tsv |  |
| chapters/18-dispatch-and-balance.md:402 | modeling_anulm.py | 107–120 | 120–133 | same | model.rebuild_rotary(device=next(model.parameters()).device) | model.rebuild_rotary(device=next(model.parameters()).device) |
| chapters/18-dispatch-and-balance.md:403 | docs/DEVELOPING.md | 291 | 295 | same | changes which experts a token is routed to, and the output collapses i | changes which experts a token is routed to, and the output collapses i |
| chapters/18-dispatch-and-balance.md:584 | model.py | 612–617 | 640–645 | same | aux = None | aux = None |
| chapters/18-dispatch-and-balance.md:584 | model.py | 618–635 | 646–663 | same | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: | if self.seq_balance_alpha > 0 or self.router_z_alpha > 0: |
| chapters/18-dispatch-and-balance.md:584 | model.py | 669–687 | 702–720 | same | def _route_sparse(self, x_flat, topk_idx, topk_w): | def _route_sparse(self, x_flat, topk_idx, topk_w): |
| chapters/18-dispatch-and-balance.md:585 | model.py | 689–698 | 722–731 | same | def _route_dense(self, x_flat, topk_idx, topk_w): | def _route_dense(self, x_flat, topk_idx, topk_w): |
| chapters/18-dispatch-and-balance.md:585 | model.py | 710–719 | 743–752 | same | # Keyed on the parameters' in-place version counters, so a | # Keyed on the parameters' in-place version counters, so a |
| chapters/18-dispatch-and-balance.md:586 | model.py | 734–749 | 768–783 | same | N, D = x_flat.shape | N, D = x_flat.shape |
| chapters/18-dispatch-and-balance.md:586 | model.py | 734–749 | 768–783 | sweep | 734–749; 'An |  |
| chapters/18-dispatch-and-balance.md:586 | model.py | 1024–1044 | 1069–1089 | same | ratios = [] | ratios = [] |
| chapters/18-dispatch-and-balance.md:587 | train.py | 710–712 | 873–875 | same | opt.step() | opt.step() |
| chapters/18-dispatch-and-balance.md:587 | export_hf.py | 271–279 | ? | UNMAPPED | # Small tensors stay float32 on disk AND must be loaded that way: the |  |
| chapters/18-dispatch-and-balance.md:588 | modeling_anulm.py | 107–120 | 120–133 | same | model.rebuild_rotary(device=next(model.parameters()).device) | model.rebuild_rotary(device=next(model.parameters()).device) |
| chapters/18-dispatch-and-balance.md:588 | modeling_anulm.py | 107–120 | 120–133 | sweep | 'modeling_anulm.py' 107–120; 'do |  |
| chapters/18-dispatch-and-balance.md:588 | docs/DEVELOPING.md | 291 | 295 | same | changes which experts a token is routed to, and the output collapses i | changes which experts a token is routed to, and the output collapses i |
| chapters/18-dispatch-and-balance.md:592 | test_model.py | 495 | 495 | sweep | 495, 'gr |  |
| chapters/19-latent-attention.md:70 | model.py | 483–500 | 509–526 | same | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, |
| chapters/19-latent-attention.md:107 | model.py | 139–144 | 144–149 | same | if self.attn == "gqa": | if self.attn == "gqa": |
| chapters/19-latent-attention.md:126 | model.py | 504–518 | 532–546 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/19-latent-attention.md:153 | model.py | 529–543 | 557–571 | same | T = compressed.shape[1] | T = compressed.shape[1] |
| chapters/19-latent-attention.md:225 | model.py | 520–528 | 548–556 | same | if cache is not None: | if cache is not None: |
| chapters/19-latent-attention.md:15 | model.py | 472–543 | 494–571 | same | class MLAttention(nn.Module): | class MLAttention(nn.Module): |
| chapters/19-latent-attention.md:62 | model.py | 515–516 |  | override | decided in repin-overrides.tsv |  |
| chapters/19-latent-attention.md:62 | model.py | 515–516 | 543–544 | same | # RoPE touches only the rope slice; the latent stays position-free -- | # RoPE touches only the rope slice; the latent stays position-free -- |
| chapters/19-latent-attention.md:120 | model.py | 787 | 830 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/19-latent-attention.md:413 | model.py | 787 |  | reattributed | quoted code is in model.py, not docs/RESULTS.md |  |
| chapters/19-latent-attention.md:413 | model.py | 787 | 830 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/19-latent-attention.md:470 | model.py | 139–144 | 144–149 | same | if self.attn == "gqa": | if self.attn == "gqa": |
| chapters/19-latent-attention.md:471 | model.py | 483–500 | 509–526 | same | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, | self.q_proj = nn.Linear(cfg.hidden_size, cfg.n_head * self.q_head_dim, |
| chapters/19-latent-attention.md:471 | model.py | 483–500 | 509–526 | sweep | 483–500, 'ML |  |
| chapters/19-latent-attention.md:471 | model.py | 504–518 | 532–546 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| chapters/19-latent-attention.md:471 | model.py | 520–528 | 548–556 | same | if cache is not None: | if cache is not None: |
| chapters/19-latent-attention.md:471 | model.py | 529–543 | 557–571 | same | T = compressed.shape[1] | T = compressed.shape[1] |
| chapters/19-latent-attention.md:472 | model.py | 787 |  | override | decided in repin-overrides.tsv |  |
| chapters/19-latent-attention.md:472 | model.py | 787 | 830 | same | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim | rope_dim = cfg.qk_rope_head_dim if cfg.attn == "mla" else cfg.head_dim |
| chapters/19-latent-attention.md:477 | sarvam/sarvam-105b/modeling_sarvam_moe.py | 486–499 | 486–499 | sweep | 486–499 (opt |  |
| chapters/20-the-whole-model.md:105 | model.py | 783–808 | 826–851 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/20-the-whole-model.md:146 | model.py | 836–844 | 879–887 | same | if embeds is not None: | if embeds is not None: |
| chapters/20-the-whole-model.md:165 | model.py | 305–324 | 317–336 | same | def _grow(self, seq_len: int, device, dtype): | def _grow(self, seq_len: int, device, dtype): |
| chapters/20-the-whole-model.md:206 | model.py | 845–856 | 888–899 | same | aux_total = None | aux_total = None |
| chapters/20-the-whole-model.md:226 | model.py | 771–779 | 809–817 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/20-the-whole-model.md:249 | model.py | 429–465 | 451–487 | same | qkv = self.query_key_value(x) | qkv = self.query_key_value(x) |
| chapters/20-the-whole-model.md:304 | model.py | 226–236 | 235–246 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/20-the-whole-model.md:324 | model.py | 656–667 | 689–700 | same | def forward(self, x):                                     # (B, S, D) | def forward(self, x):                                     # (B, S, D) |
| chapters/20-the-whole-model.md:344 | model.py | 576–587 | 604–615 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/20-the-whole-model.md:367 | model.py | 589–637 | 617–665 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/20-the-whole-model.md:415 | model.py | 857–873 | 900–916 | same | x = self.norm(x) | x = self.norm(x) |
| chapters/20-the-whole-model.md:446 | train.py | 199–208 | 241–254 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/20-the-whole-model.md:489 | model.py | 1091–1105 | ? | UNMAPPED | logits = logits[:, -1, :].float() |  |
| chapters/20-the-whole-model.md:489 | model.py | 1091 | ? | EXCERPT | header range unresolved: AnuLM.generate |  |
| chapters/20-the-whole-model.md:548 | model.py | 1047–1056 | 1092–1101 | same | def num_params(self) -> tuple[int, int]: | def num_params(self) -> tuple[int, int]: |
| chapters/20-the-whole-model.md:284 | model.py | 366 | 380 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/20-the-whole-model.md:619 | model.py | 441–447 |  | reattributed | 'GQAttention.forward' is in model.py, not api.py |  |
| chapters/20-the-whole-model.md:619 | model.py | 441–447 | 463–469 | same | if cache is not None: | if cache is not None: |
| chapters/20-the-whole-model.md:727 | model.py | 213–223 | 218–232 | same | class RMSNorm(nn.Module): | class RMSNorm(nn.Module): |
| chapters/20-the-whole-model.md:727 | model.py | 226–236 | 235–246 | same | class SwiGLU(nn.Module): | class SwiGLU(nn.Module): |
| chapters/20-the-whole-model.md:727 | model.py | 305–324 | 317–336 | same | def _grow(self, seq_len: int, device, dtype): | def _grow(self, seq_len: int, device, dtype): |
| chapters/20-the-whole-model.md:728 | model.py | 366 | 380 | same | NATIVE_GQA = False | NATIVE_GQA = False |
| chapters/20-the-whole-model.md:729 | model.py | 426–465 | 444–487 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/20-the-whole-model.md:729 | model.py | 576–587 | 604–615 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/20-the-whole-model.md:730 | model.py | 589–637 | 617–665 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| chapters/20-the-whole-model.md:730 | model.py | 656–667 | 689–700 | same | def forward(self, x):                                     # (B, S, D) | def forward(self, x):                                     # (B, S, D) |
| chapters/20-the-whole-model.md:730 | model.py | 771–779 | 809–817 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| chapters/20-the-whole-model.md:731 | model.py | 783–808 | 826–851 | same | def __init__(self, cfg: AnuLMConfig): | def __init__(self, cfg: AnuLMConfig): |
| chapters/20-the-whole-model.md:731 | model.py | 822–873 | 865–916 | same | def forward(self, idx, targets=None, cache: Optional[list] = None, sta | def forward(self, idx, targets=None, cache: Optional[list] = None, sta |
| chapters/20-the-whole-model.md:732 | model.py | 1047–1056 | 1092–1101 | same | def num_params(self) -> tuple[int, int]: | def num_params(self) -> tuple[int, int]: |
| chapters/20-the-whole-model.md:732 | model.py | 1059–1116 | 1104–1168 | same | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/20-the-whole-model.md:733 | train.py | 199–208 | 241–254 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/21-data.md:138 | train.py | 510–515 | 663–676 | same | memmapped = bool(args.data and args.data.endswith("meta.json")) | memmapped = bool(args.data and args.data.endswith("meta.json")) |
| chapters/21-data.md:138 | train.py | 663–676 |  | EXCERPT | if memmapped: | quoted line not in the new range |
| chapters/21-data.md:153 | train.py | 151–161 | 163–173 | same | def load_memmap(meta_path: str): | def load_memmap(meta_path: str): |
| chapters/21-data.md:171 | train.py | 199–208 | 241–254 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/21-data.md:268 | train.py | 118–148 | 118–160 | same | def __init__(self, n_tokens: int, block_size: int, batch_size: int, se | def __init__(self, n_tokens: int, block_size: int, batch_size: int, se |
| chapters/21-data.md:386 | train.py | 532–538 | 693–699 | edited | if memmapped: | if memmapped or (split_encoded and val_data is not None): |
| chapters/21-data.md:386 | train.py | 532 | 693 | edited-quote | if memmapped: | if memmapped or (split_encoded and val_data is not None): |
| chapters/21-data.md:386 | train.py | 533 | 694 | edited-quote | pass                                     # the build already held out  | pass |
| chapters/21-data.md:404 | train.py | 164–178 | 206–220 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/21-data.md:59 | experiments/build_v2.py | 119 |  | override | decided in repin-overrides.tsv |  |
| chapters/21-data.md:82 | experiments/build_v2.py | 42 |  | reattributed | quoted code is in experiments/build_v2.py, not None |  |
| chapters/21-data.md:308 | train.py | 139–141 | 147–150 | same | @property | @property |
| chapters/21-data.md:348 | train.py | 443 |  | reattributed | quoted code is in train.py, not None |  |
| chapters/21-data.md:348 | train.py | 443 | 594 | same | p.add_argument("--sample-with-replacement", action="store_true", | p.add_argument("--sample-with-replacement", action="store_true", |
| chapters/21-data.md:379 | model.py | 344 |  | reattributed | '_attn_mask' is defined there |  |
| chapters/21-data.md:379 | model.py | 344 | 358 | def | '_attn_mask' | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/21-data.md:463 | test_model.py | 1732 |  | reattributed | 'memmap_batches_match_in_memory_batches' is defined there |  |
| chapters/21-data.md:465 | test_model.py | 787 |  | reattributed | 'window_sampler_covers_every_window_once_per_epoch' is defined there |  |
| chapters/21-data.md:466 | test_model.py | 812 |  | reattributed | 'window_sampler_resumes_exactly' is defined there |  |
| chapters/21-data.md:467 | test_model.py | 1754 |  | reattributed | 'interleaved_split_samples_the_whole_corpus' is defined there |  |
| chapters/21-data.md:518 | experiments/build_v2.py | 119 | 119 | sweep | 'main' 84–140 ('flush' 101–112, budget check 119, 'me |  |
| chapters/21-data.md:520 | train.py | 104–148 | 104–160 | same | class WindowSampler: | class WindowSampler: |
| chapters/21-data.md:520 | train.py | 139–141 | 147–150 | same | @property | @property |
| chapters/21-data.md:520 | train.py | 151–161 | 163–173 | same | def load_memmap(meta_path: str): | def load_memmap(meta_path: str): |
| chapters/21-data.md:521 | train.py | 164–178 | 206–220 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/21-data.md:521 | train.py | 199–208 | 241–254 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| chapters/21-data.md:522 | train.py | 224–245 | 355–376 | same | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i |
| chapters/21-data.md:522 | train.py | 224–245 | 355–376 | sweep | 224–245, 'ma |  |
| chapters/21-data.md:522 | train.py | 443 | 594 | same | p.add_argument("--sample-with-replacement", action="store_true", | p.add_argument("--sample-with-replacement", action="store_true", |
| chapters/21-data.md:522 | train.py | 510–515 | 663–676 | same | memmapped = bool(args.data and args.data.endswith("meta.json")) | memmapped = bool(args.data and args.data.endswith("meta.json")) |
| chapters/21-data.md:522 | train.py | 532–538 | 693–699 | edited | if memmapped: | if memmapped or (split_encoded and val_data is not None): |
| chapters/21-data.md:522 | train.py | 692–693 | 855–856 | same | ix = (torch.randint(len(train_data) - args.block_size - 1, (args.batch | ix = (torch.randint(len(train_data) - args.block_size - 1, (args.batch |
| chapters/21-data.md:523 | model.py | 344–360 | 358–374 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/22-the-training-loop.md:43 | train.py | 553–572 | 715–734 | same | builder = {"30b": AnuLMConfig.nano_30b, | builder = {"30b": AnuLMConfig.nano_30b, |
| chapters/22-the-training-loop.md:115 | train.py | 589–612 | 751–774 | same | raw_model = model | raw_model = model |
| chapters/22-the-training-loop.md:158 | train.py | 683–707 | 846–870 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/22-the-training-loop.md:216 | train.py | 708–714 | 871–877 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/22-the-training-loop.md:284 | train.py | 716–723 | 879–886 | same | if step % args.log_every == 0 or step == args.steps - 1: | if step % args.log_every == 0 or step == args.steps - 1: |
| chapters/22-the-training-loop.md:317 | train.py | 725–742 | 888–905 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/22-the-training-loop.md:360 | train.py | 248–260 | 379–395 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/22-the-training-loop.md:10 | train.py | 399–750 | 540–921 | edited | def main(): | def main(): |
| chapters/22-the-training-loop.md:107 | train.py | 499 |  | reattributed | 'dtype' is in train.py, not None |  |
| chapters/22-the-training-loop.md:107 | train.py | 499 | 650 | same | dtype = (torch.bfloat16 if args.device.startswith("cuda") | dtype = (torch.bfloat16 if args.device.startswith("cuda") |
| chapters/22-the-training-loop.md:140 | train.py | 597–599 | 759–761 | same | # Inductor generates C++ and needs a host compiler (MSVC 'cl' on Windo | # Inductor generates C++ and needs a host compiler (MSVC 'cl' on Windo |
| chapters/22-the-training-loop.md:240 | train.py | 686–712 | 849–875 | same | lr = lr_at(step, args, start_step) | lr = lr_at(step, args, start_step) |
| chapters/22-the-training-loop.md:457 | test_model.py | 437 |  | reattributed | 'gradient_accumulation_equals_large_batch' is defined there |  |
| chapters/22-the-training-loop.md:511 | train.py | 399–750 | 540–921 | edited | def main(): | def main(): |
| chapters/22-the-training-loop.md:511 | train.py | 499 | 650 | same | dtype = (torch.bfloat16 if args.device.startswith("cuda") | dtype = (torch.bfloat16 if args.device.startswith("cuda") |
| chapters/22-the-training-loop.md:511 | train.py | 499 | 650 | sweep | - 'train.py': 'main' 399–750 (499, 553 |  |
| chapters/22-the-training-loop.md:511 | train.py | 553–572 | 715–734 | same | builder = {"30b": AnuLMConfig.nano_30b, | builder = {"30b": AnuLMConfig.nano_30b, |
| chapters/22-the-training-loop.md:511 | train.py | 553–572 | 715–734 | sweep | - 'train.py': 'main' 399–750 (499, 553–572, 576 |  |
| chapters/22-the-training-loop.md:511 | train.py | 576 | 738 | same | model = AnuLM(cfg).to(args.device) | model = AnuLM(cfg).to(args.device) |
| chapters/22-the-training-loop.md:511 | train.py | 576 | 738 | sweep | - 'train.py': 'main' 399–750 (499, 553–572, 576, 587 |  |
| chapters/22-the-training-loop.md:511 | train.py | 587 | 749 | same | f"({100*active/total:.1f}%)  \|  device={args.device} dtype={dtype}") | f"({100*active/total:.1f}%)  \|  device={args.device} dtype={dtype}") |
| chapters/22-the-training-loop.md:511 | train.py | 587 | 749 | sweep | rain.py': 'main' 399–750 (499, 553–572, 576, 587, 589 |  |
| chapters/22-the-training-loop.md:511 | train.py | 589–612 | 751–774 | same | raw_model = model | raw_model = model |
| chapters/22-the-training-loop.md:511 | train.py | 589–612 | 751–774 | sweep | py': 'main' 399–750 (499, 553–572, 576, 587, 589–612, 683 |  |
| chapters/22-the-training-loop.md:511 | train.py | 683–707 | 846–870 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/22-the-training-loop.md:511 | train.py | 683–707 | 846–870 | sweep | n' 399–750 (499, 553–572, 576, 587, 589–612, 683–707, 703 |  |
| chapters/22-the-training-loop.md:511 | train.py | 703 | 866 | same | loss_acc += lm_loss.item() / args.grad_accum | loss_acc += lm_loss.item() / args.grad_accum |
| chapters/22-the-training-loop.md:511 | train.py | 703 | 866 | sweep | 0 (499, 553–572, 576, 587, 589–612, 683–707, 703, |  |
| chapters/22-the-training-loop.md:512 | train.py | 708–714 | 871–877 | same | if args.grad_clip > 0: | if args.grad_clip > 0: |
| chapters/22-the-training-loop.md:512 | train.py | 708–714 | 871–877 | sweep | 708–714, 716 |  |
| chapters/22-the-training-loop.md:512 | train.py | 716–723 | 879–886 | same | if step % args.log_every == 0 or step == args.steps - 1: | if step % args.log_every == 0 or step == args.steps - 1: |
| chapters/22-the-training-loop.md:512 | train.py | 716–723 | 879–886 | sweep | 708–714, 716–723, 725 |  |
| chapters/22-the-training-loop.md:512 | train.py | 725–742 | 888–905 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/22-the-training-loop.md:512 | train.py | 725–742 | 888–905 | sweep | 708–714, 716–723, 725–742), 'l |  |
| chapters/22-the-training-loop.md:512 | train.py | 248–269 | 379–404 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/22-the-training-loop.md:512 | train.py | 285–288 | 420–424 | same | def set_lr_scale(self, scale: float): | def set_lr_scale(self, scale: float): |
| chapters/22-the-training-loop.md:513 | model.py | 23 | 23 | sweep | el.py': 'from __future__ import annotations' 23, |  |
| chapters/22-the-training-loop.md:514 | model.py | 1014–1044 | 1059–1089 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/23-precision-memory-speed.md:62 | train.py | 498–503 | 649–654 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/23-precision-memory-speed.md:131 | model.py | 219–223 | 227–232 | same | def forward(self, x): | def forward(self, x): |
| chapters/23-precision-memory-speed.md:154 | model.py | 580–587 | 608–615 | same | # fp32 router: top-k is discrete, and bf16 rounding would flip choices | # fp32 router: top-k is discrete, and bf16 rounding would flip choices |
| chapters/23-precision-memory-speed.md:256 | model.py | 846–854 | 889–897 | same | for i, layer in enumerate(self.layers): | for i, layer in enumerate(self.layers): |
| chapters/23-precision-memory-speed.md:312 | model.py | 887–901 | 930–944 | same | self.ce_chunk = rows | self.ce_chunk = rows |
| chapters/23-precision-memory-speed.md:357 | train.py | 189–196 | 231–238 | same | total = torch.cuda.get_device_properties(device).total_memory | total = torch.cuda.get_device_properties(device).total_memory |
| chapters/23-precision-memory-speed.md:370 | model.py | 1216–1224 | 1268–1276 | same | n = sum(p.numel() for p in model.parameters()) | n = sum(p.numel() for p in model.parameters()) |
| chapters/23-precision-memory-speed.md:142 | model.py | 869 | 912 | same | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ |
| chapters/23-precision-memory-speed.md:145 | model.py | 716 | 749 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/23-precision-memory-speed.md:145 | model.py | 743 | 777 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/23-precision-memory-speed.md:270 | model.py | 1007 | 1052 | def | 'enable_gradient_checkpointing' | def enable_gradient_checkpointing(self, enable: bool = True): |
| chapters/23-precision-memory-speed.md:340 | model.py | 864 | 907 | same | if self.ce_chunk and self.training: | if self.ce_chunk and self.training: |
| chapters/23-precision-memory-speed.md:385 | train.py | 714 | 877 | same | warn_if_spilling(args.device) | warn_if_spilling(args.device) |
| chapters/23-precision-memory-speed.md:391 | omni_train.py | 144 | 194 | same | fit_memory(self.model) | fit_memory(self.model) |
| chapters/23-precision-memory-speed.md:437 | model.py | 358–359 | 372–373 | same | if window is not None: | if window is not None: |
| chapters/23-precision-memory-speed.md:517 | train.py | 181–196 | 223–238 | same | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: |
| chapters/23-precision-memory-speed.md:517 | train.py | 498–503 | 649–654 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/23-precision-memory-speed.md:517 | train.py | 543 | 705 | same | assert not args.ce_chunk, "--bigram-teacher needs the logits; drop --c | assert not args.ce_chunk, "--bigram-teacher needs the logits; drop --c |
| chapters/23-precision-memory-speed.md:517 | train.py | 714 | 877 | same | warn_if_spilling(args.device) | warn_if_spilling(args.device) |
| chapters/23-precision-memory-speed.md:518 | omni_train.py | 144 | 194 | same | fit_memory(self.model) | fit_memory(self.model) |
| chapters/23-precision-memory-speed.md:518 | omni_train.py | 144 | 194 | sweep | - 'fit_memory' callers: 'omni_train.py' 144, 'de |  |
| chapters/23-precision-memory-speed.md:518 | decide.py | 444 | 473 | same | fit_memory(d.model) | fit_memory(d.model) |
| chapters/23-precision-memory-speed.md:518 | decide.py | 444 | 473 | sweep | y' callers: 'omni_train.py' 144, 'decide.py' 444 and |  |
| chapters/23-precision-memory-speed.md:518 | decide.py | 519 | 551 | same | fit_memory(d.model) | fit_memory(d.model) |
| chapters/23-precision-memory-speed.md:518 | decide.py | 519 | 551 | sweep | rs: 'omni_train.py' 144, 'decide.py' 444 and 519, |  |
| chapters/23-precision-memory-speed.md:519 | speech_encoder.py | 188 | 232 | same | fit_memory(model) | fit_memory(model) |
| chapters/23-precision-memory-speed.md:519 | speech_encoder.py | 188 | 232 | sweep | 'speech_encoder.py' 188, 'vi |  |
| chapters/23-precision-memory-speed.md:519 | vision_encoder.py | 382 | 457 | same | fit_memory(model) | fit_memory(model) |
| chapters/23-precision-memory-speed.md:519 | vision_encoder.py | 382 | 457 | sweep | 'speech_encoder.py' 188, 'vision_encoder.py' 382. |  |
| chapters/23-precision-memory-speed.md:520 | model.py | 219–223 | 227–232 | same | def forward(self, x): | def forward(self, x): |
| chapters/23-precision-memory-speed.md:520 | model.py | 576–587 | 604–615 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/23-precision-memory-speed.md:521 | model.py | 358–359 | 372–373 | same | if window is not None: | if window is not None: |
| chapters/23-precision-memory-speed.md:521 | model.py | 716 | 749 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/23-precision-memory-speed.md:521 | model.py | 743 | 777 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/23-precision-memory-speed.md:522 | model.py | 846–854 | 889–897 | same | for i, layer in enumerate(self.layers): | for i, layer in enumerate(self.layers): |
| chapters/23-precision-memory-speed.md:522 | model.py | 864 | 907 | same | if self.ce_chunk and self.training: | if self.ce_chunk and self.training: |
| chapters/23-precision-memory-speed.md:522 | model.py | 869 | 912 | same | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ | logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_ |
| chapters/23-precision-memory-speed.md:522 | model.py | 875–887 | 918–930 | same | def enable_chunked_loss(self, rows: int = 2048): | def enable_chunked_loss(self, rows: int = 2048): |
| chapters/23-precision-memory-speed.md:523 | model.py | 889–898 | 932–941 | same | def _chunked_loss(self, x, targets): | def _chunked_loss(self, x, targets): |
| chapters/23-precision-memory-speed.md:523 | model.py | 900–901 | 943–944 | same | def _ce_sum(self, h, t): | def _ce_sum(self, h, t): |
| chapters/23-precision-memory-speed.md:524 | model.py | 1007–1011 | 1052–1056 | same | def enable_gradient_checkpointing(self, enable: bool = True): | def enable_gradient_checkpointing(self, enable: bool = True): |
| chapters/23-precision-memory-speed.md:524 | model.py | 1205–1224 | 1257–1276 | same | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i |
| chapters/24-measuring-progress.md:37 | train.py | 211–221 | 342–352 | same | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, |
| chapters/24-measuring-progress.md:74 | train.py | 340–396 | 481–537 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/24-measuring-progress.md:74 | train.py | 341 | 482 | edited-quote | def evaluate(model, data, args, iters: int \| None = None) -> float: | def evaluate(model, data, args, iters: int \| None = None) -> tuple[flo |
| chapters/24-measuring-progress.md:209 | train.py | 224–245 | 355–376 | same | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i |
| chapters/24-measuring-progress.md:291 | train.py | 622–626 | 784–788 | same | eval_spec = ({"mode": "strided", "windows": args.eval_windows, | eval_spec = ({"mode": "strided", "windows": args.eval_windows, |
| chapters/24-measuring-progress.md:315 | train.py | 740–742 | 903–905 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| chapters/24-measuring-progress.md:340 | eval_bench.py | 49–65 | 55–71 | same | bs = cfg.block_size | bs = cfg.block_size |
| chapters/24-measuring-progress.md:377 | eval_golden.py | 77–94 | 86–103 | same | # --- multiple choice: one batch, the 4 choices right-padded ------ | # --- multiple choice: one batch, the 4 choices right-padded ------ |
| chapters/24-measuring-progress.md:28 | train.py | 532–538 | 693–699 | edited | if memmapped: | if memmapped or (split_encoded and val_data is not None): |
| chapters/24-measuring-progress.md:303 | train.py | 532–538 | 693–699 | edited | if memmapped: | if memmapped or (split_encoded and val_data is not None): |
| chapters/24-measuring-progress.md:477 | train.py | 211–221 | 342–352 | same | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, | def eval_windows(data: torch.Tensor, batch_size: int, block_size: int, |
| chapters/24-measuring-progress.md:477 | train.py | 224–245 | 355–376 | same | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i | def strided_windows(data: torch.Tensor, batch_size: int, block_size: i |
| chapters/24-measuring-progress.md:478 | train.py | 340–396 | 481–537 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/24-measuring-progress.md:478 | train.py | 340–396 | 481–537 | sweep | 340–396, 'ma |  |
| chapters/24-measuring-progress.md:478 | train.py | 622–626 | 784–788 | same | eval_spec = ({"mode": "strided", "windows": args.eval_windows, | eval_spec = ({"mode": "strided", "windows": args.eval_windows, |
| chapters/24-measuring-progress.md:478 | train.py | 740–742 | ? | CHECK | bpb = val / math.log(2) / args.bytes_per_token | 'eval_spec' spans [(622, 626, 622)] at the old pin |
| chapters/24-measuring-progress.md:478 | train.py | 740–742 | 740–742 | sweep | 340–396, 'main' 622–626 ('eval_spec') and 740–742 (bit |  |
| chapters/24-measuring-progress.md:480 | test_model.py | 971–1017 | 971–1017 | sweep | 971–1017, 'go |  |
| chapters/24-measuring-progress.md:481 | eval_bench.py | 31–74 | 31–80 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/24-measuring-progress.md:481 | eval_golden.py | 57–110 | 59–119 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/24-measuring-progress.md:482 | eval_code.py | 185–230 | ? | UNMAPPED | def score(ckpt: str, problems: list[dict], device: str, max_new: int,  |  |
| chapters/24-measuring-progress.md:484 | train.py | 532–538 | 693–699 | edited | if memmapped: | if memmapped or (split_encoded and val_data is not None): |
| chapters/25-checkpoints-and-resuming.md:27 | train.py | 725–739 | 888–902 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/25-checkpoints-and-resuming.md:169 | train.py | 143–148 | 152–160 | same | def state(self) -> dict: | def state(self) -> dict: |
| chapters/25-checkpoints-and-resuming.md:183 | test_model.py | 813 | 813 | edited-quote | from train import WindowSampler | from anulm.training.train import WindowSampler |
| chapters/25-checkpoints-and-resuming.md:209 | train.py | 628–675 | 790–837 | same | start_step, best_val = 0, float("inf") | start_step, best_val = 0, float("inf") |
| chapters/25-checkpoints-and-resuming.md:267 | train.py | 683–684 | 846–847 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/25-checkpoints-and-resuming.md:341 | train.py | 263–268 | 398–403 | same | # Extending a finished run (--resume with a larger --steps) lands mid- | # Extending a finished run (--resume with a larger --steps) lands mid- |
| chapters/25-checkpoints-and-resuming.md:369 | model.py | 1153–1186 | 1205–1238 | same | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di |
| chapters/25-checkpoints-and-resuming.md:166 | train.py | 498 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:166 | train.py | 498 | 649 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| chapters/25-checkpoints-and-resuming.md:167 | train.py | 693 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:167 | train.py | 693 | 856 | same | if args.sample_with_replacement else sampler.next()) | if args.sample_with_replacement else sampler.next()) |
| chapters/25-checkpoints-and-resuming.md:355 | train.py | 249–260 |  | override | decided in repin-overrides.tsv |  |
| chapters/25-checkpoints-and-resuming.md:355 | train.py | 249–260 | 384–395 | same | if step < args.warmup: | if step < args.warmup: |
| chapters/25-checkpoints-and-resuming.md:393 | test_model.py | 914 |  | reattributed | quoted code is in test_model.py, not None |  |
| chapters/25-checkpoints-and-resuming.md:461 | train.py | 143–148 | 152–160 | same | def state(self) -> dict: | def state(self) -> dict: |
| chapters/25-checkpoints-and-resuming.md:461 | train.py | 143–148 | 152–160 | sweep | 143–148, 'lr |  |
| chapters/25-checkpoints-and-resuming.md:461 | train.py | 248–269 | 379–404 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/25-checkpoints-and-resuming.md:461 | train.py | 628–675 | 790–837 | same | start_step, best_val = 0, float("inf") | start_step, best_val = 0, float("inf") |
| chapters/25-checkpoints-and-resuming.md:461 | train.py | 683–684 | 846–847 | same | end = min(args.steps, args.stop_at) if args.stop_at else args.steps | end = min(args.steps, args.stop_at) if args.stop_at else args.steps |
| chapters/25-checkpoints-and-resuming.md:461 | train.py | 683–684 | 846–847 | sweep | 8, 'lr_at' 248–269, 'main' 628–675 (resume), 683–684 ('-- |  |
| chapters/25-checkpoints-and-resuming.md:462 | train.py | 725–739 | 888–902 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/25-checkpoints-and-resuming.md:462 | train.py | 725–739 | 888–902 | sweep | 725–739 (sav |  |
| chapters/25-checkpoints-and-resuming.md:463 | model.py | 1153–1202 | 1205–1254 | same | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di | def load_checkpoint(path, map_location="cpu", base: bool = True) -> di |
| chapters/25-checkpoints-and-resuming.md:466 | test_model.py | 310–317 | 310–317 | sweep | 310–317, 'an |  |
| chapters/25-checkpoints-and-resuming.md:467 | finetune.py | 50 | 52 | edited | from train import OptimizerSet, atomic_save, lr_at, make_optimizer | from anulm.training.train import OptimizerSet, atomic_save, lr_at, mak |
| chapters/26-debugging-silent-failures.md:30 | test_model.py | 4 | 4 | edited-quote | python test_model.py | python -m tests.test_model |
| chapters/26-debugging-silent-failures.md:173 | test_model.py | 1757 | 1757 | edited-quote | from train import interleaved_split | from anulm.training.train import interleaved_split |
| chapters/26-debugging-silent-failures.md:260 | model.py | 576–587 | 604–615 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/26-debugging-silent-failures.md:330 | model.py | 911–929 | 954–972 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/26-debugging-silent-failures.md:363 | model.py | 931–933 | 974–978 | same | def train(self, mode: bool = True): | def train(self, mode: bool = True): |
| chapters/26-debugging-silent-failures.md:399 | train.py | 164–178 | 206–220 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/26-debugging-silent-failures.md:433 | model.py | 421–424 | 439–442 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/26-debugging-silent-failures.md:504 | test_model.py | 1769–1782 | 2213–2226 | same | def main(): | def main(): |
| chapters/26-debugging-silent-failures.md:73 | model.py | 393 | 407 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/26-debugging-silent-failures.md:77 | model.py | 454–463 | 476–485 | same | if NATIVE_GQA and self.n_rep > 1: | if NATIVE_GQA and self.n_rep > 1: |
| chapters/26-debugging-silent-failures.md:484 | test_model.py | 1448 |  | reattributed | 'golden_items_are_well_formed_and_the_scorer_runs' is defined there |  |
| chapters/26-debugging-silent-failures.md:485 | test_model.py | 1367 |  | reattributed | 'bpe_files_load_under_both_format_ids' is defined there |  |
| chapters/26-debugging-silent-failures.md:486 | test_model.py | 550 |  | reattributed | 'incremental_forward_matches_full_forward' is defined there |  |
| chapters/26-debugging-silent-failures.md:523 | test_model.py | 28 |  | override | decided in repin-overrides.tsv |  |
| chapters/26-debugging-silent-failures.md:589 | test_model.py | 1–10 | 1–10 | sweep | - 'test_model.py': module docstring 1–10; 'te |  |
| chapters/26-debugging-silent-failures.md:591 | test_model.py | 132–161 | 132–161 | sweep | 132–161; 'de |  |
| chapters/26-debugging-silent-failures.md:593 | test_model.py | 1769–1782 | 2213–2226 | same | def main(): | def main(): |
| chapters/26-debugging-silent-failures.md:595 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/26-debugging-silent-failures.md:595 | model.py | 421–424 | 439–442 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/26-debugging-silent-failures.md:596 | model.py | 576–587 | 604–615 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| chapters/26-debugging-silent-failures.md:596 | model.py | 911–929 | 954–972 | same | def _table(self, side: str): | def _table(self, side: str): |
| chapters/26-debugging-silent-failures.md:596 | model.py | 931–933 | 974–978 | same | def train(self, mode: bool = True): | def train(self, mode: bool = True): |
| chapters/26-debugging-silent-failures.md:597 | train.py | 164–178 | 206–220 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| chapters/26-debugging-silent-failures.md:597 | train.py | 181–196 | 223–238 | same | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: | def warn_if_spilling(device: str, frac: float = 0.92) -> bool: |
| chapters/27-decoding-and-the-kv-cache.md:39 | model.py | 1079–1090 | 1128–1139 | same | B = idx.shape[0] | B = idx.shape[0] |
| chapters/27-decoding-and-the-kv-cache.md:66 | model.py | 1104–1109 | 1156–1161 | reindented | probs = F.softmax(logits, dim=-1) | probs = F.softmax(logits, dim=-1) |
| chapters/27-decoding-and-the-kv-cache.md:66 | model.py | 1156–1161 |  | EXCERPT | nxt = torch.multinomial(probs, num_samples=1)              # (B, 1) | quoted line not in the new range |
| chapters/27-decoding-and-the-kv-cache.md:113 | model.py | 818–820 | 861–863 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/27-decoding-and-the-kv-cache.md:121 | model.py | 441–448 | 463–470 | same | if cache is not None: | if cache is not None: |
| chapters/27-decoding-and-the-kv-cache.md:179 | model.py | 842–844 | 885–887 | same | assert start + S <= self.cfg.block_size, \ | assert start + S <= self.cfg.block_size, \ |
| chapters/27-decoding-and-the-kv-cache.md:187 | model.py | 320–324 | 332–336 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| chapters/27-decoding-and-the-kv-cache.md:212 | model.py | 344–360 | 358–374 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/27-decoding-and-the-kv-cache.md:227 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/27-decoding-and-the-kv-cache.md:85 | model.py | 1083-1090 |  | override | decided in repin-overrides.tsv |  |
| chapters/27-decoding-and-the-kv-cache.md:85 | model.py | 1083–1090 | 1132–1139 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/27-decoding-and-the-kv-cache.md:201 | test_model.py | 625 |  | reattributed | quoted code is in test_model.py, not model.py |  |
| chapters/27-decoding-and-the-kv-cache.md:334 | api.py | 334 | 343 | same | class Generation: | class Generation: |
| chapters/27-decoding-and-the-kv-cache.md:335 | api.py | 342–343 | 351–352 | same | self.ids = self.eng.encode(prompt)[-(self.eng.cfg.block_size - 1):] | self.ids = self.eng.encode(prompt)[-(self.eng.cfg.block_size - 1):] |
| chapters/27-decoding-and-the-kv-cache.md:387 | model.py | 716 | 749 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/27-decoding-and-the-kv-cache.md:387 | model.py | 743 | 777 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/27-decoding-and-the-kv-cache.md:483 | test_model.py | 571 |  | reattributed | 'cache_holds_what_it_should' is defined there |  |
| chapters/27-decoding-and-the-kv-cache.md:545 | model.py | 320–324 | 332–336 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| chapters/27-decoding-and-the-kv-cache.md:545 | model.py | 344–360 | 358–374 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/27-decoding-and-the-kv-cache.md:545 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/27-decoding-and-the-kv-cache.md:546 | model.py | 441–448 | 463–470 | same | if cache is not None: | if cache is not None: |
| chapters/27-decoding-and-the-kv-cache.md:546 | model.py | 520–529 | 548–557 | same | if cache is not None: | if cache is not None: |
| chapters/27-decoding-and-the-kv-cache.md:547 | model.py | 818–820 | 861–863 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/27-decoding-and-the-kv-cache.md:547 | model.py | 842–844 | 885–887 | same | assert start + S <= self.cfg.block_size, \ | assert start + S <= self.cfg.block_size, \ |
| chapters/27-decoding-and-the-kv-cache.md:548 | model.py | 1079–1109 | 1128–1161 | same | B = idx.shape[0] | B = idx.shape[0] |
| chapters/27-decoding-and-the-kv-cache.md:549 | api.py | 334 | 343 | same | class Generation: | class Generation: |
| chapters/27-decoding-and-the-kv-cache.md:549 | api.py | 342–343 | 351–352 | same | self.ids = self.eng.encode(prompt)[-(self.eng.cfg.block_size - 1):] | self.ids = self.eng.encode(prompt)[-(self.eng.cfg.block_size - 1):] |
| chapters/28-choosing-the-next-token.md:19 | model.py | 1091–1105 | ? | UNMAPPED | logits = logits[:, -1, :].float() |  |
| chapters/28-choosing-the-next-token.md:19 | model.py | 1091 | ? | EXCERPT | header range unresolved: AnuLM.generate |  |
| chapters/28-choosing-the-next-token.md:187 | model.py | 980–997 | 1025–1042 | same | for _ in range(max_tokens): | for _ in range(max_tokens): |
| chapters/28-choosing-the-next-token.md:225 | model.py | 1106–1115 | 1158–1167 | same | if eos_id is not None: | if eos_id is not None: |
| chapters/28-choosing-the-next-token.md:253 | api.py | 187–190 | 192–195 | same | temperature: Optional[float] = Field(None, ge=0, le=2, description="De | temperature: Optional[float] = Field(None, ge=0, le=2, description="De |
| chapters/28-choosing-the-next-token.md:253 | api.py | 189 | 194 | edited-quote | top_k: Optional[int] = Field(None, description="AnuLM extension. Defau | top_k: Optional[int] = Field(None, ge=0, description="AnuLM extension. |
| chapters/28-choosing-the-next-token.md:77 | model.py | 1105 |  | override | decided in repin-overrides.tsv |  |
| chapters/28-choosing-the-next-token.md:77 | model.py | 1105 | ? | UNMAPPED | nxt = torch.multinomial(probs, num_samples=1)              # (B, 1) |  |
| chapters/28-choosing-the-next-token.md:90 | model.py | 1096-1104 |  | override | decided in repin-overrides.tsv |  |
| chapters/28-choosing-the-next-token.md:90 | model.py | 1096–1104 | 1145–1156 | reindented | if repetition_penalty != 1.0: | if repetition_penalty != 1.0: |
| chapters/28-choosing-the-next-token.md:120 | test_model.py | 387 |  | reattributed | 'generation_is_deterministic_at_top_k_1' is defined there |  |
| chapters/28-choosing-the-next-token.md:123 | serve.py | 135 | 153 | edited | out = self.model.generate(x, max_tokens, temperature=max(temperature,  | out = self.model.generate(x, max_tokens, temperature=temperature,   #  |
| chapters/28-choosing-the-next-token.md:123 | api.py | 387 | 399 | edited | self.eng.model.generate(x, self.max_tokens, temperature=max(self.tempe | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/28-choosing-the-next-token.md:242 | test_model.py | 609 |  | reattributed | 'generation_stops_at_eos' is defined there |  |
| chapters/28-choosing-the-next-token.md:262 | api.py | 334 | 343 | def | 'Generation' | class Generation: |
| chapters/28-choosing-the-next-token.md:263 | api.py | 387–389 | 399–401 | edited | self.eng.model.generate(x, self.max_tokens, temperature=max(self.tempe | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/28-choosing-the-next-token.md:436 | model.py | 958–1005 | 1003–1050 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/28-choosing-the-next-token.md:436 | model.py | 980–997 | 1025–1042 | same | for _ in range(max_tokens): | for _ in range(max_tokens): |
| chapters/28-choosing-the-next-token.md:436 | model.py | 980–997 | 1025–1042 | sweep | 'AnuLM.decode_from_embeds' 958–1005 (excerpt 980–997), |  |
| chapters/28-choosing-the-next-token.md:437 | model.py | 1091–1115 | 1140–1167 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/28-choosing-the-next-token.md:438 | api.py | 187–190 | 192–195 | same | temperature: Optional[float] = Field(None, ge=0, le=2, description="De | temperature: Optional[float] = Field(None, ge=0, le=2, description="De |
| chapters/28-choosing-the-next-token.md:438 | api.py | 334 | 343 | same | class Generation: | class Generation: |
| chapters/28-choosing-the-next-token.md:438 | api.py | 387–389 | 399–401 | edited | self.eng.model.generate(x, self.max_tokens, temperature=max(self.tempe | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/28-choosing-the-next-token.md:439 | api.py | 544–546 | 559–561 | same | gen = Generation(chat_prompt(question), req.max_completion_tokens or r | gen = Generation(chat_prompt(question), req.max_completion_tokens or r |
| chapters/28-choosing-the-next-token.md:439 | api.py | 544–546 | 559–561 | sweep | 544–546, com |  |
| chapters/28-choosing-the-next-token.md:439 | api.py | 582–583 | 597–598 | edited | gen = Generation(prompt, req.max_tokens or 64, 0.7 if req.temperature  | gen = Generation(prompt, req.max_tokens or 64, 0.7 if req.temperature  |
| chapters/28-choosing-the-next-token.md:439 | api.py | 582–583 | 597–598 | sweep | 544–546, completion defaults 582–583. |  |
| chapters/28-choosing-the-next-token.md:440 | serve.py | 135 | 153 | edited | out = self.model.generate(x, max_tokens, temperature=max(temperature,  | out = self.model.generate(x, max_tokens, temperature=temperature,   #  |
| chapters/28-choosing-the-next-token.md:440 | serve.py | 135 | 153 | sweep | - 'serve.py': temperature floor 135, HTT |  |
| chapters/28-choosing-the-next-token.md:440 | serve.py | 187–191 | 210–214 | same | temperature=float(req.get("temperature", 0.8)), | temperature=float(req.get("temperature", 0.8)), |
| chapters/28-choosing-the-next-token.md:440 | serve.py | 187–191 | 210–214 | sweep | ve.py': temperature floor 135, HTTP defaults 187–191. 'sa |  |
| chapters/28-choosing-the-next-token.md:440 | sample.py | 31–32 | 34–35 | same | p.add_argument("--temperature", type=float, default=0.8) | p.add_argument("--temperature", type=float, default=0.8) |
| chapters/28-choosing-the-next-token.md:440 | sample.py | 31–32 | 34–35 | sweep | oor 135, HTTP defaults 187–191. 'sample.py': 31–32. |  |
| chapters/29-long-context.md:59 | model.py | 243–256 | 253–268 | same | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): |
| chapters/29-long-context.md:84 | model.py | 286–303 | 298–315 | same | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: |
| chapters/29-long-context.md:165 | model.py | 274–279 | 286–291 | same | if cfg.yarn: | if cfg.yarn: |
| chapters/29-long-context.md:176 | model.py | 421–424 | 439–442 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/29-long-context.md:291 | model.py | 767–769 | 805–807 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/29-long-context.md:82 | model.py | 104–105 | 109–110 | same | yarn_beta_fast: float = 32.0 | yarn_beta_fast: float = 32.0 |
| chapters/29-long-context.md:185 | model.py | 269–273 |  | override | decided in repin-overrides.tsv |  |
| chapters/29-long-context.md:185 | model.py | 269–273 | 281–285 | same | # YaRN's temperature, split the way DeepSeek (and so the 105B) splits | # YaRN's temperature, split the way DeepSeek (and so the 105B) splits |
| chapters/29-long-context.md:206 | test_model.py | 133 |  | reattributed | quoted code is in test_model.py, not docs/RESULTS.md |  |
| chapters/29-long-context.md:416 | model.py | 104–105 | 109–110 | same | yarn_beta_fast: float = 32.0 | yarn_beta_fast: float = 32.0 |
| chapters/29-long-context.md:416 | model.py | 104–105 | 109–110 | sweep | - 'model.py': configuration 104–105; '_y |  |
| chapters/29-long-context.md:417 | model.py | 243–256 | 253–268 | same | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): | def _yarn_correction_dim(num_rotations, dim, base, orig_ctx): |
| chapters/29-long-context.md:417 | model.py | 269–279 | 281–291 | same | # YaRN's temperature, split the way DeepSeek (and so the 105B) splits | # YaRN's temperature, split the way DeepSeek (and so the 105B) splits |
| chapters/29-long-context.md:418 | model.py | 286–303 | 298–315 | same | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: | def _build_inv_freq(cfg: AnuLMConfig, dim: int) -> torch.Tensor: |
| chapters/29-long-context.md:418 | model.py | 344–360 | 358–374 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| chapters/29-long-context.md:419 | model.py | 421–424 | 439–442 | same | # YaRN's temperature term lives here, not in the rotary tables -- | # YaRN's temperature term lives here, not in the rotary tables -- |
| chapters/29-long-context.md:419 | model.py | 767–769 | 805–807 | same | # Sliding window on the first max_window_layers layers, full attention | # Sliding window on the first max_window_layers layers, full attention |
| chapters/30-instruction-tuning.md:74 | finetune.py | 53–58 | 55–60 | same | def _pair_ids(tok: BPE, it: dict) -> tuple[list[int], list[int]]: | def _pair_ids(tok: BPE, it: dict) -> tuple[list[int], list[int]]: |
| chapters/30-instruction-tuning.md:99 | finetune.py | 84–111 | 91–118 | same | def pack(pairs, block_size: int, eos: int) -> tuple[torch.Tensor, torc | def pack(pairs, block_size: int, eos: int) -> tuple[torch.Tensor, torc |
| chapters/30-instruction-tuning.md:142 | model.py | 867–873 | 910–916 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/30-instruction-tuning.md:259 | finetune.py | 302–317 | 359–374 | same | for step in range(start_step, end): | for step in range(start_step, end): |
| chapters/30-instruction-tuning.md:287 | finetune.py | 166–178 | 207–219 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/30-instruction-tuning.md:318 | make_chat.py | 105–112 | ? | UNMAPPED | for name, items in parts.items(): |  |
| chapters/30-instruction-tuning.md:318 | make_chat.py | 105 | ? | EXCERPT | header range unresolved: main |  |
| chapters/30-instruction-tuning.md:71 | finetune.py | 328 | 385 | def | 'qa_templates' | "val_loss": val, "qa_template": PROMPT, "qa_templates": PROMPTS, |
| chapters/30-instruction-tuning.md:391 | test_model.py | 1494 |  | reattributed | 'qa_packing_masks_prompts_and_keeps_examples_whole' is defined there |  |
| chapters/30-instruction-tuning.md:480 | finetune.py | 53–58 | 55–60 | same | def _pair_ids(tok: BPE, it: dict) -> tuple[list[int], list[int]]: | def _pair_ids(tok: BPE, it: dict) -> tuple[list[int], list[int]]: |
| chapters/30-instruction-tuning.md:480 | finetune.py | 84–111 | 91–118 | same | def pack(pairs, block_size: int, eos: int) -> tuple[torch.Tensor, torc | def pack(pairs, block_size: int, eos: int) -> tuple[torch.Tensor, torc |
| chapters/30-instruction-tuning.md:480 | finetune.py | 166–178 | 207–219 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/30-instruction-tuning.md:481 | finetune.py | 181–365 | 222–422 | same | def main(): | def main(): |
| chapters/30-instruction-tuning.md:481 | finetune.py | 302–317 | 359–374 | same | for step in range(start_step, end): | for step in range(start_step, end): |
| chapters/30-instruction-tuning.md:481 | finetune.py | 302–317 | 359–374 | sweep | 'main' 181–365 (the step 302–317, 'qa |  |
| chapters/30-instruction-tuning.md:481 | finetune.py | 328 | 385 | same | "val_loss": val, "qa_template": PROMPT, "qa_templates": PROMPTS, | "val_loss": val, "qa_template": PROMPT, "qa_templates": PROMPTS, |
| chapters/30-instruction-tuning.md:482 | make_chat.py | 97–119 | ? | UNMAPPED | def main(): |  |
| chapters/30-instruction-tuning.md:482 | make_chat.py | 109–112 | ? | UNMAPPED | rng.shuffle(rows) |  |
| chapters/30-instruction-tuning.md:482 | make_chat.py | 109–112 | 109–112 | sweep | - 'make_chat.py': 'main' 97–119 (the split 109–112). |  |
| chapters/30-instruction-tuning.md:483 | model.py | 867–873 | 910–916 | same | logits = self._head(x) | logits = self._head(x) |
| chapters/30-instruction-tuning.md:483 | model.py | 871–872 | 914–915 | same | if aux_total is not None: | if aux_total is not None: |
| chapters/30-instruction-tuning.md:483 | model.py | 871–872 | 914–915 | sweep | 867–873 ('ignore_index=-1'; the regulariser 871–872). |  |
| chapters/31-lora.md:57 | lora.py | 50–81 | 50–84 | same | class LoRALinear(nn.Module): | class LoRALinear(nn.Module): |
| chapters/31-lora.md:116 | lora.py | 84–109 | 87–112 | same | def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, | def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, |
| chapters/31-lora.md:181 | lora.py | 121–132 | 126–137 | same | def router_state_dict(model: nn.Module) -> dict: | def router_state_dict(model: nn.Module) -> dict: |
| chapters/31-lora.md:192 | finetune.py | 335–342 | 392–399 | same | if args.lora: | if args.lora: |
| chapters/31-lora.md:192 | finetune.py | 336 | 393 | edited-quote | from lora import lora_state_dict, router_state_dict | from anulm.lora import lora_state_dict, router_state_dict |
| chapters/31-lora.md:264 | lora.py | 142–158 | 150–166 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/31-lora.md:286 | lora.py | 186–194 | 196–204 | same | model.load_state_dict(base["model"]) | model.load_state_dict(base["model"]) |
| chapters/31-lora.md:376 | decide.py | 236–271 | 247–286 | same | class LoRA(nn.Module): | class LoRA(nn.Module): |
| chapters/31-lora.md:460 | lora.py | 50–81 | 50–84 | same | class LoRALinear(nn.Module): | class LoRALinear(nn.Module): |
| chapters/31-lora.md:460 | lora.py | 84–109 | 87–112 | same | def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, | def apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, |
| chapters/31-lora.md:461 | lora.py | 121–132 | 126–137 | same | def router_state_dict(model: nn.Module) -> dict: | def router_state_dict(model: nn.Module) -> dict: |
| chapters/31-lora.md:461 | lora.py | 142–158 | 150–166 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/31-lora.md:461 | lora.py | 171–204 | 181–214 | same | def _merge_cli(args) -> int: | def _merge_cli(args) -> int: |
| chapters/31-lora.md:462 | finetune.py | 335–342 | 392–399 | same | if args.lora: | if args.lora: |
| chapters/31-lora.md:462 | finetune.py | 335–342 | 392–399 | sweep | - 'finetune.py': adapter save 335–342; 'up |  |
| chapters/31-lora.md:462 | finetune.py | 317 | 374 | same | imbalance = model.update_expert_biases() | imbalance = model.update_expert_biases() |
| chapters/31-lora.md:462 | finetune.py | 317 | 374 | sweep | er save 335–342; 'update_expert_biases' call 317. |  |
| chapters/31-lora.md:463 | model.py | 1014–1044 | 1059–1089 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/31-lora.md:464 | decide.py | 236–249 | 247–262 | same | class LoRA(nn.Module): | class LoRA(nn.Module): |
| chapters/31-lora.md:464 | decide.py | 252–271 | 265–286 | same | class LoRAQV(nn.Module): | class LoRAQV(nn.Module): |
| chapters/32-classification-and-calibration.md:21 | decide.py | 88–97 | 90–99 | same | def encode(tok, text: str, options: list[str], max_len: int = 1024): | def encode(tok, text: str, options: list[str], max_len: int = 1024): |
| chapters/32-classification-and-calibration.md:60 | decide.py | 227–233 | 238–244 | same | def forward(self, idx, pos, mask=None): | def forward(self, idx, pos, mask=None): |
| chapters/32-classification-and-calibration.md:79 | decide.py | 274–285 | 289–304 | same | def batchify(examples, tok, pad, with_mask=False): | def batchify(examples, tok, pad, with_mask=False): |
| chapters/32-classification-and-calibration.md:98 | decide.py | 192–210 | 198–216 | same | def _hidden(self, idx): | def _hidden(self, idx): |
| chapters/32-classification-and-calibration.md:98 | decide.py | 200 | 206 | edited-quote | from bidirectional import bidir_hidden | from anulm.bidirectional import bidir_hidden |
| chapters/32-classification-and-calibration.md:179 | model.py | 450–452 | 472–474 | same | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio |
| chapters/32-classification-and-calibration.md:204 | decide.py | 344–350 | 366–372 | same | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: |
| chapters/32-classification-and-calibration.md:225 | decide.py | 353–370 | 375–395 | same | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d |
| chapters/32-classification-and-calibration.md:37 | decide.py | 82 | 84 | same | def readable(label: str) -> str: | def readable(label: str) -> str: |
| chapters/32-classification-and-calibration.md:74 | decide.py | 116 |  | override | decided in repin-overrides.tsv |  |
| chapters/32-classification-and-calibration.md:74 | decide.py | 116 | 118 | same | self.head = nn.Linear(hid, 1) | self.head = nn.Linear(hid, 1) |
| chapters/32-classification-and-calibration.md:76 | decide.py | 562 |  | override | decided in repin-overrides.tsv |  |
| chapters/32-classification-and-calibration.md:76 | decide.py | 562 | 594 | same | loss = F.cross_entropy(logits, y.to(a.device)) | loss = F.cross_entropy(logits, y.to(a.device)) |
| chapters/32-classification-and-calibration.md:187 | bidirectional.py | 56–101 |  | reattributed | 'adapt' is in bidirectional.py, not model.py |  |
| chapters/32-classification-and-calibration.md:187 | bidirectional.py | 56–101 | 56–105 | same | def adapt(a): | def adapt(a): |
| chapters/32-classification-and-calibration.md:218 | decide.py | 577 | 609 | same | t = fit_temperature(lv, yv) | t = fit_temperature(lv, yv) |
| chapters/32-classification-and-calibration.md:219 | model.py | 657 |  | reattributed | quoted code is in model.py, not decide.py |  |
| chapters/32-classification-and-calibration.md:219 | model.py | 657 | 690 | same | """Returns (y, aux): the routed + shared output, and the router's | """Returns (y, aux): the routed + shared output, and the router's |
| chapters/32-classification-and-calibration.md:402 | decide.py | 82 | 84 | same | def readable(label: str) -> str: | def readable(label: str) -> str: |
| chapters/32-classification-and-calibration.md:402 | decide.py | 88–97 | 90–99 | same | def encode(tok, text: str, options: list[str], max_len: int = 1024): | def encode(tok, text: str, options: list[str], max_len: int = 1024): |
| chapters/32-classification-and-calibration.md:402 | decide.py | 101–233 | 103–244 | same | class Decider(nn.Module): | class Decider(nn.Module): |
| chapters/32-classification-and-calibration.md:402 | decide.py | 116 | 118 | same | self.head = nn.Linear(hid, 1) | self.head = nn.Linear(hid, 1) |
| chapters/32-classification-and-calibration.md:403 | decide.py | 192–210 | 198–216 | same | def _hidden(self, idx): | def _hidden(self, idx): |
| chapters/32-classification-and-calibration.md:403 | decide.py | 227–233 | 238–244 | same | def forward(self, idx, pos, mask=None): | def forward(self, idx, pos, mask=None): |
| chapters/32-classification-and-calibration.md:403 | decide.py | 236–249 | 247–262 | same | class LoRA(nn.Module): | class LoRA(nn.Module): |
| chapters/32-classification-and-calibration.md:403 | decide.py | 274 | 289 | same | def batchify(examples, tok, pad, with_mask=False): | def batchify(examples, tok, pad, with_mask=False): |
| chapters/32-classification-and-calibration.md:404 | decide.py | 288 | 307 | same | def make_examples(rows, tok, labs, rng=None): | def make_examples(rows, tok, labs, rng=None): |
| chapters/32-classification-and-calibration.md:404 | decide.py | 344–350 | 366–372 | same | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: | def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float: |
| chapters/32-classification-and-calibration.md:404 | decide.py | 353–370 | 375–395 | same | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d | def metrics(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> d |
| chapters/32-classification-and-calibration.md:405 | decide.py | 514–582 | 543–614 | same | def train(a): | def train(a): |
| chapters/32-classification-and-calibration.md:405 | decide.py | 514–582 | 543–614 | sweep | 514–582 (los |  |
| chapters/32-classification-and-calibration.md:405 | decide.py | 562 | 594 | same | loss = F.cross_entropy(logits, y.to(a.device)) | loss = F.cross_entropy(logits, y.to(a.device)) |
| chapters/32-classification-and-calibration.md:405 | decide.py | 562 | 594 | sweep | 514–582 (loss 562, tem |  |
| chapters/32-classification-and-calibration.md:405 | decide.py | 577 | 609 | same | t = fit_temperature(lv, yv) | t = fit_temperature(lv, yv) |
| chapters/32-classification-and-calibration.md:405 | decide.py | 577 | 609 | sweep | 514–582 (loss 562, temperature 577), 'D |  |
| chapters/32-classification-and-calibration.md:405 | decide.py | 653–663 | 687–697 | same | def __call__(self, message: str, options: list[str] \| None = None, top | def __call__(self, message: str, options: list[str] \| None = None, top |
| chapters/32-classification-and-calibration.md:406 | bidirectional.py | 56–101 | 56–105 | same | def adapt(a): | def adapt(a): |
| chapters/32-classification-and-calibration.md:407 | model.py | 450–452 | 472–474 | same | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio | # bidir_mask (B, 1, 1, T), True = a real token: BIDIRECTIONAL attentio |
| chapters/33-retrieval-and-readers.md:28 | rag.py | 42–56 | 42–60 | same | BUCKETS = 1 << 24 | BUCKETS = 1 << 24 |
| chapters/33-retrieval-and-readers.md:66 | rag.py | 127–136 | 136–145 | same | N, avgdl = len(pairs), float(lens.mean()) | N, avgdl = len(pairs), float(lens.mean()) |
| chapters/33-retrieval-and-readers.md:94 | rag.py | 264–284 | 284–304 | same | scores = np.zeros(self.n, dtype=np.float32) | scores = np.zeros(self.n, dtype=np.float32) |
| chapters/33-retrieval-and-readers.md:159 | rag.py | 186–194 | 195–205 | same | def embed(self, texts: list[str], max_len: int = 256): | def embed(self, texts: list[str], max_len: int = 256): |
| chapters/33-retrieval-and-readers.md:185 | rag.py | 243–257 | 263–277 | same | mode = mode or self.mode | mode = mode or self.mode |
| chapters/33-retrieval-and-readers.md:249 | rag.py | 318–322 | 340–347 | same | def prompt(self, question: str, passages: list[str]) -> tuple[str, str | def prompt(self, question: str, passages: list[str]) -> tuple[str, str |
| chapters/33-retrieval-and-readers.md:285 | rc_pointer.py | 59–69 | 61–73 | same | def forward(self, idx): | def forward(self, idx): |
| chapters/33-retrieval-and-readers.md:305 | rc_pointer.py | 172–177 | 188–193 | same | def loss_of(items): | def loss_of(items): |
| chapters/33-retrieval-and-readers.md:319 | rc_pointer.py | 79–97 | 83–103 | same | def encode(tok, qword, q, body, answer=None): | def encode(tok, qword, q, body, answer=None): |
| chapters/33-retrieval-and-readers.md:348 | rc_pointer.py | 111–124 | 120–133 | same | def best_span(s, e, n): | def best_span(s, e, n): |
| chapters/33-retrieval-and-readers.md:389 | rag.py | 345–364 | 374–396 | same | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") |
| chapters/33-retrieval-and-readers.md:49 | rag.py | 87 | 91 | same | ids = term_ids(title + " " + chunk) | ids = term_ids(title + " " + chunk) |
| chapters/33-retrieval-and-readers.md:181 | rag.py | 200 |  | reattributed | quoted code is in rag.py, not None |  |
| chapters/33-retrieval-and-readers.md:181 | rag.py | 200 | 213 | same | out = torch.empty(len(chunks), self.m.config.hidden_size, dtype=torch. | out = torch.empty(len(chunks), self.m.config.hidden_size, dtype=torch. |
| chapters/33-retrieval-and-readers.md:236 | rag.py | 234 | 252 | same | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" |
| chapters/33-retrieval-and-readers.md:237 | rag.py | 232 |  | override | decided in repin-overrides.tsv |  |
| chapters/33-retrieval-and-readers.md:237 | rag.py | 232 | 250 | same | # the top 3 every time (BM25: 15/20) and took our own reader from 0/20 | # the top 3 every time (BM25: 15/20) and took our own reader from 0/20 |
| chapters/33-retrieval-and-readers.md:272 | rag.py | 325–333 |  | override | decided in repin-overrides.tsv |  |
| chapters/33-retrieval-and-readers.md:272 | rag.py | 325–333 | 350–360 | same | def read(self, prompt: str, max_tokens: int = 32) -> str: | def read(self, prompt: str, max_tokens: int = 32) -> str: |
| chapters/33-retrieval-and-readers.md:497 | rag.py | 42–43 | 42–43 | sweep | - 'rag.py': constants 42–43, 'wo |  |
| chapters/33-retrieval-and-readers.md:497 | rag.py | 55 | 57 | same | def term_ids(text: str) -> np.ndarray: | def term_ids(text: str) -> np.ndarray: |
| chapters/33-retrieval-and-readers.md:497 | rag.py | 87 | ? | CHECK | ids = term_ids(title + " " + chunk) | 'term_ids' spans [(55, 56, 55)] at the old pin |
| chapters/33-retrieval-and-readers.md:497 | rag.py | 87 | 87 | sweep | rds' 51, 'term_ids' 55, title + text indexed 87, |  |
| chapters/33-retrieval-and-readers.md:498 | rag.py | 345–364 | 374–396 | same | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") | _PUNCT = re.compile(r"[\"'()\[\],;:।.!?\-–—]") |
| chapters/33-retrieval-and-readers.md:498 | rag.py | 99–159 | 103–168 | same | def build(langs=("simple", "hi"), workers: int = 12) -> None: | def build(langs=("simple", "hi"), workers: int = 12) -> None: |
| chapters/33-retrieval-and-readers.md:499 | rag.py | 127–136 | 136–145 | same | N, avgdl = len(pairs), float(lens.mean()) | N, avgdl = len(pairs), float(lens.mean()) |
| chapters/33-retrieval-and-readers.md:499 | rag.py | 127–136 | 136–145 | sweep | 127–136), 'D |  |
| chapters/33-retrieval-and-readers.md:499 | rag.py | 165–215 | 174–230 | same | class Dense: | class Dense: |
| chapters/33-retrieval-and-readers.md:499 | rag.py | 186–194 | 195–205 | same | def embed(self, texts: list[str], max_len: int = 256): | def embed(self, texts: list[str], max_len: int = 256): |
| chapters/33-retrieval-and-readers.md:499 | rag.py | 218–284 | 233–304 | same | class Retriever: | class Retriever: |
| chapters/33-retrieval-and-readers.md:500 | rag.py | 234 | 252 | same | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" | self.mode = "dense" if (root / "dense_e5s.npy").exists() else "bm25" |
| chapters/33-retrieval-and-readers.md:500 | rag.py | 234 | 252 | sweep | mode 234, 'se |  |
| chapters/33-retrieval-and-readers.md:500 | rag.py | 239–257 | 257–277 | same | def search(self, query: str, k: int = 3, lang: str \| None = None, | def search(self, query: str, k: int = 3, lang: str \| None = None, |
| chapters/33-retrieval-and-readers.md:500 | rag.py | 259–284 | 279–304 | same | def bm25(self, query: str, k: int = 3, lang: str \| None = None, | def bm25(self, query: str, k: int = 3, lang: str \| None = None, |
| chapters/33-retrieval-and-readers.md:500 | rag.py | 298–342 | 320–371 | same | class RagQA: | class RagQA: |
| chapters/33-retrieval-and-readers.md:501 | rc_pointer.py | 51–69 | 51–73 | same | class Pointer(nn.Module): | class Pointer(nn.Module): |
| chapters/33-retrieval-and-readers.md:501 | rc_pointer.py | 79–97 | 83–103 | same | def encode(tok, qword, q, body, answer=None): | def encode(tok, qword, q, body, answer=None): |
| chapters/33-retrieval-and-readers.md:501 | rc_pointer.py | 111–124 | 120–133 | same | def best_span(s, e, n): | def best_span(s, e, n): |
| chapters/33-retrieval-and-readers.md:502 | rc_pointer.py | 144–216 | 157–232 | same | def train(a): | def train(a): |
| chapters/33-retrieval-and-readers.md:502 | rc_pointer.py | 172–177 | 188–193 | same | def loss_of(items): | def loss_of(items): |
| chapters/33-retrieval-and-readers.md:502 | rc_pointer.py | 172–177 | 188–193 | sweep | 'train' 144–216 (loss 172–177). |  |
| chapters/34-preference-optimisation-and-rl.md:65 | model.py | 860–866 | 903–909 | same | if targets is None: | if targets is None: |
| chapters/34-preference-optimisation-and-rl.md:319 | model.py | 1101–1105 | ? | UNMAPPED | if top_k is not None: |  |
| chapters/34-preference-optimisation-and-rl.md:319 | model.py | 1101 | ? | EXCERPT | header range unresolved: AnuLM.generate |  |
| chapters/34-preference-optimisation-and-rl.md:400 | model.py | 860–866 | 903–909 | same | if targets is None: | if targets is None: |
| chapters/34-preference-optimisation-and-rl.md:400 | model.py | 1101–1105 | ? | UNMAPPED | if top_k is not None: |  |
| chapters/35-wider-vocabulary.md:86 | audio_codec.py | 108–126 | 121–139 | same | def flatten(codes: list) -> np.ndarray: | def flatten(codes: list) -> np.ndarray: |
| chapters/35-wider-vocabulary.md:237 | model.py | 832–841 | 875–884 | same | # 'embeds' bypasses the embedding table: (B, S, D) straight in. Audio | # 'embeds' bypasses the embedding table: (B, S, D) straight in. Audio |
| chapters/35-wider-vocabulary.md:264 | speech_encoder.py | 106–126 | ? | UNMAPPED | def build(self, states: torch.Tensor, text_ids: list[int], answer_ids: |  |
| chapters/35-wider-vocabulary.md:264 | speech_encoder.py | 106 | ? | EXCERPT | header range unresolved: ContinuousSpeech.build |  |
| chapters/35-wider-vocabulary.md:331 | model.py | 1091–1095 | 1140–1144 | same | logits = logits[:, -1, :].float() | logits = logits[:, -1, :].float() |
| chapters/35-wider-vocabulary.md:384 | model.py | 976–1000 | 1021–1045 | same | dev = prefix.device | dev = prefix.device |
| chapters/35-wider-vocabulary.md:410 | speech_encoder.py | 342–346 | ? | UNMAPPED | with autocast: |  |
| chapters/35-wider-vocabulary.md:410 | speech_encoder.py | 342 | ? | EXCERPT | header range unresolved: transcribe |  |
| chapters/35-wider-vocabulary.md:62 | speech_vocab.py | 58–61 |  | reattributed | quoted code is in speech_vocab.py, not audio_codec.py |  |
| chapters/35-wider-vocabulary.md:108 | audio_codec.py | 129–157 | 142–170 | same | def unflatten(ids: np.ndarray, report: bool = False): | def unflatten(ids: np.ndarray, report: bool = False): |
| chapters/35-wider-vocabulary.md:253 | model.py | 841 |  | override | decided in repin-overrides.tsv |  |
| chapters/35-wider-vocabulary.md:253 | model.py | 841 | 884 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| chapters/35-wider-vocabulary.md:321 | speech_encoder.py | 244 | 288 | same | _, loss = model(None, tb, embeds=eb) | _, loss = model(None, tb, embeds=eb) |
| chapters/35-wider-vocabulary.md:482 | audio_codec.py | 108–126 | 121–139 | same | def flatten(codes: list) -> np.ndarray: | def flatten(codes: list) -> np.ndarray: |
| chapters/35-wider-vocabulary.md:482 | audio_codec.py | 129–157 | 142–170 | same | def unflatten(ids: np.ndarray, report: bool = False): | def unflatten(ids: np.ndarray, report: bool = False): |
| chapters/35-wider-vocabulary.md:483 | model.py | 822–873 | 865–916 | same | def forward(self, idx, targets=None, cache: Optional[list] = None, sta | def forward(self, idx, targets=None, cache: Optional[list] = None, sta |
| chapters/35-wider-vocabulary.md:483 | model.py | 958–1005 | 1003–1050 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/35-wider-vocabulary.md:484 | model.py | 1058–1116 | 1103–1168 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/35-wider-vocabulary.md:485 | speech_encoder.py | 106–126 | ? | UNMAPPED | def build(self, states: torch.Tensor, text_ids: list[int], answer_ids: |  |
| chapters/35-wider-vocabulary.md:486 | speech_encoder.py | 244 | 288 | same | _, loss = model(None, tb, embeds=eb) | _, loss = model(None, tb, embeds=eb) |
| chapters/35-wider-vocabulary.md:486 | speech_encoder.py | 244 | 288 | sweep | 244, 'tr |  |
| chapters/35-wider-vocabulary.md:486 | speech_encoder.py | 342–346 | ? | UNMAPPED | with autocast: |  |
| chapters/36-frozen-encoders.md:27 | vision_encoder.py | 72–97 | 74–106 | same | def __init__(self, name: str = DEFAULT_TOWER, device: str \| None = Non | def __init__(self, name: str = DEFAULT_TOWER, device: str \| None = Non |
| chapters/36-frozen-encoders.md:57 | audio_codec.py | 207–229 | 220–255 | same | def __init__(self, size: str = "small", device: str = "auto", compute_ | def __init__(self, size: str = "small", device: str = "auto", compute_ |
| chapters/36-frozen-encoders.md:57 | audio_codec.py | 226 | 252 | edited-quote | # Whisper's own 30 s window; longer clips are chunked by the caller. | # Whisper's fixed maximum feature window, after the explicit duration  |
| chapters/36-frozen-encoders.md:148 | vision_encoder.py | 118–132 | 127–141 | same | def forward(self, patches: torch.Tensor) -> torch.Tensor: | def forward(self, patches: torch.Tensor) -> torch.Tensor: |
| chapters/36-frozen-encoders.md:168 | vision_encoder.py | 109–116 | 118–125 | same | def __init__(self, in_dim: int, out_dim: int, pool: int = POOL, hidden | def __init__(self, in_dim: int, out_dim: int, pool: int = POOL, hidden |
| chapters/36-frozen-encoders.md:221 | vision_encoder.py | 152–165 | 162–183 | same | dev = patches.device | dev = patches.device |
| chapters/36-frozen-encoders.md:338 | vision_encoder.py | 411–414 | 486–489 | same | opt = torch.optim.AdamW( | opt = torch.optim.AdamW( |
| chapters/36-frozen-encoders.md:353 | vision_encoder.py | 269–287 | 312–361 | same | path = Path(path) | path = Path(path) |
| chapters/36-frozen-encoders.md:353 | vision_encoder.py | 312–361 |  | EXCERPT | if path.exists() and path.stat().st_size == int(np.prod(shape)) * 2: | quoted line not in the new range |
| chapters/36-frozen-encoders.md:353 | vision_encoder.py | 312–361 |  | EXCERPT | mm = np.memmap(path, dtype=dtype, mode="w+", shape=shape) | quoted line not in the new range |
| chapters/36-frozen-encoders.md:353 | vision_encoder.py | 281 | 350 | edited-quote | mm[i] = tower.encode(img).cpu().numpy().astype(dtype) | mm[i] = tower.encode(img).detach().cpu().numpy().astype(dtype) |
| chapters/36-frozen-encoders.md:396 | speech_encoder.py | 154–160 | 198–204 | edited | for r in chunk: | for r in chunk: |
| chapters/36-frozen-encoders.md:396 | speech_encoder.py | 160 | 204 | edited-quote | built.append(sm.build(states, [], answer)) | built.append(sm.build(states, [], answer, example=r["audio"], task=tas |
| chapters/36-frozen-encoders.md:348 | vision_encoder.py | 579–580 | 654–655 | same | t.add_argument("--lr", type=float, default=5e-5) | t.add_argument("--lr", type=float, default=5e-5) |
| chapters/36-frozen-encoders.md:492 | vision_encoder.py | 53 | 55 | same | POOL = 2                        # 2x2 patch pooling: 196 patches -> 49 | POOL = 2                        # 2x2 patch pooling: 196 patches -> 49 |
| chapters/36-frozen-encoders.md:492 | vision_encoder.py | 67–97 | 69–106 | same | class VisionTower: | class VisionTower: |
| chapters/36-frozen-encoders.md:492 | vision_encoder.py | 100–132 | 109–141 | same | class ImageProjector(nn.Module): | class ImageProjector(nn.Module): |
| chapters/36-frozen-encoders.md:493 | vision_encoder.py | 144–165 | 153–183 | same | def build(self, patches: torch.Tensor, answer_ids: list[int], | def build(self, patches: torch.Tensor, answer_ids: list[int], |
| chapters/36-frozen-encoders.md:493 | vision_encoder.py | 257–287 | 298–361 | same | def build_cache(rows, tower, path: str \| Path, dtype="float16"): | def build_cache(rows, tower, path: str \| Path, dtype="float16"): |
| chapters/36-frozen-encoders.md:493 | vision_encoder.py | 368–460 | 443–535 | same | def train(args) -> None: | def train(args) -> None: |
| chapters/36-frozen-encoders.md:494 | vision_encoder.py | 411–414 | 486–489 | same | opt = torch.optim.AdamW( | opt = torch.optim.AdamW( |
| chapters/36-frozen-encoders.md:494 | vision_encoder.py | 411–414 | 486–489 | sweep | (optimiser 411–414, def |  |
| chapters/36-frozen-encoders.md:494 | vision_encoder.py | 579–580 | 654–655 | same | t.add_argument("--lr", type=float, default=5e-5) | t.add_argument("--lr", type=float, default=5e-5) |
| chapters/36-frozen-encoders.md:494 | vision_encoder.py | 579–580 | 654–655 | sweep | (optimiser 411–414, defaults 579–580). |  |
| chapters/36-frozen-encoders.md:495 | speech_encoder.py | 142–170 | 186–214 | same | def batches(rows, enc: WhisperEncoder, tok, sm: ContinuousSpeech, devi | def batches(rows, enc: WhisperEncoder, tok, sm: ContinuousSpeech, devi |
| chapters/36-frozen-encoders.md:496 | audio_codec.py | 198–229 | 211–255 | same | class WhisperEncoder: | class WhisperEncoder: |
| chapters/37-one-model-many-tasks.md:31 | omni_train.py | 101–124 | 104–174 | same | def pad(built, device): | def pad(built, device): |
| chapters/37-one-model-many-tasks.md:82 | omni_train.py | 66–79 | 69–82 | same | def __init__(self, spec: str): | def __init__(self, spec: str): |
| chapters/37-one-model-many-tasks.md:113 | omni_train.py | 163–178 | 217–235 | same | held = dict(h.split("=", 1) for h in (args.held or [])) | held = dict(h.split("=", 1) for h in (args.held or [])) |
| chapters/37-one-model-many-tasks.md:113 | omni_train.py | 217–235 |  | EXCERPT | cut = max(1, int(len(rows) * 0.02)) | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:113 | omni_train.py | 217–235 |  | EXCERPT | t.held, t.rows = rows[:cut], rows[cut:] | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:113 | omni_train.py | 217–235 |  | EXCERPT | from vision_encoder import build_cache | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:113 | omni_train.py | 217–235 |  | EXCERPT | self.caches[t.name] = build_cache(rows, self.tower, Path(t.path).with_ | quoted line not in the new range |
| chapters/37-one-model-many-tasks.md:162 | omni_train.py | 282–305 | ? | UNMAPPED | w = np.array([t.weight for t in om.tasks], dtype=float) |  |
| chapters/37-one-model-many-tasks.md:162 | omni_train.py | 282 | ? | EXCERPT | header range unresolved: train |  |
| chapters/37-one-model-many-tasks.md:241 | omni_train.py | 311–318 | 457–464 | same | if step % args.eval_every == 0 or step == args.steps: | if step % args.eval_every == 0 or step == args.steps: |
| chapters/37-one-model-many-tasks.md:299 | omni_train.py | 223–233 | 336–350 | same | def state(self, step, score, losses) -> dict: | def state(self, step, score, losses) -> dict: |
| chapters/37-one-model-many-tasks.md:299 | omni_train.py | 228 | 345 | edited-quote | "qa_templates": self.ck.get("qa_templates")} | "qa_templates": qa_templates} |
| chapters/37-one-model-many-tasks.md:321 | omni.py | 50 | 50 | edited-quote | from audio_codec import WhisperEncoder | from .audio_codec import WhisperEncoder |
| chapters/37-one-model-many-tasks.md:347 | experiments/omni_one.py | 230–238 | 375–383 | same | def gates(res) -> dict: | def gates(res) -> dict: |
| chapters/37-one-model-many-tasks.md:384 | omni_train.py | 211–221 | ? | UNMAPPED | @torch.no_grad() |  |
| chapters/37-one-model-many-tasks.md:384 | omni_train.py | 211 | ? | EXCERPT | header range unresolved: Omni.held_loss |  |
| chapters/37-one-model-many-tasks.md:403 | omni_eval.py | 116 | 116 | edited-quote | from finetune import _pair_ids | from anulm.training.finetune import _pair_ids |
| chapters/37-one-model-many-tasks.md:61 | omni_train.py | 82–93 | 85–96 | same | def load_rows(task: Task, tok, max_len: int) -> list[dict]: | def load_rows(task: Task, tok, max_len: int) -> list[dict]: |
| chapters/37-one-model-many-tasks.md:101 | omni_train.py | 61 | 64 | def | 'image' | KINDS = ("text", "asr", "image") |
| chapters/37-one-model-many-tasks.md:102 | omni_train.py | 62 | 65 | same | DEFAULT_BATCH = {"text": 8, "asr": 4, "image": 8} | DEFAULT_BATCH = {"text": 8, "asr": 4, "image": 8} |
| chapters/37-one-model-many-tasks.md:137 | experiments/omni_one.py | 61–69 | 77–87 | same | TASKS = [ | TASKS = [ |
| chapters/37-one-model-many-tasks.md:195 | omni_train.py | 202–209 | 309–317 | same | def next_batch(self, t: Task): | def next_batch(self, t: Task): |
| chapters/37-one-model-many-tasks.md:201 | omni_train.py | 241–245 | 372–376 | same | groups = [{"params": model.parameters(), "lr": args.lr}] | groups = [{"params": model.parameters(), "lr": args.lr}] |
| chapters/37-one-model-many-tasks.md:201 | omni_train.py | 345–346 | 493–494 | same | p.add_argument("--lr", type=float, default=5e-5) | p.add_argument("--lr", type=float, default=5e-5) |
| chapters/37-one-model-many-tasks.md:254 | omni_train.py | 278 | 415 | same | first = evaluate() | first = evaluate() |
| chapters/37-one-model-many-tasks.md:282 | experiments/omni_one.py | 314 | 456 | edited | run("3_grow", ["speech_vocab.py", "--ckpt", "ckpt_chat_v3.pt", "--out" | run("3_grow", ["-m", "anulm.multimodal.speech_vocab", "--ckpt", "ckpt_ |
| chapters/37-one-model-many-tasks.md:363 | experiments/omni_one.py | 76–79 |  | override | decided in repin-overrides.tsv |  |
| chapters/37-one-model-many-tasks.md:363 | experiments/omni_one.py | 76–79 | 94–97 | same | # Hindi: the real-speaker data (Kathbath, MUCS) trades FLEURS for ever | # Hindi: the real-speaker data (Kathbath, MUCS) trades FLEURS for ever |
| chapters/37-one-model-many-tasks.md:427 | experiments/omni_one.py | 82 | 100 | same | "chat_heldout_loss": ("<=", 3.30),     # chat_v3: 3.146 by omni_eval.p | "chat_heldout_loss": ("<=", 3.30),     # chat_v3: 3.146 by omni_eval.p |
| chapters/37-one-model-many-tasks.md:491 | omni_train.py | 61–62 | 64–65 | same | KINDS = ("text", "asr", "image") | KINDS = ("text", "asr", "image") |
| chapters/37-one-model-many-tasks.md:491 | omni_train.py | 65–79 | 68–82 | same | class Task: | class Task: |
| chapters/37-one-model-many-tasks.md:492 | omni_train.py | 82–98 | 85–101 | same | def load_rows(task: Task, tok, max_len: int) -> list[dict]: | def load_rows(task: Task, tok, max_len: int) -> list[dict]: |
| chapters/37-one-model-many-tasks.md:492 | omni_train.py | 82–98 | 85–101 | sweep | 82–98, 'pa |  |
| chapters/37-one-model-many-tasks.md:492 | omni_train.py | 101–108 | 104–111 | same | def pad(built, device): | def pad(built, device): |
| chapters/37-one-model-many-tasks.md:492 | omni_train.py | 111–124 | 161–174 | same | def text_batches(rows, model, device, batch_size, shuffle=True, seed=0 | def text_batches(rows, model, device, batch_size, shuffle=True, seed=0 |
| chapters/37-one-model-many-tasks.md:492 | omni_train.py | 131–178 | 181–235 | same | def __init__(self, args): | def __init__(self, args): |
| chapters/37-one-model-many-tasks.md:493 | omni_train.py | 202–209 | 309–317 | same | def next_batch(self, t: Task): | def next_batch(self, t: Task): |
| chapters/37-one-model-many-tasks.md:493 | omni_train.py | 223–233 | 336–350 | same | def state(self, step, score, losses) -> dict: | def state(self, step, score, losses) -> dict: |
| chapters/37-one-model-many-tasks.md:493 | omni_train.py | 236–330 | 362–478 | same | def train(args) -> None: | def train(args) -> None: |
| chapters/37-one-model-many-tasks.md:494 | omni_train.py | 333–361 | ? | UNMAPPED | def main() -> None: |  |
| chapters/37-one-model-many-tasks.md:494 | omni_train.py | 333–361 | 333–361 | sweep | 333–361. |  |
| chapters/37-one-model-many-tasks.md:496 | experiments/omni_one.py | 61–69 | 77–87 | same | TASKS = [ | TASKS = [ |
| chapters/37-one-model-many-tasks.md:496 | experiments/omni_one.py | 74–86 | 92–104 | same | GATES = { | GATES = { |
| chapters/37-one-model-many-tasks.md:496 | experiments/omni_one.py | 230–238 | 375–383 | same | def gates(res) -> dict: | def gates(res) -> dict: |
| chapters/38-serving.md:74 | serve.py | 129–141 | ? | UNMAPPED | ids = self.encode(prompt)[-(self.cfg.block_size - 1):] |  |
| chapters/38-serving.md:74 | serve.py | 129 | ? | EXCERPT | header range unresolved: Engine.generate |  |
| chapters/38-serving.md:112 | serve.py | 175–195 | 198–218 | same | def do_POST(self): | def do_POST(self): |
| chapters/38-serving.md:150 | api.py | 187–193 | 192–198 | same | temperature: Optional[float] = Field(None, ge=0, le=2, description="De | temperature: Optional[float] = Field(None, ge=0, le=2, description="De |
| chapters/38-serving.md:150 | api.py | 189 | 194 | edited-quote | top_k: Optional[int] = Field(None, description="AnuLM extension. Defau | top_k: Optional[int] = Field(None, ge=0, description="AnuLM extension. |
| chapters/38-serving.md:166 | api.py | 503–508 | 518–523 | same | users = [m for m in req.messages if m.role == "user"] | users = [m for m in req.messages if m.role == "user"] |
| chapters/38-serving.md:182 | api.py | 259–260 | 264–265 | same | models: A.Models \| None = None | models: A.Models \| None = None |
| chapters/38-serving.md:204 | api.py | 371–389 | 383–401 | same | def on_token(nxt): | def on_token(nxt): |
| chapters/38-serving.md:204 | api.py | 387 | 399 | edited-quote | self.eng.model.generate(x, self.max_tokens, temperature=max(self.tempe | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/38-serving.md:232 | api.py | 350–364 | 360–376 | same | def _release(self, final: bool) -> str: | def _release(self, final: bool) -> str: |
| chapters/38-serving.md:232 | api.py | 360–376 |  | EXCERPT | text = self.eng.decode(self.new) | quoted line not in the new range |
| chapters/38-serving.md:232 | api.py | 356 | 368 | edited-quote | cut, self.finish = min(cut, i), "stop" | cut, self.finish, self.matched_stop = min(cut, i), "stop", True |
| chapters/38-serving.md:232 | api.py | 360–376 |  | EXCERPT | if self.finish != "stop" and not final: | quoted line not in the new range |
| chapters/38-serving.md:232 | api.py | 360–376 |  | EXCERPT | text = text.rstrip("�") | quoted line not in the new range |
| chapters/38-serving.md:294 | api.py | 390–394 | ? | UNMAPPED | if self.finish != "stop": |  |
| chapters/38-serving.md:294 | api.py | 390 | ? | EXCERPT | header range unresolved: Generation.run |  |
| chapters/38-serving.md:337 | api.py | 413–447 | 428–462 | same | q: queue.Queue = queue.Queue() | q: queue.Queue = queue.Queue() |
| chapters/38-serving.md:105 | serve.py | 129 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:105 | serve.py | 129 | 147 | same | ids = self.encode(prompt)[-(self.cfg.block_size - 1):] | ids = self.encode(prompt)[-(self.cfg.block_size - 1):] |
| chapters/38-serving.md:106 | serve.py | 123–128 |  | override | decided in repin-overrides.tsv |  |
| chapters/38-serving.md:106 | serve.py | 123–128 | 141–146 | same | if mode == "translate" and self.qa_templates and "en-hi" in self.qa_te | if mode == "translate" and self.qa_templates and "en-hi" in self.qa_te |
| chapters/38-serving.md:140 | serve.py | 217 | 242 | def | 'ThreadingHTTPServer' | server = ThreadingHTTPServer((args.host, args.port), make_handler(engi |
| chapters/38-serving.md:202 | model.py | 1110–1111 |  | reattributed | quoted code is in model.py, not None |  |
| chapters/38-serving.md:202 | model.py | 1110–1111 | 1162–1163 | same | if on_token is not None and on_token(nxt): | if on_token is not None and on_token(nxt): |
| chapters/38-serving.md:282 | api.py | 473–490 |  | reattributed | 'one_shot_reply' is in api.py, not None |  |
| chapters/38-serving.md:282 | api.py | 473–490 | 488–505 | same | def one_shot_reply(req: ChatRequest, content: str, extra: dict \| None  | def one_shot_reply(req: ChatRequest, content: str, extra: dict \| None  |
| chapters/38-serving.md:400 | api.py | 500 |  | reattributed | quoted code is in api.py, not docs/RESULTS.md |  |
| chapters/38-serving.md:400 | api.py | 500 | 515 | same | raise APIError(400, "Tool calling is not supported: the model was not  | raise APIError(400, "Tool calling is not supported: the model was not  |
| chapters/38-serving.md:419 | api.py | 343 | 352 | same | self.max_tokens = max(1, min(max_tokens, self.eng.cfg.block_size - len | self.max_tokens = max(1, min(max_tokens, self.eng.cfg.block_size - len |
| chapters/38-serving.md:460 | serve.py | 50–151 | 50–171 | same | class Engine: | class Engine: |
| chapters/38-serving.md:460 | serve.py | 121–151 | 134–171 | same | def generate(self, prompt: str, max_tokens: int, temperature: float, t | def generate(self, prompt: str, max_tokens: int, temperature: float, t |
| chapters/38-serving.md:461 | serve.py | 154–200 | 174–223 | same | def make_handler(engine: Engine): | def make_handler(engine: Engine): |
| chapters/38-serving.md:461 | serve.py | 203–222 | 226–247 | same | def main(): | def main(): |
| chapters/38-serving.md:462 | api.py | 181–198 | 186–203 | same | class ChatRequest(BaseModel): | class ChatRequest(BaseModel): |
| chapters/38-serving.md:462 | api.py | 260 | 265 | same | GPU = threading.Lock()                # one request on the device at a | GPU = threading.Lock()                # one request on the device at a |
| chapters/38-serving.md:462 | api.py | 334–394 | 343–409 | same | class Generation: | class Generation: |
| chapters/38-serving.md:463 | api.py | 473–490 | 488–505 | same | def one_shot_reply(req: ChatRequest, content: str, extra: dict \| None  | def one_shot_reply(req: ChatRequest, content: str, extra: dict \| None  |
| chapters/38-serving.md:464 | api.py | 408–449 | 423–464 | same | def stream_generation(gen: Generation, chunk, final_chunk, usage_chunk | def stream_generation(gen: Generation, chunk, final_chunk, usage_chunk |
| chapters/38-serving.md:464 | api.py | 493–571 | 508–586 | same | @app.post("/v1/chat/completions", tags=["OpenAI-compatible"], dependen | @app.post("/v1/chat/completions", tags=["OpenAI-compatible"], dependen |
| chapters/38-serving.md:465 | model.py | 1058–1116 | 1103–1168 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/38-serving.md:465 | model.py | 1110–1111 | 1162–1163 | same | if on_token is not None and on_token(nxt): | if on_token is not None and on_token(nxt): |
| chapters/39-exporting.md:57 | export_hf.py | 50–59 | 51–60 | same | def to_bf16(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: | def to_bf16(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: |
| chapters/39-exporting.md:160 | export_hf.py | 256–303 | 307–356 | edited | if cfg is not None: | if cfg is not None: |
| chapters/39-exporting.md:160 | export_hf.py | 303 | 356 | edited-quote | shutil.copyfile(Path(__file__).parent / mod, out / mod) | shutil.copyfile(Path(__file__).resolve().parents[1 if mod == "model.py |
| chapters/39-exporting.md:213 | model.py | 1169–1189 | 1221–1241 | same | p = _P(path) | p = _P(path) |
| chapters/39-exporting.md:254 | modeling_anulm.py | 134–146 | 155–167 | same | def __init__(self, config: AnuLMConfig): | def __init__(self, config: AnuLMConfig): |
| chapters/39-exporting.md:277 | modeling_anulm.py | 195–230 | 220–255 | same | if inputs_embeds is not None: | if inputs_embeds is not None: |
| chapters/39-exporting.md:354 | modeling_anulm.py | 257–264 | 284–291 | same | @staticmethod | @staticmethod |
| chapters/39-exporting.md:374 | modeling_anulm.py | 104–121 | 111–134 | same | @classmethod | @classmethod |
| chapters/39-exporting.md:149 | modeling_anulm.py | 107–120 | 120–133 | same | model.rebuild_rotary(device=next(model.parameters()).device) | model.rebuild_rotary(device=next(model.parameters()).device) |
| chapters/39-exporting.md:205 | export_hf.py | 231–236 | 281–287 | same | if groups: | if groups: |
| chapters/39-exporting.md:246 | test_model.py | 960–962 |  | override | decided in repin-overrides.tsv |  |
| chapters/39-exporting.md:246 | test_model.py | 960–962 | 960–962 | edited | root = P(__file__).parent | root = P(__file__).resolve().parents[1] |
| chapters/39-exporting.md:272 | configuration_anulm.py | 67–70 | 75–78 | same | def to_native(self) -> NativeConfig: | def to_native(self) -> NativeConfig: |
| chapters/39-exporting.md:315 | modeling_anulm.py | 232–247 | 257–272 | same | def _prepare_cache_for_generation(self, generation_config, model_kwarg | def _prepare_cache_for_generation(self, generation_config, model_kwarg |
| chapters/39-exporting.md:347 | modeling_anulm.py | 84–102 |  | reattributed | 'rebuild_rotary' is in modeling_anulm.py, not test_model.py |  |
| chapters/39-exporting.md:347 | modeling_anulm.py | 84–102 | 91–109 | same | def rebuild_rotary(self, device=None) -> None: | def rebuild_rotary(self, device=None) -> None: |
| chapters/39-exporting.md:369 | modeling_anulm.py | 187–193 | 212–218 | same | 'attention_mask' is accepted and ignored: this model is trained and | 'attention_mask' is accepted and ignored: this model is trained and |
| chapters/40-the-production-gap.md:24 | model.py | 818–820 | 861–863 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/40-the-production-gap.md:32 | model.py | 441–447 | 463–469 | same | if cache is not None: | if cache is not None: |
| chapters/40-the-production-gap.md:48 | model.py | 1083–1090 | 1132–1139 | same | for _ in range(max_new_tokens): | for _ in range(max_new_tokens): |
| chapters/40-the-production-gap.md:95 | api.py | 382–389 | 394–401 | same | with GPU: | with GPU: |
| chapters/40-the-production-gap.md:95 | api.py | 387 | 399 | edited-quote | self.eng.model.generate(x, self.max_tokens, temperature=max(self.tempe | self.eng.model.generate(x, self.max_tokens, temperature=self.temperatu |
| chapters/40-the-production-gap.md:174 | model.py | 712–719 | 745–752 | same | key = sum(p._version for e in self.experts for p in e.parameters()) | key = sum(p._version for e in self.experts for p in e.parameters()) |
| chapters/40-the-production-gap.md:207 | audio_codec.py | 214–218 | 227–231 | same | if compute_type is None: | if compute_type is None: |
| chapters/40-the-production-gap.md:260 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/40-the-production-gap.md:294 | train.py | 326–334 | 467–475 | same | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam |
| chapters/40-the-production-gap.md:294 | train.py | 328 | 469 | edited-quote | from muon import build_optimizers | from anulm.muon import build_optimizers |
| chapters/40-the-production-gap.md:342 | api.py | 165–169 | 170–174 | same | def check_key(authorization: Optional[str] = Header(default=None)): | def check_key(authorization: Optional[str] = Header(default=None)): |
| chapters/40-the-production-gap.md:431 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| chapters/40-the-production-gap.md:431 | model.py | 441–447 | 463–469 | same | if cache is not None: | if cache is not None: |
| chapters/40-the-production-gap.md:432 | model.py | 818–820 | 861–863 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| chapters/40-the-production-gap.md:433 | train.py | 310–337 | 451–478 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| chapters/40-the-production-gap.md:434 | audio_codec.py | 207–219 | 220–232 | same | def __init__(self, size: str = "small", device: str = "auto", compute_ | def __init__(self, size: str = "small", device: str = "auto", compute_ |
| chapters/40-the-production-gap.md:435 | api.py | 165–169 | 170–174 | same | def check_key(authorization: Optional[str] = Header(default=None)): | def check_key(authorization: Optional[str] = Header(default=None)): |
| chapters/40-the-production-gap.md:435 | api.py | 260 | 265 | same | GPU = threading.Lock()                # one request on the device at a | GPU = threading.Lock()                # one request on the device at a |
| chapters/40-the-production-gap.md:435 | api.py | 366–394 | 378–409 | same | def run(self, emit=None): | def run(self, emit=None): |
| chapters/41-capstone.md:65 | train.py | 248–250 | 379–385 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/41-capstone.md:77 | train.py | 553–571 | 715–733 | same | builder = {"30b": AnuLMConfig.nano_30b, | builder = {"30b": AnuLMConfig.nano_30b, |
| chapters/41-capstone.md:184 | train.py | 689–712 | 852–875 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/41-capstone.md:223 | train.py | 725–742 | 888–905 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/41-capstone.md:365 | model.py | 1100–1105 | ? | UNMAPPED | logits = logits / max(temperature, 1e-5) |  |
| chapters/41-capstone.md:365 | model.py | 1100 | ? | EXCERPT | header range unresolved: AnuLM.generate |  |
| chapters/41-capstone.md:610 | test_model.py | 1769–1782 | 2213–2226 | same | def main(): | def main(): |
| chapters/41-capstone.md:685 | model.py | 743–746 | 777–780 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/41-capstone.md:50 | train.py | 428–429 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:50 | train.py | 428–429 | 579–580 | same | p.add_argument("--device", type=str, | p.add_argument("--device", type=str, |
| chapters/41-capstone.md:55 | .github/workflows/tests.yml | 76 | 84 | same | run: \| | run: \| |
| chapters/41-capstone.md:62 | train.py | 411 | 558 | same | p.add_argument("--warmup", type=int, default=100) | p.add_argument("--warmup", type=int, default=100) |
| chapters/41-capstone.md:140 | train.py | 566–570 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:140 | train.py | 566–570 | 728–732 | same | kw = dict(block_size=args.block_size, yarn=args.yarn, | kw = dict(block_size=args.block_size, yarn=args.yarn, |
| chapters/41-capstone.md:252 | train.py | 104–148 | 104–160 | same | class WindowSampler: | class WindowSampler: |
| chapters/41-capstone.md:379 | model.py | 1104 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:379 | model.py | 1104 | 1156 | reindented | probs = F.softmax(logits, dim=-1) | probs = F.softmax(logits, dim=-1) |
| chapters/41-capstone.md:630 | test_model.py | 1786 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:630 | test_model.py | 1786 | 2230 | same | sys.exit(main()) | sys.exit(main()) |
| chapters/41-capstone.md:645 | test_model.py | 688, 709, 727 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:653 | export_hf.py | 271–279 | ? | UNMAPPED | # Small tensors stay float32 on disk AND must be loaded that way: the |  |
| chapters/41-capstone.md:682 | model.py | 716 |  | override | decided in repin-overrides.tsv |  |
| chapters/41-capstone.md:682 | model.py | 716 | 749 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/41-capstone.md:872 | train.py | 399–750 | 540–921 | edited | def main(): | def main(): |
| chapters/41-capstone.md:872 | train.py | 553–563 | 715–725 | same | builder = {"30b": AnuLMConfig.nano_30b, | builder = {"30b": AnuLMConfig.nano_30b, |
| chapters/41-capstone.md:872 | train.py | 553–563 | 715–725 | sweep | - 'train.py': 'main' 399–750 (preset builder 553–563, key |  |
| chapters/41-capstone.md:873 | train.py | 566–571 | 728–733 | same | kw = dict(block_size=args.block_size, yarn=args.yarn, | kw = dict(block_size=args.block_size, yarn=args.yarn, |
| chapters/41-capstone.md:873 | train.py | 566–571 | 728–733 | sweep | 566–571, the |  |
| chapters/41-capstone.md:873 | train.py | 689–712 | 852–875 | same | opt.zero_grad(set_to_none=True) | opt.zero_grad(set_to_none=True) |
| chapters/41-capstone.md:873 | train.py | 689–712 | 852–875 | sweep | 566–571, the step 689–712, eva |  |
| chapters/41-capstone.md:873 | train.py | 725–742 | 888–905 | same | if (step + 1) % args.eval_every == 0 or step == end - 1: | if (step + 1) % args.eval_every == 0 or step == end - 1: |
| chapters/41-capstone.md:873 | train.py | 725–742 | 888–905 | sweep | –571, the step 689–712, evaluation and saves 725–742), 'l |  |
| chapters/41-capstone.md:874 | train.py | 104–148 | 104–160 | same | class WindowSampler: | class WindowSampler: |
| chapters/41-capstone.md:874 | train.py | 248–269 | 379–404 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| chapters/41-capstone.md:874 | train.py | 340–396 | 481–537 | same | @torch.no_grad() | @torch.no_grad() |
| chapters/41-capstone.md:875 | train.py | 405 | 552 | same | p.add_argument("--steps", type=int, default=2000) | p.add_argument("--steps", type=int, default=2000) |
| chapters/41-capstone.md:875 | train.py | 411 | 558 | same | p.add_argument("--warmup", type=int, default=100) | p.add_argument("--warmup", type=int, default=100) |
| chapters/41-capstone.md:875 | train.py | 411 | 558 | sweep | '--steps' 2000 (line 405), '--warmup' 100 (411), '- |  |
| chapters/41-capstone.md:875 | train.py | 428–429 | 579–580 | same | p.add_argument("--device", type=str, | p.add_argument("--device", type=str, |
| chapters/41-capstone.md:875 | train.py | 428–429 | 579–580 | sweep | line 405), '--warmup' 100 (411), '--device' (428–429). |  |
| chapters/41-capstone.md:876 | .github/workflows/tests.yml | 76 | 84 | same | run: \| | run: \| |
| chapters/41-capstone.md:877 | model.py | 1059–1116 | 1104–1168 | same | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, | def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, |
| chapters/41-capstone.md:877 | model.py | 1100–1105 | ? | UNMAPPED | logits = logits / max(temperature, 1e-5) |  |
| chapters/41-capstone.md:877 | model.py | 1100–1105 | 1100–1105 | sweep | el.py': 'AnuLM.generate' 1059–1116 (sampling 1100–1105), |  |
| chapters/41-capstone.md:878 | model.py | 1047–1056 | 1092–1101 | same | def num_params(self) -> tuple[int, int]: | def num_params(self) -> tuple[int, int]: |
| chapters/41-capstone.md:878 | model.py | 725–749 | 759–783 | same | def _route_grouped(self, x_flat, topk_idx, topk_w): | def _route_grouped(self, x_flat, topk_idx, topk_w): |
| chapters/41-capstone.md:879 | model.py | 716 | 749 | same | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to | torch.stack([getattr(e, name).weight.t() for e in self.experts]).to(to |
| chapters/41-capstone.md:879 | model.py | 716 | 749 | sweep | 716 and |  |
| chapters/41-capstone.md:879 | model.py | 743 | 777 | same | xs = x_flat[order // k].to(torch.bfloat16)                          #  | xs = x_flat[order // k].to(torch.bfloat16)                          #  |
| chapters/41-capstone.md:879 | model.py | 743 | 777 | sweep | 716 and 743). |  |
| chapters/41-capstone.md:882 | test_model.py | 399 | 399 | sweep | 399, 'gr |  |
| chapters/41-capstone.md:884 | test_model.py | 1769–1782 | 2213–2226 | same | def main(): | def main(): |
| chapters/41-capstone.md:885 | test_model.py | 688 | 688 | sweep | returns at 688, 709 |  |
| chapters/41-capstone.md:885 | test_model.py | 709 | 709 | sweep | returns at 688, 709, 727 |  |
| chapters/41-capstone.md:885 | test_model.py | 727 | 727 | sweep | returns at 688, 709, 727 and |  |
| chapters/41-capstone.md:885 | test_model.py | 888–891 | 888–891 | sweep | returns at 688, 709, 727 and 888–891; 66 |  |
| chapters/41-capstone.md:885 | test_model.py | 66 | 66 | sweep | returns at 688, 709, 727 and 888–891; 66 test |  |
| chapters/41-capstone.md:886 | export_hf.py | 271–279 | ? | UNMAPPED | # Small tensors stay float32 on disk AND must be loaded that way: the |  |
| chapters/41-capstone.md:893 | bpe.py | 11 | 11 | sweep | 11 and |  |
| chapters/41-capstone.md:893 | bpe.py | 15 | 15 | sweep | 11 and 15, 'co |  |
| chapters/A1-shape-trace.md:64 | model.py | 841 | 884 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| chapters/A1-shape-trace.md:76 | model.py | 864 | 907 | same | if self.ce_chunk and self.training: | if self.ce_chunk and self.training: |
| chapters/A2-parameter-arithmetic.md:34 | model.py | 39–206 | 39–211 | same | @dataclass | @dataclass |
| chapters/A2-parameter-arithmetic.md:85 | model.py | 121–144 |  | reattributed | '__post_init__' is in model.py, not data/multi32k.json |  |
| chapters/A2-parameter-arithmetic.md:85 | model.py | 121–144 | 126–149 | same | def __post_init__(self): | def __post_init__(self): |
| chapters/A2-parameter-arithmetic.md:173 | model.py | 175–176 | 180–181 | same | Requires '--grad-ckpt'. Without it the same model peaks at 15.75 GB an | Requires '--grad-ckpt'. Without it the same model peaks at 15.75 GB an |
| chapters/A3-test-index.md:12 | test_model.py | 1769 | 2213 | same | def main(): | def main(): |
| chapters/A3-test-index.md:13 | test_model.py | 33 |  | override | decided in repin-overrides.tsv |  |
| chapters/A3-test-index.md:115 | test_model.py | 1590 |  | reattributed | 'native_gqa_matches_repeat_interleave' is in test_model.py, not None |  |
| chapters/A3-test-index.md:116 | test_model.py | 1102–1106 |  | reattributed | 'lora_starts_as_the_base_model_and_merges_back_exactly' is in test_mod |  |
| chapters/A3-test-index.md:117 | test_model.py | 688, 709, 727 |  | override | decided in repin-overrides.tsv |  |
| chapters/A3-test-index.md:118 | test_model.py | 1282–1286, 1342–1346 |  | reattributed | quoted code is in test_model.py, not None |  |
| chapters/A3-test-index.md:119 | test_model.py | 888–891 |  | reattributed | quoted code is in test_model.py, not None |  |
| chapters/A4-glossary.md:52 | export_hf.py | 271–279 | ? | UNMAPPED | # Small tensors stay float32 on disk AND must be loaded that way: the |  |
| REPOSITORY_SOURCE_MAP.md:87 | train.py | 101 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_SOURCE_MAP.md:150 | omni_train.py | 196–211 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_SOURCE_MAP.md:150 | omni_train.py | 196–211 | ? | UNMAPPED | from speech_encoder import batches |  |
| REPOSITORY_SOURCE_MAP.md:162 | audio_codec.py | 214–215 | 227–228 | same | if compute_type is None: | if compute_type is None: |
| REPOSITORY_ARCHITECTURE.md:118 | train.py | 264–312 | 399–453 | same | # several times the lr the weights last saw. --rewarmup ramps back up  | # several times the lr the weights last saw. --rewarmup ramps back up  |
| REPOSITORY_ARCHITECTURE.md:118 | train.py | 151–161 | 163–173 | same | def load_memmap(meta_path: str): | def load_memmap(meta_path: str): |
| REPOSITORY_ARCHITECTURE.md:119 | train.py | 104–148 | 104–160 | same | class WindowSampler: | class WindowSampler: |
| REPOSITORY_ARCHITECTURE.md:120 | train.py | 199–208 | 241–254 | same | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  | def make_batch(data: torch.Tensor, ix: torch.Tensor, block_size: int,  |
| REPOSITORY_ARCHITECTURE.md:121 | model.py | 841 |  | reattributed | 'AnuLM.forward' is in model.py, not train.py |  |
| REPOSITORY_ARCHITECTURE.md:121 | model.py | 841 | 884 | same | x = F.embedding(idx, self._table("in"))               # (B, S, D) | x = F.embedding(idx, self._table("in"))               # (B, S, D) |
| REPOSITORY_ARCHITECTURE.md:121 | model.py | 911–929 |  | reattributed | '_table' is in model.py, not train.py |  |
| REPOSITORY_ARCHITECTURE.md:121 | model.py | 911–929 | 954–972 | same | def _table(self, side: str): | def _table(self, side: str): |
| REPOSITORY_ARCHITECTURE.md:122 | model.py | 320–324 |  | reattributed | 'Rotary.forward' is in model.py, not train.py |  |
| REPOSITORY_ARCHITECTURE.md:122 | model.py | 320–324 | 332–336 | same | def forward(self, seq_len: int, device, dtype, start: int = 0): | def forward(self, seq_len: int, device, dtype, start: int = 0): |
| REPOSITORY_ARCHITECTURE.md:123 | model.py | 771–779 |  | reattributed | 'Block.forward' is in model.py, not train.py |  |
| REPOSITORY_ARCHITECTURE.md:123 | model.py | 771–779 | 809–817 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| REPOSITORY_ARCHITECTURE.md:124 | model.py | 426–465 |  | reattributed | 'GQAttention.forward' is in model.py, not train.py |  |
| REPOSITORY_ARCHITECTURE.md:124 | model.py | 426–465 | 444–487 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| REPOSITORY_ARCHITECTURE.md:125 | model.py | 344–360 |  | reattributed | '_attn_mask' is in model.py, not train.py |  |
| REPOSITORY_ARCHITECTURE.md:125 | model.py | 344–360 | 358–374 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| REPOSITORY_ARCHITECTURE.md:125 | model.py | 393–397 |  | reattributed | '_sdpa' is in model.py, not train.py |  |
| REPOSITORY_ARCHITECTURE.md:125 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| REPOSITORY_ARCHITECTURE.md:126 | model.py | 236 |  | reattributed | 'SwiGLU.forward' is in model.py, not train.py |  |
| REPOSITORY_ARCHITECTURE.md:126 | model.py | 236 | 246 | same | return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) | return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)) |
| REPOSITORY_ARCHITECTURE.md:126 | model.py | 656–667 |  | reattributed | 'MoE.forward' is in model.py, not train.py |  |
| REPOSITORY_ARCHITECTURE.md:126 | model.py | 656–667 | 689–700 | same | def forward(self, x):                                     # (B, S, D) | def forward(self, x):                                     # (B, S, D) |
| REPOSITORY_ARCHITECTURE.md:127 | model.py | 589–637 |  | reattributed | 'Router._gate' is in model.py, not train.py |  |
| REPOSITORY_ARCHITECTURE.md:127 | model.py | 589–637 | 617–665 | same | def _gate(self, x_flat, seq_len: Optional[int]): | def _gate(self, x_flat, seq_len: Optional[int]): |
| REPOSITORY_ARCHITECTURE.md:128 | model.py | 857–873 |  | reattributed | 'AnuLM.forward' is in model.py, not train.py |  |
| REPOSITORY_ARCHITECTURE.md:128 | model.py | 857–873 | 900–916 | same | x = self.norm(x) | x = self.norm(x) |
| REPOSITORY_ARCHITECTURE.md:129 | train.py | 310–337 | 451–478 | same | def make_optimizer(model: AnuLM, args): | def make_optimizer(model: AnuLM, args): |
| REPOSITORY_ARCHITECTURE.md:130 | train.py | 248–269 |  | reattributed | 'lr_at' is in train.py, not muon.py |  |
| REPOSITORY_ARCHITECTURE.md:130 | train.py | 248–269 | 379–404 | same | def lr_at(step: int, args, start_step: int = 0) -> float: | def lr_at(step: int, args, start_step: int = 0) -> float: |
| REPOSITORY_ARCHITECTURE.md:131 | train.py | 684–742 |  | reattributed | quoted code is in train.py, not muon.py |  |
| REPOSITORY_ARCHITECTURE.md:131 | train.py | 684–742 | 847–905 | same | for step in range(start_step, end): | for step in range(start_step, end): |
| REPOSITORY_ARCHITECTURE.md:146 | model.py | 818 |  | reattributed | 'new_cache' is in model.py, not sample.py |  |
| REPOSITORY_ARCHITECTURE.md:146 | model.py | 818 | 861 | same | def new_cache(self) -> list[dict]: | def new_cache(self) -> list[dict]: |
| REPOSITORY_ARCHITECTURE.md:148 | model.py | 441–447 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:148 | model.py | 441–447 | 463–469 | same | if cache is not None: | if cache is not None: |
| REPOSITORY_ARCHITECTURE.md:149 | model.py | 520–528 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:149 | model.py | 520–528 | 548–556 | same | if cache is not None: | if cache is not None: |
| REPOSITORY_ARCHITECTURE.md:154 | model.py | 344–360 |  | reattributed | quoted code is in model.py, not sample.py |  |
| REPOSITORY_ARCHITECTURE.md:154 | model.py | 344–360 | 358–374 | same | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi | def _attn_mask(S: int, T: int, start: int, window: Optional[int], devi |
| REPOSITORY_ARCHITECTURE.md:156 | model.py | 1087–1090 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:156 | model.py | 1087–1090 | 1136–1139 | same | if cache is None or start + pending.shape[1] > bs: | if cache is None or start + pending.shape[1] > bs: |
| REPOSITORY_ARCHITECTURE.md:161 | api.py | 188 | 193 | same | top_p: Optional[float] = Field(None, description="Accepted for compati | top_p: Optional[float] = Field(None, description="Accepted for compati |
| REPOSITORY_ARCHITECTURE.md:162 | modeling_anulm.py | 258–263 | 285–290 | same | def _reorder_cache(past_key_values, beam_idx): | def _reorder_cache(past_key_values, beam_idx): |
| REPOSITORY_ARCHITECTURE.md:164 | model.py | 836–839 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:164 | model.py | 836–839 | 879–882 | same | if embeds is not None: | if embeds is not None: |
| REPOSITORY_ARCHITECTURE.md:164 | model.py | 959–1005 |  | reattributed | 'decode_from_embeds' is in model.py, not modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:164 | model.py | 959–1005 | 1004–1050 | same | def decode_from_embeds(self, prefix, max_tokens: int, stop, no_repeat_ | def decode_from_embeds(self, prefix, max_tokens: int, stop, no_repeat_ |
| REPOSITORY_ARCHITECTURE.md:173 | model.py | 669–749 |  | reattributed | '_route_grouped' is in model.py, not modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:173 | model.py | 669–749 | 702–783 | same | def _route_sparse(self, x_flat, topk_idx, topk_w): | def _route_sparse(self, x_flat, topk_idx, topk_w): |
| REPOSITORY_ARCHITECTURE.md:174 | model.py | 576–587 |  | reattributed | 'Router.forward' is in model.py, not modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:174 | model.py | 576–587 | 604–615 | same | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) | def forward(self, x_flat, seq_len: Optional[int] = None):   # (N, D) |
| REPOSITORY_ARCHITECTURE.md:175 | model.py | 612–635 |  | reattributed | 'Router._gate' is in model.py, not modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:175 | model.py | 612–635 | 640–663 | same | aux = None | aux = None |
| REPOSITORY_ARCHITECTURE.md:175 | model.py | 1014–1044 |  | reattributed | 'AnuLM.update_expert_biases' is in model.py, not modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:175 | model.py | 1014–1044 | 1059–1089 | same | @torch.no_grad() | @torch.no_grad() |
| REPOSITORY_ARCHITECTURE.md:176 | model.py | 847–852 |  | reattributed | 'AnuLM.forward' is in model.py, not modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:176 | model.py | 847–852 | 890–895 | same | if self.grad_ckpt and self.training: | if self.grad_ckpt and self.training: |
| REPOSITORY_ARCHITECTURE.md:177 | model.py | 875–901 |  | reattributed | '_chunked_loss' is in model.py, not modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:177 | model.py | 875–901 | 918–944 | same | def enable_chunked_loss(self, rows: int = 2048): | def enable_chunked_loss(self, rows: int = 2048): |
| REPOSITORY_ARCHITECTURE.md:178 | model.py | 369–390 |  | reattributed | 'merge_tree_matrix' is in model.py, not modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:178 | model.py | 369–390 | 383–404 | same | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: | def merge_tree_matrix(merges, vocab_size: int) -> torch.Tensor: |
| REPOSITORY_ARCHITECTURE.md:179 | model.py | 805–808 |  | reattributed | 'AnuLM.__init__' is in model.py, not modeling_anulm.py |  |
| REPOSITORY_ARCHITECTURE.md:179 | model.py | 805–808 | 848–851 | same | # nanoGPT's scaled init for the residual output projections. | # nanoGPT's scaled init for the residual output projections. |
| REPOSITORY_ARCHITECTURE.md:180 | train.py | 498–503 | 649–654 | same | torch.manual_seed(args.seed) | torch.manual_seed(args.seed) |
| REPOSITORY_ARCHITECTURE.md:180 | train.py | 326 | 467 | same | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam | fused = args.device.startswith("cuda") and "fused" in torch.optim.Adam |
| REPOSITORY_ARCHITECTURE.md:180 | train.py | 590–610 | 752–772 | same | if args.compile: | if args.compile: |
| REPOSITORY_ARCHITECTURE.md:181 | train.py | 691–707 | 854–870 | same | for _ in range(args.grad_accum): | for _ in range(args.grad_accum): |
| REPOSITORY_ARCHITECTURE.md:182 | train.py | 628–675 |  | override | decided in repin-overrides.tsv |  |
| REPOSITORY_ARCHITECTURE.md:182 | train.py | 628–675 | 790–837 | same | start_step, best_val = 0, float("inf") | start_step, best_val = 0, float("inf") |
| REPOSITORY_ARCHITECTURE.md:183 | train.py | 340–396 | 481–537 | same | @torch.no_grad() | @torch.no_grad() |
| REPOSITORY_ARCHITECTURE.md:184 | train.py | 164–178 | 206–220 | same | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  | def interleaved_split(data: torch.Tensor, chunk: int = 262144, every:  |
| REPOSITORY_ARCHITECTURE.md:187 | model.py | 1205–1224 |  | reattributed | 'fit_memory' is in model.py, not bigram_teacher.py |  |
| REPOSITORY_ARCHITECTURE.md:187 | model.py | 1205–1224 | 1257–1276 | same | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i | def fit_memory(model: "AnuLM", big_params: float = 500e6, big_vocab: i |
| FRONT_MATTER.md:33 | model.py | 393–397 | 407–411 | same | def _sdpa(q, k, v, mask, S, scale, dropout_p): | def _sdpa(q, k, v, mask, S, scale, dropout_p): |
| README.md:61 | model.py | 580–586 | 608–614 | same | # fp32 router: top-k is discrete, and bf16 rounding would flip choices | # fp32 router: top-k is discrete, and bf16 rounding would flip choices |
| BOOK_PLAN.md:87 | model.py | 867–873 |  | reattributed | 'AnuLM.forward' is in model.py, not bidirectional.py |  |
| BOOK_PLAN.md:87 | model.py | 867–873 | 910–916 | same | logits = self._head(x) | logits = self._head(x) |
| BOOK_PLAN.md:87 | train.py | 740 | 903 | same | bpb = val / math.log(2) / args.bytes_per_token | bpb = val / math.log(2) / args.bytes_per_token |
| BOOK_PLAN.md:94 | model.py | 428–434 |  | reattributed | 'GQAttention.forward' is in model.py, not train.py |  |
| BOOK_PLAN.md:94 | model.py | 428–434 | 450–456 | same | B, S, _ = x.shape | B, S, _ = x.shape |
| BOOK_PLAN.md:102 | train.py | 702 | 865 | same | (loss / args.grad_accum).backward() | (loss / args.grad_accum).backward() |
| STYLE.md:111 | model.py | 426–465 | 444–487 | same | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in | def forward(self, x, cos, sin, cache: Optional[dict] = None, start: in |
| RESEARCH_REFERENCES.md:203 | model.py | 126 | 131 | same | "moe_impl='grouped' needs torch.nn.functional.grouped_mm (present in 2 | "moe_impl='grouped' needs torch.nn.functional.grouped_mm (present in 2 |
| RESEARCH_REFERENCES.md:270 | model.py | 104–105 |  | override | decided in repin-overrides.tsv |  |
| RESEARCH_REFERENCES.md:270 | model.py | 104–105 | 109–110 | same | yarn_beta_fast: float = 32.0 | yarn_beta_fast: float = 32.0 |
| RESEARCH_REFERENCES.md:270 | model.py | 424 |  | override | decided in repin-overrides.tsv |  |
| RESEARCH_REFERENCES.md:270 | model.py | 424 | 442 | same | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) | self.scale = cfg.head_dim ** -0.5 * (rotary.attn_mscale ** 2) |
| RESEARCH_REFERENCES.md:273 | model.py | 418–420 |  | override | decided in repin-overrides.tsv |  |
| RESEARCH_REFERENCES.md:273 | model.py | 418–420 | 436–438 | same | if cfg.use_qk_norm: | if cfg.use_qk_norm: |
