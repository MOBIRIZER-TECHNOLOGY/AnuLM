"""Offline regressions for serving, UTF-8, grouped chat data and tokenizers."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from anulm.apps import api
from anulm.data.make_chat import split_chat_rows, question_group
from anulm.tokenization.utf8 import complete_utf8


class Engine:
    device, eos_id = 'cpu', 255
    cfg = SimpleNamespace(block_size=128)
    qa_templates = None
    def __init__(self, values):
        self.values, self.model, self.kwargs = values, self, None
    def encode(self, text):
        return list(text.encode()) or [1]
    def decode_complete(self, values):
        return complete_utf8(bytes(values))
    def generate(self, x, max_tokens, **kwargs):
        self.kwargs = kwargs
        for token in self.values[:max_tokens]:
            if kwargs['on_token'](torch.tensor([[token]])):
                break


class ReleaseFixes(unittest.TestCase):
    def run_generation(self, values, stops=()):
        engine = Engine(values)
        with patch.object(api, 'models', SimpleNamespace(brain=lambda: engine)):
            gen = api.Generation('Q', 100, .3, 0, 1., None, list(stops))
            chunks = []
            value = gen.run(chunks.append)
            self.assertEqual(value, ''.join(chunks))
            return value

    def test_eos_flushes_unmatched_stop_buffer(self):
        self.assertEqual(self.run_generation(list(b'Hello') + [255], ['###']), 'Hello')

    def test_usage_counts_generated_eos_token(self):
        with patch.object(api, 'models', SimpleNamespace(brain=lambda: Engine([65, 255]))):
            gen = api.Generation('Q', 10, .3, 0, 1., 0, [])
            self.assertEqual(gen.run(), 'A')
            self.assertEqual(gen.generated_tokens, 2)

    def test_empty_stop_is_not_an_immediate_stop(self):
        self.assertEqual(api.stops_of(''), [])

    def test_native_engine_omits_eos_control_token_from_hf_text(self):
        import threading
        from anulm.inference.serve import Engine as RealEngine
        engine = RealEngine.__new__(RealEngine)
        engine.cfg, engine.device, engine.eos_id = SimpleNamespace(block_size=128), 'cpu', 0
        engine.lock, engine.qa_templates, engine.coder = threading.Lock(), None, False
        engine.tok = SimpleNamespace(encode=lambda text: [2],
            decode_complete=lambda ids: ''.join({1: 'A', 0: '<|endoftext|>'}[i] for i in ids))
        engine.model = SimpleNamespace(generate=lambda x, *a, **k: torch.cat([x, torch.tensor([[1, 0]])], dim=1))
        self.assertEqual(engine.generate('Q', 10, .3, 0, 0)['completion'], 'A')

    def test_stop_crossing_tokens_is_never_emitted(self):
        self.assertEqual(self.run_generation(list(b'Hello###tail') + [255], ['###']), 'Hello')

    def test_incomplete_utf8_is_omitted_but_valid_replacement_character_survives(self):
        self.assertEqual(self.run_generation(list(b'Hi ') + [0xe0]), 'Hi ')
        self.assertEqual(self.run_generation(list('नमस्ते'.encode()) + [255]), 'नमस्ते')
        self.assertEqual(self.run_generation(list('Hi �'.encode()) + [255]), 'Hi �')

    def test_http_semantics_reject_unsupported_history_sampling_and_bad_limits(self):
        engine = Engine(list(b'OK') + [255])
        model = SimpleNamespace(brain=lambda: engine, device='cpu', _m={})
        with patch.object(api, 'models', model), patch.object(api, 'available', return_value=True), \
             patch.object(api, 'chat_prompt', side_effect=lambda q: q), \
             patch.dict(os.environ, {'ANULM_API_KEY': 'fixture'}), TestClient(api.app) as client:
            headers = {'Authorization': 'Bearer fixture'}
            base = {'messages': [{'role': 'user', 'content': 'Q'}]}
            self.assertEqual(client.get('/v1/models').status_code, 401)
            for change in [{'top_p': .2}, {'messages': [{'role': 'system', 'content': 'X'}, *base['messages']]}]:
                response = client.post('/v1/chat/completions', json={**base, **change}, headers=headers)
                self.assertEqual(response.status_code, 400, response.text)
            self.assertEqual(client.post('/v1/chat/completions', json={**base, 'max_tokens': -5}, headers=headers).status_code, 422)
            self.assertEqual(client.post('/v1/chat/completions', json={**base, 'frequency_penalty': 1}, headers=headers).status_code, 422)
            response = client.post('/v1/chat/completions', json={**base, 'temperature': .3, 'top_k': 0}, headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['choices'][0]['message']['content'], 'OK')
            self.assertIsNone(engine.kwargs['top_k'])

    def test_exact_greedy_selects_first_argmax_even_on_ties(self):
        from anulm.inference.decoding import greedy_generate
        class Model:
            cfg = SimpleNamespace(block_size=16)
            def new_cache(self):
                return []
            def __call__(self, x, **kwargs):
                logits = torch.zeros(x.shape[0], x.shape[1], 100)
                logits[..., 65:67] = 1.
                return logits, None
        for seed in [1, 11]:
            torch.manual_seed(seed)
            self.assertEqual(greedy_generate(Model(), torch.tensor([[81]]), 4)[0].tolist(), [81, 65, 65, 65, 65])

    def test_chat_duplicates_and_identity_prompts_do_not_cross_partitions(self):
        rows = [{'question': '  SAME question ', 'answer': 'A', 'lang': 'en', 'source': 'one'},
                {'question': 'same QUESTION', 'answer': 'A', 'lang': 'en', 'source': 'two'},
                {'question': 'other', 'answer': 'B', 'lang': 'en', 'source': 'one'},
                {'question': 'Who are you?', 'answer': 'AnuLM', 'lang': 'en', 'source': 'identity'},
                {'question': 'who ARE you?', 'answer': 'AnuLM', 'lang': 'en', 'source': 'two'}]
        train, held = split_chat_rows(rows, held_groups=1)
        self.assertFalse({question_group(r) for r in train} & {question_group(r) for r in held})
        self.assertTrue(all(question_group(r) != 'who are you?' for r in held))

    def test_hf_tokenizer_loads_after_export_rename_and_decodes_utf8_boundaries(self):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        from anulm.tokenization.hf import load_tokenizer
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'tokenizer.tokenizer.json'
            tok = Tokenizer(models.BPE())
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
            tok.train_from_iterator(['Hello नमस्ते �'] * 8, trainers.BpeTrainer(vocab_size=280,
                special_tokens=['<|endoftext|>'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
            tok.save(str(path))
            loaded = load_tokenizer(path)
            values = loaded.encode('Hello नमस्ते �')
            self.assertEqual(loaded.decode_complete(values), 'Hello नमस्ते �')
            for cut in range(len(values) + 1):
                text = loaded.decode_complete(values[:cut])
                self.assertFalse(text.endswith('�') and cut < len(values))


if __name__ == '__main__':
    unittest.main()
