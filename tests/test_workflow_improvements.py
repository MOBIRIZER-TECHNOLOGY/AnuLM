"""Regression checks for selectively adapted V2 workflows; offline CPU only."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from anulm.training import finetune as F
from anulm.multimodal import omni_train as O, audio_codec as A
from anulm.evaluation.eval_code import mbpp_entry
from anulm.evaluation.report import gate, load_plan
from anulm.data.decontaminate import ngrams, filter_corpus
from anulm.training.from_scratch import split_documents

torch.set_num_threads(2)


class Workflows(unittest.TestCase):
    def test_packed_cache_binds_content_tokenizer_and_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, tokenizer = root / 'data.jsonl', root / 'tok.json'
            rows.write_text('{"question":"a","answer":"b"}\n')
            tokenizer.write_text('{}')
            with patch.object(F, 'encode_pairs_parallel', return_value=[([1], [2])]) as encode:
                F.load_packed(str(rows), str(tokenizer), 8, 3, 1)
                F.load_packed(str(rows), str(tokenizer), 8, 3, 1)
                self.assertEqual(encode.call_count, 1)
                rows.write_text('{"question":"x","answer":"b"}\n')
                F.load_packed(str(rows), str(tokenizer), 8, 3, 1)
                tokenizer.write_text('{"changed":true}')
                F.load_packed(str(rows), str(tokenizer), 8, 3, 1)
                self.assertEqual(encode.call_count, 3)
                cache = Path(f'{rows}.tok.b8.pt')
                blob = torch.load(cache, weights_only=True)
                blob['x'] = torch.zeros(2, 2)
                torch.save(blob, cache)
                F.load_packed(str(rows), str(tokenizer), 8, 3, 1)
                cache.write_bytes(b'broken')
                F.load_packed(str(rows), str(tokenizer), 8, 3, 1)
                self.assertEqual(encode.call_count, 5)

    def test_long_audio_windows_and_input_validation(self):
        audio = np.arange(65 * A.ASR_RATE, dtype=np.float32)
        chunks = A.audio_chunks(audio)
        self.assertEqual([len(c) for c in chunks], [480000, 480000, 80000])
        np.testing.assert_array_equal(np.concatenate(chunks), audio)
        for invalid in [[], np.zeros((2, 2)), [float('nan')]]:
            with self.assertRaises(ValueError):
                A.audio_chunks(invalid)

    def test_mbpp_wrapped_call_ignores_helper_substrings(self):
        row = {'code': 'def a():\n return 0\ndef volume_sphere(r):\n return r**3',
               'test_list': ['assert math.isclose(volume_sphere(10), 1000)']}
        self.assertEqual(mbpp_entry(row), 'volume_sphere')

    def test_missing_and_nonfinite_metrics_fail_quality_gate(self):
        rule = [{'metric': 'wer', 'op': '<=', 'threshold': .2}]
        self.assertFalse(gate({}, rule))
        self.assertFalse(gate({'wer': float('nan')}, rule))
        self.assertFalse(gate({'wer': .1}, []))
        self.assertTrue(gate({'wer': .1}, rule))

    def test_evaluation_plan_requires_full_declared_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / 'plan.json'
            p.write_text(json.dumps({'required_tasks': ['hi', 'en'], 'tasks': [{'name': 'hi'}]}))
            with self.assertRaisesRegex(ValueError, 'coverage'):
                load_plan(p)

    def test_decontamination_checks_every_offset_and_fails_without_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ref, corpus = root / 'ref.jsonl', root / 'corpus.txt'
            reference = 'one two three four five six seven eight nine ten eleven twelve thirteen'
            ref.write_text(json.dumps({'text': reference}) + '\n')
            from anulm.tokenization.bpe import DOC_SEP
            corpus.write_text('extra ' + reference + DOC_SEP + 'unrelated document' + DOC_SEP)
            result = filter_corpus(corpus, [ref], root / 'clean.txt', ['text'])
            self.assertEqual((result['kept'], result['dropped']), (1, 1))
            with self.assertRaisesRegex(ValueError, 'No usable'):
                filter_corpus(corpus, [ref], root / 'bad.txt', ['missing'])
            hindi = 'भारत में अनेक भाषाएँ बोली जाती हैं और सभी भाषाओं की अपनी अलग पहचान है'
            self.assertTrue(ngrams(hindi, 5) & ngrams('नमस्ते ' + hindi, 5))

    def test_document_split_deduplicates_before_tokenizer_training(self):
        train, validation = split_documents('first document\n\nsecond document\n\nfirst document')
        self.assertEqual(len(train) + len(validation), 2)
        self.assertFalse(set(train) & set(validation))

    def test_invalid_accumulation_fails_before_loading(self):
        with patch.object(O, 'Omni') as loader:
            with self.assertRaises(ValueError):
                O.train(SimpleNamespace(accum=0))
            loader.assert_not_called()

    def test_opt_in_projector_norm_scaling_and_zero_guard(self):
        omni = O.Omni.__new__(O.Omni)
        embeddings = torch.nn.Embedding(8, 4)
        with torch.no_grad():
            embeddings.weight.fill_(.2)
        omni.model = SimpleNamespace(embed_tokens=embeddings)
        omni.vocab = SimpleNamespace(text=8)
        projector = SimpleNamespace(net=torch.nn.Sequential(torch.nn.LayerNorm(4)))
        omni._match_scale(projector)
        self.assertAlmostEqual(projector.net[-1].weight.norm().item(), .4, places=5)
        with torch.no_grad():
            embeddings.weight.zero_()
        with self.assertRaises(ValueError):
            omni._match_scale(projector)

    def test_legacy_precision_preservation_is_explicit_opt_in(self):
        from anulm.apps.omni import Models
        model = torch.nn.Module()
        model.large = torch.nn.Parameter(torch.ones(100000))
        model.small = torch.nn.Parameter(torch.ones(8))
        Models('cuda', '')._half(model, preserve_small_fp32=True)
        self.assertEqual(model.large.dtype, torch.bfloat16)
        self.assertEqual(model.small.dtype, torch.float32)

    def test_accumulation_parity_resume_guard_and_recovery(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(()))
            def forward(self, *args, **kwargs):
                return None, (self.weight - 1).square()
            def update_expert_biases(self):
                return 1.0
        class Omni:
            def __init__(self, args):
                self.model, self.dev, self.sm, self.vl = Model(), 'cpu', None, None
                self.tasks = [SimpleNamespace(name='text', weight=1., epoch=0)]
            def held_loss(self, *args):
                return 1.
            def next_batch(self, task):
                return None, torch.ones(1)
            def state(self, step, score, losses):
                return {'model': self.model.state_dict(), 'step': step}
        with tempfile.TemporaryDirectory() as temp, patch.object(O, 'Omni', Omni), contextlib.redirect_stdout(io.StringIO()):
            states = []
            for accum in [1, 2]:
                args = SimpleNamespace(out=str(Path(temp) / f'{accum}.pt'), accum=accum,
                    seed=1, lr=.01, proj_lr=.01, resume=False, steps=1, warmup=1, log_every=1, eval_every=1)
                O.train(args)
                state = torch.load(args.out + '.last', weights_only=False)
                self.assertEqual(state['seen']['text'], accum)
                states.append(state['model']['weight'])
                args.resume, args.accum, args.steps = True, accum + 1, 2
                with self.assertRaisesRegex(ValueError, 'Resume'):
                    O.train(args)
            torch.testing.assert_close(states[0], states[1])


    def test_long_audio_transcription_preserves_chunk_order(self):
        """Continuous ASR must decode each long-recording chunk independently and
        join the texts in audio order, not return only Whisper's first 30 seconds."""
        from types import SimpleNamespace
        import numpy as np
        from anulm.multimodal.audio_codec import ASR_RATE
        from anulm.multimodal.speech_encoder import transcribe

        class Encoder:
            def __init__(self):
                self.sizes = []

            def encode(self, audio):
                self.sizes.append(len(audio))
                return np.zeros((4, 2), dtype=np.float32)

        class Projector:
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def parameters(self):
                return iter([self.weight])

            def __call__(self, states):
                return states

        class Model:
            def __init__(self):
                self.calls = 0

            def embed_tokens(self, ids):
                return torch.zeros((len(ids), 2), dtype=torch.float32)

            def decode_from_embeds(self, prefix, *args, **kwargs):
                self.calls += 1
                return torch.tensor([self.calls])

        class Tokenizer:
            eos_id = 99

            def __init__(self, texts):
                self.texts = texts

            def decode(self, ids):
                return self.texts[int(ids[0]) - 1]

        def run(audio, texts):
            model = Model()
            sm = SimpleNamespace(
                proj=Projector(), model=model,
                vocab=SimpleNamespace(speech_bos=1, text_bos=2, text=100))
            enc = Encoder()
            text = transcribe(sm, enc, Tokenizer(texts), audio, device="cpu")
            return text, enc.sizes

        text, sizes = run(np.zeros(65 * ASR_RATE, dtype=np.float32), ["one", "two", "three"])
        assert text == "one two three"
        assert sizes == [30 * ASR_RATE, 30 * ASR_RATE, 5 * ASR_RATE]
        text, sizes = run(np.zeros(10, dtype=np.float32), ["short\nignored"])
        assert text == "short" and len(sizes) == 1


if __name__ == '__main__':
    unittest.main()
