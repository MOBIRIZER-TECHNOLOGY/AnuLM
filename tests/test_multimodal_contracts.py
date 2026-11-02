"""CPU-only multimodal contract audit; no optimizer, corpus, or pretrained download.

Run with python -m unittest tests.test_multimodal_contracts. Frozen
encoders are represented by synthetic feature producers; this does not assess
their quality or replace a later real-encoder integration check.
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from anulm import model as M
from anulm.multimodal import speech_encoder as A, vision_encoder as V
from anulm.multimodal import speech_vocab as S, omni_train as O, audio_codec as C
from anulm.tokenization import hf as H

import numpy as np
import torch

torch.set_num_threads(2)


def tiny():
    vocab = S.SpeechVocab(64)
    cfg = M.AnuLMConfig(vocab_size=vocab.total, block_size=64, n_layer=2,
                       hidden_size=16, n_head=2, head_dim=8, n_kv_head=1,
                       intermediate_size=32, first_k_dense_replace=2,
                       num_experts=4, num_experts_per_tok=2, n_group=1, topk_group=1)
    return M.AnuLM(cfg).cpu(), vocab


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_modality_ranges_are_disjoint_and_roundtrip(self):
        vocab = S.SpeechVocab(64)
        control = [vocab.speech_bos, vocab.speech_eos, vocab.text_bos, vocab.image_bos, vocab.image_eos]
        self.assertEqual(len(set(control)), 5)
        self.assertEqual(min(control), 64 + C.AUDIO_VOCAB)
        self.assertEqual(max(control), vocab.total - 1)
        raw = torch.tensor([0, C.AUDIO_VOCAB - 1])
        self.assertTrue(torch.equal(vocab.to_snac(vocab.audio(raw)), raw))
        self.assertFalse(vocab.is_audio(torch.tensor(control)).any())

    def test_discrete_audio_frames_roundtrip_with_separate_slot_ranges(self):
        codes = [np.array([[0, 4095]]), np.array([[1, 2, 4094, 4095]]),
                 np.arange(8).reshape(1, 8)]
        flat = C.flatten(codes)
        self.assertEqual(flat.size, 2 * C.FRAME_SLOTS)
        for slot in range(C.FRAME_SLOTS):
            self.assertTrue(np.all(flat[slot::C.FRAME_SLOTS] // C.LEVEL_VOCAB == slot))
        restored, out_of_range = C.unflatten(flat, report=True)
        self.assertEqual(out_of_range, 0)
        for actual, expected in zip(restored, codes):
            self.assertTrue(np.array_equal(actual, expected))

    def test_image_pooling_preserves_two_dimensional_neighbours(self):
        projector = V.ImageProjector(1, 4, pool=2, hidden=4)
        observed = []
        handle = projector.net[0].register_forward_pre_hook(lambda module, args: observed.append(args[0].detach()))
        projector(torch.arange(16, dtype=torch.float32).reshape(16, 1))
        handle.remove()
        expected = torch.tensor([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]], dtype=torch.float32)
        self.assertTrue(torch.equal(observed[0], expected))

    def test_audio_stacking_uses_only_complete_frame_groups(self):
        projector = A.AudioProjector(2, 4, stack=4, hidden=8)
        observed = []
        hook = projector.net[0].register_forward_pre_hook(lambda module, args: observed.append(args[0].detach()))
        states = torch.arange(14, dtype=torch.float32).reshape(1, 7, 2)
        output = projector(states)
        hook.remove()
        self.assertEqual(tuple(output.shape), (1, 1, 4))
        self.assertTrue(torch.equal(observed[0], states[:, :4].reshape(1, 1, 8)))

    def test_audio_answer_mask_is_shifted_and_prompt_is_ungraded(self):
        model, vocab = tiny()
        speech = A.ContinuousSpeech(model, vocab, in_dim=4, stack=4)
        answer = [7, 8, 63]
        embeds, targets = speech.build(torch.randn(12, 4), [], answer)
        start = 1 + 3 + 1
        self.assertEqual(tuple(embeds.shape), (start + len(answer), 16))
        self.assertTrue(torch.all(targets[:start - 1] == -1))
        self.assertEqual(targets[start - 1:start - 1 + len(answer)].tolist(), answer)
        self.assertEqual(targets[-1].item(), -1)

    def test_image_question_and_prompt_are_not_target_tokens(self):
        model, vocab = tiny()
        vision = V.VisionLanguage(model, vocab, in_dim=4, pool=2)
        answer, question = [7, 8, 63], [3, 4, 5]
        embeds, targets = vision.build(torch.randn(16, 4), answer, question)
        start = 1 + 4 + 1 + len(question)
        self.assertEqual(tuple(embeds.shape), (start + len(answer), 16))
        self.assertTrue(torch.all(targets[:start - 1] == -1))
        self.assertEqual(targets[start - 1:start - 1 + len(answer)].tolist(), answer)
        self.assertEqual(targets[-1].item(), -1)

    def test_shared_decoder_and_both_projectors_receive_finite_gradients(self):
        model, vocab = tiny()
        speech = A.ContinuousSpeech(model, vocab, 4, stack=4)
        vision = V.VisionLanguage(model, vocab, 4, pool=2)
        self.assertIs(speech.model, vision.model)
        audio_embeds, audio_targets = speech.build(torch.randn(12, 4), [], [7, 8, 63])
        image_embeds, image_targets = vision.build(torch.randn(16, 4), [9, 10, 63], [3, 4])
        _, audio_loss = model(None, audio_targets[None], embeds=audio_embeds[None])
        _, image_loss = model(None, image_targets[None], embeds=image_embeds[None])
        (audio_loss + image_loss).backward()  # Differentiability only: no optimizer/update.
        for component in [model, speech.proj, vision.proj]:
            gradients = [p.grad for p in component.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(torch.count_nonzero(g).item() for g in gradients))

    def test_joint_checkpoint_keeps_both_projectors_and_backbone(self):
        model, vocab = tiny()
        audio = A.ContinuousSpeech(model, vocab, 4, stack=4)
        image = V.VisionLanguage(model, vocab, 4, pool=2)
        omni = O.Omni.__new__(O.Omni)
        omni.model, omni.sm, omni.vl, omni.cfg = model, audio, image, model.cfg
        omni.ck = {'cfg': model.cfg, 'speech_text_vocab': vocab.text}
        omni.args = SimpleNamespace(ckpt='synthetic-only', whisper='synthetic', stack=4, tower='synthetic', pool=2)
        omni.tasks = [O.Task('chat=text:unused'), O.Task('asr=asr:unused'), O.Task('caption=image:unused')]
        state = omni.state(0, 1.0, {'chat': 1., 'asr': 1., 'caption': 1.})
        self.assertTrue({'model', 'proj', 'vision_proj', 'speech_text_vocab', 'whisper', 'tower'} <= state.keys())
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, map_location='cpu', weights_only=False)
        clone, _ = tiny()
        clone.load_state_dict(restored['model'])
        audio_clone = A.ContinuousSpeech(clone, vocab, 4, stack=4)
        image_clone = V.VisionLanguage(clone, vocab, 4, pool=2)
        audio_clone.proj.load_state_dict(restored['proj'])
        image_clone.proj.load_state_dict(restored['vision_proj'])
        for actual, expected in [(clone, model), (audio_clone.proj, audio.proj), (image_clone.proj, image.proj)]:
            for key, tensor in actual.state_dict().items():
                self.assertTrue(torch.equal(tensor, expected.state_dict()[key]))

    def test_image_cache_does_not_reuse_features_for_different_images_of_same_count(self):
        class Tower:
            dim, name = 2, 'synthetic-only'
            def encode(self, path):
                return torch.full((4, self.dim), float(Path(path).read_text()))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / 'a.fake', root / 'b.fake'
            a.write_text('1')
            b.write_text('9')
            first, _ = V.build_cache([{'image': str(a)}], Tower(), root / 'cache.f16')
            self.assertTrue(np.all(first == 1))
            first._mmap.close()
            del first
            second, _ = V.build_cache([{'image': str(b)}], Tower(), root / 'cache.f16')
            actual = np.array(second, copy=True)
            second._mmap.close()
            del second
            self.assertTrue(np.all(actual == 9), 'Cache silently reused features from a different image.')

    def test_cache_tracks_content_encoder_and_dtype_and_reuses_valid_features(self):
        class Tower:
            dim, name, transform = 2, 'synthetic-only', 'transform-v1'
            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False).eval().requires_grad_(False)
                self.model.weight.fill_(1)
                self.calls = 0
            def encode(self, path):
                self.calls += 1
                return torch.full((4, 2), float(Path(path).read_text()) * self.model.weight.item())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'a.fake'
            image.write_text('2')
            tower = Tower()
            def cached(dtype='float16'):
                mm, _ = V.build_cache([{'image': str(image)}], tower, root / 'cache', dtype)
                result = np.array(mm, copy=True)
                mm._mmap.close()
                return result
            self.assertTrue(np.all(cached() == 2))
            before = tower.calls
            cached()
            self.assertEqual(tower.calls - before, 1)  # Shape probe only: valid cache reused.
            image.write_text('3')
            self.assertTrue(np.all(cached() == 3))
            tower.model.weight.fill_(2)
            self.assertTrue(np.all(cached() == 6))
            tower.transform = 'transform-v2'
            before = tower.calls
            cached()
            self.assertEqual(tower.calls - before, 2)
            self.assertEqual(cached('float32').dtype, np.dtype('float32'))
            tower.model.requires_grad_(True)
            with self.assertRaisesRegex(ValueError, 'frozen'):
                cached()

    def test_explicit_split_rejects_overlap_and_empty_partitions(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            O.check_split([{'image': 'a'}], [{'image': './a'}], 'image', 'caption')
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            O.check_split([{'image': 'a'}], [], 'image', 'caption')
        with self.assertRaisesRegex(ValueError, 'two independent'):
            O.split_rows([{'image': 'a'}, {'image': './a'}], 'image')

    def test_automatic_image_split_keeps_same_image_out_of_both_partitions(self):
        model, vocab = tiny()
        rows = [{'image': 'same.fake', 'text': 'caption one'},
                {'image': 'same.fake', 'text': 'caption two'},
                {'image': 'other.fake', 'text': 'caption three'}]
        args = SimpleNamespace(device='cpu', ckpt='unused', grad_ckpt=False,
                               task=['caption=image:unused'], held=None, tower='synthetic', pool=2,
                               vision_proj_from=None, audio_proj_from=None)
        ck = {'model': model.state_dict(), 'cfg': model.cfg, 'speech_text_vocab': vocab.text}
        with mock.patch.object(M, 'load_checkpoint', return_value=ck), \
             mock.patch.object(M, 'fit_memory'), \
             mock.patch.object(H, 'load_tokenizer', return_value=object()), \
             mock.patch.object(V, 'VisionTower', return_value=SimpleNamespace(dim=4)), \
             mock.patch.object(V, 'build_cache', return_value=(None, {})), \
             mock.patch.object(O, 'load_rows', return_value=rows):
            omni = O.Omni(args)
        training = {r['image'] for r in omni.tasks[0].rows}
        held = {r['image'] for r in omni.tasks[0].held}
        self.assertFalse(training & held, 'The same image is in training and held-out data.')

    def test_long_audio_is_handled_or_explicitly_rejected_instead_of_silently_cropped(self):
        encoder = C.WhisperEncoder.__new__(C.WhisperEncoder)
        class Encoded:
            def __init__(self, features):
                self.array = np.zeros((1, features.shape[-1] // 2, 4), dtype=np.float32)
            def to_device(self, device):
                return self.array
        encoder.ct2 = SimpleNamespace(StorageView=SimpleNamespace(from_array=lambda x: x), Device=SimpleNamespace(cpu='cpu'))
        encoder.whisper = SimpleNamespace(feature_extractor=lambda x: np.zeros((80, 3100), np.float32),
                                            model=SimpleNamespace(encode=lambda features: Encoded(features)))
        try:
            features = encoder.encode(np.zeros(31 * C.ASR_RATE, dtype=np.float32))
        except (ValueError, NotImplementedError) as error:
            self.assertRegex(str(error).lower(), r'long|duration|second|chunk')
        else:
            self.assertGreaterEqual(features.shape[0], 1550, 'A 31-second clip was silently cropped to 30 seconds.')


if __name__ == '__main__':
    unittest.main(verbosity=2)
