"""Input-contract and real-model failure safeguards without synthetic research."""
import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from src.recognition.adaface import AdaFaceEmbedder, bgr_tensor, load_official_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class AdaFaceTests(unittest.TestCase):
    def test_preprocessing_preserves_bgr_channel_order_and_exact_range(self):
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        crop[:] = [0, 127, 255]
        tensor = bgr_tensor(crop)
        self.assertEqual(tuple(tensor.shape), (1, 3, 112, 112))
        self.assertEqual(tensor.dtype, torch.float32)
        np.testing.assert_allclose(tensor[0, :, 0, 0].numpy(), [-1., (127 / 255. - .5) / .5, 1.])

    def test_single_and_batch_preprocessing_match(self):
        crop = np.random.default_rng(12).integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
        np.testing.assert_array_equal(bgr_tensor(crop)[0], bgr_tensor(np.stack([crop, crop]))[1])

    def test_matches_pinned_official_reference_function(self):
        source = ROOT / 'data/models/adaface/upstream/inference.py'
        if not source.exists():
            self.skipTest('Official reference source not downloaded')
        function = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef) and n.name == 'to_input')
        scope = dict(np=np, torch=torch)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), scope)
        crop = np.random.default_rng(42).integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
        expected = scope['to_input'](crop[:, :, ::-1].copy())
        np.testing.assert_array_equal(bgr_tensor(crop).numpy(), expected.numpy())

    def test_invalid_crops_fail_explicitly(self):
        for crop in [None, np.zeros((112, 112, 3)), np.zeros((111, 112, 3), dtype=np.uint8),
                     np.zeros((0, 112, 112, 3), dtype=np.uint8), np.zeros((112, 112), dtype=np.uint8)]:
            with self.assertRaises(ValueError):
                bgr_tensor(crop)

    def test_missing_real_model_cannot_produce_a_feature(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                AdaFaceEmbedder(directory)

    def test_unknown_checkpoint_globals_are_rejected_before_loading(self):
        with patch('torch.serialization.get_unsafe_globals_in_checkpoint', return_value=['unrelated.CustomClass']):
            with patch('torch.load') as loader:
                with self.assertRaisesRegex(ValueError, 'Unexpected checkpoint'):
                    load_official_checkpoint('unused.ckpt')
                loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
