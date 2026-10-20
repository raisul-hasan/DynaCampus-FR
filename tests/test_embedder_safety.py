"""Offline safeguards against accidentally reporting synthetic recognition."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.recognition import FaceEmbedder


class EmbedderSafetyTests(unittest.TestCase):
    def test_missing_checkpoint_fails(self):
        with self.assertRaises(FileNotFoundError):
            FaceEmbedder(model_name='arcface', model_path='missing-checkpoint-for-safety-test.onnx')

    def test_unimplemented_backbone_is_not_renamed_arcface(self):
        with self.assertRaisesRegex(ValueError, 'not implemented'):
            FaceEmbedder(model_name='adaface_ir50')

    def test_loader_failure_never_creates_mock(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'invalid.onnx'
            path.write_bytes(b'invalid')
            with patch('insightface.model_zoo.get_model', side_effect=RuntimeError('load failed')):
                with self.assertRaisesRegex(RuntimeError, 'synthetic fallback is disabled'):
                    FaceEmbedder(model_name='arcface', model_path=path)

    def test_explicit_mock_remains_available_for_offline_tests(self):
        embedder = FaceEmbedder(mock=True)
        result = embedder.extract_embedding(np.full((112, 112, 3), 128, dtype=np.uint8))
        self.assertEqual(result.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1, places=5)

    def test_lost_real_model_fails_at_inference(self):
        embedder = FaceEmbedder(mock=True)
        embedder.mock = False
        with self.assertRaisesRegex(RuntimeError, 'synthetic fallback is disabled'):
            embedder.extract_embedding(np.zeros((112, 112, 3), dtype=np.uint8))

    def test_bad_channels_are_rejected(self):
        with self.assertRaises(ValueError):
            FaceEmbedder(mock=True).extract_embedding(np.zeros((112, 112, 1), dtype=np.uint8))


if __name__ == '__main__':
    unittest.main()
