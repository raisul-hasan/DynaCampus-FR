"""Verify synthetic degradations and end-to-end failure accounting."""
import unittest
import numpy as np

from scripts.run_robustness import degrade, summarize_stress
from scripts.compare_decisions import apply_rule


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.image = np.random.default_rng(42).integers(0, 256, (80, 120, 3), dtype=np.uint8)

    def test_downsampling_keeps_aspect_and_never_upscales(self):
        small = degrade(self.image, dict(type='resolution', value=60))
        self.assertEqual(small.shape, (40, 60, 3))
        np.testing.assert_array_equal(degrade(self.image, dict(type='resolution', value=200)), self.image)

    def test_transformations_are_deterministic_and_do_not_mutate_source(self):
        source_copy = self.image.copy()
        for kind, value in [('blur', .01), ('resolution', 32), ('jpeg', 10), ('original', 0)]:
            condition = dict(type=kind, value=value)
            first = degrade(self.image, condition)
            np.testing.assert_array_equal(first, degrade(self.image, condition))
            self.assertFalse(np.shares_memory(first, self.image))
        np.testing.assert_array_equal(self.image, source_copy)

    def test_blur_and_jpeg_change_pixels_without_changing_dimensions(self):
        for kind, value in [('blur', .02), ('jpeg', 10)]:
            result = degrade(self.image, dict(type=kind, value=value))
            self.assertEqual(result.shape, self.image.shape)
            self.assertFalse(np.array_equal(result, self.image))

    def test_invalid_conditions_raise(self):
        for kind, value in [('blur', 0), ('resolution', 0), ('jpeg', 101), ('noise', 1)]:
            with self.assertRaises(ValueError):
                degrade(self.image, dict(type=kind, value=value))

    def test_failures_remain_in_end_to_end_denominators(self):
        rows = [dict(subject_id='a', truth_status='known', status='ok', quality=.8,
                     top1_id='a', top1_score=.9, top2_score=.1, rank1_correct=True),
                dict(subject_id='b', truth_status='known', status='no_face', quality='',
                     top1_id='', top1_score='', top2_score='', rank1_correct=False),
                dict(subject_id='u', truth_status='unknown', status='ok', quality=.8,
                     top1_id='a', top1_score=.1, top2_score=.05, rank1_correct=False)]
        thresholds = dict(score_threshold=.5, quality_threshold=0., margin_threshold=.15)
        result = summarize_stress(apply_rule(rows, thresholds, 'test'), thresholds, 'test', dict(name='blur', type='blur', value=.01))
        self.assertEqual(result['rank1_end_to_end'], .5)
        self.assertEqual(result['correct_known_accept_end_to_end'], .5)
        self.assertEqual(result['extraction_success_rate'], 2/3)
        self.assertEqual(result['extraction_counts']['no_face'], 1)


if __name__ == '__main__':
    unittest.main()
