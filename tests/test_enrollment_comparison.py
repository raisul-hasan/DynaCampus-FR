"""Guard against image reuse and incorrect paired enrollment comparisons."""
import unittest

import numpy as np

from scripts.compare_enrollment import gallery_candidates, paired_bootstrap, reject_overlap, unit_mean


class EnrollmentComparisonTests(unittest.TestCase):
    def test_mean_is_normalized_after_averaging(self):
        actual = unit_mean([[1., 0.], [0., 1.]])
        np.testing.assert_allclose(actual, [2 ** -0.5, 2 ** -0.5], atol=1e-6)

    def test_masks_never_enter_clean_enrollment(self):
        rows = [dict(dataset='halfface', subject_id='s1', mask_status=mask, pose=pose, path=path)
                for mask, pose, path in [('masked', 'front', 'a'), ('unmasked', 'right', 'b'),
                                         ('unmasked', 'front', 'c'), ('unmasked', 'left', 'd')]]
        one = gallery_candidates(rows, 'halfface', ['s1'], 'one_photo')['s1']
        few = gallery_candidates(rows, 'halfface', ['s1'], 'up_to_three_photos')['s1']
        self.assertEqual([r['path'] for r in one], ['c'])
        self.assertEqual([r['path'] for r in few], ['c', 'd', 'b'])

    def test_enrollment_probe_reuse_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'leakage'):
            reject_overlap([{'path': 'same.jpg'}], [{'path': 'same.jpg'}])

    def test_paired_bootstrap_requires_identical_probes(self):
        one = [{'subject_id': 's1', 'path': 'a', 'truth_status': 'known', 'rank1_correct': True}]
        few = [{'subject_id': 's1', 'path': 'b', 'truth_status': 'known', 'rank1_correct': True}]
        with self.assertRaisesRegex(ValueError, 'different known probes'):
            paired_bootstrap(one, few, ['s1'], 20, 42, 'rank1_correct')

    def test_identical_results_have_zero_paired_effect(self):
        rows = [{'subject_id': subject, 'path': subject, 'truth_status': 'known', 'rank1_correct': value}
                for subject, value in [('s1', True), ('s2', False)]]
        result = paired_bootstrap(rows, rows, ['s1', 's2'], 100, 42, 'rank1_correct')
        self.assertEqual(result['delta_percentage_points'], 0)
        self.assertEqual(result['ci95_low_percentage_points'], 0)
        self.assertEqual(result['ci95_high_percentage_points'], 0)


if __name__ == '__main__':
    unittest.main()
