"""Scientific safeguards for the development enrollment-pooling comparison."""
import unittest
import numpy as np
from scripts.compare_enrollment_pooling import assert_development, pool_template, apply_decision, summarize


class EnrollmentPoolingTests(unittest.TestCase):
    def setUp(self):
        self.features = np.eye(3, dtype=np.float32)
        self.rows = [dict(status='ok', embedding_index=i, quality=q, det_score=.8)
                     for i, q in enumerate([.1, .9])]

    def test_uniform_weights_recover_mean_and_weighting_changes_direction(self):
        uniform, _, _ = pool_template(self.rows, self.features, 'uniform_mean')
        weighted, weights, _ = pool_template(self.rows, self.features, 'heuristic_weighted')
        np.testing.assert_allclose(uniform, [2**-.5, 2**-.5, 0], atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.norm(weighted)), 1.)
        self.assertGreater(weighted[1], weighted[0])
        np.testing.assert_allclose(weights, [.1, .9])

    def test_weights_are_scale_invariant_and_zero_weights_are_disclosed(self):
        a, _, _ = pool_template(self.rows, self.features, 'heuristic_weighted')
        scaled = [dict(r, quality=float(r['quality']) * .5) for r in self.rows]
        b, _, _ = pool_template(scaled, self.features, 'heuristic_weighted')
        np.testing.assert_array_equal(a, b)
        zeros = [dict(r, quality=0.) for r in self.rows]
        actual, weights, fallback = pool_template(zeros, self.features, 'heuristic_weighted')
        expected, _, _ = pool_template(zeros, self.features, 'uniform_mean')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(weights, [.5, .5])
        self.assertTrue(fallback)

    def test_best_quality_uses_an_actual_sample(self):
        template, weights, _ = pool_template(self.rows, self.features, 'best_quality')
        np.testing.assert_array_equal(template, self.features[1])
        np.testing.assert_array_equal(weights, [0., 1.])

    def test_invalid_quality_cannot_silently_change_a_template(self):
        for value in [float('nan'), float('inf'), -.1, 1.1]:
            with self.assertRaises(ValueError):
                pool_template([dict(self.rows[0], quality=value)], self.features, 'heuristic_weighted')

    def test_reserved_and_unlisted_identities_are_rejected(self):
        split = dict(halfface=dict(development=dict(known=['dev'], unknown=['unk']), test_reserved=dict(known=['test'], unknown=[])),
                     icfd_sample=dict(engineering_development=['icfd_dev'], reserved=['icfd_test']))
        row = dict(dataset='halfface', subject_id='dev', status='ok', embedding_index=0)
        assert_development([row], split)
        for sid in ['test', 'icfd_test', 'unlisted']:
            with self.assertRaisesRegex(ValueError, 'reserved identity'):
                assert_development([dict(row, subject_id=sid)], split)

    def test_failures_and_wrong_identities_remain_in_end_to_end_counts(self):
        base = dict(dataset='halfface', quality=.8, method='uniform_mean')
        rows = [dict(base, subject_id='a', truth_status='known', status='ok', top1_id='a', top1_score=.8, top2_score=.1, rank1_correct=True),
                dict(base, subject_id='b', truth_status='known', status='no_face', top1_id='', top1_score='', top2_score='', rank1_correct=False),
                dict(base, subject_id='u', truth_status='unknown', status='ok', top1_id='a', top1_score=.7, top2_score=.1, rank1_correct=False)]
        predictions = apply_decision(rows, .5, .1)
        summary = summarize(predictions, 'halfface', 'uniform_mean', 'test', .5, .1, 2, 4)
        self.assertEqual(summary['known_attempts'], 2)
        self.assertEqual(summary['capture_failures'], 1)
        self.assertEqual(summary['correct_known_accept_rate'], .5)
        self.assertEqual(summary['false_known_unknowns'], 1)
        self.assertEqual(summary['fpir_conditional'], 1.)


if __name__ == '__main__':
    unittest.main()
