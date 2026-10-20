"""Keep backbone comparisons paired and prevent enrollment/probe contamination."""
import unittest
import numpy as np
from scripts.compare_backbones import score_probes


class BackboneComparisonTests(unittest.TestCase):
    def rows(self):
        return [dict(dataset='halfface', subject_id=sid, path=path, mask_status=mask, pose=pose,
                     truth_status=truth, status=status, embedding_index=index, quality=.8, evaluation_role=role)
                for sid, path, mask, pose, truth, status, index, role in [
                    ('a', 'a_front', 'unmasked', 'front', 'known', 'ok', 0, 'gallery'),
                    ('a', 'a_left', 'unmasked', 'left', 'known', 'ok', 1, 'probe'),
                    ('b', 'b_front', 'unmasked', 'front', 'known', 'ok', 2, 'gallery'),
                    ('a', 'a_mask', 'masked', 'front', 'known', 'ok', 3, 'probe'),
                    ('b', 'b_mask', 'masked', 'front', 'known', 'no_face', -1, 'probe'),
                    ('u', 'u_probe', 'unmasked', 'front', 'unknown', 'ok', 4, 'probe')]]

    def test_policies_keep_identical_probes_and_include_capture_failures(self):
        rows = self.rows()
        features = np.eye(5, dtype=np.float32)
        first, source_one, count = score_probes(rows, features, 'halfface', ['a', 'b'], 'one_photo')
        few, source_few, _ = score_probes(rows, features, 'halfface', ['a', 'b'], 'up_to_three_photos')
        self.assertEqual(count, 2)
        self.assertEqual([r['path'] for r in first], ['a_mask', 'b_mask', 'u_probe'])
        self.assertEqual([r['path'] for r in first], [r['path'] for r in few])
        self.assertEqual(first[1]['status'], 'no_face')
        self.assertEqual(first[1]['top1_id'], '')
        self.assertEqual(len(source_one), 2)
        self.assertEqual(len(source_few), 3)
        self.assertFalse({r['path'] for r in source_few} & {r['path'] for r in few})

    def test_missing_gallery_identity_cannot_be_silently_dropped(self):
        rows = [dict(r, status='no_face', embedding_index=-1) if r['path'] == 'b_front' else r for r in self.rows()]
        with self.assertRaisesRegex(ValueError, 'Enrollment failure'):
            score_probes(rows, np.eye(5, dtype=np.float32), 'halfface', ['a', 'b'], 'one_photo')

    def test_best_match_uses_actual_mean_template(self):
        features = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., 1.], [0., -1.]], dtype=np.float32)
        scored, _, _ = score_probes(self.rows(), features, 'halfface', ['a', 'b'], 'up_to_three_photos')
        self.assertEqual(scored[0]['top1_id'], 'a')
        self.assertAlmostEqual(scored[0]['top1_score'], 2**-.5, places=6)


if __name__ == '__main__':
    unittest.main()
