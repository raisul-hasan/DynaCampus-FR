"""Protocol checks: held-out identities and score-boundary calibration."""
import json
import unittest

import numpy as np

from scripts.run_arcface_baseline import ROOT, choose_threshold, read_csv, select_rows


class BaselineProtocolTests(unittest.TestCase):
    def test_threshold_respects_equal_score_boundary(self):
        scores = np.array([0.7, 0.7, 0.2, 0.1])
        threshold = choose_threshold(scores, 0.25)
        self.assertLessEqual(np.mean(scores >= threshold), 0.25)
        self.assertGreater(threshold, 0.7)

    def test_small_nonmated_set_requires_zero_accepts_at_one_percent(self):
        scores = np.array([0.3, 0.4, 0.5])
        self.assertFalse(np.any(scores >= choose_threshold(scores, 0.01)))

    def test_missing_nonmated_evidence_is_an_error(self):
        with self.assertRaises(ValueError):
            choose_threshold([], 0.01)

    def test_reserved_identities_never_enter_development(self):
        split = json.loads((ROOT / 'configs/splits/arcface_baseline_v1.json').read_text(encoding='utf-8'))
        hf = read_csv(ROOT / 'data/manifests/halfface_usable.csv')
        icfd = read_csv(ROOT / 'data/manifests/icfd_sample.csv')
        rows = select_rows(hf, icfd, split)
        dev = split['halfface']['development']
        test = split['halfface']['test_reserved']
        reserved = set(test['known'] + test['unknown'] + split['icfd_sample']['reserved'])
        self.assertFalse(reserved & {r['subject_id'] for r in rows})
        self.assertFalse(set(dev['known']) & set(dev['unknown']))
        self.assertFalse(set(dev['known'] + dev['unknown']) & set(test['known'] + test['unknown']))
        gallery = [r for r in rows if r['dataset'] == 'halfface' and r['evaluation_role'] == 'gallery']
        self.assertEqual({r['subject_id'] for r in gallery}, set(dev['known']))
        self.assertEqual(len(gallery), len(dev['known']))
        self.assertTrue(all(r['mask_status'] == 'unmasked' and r['pose'] == 'front' for r in gallery))


if __name__ == '__main__':
    unittest.main()
