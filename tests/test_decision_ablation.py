"""Check refusal semantics, score boundaries and calibration denominators."""
import unittest

from scripts.compare_decisions import apply_rule, calibrate, summarize
from src.open_set import OpenSetDecisionEngine


def probe(subject, truth, score, second, quality=1., status='ok'):
    return dict(subject_id=subject, truth_status=truth, top1_id='a',
                top1_score=score, top2_score=second, quality=quality, status=status)


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.engine = OpenSetDecisionEngine(.5, .25, .125)

    def test_high_score_near_tie_is_ambiguous_without_identity(self):
        self.assertEqual(self.engine.evaluate_scores('a', .9, .85, 1.)[:2], ('AMBIGUOUS', None))

    def test_quality_precedes_score_and_margin(self):
        self.assertEqual(self.engine.evaluate_scores('a', .9, .1, .1)[:2], ('INSUFFICIENT_QUALITY', None))

    def test_low_score_remains_unknown(self):
        self.assertEqual(self.engine.evaluate_scores('a', .49, -.5, 1.)[:2], ('UNKNOWN', None))

    def test_equal_thresholds_are_accepted(self):
        self.assertEqual(self.engine.evaluate_scores('a', .5, .375, .25)[:2], ('KNOWN', 'a'))

    def test_single_candidate_and_empty_gallery(self):
        self.assertEqual(self.engine.evaluate([('a', .8, {})], 1.)[:2], ('KNOWN', 'a'))
        self.assertEqual(self.engine.evaluate([], 1.)[:2], ('UNKNOWN', None))

    def test_invalid_scores_and_quality_are_rejected(self):
        for scores in [(.8, .9, 1.), (float('nan'), .1, 1.), (.8, .1, float('nan')), (.8, .1, 1.1)]:
            with self.assertRaises(ValueError):
                self.engine.evaluate_scores('a', *scores)

    def test_refused_unknowns_stay_in_fpir_denominator(self):
        rows = [probe('a', 'known', .9, .1), probe('u1', 'unknown', .8, .1),
                probe('u2', 'unknown', .7, .1, quality=.1),
                probe('u3', 'unknown', 0, 0, status='no_face')]
        thresholds = dict(score_threshold=.5, quality_threshold=.25, margin_threshold=.125)
        result = summarize(apply_rule(rows, thresholds, 'test'), thresholds, 'test')
        self.assertEqual(result['fpir_conditional'], .5)
        self.assertEqual(result['unknown_probe_attempts'], 3)
        self.assertEqual(result['valid_decision_coverage'], 2/3)
        self.assertEqual(result['end_to_end_decision_coverage'], .5)

    def test_calibration_matches_rule_and_can_disable_quality(self):
        rows = [probe('a', 'known', .75, .125, quality=.2),
                probe('u', 'unknown', .5, .125)]
        config = dict(margin_grid=[0., .125], quality_grid=[0., .5],
                      target_conditional_fpir=0., minimum_known_valid_coverage=.85,
                      minimum_unknown_valid_coverage=.85)
        for family in ['score_only', 'score_margin', 'score_margin_quality']:
            thresholds, _ = calibrate(rows, family, config)
            result = summarize(apply_rule(rows, thresholds, family), thresholds, family)
            self.assertEqual(result['correct_known_accepts'], 1)
            self.assertEqual(result['false_known_unknowns'], 0)
            self.assertEqual(thresholds['quality_threshold'], 0.)


if __name__ == '__main__':
    unittest.main()
