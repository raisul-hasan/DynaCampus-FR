"""Guard outcome labels and distinguish template effects from calibration."""
import unittest
from scripts.prepare_defense_cases import category, compare_pair


class DefenseCaseTests(unittest.TestCase):
    def row(self, decision='KNOWN', correct=True, truth='known', path='probe', status='ok'):
        return dict(decision=decision, correct_known_accept=str(correct), truth_status=truth,
                    subject_id='example', path=path, status=status)

    def test_harmed_examples_are_retained(self):
        self.assertEqual(category(self.row(),self.row('UNKNOWN',False)),'harmed_known')
        self.assertEqual(category(self.row('UNKNOWN',False),self.row()),'helped_known')
        self.assertEqual(category(self.row(),self.row()),'unchanged_accept')

    def test_false_accept_and_ambiguity_are_not_treated_as_correct(self):
        self.assertEqual(category(self.row(truth='unknown'),self.row(truth='unknown')),'unknown_false_accept')
        self.assertEqual(category(self.row(truth='unknown'),self.row('AMBIGUOUS',False,'unknown')),'ambiguous_unknown')

    def test_capture_failure_is_not_a_known_or_unknown_decision(self):
        self.assertEqual(category(self.row(status='no_face'),self.row(status='no_face')),'capture_failure')

    def test_mismatched_probe_sets_cannot_be_compared(self):
        with self.assertRaisesRegex(ValueError,'Different probes'):
            compare_pair([self.row(path='a')],[self.row(path='b')],'fixed','arcface',[(.4,.1)]*2)

    def test_separate_thresholds_are_preserved_in_case_records(self):
        pair=compare_pair([self.row('UNKNOWN',False)],[self.row()],'calibrated','arcface',[(.3,.1),(.4,.15)])[0]
        self.assertEqual(pair['one_threshold'],.3)
        self.assertEqual(pair['few_threshold'],.4)
        self.assertEqual(pair['few_margin'],.15)


if __name__=='__main__':
    unittest.main()
