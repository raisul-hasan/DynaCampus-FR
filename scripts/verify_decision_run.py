"""Independently recompute saved decisions and constrained selection from CSV."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_arcface_baseline import read_csv, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    run = ROOT / args.run
    info = json.loads((run / 'comparison.json').read_text())
    inputs = read_csv(run / 'input_scores.csv')
    predictions = read_csv(run / 'predictions.csv')
    grid = read_csv(run / 'calibration_grid.csv')
    split = json.loads((run / 'split_snapshot.json').read_text())
    reserved = set(sum(split['halfface']['test_reserved'].values(), []))
    assert not reserved & {r['subject_id'] for r in inputs}
    assert len({r['path'] for r in inputs}) == len(inputs)
    assert sha256(ROOT / 'src/open_set/decision.py') == info['engine_sha256']
    source = ROOT / info['configuration']['enrollment_run']
    assert sha256(source / 'predictions.csv') == info['source_predictions_sha256']
    assert sha256(source / 'combined_extraction.csv') == info['source_extraction_sha256']
    checked = 0
    for summary in info['summaries']:
        family = summary['family']
        t = info['selected_thresholds'][family]
        rows = [r for r in predictions if r['family'] == family]
        assert {r['path'] for r in rows} == {r['path'] for r in inputs}
        counts = Counter()
        known_correct = unknown_false = covered = unknown_valid = 0
        for r in rows:
            identity = ''
            if r['status'] != 'ok':
                state = 'CAPTURE_FAILURE'
            else:
                score, second, quality = map(float, [r['top1_score'], r['top2_score'], r['quality']])
                unknown_valid += r['truth_status'] == 'unknown'
                if quality < t['quality_threshold']:
                    state = 'INSUFFICIENT_QUALITY'
                elif score < t['score_threshold']:
                    state = 'UNKNOWN'
                elif score - second < t['margin_threshold']:
                    state = 'AMBIGUOUS'
                else:
                    state, identity = 'KNOWN', r['top1_id']
            assert (state, identity) == (r['decision'], r['accepted_id'])
            counts[state] += 1
            known_correct += state == 'KNOWN' and identity == r['subject_id'] and r['truth_status'] == 'known'
            unknown_false += state == 'KNOWN' and r['truth_status'] == 'unknown'
            covered += state in {'KNOWN', 'UNKNOWN'}
            checked += 1
        assert dict(counts) == summary['decision_counts']
        assert known_correct == summary['correct_known_accepts']
        assert unknown_false == summary['false_known_unknowns']
        assert unknown_false / unknown_valid == summary['fpir_conditional']
        assert covered / len(rows) == summary['end_to_end_decision_coverage']
        feasible = [r for r in grid if r['family'] == family and r['feasible'] == 'True']
        def key(r):
            return (-int(r['correct_known_accepts']), int(r['false_known_unknowns']),
                    int(r['wrong_known_identity_accepts']), -float(r['valid_decision_coverage']),
                    float(r['quality_threshold']), float(r['margin_threshold']), -float(r['score_threshold']))
        best = min(feasible, key=key)
        assert all(float(best[k]) == t[k] for k in ['score_threshold', 'quality_threshold', 'margin_threshold'])
    assert not info['final_test_evaluated']
    report = dict(verified_attempts=checked, input_probes=len(inputs),
                  grid_configurations=len(grid), reserved_identity_overlap=0,
                  source_hashes_match=True, rule_and_selected_grid_match=True,
                  verifier_sha256=sha256(__file__))
    (run / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
