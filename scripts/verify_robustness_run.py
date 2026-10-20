"""Verify saved stress-test searches, decisions, fixed settings and source integrity."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_enrollment import gallery_candidates, unit_mean
from scripts.run_arcface_baseline import read_csv, sha256


def verify(run):
    info = json.loads((run/'comparison.json').read_text())
    config = info['configuration']
    decision_info = json.loads((ROOT/config['decision_run']/'comparison.json').read_text())
    enrollment = ROOT/decision_info['configuration']['enrollment_run']
    enrollment_info = json.loads((enrollment/'comparison.json').read_text())
    old_records = read_csv(enrollment/'combined_extraction.csv')
    old_features = np.load(ROOT/enrollment_info['feature_cache'], allow_pickle=False)['embeddings']
    for path, digest in {**info['source_sha256'], **info['input_sha256']}.items():
        assert sha256(ROOT/path) == digest, path
    source = read_csv(run/'source_manifest.csv')
    for row in source:
        assert sha256(ROOT/row['path']) == row['sha256'], row['path']
    split = json.loads((run/'split_snapshot.json').read_text())
    reserved = set(sum(split['halfface']['test_reserved'].values(), []))
    assert not reserved & {r['subject_id'] for r in source}
    assert len({r['path'] for r in source}) == len(source)
    cache = ROOT/info['feature_cache']
    gallery = np.load(cache/'gallery.npz', allow_pickle=False)
    ids, prototypes = gallery['identities'].tolist(), gallery['embeddings']
    candidates = gallery_candidates(old_records, 'halfface', split['halfface']['development']['known'], 'up_to_three_photos')
    assert ids == list(candidates)
    expected = np.stack([unit_mean([old_features[int(r['embedding_index'])] for r in candidates[s] if r['status'] == 'ok']) for s in ids])
    np.testing.assert_array_equal(prototypes, expected)
    assert not {r['path'] for r in source} & {r['path'] for rows in candidates.values() for r in rows}
    predictions = read_csv(run/'predictions.csv')
    checked = searches = 0
    for condition in config['conditions']:
        name = condition['name']
        extraction = read_csv(run/f'{name}_extraction.csv')
        assert {r['path'] for r in extraction} == {r['path'] for r in source}
        extraction_by_path = {r['path']: r for r in extraction}
        matrix = np.load(cache/f'{name}.npz', allow_pickle=False)['embeddings']
        assert len(matrix) == sum(r['status'] == 'ok' for r in extraction)
        assert np.isfinite(matrix).all()
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1., atol=1e-5)
        scores_by_path = {}
        for row in extraction:
            if row['status'] == 'ok':
                scores = prototypes @ matrix[int(row['embedding_index'])]
                order = np.argsort(scores)[::-1]
                scores_by_path[row['path']] = (ids[int(order[0])], float(scores[order[0]]), float(scores[order[1]]))
                searches += 1
        for family in config['families']:
            thresholds = decision_info['selected_thresholds'][family]
            assert info['selected_thresholds'][family] == thresholds
            rows = [r for r in predictions if r['condition'] == name and r['family'] == family]
            assert len(rows) == len(source)
            assert {r['path'] for r in rows} == {r['path'] for r in source}
            counts, extraction_counts = Counter(), Counter()
            true_known = rank1 = false_unknown = wrong_known = covered = known_valid = unknown_valid = 0
            for row in rows:
                extracted = extraction_by_path[row['path']]
                assert row['status'] == extracted['status']
                assert row['quality'] == extracted['quality']
                identity, rank_correct = '', False
                if row['status'] != 'ok':
                    state = 'CAPTURE_FAILURE'
                else:
                    best, score, second = scores_by_path[row['path']]
                    assert best == row['top1_id']
                    assert score == float(row['top1_score']) and second == float(row['top2_score'])
                    rank_correct = row['truth_status'] == 'known' and best == row['subject_id']
                    known_valid += row['truth_status'] == 'known'
                    unknown_valid += row['truth_status'] == 'unknown'
                    if float(row['quality']) < thresholds['quality_threshold']:
                        state = 'INSUFFICIENT_QUALITY'
                    elif score < thresholds['score_threshold']:
                        state = 'UNKNOWN'
                    elif score-second < thresholds['margin_threshold']:
                        state = 'AMBIGUOUS'
                    else:
                        state, identity = 'KNOWN', best
                assert (state, identity) == (row['decision'], row['accepted_id'])
                assert (row['rank1_correct'] == 'True') == rank_correct
                counts[state] += 1
                extraction_counts[row['status']] += 1
                true_known += state == 'KNOWN' and rank_correct
                rank1 += rank_correct
                false_unknown += state == 'KNOWN' and row['truth_status'] == 'unknown'
                wrong_known += state == 'KNOWN' and row['truth_status'] == 'known' and not rank_correct
                covered += state in {'KNOWN', 'UNKNOWN'}
                checked += 1
            summary = next(r for r in info['summaries'] if r['family'] == family and r['condition'] == name)
            assert dict(counts) == summary['decision_counts']
            assert dict(extraction_counts) == summary['extraction_counts']
            assert (true_known, rank1, false_unknown, wrong_known) == (summary['correct_known_accepts'], summary['rank1_correct'], summary['false_known_unknowns'], summary['wrong_known_identity_accepts'])
            assert true_known/summary['known_probe_attempts'] == summary['correct_known_accept_end_to_end']
            assert false_unknown/unknown_valid == summary['fpir_conditional']
            assert false_unknown/summary['unknown_probe_attempts'] == summary['false_known_end_to_end']
            assert covered/len(rows) == summary['end_to_end_decision_coverage']
            assert known_valid+unknown_valid == summary['extraction_successes']
            assert known_valid == summary['known_with_embedding'] and unknown_valid == summary['unknown_with_embedding']
    assert not info['thresholds_retuned'] and not info['gallery_changed'] and not info['final_test_evaluated']
    assert len(predictions) == checked
    report = dict(verified_prediction_attempts=checked, verified_cosine_searches=searches,
                  source_images_unchanged=len(source), conditions=len(config['conditions']),
                  unchanged_gallery_and_thresholds=True, reserved_identity_overlap=0,
                  verifier_sha256=sha256(__file__),
                  prediction_sha256=sha256(run/'predictions.csv'),
                  cache_sha256={p.name:sha256(p) for p in cache.glob('*.npz')})
    (run/'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'cache_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    verify(ROOT/args.run)
