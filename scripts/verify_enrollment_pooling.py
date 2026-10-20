"""Audit saved pooling results without importing the pooling implementation.

Reconstruct enrollment candidates and weighted templates independently, then
check every cosine score, threshold, decision, summary and source fingerprint.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with Path(path).open(encoding='utf-8', newline='') as source:
        return list(csv.DictReader(source))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    run = ROOT / args.run
    info = json.loads((run / 'comparison.json').read_text(encoding='utf-8'))
    for path, expected in {**info['input_sha256'], **info['source_sha256']}.items():
        assert digest(ROOT / path) == expected, path
    assert not info['final_test_evaluated'] and not info['new_model_inference'] and not info['training_performed']
    source = ROOT / info['configuration']['enrollment_run']
    upstream = json.loads((source / 'comparison.json').read_text(encoding='utf-8'))
    split = json.loads((run / 'split_snapshot.json').read_text(encoding='utf-8'))
    assert split == json.loads((source / 'split_snapshot.json').read_text(encoding='utf-8'))
    records = read(source / 'combined_extraction.csv')
    by_path = {r['path']: r for r in records}
    allowed = set(sum(split['halfface']['development'].values(), []) + split['icfd_sample']['engineering_development'])
    assert {r['subject_id'] for r in records} <= allowed
    features = np.load(ROOT / upstream['feature_cache'], allow_pickle=False)['embeddings']
    saved = read(run / 'predictions.csv')
    logged_weights = read(run / 'enrollment_weights.csv')
    summaries = read(run / 'summary.csv')
    keys = [(r['dataset'], r['method'], r['rule'], r['path']) for r in saved]
    assert len(keys) == len(set(keys)) == info['probe_method_rule_attempts']
    checked, searches = 0, 0
    for summary in summaries:
        dataset, method, rule = (summary[k] for k in ('dataset', 'method', 'rule'))
        ids = sorted(split['halfface']['development']['known'] if dataset == 'halfface' else split['icfd_sample']['engineering_development'])
        vectors, enrollment_paths = [], set()
        for subject in ids:
            candidates = [r for r in records if r['dataset'] == dataset and r['subject_id'] == subject and
                          (r['mask_status'] == 'unmasked' if dataset == 'halfface' else r['evaluation_role'] == 'gallery')]
            candidates.sort(key=lambda r: (({'front': 0, 'left': 1, 'right': 2}[r['pose']], r['path']) if dataset == 'halfface' else (r['path'],)))
            if method == 'one_photo' and dataset == 'halfface':
                candidates = [r for r in candidates if r['pose'] == 'front']
            candidates = candidates[:1 if method == 'one_photo' else 3]
            enrollment_paths.update(r['path'] for r in candidates)
            usable = [r for r in candidates if r['status'] == 'ok']
            coefficients = np.ones(len(usable), dtype=np.float64)
            if method in ('heuristic_weighted', 'detector_weighted'):
                field = 'quality' if method == 'heuristic_weighted' else 'det_score'
                coefficients = np.array([float(r[field]) for r in usable])
            elif method == 'best_quality':
                coefficients[:] = 0.
                coefficients[np.argmax([float(r['quality']) for r in usable])] = 1.
            fallback = coefficients.sum() == 0
            if fallback:
                coefficients[:] = 1.
            coefficients /= coefficients.sum()
            weights = [r for r in logged_weights if r['dataset'] == dataset and r['method'] == method and r['subject_id'] == subject]
            assert [r['path'] for r in weights] == [r['path'] for r in usable]
            np.testing.assert_allclose([float(r['weight']) for r in weights], coefficients, atol=1e-14)
            assert all((r['zero_weight_fallback'] == 'True') == fallback for r in weights)
            mean = np.array([features[int(r['embedding_index'])] for r in usable], dtype=np.float64).T @ coefficients
            vectors.append((mean / np.linalg.norm(mean)).astype(np.float32))
        matrix = np.stack(vectors)
        expected_probes = {r['path'] for r in records if r['dataset'] == dataset and
                           ((r['truth_status'] == 'unknown' or r['mask_status'] == 'masked') if dataset == 'halfface' else r['evaluation_role'] == 'probe')}
        selected = [r for r in saved if (r['dataset'], r['method'], r['rule']) == (dataset, method, rule)]
        assert {r['path'] for r in selected} == expected_probes
        assert not expected_probes & enrollment_paths
        recomputed = []
        for row in selected:
            original = by_path[row['path']]
            assert all(row[k] == original[k] for k in ('dataset', 'subject_id', 'truth_status', 'status', 'quality'))
            top, first, second = '', None, None
            if original['status'] == 'ok':
                scores = matrix @ features[int(original['embedding_index'])]
                order = np.argsort(scores)[::-1]
                top, first, second = ids[order[0]], float(scores[order[0]]), float(scores[order[1]])
                assert row['top1_id'] == top
                assert abs(first - float(row['top1_score'])) < 1e-7
                assert abs(second - float(row['top2_score'])) < 1e-7
                searches += 1
            else:
                assert row['top1_id'] == row['top1_score'] == row['top2_score'] == ''
            recomputed.append((row, top, first, second))
        if rule == 'recalibrated_score':
            nonmated = sorted([first for row, _, first, _ in recomputed if first is not None and row['truth_status'] == 'unknown'], reverse=True)
            threshold = float(np.nextafter(nonmated[math.floor(info['configuration']['target_conditional_fpir'] * len(nonmated))], np.inf))
            assert threshold == float(summary['threshold'])
        accepted, wrong, false_known, correct_rank, ambiguous, failures = 0, 0, 0, 0, 0, 0
        for row, top, first, second in recomputed:
            rank_correct = first is not None and row['truth_status'] == 'known' and top == row['subject_id']
            correct_rank += rank_correct
            assert (row['rank1_correct'] == 'True') == rank_correct
            identity = ''
            if first is None:
                state = 'CAPTURE_FAILURE'
                failures += 1
            elif rule == 'closed_set_engineering':
                state = 'CLOSED_SET_ONLY'
            else:
                assert float(row['threshold']) == float(summary['threshold'])
                assert float(row['margin_threshold']) == float(summary['margin_threshold'])
                state = 'UNKNOWN' if first < float(summary['threshold']) else (
                    'AMBIGUOUS' if first - second < float(summary['margin_threshold']) else 'KNOWN')
                if state == 'KNOWN':
                    identity = top
            correct = state == 'KNOWN' and rank_correct
            assert (row['decision'], row['accepted_id'], row['correct_known_accept'] == 'True') == (state, identity, correct)
            accepted += correct
            wrong += state == 'KNOWN' and row['truth_status'] == 'known' and not rank_correct
            false_known += state == 'KNOWN' and row['truth_status'] == 'unknown'
            ambiguous += state == 'AMBIGUOUS'
            checked += 1
        assert int(summary['known_attempts']) == sum(r['truth_status'] == 'known' for r in selected)
        assert int(summary['unknown_attempts']) == sum(r['truth_status'] == 'unknown' for r in selected)
        assert int(summary['rank1_correct']) == correct_rank
        assert int(summary['capture_failures']) == failures
        assert int(summary['ambiguous']) == ambiguous
        if rule != 'closed_set_engineering':
            assert int(summary['correct_known_accepts']) == accepted
            assert int(summary['wrong_known_identity_accepts']) == wrong
            assert int(summary['false_known_unknowns']) == false_known
    assert checked == info['probe_method_rule_attempts']
    receipt = dict(verification='passed', attempts=checked, successful_cosine_searches=searches,
        summaries_checked=len(summaries), independent_template_reconstruction=True,
        reserved_test_evaluated=False, image_inference_performed=False,
        verified_artifact_sha256={name: digest(run / name) for name in ['comparison.json', 'summary.csv', 'predictions.csv', 'enrollment_weights.csv']},
        verifier_sha256=digest(Path(__file__)))
    (run / 'verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
