"""Independent cached audit of paired backbone scores, decisions and calibration."""
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


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify_grid(rows, logged, selected, family, config):
    valid = np.array([r['status'] == 'ok' for r in rows])
    known = np.array([r['truth_status'] == 'known' for r in rows])
    correct = np.array([r['rank1_correct'] == 'True' for r in rows])
    score = np.array([float(r['top1_score']) if r['status'] == 'ok' else -2. for r in rows])
    gap = np.array([float(r['top1_score']) - float(r['top2_score']) if r['status'] == 'ok' else 0. for r in rows])
    levels = np.unique(score[valid])
    thresholds = np.unique(np.r_[-1.000001, levels, np.nextafter(levels, np.inf), 1.000001])
    margins = [0.] if family == 'score_only' else config['margin_grid']
    by_key = {(float(r['score_threshold']), float(r['margin_threshold'])): r for r in logged}
    assert len(logged) == len(by_key) == len(thresholds) * len(margins)
    best, checked = None, 0
    for margin in margins:
        accepted = valid[None, :] & (score[None, :] >= thresholds[:, None]) & (gap[None, :] >= margin)
        covered = valid[None, :] & ((score[None, :] < thresholds[:, None]) | (gap[None, :] >= margin))
        true_count = (accepted & correct).sum(axis=1)
        false_count = (accepted & ~known).sum(axis=1)
        wrong = (accepted & known & ~correct).sum(axis=1)
        coverage_count = covered.sum(axis=1)
        known_coverage = (covered & known).sum(axis=1) / (valid & known).sum()
        unknown_coverage = (covered & ~known).sum(axis=1) / (valid & ~known).sum()
        fpir = false_count / (valid & ~known).sum()
        feasible = ((fpir <= config['target_conditional_fpir'] + 1e-12) &
                    (known_coverage >= config['minimum_known_valid_coverage']) &
                    (unknown_coverage >= config['minimum_unknown_valid_coverage']))
        for i, threshold in enumerate(thresholds):
            row = by_key[(float(threshold), float(margin))]
            expected = dict(correct_known_accepts=int(true_count[i]), false_known_unknowns=int(false_count[i]),
                wrong_known_identity_accepts=int(wrong[i]), valid_decision_coverage=float(coverage_count[i] / valid.sum()),
                known_valid_coverage=float(known_coverage[i]), unknown_valid_coverage=float(unknown_coverage[i]), fpir_conditional=float(fpir[i]))
            assert float(row['quality_threshold']) == 0.
            assert (row['feasible'] == 'True') == bool(feasible[i])
            for field, value in expected.items():
                assert abs(float(row[field]) - value) < 1e-12, field
            if feasible[i]:
                key = (-int(true_count[i]), int(false_count[i]), int(wrong[i]), -int(coverage_count[i]), 0., float(margin), -float(threshold))
                if best is None or key < best[0]:
                    best = key, float(threshold), float(margin)
            checked += 1
    assert best is not None
    assert float(selected['score_threshold']) == best[1]
    assert float(selected['margin_threshold']) == best[2]
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    run = ROOT / args.run
    info = json.loads((run / 'comparison.json').read_text(encoding='utf-8'))
    for path, expected in {**info['source_sha256'], **info['input_sha256']}.items():
        assert sha(ROOT / path) == expected, path
    assert not info['final_test_evaluated'] and not info['training_performed']
    split = json.loads((run / 'split_snapshot.json').read_text(encoding='utf-8'))
    records = read(run / 'extraction.csv')
    by_path = {r['path']: r for r in records}
    allowed = set(sum(split['halfface']['development'].values(), []) + split['icfd_sample']['engineering_development'])
    assert len(by_path) == len(records) == info['extraction_attempts']
    assert {r['subject_id'] for r in records} <= allowed
    for row in records:
        assert sha(ROOT / row['path']) == row['source_sha256']
    source = ROOT / info['configuration']['enrollment_run']
    original = read(source / 'combined_extraction.csv')
    assert [r['path'] for r in original] == [r['path'] for r in records]
    assert all(a['status'] == b['status'] and a['embedding_index'] == b['embedding_index'] for a, b in zip(original, records))
    source_info = json.loads((source / 'comparison.json').read_text(encoding='utf-8'))
    old_arc = np.load(ROOT / source_info['feature_cache'], allow_pickle=False)['embeddings']
    assert sha(ROOT / info['model']['features']) == info['model']['feature_sha256']
    cache = np.load(ROOT / info['model']['features'], allow_pickle=False)
    assert set(cache.files) == {'arcface', 'adaface'}
    for name in cache.files:
        assert cache[name].shape == old_arc.shape and np.isfinite(cache[name]).all()
        np.testing.assert_allclose(np.linalg.norm(cache[name], axis=1), 1., atol=1e-6)
        assert sha(Path(info['model'][name]['checkpoint']) if Path(info['model'][name]['checkpoint']).is_absolute() else ROOT / info['model'][name]['checkpoint']) == info['model'][name]['sha256']
    delta = float(np.max(np.abs(cache['arcface'] - old_arc)))
    assert delta <= 1e-6 and delta == info['model']['arcface_max_cache_difference']
    smoke = json.loads((run / 'real_model_smoke.json').read_text(encoding='utf-8'))
    assert smoke['official_preprocessing_exact'] and smoke['strict_checkpoint_loading']
    predictions, summaries = read(run / 'predictions.csv'), read(run / 'summary.csv')
    selections, grid = read(run / 'selected_thresholds.csv'), read(run / 'calibration_grid.csv')
    sources = read(run / 'enrollment_sources.csv')
    checked = searches = grid_checks = 0
    keys = [(r['model'], r['dataset'], r['strategy'], r['family'], r['path']) for r in predictions]
    assert len(keys) == len(set(keys)) == info['probe_method_rule_attempts']
    for summary in summaries:
        model, dataset, strategy, family = (summary[k] for k in ['model', 'dataset', 'strategy', 'family'])
        selected_rows = [r for r in predictions if (r['model'], r['dataset'], r['strategy'], r['family']) == (model, dataset, strategy, family)]
        ids = sorted(split['halfface']['development']['known'] if dataset == 'halfface' else split['icfd_sample']['engineering_development'])
        prototypes, used_paths = [], set()
        for subject in ids:
            candidates = [r for r in records if r['dataset'] == dataset and r['subject_id'] == subject and
                          (r['mask_status'] == 'unmasked' if dataset == 'halfface' else r['evaluation_role'] == 'gallery')]
            candidates.sort(key=lambda r: (({'front': 0, 'left': 1, 'right': 2}[r['pose']], r['path']) if dataset == 'halfface' else (r['path'],)))
            if strategy == 'one_photo' and dataset == 'halfface':
                candidates = [r for r in candidates if r['pose'] == 'front']
            candidates = candidates[:1 if strategy == 'one_photo' else 3]
            logged = [r for r in sources if (r['model'], r['dataset'], r['strategy'], r['subject_id']) == (model, dataset, strategy, subject)]
            assert [r['path'] for r in logged] == [r['path'] for r in candidates]
            used_paths.update(r['path'] for r in candidates)
            vectors = [cache[model][int(r['embedding_index'])] for r in candidates if r['status'] == 'ok']
            mean = np.asarray(vectors, dtype=np.float64).mean(axis=0)
            prototypes.append((mean / np.linalg.norm(mean)).astype(np.float32))
        matrix = np.stack(prototypes)
        expected_paths = {r['path'] for r in records if r['dataset'] == dataset and
                          ((r['truth_status'] == 'unknown' or r['mask_status'] == 'masked') if dataset == 'halfface' else r['evaluation_role'] == 'probe')}
        assert {r['path'] for r in selected_rows} == expected_paths
        assert not expected_paths & used_paths
        for row in selected_rows:
            original_row = by_path[row['path']]
            assert all(row[k] == original_row[k] for k in ['subject_id', 'truth_status', 'status', 'quality'])
            state, identity, rank = 'CAPTURE_FAILURE', '', False
            if row['status'] == 'ok':
                scores = matrix @ cache[model][int(original_row['embedding_index'])]
                order = np.argsort(scores)[::-1]
                first, second = float(scores[order[0]]), float(scores[order[1]])
                top = ids[order[0]]
                assert row['top1_id'] == top
                assert abs(first - float(row['top1_score'])) < 1e-7
                assert abs(second - float(row['top2_score'])) < 1e-7
                rank = row['truth_status'] == 'known' and top == row['subject_id']
                if dataset == 'icfd_sample':
                    state = 'CLOSED_SET_ONLY'
                else:
                    state = 'UNKNOWN' if first < float(summary['score_threshold']) else (
                        'AMBIGUOUS' if first - second < float(summary['margin_threshold']) else 'KNOWN')
                    if state == 'KNOWN':
                        identity = top
                searches += 1
            else:
                assert row['top1_id'] == row['top1_score'] == row['top2_score'] == ''
            assert (row['rank1_correct'] == 'True') == rank
            assert (row['decision'], row['accepted_id'], row['correct_known_accept'] == 'True') == (state, identity, state == 'KNOWN' and rank)
            checked += 1
        known = [r for r in selected_rows if r['truth_status'] == 'known']
        unknown = [r for r in selected_rows if r['truth_status'] == 'unknown']
        assert int(summary['known_probe_attempts']) == len(known)
        assert int(summary['unknown_probe_attempts']) == len(unknown)
        assert int(summary['rank1_correct']) == sum(r['rank1_correct'] == 'True' for r in known)
        if dataset == 'halfface':
            selected = next(r for r in selections if (r['model'], r['dataset'], r['strategy'], r['family']) == (model, dataset, strategy, family))
            logged = [r for r in grid if (r['model'], r['dataset'], r['strategy'], r['family']) == (model, dataset, strategy, family)]
            grid_checks += verify_grid(selected_rows, logged, selected, family, info['configuration'])
            assert float(selected['score_threshold']) == float(summary['score_threshold'])
            assert float(selected['margin_threshold']) == float(summary['margin_threshold'])
            assert int(summary['correct_known_accepts']) == sum(r['correct_known_accept'] == 'True' for r in known)
            assert int(summary['false_known_unknowns']) == sum(r['decision'] == 'KNOWN' for r in unknown)
    assert checked == info['probe_method_rule_attempts']
    assert grid_checks == len(grid)
    timings = read(run / 'timings.csv')
    for row in read(run / 'timing_summary.csv'):
        values = [float(r[row['stage']]) for r in timings if r['model'] == row['model'] and r['dataset'] == row['dataset'] and r['status'] == 'ok']
        assert len(values) == int(row['samples'])
        assert abs(float(np.median(values)) - float(row['median_ms'])) < 1e-10
        assert abs(float(np.percentile(values, 95)) - float(row['p95_ms'])) < 1e-10
    receipt = dict(verification='passed', prediction_attempts=checked, cosine_searches=searches,
        calibration_grid_entries=grid_checks, shared_source_images=len(records), arcface_cache_max_difference=delta,
        shared_detector_outcomes=True, input_images_unchanged=True, final_test_evaluated=False,
        verifier_sha256=sha(Path(__file__)), artifact_sha256={p.name: sha(p) for p in run.iterdir() if p.is_file() and p.name != 'verification.json'})
    (run / 'verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'artifact_sha256'}, indent=2))


if __name__ == '__main__':
    main()
