"""Compare existing enrollment pooling policies on real development caches.

No new model inference, training, reserved-test access or app changes. Threshold
recalibration reuses development unknown probes and is explicitly descriptive.
A second comparison holds the previous mean-gallery score/margin rule fixed.
"""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_enrollment import gallery_candidates, paired_bootstrap, reject_overlap, unit_mean
from scripts.run_arcface_baseline import choose_threshold, read_csv, sha256, write_csv
from src.open_set import OpenSetDecisionEngine

METHODS = ('one_photo', 'uniform_mean', 'heuristic_weighted', 'detector_weighted', 'best_quality')


def assert_development(rows, split):
    allowed = {
        'halfface': set(sum(split['halfface']['development'].values(), [])),
        'icfd_sample': set(split['icfd_sample']['engineering_development']),
    }
    reserved = set(sum(split['halfface']['test_reserved'].values(), []) + split['icfd_sample']['reserved'])
    if not rows:
        raise ValueError('Empty development extraction')
    for row in rows:
        if row['subject_id'] in reserved or row['subject_id'] not in allowed.get(row['dataset'], set()):
            raise ValueError('Non-development or reserved identity in pooling inputs')
        if row['status'] == 'ok' and int(row['embedding_index']) < 0:
            raise ValueError('Successful extraction has no embedding')


def pool_template(rows, features, method):
    if method not in METHODS or not rows or any(r['status'] != 'ok' for r in rows):
        raise ValueError('Expected a supported method and successful enrollment rows')
    vectors = np.stack([features[int(r['embedding_index'])] for r in rows])
    quality = np.array([float(r['quality']) for r in rows], dtype=np.float64)
    confidence = np.array([float(r['det_score']) for r in rows], dtype=np.float64)
    if not np.isfinite(quality).all() or not np.isfinite(confidence).all():
        raise ValueError('Non-finite enrollment quality or detector confidence')
    if ((quality < 0) | (quality > 1) | (confidence < 0) | (confidence > 1)).any():
        raise ValueError('Enrollment weights must be in [0, 1]')
    if method in ('uniform_mean', 'one_photo'):
        weights = np.ones(len(rows), dtype=np.float64)
    elif method == 'heuristic_weighted':
        weights = quality.copy()
    elif method == 'detector_weighted':
        weights = confidence.copy()
    else:
        weights = np.zeros(len(rows), dtype=np.float64)
        weights[int(np.argmax(quality))] = 1.
    fallback = float(weights.sum()) == 0
    if fallback:
        weights[:] = 1.
    weights /= weights.sum()
    if method in ('uniform_mean', 'one_photo') or fallback:
        template = unit_mean(vectors)
    else:
        mean = np.sum(vectors.astype(np.float64) * weights[:, None], axis=0)
        norm = np.linalg.norm(mean)
        if not np.isfinite(norm) or norm <= 1e-10:
            raise ValueError('Invalid weighted template')
        template = (mean / norm).astype(np.float32)
    return template, weights, fallback


def apply_decision(rows, threshold, margin):
    engine = OpenSetDecisionEngine(base_threshold=threshold, margin_threshold=margin, quality_threshold=0.)
    output = []
    for source in rows:
        row = dict(source)
        state, identity = 'CAPTURE_FAILURE', None
        if row['status'] == 'ok':
            state, identity, _ = engine.evaluate_scores(row['top1_id'], row['top1_score'], row['top2_score'], float(row['quality']))
        row.update(decision=state, accepted_id=identity or '', threshold=threshold, margin_threshold=margin,
                   correct_known_accept=state == 'KNOWN' and identity == row['subject_id'] and row['truth_status'] == 'known')
        output.append(row)
    return output


def summarize(rows, dataset, method, rule, threshold, margin, gallery_count, source_count):
    known = [r for r in rows if r['truth_status'] == 'known']
    unknown = [r for r in rows if r['truth_status'] == 'unknown']
    unknown_ok = [r for r in unknown if r['status'] == 'ok']
    closed = rule == 'closed_set_engineering'
    return dict(dataset=dataset, method=method, rule=rule, gallery_identities=gallery_count,
        enrollment_images_used=source_count, known_attempts=len(known), unknown_attempts=len(unknown),
        known_with_embedding=sum(r['status'] == 'ok' for r in known), unknown_with_embedding=len(unknown_ok),
        rank1_correct=sum(r['rank1_correct'] for r in known),
        correct_known_accepts=None if closed else sum(r['correct_known_accept'] for r in known),
        false_known_unknowns=None if closed else sum(r['decision'] == 'KNOWN' for r in unknown),
        wrong_known_identity_accepts=None if closed else sum(r['decision'] == 'KNOWN' and r['accepted_id'] != r['subject_id'] for r in known),
        correct_known_accept_rate=None if closed else sum(r['correct_known_accept'] for r in known) / len(known),
        fpir_conditional=None if not unknown_ok else sum(r['decision'] == 'KNOWN' for r in unknown_ok) / len(unknown_ok),
        capture_failures=sum(r['status'] != 'ok' for r in rows),
        ambiguous=sum(r['decision'] == 'AMBIGUOUS' for r in rows), threshold=threshold, margin_threshold=margin)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/enrollment_pooling.yaml')
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if config['final_test_evaluation'] or tuple(config['methods']) != METHODS:
        raise ValueError('Only the declared development comparison is supported')
    if not 0 <= config['target_conditional_fpir'] < 1:
        raise ValueError('Invalid empirical FPIR target')
    source, decisions = ROOT / config['enrollment_run'], ROOT / config['decision_run']
    info = json.loads((source / 'comparison.json').read_text(encoding='utf-8'))
    decision_info = json.loads((decisions / 'comparison.json').read_text(encoding='utf-8'))
    if info['final_test_evaluated'] or info['model']['mock'] or decision_info['final_test_evaluated']:
        raise ValueError('Expected real-model development sources')
    split = json.loads((source / 'split_snapshot.json').read_text(encoding='utf-8'))
    if split != json.loads((decisions / 'split_snapshot.json').read_text(encoding='utf-8')):
        raise ValueError('Enrollment and decision identity splits differ')
    rows = read_csv(source / 'combined_extraction.csv')
    assert_development(rows, split)
    feature_path = ROOT / info['feature_cache']
    features = np.load(feature_path, allow_pickle=False)['embeddings']
    used_indices = [int(r['embedding_index']) for r in rows if r['status'] == 'ok']
    if features.ndim != 2 or features.shape[1] != 512 or max(used_indices) >= len(features):
        raise ValueError('Embedding cache does not match extraction indices')
    if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1., atol=1e-5):
        raise ValueError('Expected finite unit embeddings')
    inputs = [config_path, source / 'comparison.json', source / 'split_snapshot.json',
              source / 'combined_extraction.csv', source / 'predictions.csv', feature_path,
              decisions / 'comparison.json', decisions / 'split_snapshot.json']
    fingerprints = {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs}
    destination = ROOT / 'results/enrollment_pooling' / ('pooling_development_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (destination / 'split_snapshot.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    frozen = decision_info['selected_thresholds']['score_margin']
    all_predictions, summaries, enrollment_log, effects = [], [], [], []
    baseline_rows = read_csv(source / 'predictions.csv')
    baseline_checks = 0
    for dataset, known_ids in [('halfface', split['halfface']['development']['known']),
                               ('icfd_sample', split['icfd_sample']['engineering_development'])]:
        probes = [r for r in rows if r['dataset'] == dataset and
                  ((r['truth_status'] == 'known' and r['mask_status'] == 'masked') or r['truth_status'] == 'unknown')]
        if dataset == 'icfd_sample':
            probes = [r for r in rows if r['dataset'] == dataset and r['evaluation_role'] == 'probe']
        paired = {}
        for method in METHODS:
            strategy = 'one_photo' if method == 'one_photo' else 'up_to_three_photos'
            candidates = gallery_candidates(rows, dataset, known_ids, strategy)
            reject_overlap([r for selected in candidates.values() for r in selected], probes)
            prototypes, ids, used_count = [], [], 0
            for sid, selected in candidates.items():
                usable = [r for r in selected if r['status'] == 'ok']
                if not usable:
                    raise ValueError('Missing gallery identity; do not silently exclude it')
                template, weights, fallback = pool_template(usable, features, method)
                ids.append(sid)
                prototypes.append(template)
                used_count += int(np.count_nonzero(weights))
                for row, weight in zip(usable, weights):
                    enrollment_log.append(dict(dataset=dataset, method=method, subject_id=sid,
                        path=row['path'], weight=float(weight), quality=float(row['quality']),
                        detector_confidence=float(row['det_score']), zero_weight_fallback=fallback))
            matrix = np.stack(prototypes)
            scored = []
            for row in probes:
                out = {k: row[k] for k in ('dataset', 'subject_id', 'path', 'truth_status', 'status', 'quality')}
                out.update(method=method, top1_id='', top1_score='', top2_score='', rank1_correct=False)
                if row['status'] == 'ok':
                    values = matrix @ features[int(row['embedding_index'])]
                    order = np.argsort(values)[::-1]
                    out.update(top1_id=ids[int(order[0])], top1_score=float(values[order[0]]), top2_score=float(values[order[1]]))
                    out['rank1_correct'] = out['truth_status'] == 'known' and out['top1_id'] == out['subject_id']
                scored.append(out)
            if method in ('one_photo', 'uniform_mean'):
                reference = {r['path']: r for r in baseline_rows if r['dataset'] == dataset and r['strategy'] == strategy}
                if set(reference) != {r['path'] for r in scored}:
                    raise ValueError('Baseline probes differ')
                for row in scored:
                    old = reference[row['path']]
                    if row['status'] != old['status'] or row['top1_id'] != old['top1_id']:
                        raise ValueError('Baseline identity/status mismatch')
                    if row['status'] == 'ok' and any(abs(row[k] - float(old[k])) > 1e-7 for k in ('top1_score', 'top2_score')):
                        raise ValueError('Uniform/original enrollment failed baseline reproduction')
                    baseline_checks += 1
            if dataset == 'halfface':
                threshold = choose_threshold([r['top1_score'] for r in scored if r['status'] == 'ok' and r['truth_status'] == 'unknown'], config['target_conditional_fpir'])
                rules = [('recalibrated_score', threshold, 0.),
                         ('frozen_margin_sensitivity', frozen['score_threshold'], frozen['margin_threshold'])]
            else:
                rules = [('closed_set_engineering', None, None)]
            for rule, threshold, margin in rules:
                predicted = apply_decision(scored, threshold, margin) if threshold is not None else [
                    dict(r, decision='CLOSED_SET_ONLY' if r['status'] == 'ok' else 'CAPTURE_FAILURE',
                         accepted_id='', threshold='', margin_threshold='', correct_known_accept=False) for r in scored]
                for row in predicted:
                    row['rule'] = rule
                paired[(method, rule)] = predicted
                all_predictions.extend(predicted)
                summaries.append(summarize(predicted, dataset, method, rule, threshold, margin, len(ids), used_count))
        for rule in sorted({key[1] for key in paired}):
            for method in METHODS:
                if method == 'uniform_mean':
                    continue
                key = 'rank1_correct' if dataset == 'icfd_sample' else 'correct_known_accept'
                effects.append(dict(dataset=dataset, method=method, reference='uniform_mean', rule=rule,
                    **paired_bootstrap(paired[('uniform_mean', rule)], paired[(method, rule)], known_ids,
                                       config['bootstrap_samples'], config['seed'], key)))
    if any(sha256(ROOT / p) != digest for p, digest in fingerprints.items()):
        raise ValueError('Input mutated during comparison')
    write_csv(destination / 'summary.csv', summaries)
    write_csv(destination / 'predictions.csv', all_predictions)
    write_csv(destination / 'enrollment_weights.csv', enrollment_log)
    write_csv(destination / 'paired_effects.csv', effects)
    report = dict(run_directory=destination.relative_to(ROOT).as_posix(), configuration=config,
        input_sha256=fingerprints, source_sha256={p: sha256(ROOT / p) for p in [
            'scripts/compare_enrollment_pooling.py', 'scripts/compare_enrollment.py', 'scripts/run_arcface_baseline.py', 'src/open_set/decision.py']},
        model=info['model'], summaries=summaries, paired_effects=effects,
        baseline_probe_method_checks=baseline_checks, probe_method_rule_attempts=len(all_predictions),
        final_test_evaluated=False, new_model_inference=False, training_performed=False,
        python=platform.python_version(), numpy=np.__version__,
        git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True).stdout.strip(),
        limitations=[
            'Descriptive development comparison: calibration and evaluation reuse identities and images; not unseen-test performance.',
            'Frozen-margin sensitivity applies the existing uniform-mean settings to other pooling policies; it is not equal-FPIR calibration.',
            'ICFD public sample is closed-set engineering evidence with five development identities, not an open-set benchmark.',
            'Paired intervals condition on the calibrated thresholds and omit calibration and unknown-sample uncertainty.',
            'Heuristic quality and detector confidence are existing proxies, not new learned face-quality methods.',
            'No real CCTV, temporal, population-generalization or method-novelty claim follows from these experiments.'])
    (destination / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    lines = ['# Enrollment pooling: descriptive development evidence', '',
        'Same real ArcFace features, enrollment candidates and probes. No training or reserved-test access.', '',
        '| Dataset | Pooling | Decision rule | Correct known / attempts | False known / extracted unknown |',
        '|---|---|---|---:|---:|']
    for row in summaries:
        known = f"{row['rank1_correct']}/{row['known_attempts']} retrieval" if row['correct_known_accepts'] is None else f"{row['correct_known_accepts']}/{row['known_attempts']}"
        unknown = 'not available' if row['false_known_unknowns'] is None else f"{row['false_known_unknowns']}/{row['unknown_with_embedding']}"
        lines.append(f"| {row['dataset']} | {row['method']} | {row['rule']} | {known} | {unknown} |")
    lines += ['', '## Interpretation limits', ''] + [f'- {item}' for item in report['limitations']]
    lines += ['', f'Baseline reproduction: {baseline_checks} original probe/method attempts.',
              'Per-probe scores and decisions, weights, paired effects, input hashes and configuration are saved beside this report.', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(run_directory=report['run_directory'], summaries=summaries,
                         baseline_probe_method_checks=baseline_checks, attempts=len(all_predictions)), indent=2))


if __name__ == '__main__':
    main()
