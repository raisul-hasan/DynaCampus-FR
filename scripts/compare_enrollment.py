"""Paired development comparison of one versus up to three enrollment photos.

Uses frozen baseline embeddings, extracting only additional ICFD gallery images.
halfFace masked probes are identical across methods and never enter enrollment.
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_arcface_baseline import choose_threshold, extract, read_csv, sha256, write_csv

STRATEGIES = ('one_photo', 'up_to_three_photos')


def unit_mean(vectors):
    mean = np.asarray(vectors, dtype=np.float64).mean(axis=0)
    norm = np.linalg.norm(mean)
    if not np.isfinite(norm) or norm <= 1e-10:
        raise ValueError('Enrollment prototype is non-finite or zero')
    return (mean / norm).astype(np.float32)


def gallery_candidates(rows, dataset, known_ids, strategy):
    if strategy not in STRATEGIES:
        raise ValueError('Unknown enrollment strategy')
    selected = {}
    for subject in sorted(known_ids):
        candidates = [r for r in rows if r['dataset'] == dataset and r['subject_id'] == subject and
                      (r['mask_status'] == 'unmasked' if dataset == 'halfface' else r['evaluation_role'] == 'gallery')]
        if dataset == 'halfface':
            candidates.sort(key=lambda r: ({'front': 0, 'left': 1, 'right': 2}[r['pose']], r['path']))
            if strategy == 'one_photo':
                candidates = [r for r in candidates if r['pose'] == 'front']
        else:
            candidates.sort(key=lambda r: r['path'])
        selected[subject] = candidates[:1 if strategy == 'one_photo' else 3]
        if not selected[subject]:
            raise ValueError(f'No source enrollment candidates for {subject}')
    return selected


def reject_overlap(gallery_rows, probes):
    if {r['path'] for r in gallery_rows} & {r['path'] for r in probes}:
        raise ValueError('Enrollment/probe image leakage')


def paired_bootstrap(one, few, known_ids, samples, seed, key):
    ids = sorted(known_ids)
    one_by_path = {r['path']: r for r in one if r['truth_status'] == 'known'}
    few_by_path = {r['path']: r for r in few if r['truth_status'] == 'known'}
    if one_by_path.keys() != few_by_path.keys():
        raise ValueError('Paired methods have different known probes')
    counts = np.array([sum(r['subject_id'] == s for r in one_by_path.values()) for s in ids])
    if not counts.sum():
        raise ValueError('No paired known probes')
    first = np.array([sum(bool(r[key]) for r in one_by_path.values() if r['subject_id'] == s) for s in ids])
    second = np.array([sum(bool(r[key]) for r in few_by_path.values() if r['subject_id'] == s) for s in ids])
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(samples, len(ids)))
    deltas = (second[draws].sum(axis=1) - first[draws].sum(axis=1)) / counts[draws].sum(axis=1)
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {'metric': key, 'delta_percentage_points': float(100 * (second.sum() - first.sum()) / counts.sum()),
            'ci95_low_percentage_points': float(100 * low), 'ci95_high_percentage_points': float(100 * high),
            'bootstrap_unit': 'known_identity', 'bootstrap_samples': samples,
            'note': 'Paired development bootstrap with fixed calibrated thresholds; calibration and unknown-sample uncertainty are not included.'}


def evaluate_method(rows, features, dataset, known_ids, strategy, target):
    candidates = gallery_candidates(rows, dataset, known_ids, strategy)
    prototypes, gallery_ids, enrollment_log = [], [], []
    for subject, selected in candidates.items():
        usable = [r for r in selected if r['status'] == 'ok']
        for row in selected:
            enrollment_log.append({'dataset': dataset, 'strategy': strategy, 'subject_id': subject,
                                   'path': row['path'], 'status': row['status'], 'used_in_template': row['status'] == 'ok'})
        if usable:
            gallery_ids.append(subject)
            prototypes.append(unit_mean([features[int(r['embedding_index'])] for r in usable]))
    if len(gallery_ids) != len(known_ids):
        raise ValueError('Enrollment failures require explicit handling before a paired comparison')
    matrix = np.stack(prototypes)
    probes = [r for r in rows if r['dataset'] == dataset and
              ((r['truth_status'] == 'known' and r['mask_status'] == 'masked') or r['truth_status'] == 'unknown')]
    if dataset == 'icfd_sample':
        probes = [r for r in rows if r['dataset'] == dataset and r['evaluation_role'] == 'probe']
    reject_overlap([r for selected in candidates.values() for r in selected], probes)
    predictions = []
    for row in probes:
        output = {key: row[key] for key in ('dataset', 'subject_id', 'path', 'truth_status', 'mask_status', 'status')}
        output.update({'strategy': strategy, 'top1_id': '', 'top1_score': '', 'top2_score': '',
                       'rank1_correct': False, 'decision': '', 'correct_known_accept': False, 'threshold': ''})
        if row['status'] == 'ok':
            scores = matrix @ features[int(row['embedding_index'])]
            order = np.argsort(scores)[::-1]
            output.update({'top1_id': gallery_ids[int(order[0])], 'top1_score': float(scores[order[0]]),
                           'top2_score': float(scores[order[1]]) if len(order) > 1 else -1.0})
            output['rank1_correct'] = output['truth_status'] == 'known' and output['top1_id'] == output['subject_id']
        predictions.append(output)
    unknown = [r for r in predictions if r['truth_status'] == 'unknown']
    unknown_ok = [r for r in unknown if r['status'] == 'ok']
    threshold = choose_threshold([r['top1_score'] for r in unknown_ok], target) if dataset == 'halfface' else None
    for row in predictions:
        row['threshold'] = threshold if threshold is not None else ''
        row['decision'] = (('KNOWN' if row['top1_score'] >= threshold else 'UNKNOWN') if threshold is not None else 'CLOSED_SET_ONLY') if row['status'] == 'ok' else row['status'].upper()
        row['correct_known_accept'] = row['rank1_correct'] and row['decision'] == 'KNOWN'
    known = [r for r in predictions if r['truth_status'] == 'known']
    successes = sum(r['rank1_correct'] for r in known)
    accepted = sum(r['correct_known_accept'] for r in known)
    false_known = sum(r['decision'] == 'KNOWN' for r in unknown_ok)
    used_counts = Counter(r['subject_id'] for r in enrollment_log if r['used_in_template'])
    summary = {'dataset': dataset, 'scope': 'development_only' if dataset == 'halfface' else 'engineering_sample_only',
               'strategy': strategy, 'gallery_identities': len(gallery_ids), 'gallery_images': sum(used_counts.values()),
               'mean_gallery_images_per_identity': float(np.mean(list(used_counts.values()))),
               'known_probe_attempts': len(known), 'known_probes_with_embedding': sum(r['status'] == 'ok' for r in known),
               'rank1_correct': successes, 'rank1_end_to_end': successes / len(known),
               'correct_known_accepts': accepted if threshold is not None else None,
               'correct_known_accept_end_to_end': accepted / len(known) if threshold is not None else None,
               'unknown_attempts': len(unknown), 'unknown_with_embedding': len(unknown_ok),
               'false_known_count': false_known if unknown_ok else None,
               'fpir_conditional': false_known / len(unknown_ok) if unknown_ok else None,
               'threshold': threshold, 'enrollment_image_count_distribution': dict(Counter(used_counts.values()))}
    return summary, predictions, enrollment_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/enrollment_comparison.yaml')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding='utf-8'))
    baseline = ROOT / config['baseline_run']
    info = json.loads((baseline / 'run_info.json').read_text())
    if info['model']['mock'] or info['final_test_evaluated']:
        raise ValueError('Expected an explicit real-model development run')
    for source, expected in json.loads((baseline / 'source_fingerprints.json').read_text()).items():
        if sha256(ROOT / source) != expected:
            raise ValueError(f'Baseline source has changed: {source}. Cache compatibility must be reviewed.')
    split = json.loads((baseline / 'split_snapshot.json').read_text())
    reserved = set(sum(split['halfface']['test_reserved'].values(), []) + split['icfd_sample']['reserved'])
    baseline_config = yaml.safe_load((baseline / 'config.yaml').read_text())
    for name, key in [('halfface', 'halfface_manifest'), ('icfd', 'icfd_manifest')]:
        if sha256(ROOT / baseline_config[key]) != split['manifest_sha256'][name]:
            raise ValueError('Source manifest changed after split freeze')
    records = read_csv(baseline / 'extraction.csv')
    features = np.load(ROOT / info['model']['embedding_cache'], allow_pickle=False)['embeddings']
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = ROOT / 'results' / 'enrollment' / f'enrollment_development_{timestamp}'
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    (destination / 'split_snapshot.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    sample_ids = set(split['icfd_sample']['engineering_development'])
    source_rows = read_csv(ROOT / baseline_config['icfd_manifest'])
    extra = []
    existing = {r['path'] for r in records}
    for subject in sorted(sample_ids):
        candidates = sorted([r for r in source_rows if r['subject_id'] == subject and r['source_role'] == 'gallery'], key=lambda r: r['path'])[:3]
        for row in candidates:
            if row['path'] not in existing:
                extra.append({**row, 'mask_status': 'unspecified', 'pose': 'unspecified',
                              'evaluation_role': 'gallery', 'truth_status': 'known'})
    if any(r['subject_id'] in reserved for r in records + extra):
        raise ValueError('Reserved identity entered development data')
    model_info = info['model']
    if extra:
        additional, extra_features, extra_model = extract(extra, baseline_config, destination)
        for key in ('recognizer_sha256', 'detector_sha256'):
            if extra_model[key] != model_info[key]:
                raise ValueError('Additional embeddings use different checkpoints')
        offset = len(features)
        for row in additional:
            if int(row['embedding_index']) >= 0:
                row['embedding_index'] = int(row['embedding_index']) + offset
        records += additional
        features = np.concatenate([features, extra_features])
    write_csv(destination / 'combined_extraction.csv', records)
    cache = ROOT / 'data' / 'embeddings' / destination.name
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache / 'embeddings.npz', embeddings=features)
    summaries, all_predictions, all_enrollment, effects = [], [], [], []
    for dataset, known_ids in [('halfface', split['halfface']['development']['known']), ('icfd_sample', sorted(sample_ids))]:
        paired = {}
        for strategy in STRATEGIES:
            summary, predictions, enrollment = evaluate_method(records, features, dataset, known_ids, strategy, config['target_conditional_fpir'])
            summaries.append(summary)
            all_predictions.extend(predictions)
            all_enrollment.extend(enrollment)
            paired[strategy] = predictions
        if {r['path'] for r in paired[STRATEGIES[0]]} != {r['path'] for r in paired[STRATEGIES[1]]}:
            raise ValueError('Methods use different probes')
        for key in ['rank1_correct'] + (['correct_known_accept'] if dataset == 'halfface' else []):
            effects.append({'dataset': dataset, **paired_bootstrap(paired[STRATEGIES[0]], paired[STRATEGIES[1]], known_ids,
                config['bootstrap_samples'], config['seed'], key)})
    write_csv(destination / 'predictions.csv', all_predictions)
    write_csv(destination / 'enrollment_sources.csv', all_enrollment)
    write_csv(destination / 'comparison.csv', [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in summaries])
    write_csv(destination / 'paired_bootstrap.csv', effects)
    result = {'run_directory': destination.relative_to(ROOT).as_posix(), 'baseline_run': config['baseline_run'],
              'configuration': config, 'model': model_info, 'summaries': summaries, 'paired_effects': effects,
              'additional_icfd_images': len(extra), 'feature_cache': (cache / 'embeddings.npz').relative_to(ROOT).as_posix(),
              'final_test_evaluated': False, 'source_sha256': sha256(__file__),
              'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True).stdout.strip(),
              'evidence_scope': 'Development comparison. ICFD public sample is engineering evidence only.',
              'protocol_note': 'halfFace known probes are masked-only for both methods; all development unknown images are unchanged. This denominator differs from the earlier mixed-mask baseline.'}
    (destination / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
