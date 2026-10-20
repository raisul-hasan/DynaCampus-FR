"""Calibrate nested decision families on cached development scores only.

No model inference or reserved-test access. Failures stay in end-to-end counts;
unknown FPIR uses every successfully embedded unknown, including refused inputs.
"""
import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_enrollment import paired_bootstrap
from scripts.run_arcface_baseline import read_csv, sha256, write_csv
from src.open_set import OpenSetDecisionEngine


def data_arrays(rows):
    valid = np.array([r['status'] == 'ok' for r in rows])
    known = np.array([r['truth_status'] == 'known' for r in rows])
    score = np.array([float(r['top1_score']) if r['status'] == 'ok' else -2 for r in rows])
    margin = np.array([float(r['top1_score']) - float(r['top2_score']) if r['status'] == 'ok' else 0 for r in rows])
    quality = np.array([float(r['quality']) if r['status'] == 'ok' else 0 for r in rows])
    correct = np.array([r['top1_id'] == r['subject_id'] and r['truth_status'] == 'known' and r['status'] == 'ok' for r in rows])
    return valid, known, score, margin, quality, correct


def calibrate(rows, family, config):
    valid, known, score, margin, quality, correct = data_arrays(rows)
    if not (valid & ~known).any() or not (valid & known).any():
        raise ValueError('Known and unknown development evidence is required')
    levels = np.unique(score[valid])
    thresholds = np.unique(np.r_[-1.000001, levels, np.nextafter(levels, np.inf), 1.000001])
    margins = [0.] if family == 'score_only' else config['margin_grid']
    qualities = config['quality_grid'] if family == 'score_margin_quality' else [0.]
    grid, best = [], None
    for qm in qualities:
        for mm in margins:
            accepted = valid[None, :] & (quality[None, :] >= qm) & (margin[None, :] >= mm) & (score[None, :] >= thresholds[:, None])
            covered = valid[None, :] & (quality[None, :] >= qm) & ((score[None, :] < thresholds[:, None]) | (margin[None, :] >= mm))
            true_accepts = (accepted & correct).sum(axis=1)
            false_known = (accepted & ~known).sum(axis=1)
            misidentified_known = (accepted & known & ~correct).sum(axis=1)
            known_coverage = (covered & known).sum(axis=1) / (valid & known).sum()
            unknown_coverage = (covered & ~known).sum(axis=1) / (valid & ~known).sum()
            fpir = false_known / (valid & ~known).sum()
            feasible = ((fpir <= config['target_conditional_fpir'] + 1e-12) &
                        (known_coverage >= config['minimum_known_valid_coverage']) &
                        (unknown_coverage >= config['minimum_unknown_valid_coverage']))
            coverage_count = covered.sum(axis=1)
            for index, threshold in enumerate(thresholds):
                row = {'family': family, 'score_threshold': float(threshold), 'margin_threshold': float(mm),
                       'quality_threshold': float(qm), 'correct_known_accepts': int(true_accepts[index]),
                       'false_known_unknowns': int(false_known[index]), 'wrong_known_identity_accepts': int(misidentified_known[index]),
                       'valid_decision_coverage': float(coverage_count[index] / valid.sum()),
                       'known_valid_coverage': float(known_coverage[index]), 'unknown_valid_coverage': float(unknown_coverage[index]),
                       'fpir_conditional': float(fpir[index]), 'feasible': bool(feasible[index])}
                grid.append(row)
                if feasible[index]:
                    key = (-row['correct_known_accepts'], row['false_known_unknowns'], row['wrong_known_identity_accepts'],
                           -int(coverage_count[index]), float(qm), float(mm), -float(threshold))
                    if best is None or key < best[0]:
                        best = (key, row)
    if best is None:
        raise ValueError(f'No feasible calibration for {family}')
    return best[1], grid


def apply_rule(rows, thresholds, family):
    engine = OpenSetDecisionEngine(base_threshold=thresholds['score_threshold'],
        margin_threshold=thresholds['margin_threshold'], quality_threshold=thresholds['quality_threshold'])
    predictions = []
    for source in rows:
        row = dict(source)
        row['family'] = family
        if row['status'] != 'ok':
            state, identity = 'CAPTURE_FAILURE', None
        else:
            state, identity, _ = engine.evaluate_scores(row['top1_id'], float(row['top1_score']), float(row['top2_score']), float(row['quality']))
        row['decision'] = state
        row['accepted_id'] = identity or ''
        row['correct_known_accept'] = state == 'KNOWN' and identity == row['subject_id'] and row['truth_status'] == 'known'
        row['covered'] = state in {'KNOWN', 'UNKNOWN'}
        predictions.append(row)
    return predictions


def summarize(predictions, thresholds, family):
    known = [r for r in predictions if r['truth_status'] == 'known']
    unknown = [r for r in predictions if r['truth_status'] == 'unknown']
    valid = [r for r in predictions if r['status'] == 'ok']
    unknown_ok = [r for r in unknown if r['status'] == 'ok']
    covered = [r for r in predictions if r['covered']]
    true_accept = sum(r['correct_known_accept'] for r in known)
    false_known = sum(r['decision'] == 'KNOWN' for r in unknown)
    correct_covered = sum(r['correct_known_accept'] or (r['truth_status'] == 'unknown' and r['decision'] == 'UNKNOWN') for r in covered)
    return {'family': family, 'known_probe_attempts': len(known), 'unknown_probe_attempts': len(unknown),
            'known_with_embedding': sum(r['status'] == 'ok' for r in known), 'unknown_with_embedding': len(unknown_ok),
            'correct_known_accepts': true_accept, 'correct_known_accept_end_to_end': true_accept / len(known),
            'false_known_unknowns': false_known, 'fpir_conditional': false_known / len(unknown_ok),
            'wrong_known_identity_accepts': sum(r['decision'] == 'KNOWN' and not r['correct_known_accept'] for r in known),
            'valid_decision_coverage': len(covered) / len(valid), 'end_to_end_decision_coverage': len(covered) / len(predictions),
            'accuracy_among_covered': correct_covered / len(covered) if covered else None,
            'decision_counts': dict(Counter(r['decision'] for r in predictions)),
            'score_threshold': thresholds['score_threshold'], 'margin_threshold': thresholds['margin_threshold'],
            'quality_threshold': thresholds['quality_threshold']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/decision_ablation.yaml')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    if config['final_test_evaluation'] or not 0 <= config['target_conditional_fpir'] < 1:
        raise ValueError('Only valid development calibration is supported')
    source = ROOT / config['enrollment_run']
    source_info = json.loads((source / 'comparison.json').read_text())
    if source_info['final_test_evaluated'] or source_info['model']['mock']:
        raise ValueError('Expected real-model development evidence')
    rows = [r for r in read_csv(source / 'predictions.csv') if r['dataset'] == config['dataset'] and r['strategy'] == config['enrollment_strategy']]
    extraction = {r['path']: r for r in read_csv(source / 'combined_extraction.csv')}
    split = json.loads((source / 'split_snapshot.json').read_text())
    reserved = set(sum(split['halfface']['test_reserved'].values(), []))
    if not rows or reserved & {r['subject_id'] for r in rows}:
        raise ValueError('Empty development inputs or reserved identity leakage')
    for row in rows:
        row['quality'] = extraction[row['path']]['quality']
        if row['status'] == 'ok' and not math.isfinite(float(row['quality'])):
            raise ValueError('Quality must be available for every embedded probe')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = ROOT / 'results' / 'decisions' / f'decisions_development_{timestamp}'
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (destination / 'split_snapshot.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    write_csv(destination / 'input_scores.csv', rows)
    summaries, grid_rows, all_predictions, effects = [], [], [], []
    predictions_by_family = {}
    selected = {}
    for family in config['families']:
        thresholds, grid = calibrate(rows, family, config)
        predictions = apply_rule(rows, thresholds, family)
        summary = summarize(predictions, thresholds, family)
        assert summary['correct_known_accepts'] == thresholds['correct_known_accepts']
        assert summary['false_known_unknowns'] == thresholds['false_known_unknowns']
        summaries.append(summary)
        selected[family] = thresholds
        predictions_by_family[family] = predictions
        grid_rows.extend(grid)
        all_predictions.extend(predictions)
        print(json.dumps(summary), flush=True)
    for family in config['families'][1:]:
        effects.append({'comparison': f'{family}_minus_score_only', **paired_bootstrap(
            predictions_by_family['score_only'], predictions_by_family[family], split['halfface']['development']['known'],
            config['bootstrap_samples'], config['seed'], 'correct_known_accept')})
    sensitivity = []
    fixed = selected['score_margin']
    for q in np.arange(0, 1.000001, config['quality_sensitivity_step']):
        thresholds = {**fixed, 'quality_threshold': float(q)}
        summary = summarize(apply_rule(rows, thresholds, 'quality_sensitivity'), thresholds, 'quality_sensitivity')
        sensitivity.append({k: v for k, v in summary.items() if not isinstance(v, dict)})
    write_csv(destination / 'predictions.csv', all_predictions)
    write_csv(destination / 'calibration_grid.csv', grid_rows)
    write_csv(destination / 'ablation.csv', [{k: v for k, v in s.items() if not isinstance(v, dict)} for s in summaries])
    write_csv(destination / 'quality_sensitivity.csv', sensitivity)
    write_csv(destination / 'paired_effects.csv', effects)
    result = {'run_directory': destination.relative_to(ROOT).as_posix(), 'configuration': config,
              'selected_thresholds': selected, 'summaries': summaries, 'paired_effects': effects,
              'source_predictions_sha256': sha256(source / 'predictions.csv'),
              'source_extraction_sha256': sha256(source / 'combined_extraction.csv'),
              'script_sha256': sha256(__file__), 'engine_sha256': sha256(ROOT / 'src/open_set/decision.py'),
              'grid_configurations': len(grid_rows), 'final_test_evaluated': False, 'new_model_inference': False,
              'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True).stdout.strip(),
              'limitations': ['Development calibration and evaluation reuse the same identities; not held-out performance.',
                              'Zero margin/quality gates are valid nested choices; do not claim benefit when a gate selects zero.',
                              'Conditional unknown FPIR denominator includes all extracted unknowns, even those refused by gates.',
                              'Capture failures stay in end-to-end known counts and are separate from the four decision states.']}
    (destination / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('RESULTS', destination, 'GRID', len(grid_rows), flush=True)


if __name__ == '__main__':
    main()
