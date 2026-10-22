"""Prepare, then run corrected sequence-separated ChokePoint temporal ablations.

No halfFace/ICFD inputs, gallery files or app defaults are used or modified.
All biometric arrays remain in ignored data/. Public results contain scores,
anonymous benchmark IDs, manifests and aggregate statistics, never face images.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.chokepoint import (LEVELS, stable_seed, unit, degrade,
    laplacian_quality, consecutive_tracks, aggregate, accept, calibrate,
    verification_metrics, paired_identity_interval)
from src.recognition import FaceEmbedder


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty table: {path}')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(config_path):
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    raw = ROOT / cfg['raw_root']
    roles = {'gallery': [cfg['gallery_sequence']],
             'calibration': cfg['calibration_sequences'], 'evaluation': cfg['evaluation_sequences']}
    sequences = sum(roles.values(), [])
    if len(set(sequences)) != len(sequences):
        raise ValueError('Enrollment/calibration/evaluation sequences overlap')
    inventory, by_sequence, groundtruth_hashes = [], {}, {}
    for seq in sequences:
        xml = raw / 'groundtruth' / (seq + '.xml')
        groundtruth_hashes[xml.relative_to(ROOT).as_posix()] = sha(xml)
        truth = {int(f.attrib['number']): {p.attrib['id'] for p in f.findall('person')}
                 for f in ET.parse(xml).getroot().findall('frame')}
        by_sequence[seq] = {}
        for path in sorted((raw / seq).rglob('*.pgm')):
            identity, frame = path.parent.name, int(path.stem)
            if identity not in truth.get(frame, set()):
                raise ValueError(f'Crop/ground-truth identity mismatch: {path}')
            image = cv2.imread(str(path))
            if image is None or image.shape != (96, 96, 3):
                raise ValueError(f'Invalid official crop: {path}')
            row = {'identity': identity, 'sequence': seq, 'frame': frame,
                   'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
                   'width': 96, 'height': 96}
            inventory.append(row)
            by_sequence[seq].setdefault(identity, []).append(row)
    identities = sorted(by_sequence[cfg['gallery_sequence']])
    if len(identities) != 25 or any(set(by_sequence[s]) != set(identities) for s in sequences):
        raise ValueError('Expected the same 25 Portal-1 identities in each selected sequence')
    shuffled = identities.copy()
    np.random.RandomState(cfg['seed']).shuffle(shuffled)
    known = shuffled[:cfg['known_identities']]
    unknown = shuffled[cfg['known_identities']:]
    galleries = [min(by_sequence[cfg['gallery_sequence']][i], key=lambda x: x['frame']) for i in known]
    tracks, accounting = [], []
    for role in ['calibration', 'evaluation']:
        for seq in roles[role]:
            for identity in identities:
                source = by_sequence[seq][identity]
                chunks = consecutive_tracks(source, cfg['track_size'], cfg['max_frame_gap'],
                                             cfg['max_tracks_per_identity_per_sequence'])
                accounting.append({'role': role, 'sequence': seq, 'identity': identity,
                                   'available_frames': len(source), 'selected_tracks': len(chunks),
                                   'unused_frames': len(source) - sum(map(len, chunks))})
                if not chunks:
                    raise ValueError(f'No valid track for {role}/{seq}/{identity}')
                for index, chunk in enumerate(chunks):
                    tid = f'{seq}/{identity}/{index:03d}'
                    mixed = np.random.default_rng(stable_seed(cfg['seed'], tid, 'mixed')).choice(
                        LEVELS, size=cfg['track_size'], p=cfg['mixed_probabilities']).tolist()
                    tracks.append({'track_id': tid, 'role': role, 'sequence': seq, 'identity': identity,
                                   'truth': 'known' if identity in known else 'unknown',
                                   'paths': [r['path'] for r in chunk], 'frames': [r['frame'] for r in chunk],
                                   'mixed_levels': mixed,
                                   'random_index': int(stable_seed(cfg['seed'], tid, 'random') % cfg['track_size'])})
    for t in tracks:
        # One intruder path per track, reused across every method and control.
        if t['truth'] == 'known':
            pool = sorted({p for other in tracks if other['role'] == t['role']
                           and other['truth'] == 'known' and other['identity'] != t['identity']
                           for p in other['paths']})
            t['intruder_path'] = pool[stable_seed(cfg['seed'], t['track_id'], 'intruder') % len(pool)]
        else:
            t['intruder_path'] = None
    runid = 'temporal_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest = ROOT / 'results/chokepoint' / runid
    dest.mkdir(parents=True, exist_ok=False)
    save_json(dest / 'config.json', cfg)
    save_json(dest / 'split.json', {'known': known, 'unknown': unknown, 'sequences': roles,
                                   'identity_selection': 'sorted official IDs, NumPy RandomState seed 42'})
    save_json(dest / 'tracks.json', tracks)
    write_csv(dest / 'inventory.csv', inventory)
    write_csv(dest / 'selection_accounting.csv', accounting)
    write_csv(dest / 'gallery.csv', galleries)
    source_files = ['src/evaluation/chokepoint.py', 'scripts/run_chokepoint_temporal.py',
                    'src/recognition/embedder.py']
    receipt = {'prepared_utc': datetime.now(timezone.utc).isoformat(), 'status': 'prepared_before_inference',
               'config_sha256': sha(dest / 'config.json'),
               'source_sha256': {s: sha(ROOT / s) for s in source_files},
               'manifest_sha256': {s: sha(dest / s) for s in ['tracks.json', 'inventory.csv', 'gallery.csv', 'split.json']},
               'groundtruth_sha256': groundtruth_hashes,
               'dataset_receipt_sha256': sha(ROOT / 'data/sources/chokepoint/receipt.json'),
               'counts': dict(Counter(f"{t['role']}_{t['truth']}" for t in tracks)),
               'scope': cfg['scope']}
    save_json(dest / 'preparation.json', receipt)
    print('Prepared:', dest.relative_to(ROOT), flush=True)
    print(json.dumps(receipt['counts']), flush=True)


def extract(dest, cfg, tracks):
    cache = ROOT / 'data/embeddings/chokepoint' / dest.name
    cache.mkdir(parents=True, exist_ok=True)
    gallery_rows = list(csv.DictReader((dest / 'gallery.csv').open(encoding='utf-8')))
    selected = sorted({p for t in tracks for p in t['paths']} | {r['path'] for r in gallery_rows})
    inventory = {r['path']: r for r in csv.DictReader((dest / 'inventory.csv').open(encoding='utf-8'))}
    for p in selected:
        if sha(ROOT / p) != inventory[p]['sha256']:
            raise ValueError('Original crop changed: ' + p)
    cv2.setNumThreads(1)
    ort.set_default_logger_severity(3)
    embedder = FaceEmbedder(cpu_threads=cfg['cpu_threads'])
    embedder.model.session.set_providers(['CPUExecutionProvider'])
    runtime = {'python': sys.version, 'platform': platform.platform(), 'opencv': cv2.__version__,
               'numpy': np.__version__, 'onnxruntime': ort.__version__,
               'providers': embedder.model.session.get_providers(),
               'model_sha256': sha(embedder.model_path), 'model_path': str(embedder.model_path),
               'cpu_threads': cfg['cpu_threads'], 'batch_size': cfg['inference_batch_size']}
    print('Extracting', len(selected), 'source crops x four corruption levels', flush=True)
    lookup, frame_records = {}, []
    for level in LEVELS:
        npz, meta = cache / (level + '.npz'), cache / (level + '.json')
        if npz.exists() and meta.exists():
            records = json.loads(meta.read_text())
            features = np.load(npz, allow_pickle=False)['embeddings']
            print('Resume verified level:', level, flush=True)
        else:
            features, records, batch, pending = [], [], [], []
            start = time.perf_counter()
            # Clean gallery only; its other corruption features are harmless cache overhead.
            for path in selected:
                image = cv2.imread(str(ROOT / path))
                image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)
                seed = stable_seed(cfg['seed'], path, level)
                corrupted = degrade(image, level, seed)
                pending.append({'path': path, 'level': level, 'seed': seed,
                                'pixel_sha256': hashlib.sha256(corrupted.tobytes()).hexdigest(),
                                'laplacian': laplacian_quality(corrupted)})
                batch.append(corrupted)
                if len(batch) == cfg['inference_batch_size'] or path == selected[-1]:
                    raw = np.asarray(embedder.model.get_feat(batch))
                    if raw.shape != (len(batch), 512) or not np.isfinite(raw).all():
                        raise ValueError('Invalid real ArcFace output')
                    for feat, row in zip(raw, pending):
                        row['norm'] = float(np.linalg.norm(feat))
                        features.append(unit(feat))
                        records.append(row)
                    batch, pending = [], []
                    if len(records) % 256 == 0 or len(records) == len(selected):
                        print(f'{level}: {len(records)}/{len(selected)} in {time.perf_counter()-start:.1f}s', flush=True)
            features = np.stack(features)
            np.savez_compressed(npz, embeddings=features)
            save_json(meta, records)
        if len(features) != len(selected) or [r['path'] for r in records] != selected:
            raise ValueError('Feature cache manifest mismatch')
        for row, feat in zip(records, features):
            lookup[(row['path'], level)] = (feat, row['norm'], row['laplacian'])
        frame_records.extend(records)
    save_json(cache / 'runtime.json', runtime)
    write_csv(dest / 'frame_quality.csv', frame_records)
    save_json(dest / 'cache_receipt.json', {'cache': cache.relative_to(ROOT).as_posix(),
        'files': {p.name: sha(p) for p in cache.iterdir() if p.suffix in ['.npz', '.json']},
        'runtime': runtime})
    gallery = np.stack([lookup[(r['path'], 'clean')][0] for r in gallery_rows])
    return lookup, gallery, [r['identity'] for r in gallery_rows]


def evaluate(dest, cfg, tracks, lookup, gallery, ids):
    predictions, summaries, calibration_rows, pairwise, intervals = [], [], [], [], []
    vectors, score_arrays, truth_arrays, keys = [], [], [], []
    groups = [(c, 'none') for c in cfg['conditions']] + [('mixed', c) for c in cfg['contamination_conditions']]
    for condition, contamination in groups:
        for method in cfg['methods']:
            rows, scores_all, truth_indices = [], [], []
            for t in tracks:
                levels = t['mixed_levels'] if condition == 'mixed' else [condition] * cfg['track_size']
                paths = t['paths'].copy()
                levels = levels.copy()
                if contamination != 'none' and t['truth'] == 'known':
                    paths[-1] = t['intruder_path']
                    if contamination == 'mild':
                        levels[-1] = 'mild'
                items = [lookup[(p, level)] for p, level in zip(paths, levels)]
                norms = np.array([x[1] for x in items])
                vector = aggregate([x[0] for x in items], norms, [x[2] for x in items],
                                   method, t['random_index'])
                scores = gallery @ vector
                order = np.argsort(-scores, kind='stable')
                quality = float(np.clip((norms - cfg['norm_bounds'][0]) /
                                        (cfg['norm_bounds'][1] - cfg['norm_bounds'][0]), 0, 1).mean())
                truth_index = ids.index(t['identity']) if t['truth'] == 'known' else -1
                row = {'condition': condition, 'contamination': contamination, 'method': method,
                       'track_id': t['track_id'], 'role': t['role'], 'sequence': t['sequence'],
                       'identity': t['identity'], 'truth': t['truth'], 'top1_id': ids[order[0]],
                       'top1_score': float(scores[order[0]]), 'top2_score': float(scores[order[1]]),
                       'true_score': float(scores[truth_index]) if truth_index >= 0 else '',
                       'quality': quality, 'correct': ids[order[0]] == t['identity']}
                rows.append(row)
                scores_all.append(scores)
                truth_indices.append(truth_index)
                vectors.append(vector)
            scores_all, truth_indices = np.stack(scores_all), np.array(truth_indices)
            eval_mask = np.array([r['role'] == 'evaluation' for r in rows])
            pairwise.append({'condition': condition, 'contamination': contamination, 'method': method,
                             **verification_metrics(scores_all[eval_mask], truth_indices[eval_mask])})
            score_arrays.append(scores_all)
            truth_arrays.append(truth_indices)
            keys.append([condition, contamination, method])
            for rule in ['fixed', 'aosm']:
                cal = [r for r in rows if r['role'] == 'calibration' and r['truth'] == 'unknown']
                threshold = calibrate([r['top1_score'] for r in cal], [r['quality'] for r in cal], rule,
                                      cfg['target_fpir'], cfg['aosm_margin'], cfg['aosm_floor'])
                for r in rows:
                    r[rule + '_accepted'] = bool(accept(r['top1_score'], r['quality'], threshold, rule,
                                                       cfg['aosm_margin'], cfg['aosm_floor']))
                cfa = sum(r[rule + '_accepted'] for r in cal)
                calibration_rows.append({'condition': condition, 'contamination': contamination,
                    'method': method, 'rule': rule, 'threshold': threshold,
                    'unknown_queries': len(cal), 'false_accepts': cfa, 'observed_fpir': cfa / len(cal)})
                for role in ['calibration', 'evaluation']:
                    known = [r for r in rows if r['role'] == role and r['truth'] == 'known']
                    unknown = [r for r in rows if r['role'] == role and r['truth'] == 'unknown']
                    success = [r['correct'] and r[rule + '_accepted'] for r in known]
                    false_accepts = [r[rule + '_accepted'] for r in unknown]
                    ci = paired_identity_interval(success, [r['identity'] for r in known],
                                                   cfg['bootstrap_resamples'], cfg['seed'])
                    summaries.append({'condition': condition, 'contamination': contamination,
                        'method': method, 'rule': rule, 'role': role, 'threshold': threshold,
                        'known_tracks': len(known), 'unknown_tracks': len(unknown),
                        'rank1_correct': sum(r['correct'] for r in known),
                        'rank1': np.mean([r['correct'] for r in known]).item(),
                        'correct_known_accepted': sum(success), 'dir': float(np.mean(success)),
                        'false_known_accepted': sum(false_accepts), 'fpir': float(np.mean(false_accepts)),
                        'dir_identity_bootstrap_low': ci[0], 'dir_identity_bootstrap_high': ci[1]})
            predictions.extend(rows)
            print('Scored', condition, contamination, method, flush=True)
    by_key = {(r['condition'], r['contamination'], r['method'], r['track_id']): r for r in predictions}
    for condition, contamination in groups:
        base = [r for r in predictions if r['condition'] == condition and r['contamination'] == contamination
                and r['method'] == 'first' and r['role'] == 'evaluation' and r['truth'] == 'known']
        for reference, other in [('first', 'mean'), ('mean', 'best_norm'), ('mean', 'softmax_norm_1')]:
            a = [by_key[(condition, contamination, reference, r['track_id'])] for r in base]
            b = [by_key[(condition, contamination, other, r['track_id'])] for r in base]
            for metric in ['rank1', 'fixed_dir']:
                def success(r):
                    return int(r['correct'] and (metric == 'rank1' or r['fixed_accepted']))
                delta = [success(y) - success(x) for x, y in zip(a, b)]
                ci = paired_identity_interval(delta, [r['identity'] for r in base],
                                               cfg['bootstrap_resamples'], cfg['seed'])
                intervals.append({'condition': condition, 'contamination': contamination, 'reference': reference,
                    'comparison': other, 'metric': metric, 'known_tracks': len(base),
                    'delta': float(np.mean(delta)), 'ci_low': ci[0], 'ci_high': ci[1],
                    'helped': sum(d > 0 for d in delta), 'harmed': sum(d < 0 for d in delta)})
    write_csv(dest / 'predictions.csv', predictions)
    write_csv(dest / 'summary.csv', summaries)
    write_csv(dest / 'calibration.csv', calibration_rows)
    write_csv(dest / 'pairwise_verification.csv', pairwise)
    write_csv(dest / 'paired_intervals.csv', intervals)
    cache = ROOT / 'data/embeddings/chokepoint' / dest.name
    np.savez_compressed(cache / 'query_scores.npz', scores=np.stack(score_arrays),
                        truths=np.stack(truth_arrays), vectors=np.stack(vectors).reshape(len(keys),len(tracks),512),
                        gallery=gallery)
    save_json(cache / 'query_keys.json', {'keys': keys, 'gallery_ids': ids})
    quality_rows = []
    for level in LEVELS:
        ns, laps, sims, labels = [], [], [], []
        # All selected known frames once, not once per overlapping analysis.
        seen = set()
        for t in tracks:
            if t['truth'] != 'known' or t['role'] != 'evaluation':
                continue
            for path in t['paths']:
                if path in seen:
                    continue
                seen.add(path)
                feat, norm, lap = lookup[(path, level)]
                ns.append(norm)
                laps.append(lap)
                sims.append(float(gallery[ids.index(t['identity'])] @ feat))
                labels.append(t['identity'])
        norm_corr = spearmanr(ns, sims)
        lap_corr = spearmanr(laps, sims)
        quality_rows.append({'level': level, 'frames': len(ns), 'identities': len(set(labels)),
            'mean_norm': float(np.mean(ns)), 'mean_laplacian': float(np.mean(laps)),
            'norm_spearman_r': float(norm_corr.statistic), 'laplacian_spearman_r': float(lap_corr.statistic),
            'norm_nominal_p': float(norm_corr.pvalue), 'laplacian_nominal_p': float(lap_corr.pvalue)})
    write_csv(dest / 'quality_diagnostics.csv', quality_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--config', default='configs/chokepoint_temporal.json')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(ROOT / args.config)
        return
    if args.run is None:
        parser.error('Use --prepare, then --run results/chokepoint/<prepared-run>')
    dest = (ROOT / args.run).resolve()
    if not dest.is_relative_to(ROOT / 'results/chokepoint'):
        raise ValueError('Run must be in results/chokepoint')
    if (dest / 'completion.json').exists():
        raise ValueError('Completed runs are immutable; prepare another run')
    prep = json.loads((dest / 'preparation.json').read_text())
    for name, digest in prep['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Prepared implementation changed: ' + name)
    for name, digest in prep['manifest_sha256'].items():
        if sha(dest / name) != digest:
            raise ValueError('Prepared manifest changed: ' + name)
    if sha(dest / 'config.json') != prep['config_sha256']:
        raise ValueError('Prepared protocol changed')
    cfg = json.loads((dest / 'config.json').read_text())
    tracks = json.loads((dest / 'tracks.json').read_text())
    start = time.perf_counter()
    lookup, gallery, ids = extract(dest, cfg, tracks)
    evaluate(dest, cfg, tracks, lookup, gallery, ids)
    save_json(dest / 'completion.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds': time.perf_counter()-start, 'independently_verified': False,
        'scope': cfg['scope'], 'output_sha256': {p.name: sha(p) for p in dest.iterdir() if p.is_file()}})
    print('Completed; independent verification still required:', dest, flush=True)


if __name__ == '__main__':
    main()
