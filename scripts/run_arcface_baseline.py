"""Run a reproducible real-model development baseline, leaving test IDs untouched.

Original crops -> SCRFD -> 5-point alignment -> ArcFace -> exact cosine search.
Failed detections are recorded, never replaced with embeddings from fake models.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.alignment import FaceAligner
from src.detection import SCRFDDetector
from src.quality import FaceQualityEstimator
from src.recognition import FaceEmbedder


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def partition(identities, eligible, rng, enrolled_fraction):
    count = round(len(identities) * enrolled_fraction)
    enrolled = [subject for subject in identities if subject in eligible][:count]
    if len(enrolled) != count:
        raise ValueError('Not enough identities with documented clean frontal enrollment images')
    return {'known': enrolled, 'unknown': [s for s in identities if s not in enrolled]}


def frozen_split(config, hf_rows, icfd_rows):
    path = ROOT / config['split_file']
    fingerprint = {label: sha256(ROOT / config[key]) for label, key in
                   [('halfface', 'halfface_manifest'), ('icfd', 'icfd_manifest')]}
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        if result['manifest_sha256'] != fingerprint:
            raise ValueError('Source manifest changed after the split was frozen; create a new protocol version')
        if result['seed'] != config['seed'] or result['protocol_version'] != config['protocol_version']:
            raise ValueError('Split configuration changed; use a new split file')
        return result
    rng = np.random.default_rng(config['seed'])
    identities = sorted({r['subject_id'] for r in hf_rows})
    rng.shuffle(identities)
    cut = round(len(identities) * config['development_fraction'])
    eligible = {r['subject_id'] for r in hf_rows if r['mask_status'] == 'unmasked' and r['pose'] == 'front'}
    development = partition(identities[:cut], eligible, rng, config['enrolled_fraction'])
    test = partition(identities[cut:], eligible, rng, config['enrolled_fraction'])
    sample_ids = sorted({r['subject_id'] for r in icfd_rows})
    rng.shuffle(sample_ids)
    result = {
        'protocol_version': config['protocol_version'], 'seed': config['seed'],
        'manifest_sha256': fingerprint, 'halfface': {'development': development, 'test_reserved': test},
        'icfd_sample': {'engineering_development': sample_ids[:len(sample_ids) // 2],
                        'reserved': sample_ids[len(sample_ids) // 2:]},
        'note': 'Frozen before first recognition run. ICFD sample is engineering evidence only. No video/session-disjoint guarantee exists for this sample.'
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def select_rows(hf_rows, icfd_rows, split):
    dev = split['halfface']['development']
    selected = []
    known_ids = set(dev['known'])
    unknown_ids = set(dev['unknown'])
    gallery = {}
    for row in sorted(hf_rows, key=lambda r: r['path']):
        if row['subject_id'] in known_ids and row['mask_status'] == 'unmasked' and row['pose'] == 'front':
            gallery.setdefault(row['subject_id'], row['path'])
    if set(gallery) != known_ids:
        raise ValueError('Every known identity must have an original documented enrollment image')
    for source in hf_rows:
        subject = source['subject_id']
        if subject not in known_ids | unknown_ids:
            continue
        selected.append({**source, 'evaluation_role': 'gallery' if gallery.get(subject) == source['path'] else 'probe',
                         'truth_status': 'known' if subject in known_ids else 'unknown'})
    sample_ids = set(split['icfd_sample']['engineering_development'])
    icfd_gallery = {}
    for row in sorted(icfd_rows, key=lambda r: r['path']):
        if row['subject_id'] in sample_ids and row['source_role'] == 'gallery':
            icfd_gallery.setdefault(row['subject_id'], row['path'])
    for source in icfd_rows:
        if source['subject_id'] not in sample_ids:
            continue
        if source['source_role'] == 'gallery' and icfd_gallery[source['subject_id']] != source['path']:
            continue
        selected.append({**source, 'mask_status': 'unspecified', 'pose': 'unspecified',
                         'evaluation_role': source['source_role'], 'truth_status': 'known'})
    return selected


def extract(rows, config, destination):
    detector = SCRFDDetector(confidence_threshold=config['detector_confidence'],
        input_size=tuple(config['detector_input_size']), providers=config['providers'], cpu_threads=config['cpu_threads'])
    embedder = FaceEmbedder(model_name=config['backbone'], providers=config['providers'], cpu_threads=config['cpu_threads'])
    if embedder.mock:
        raise RuntimeError('Mock models are prohibited in research runs')
    aligner, quality = FaceAligner(), FaceQualityEstimator()
    features, records = [], []
    # Warm-up is excluded from per-image inference timings.
    detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    embedder.extract_embedding(np.zeros((112, 112, 3), dtype=np.uint8))
    for index, row in enumerate(rows):
        record = {key: row[key] for key in ['dataset', 'subject_id', 'path', 'mask_status', 'pose', 'evaluation_role', 'truth_status']}
        record.update({'status': '', 'embedding_index': -1, 'face_count': 0, 'native_width': int(row['width']),
                       'native_height': int(row['height']), 'det_score': '', 'quality': '',
                       'detector_ms': 0.0, 'recognizer_ms': 0.0})
        path = ROOT / row['path']
        if sha256(path) != row['sha256']:
            raise ValueError(f'Original image has changed since inventory: {path}')
        image = cv2.imread(str(path))
        if image is None:
            record['status'] = 'unreadable_image'
        else:
            pad_y = round(image.shape[0] * config.get('crop_padding_fraction', 0))
            pad_x = round(image.shape[1] * config.get('crop_padding_fraction', 0))
            value = config.get('crop_padding_value', 127)
            detector_image = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x,
                                               cv2.BORDER_CONSTANT, value=(value, value, value))
            start = time.perf_counter()
            faces = detector.detect(detector_image)
            record['detector_ms'] = (time.perf_counter() - start) * 1000
            record['face_count'] = len(faces)
            if len(faces) != 1:
                record['status'] = 'no_face' if not faces else 'multiple_faces'
            elif faces[0]['kps'] is None or not np.isfinite(faces[0]['kps']).all():
                record['status'] = 'invalid_landmarks'
            else:
                face = faces[0]
                crop = aligner.align(detector_image, face['kps'])
                native_bbox = [max(0, face['bbox'][0] - pad_x), max(0, face['bbox'][1] - pad_y),
                               min(image.shape[1], face['bbox'][2] - pad_x), min(image.shape[0], face['bbox'][3] - pad_y)]
                start = time.perf_counter()
                feature = embedder.extract_embedding(crop)
                record['recognizer_ms'] = (time.perf_counter() - start) * 1000
                record.update({'status': 'ok', 'embedding_index': len(features), 'det_score': face['score'],
                               'quality': quality.estimate_quality(crop, bbox=native_bbox, landmarks=face['kps'])['overall_quality']})
                features.append(feature)
        records.append(record)
        if (index + 1) % 50 == 0 or index == 0 or index + 1 == len(rows):
            print(f'Processed {index + 1}/{len(rows)}; successful embeddings: {len(features)}', flush=True)
            write_csv(destination / 'extraction.csv', records)
    matrix = np.stack(features) if features else np.empty((0, 512), dtype=np.float32)
    cache_dir = ROOT / 'data' / 'embeddings' / destination.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / 'arcface_embeddings.npz', embeddings=matrix)
    write_csv(destination / 'extraction.csv', records)
    model_info = {
        'recognizer': embedder.model_name, 'recognizer_filename': embedder.model_path.name,
        'recognizer_sha256': sha256(embedder.model_path), 'detector_filename': detector.model_path.name,
        'detector_sha256': sha256(detector.model_path), 'providers': embedder.model.session.get_providers(),
        'preprocessing': f"Fixed neutral border of {config.get('crop_padding_fraction', 0)} times each source dimension; SCRFD; exactly one face required; five-point norm_crop 112x112; InsightFace BGR input get_feat (swapRB in model wrapper); L2 normalization",
        'mock': False, 'embedding_cache': (cache_dir / 'arcface_embeddings.npz').relative_to(ROOT).as_posix()
    }
    return records, matrix, model_info


def choose_threshold(unknown_scores, target):
    scores = np.asarray(unknown_scores, dtype=np.float64)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite development non-mated scores are required for threshold calibration')
    allowed = math.floor(target * len(scores))
    if allowed >= len(scores):
        raise ValueError('FPIR target must be below 1')
    # A strict boundary handles ties and the score >= threshold acceptance rule.
    return float(np.nextafter(np.sort(scores)[::-1][allowed], np.inf))


def evaluate(records, matrix, config, destination):
    predictions, summaries = [], []
    threshold = None
    for dataset in ['halfface', 'icfd_sample']:
        subset = [r for r in records if r['dataset'] == dataset]
        gallery = [r for r in subset if r['evaluation_role'] == 'gallery' and r['status'] == 'ok']
        probes = [r for r in subset if r['evaluation_role'] == 'probe']
        if not gallery:
            raise RuntimeError(f'No valid gallery templates for {dataset}')
        gallery_ids = [r['subject_id'] for r in gallery]
        gallery_matrix = np.stack([matrix[r['embedding_index']] for r in gallery])
        dataset_predictions = []
        for row in probes:
            result = {key: row[key] for key in ['dataset', 'subject_id', 'path', 'mask_status', 'pose', 'truth_status', 'status', 'quality']}
            result.update({'top1_id': '', 'top1_score': '', 'top2_score': '', 'margin': '', 'rank1_correct': False,
                           'has_gallery_mate': row['subject_id'] in gallery_ids, 'decision': '', 'threshold': '', 'correct_known_accept': False})
            if row['status'] == 'ok':
                scores = gallery_matrix @ matrix[row['embedding_index']]
                ranking = np.argsort(scores)[::-1]
                result.update({'top1_id': gallery_ids[int(ranking[0])], 'top1_score': float(scores[ranking[0]]),
                               'top2_score': float(scores[ranking[1]]) if len(ranking) > 1 else -1.0})
                result['margin'] = result['top1_score'] - result['top2_score']
                result['rank1_correct'] = row['truth_status'] == 'known' and result['top1_id'] == row['subject_id']
            dataset_predictions.append(result)
        if dataset == 'halfface':
            unknown_scores = [r['top1_score'] for r in dataset_predictions if r['truth_status'] == 'unknown' and r['status'] == 'ok']
            threshold = choose_threshold(unknown_scores, config['calibration_fpir'])
            for row in dataset_predictions:
                row['threshold'] = threshold
                row['decision'] = ('KNOWN' if row['top1_score'] >= threshold else 'UNKNOWN') if row['status'] == 'ok' else row['status'].upper()
                row['correct_known_accept'] = row['decision'] == 'KNOWN' and row['rank1_correct']
        else:
            for row in dataset_predictions:
                row['decision'] = 'CLOSED_SET_ONLY' if row['status'] == 'ok' else row['status'].upper()
        known = [r for r in dataset_predictions if r['truth_status'] == 'known']
        unknown = [r for r in dataset_predictions if r['truth_status'] == 'unknown']
        embedded_known = [r for r in known if r['status'] == 'ok' and r['has_gallery_mate']]
        embedded_unknown = [r for r in unknown if r['status'] == 'ok']
        ratio = lambda numerator, denominator: numerator / denominator if denominator else None
        summary = {
            'dataset': dataset, 'evaluation_scope': 'development_only' if dataset == 'halfface' else 'engineering_sample_only',
            'gallery_attempts': sum(r['evaluation_role'] == 'gallery' for r in subset), 'gallery_successes': len(gallery),
            'known_probe_attempts': len(known), 'known_with_embedding_and_gallery': len(embedded_known),
            'unknown_probe_attempts': len(unknown), 'unknown_with_embedding': len(embedded_unknown),
            'rank1_conditional': ratio(sum(r['rank1_correct'] for r in embedded_known), len(embedded_known)),
            'rank1_end_to_end': ratio(sum(r['rank1_correct'] for r in known), len(known)),
            'correct_known_accept_end_to_end': ratio(sum(r['correct_known_accept'] for r in known), len(known)) if dataset == 'halfface' else None,
            'fpir_conditional': ratio(sum(r['decision'] == 'KNOWN' for r in embedded_unknown), len(embedded_unknown)),
            'fpir_end_to_end': ratio(sum(r['decision'] == 'KNOWN' for r in unknown), len(unknown)),
            'threshold': threshold if dataset == 'halfface' else None,
            'extraction_status_counts': dict(Counter(r['status'] for r in subset)),
        }
        summaries.append(summary)
        predictions.extend(dataset_predictions)
    write_csv(destination / 'predictions.csv', predictions)
    (destination / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    (destination / 'threshold.json').write_text(json.dumps({
        'scope': 'halfface_development_only', 'threshold': threshold,
        'target_conditional_fpir': config['calibration_fpir'], 'acceptance_rule': 'score >= threshold',
        'note': 'Empirical calibration on development non-mated searches; this is not held-out performance or a statistical guarantee.'
    }, indent=2), encoding='utf-8')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/arcface_baseline.yaml')
    parser.add_argument('--limit', type=int, help='Engineering smoke run only; no metrics or calibration')
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    hf_rows = read_csv(ROOT / config['halfface_manifest'])
    icfd_rows = read_csv(ROOT / config['icfd_manifest'])
    if len({r['sha256'] for r in hf_rows}) != len(hf_rows):
        raise ValueError('Duplicate images must be excluded before the experiment')
    split = frozen_split(config, hf_rows, icfd_rows)
    dev_ids = set(split['halfface']['development']['known'] + split['halfface']['development']['unknown'])
    test_ids = set(split['halfface']['test_reserved']['known'] + split['halfface']['test_reserved']['unknown'])
    if dev_ids & test_ids:
        raise ValueError('Development/test identity leakage')
    rows = select_rows(hf_rows, icfd_rows, split)
    if any(r['subject_id'] in test_ids for r in rows):
        raise ValueError('Reserved test identity entered development extraction')
    if args.limit:
        rows = rows[:args.limit]
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_id = f'arcface_{"smoke" if args.limit else "development"}_{timestamp}'
    destination = ROOT / 'results' / 'baselines' / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (destination / 'split_snapshot.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    print('RUN', destination, 'IMAGES', len(rows), flush=True)
    records, features, model_info = extract(rows, config, destination)
    summaries = evaluate(records, features, config, destination) if not args.limit else []
    git_hash = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    info = {
        'run_id': run_id, 'git_commit': git_hash, 'source_script_sha256': sha256(__file__),
        'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__, 'onnxruntime': ort.__version__,
        'model': model_info, 'images_processed': len(records), 'successful_embeddings': len(features),
        'final_test_evaluated': False, 'summaries': summaries,
        'limitations': ['Development results only; not final test evidence.',
                        'ICFD uses only the public sample, not the full dataset.',
                        'Exact duplicates were filtered; near-duplicate and pretraining overlap are not ruled out.',
                        'No adaptation, quality gate, augmentation or temporal aggregation is applied in this baseline.']
    }
    (destination / 'run_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
