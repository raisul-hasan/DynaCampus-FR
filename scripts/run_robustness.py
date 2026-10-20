"""Fixed-setting synthetic stress tests on development probes only.

Transforms are applied in memory before detection. Original files, gallery
templates and calibrated thresholds remain unchanged. No final-test access.
"""
import argparse
import hashlib
import json
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
from scripts.compare_decisions import apply_rule, summarize
from scripts.compare_enrollment import gallery_candidates, paired_bootstrap, reject_overlap, unit_mean
from scripts.run_arcface_baseline import read_csv, sha256, write_csv
from src.alignment import FaceAligner
from src.detection import SCRFDDetector
from src.quality import FaceQualityEstimator
from src.recognition import FaceEmbedder


def degrade(image, condition):
    """Return an independent BGR uint8 image; never overwrite the source."""
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Expected a decoded BGR uint8 image')
    kind, value = condition['type'], condition['value']
    if kind == 'original':
        return image.copy()
    if kind == 'blur':
        if not 0 < value < 1:
            raise ValueError('Blur sigma fraction must be in (0,1)')
        sigma = min(image.shape[:2]) * value
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
    if kind == 'resolution':
        if int(value) != value or value < 2:
            raise ValueError('Resolution long edge must be an integer >= 2')
        scale = min(1., value / max(image.shape[:2]))
        size = (max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale)))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if kind == 'jpeg':
        if int(value) != value or not 1 <= value <= 100:
            raise ValueError('JPEG quality must be an integer in [1,100]')
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(value)])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError('JPEG decoding failed')
        return decoded
    raise ValueError(f'Unsupported degradation: {kind}')


def score_records(records, features, gallery_ids, prototypes):
    rows = []
    for record in records:
        row = {k: record[k] for k in ['dataset', 'subject_id', 'path', 'truth_status', 'mask_status', 'status', 'quality']}
        row.update(top1_id='', top1_score='', top2_score='', rank1_correct=False)
        if record['status'] == 'ok':
            scores = prototypes @ features[int(record['embedding_index'])]
            order = np.argsort(scores)[::-1]
            row.update(top1_id=gallery_ids[int(order[0])], top1_score=float(scores[order[0]]), top2_score=float(scores[order[1]]))
            row['rank1_correct'] = row['truth_status'] == 'known' and row['top1_id'] == row['subject_id']
        rows.append(row)
    return rows


def summarize_stress(predictions, thresholds, family, condition):
    summary = summarize(predictions, thresholds, family)
    known = [r for r in predictions if r['truth_status'] == 'known']
    valid = [r for r in predictions if r['status'] == 'ok']
    summary.update(condition=condition['name'], degradation=condition['type'], severity=condition['value'],
                   rank1_correct=sum(r['rank1_correct'] for r in known),
                   rank1_end_to_end=sum(r['rank1_correct'] for r in known)/len(known),
                   extraction_successes=len(valid), extraction_success_rate=len(valid)/len(predictions),
                   false_known_end_to_end=summary['false_known_unknowns']/summary['unknown_probe_attempts'],
                   mean_quality=float(np.mean([float(r['quality']) for r in valid])) if valid else None,
                   extraction_counts=dict(Counter(r['status'] for r in predictions)))
    return summary


def extract_condition(rows, condition, config, detector, embedder, aligner, quality, destination, cache):
    records, features = [], []
    for index, source in enumerate(rows):
        record = {k: source[k] for k in ['dataset', 'subject_id', 'path', 'mask_status', 'truth_status']}
        record.update(condition=condition['name'], source_sha256=source['sha256'], status='', embedding_index=-1,
                      face_count=0, quality='', transformed_width=0, transformed_height=0,
                      transformed_pixel_sha256='', detector_ms=0., recognizer_ms=0.)
        path = ROOT / source['path']
        if sha256(path) != source['sha256']:
            raise ValueError(f'Source image changed: {source["path"]}')
        original = cv2.imread(str(path))
        if original is None:
            raise ValueError('Inventoried source cannot be decoded')
        image = degrade(original, condition)
        record.update(transformed_width=image.shape[1], transformed_height=image.shape[0],
                      transformed_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest())
        py, px = (round(dim*config['crop_padding_fraction']) for dim in image.shape[:2])
        v = config['crop_padding_value']
        padded = cv2.copyMakeBorder(image, py, py, px, px, cv2.BORDER_CONSTANT, value=(v, v, v))
        start = time.perf_counter()
        faces = detector.detect(padded)
        record.update(detector_ms=(time.perf_counter()-start)*1000, face_count=len(faces))
        if len(faces) != 1:
            record['status'] = 'no_face' if not faces else 'multiple_faces'
        elif faces[0]['kps'] is None or not np.isfinite(faces[0]['kps']).all():
            record['status'] = 'invalid_landmarks'
        else:
            face = faces[0]
            crop = aligner.align(padded, face['kps'])
            bbox = [max(0, face['bbox'][0]-px), max(0, face['bbox'][1]-py),
                    min(image.shape[1], face['bbox'][2]-px), min(image.shape[0], face['bbox'][3]-py)]
            start = time.perf_counter()
            feature = embedder.extract_embedding(crop)
            record.update(status='ok', embedding_index=len(features), recognizer_ms=(time.perf_counter()-start)*1000,
                          quality=quality.estimate_quality(crop, bbox=bbox, landmarks=face['kps'])['overall_quality'])
            features.append(feature)
        records.append(record)
        if (index+1) % 50 == 0 or index+1 == len(rows):
            print(f'{condition["name"]}: {index+1}/{len(rows)} probes, {len(features)} extracted', flush=True)
            write_csv(destination / f'{condition["name"]}_extraction.csv', records)
    matrix = np.stack(features) if features else np.empty((0, 512), dtype=np.float32)
    np.savez_compressed(cache / f'{condition["name"]}.npz', embeddings=matrix)
    return records, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/robustness_development.yaml')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    if config['final_test_evaluation'] or config['thresholds_retuned'] or config['gallery_changed'] or config['degradations_combined']:
        raise ValueError('Only isolated, fixed-setting development stress tests are supported')
    decisions = ROOT / config['decision_run']
    decision_info = json.loads((decisions / 'comparison.json').read_text())
    if decision_info['final_test_evaluated'] or sha256(ROOT/'src/open_set/decision.py') != decision_info['engine_sha256']:
        raise ValueError('Decision engine or development scope changed')
    enrollment = ROOT / decision_info['configuration']['enrollment_run']
    enrollment_info = json.loads((enrollment / 'comparison.json').read_text())
    baseline = ROOT / enrollment_info['baseline_run']
    baseline_config = yaml.safe_load((baseline / 'config.yaml').read_text())
    fingerprints = json.loads((baseline / 'source_fingerprints.json').read_text())
    for path, digest in fingerprints.items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f'Baseline implementation changed: {path}')
    if sha256(enrollment/'combined_extraction.csv') != decision_info['source_extraction_sha256']:
        raise ValueError('Enrollment extraction changed')
    records = read_csv(enrollment / 'combined_extraction.csv')
    features = np.load(ROOT / enrollment_info['feature_cache'], allow_pickle=False)['embeddings']
    split = json.loads((decisions / 'split_snapshot.json').read_text())
    known_ids = split['halfface']['development']['known']
    reserved = set(sum(split['halfface']['test_reserved'].values(), []))
    inputs = read_csv(decisions / 'input_scores.csv')
    paths = {r['path'] for r in inputs}
    if len(paths) != len(inputs) or reserved & {r['subject_id'] for r in inputs}:
        raise ValueError('Duplicate probes or reserved identity leakage')
    manifest_path = ROOT / baseline_config['halfface_manifest']
    if sha256(manifest_path) != split['manifest_sha256']['halfface']:
        raise ValueError('Original manifest changed')
    manifest = {r['path']: r for r in read_csv(manifest_path)}
    source_rows = [{**manifest[r['path']], 'truth_status': r['truth_status']} for r in inputs]
    candidates = gallery_candidates(records, 'halfface', known_ids, 'up_to_three_photos')
    gallery_ids = list(candidates)
    gallery_rows = [r for selected in candidates.values() for r in selected]
    reject_overlap(gallery_rows, source_rows)
    prototypes = np.stack([unit_mean([features[int(r['embedding_index'])] for r in candidates[s] if r['status'] == 'ok']) for s in gallery_ids])
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = ROOT / 'results/robustness' / f'robustness_development_{timestamp}'
    destination.mkdir(parents=True)
    cache = ROOT / 'data/embeddings' / destination.name
    cache.mkdir(parents=True)
    np.savez_compressed(cache / 'gallery.npz', embeddings=prototypes, identities=np.array(gallery_ids))
    (destination / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (destination / 'split_snapshot.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    (destination / 'preprocessing.yaml').write_text(yaml.safe_dump(baseline_config), encoding='utf-8')
    write_csv(destination / 'source_manifest.csv', source_rows)
    write_csv(destination / 'gallery_sources.csv', gallery_rows)
    detector = SCRFDDetector(confidence_threshold=baseline_config['detector_confidence'],
        input_size=tuple(baseline_config['detector_input_size']), providers=baseline_config['providers'], cpu_threads=baseline_config['cpu_threads'])
    embedder = FaceEmbedder(model_name=baseline_config['backbone'], providers=baseline_config['providers'], cpu_threads=baseline_config['cpu_threads'])
    if embedder.mock or any(sha256(path) != enrollment_info['model'][key] for path, key in
                           [(embedder.model_path, 'recognizer_sha256'), (detector.model_path, 'detector_sha256')]):
        raise ValueError('Model does not match the real-model enrollment cache')
    aligner, quality = FaceAligner(), FaceQualityEstimator()
    detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
    embedder.extract_embedding(np.zeros((112, 112, 3), dtype=np.uint8))
    print('RUN', destination, 'PROBES', len(inputs), 'DEGRADED CONDITIONS', len(config['conditions'])-1, flush=True)
    summaries, predictions, effects = [], [], []
    clean_by_family = {}
    existing = {r['path']: r for r in records}
    for condition in config['conditions']:
        if condition['type'] == 'original':
            extraction = [existing[r['path']] for r in inputs]
            selected_features = np.stack([features[int(r['embedding_index'])] for r in extraction if r['status'] == 'ok'])
            offset = 0
            extraction = [dict(r) for r in extraction]
            for r in extraction:
                if r['status'] == 'ok':
                    r['embedding_index'] = offset
                    offset += 1
            np.savez_compressed(cache / 'original.npz', embeddings=selected_features)
            write_csv(destination / 'original_extraction.csv', extraction)
        else:
            extraction, selected_features = extract_condition(source_rows, condition, baseline_config, detector, embedder, aligner, quality, destination, cache)
        scores = score_records(extraction, selected_features, gallery_ids, prototypes)
        if condition['type'] == 'original':
            for row, old in zip(scores, inputs):
                assert row['status'] == old['status']
                if row['status'] == 'ok':
                    assert row['top1_id'] == old['top1_id']
                    assert abs(row['top1_score']-float(old['top1_score'])) < 1e-7
        for family in config['families']:
            thresholds = decision_info['selected_thresholds'][family]
            output = apply_rule(scores, thresholds, family)
            summary = summarize_stress(output, thresholds, family, condition)
            for row in output:
                row.update(condition=condition['name'])
            if condition['type'] == 'original':
                expected = next(r for r in decision_info['summaries'] if r['family'] == family)
                assert summary['correct_known_accepts'] == expected['correct_known_accepts']
                assert summary['decision_counts'] == expected['decision_counts']
                clean_by_family[family] = output
            else:
                effects.append({'condition': condition['name'], 'family': family, **paired_bootstrap(
                    clean_by_family[family], output, known_ids, config['bootstrap_samples'], config['seed'], 'correct_known_accept')})
            summaries.append(summary)
            predictions.extend(output)
            print(condition['name'], family, 'known', summary['correct_known_accepts'], '/', summary['known_probe_attempts'],
                  'false known', summary['false_known_unknowns'], '/', summary['unknown_with_embedding'], flush=True)
        write_csv(destination / 'predictions.csv', predictions)
        write_csv(destination / 'summary.csv', [{k:v for k,v in s.items() if not isinstance(v, dict)} for s in summaries])
    write_csv(destination / 'paired_effects.csv', effects)
    source_files = list(fingerprints) + ['scripts/run_robustness.py', 'src/open_set/decision.py',
                   'scripts/compare_decisions.py', 'scripts/compare_enrollment.py', args.config]
    result = dict(run_directory=destination.relative_to(ROOT).as_posix(), configuration=config, summaries=summaries,
                  selected_thresholds={f: decision_info['selected_thresholds'][f] for f in config['families']},
                  final_test_evaluated=False, thresholds_retuned=False, gallery_changed=False,
                  clean_control_matches_previous_run=True, model=enrollment_info['model'],
                  feature_cache=cache.relative_to(ROOT).as_posix(),
                  source_sha256={p:sha256(ROOT/p) for p in source_files},
                  input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in
                      [decisions/'comparison.json', decisions/'input_scores.csv', enrollment/'combined_extraction.csv', ROOT/enrollment_info['feature_cache']]},
                  python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__, onnxruntime=ort.__version__,
                  new_degraded_image_attempts=len(inputs)*(len(config['conditions'])-1),
                  git_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True).stdout.strip(),
                  limitations=['Synthetic isolated transformations on development face crops, not measured CCTV conditions.',
                               'Thresholds were selected on original development data; no degradation-specific retuning.',
                               'Derived versions of a source are paired observations, not new independent people.',
                               'Unknown extraction denominators change with degradation; report attempted-probe counts too.',
                               'Original enrollment templates remain fixed. Quality gate stays disabled as selected previously.'])
    (destination/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('COMPLETE', destination, flush=True)


if __name__ == '__main__':
    main()
