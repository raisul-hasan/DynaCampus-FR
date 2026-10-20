"""Paired AdaFace/ArcFace development experiment on identical aligned crops.

One and few original enrollment photos, score and score/margin calibration, raw
predictions, failure cases and CPU component timings. No held-out inference.
"""
import argparse
import ast
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
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_arcface_baseline import read_csv, write_csv, sha256
from scripts.compare_enrollment import gallery_candidates, paired_bootstrap, reject_overlap, unit_mean
from scripts.compare_decisions import calibrate, apply_rule, summarize
from scripts.compare_enrollment_pooling import assert_development
from src.recognition import FaceEmbedder
from src.recognition.adaface import AdaFaceEmbedder, bgr_tensor
from src.detection import SCRFDDetector
from src.alignment import FaceAligner
from src.quality import FaceQualityEstimator

MODELS = ('arcface', 'adaface')


def timed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns() - start) / 1e6


def official_smoke(model, crop, source_path):
    function = next(node for node in ast.parse(source_path.read_text(encoding='utf-8')).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'to_input')
    scope = dict(np=np, torch=torch)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), 'exec'), scope)
    reference = scope['to_input'](crop[:, :, ::-1].copy())
    actual = bgr_tensor(crop)
    if not torch.equal(actual, reference):
        raise ValueError('AdaFace preprocessing differs from pinned official reference')
    with torch.inference_mode():
        expected, _ = model.model(reference.to(model.device))
    one, norm = model.extract_with_norm(crop)
    pair = model.extract_embedding(np.stack([crop, crop]))
    reference_delta = float(np.max(np.abs(one - expected.cpu().numpy()[0])))
    batch_delta = float(np.max(np.abs(pair - one)))
    if reference_delta > 1e-6 or batch_delta > 1e-5:
        raise ValueError('Real AdaFace reference or batch equivalence failed')
    return dict(official_preprocessing_exact=True, reference_feature_max_difference=reference_delta,
                single_batch_max_difference=batch_delta, pre_normalization_norm=norm,
                real_model_loaded=True, strict_checkpoint_loading=True)


def extract(rows, old_features, config, destination):
    torch.set_num_threads(config['cpu_threads'])
    torch.set_num_interop_threads(1)
    detector = SCRFDDetector(confidence_threshold=config['detector_confidence'],
        input_size=tuple(config['detector_input_size']), providers=config['providers'], cpu_threads=config['cpu_threads'])
    arc, arc_load_ms = timed(lambda: FaceEmbedder(model_name='insightface_arcface', providers=config['providers'], cpu_threads=config['cpu_threads']))
    ada, ada_load_ms = timed(lambda: AdaFaceEmbedder(ROOT / config['adaface_model_dir'], config['device']))
    models = dict(arcface=arc, adaface=ada)
    aligner, quality = FaceAligner(), FaceQualityEstimator()
    manifests = {r['path']: r for path in ['data/manifests/halfface_usable.csv', 'data/manifests/icfd_sample.csv'] for r in read_csv(ROOT / path)}
    for _ in range(config['warmup_calls']):
        detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        for model in models.values():
            model.extract_embedding(np.zeros((112, 112, 3), dtype=np.uint8))
    features, records, timings, smoke = {name: [] for name in MODELS}, [], [], None
    max_delta = 0.
    for index, original in enumerate(rows):
        path = ROOT / original['path']
        if sha256(path) != manifests[original['path']]['sha256']:
            raise ValueError(f'Original source changed: {original["path"]}')
        image, decode_ms = timed(lambda: cv2.imread(str(path)))
        record = {key: original[key] for key in ['dataset', 'subject_id', 'path', 'mask_status', 'pose', 'evaluation_role', 'truth_status']}
        record.update(status='', embedding_index=-1, face_count=0, native_width=original['native_width'],
                      native_height=original['native_height'], det_score='', quality='', adaface_norm='',
                      source_sha256=manifests[original['path']]['sha256'], aligned_crop_sha256='', arcface_cache_max_difference='')
        padding_ms = detector_ms = alignment_ms = quality_ms = 0.
        model_times = dict.fromkeys(MODELS, 0.)
        if image is None:
            record['status'] = 'unreadable_image'
        else:
            py, px = [round(dim * config['crop_padding_fraction']) for dim in image.shape[:2]]
            value = config['crop_padding_value']
            padded, padding_ms = timed(lambda: cv2.copyMakeBorder(image, py, py, px, px, cv2.BORDER_CONSTANT, value=(value, value, value)))
            faces, detector_ms = timed(lambda: detector.detect(padded))
            record['face_count'] = len(faces)
            if len(faces) != 1:
                record['status'] = 'no_face' if not faces else 'multiple_faces'
            elif faces[0]['kps'] is None or not np.isfinite(faces[0]['kps']).all():
                record['status'] = 'invalid_landmarks'
            else:
                face = faces[0]
                crop, alignment_ms = timed(lambda: aligner.align(padded, face['kps']))
                bbox = [max(0, face['bbox'][0] - px), max(0, face['bbox'][1] - py),
                        min(image.shape[1], face['bbox'][2] - px), min(image.shape[0], face['bbox'][3] - py)]
                q, quality_ms = timed(lambda: quality.estimate_quality(crop, bbox=bbox, landmarks=face['kps']))
                if smoke is None:
                    smoke = official_smoke(ada, crop, ROOT / config['adaface_model_dir'] / 'upstream/inference.py')
                    smoke.update(source_path=original['path'], aligned_crop_sha256=hashlib.sha256(crop.tobytes()).hexdigest())
                    (destination / 'real_model_smoke.json').write_text(json.dumps(smoke, indent=2), encoding='utf-8')
                output = {}
                # Alternate forward order. This remains a backend/checkpoint comparison,
                # not a controlled loss-function or runtime-engine ablation.
                for name in MODELS if index % 2 == 0 else MODELS[::-1]:
                    if name == 'adaface':
                        (output[name], norm), model_times[name] = timed(lambda: ada.extract_with_norm(crop))
                        record['adaface_norm'] = norm
                    else:
                        output[name], model_times[name] = timed(lambda: arc.extract_embedding(crop))
                delta = float(np.max(np.abs(output['arcface'] - old_features[int(original['embedding_index'])])))
                if delta > 1e-6:
                    raise ValueError(f'ArcFace cache reproduction failed: {original["path"]}: {delta}')
                max_delta = max(max_delta, delta)
                embedding_index = len(features['arcface'])
                if embedding_index != int(original['embedding_index']):
                    raise ValueError('Extraction ordering differs from the original cache')
                for name in MODELS:
                    features[name].append(output[name])
                record.update(status='ok', embedding_index=embedding_index, det_score=face['score'], quality=q['overall_quality'],
                              aligned_crop_sha256=hashlib.sha256(crop.tobytes()).hexdigest(), arcface_cache_max_difference=delta)
        if record['status'] != original['status']:
            raise ValueError(f'Detector outcome changed for {original["path"]}; paired protocol requires review')
        records.append(record)
        for name in MODELS:
            timings.append(dict(dataset=record['dataset'], subject_id=record['subject_id'], path=record['path'], model=name,
                status=record['status'], decode_ms=decode_ms, padding_ms=padding_ms, detector_ms=detector_ms,
                alignment_ms=alignment_ms, quality_ms=quality_ms, recognizer_ms=model_times[name],
                composed_image_to_embedding_ms=decode_ms + padding_ms + detector_ms + alignment_ms + quality_ms + model_times[name]))
        if index % 50 == 0 or index + 1 == len(rows):
            write_csv(destination / 'extraction.csv', records)
            write_csv(destination / 'timings.csv', timings)
            print(f'Paired extraction {index+1}/{len(rows)}; real aligned features {len(features["arcface"])}', flush=True)
    cache = ROOT / 'data/embeddings' / destination.name
    cache.mkdir(parents=True, exist_ok=False)
    matrices = {name: np.stack(vectors) for name, vectors in features.items()}
    np.savez_compressed(cache / 'embeddings.npz', **matrices)
    model_info = dict(
        arcface=dict(checkpoint=str(arc.model_path), sha256=sha256(arc.model_path), backend='ONNX Runtime CPUExecutionProvider',
                     pretraining='WebFace600K as documented for buffalo_l; not matched to AdaFace MS1MV2', load_ms=arc_load_ms),
        adaface=dict(checkpoint=ada.model_path.relative_to(ROOT).as_posix(), sha256=sha256(ada.model_path), backend='PyTorch CPU float32',
                     pretraining='MS1MV2', parameters=ada.parameter_count, load_ms=ada_load_ms, official_provenance=ada.provenance),
        detector=dict(checkpoint=str(detector.model_path), sha256=sha256(detector.model_path)),
        features=(cache / 'embeddings.npz').relative_to(ROOT).as_posix(),
        feature_sha256=sha256(cache / 'embeddings.npz'), arcface_max_cache_difference=max_delta,
        arcface_reproduced_images=len(features['arcface']))
    return records, matrices, model_info, timings


def score_probes(records, features, dataset, known_ids, strategy):
    candidates = gallery_candidates(records, dataset, known_ids, strategy)
    probes = [r for r in records if r['dataset'] == dataset and
              ((r['truth_status'] == 'known' and r['mask_status'] == 'masked') or r['truth_status'] == 'unknown')]
    if dataset == 'icfd_sample':
        probes = [r for r in records if r['dataset'] == dataset and r['evaluation_role'] == 'probe']
    reject_overlap([r for selected in candidates.values() for r in selected], probes)
    vectors, ids, sources = [], [], []
    for sid, selected in candidates.items():
        usable = [r for r in selected if r['status'] == 'ok']
        if not usable:
            raise ValueError('Enrollment failure must not silently remove an identity')
        ids.append(sid)
        vectors.append(unit_mean([features[int(r['embedding_index'])] for r in usable]))
        sources.extend(dict(subject_id=sid, path=r['path'], status=r['status'], used_in_template=r['status'] == 'ok') for r in selected)
    matrix = np.stack(vectors)
    output = []
    for record in probes:
        row = {key: record[key] for key in ['dataset', 'subject_id', 'path', 'truth_status', 'status', 'quality']}
        row.update(top1_id='', top1_score='', top2_score='', rank1_correct=False)
        if row['status'] == 'ok':
            scores = matrix @ features[int(record['embedding_index'])]
            order = np.argsort(scores)[::-1]
            row.update(top1_id=ids[int(order[0])], top1_score=float(scores[order[0]]), top2_score=float(scores[order[1]]))
            row['rank1_correct'] = row['truth_status'] == 'known' and row['top1_id'] == row['subject_id']
        output.append(row)
    return output, sources, len(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/backbone_comparison.yaml')
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if config['final_test_evaluation'] or config['device'] != 'cpu':
        raise ValueError('This protocol requires CPU development inference only')
    source = ROOT / config['enrollment_run']
    info = json.loads((source / 'comparison.json').read_text(encoding='utf-8'))
    if info['final_test_evaluated'] or info['model']['mock']:
        raise ValueError('Expected a real-model development source')
    split = json.loads((source / 'split_snapshot.json').read_text(encoding='utf-8'))
    original = read_csv(source / 'combined_extraction.csv')
    assert_development(original, split)
    if len({r['path'] for r in original}) != len(original):
        raise ValueError('Repeated source image in extraction records')
    baseline_features = np.load(ROOT / info['feature_cache'], allow_pickle=False)['embeddings']
    source_files = [config_path, source / 'comparison.json', source / 'combined_extraction.csv', source / 'predictions.csv',
                    source / 'split_snapshot.json', ROOT / info['feature_cache'], ROOT / 'configs/decision_ablation.yaml',
                    ROOT / 'data/manifests/halfface_usable.csv', ROOT / 'data/manifests/icfd_sample.csv']
    fingerprints = {p.relative_to(ROOT).as_posix(): sha256(p) for p in source_files}
    destination = ROOT / 'results/backbones' / ('backbones_development_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    (destination / 'split_snapshot.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    (destination / 'input_fingerprints.json').write_text(json.dumps(fingerprints, indent=2), encoding='utf-8')
    print(f'RUN {destination.relative_to(ROOT).as_posix()}', flush=True)
    records, matrices, model_info, timings = extract(original, baseline_features, config, destination)
    if model_info['arcface']['sha256'] != info['model']['recognizer_sha256'] or model_info['detector']['sha256'] != info['model']['detector_sha256']:
        raise ValueError('ArcFace/detector checkpoints differ from baseline')
    summaries, predictions, grids, sources, selections, effects = [], [], [], [], [], []
    paired = {}
    for model in MODELS:
        for dataset, ids in [('halfface', split['halfface']['development']['known']),
                             ('icfd_sample', split['icfd_sample']['engineering_development'])]:
            for strategy in config['strategies']:
                scored, gallery_sources, gallery_count = score_probes(records, matrices[model], dataset, ids, strategy)
                sources.extend(dict(model=model, dataset=dataset, strategy=strategy, **r) for r in gallery_sources)
                if dataset == 'halfface':
                    for family in config['decision_families']:
                        selection, grid = calibrate(scored, family, config)
                        selected = dict(model=model, dataset=dataset, strategy=strategy, **selection)
                        selections.append(selected)
                        grids.extend(dict(model=model, dataset=dataset, strategy=strategy, **r) for r in grid)
                        predicted = apply_rule(scored, selection, family)
                        summary = summarize(predicted, selection, family)
                        summary.update(model=model, dataset=dataset, strategy=strategy, gallery_identities=gallery_count,
                            gallery_images=sum(r['used_in_template'] for r in gallery_sources), rank1_correct=sum(r['rank1_correct'] for r in scored))
                        summaries.append(summary)
                        paired[(model, dataset, strategy, family)] = predicted
                        predictions.extend(dict(model=model, strategy=strategy, **r) for r in predicted)
                        print(f'{model} {strategy} {family}: {summary["correct_known_accepts"]}/{summary["known_probe_attempts"]}, false {summary["false_known_unknowns"]}/{summary["unknown_with_embedding"]}', flush=True)
                else:
                    family = 'closed_set_engineering'
                    predicted = [dict(r, family=family, decision='CLOSED_SET_ONLY' if r['status'] == 'ok' else 'CAPTURE_FAILURE',
                                      accepted_id='', correct_known_accept=False, covered=r['status'] == 'ok') for r in scored]
                    paired[(model, dataset, strategy, family)] = predicted
                    predictions.extend(dict(model=model, strategy=strategy, **r) for r in predicted)
                    summaries.append(dict(family=family, known_probe_attempts=len(scored), unknown_probe_attempts=0,
                        known_with_embedding=sum(r['status'] == 'ok' for r in scored), unknown_with_embedding=0,
                        correct_known_accepts=None, correct_known_accept_end_to_end=None, false_known_unknowns=None,
                        fpir_conditional=None, wrong_known_identity_accepts=None, valid_decision_coverage=None,
                        end_to_end_decision_coverage=None, accuracy_among_covered=None, decision_counts=dict(Counter(r['decision'] for r in predicted)),
                        score_threshold=None, margin_threshold=None, quality_threshold=None, model=model, dataset=dataset, strategy=strategy,
                        gallery_identities=gallery_count, gallery_images=sum(r['used_in_template'] for r in gallery_sources), rank1_correct=sum(r['rank1_correct'] for r in scored)))
    for model in MODELS:
        for family in config['decision_families']:
            effects.append(dict(comparison='few_minus_one', model=model, strategy='up_to_three_photos', family=family,
                **paired_bootstrap(paired[(model, 'halfface', 'one_photo', family)], paired[(model, 'halfface', 'up_to_three_photos', family)],
                                   split['halfface']['development']['known'], config['bootstrap_samples'], config['seed'], 'correct_known_accept')))
    for strategy in config['strategies']:
        for family in config['decision_families']:
            effects.append(dict(comparison='adaface_minus_arcface', model='adaface', strategy=strategy, family=family,
                **paired_bootstrap(paired[('arcface', 'halfface', strategy, family)], paired[('adaface', 'halfface', strategy, family)],
                                   split['halfface']['development']['known'], config['bootstrap_samples'], config['seed'], 'correct_known_accept')))
    timing_summary = []
    for model in MODELS:
        for dataset in ('halfface', 'icfd_sample'):
            rows = [r for r in timings if r['model'] == model and r['dataset'] == dataset and r['status'] == 'ok']
            for stage in ['decode_ms', 'padding_ms', 'detector_ms', 'alignment_ms', 'quality_ms', 'recognizer_ms', 'composed_image_to_embedding_ms']:
                values = [r[stage] for r in rows]
                timing_summary.append(dict(model=model, dataset=dataset, stage=stage, samples=len(values),
                    median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95))))
    failures, cases = [], []
    for model in MODELS:
        a = {r['path']: r for r in paired[(model, 'halfface', 'one_photo', 'score_margin')]}
        b = {r['path']: r for r in paired[(model, 'halfface', 'up_to_three_photos', 'score_margin')]}
        if a.keys() != b.keys():
            raise ValueError('Paired policies use different probes')
        for path, row in b.items():
            if row['status'] != 'ok' or (row['truth_status'] == 'known' and not row['correct_known_accept']) or (row['truth_status'] == 'unknown' and row['decision'] == 'KNOWN'):
                category = ('capture_failure' if row['status'] != 'ok' else
                    'unknown_false_accept' if row['truth_status'] == 'unknown' else
                    'wrong_known_identity' if row['decision'] == 'KNOWN' else
                    'ambiguous_known' if row['decision'] == 'AMBIGUOUS' else 'rejected_known')
                failures.append(dict(model=model, category=category, **row))
            first = a[path]
            if first['decision'] != row['decision'] or first['correct_known_accept'] != row['correct_known_accept']:
                cases.append(dict(model=model, path=path, subject_id=row['subject_id'], truth_status=row['truth_status'],
                    one_decision=first['decision'], few_decision=row['decision'], one_correct=first['correct_known_accept'], few_correct=row['correct_known_accept'],
                    one_score=first['top1_score'], few_score=row['top1_score'],
                    effect='helped_known' if row['correct_known_accept'] else 'harmed_known' if first['correct_known_accept'] else 'decision_changed'))
    write_csv(destination / 'predictions.csv', predictions)
    write_csv(destination / 'summary.csv', [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in summaries])
    write_csv(destination / 'calibration_grid.csv', grids)
    write_csv(destination / 'selected_thresholds.csv', selections)
    write_csv(destination / 'enrollment_sources.csv', sources)
    write_csv(destination / 'paired_effects.csv', effects)
    write_csv(destination / 'timing_summary.csv', timing_summary)
    write_csv(destination / 'failure_cases.csv', failures)
    write_csv(destination / 'paired_cases.csv', cases)
    if any(sha256(ROOT / p) != digest for p, digest in fingerprints.items()):
        raise ValueError('Original inputs changed during comparison')
    result = dict(run_directory=destination.relative_to(ROOT).as_posix(), configuration=config, model=model_info,
        input_sha256=fingerprints, summaries=summaries, paired_effects=effects,
        source_sha256={p: sha256(ROOT / p) for p in ['scripts/compare_backbones.py', 'src/recognition/adaface.py',
            'scripts/compare_decisions.py', 'scripts/compare_enrollment.py', 'scripts/run_arcface_baseline.py',
            'scripts/compare_enrollment_pooling.py', 'src/detection/detector.py', 'src/quality/estimator.py',
            'src/alignment/aligner.py', 'src/recognition/embedder.py', 'src/open_set/decision.py']},
        extraction_attempts=len(records), shared_extraction_failures=sum(r['status'] != 'ok' for r in records),
        probe_method_rule_attempts=len(predictions), final_test_evaluated=False, training_performed=False,
        environment=dict(python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
                         torch=torch.__version__, numpy=np.__version__, onnxruntime=ort.__version__, opencv=cv2.__version__,
                         cpu_threads=config['cpu_threads'], cuda_used=False),
        git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        limitations=[
            'Calibration and evaluation reuse development identities/images; descriptive comparison, not held-out generalization.',
            'Pretraining data and runtime engines differ between public checkpoints; no isolated AdaFace loss-function superiority claim.',
            'Both models use the same SCRFD five-point crops. This is not the official AdaFace MTCNN benchmark protocol.',
            'ICFD is a five-identity public-sample closed-set engineering check, without unknown probes.',
            'Paired known-identity intervals condition on the selected thresholds and omit calibration and unknown-sample uncertainty.',
            'Timings use alternating CPU forwards. Composed image-to-embedding sums measured stages; it is not camera FPS or complete UI latency.',
            'No model retraining, adapter, augmentation, reserved-test inference or final-method replacement was performed.'])
    (destination / 'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# AdaFace and ArcFace: development comparison', '',
        '| Model | Enrollment | Rule | Correct known / attempts | False known / extracted unknown |',
        '|---|---|---|---:|---:|']
    for row in summaries:
        if row['dataset'] == 'halfface':
            lines.append(f"| {row['model']} | {row['strategy']} | {row['family']} | {row['correct_known_accepts']}/{row['known_probe_attempts']} | {row['false_known_unknowns']}/{row['unknown_with_embedding']} |")
    lines += ['', '## Scope and limitations', ''] + [f'- {item}' for item in result['limitations']]
    lines += ['', 'Official source: https://github.com/mk-minchul/AdaFace', '',
              'Raw per-probe predictions, all calibration candidates, selected thresholds, paired effects, timing stages and failure cases are saved beside this report.', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(run_directory=result['run_directory'], attempts=len(predictions), shared_failures=result['shared_extraction_failures'],
                         arcface_reproduction_max_difference=model_info['arcface_max_cache_difference']), indent=2))


if __name__ == '__main__':
    main()
