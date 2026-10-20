"""Freeze development-selected settings, then evaluate reserved originals once.

Freeze reads manifests and development evidence only. Evaluation opens reserved
images after writing a start receipt. Further figures/verification use caches.
"""
import argparse
import importlib.metadata
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
from scripts.compare_decisions import apply_rule, summarize
from scripts.compare_enrollment import gallery_candidates, paired_bootstrap, reject_overlap, unit_mean
from scripts.run_arcface_baseline import extract, read_csv, sha256, write_csv
from scripts.run_robustness import score_records


def now():
    return datetime.now(timezone.utc).isoformat()


def assert_hashes(hashes, root=ROOT):
    for path, expected in hashes.items():
        if sha256(root/path) != expected:
            raise ValueError(f'Frozen source changed: {path}')


def claim_evaluation(directory):
    # Exclusive creation prevents an accidental second inference run.
    receipt = {'started_utc': now(), 'freeze_sha256': sha256(directory/'freeze.json')}
    with (directory/'evaluation_started.json').open('x', encoding='utf-8') as out:
        json.dump(receipt, out, indent=2)


def select_test_rows(hf_rows, icfd_rows, split):
    reserved = split['halfface']['test_reserved']
    known, unknown = set(reserved['known']), set(reserved['unknown'])
    development = set(sum(split['halfface']['development'].values(), []))
    if known & unknown or (known | unknown) & development:
        raise ValueError('Identity partition overlap')
    hf = [{**r, 'truth_status': 'known' if r['subject_id'] in known else 'unknown',
           'evaluation_role': 'gallery' if r['subject_id'] in known and r['mask_status'] == 'unmasked' else 'probe'}
          for r in hf_rows if r['subject_id'] in known | unknown]
    candidates = gallery_candidates(hf, 'halfface', known, 'up_to_three_photos')
    # Require the documented one-photo frontal choice before inference too.
    gallery_candidates(hf, 'halfface', known, 'one_photo')
    chosen = {r['path'] for rows in candidates.values() for r in rows}
    probes = {r['path'] for r in hf if r['truth_status'] == 'unknown' or r['mask_status'] == 'masked'}
    hf = [r for r in hf if r['path'] in chosen | probes]
    sample_ids = set(split['icfd_sample']['reserved'])
    if sample_ids & set(split['icfd_sample']['engineering_development']):
        raise ValueError('ICFD identity partition overlap')
    icfd = [{**r, 'truth_status': 'known', 'mask_status': 'unspecified', 'pose': 'unspecified',
             'evaluation_role': r['source_role']} for r in icfd_rows if r['subject_id'] in sample_ids]
    selected = gallery_candidates(icfd, 'icfd_sample', sample_ids, 'up_to_three_photos')
    chosen = {r['path'] for rows in selected.values() for r in rows}
    icfd = [r for r in icfd if r['evaluation_role'] == 'probe' or r['path'] in chosen]
    rows = sorted(hf+icfd, key=lambda r:(r['dataset'], r['path']))
    if len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Duplicate evaluation paths')
    if len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Exact duplicate evaluation images')
    dev_hashes = {r['sha256'] for r in hf_rows if r['subject_id'] in development}
    dev_hashes |= {r['sha256'] for r in icfd_rows if r['subject_id'] in split['icfd_sample']['engineering_development']}
    if dev_hashes & {r['sha256'] for r in rows}:
        raise ValueError('Exact image overlap with development')
    # The two source manifests have different metadata columns.
    keys = list(dict.fromkeys(key for row in rows for key in row))
    return [{key:row.get(key,'') for key in keys} for row in rows]


def cluster_interval(rows, truth, numerator, denominator, samples, seed):
    subset = [r for r in rows if r['truth_status'] == truth]
    subjects = sorted({r['subject_id'] for r in subset})
    counts = np.array([sum(denominator(r) for r in subset if r['subject_id'] == s) for s in subjects])
    successes = np.array([sum(numerator(r) for r in subset if r['subject_id'] == s) for s in subjects])
    draws = np.random.default_rng(seed).integers(0, len(subjects), size=(samples, len(subjects)))
    total = counts[draws].sum(axis=1)
    values = successes[draws].sum(axis=1)[total > 0] / total[total > 0]
    return [float(v) for v in np.percentile(values, [2.5,97.5])]


def make_gallery(records, features, dataset, known_ids, strategy):
    candidates = gallery_candidates(records, dataset, known_ids, strategy)
    ids, prototypes, sources = [], [], []
    for subject, rows in candidates.items():
        usable = [r for r in rows if r['status'] == 'ok']
        sources += [{'dataset':dataset, 'strategy':strategy, 'subject_id':subject, 'path':r['path'],
                     'status':r['status'], 'used_in_template':r['status'] == 'ok'} for r in rows]
        if usable:
            ids.append(subject)
            prototypes.append(unit_mean([features[int(r['embedding_index'])] for r in usable]))
    if len(ids) < 2:
        raise ValueError('Fewer than two enrolled identities; evaluation cannot proceed')
    return ids, np.stack(prototypes), sources


def freeze(config_path):
    config = yaml.safe_load((ROOT/config_path).read_text())
    if config['retune_on_test'] or config['training'] or config['degradations'] != 'none':
        raise ValueError('Only frozen original-image evaluation is supported')
    decisions = ROOT/config['decision_run']
    decision_info = json.loads((decisions/'comparison.json').read_text())
    enrollment = ROOT/decision_info['configuration']['enrollment_run']
    enrollment_info = json.loads((enrollment/'comparison.json').read_text())
    baseline = ROOT/enrollment_info['baseline_run']
    if decision_info['final_test_evaluated'] or enrollment_info['model']['mock']:
        raise ValueError('Expected real-model development calibration')
    fingerprints = json.loads((baseline/'source_fingerprints.json').read_text())
    assert_hashes(fingerprints)
    if sha256(ROOT/'src/open_set/decision.py') != decision_info['engine_sha256']:
        raise ValueError('Decision engine differs from development calibration')
    split_path = decisions/'split_snapshot.json'
    split = json.loads(split_path.read_text())
    preprocessing = yaml.safe_load((baseline/'config.yaml').read_text())
    hf_path, icfd_path = ROOT/preprocessing['halfface_manifest'], ROOT/preprocessing['icfd_manifest']
    if (sha256(hf_path), sha256(icfd_path)) != (split['manifest_sha256']['halfface'], split['manifest_sha256']['icfd']):
        raise ValueError('Source manifests changed')
    rows = select_test_rows(read_csv(hf_path), read_csv(icfd_path), split)
    one = next(r for r in enrollment_info['summaries'] if r['dataset'] == 'halfface' and r['strategy'] == 'one_photo')
    methods = {
        'one_photo_score':dict(strategy='one_photo', score_threshold=one['threshold'], margin_threshold=0., quality_threshold=0.),
        'few_photo_score':dict(strategy='up_to_three_photos', **{k:decision_info['selected_thresholds']['score_only'][k] for k in ['score_threshold','margin_threshold','quality_threshold']}),
        'few_photo_margin':dict(strategy='up_to_three_photos', **{k:decision_info['selected_thresholds']['score_margin'][k] for k in ['score_threshold','margin_threshold','quality_threshold']})}
    if config['methods'] != list(methods) or config['primary_method'] != 'few_photo_margin':
        raise ValueError('Unexpected method selection')
    source_paths = list(fingerprints)+[config_path, 'scripts/run_final_evaluation.py','scripts/run_robustness.py',
        'scripts/compare_decisions.py','scripts/compare_enrollment.py','src/open_set/decision.py','configs/models.yaml']
    evidence_paths = [decisions/'comparison.json', enrollment/'comparison.json', baseline/'config.yaml', split_path, hf_path, icfd_path]
    fingerprints.update({p:sha256(ROOT/p) for p in source_paths})
    fingerprints.update({p.relative_to(ROOT).as_posix():sha256(p) for p in evidence_paths})
    destination = ROOT/'results/final'/('heldout_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    destination.mkdir(parents=True)
    write_csv(destination/'evaluation_manifest.csv', rows)
    receipt = dict(frozen_utc=now(), configuration=config, methods=methods, preprocessing=preprocessing,
                   primary_method=config['primary_method'], split=split, expected_model=enrollment_info['model'],
                   source_sha256=fingerprints, evaluation_manifest_sha256=sha256(destination/'evaluation_manifest.csv'),
                   pre_inference_counts=dict(Counter(f'{r["dataset"]}:{r["evaluation_role"]}:{r["truth_status"]}' for r in rows)),
                   test_images_opened_during_freeze=False)
    (destination/'freeze.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('FROZEN', (destination/'freeze.json').relative_to(ROOT).as_posix())
    print('METHODS', json.dumps(methods))
    print('MANIFEST COUNTS', json.dumps(receipt['pre_inference_counts']))


def evaluate(freeze_path):
    directory = (ROOT/freeze_path).parent
    receipt = json.loads((directory/'freeze.json').read_text())
    assert_hashes(receipt['source_sha256'])
    if sha256(directory/'evaluation_manifest.csv') != receipt['evaluation_manifest_sha256']:
        raise ValueError('Evaluation manifest changed after freezing')
    claim_evaluation(directory)
    print('START', directory, flush=True)
    rows = read_csv(directory/'evaluation_manifest.csv')
    records, features, model = extract(rows, receipt['preprocessing'], directory)
    if model['mock'] or any(model[k] != receipt['expected_model'][k] for k in ['recognizer_sha256','detector_sha256']):
        raise ValueError('Model checkpoint changed')
    split, config = receipt['split'], receipt['configuration']
    known_ids = split['halfface']['test_reserved']['known']
    probes = [r for r in records if r['dataset'] == 'halfface' and r['evaluation_role'] == 'probe']
    summaries, all_predictions, enrollment_log, paired = [], [], [], {}
    cache = ROOT/model['embedding_cache']
    for method, settings in receipt['methods'].items():
        ids, prototypes, sources = make_gallery(records, features, 'halfface', known_ids, settings['strategy'])
        reject_overlap(sources, probes)
        np.savez_compressed(cache.parent/f'{method}_gallery.npz', embeddings=prototypes, identities=np.array(ids))
        scored = score_records(probes, features, ids, prototypes)
        output = apply_rule(scored, settings, method)
        summary = summarize(output, settings, method)
        ci = cluster_interval(output,'known',lambda r:r['correct_known_accept'],lambda r:1,config['bootstrap_samples'],config['seed'])
        fpir_ci = cluster_interval(output,'unknown',lambda r:r['decision'] == 'KNOWN',lambda r:r['status'] == 'ok',config['bootstrap_samples'],config['seed'])
        summary.update(dataset='halfface', scope='identity_disjoint_heldout_originals', method=method, strategy=settings['strategy'],
            primary=method == receipt['primary_method'], gallery_identities=len(ids), gallery_images=sum(r['used_in_template'] for r in sources),
            intended_gallery_identities=len(known_ids), missing_gallery_ids=sorted(set(known_ids)-set(ids)),
            rank1_correct=sum(r['rank1_correct'] for r in output),
            rank1_end_to_end=sum(r['rank1_correct'] for r in output)/summary['known_probe_attempts'],
            false_known_end_to_end=summary['false_known_unknowns']/summary['unknown_probe_attempts'],
            known_accept_ci95_low=ci[0],known_accept_ci95_high=ci[1],fpir_cluster_ci95_low=fpir_ci[0],fpir_cluster_ci95_high=fpir_ci[1])
        summaries.append(summary)
        for row in output:
            row['method'] = method
        all_predictions += output
        enrollment_log += [{**r,'method':method} for r in sources]
        paired[method] = output
        print(json.dumps(summary), flush=True)
    effects = []
    for first, second in [('one_photo_score','few_photo_score'),('few_photo_score','few_photo_margin'),('one_photo_score','few_photo_margin')]:
        effect = paired_bootstrap(paired[first],paired[second],known_ids,config['bootstrap_samples'],config['seed'],'correct_known_accept')
        effect['note'] = 'Paired held-out known-identity bootstrap with fixed development thresholds; calibration and unknown-sample uncertainty excluded.'
        effects.append(dict(comparison=f'{second}_minus_{first}',**effect))
    icfd_summaries, icfd_predictions = [], []
    sample_ids = split['icfd_sample']['reserved']
    sample_probes = [r for r in records if r['dataset'] == 'icfd_sample' and r['evaluation_role'] == 'probe']
    for strategy in config['icfd_enrollment']:
        ids, prototypes, sources = make_gallery(records,features,'icfd_sample',sample_ids,strategy)
        reject_overlap(sources,sample_probes)
        np.savez_compressed(cache.parent/f'icfd_{strategy}_gallery.npz',embeddings=prototypes,identities=np.array(ids))
        scored = score_records(sample_probes,features,ids,prototypes)
        for row in scored:
            row.update(strategy=strategy,decision='CLOSED_SET_ONLY' if row['status'] == 'ok' else 'CAPTURE_FAILURE')
        icfd_predictions += scored
        summary = dict(dataset='icfd_sample',scope='reserved_public_sample_closed_set_engineering_only',strategy=strategy,
            gallery_identities=len(ids), gallery_images=sum(r['used_in_template'] for r in sources),
            probe_attempts=len(scored), successful_extractions=sum(r['status'] == 'ok' for r in scored),
            rank1_correct=sum(r['rank1_correct'] for r in scored), rank1_end_to_end=sum(r['rank1_correct'] for r in scored)/len(scored),
            unknown_probes=0,fpir_not_estimable=True,missing_gallery_ids=sorted(set(sample_ids)-set(ids)))
        icfd_summaries.append(summary)
        enrollment_log += [{**r,'method':f'icfd_{strategy}'} for r in sources]
        print(json.dumps(summary), flush=True)
    write_csv(directory/'predictions.csv',all_predictions)
    write_csv(directory/'icfd_predictions.csv',icfd_predictions)
    write_csv(directory/'enrollment_sources.csv',enrollment_log)
    write_csv(directory/'summary.csv',[{k:v for k,v in s.items() if not isinstance(v,(dict,list))} for s in summaries])
    write_csv(directory/'icfd_summary.csv',[{k:v for k,v in s.items() if not isinstance(v,(dict,list))} for s in icfd_summaries])
    write_csv(directory/'paired_effects.csv',effects)
    result = dict(run_directory=directory.relative_to(ROOT).as_posix(),completed_utc=now(),freeze_sha256=sha256(directory/'freeze.json'),
        final_test_evaluated=True,thresholds_retuned=False,primary_method=receipt['primary_method'],model=model,summaries=summaries,
        icfd_summaries=icfd_summaries,paired_effects=effects,images_attempted=len(records),successful_embeddings=len(features),
        extraction_counts=dict(Counter(r['status'] for r in records)),
        versions={n:importlib.metadata.version(n) for n in ['numpy','opencv-python','onnxruntime','insightface','scikit-image','matplotlib']},
        git_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True).stdout.strip(),
        limitations=['Identity-disjoint within the public halfFace dataset; pretraining overlap and near duplicates are not ruled out.',
                     'Known halfFace probes are masked; unknown probes mix mask conditions. Not a Bangladesh-wide or CCTV-deployment benchmark.',
                     'ICFD uses five reserved public-sample IDs without unknowns; FPIR cannot be estimated there.',
                     'Cluster bootstrap conditions on frozen development calibration; small unknown samples yield unstable error estimates.',
                     'Quality gate is disabled. No AdaFace, adapter training or temporal tracking has been validated.'])
    (directory/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('COMPLETE',directory,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true')
    group.add_argument('--evaluate',metavar='FREEZE_JSON')
    parser.add_argument('--config',default='configs/final_evaluation.yaml')
    args = parser.parse_args()
    freeze(args.config) if args.freeze else evaluate(args.evaluate)
