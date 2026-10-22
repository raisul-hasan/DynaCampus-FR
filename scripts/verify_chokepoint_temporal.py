"""Independently recompute temporal results from frozen manifests and real features.

Does not import aggregation, decision, degradation or metric helpers from the runner.
Checks data hashes/ground truth, corruption pixels, pairing, all score vectors,
calibration choices, counts, ROC summaries and paired uncertainty estimates.
"""
import argparse
import csv
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def hash_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seeded(*values):
    return int.from_bytes(hashlib.sha256('\x00'.join(map(str, values)).encode()).digest()[:4], 'little')


def normalize(x):
    y = np.asarray(x, dtype=np.float64)
    return (y / np.linalg.norm(y)).astype(np.float32)


def check_close(actual, expected, message='', tol=2e-6):
    if not np.allclose(actual, expected, atol=tol, rtol=0):
        raise AssertionError(f'{message}: {actual} != {expected}')


def bootstrap(values, identities, cfg):
    identities = np.array(identities)
    labels = sorted(set(identities))
    values = np.asarray(values, float)
    indices = np.random.default_rng(cfg['seed']).integers(0, len(labels),
                   size=(cfg['bootstrap_resamples'], len(labels)))
    numerator = np.array([values[identities == i].sum() for i in labels])
    denominator = np.array([np.sum(identities == i) for i in labels])
    rates = numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)
    return np.quantile(rates, [.025, .975])


def rule_accept(s, q, tau, rule, cfg):
    boundary = tau if rule == 'fixed' else np.maximum(
        tau - cfg['aosm_margin'] * (1 - np.asarray(q)), cfg['aosm_floor'])
    return np.asarray(s) >= boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--real-sample', action='store_true', help='Also repeat real inference on two frames per level')
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    cfg = json.loads((run / 'config.json').read_text())
    prep = json.loads((run / 'preparation.json').read_text())
    complete = json.loads((run / 'completion.json').read_text())
    for name, digest in complete['output_sha256'].items():
        assert hash_file(run / name) == digest, name
    for name, digest in prep['source_sha256'].items():
        assert hash_file(ROOT / name) == digest, name
    for name, digest in prep['groundtruth_sha256'].items():
        assert hash_file(ROOT / name) == digest, name
    assert hash_file(ROOT / 'data/sources/chokepoint/receipt.json') == prep['dataset_receipt_sha256']
    split = json.loads((run / 'split.json').read_text())
    assert not set(split['known']) & set(split['unknown'])
    assert len(split['known']) == 17 and len(split['unknown']) == 8
    seqs = sum(split['sequences'].values(), [])
    assert len(seqs) == len(set(seqs))
    inventory = read_csv(run / 'inventory.csv')
    source_map = {r['path']: r for r in inventory}
    assert len(source_map) == len(inventory)
    expected_ids = sorted({r['identity'] for r in inventory})
    np.random.RandomState(cfg['seed']).shuffle(expected_ids)
    assert split['known'] == expected_ids[:17] and split['unknown'] == expected_ids[17:]
    groundtruth = {}
    for seq in seqs:
        tree = ET.parse(ROOT / cfg['raw_root'] / 'groundtruth' / (seq + '.xml'))
        groundtruth[seq] = {int(f.attrib['number']): {p.attrib['id'] for p in f.findall('person')}
                            for f in tree.getroot().findall('frame')}
    for r in inventory:
        assert hash_file(ROOT / r['path']) == r['sha256']
        assert r['identity'] in groundtruth[r['sequence']][int(r['frame'])]
    tracks = json.loads((run / 'tracks.json').read_text())
    for accounting in read_csv(run/'selection_accounting.csv'):
        candidates = sorted([r for r in inventory if r['identity']==accounting['identity']
                             and r['sequence']==accounting['sequence']],key=lambda r:int(r['frame']))
        selected, pending, previous = [],[],None
        for frame in candidates:
            if previous is not None and int(frame['frame'])-previous != 1:
                pending=[]
            pending.append(frame['path']); previous=int(frame['frame'])
            if len(pending)==5:
                selected.append(pending); pending=[]
                if len(selected)==cfg['max_tracks_per_identity_per_sequence']:
                    break
        saved = [t['paths'] for t in tracks if t['identity']==accounting['identity']
                 and t['sequence']==accounting['sequence'] and t['role']==accounting['role']]
        assert selected == saved
        assert int(accounting['available_frames'])==len(candidates)
        assert int(accounting['selected_tracks'])==len(selected)
        assert int(accounting['unused_frames'])==len(candidates)-5*len(selected)
    tmap, used = {}, set()
    for t in tracks:
        assert t['track_id'] not in tmap
        tmap[t['track_id']] = t
        assert len(t['paths']) == cfg['track_size']
        assert t['sequence'] in split['sequences'][t['role']]
        assert all(b-a == 1 for a,b in zip(t['frames'], t['frames'][1:]))
        assert t['truth'] == ('known' if t['identity'] in split['known'] else 'unknown')
        for path, frame in zip(t['paths'], t['frames']):
            assert path not in used
            used.add(path)
            assert source_map[path]['identity'] == t['identity']
            assert source_map[path]['sequence'] == t['sequence']
            assert int(source_map[path]['frame']) == frame
        rng = np.random.default_rng(seeded(cfg['seed'], t['track_id'], 'mixed'))
        assert t['mixed_levels'] == rng.choice(['clean','mild','medium','severe'], 5,
                                              p=cfg['mixed_probabilities']).tolist()
        assert t['random_index'] == seeded(cfg['seed'], t['track_id'], 'random') % 5
        if t['truth'] == 'known':
            pool = sorted({p for other in tracks if other['role'] == t['role']
                           and other['truth'] == 'known' and other['identity'] != t['identity']
                           for p in other['paths']})
            assert t['intruder_path'] == pool[seeded(cfg['seed'], t['track_id'], 'intruder') % len(pool)]
    gallery_rows = read_csv(run / 'gallery.csv')
    assert {r['identity'] for r in gallery_rows} == set(split['known'])
    assert not used & {r['path'] for r in gallery_rows}
    role_hashes = {role: {source_map[p]['sha256'] for t in tracks if t['role']==role
                         for p in t['paths']} for role in ['calibration','evaluation']}
    gallery_hashes = {r['sha256'] for r in gallery_rows}
    assert not role_hashes['calibration'] & role_hashes['evaluation']
    assert not gallery_hashes & (role_hashes['calibration'] | role_hashes['evaluation'])
    for r in gallery_rows:
        candidates = [x for x in inventory if x['sequence'] == cfg['gallery_sequence']
                      and x['identity'] == r['identity']]
        assert int(r['frame']) == min(int(x['frame']) for x in candidates)
    receipt = json.loads((run / 'cache_receipt.json').read_text())
    cache = ROOT / receipt['cache']
    for name, digest in receipt['files'].items():
        assert hash_file(cache / name) == digest, name
    if args.real_sample:
        import onnxruntime as ort
        from src.recognition import FaceEmbedder
        ort.set_default_logger_severity(3)
        embedder = FaceEmbedder(cpu_threads=cfg['cpu_threads'])
        assert hash_file(embedder.model_path) == receipt['runtime']['model_sha256']
    lookup, pixel_checks, inference_checks = {}, 0, 0
    cv2.setNumThreads(1)
    for level in ['clean', 'mild', 'medium', 'severe']:
        records = json.loads((cache / (level + '.json')).read_text())
        features = np.load(cache / (level + '.npz'), allow_pickle=False)['embeddings']
        check_close(np.linalg.norm(features, axis=1), 1, 'Unit frame embeddings')
        for index, (r, feature) in enumerate(zip(records, features)):
            assert r['seed'] == seeded(cfg['seed'], r['path'], level)
            image = cv2.imread(str(ROOT / r['path']))
            image = cv2.resize(image, (112,112), interpolation=cv2.INTER_CUBIC)
            if level != 'clean':
                kernel, sigma, quality = {'mild': (3,1.,75), 'medium': (5,2.,50), 'severe': (7,3.5,25)}[level]
                image = cv2.GaussianBlur(image, (kernel,kernel), sigma)
                if level in ['medium','severe']:
                    n = 56 if level == 'medium' else 28
                    image = cv2.resize(cv2.resize(image, (n,n), interpolation=cv2.INTER_CUBIC),
                                       (112,112), interpolation=cv2.INTER_CUBIC)
                if level == 'severe':
                    image = np.clip(np.round(image.astype(float) + np.random.default_rng(r['seed']).normal(
                        0,15,image.shape)),0,255).astype(np.uint8)
                ok, encoded = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,quality])
                assert ok
                image = cv2.imdecode(encoded,cv2.IMREAD_COLOR)
            assert hashlib.sha256(image.tobytes()).hexdigest() == r['pixel_sha256']
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lap = np.clip(.6*min(cv2.Laplacian(gray,cv2.CV_64F).var()/500,1)
                          + .4*min(gray.std()/80,1),1e-4,1)
            check_close(lap, r['laplacian'], 'quality')
            if args.real_sample and index in [0, len(records)//2]:
                raw = embedder.model.get_feat(image).flatten()
                check_close(normalize(raw),feature,'Repeated real inference',tol=2e-5)
                check_close(np.linalg.norm(raw),r['norm'],'Repeated raw norm',tol=2e-4)
                inference_checks += 1
            lookup[(r['path'],level)] = (feature,r['norm'],r['laplacian'])
            pixel_checks += 1
        print('Verified original/corrupted pixels:', level,len(records),flush=True)
    # Materialize once: NpzFile lazily decompresses a full array on EVERY access.
    with np.load(cache / 'query_scores.npz',allow_pickle=False) as archive:
        stored = {key: archive[key] for key in archive.files}
    keys = json.loads((cache / 'query_keys.json').read_text())
    ids = keys['gallery_ids']
    gallery = np.stack([lookup[(r['path'],'clean')][0] for r in gallery_rows])
    check_close(gallery,stored['gallery'])
    predictions = read_csv(run / 'predictions.csv')
    pred_map = {(r['condition'],r['contamination'],r['method'],r['track_id']):r for r in predictions}
    assert len(pred_map) == len(predictions)
    paired = read_csv(run / 'paired_intervals.csv')
    cal_table = read_csv(run / 'calibration.csv')
    summary_table = read_csv(run / 'summary.csv')
    pair_table = read_csv(run / 'pairwise_verification.csv')
    score_checks = 0
    for ki, (condition, contamination, method) in enumerate(keys['keys']):
        group, scores_all, truth_all = [], [], []
        for ti,t in enumerate(tracks):
            levels = t['mixed_levels'].copy() if condition == 'mixed' else [condition]*5
            paths = t['paths'].copy()
            if contamination != 'none' and t['truth'] == 'known':
                paths[-1] = t['intruder_path']
                if contamination == 'mild':
                    levels[-1] = 'mild'
            features, norms, laps = map(np.array,zip(*(lookup[(p,l)] for p,l in zip(paths,levels))))
            if method in ['first','random','best_norm','best_laplacian']:
                selected = {'first':0,'random':t['random_index'],'best_norm':int(np.argmax(norms)),
                            'best_laplacian':int(np.argmax(laps))}[method]
                vector = features[selected]
            elif method == 'mean':
                vector = normalize(features.mean(axis=0))
            else:
                beta = float(method.split('_')[-1])
                z = beta*(norms-np.mean(norms))/(np.std(norms)+1e-8)
                w = np.exp(z-z.max()); w /= w.sum()
                vector = normalize(np.sum(features*w[:,None],axis=0))
            scores = gallery @ vector
            check_close(vector,stored['vectors'][ki,ti],'Query vector')
            check_close(scores,stored['scores'][ki,ti],'All gallery scores')
            row = pred_map[(condition,contamination,method,t['track_id'])]
            order = np.argsort(-scores,kind='stable')
            assert row['top1_id'] == ids[order[0]]
            check_close(float(row['top1_score']),scores[order[0]])
            check_close(float(row['top2_score']),scores[order[1]])
            q = np.clip((norms-cfg['norm_bounds'][0]) / np.ptp(cfg['norm_bounds']),0,1).mean()
            check_close(float(row['quality']),q)
            correct = ids[order[0]] == t['identity']
            assert (row['correct']=='True') == correct
            truth = ids.index(t['identity']) if t['truth']=='known' else -1
            assert truth == stored['truths'][ki,ti]
            if truth >= 0:
                check_close(float(row['true_score']),scores[truth])
            group.append(row); scores_all.append(scores); truth_all.append(truth)
            score_checks += len(scores)
        for rule in ['fixed','aosm']:
            crow = next(r for r in cal_table if (r['condition'],r['contamination'],r['method'],r['rule'])
                        == (condition,contamination,method,rule))
            tau = float(crow['threshold'])
            cal = [r for r in group if r['role']=='calibration' and r['truth']=='unknown']
            s = np.array([float(r['top1_score']) for r in cal]); q = np.array([float(r['quality']) for r in cal])
            budget = int(np.floor(len(s)*cfg['target_fpir']+1e-12))
            assert rule_accept(s,q,tau,rule,cfg).sum() <= budget
            critical = s if rule=='fixed' else s+(1-q)*cfg['aosm_margin']
            candidates = np.unique(np.r_[-1.,critical,np.nextafter(critical,np.inf),1.1])
            admissible = [v for v in candidates if rule_accept(s,q,v,rule,cfg).sum()<=budget]
            assert tau == min(admissible)
            assert int(crow['false_accepts']) == rule_accept(s,q,tau,rule,cfg).sum()
            for row in group:
                expected = bool(rule_accept(float(row['top1_score']),float(row['quality']),tau,rule,cfg))
                assert (row[rule+'_accepted']=='True') == expected
            for role in ['calibration','evaluation']:
                saved = next(r for r in summary_table if (r['condition'],r['contamination'],r['method'],r['rule'],r['role'])
                             == (condition,contamination,method,rule,role))
                known = [r for r in group if r['role']==role and r['truth']=='known']
                unknown = [r for r in group if r['role']==role and r['truth']=='unknown']
                ok = np.array([r['correct']=='True' for r in known])
                success = ok & np.array([r[rule+'_accepted']=='True' for r in known])
                fa = sum(r[rule+'_accepted']=='True' for r in unknown)
                assert int(saved['known_tracks']) == len(known)
                assert int(saved['unknown_tracks']) == len(unknown)
                assert int(saved['rank1_correct']) == ok.sum()
                assert int(saved['correct_known_accepted']) == success.sum()
                assert int(saved['false_known_accepted']) == fa
                check_close(float(saved['rank1']),ok.mean())
                check_close(float(saved['dir']),success.mean())
                check_close(float(saved['fpir']),fa/len(unknown))
                check_close([float(saved['dir_identity_bootstrap_low']),float(saved['dir_identity_bootstrap_high'])],
                            bootstrap(success,[r['identity'] for r in known],cfg))
        mask = np.array([r['role']=='evaluation' for r in group])
        scores, truth = np.array(scores_all)[mask], np.array(truth_all)[mask]
        labels = np.zeros_like(scores,dtype=int)
        labels[np.flatnonzero(truth>=0),truth[truth>=0]] = 1
        fpr,tpr,_ = roc_curve(labels.ravel(),scores.ravel(),drop_intermediate=False)
        ei = np.argmin(np.abs(fpr-(1-tpr)))
        expected = {'genuine_pairs':labels.sum(),'impostor_pairs':labels.size-labels.sum(),
                    'eer':(fpr[ei]+1-tpr[ei])/2,'roc_auc':roc_auc_score(labels.ravel(),scores.ravel()),
                    'tar_pair_far_1pct':tpr[fpr<=.01].max(),'tar_pair_far_0_1pct':tpr[fpr<=.001].max()}
        prow = next(r for r in pair_table if (r['condition'],r['contamination'],r['method'])==(condition,contamination,method))
        for k,v in expected.items():
            check_close(float(prow[k]),v,k)
    for row in paired:
        base = [r for r in predictions if (r['condition'],r['contamination'],r['method'],r['role'],r['truth'])
                == (row['condition'],row['contamination'],row['reference'],'evaluation','known')]
        delta = []
        for a in base:
            b = pred_map[(row['condition'],row['contamination'],row['comparison'],a['track_id'])]
            def success(r):
                return int(r['correct']=='True' and (row['metric']=='rank1' or r['fixed_accepted']=='True'))
            delta.append(success(b)-success(a))
        check_close(float(row['delta']),np.mean(delta))
        check_close([float(row['ci_low']),float(row['ci_high'])],bootstrap(delta,[r['identity'] for r in base],cfg))
        assert int(row['helped']) == sum(x>0 for x in delta)
        assert int(row['harmed']) == sum(x<0 for x in delta)
    for row in read_csv(run/'quality_diagnostics.csv'):
        ns,laps,sims = [],[],[]
        for t in tracks:
            if t['role']=='evaluation' and t['truth']=='known':
                for p in t['paths']:
                    f,n,l = lookup[(p,row['level'])]
                    ns.append(n); laps.append(l); sims.append(float(gallery[ids.index(t['identity'])]@f))
        check_close(float(row['mean_norm']),np.mean(ns))
        check_close(float(row['mean_laplacian']),np.mean(laps))
        check_close(float(row['norm_spearman_r']),spearmanr(ns,sims).statistic)
        check_close(float(row['laplacian_spearman_r']),spearmanr(laps,sims).statistic)
    result = {'verified_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
              'verifier_sha256':hash_file(Path(__file__)), 'original_crops_checked':len(inventory),
              'corrupted_pixels_checked':pixel_checks,'gallery_scores_checked':score_checks,
              'query_predictions_checked':len(predictions),'summary_rows_checked':len(summary_table),
              'paired_intervals_checked':len(paired),'repeated_real_inference_checks':inference_checks,
              'completion_sha256':hash_file(run/'completion.json'),
              'query_score_cache_sha256':hash_file(cache/'query_scores.npz'),
              'scope':cfg['scope']}
    (run/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
