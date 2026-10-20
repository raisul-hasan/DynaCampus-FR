"""Independently audit held-out searches and decisions from saved real embeddings."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_arcface_baseline import read_csv,sha256


def expected_gallery(records,features,dataset,intended,strategy):
    ids,vectors,used_paths = [],[],[]
    for subject in sorted(intended):
        rows = [r for r in records if r['dataset'] == dataset and r['subject_id'] == subject and r['evaluation_role'] == 'gallery']
        if dataset == 'halfface':
            rows.sort(key=lambda r:({'front':0,'left':1,'right':2}[r['pose']],r['path']))
            if strategy == 'one_photo':
                rows = [r for r in rows if r['pose'] == 'front']
        else:
            rows.sort(key=lambda r:r['path'])
        rows = rows[:1 if strategy == 'one_photo' else 3]
        used_paths += [r['path'] for r in rows]
        selected = [features[int(r['embedding_index'])] for r in rows if r['status'] == 'ok']
        if selected:
            vector = np.asarray(selected,dtype=np.float64).mean(axis=0)
            vector = (vector/np.linalg.norm(vector)).astype(np.float32)
            vectors.append(vector)
            ids.append(subject)
    return ids,np.stack(vectors),used_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    run = ROOT/args.run
    info = json.loads((run/'comparison.json').read_text())
    freeze = json.loads((run/'freeze.json').read_text())
    start = json.loads((run/'evaluation_started.json').read_text())
    assert freeze['frozen_utc'] <= start['started_utc'] <= info['completed_utc']
    assert sha256(run/'freeze.json') == start['freeze_sha256'] == info['freeze_sha256']
    assert sha256(run/'evaluation_manifest.csv') == freeze['evaluation_manifest_sha256']
    assert not freeze['test_images_opened_during_freeze'] and not info['thresholds_retuned']
    assert info['primary_method'] == freeze['primary_method'] == 'few_photo_margin'
    for path,digest in freeze['source_sha256'].items():
        assert sha256(ROOT/path) == digest,path
    manifest = read_csv(run/'evaluation_manifest.csv')
    for row in manifest:
        assert sha256(ROOT/row['path']) == row['sha256'],row['path']
    split = freeze['split']
    dev_ids = set(sum(split['halfface']['development'].values(),[])+split['icfd_sample']['engineering_development'])
    test_ids = set(sum(split['halfface']['test_reserved'].values(),[])+split['icfd_sample']['reserved'])
    assert not dev_ids & test_ids
    assert {r['subject_id'] for r in manifest} <= test_ids
    records = read_csv(run/'extraction.csv')
    assert {r['path'] for r in records} == {r['path'] for r in manifest}
    by_path = {r['path']:r for r in records}
    matrix_path = ROOT/info['model']['embedding_cache']
    features = np.load(matrix_path,allow_pickle=False)['embeddings']
    assert np.isfinite(features).all()
    np.testing.assert_allclose(np.linalg.norm(features,axis=1),1.,atol=1e-5)
    assert sorted(int(r['embedding_index']) for r in records if r['status'] == 'ok') == list(range(len(features)))
    assert len(records) == info['images_attempted'] and len(features) == info['successful_embeddings']
    assert dict(Counter(r['status'] for r in records)) == info['extraction_counts']
    for key in ['recognizer_sha256','detector_sha256']:
        assert info['model'][key] == freeze['expected_model'][key]
    predictions = read_csv(run/'predictions.csv')
    attempts,searches = 0,0
    for method,settings in freeze['methods'].items():
        intended = split['halfface']['test_reserved']['known']
        ids,prototypes,gallery_paths = expected_gallery(records,features,'halfface',intended,settings['strategy'])
        saved = np.load(matrix_path.parent/f'{method}_gallery.npz',allow_pickle=False)
        assert ids == saved['identities'].tolist()
        np.testing.assert_array_equal(prototypes,saved['embeddings'])
        rows = [r for r in predictions if r['method'] == method]
        expected_probes = {r['path'] for r in records if r['dataset'] == 'halfface' and r['evaluation_role'] == 'probe'}
        assert len(rows) == len(expected_probes) and {r['path'] for r in rows} == expected_probes
        assert not expected_probes & set(gallery_paths)
        counts = Counter()
        correct = rank1 = false_known = wrong_known = covered = unknown_valid = known_valid = 0
        for row in rows:
            source = by_path[row['path']]
            assert row['status'] == source['status'] and row['quality'] == source['quality']
            assert all(row[k] == source[k] for k in ['dataset','subject_id','truth_status','mask_status'])
            identity,rank_correct = '',False
            if row['status'] != 'ok':
                state = 'CAPTURE_FAILURE'
            else:
                values = prototypes@features[int(source['embedding_index'])]
                order = np.argsort(values)[::-1]
                best,score,second = ids[int(order[0])],float(values[order[0]]),float(values[order[1]])
                assert (best,score,second) == (row['top1_id'],float(row['top1_score']),float(row['top2_score']))
                rank_correct = row['truth_status'] == 'known' and best == row['subject_id']
                if float(row['quality']) < settings['quality_threshold']:
                    state = 'INSUFFICIENT_QUALITY'
                elif score < settings['score_threshold']:
                    state = 'UNKNOWN'
                elif score-second < settings['margin_threshold']:
                    state = 'AMBIGUOUS'
                else:
                    state,identity = 'KNOWN',best
                unknown_valid += row['truth_status'] == 'unknown'
                known_valid += row['truth_status'] == 'known'
                searches += 1
            assert (state,identity) == (row['decision'],row['accepted_id'])
            assert rank_correct == (row['rank1_correct'] == 'True')
            assert (rank_correct and state == 'KNOWN') == (row['correct_known_accept'] == 'True')
            assert (state in {'KNOWN','UNKNOWN'}) == (row['covered'] == 'True')
            counts[state] += 1
            correct += rank_correct and state == 'KNOWN'
            rank1 += rank_correct
            false_known += row['truth_status'] == 'unknown' and state == 'KNOWN'
            wrong_known += row['truth_status'] == 'known' and state == 'KNOWN' and not rank_correct
            covered += state in {'KNOWN','UNKNOWN'}
            attempts += 1
        s = next(r for r in info['summaries'] if r['method'] == method)
        assert (correct,rank1,false_known,wrong_known) == (s['correct_known_accepts'],s['rank1_correct'],s['false_known_unknowns'],s['wrong_known_identity_accepts'])
        assert dict(counts) == s['decision_counts']
        assert s['gallery_identities'] == len(ids) and s['missing_gallery_ids'] == sorted(set(intended)-set(ids))
        assert correct/s['known_probe_attempts'] == s['correct_known_accept_end_to_end']
        assert false_known/unknown_valid == s['fpir_conditional']
        assert known_valid == s['known_with_embedding'] and unknown_valid == s['unknown_with_embedding']
        assert covered/len(rows) == s['end_to_end_decision_coverage']
        assert all(s[k] == settings[k] for k in ['score_threshold','margin_threshold','quality_threshold'])
    sample = read_csv(run/'icfd_predictions.csv')
    for strategy in freeze['configuration']['icfd_enrollment']:
        ids,prototypes,gallery_paths = expected_gallery(records,features,'icfd_sample',split['icfd_sample']['reserved'],strategy)
        saved = np.load(matrix_path.parent/f'icfd_{strategy}_gallery.npz',allow_pickle=False)
        assert ids == saved['identities'].tolist()
        np.testing.assert_array_equal(prototypes,saved['embeddings'])
        rows = [r for r in sample if r['strategy'] == strategy]
        assert {r['path'] for r in rows} == {r['path'] for r in records if r['dataset'] == 'icfd_sample' and r['evaluation_role'] == 'probe'}
        assert not {r['path'] for r in rows} & set(gallery_paths)
        correct = 0
        for row in rows:
            source = by_path[row['path']]
            assert row['status'] == source['status']
            assert row['subject_id'] == source['subject_id'] and row['truth_status'] == source['truth_status']
            if row['status'] == 'ok':
                scores = prototypes@features[int(source['embedding_index'])]
                ranking = np.argsort(scores)[::-1]
                assert row['top1_id'] == ids[int(ranking[0])]
                assert float(row['top1_score']) == float(scores[ranking[0]])
                correct += row['top1_id'] == row['subject_id']
                searches += 1
            attempts += 1
        summary = next(r for r in info['icfd_summaries'] if r['strategy'] == strategy)
        assert summary['rank1_correct'] == correct and summary['probe_attempts'] == len(rows)
        assert summary['unknown_probes'] == 0 and summary['fpir_not_estimable']
    report = dict(verified_prediction_attempts=attempts,verified_cosine_searches=searches,
        source_images_unchanged=len(manifest),development_identity_overlap=0,
        frozen_before_inference=True,unchanged_development_thresholds=True,primary_method_preserved=True,
        verifier_sha256=sha256(__file__),freeze_sha256=sha256(run/'freeze.json'),
        predictions_sha256=sha256(run/'predictions.csv'),icfd_predictions_sha256=sha256(run/'icfd_predictions.csv'),
        cache_sha256={p.name:sha256(p) for p in matrix_path.parent.glob('*.npz')})
    (run/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'cache_sha256'},indent=2))


if __name__ == '__main__':
    main()
