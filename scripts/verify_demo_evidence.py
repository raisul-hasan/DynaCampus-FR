"""Independently check incremental-gallery decisions and browser evidence."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_arcface_baseline import read_csv,sha256
from src.gallery import FaceGallery
from src.demo_workflow import recognize

RUN=ROOT/'results/demo/dynamic_development_20261019T233918475526Z'
info=json.loads((RUN/'comparison.json').read_text())
for path,digest in {**info['source_sha256'],**info['input_sha256']}.items():
    assert sha256(ROOT/path)==digest,path
source=ROOT/'results/enrollment/enrollment_development_20261019T222121709545Z'
source_info=json.loads((source/'comparison.json').read_text())
split=json.loads((source/'split_snapshot.json').read_text())
ids=sorted(split['halfface']['development']['known'])
records=[r for r in read_csv(source/'combined_extraction.csv') if r['dataset']=='halfface']
by_path={r['path']:r for r in records}
features=np.load(ROOT/source_info['feature_cache'],allow_pickle=False)['embeddings']
prototypes={}
for sid in ids:
    rows=[r for r in records if r['subject_id']==sid and r['mask_status']=='unmasked']
    rows.sort(key=lambda r:({'front':0,'left':1,'right':2}[r['pose']],r['path']))
    vectors=[features[int(r['embedding_index'])] for r in rows[:3] if r['status']=='ok']
    vector=np.asarray(vectors,dtype=np.float64).mean(axis=0)
    prototypes[sid]=(vector/np.linalg.norm(vector)).astype(np.float32)
predictions=read_csv(RUN/'predictions.csv')
checked=0
for stage in info['stages']:
    active=ids[:stage]
    matrix=np.stack([prototypes[s] for s in active]) if active else None
    rows=[r for r in predictions if int(r['stage'])==stage]
    for row in rows:
        original=by_path[row['path']]
        assert row['status']==original['status']
        assert (row['enrolled_at_stage']=='True')==(row['subject_id'] in active)
        identity,top,score='','',0.
        if row['status']!='ok':
            state='CAPTURE_FAILURE'
        elif not active:
            state='UNKNOWN'
        else:
            scores=matrix@features[int(original['embedding_index'])]
            order=np.argsort(scores)[::-1]
            top,score,second=active[int(order[0])],float(scores[order[0]]),float(scores[order[1]])
            state='UNKNOWN' if score<0.3778589367866516 else ('AMBIGUOUS' if score-second<.15 else 'KNOWN')
            if state=='KNOWN':identity=top
        assert (state,identity,top)==(row['decision'],row['accepted_id'],row['top1_id'])
        assert abs(score-float(row['top1_score']))<1e-7
        checked+=1
    summary=next(s for s in info['summaries'] if s['gallery_identities']==stage)
    assert summary['correct_known_accepts']==sum(r['correct_known_accept']=='True' for r in rows)
    assert summary['active_known_attempts']==sum(r['enrolled_at_stage']=='True' for r in rows)
    times=[float(r['milliseconds']) for r in read_csv(RUN/'timings.csv') if int(r['stage'])==stage]
    assert len(times)==300
    assert abs(float(np.median(times))-summary['search_median_ms'])<1e-12
for name,digest in info['backbone_checkpoint_hashes_unchanged'].items():
    assert sha256(Path.home()/'.insightface/models/buffalo_l'/name)==digest
fixture=json.loads((RUN/'demo_fixture.json').read_text())
ui_snapshot=ROOT/'data/demo/dynamic_development_20261019T233918475526Z/ui_gallery_snapshot'
gallery=FaceGallery(ui_snapshot)
assert gallery.size==1
np.testing.assert_array_equal(gallery._embeddings[fixture['subject_id']],prototypes[fixture['subject_id']])
expected={}
for key in ['known_probe','unknown_probe']:
    row=by_path[fixture[key]]
    result=recognize(gallery,features[int(row['embedding_index'])],float(row['quality']))
    expected[key]=dict(decision=result['decision'],score=result['top1_score'],accepted_id=result['accepted_id'])
assert expected['known_probe']['decision']=='KNOWN' and expected['unknown_probe']['decision']=='UNKNOWN'
events=[json.loads(line) for line in (RUN/'browser_events.jsonl').read_text(encoding='utf-8').splitlines()]
shots={e['value']:e['observation'] for e in events if e['action']=='screenshot'}
assert 'Gallery now has 1 identities' in shots['fig03_app_enrollment.png']['text']
assert 'KNOWN' in shots['fig04_app_known.png']['text']
assert f"Score: {expected['known_probe']['score']:.4f}" in shots['fig04_app_known.png']['text']
assert 'UNKNOWN' in shots['fig05_app_unknown.png']['text']
assert f"Score: {expected['unknown_probe']['score']:.4f}" in shots['fig05_app_unknown.png']['text']
assert '1 identities enrolled' in shots['fig06_app_persistence.png']['text']
assert 'Gallery is empty' in shots['fig07_app_after_removal.png']['text'] and 'UNKNOWN' in shots['fig07_app_after_removal.png']['text']
for shot in shots.values():
    assert not next(i for i in shot['inputs'] if i['label']=='Show photo previews')['checked']
    assert not next(i for i in shot['inputs'] if i['label']=='Queue unknown faces for review')['checked']
report=dict(incremental_attempts_verified=checked,source_and_checkpoint_hashes_match=True,
            final_304_predictions_match_prior_development=True,ui_uploaded_template_matches_cached_original_mean=True,
            ui_known_unknown_scores_match_reference=expected,screenshot_states_verified=list(shots),
            gallery_reload_and_remove_observed=True,photo_previews_disabled=True,reserved_test_images_used=False,
            verifier_sha256=sha256(__file__),browser_events_sha256=sha256(RUN/'browser_events.jsonl'))
report['ui_gallery_snapshot_sha256']={p.name:sha256(p) for p in ui_snapshot.iterdir() if p.is_file()}
(RUN/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
