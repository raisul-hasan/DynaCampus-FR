"""Audit example selection, source scores, private images and real rehearsal logs."""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_arcface_baseline import read_csv,sha256


def main():
    pointer=json.loads((ROOT/'paper_assets/development/defense_v1/source_run.json').read_text())
    run=ROOT/pointer['run']
    manifest=json.loads((run/'case_manifest.json').read_text())
    assert sha256(run/'case_manifest.json')==pointer['case_manifest_sha256']
    assert not manifest['final_test_used'] and not manifest['new_model_inference']
    for path,digest in manifest['input_sha256'].items():
        assert sha256(ROOT/path)==digest,path
    assert sha256(ROOT/'scripts/prepare_defense_cases.py')==manifest['script_sha256']
    ledger=read_csv(run/'candidate_ledger.csv')
    assert len(ledger)==manifest['candidate_count']
    assert len({(r['scope'],r['model'],r['category'],r['path']) for r in ledger})==len(ledger)
    actual=Counter((r['scope'],r['model'],r['category']) for r in ledger)
    logged={(r['scope'],r['model'],r['category']):int(r['count']) for r in read_csv(run/'candidate_counts.csv')}
    assert actual==logged
    fixed=read_csv(ROOT/'results/enrollment_pooling/pooling_development_20261020T181729489649Z/predictions.csv')
    backbones=read_csv(ROOT/'results/backbones/backbones_development_20261020T200004286474Z/predictions.csv')
    split=json.loads((ROOT/'configs/splits/arcface_baseline_v1.json').read_text())
    allowed=set(sum(split['halfface']['development'].values(),[]))
    selected=[]
    for case in manifest['cases']:
        assert case['subject_id'] in allowed
        pool=[r['path'] for r in ledger if (r['scope'],r['model'],r['category'])==(case['scope'],case['model'],case['category'])]
        assert case['path']==min(pool)
        assert case['path'] not in case['enrollment_paths']
        for path,digest in case['source_sha256'].items():
            assert sha256(ROOT/path)==digest,path
        for name,strategy in [('one','one_photo'),('few','up_to_three_photos')]:
            if case['scope']=='fixed_rule_sensitivity':
                method='one_photo' if name=='one' else 'uniform_mean'
                original=next(r for r in fixed if r['path']==case['path'] and r['method']==method and r['rule']=='frozen_margin_sensitivity')
                assert case[f'{name}_threshold']==float(original['threshold'])
                assert case[f'{name}_margin']==float(original['margin_threshold'])
            else:
                original=next(r for r in backbones if r['path']==case['path'] and r['model']==case['model'] and r['strategy']==strategy and r['family']=='score_margin')
            assert original==case[name]
        a,b=case['one'],case['few']
        if case['category']=='helped_known':
            assert a['correct_known_accept']=='False' and b['correct_known_accept']=='True'
        elif case['category']=='harmed_known':
            assert a['correct_known_accept']=='True' and b['correct_known_accept']=='False'
        elif case['category']=='unknown_false_accept':
            assert b['truth_status']=='unknown' and b['decision']=='KNOWN' and b['accepted_id']
        elif case['category']=='ambiguous_unknown':
            assert b['decision']=='AMBIGUOUS' and not b['accepted_id']
        elif case['category']=='capture_failure':
            assert b['status']!='ok' and not b['top1_score']
        selected.append(case['case_id'])
    assert len(selected)==len(set(selected))==12
    fixture=json.loads((run/'demo_fixture.json').read_text())
    assert set(fixture['seed_ids'])<=allowed and len(fixture['seed_ids'])==54
    assert fixture['subject_id'] not in fixture['seed_ids']
    assert fixture['expected_before']['decision']=='UNKNOWN'
    assert fixture['expected_after']['accepted_id']==fixture['subject_id']
    assert fixture['expected_ambiguous']['decision']=='AMBIGUOUS'
    rehearsal=json.loads((run/'backend_rehearsal.json').read_text())
    session=ROOT/rehearsal['session_directory']
    assert rehearsal['verification']=='passed' and sha256(session/'events.jsonl')==rehearsal['event_log_sha256']
    for path,digest in rehearsal['source_sha256'].items():
        assert sha256(ROOT/path)==digest,path
    events=[json.loads(line) for line in (session/'events.jsonl').read_text().splitlines()]
    assert len(events)==11 and [e.get('decision') for e in events[1:]]==rehearsal['state_sequence']
    assert [e['gallery_size'] for e in events[1:]]==[54,55,55,55,55,55,55,54,54,54]
    assert events[5]['top1_id']!=events[5]['top2_id']
    assert events[5]['top1_score']>=fixture['score_threshold'] and events[5]['margin']<fixture['margin_threshold']
    assert all(e['source']=='fresh original-image inference' and e['feature_cache_max_difference']==0. for e in events[1:] if e.get('face_count')==1)
    for path,digest in rehearsal['checkpoint_sha256'].items():
        assert sha256(Path(path))==digest
    app=json.loads((run/'app_checks.json').read_text())
    assert app['verification']=='passed' and app['real_walkthrough_steps']==10 and app['case_views_checked']==12
    for path,digest in app['source_sha256'].items():
        assert sha256(ROOT/path)==digest,path
    freeze=json.loads((ROOT/'results/final/heldout_20261019T231512869782Z/freeze.json').read_text())
    for path,digest in freeze['source_sha256'].items():
        assert sha256(ROOT/path)==digest,path
    receipt=dict(verification='passed',selected_examples=12,candidate_records=len(ledger),deterministic_first_path_selection=True,
        original_image_hashes_match=True,case_values_match_verified_research=True,
        actual_real_image_backend_steps=10,app_test_steps=10,fresh_model_features_match_cache=True,
        genuine_multi_identity_ambiguity=True,frozen_final_source_hashes_unchanged=True,
        fresh_browser_capture=False,live_screen_recording=False,
        verifier_sha256=sha256(Path(__file__)))
    (run/'verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
