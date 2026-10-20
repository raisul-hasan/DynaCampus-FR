"""Engineering evidence for dynamic enrollment using real development caches only."""
import json
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_arcface_baseline import read_csv,sha256,write_csv
from scripts.compare_enrollment import gallery_candidates,unit_mean
from src.gallery import FaceGallery
from src.demo_workflow import process_image,recognize
from src.detection import SCRFDDetector
from src.alignment import FaceAligner
from src.recognition import FaceEmbedder
from src.quality import FaceQualityEstimator


def main():
    source = ROOT/'results/enrollment/enrollment_development_20261019T222121709545Z'
    decision_source = ROOT/'results/decisions/decisions_development_20261019T222927022126Z'
    info = json.loads((source/'comparison.json').read_text())
    split = json.loads((source/'split_snapshot.json').read_text())
    known_ids = sorted(split['halfface']['development']['known'])
    reserved = set(sum(split['halfface']['test_reserved'].values(),[]))
    records = [r for r in read_csv(source/'combined_extraction.csv') if r['dataset'] == 'halfface']
    assert not reserved & {r['subject_id'] for r in records}
    features = np.load(ROOT/info['feature_cache'],allow_pickle=False)['embeddings']
    candidates = gallery_candidates(records,'halfface',known_ids,'up_to_three_photos')
    probes = [r for r in records if r['truth_status'] == 'unknown' or r['mask_status'] == 'masked']
    run_id = 'dynamic_development_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = ROOT/'results/demo'/run_id
    destination.mkdir(parents=True)
    gallery_path = ROOT/'data/demo'/run_id/'gallery'
    gallery = FaceGallery(gallery_path)
    weights_dir = Path.home()/'.insightface/models/buffalo_l'
    checkpoints = {name:sha256(weights_dir/name) for name in ['w600k_r50.onnx','det_10g.onnx']}
    predictions,summaries,timings,operations = [],[],[],[]
    stages = [0,10,20,30,40,55]
    enrolled = set()
    for count in stages:
        for sid in known_ids[:count]:
            if sid in enrolled:
                continue
            before = {s:e.copy() for s,e in gallery._embeddings.items()}
            selected = [r for r in candidates[sid] if r['status'] == 'ok']
            start = time.perf_counter_ns()
            assert gallery.enroll_multi(sid,sid,[features[int(r['embedding_index'])] for r in selected],
                                       [float(r['quality']) for r in selected])
            elapsed = (time.perf_counter_ns()-start)/1e6
            assert all(np.array_equal(before[s],gallery._embeddings[s]) for s in before)
            np.testing.assert_array_equal(gallery._embeddings[sid],unit_mean([features[int(r['embedding_index'])] for r in selected]))
            enrolled.add(sid)
            operations.append(dict(operation='insert',subject_id=sid,active_gallery_size=gallery.size,milliseconds=elapsed,
                                   prior_templates_unchanged=True))
        output = []
        for row in probes:
            if row['status'] == 'ok':
                result = recognize(gallery,features[int(row['embedding_index'])],float(row['quality']))
                state,accepted = result['decision'],result['accepted_id'] or ''
                score,top = result['top1_score'],result['top1_id'] or ''
            else:
                state,accepted,score,top = 'CAPTURE_FAILURE','',0.,''
            active = row['subject_id'] in enrolled
            output.append(dict(stage=count,path=row['path'],subject_id=row['subject_id'],original_truth=row['truth_status'],
                enrolled_at_stage=active,status=row['status'],decision=state,accepted_id=accepted,top1_id=top,top1_score=score,
                correct_known_accept=active and state == 'KNOWN' and accepted == row['subject_id']))
        active = [r for r in output if r['enrolled_at_stage']]
        unknown = [r for r in output if not r['enrolled_at_stage']]
        successful = [r for r in probes if r['status'] == 'ok']
        for row in successful[:20]:
            gallery.search(features[int(row['embedding_index'])])
        stage_times = []
        for index in range(300):
            row = successful[index%len(successful)]
            start = time.perf_counter_ns()
            gallery.search(features[int(row['embedding_index'])])
            ms = (time.perf_counter_ns()-start)/1e6
            stage_times.append(ms)
            timings.append(dict(stage=count,operation='search_only',repetition=index,milliseconds=ms))
        summaries.append(dict(gallery_identities=count,active_known_attempts=len(active),
            correct_known_accepts=sum(r['correct_known_accept'] for r in active),
            not_enrolled_attempts=len(unknown),false_known_not_enrolled=sum(r['decision'] == 'KNOWN' for r in unknown),
            search_median_ms=float(np.median(stage_times)),search_p95_ms=float(np.percentile(stage_times,95))))
        predictions += output
    reference = {r['path']:r for r in read_csv(decision_source/'predictions.csv') if r['family'] == 'score_margin'}
    final = {r['path']:r for r in predictions if r['stage'] == 55}
    assert final.keys() == reference.keys()
    for path,row in final.items():
        assert (row['decision'],row['accepted_id']) == (reference[path]['decision'],reference[path]['accepted_id'])
        if row['status'] == 'ok':
            assert row['top1_id'] == reference[path]['top1_id']
            assert abs(row['top1_score']-float(reference[path]['top1_score'])) < 1e-7
    reloaded = FaceGallery(gallery_path)
    assert reloaded.size == 55 and all(np.array_equal(gallery._embeddings[s],reloaded._embeddings[s]) for s in enrolled)
    sample = next(r for r in probes if r['status'] == 'ok')
    assert gallery.search(features[int(sample['embedding_index'])]) == reloaded.search(features[int(sample['embedding_index'])])
    remove_id = known_ids[0]
    saved = gallery._embeddings[remove_id].copy()
    assert gallery.remove(remove_id) and FaceGallery(gallery_path).size == 54
    assert gallery.enroll(remove_id,remove_id,saved) and gallery.size == 55
    # Real-image equivalence checks and a disclosed successful development demo fixture.
    eligible = sorted({r['subject_id'] for r in probes if r['status'] == 'ok' and r['truth_status'] == 'known'
                       and final[r['path']]['correct_known_accept'] and len(candidates[r['subject_id']]) == 3})
    subject = eligible[0]
    enrollment = candidates[subject]
    known = next(r for r in probes if r['subject_id'] == subject and r['status'] == 'ok' and final[r['path']]['correct_known_accept'])
    unknown = next(r for r in probes if r['truth_status'] == 'unknown' and r['status'] == 'ok' and final[r['path']]['decision'] == 'UNKNOWN')
    detector,aligner,embedder,quality = SCRFDDetector(),FaceAligner(),FaceEmbedder(),FaceQualityEstimator()
    checks = []
    for row in enrollment+[known,unknown]:
        decoded = cv2.imread(str(ROOT/row['path']))
        faces = process_image(decoded,detector,aligner,embedder,quality,face_crop=True)
        assert len(faces) == 1
        delta = float(np.max(np.abs(faces[0]['embedding']-features[int(row['embedding_index'])])))
        assert delta < 1e-6
        checks.append(dict(path=row['path'],subject_id=row['subject_id'],max_embedding_difference=delta,
                           quality_matches=float(row['quality']) == faces[0]['quality']['overall_quality']))
        assert checks[-1]['quality_matches']
    assert {n:sha256(weights_dir/n) for n in checkpoints} == checkpoints
    fixture = dict(subject_id=subject,display_name='Development Example',enrollment_paths=[r['path'] for r in enrollment],
                   known_probe=known['path'],unknown_probe=unknown['path'],
                   note='Selected successful development example for engineering demonstration, not an accuracy sample.')
    (destination/'demo_fixture.json').write_text(json.dumps(fixture,indent=2),encoding='utf-8')
    write_csv(destination/'stage_summary.csv',summaries)
    write_csv(destination/'predictions.csv',predictions)
    write_csv(destination/'timings.csv',timings)
    write_csv(destination/'insertions.csv',operations)
    write_csv(destination/'image_equivalence.csv',checks)
    result = dict(run_directory=destination.relative_to(ROOT).as_posix(),scope='development_engineering_only',stages=stages,
        summaries=summaries,insertions=len(operations),old_templates_unchanged=True,persistence_reload_exact=True,
        remove_and_reinsert_verified=True,full_gallery_predictions_match_development_reference=len(final),
        actual_image_checks=len(checks),backbone_checkpoint_hashes_unchanged=checkpoints,
        training_performed=False,reserved_test_images_used=False,probe_method_attempts=len(predictions),
        source_sha256={p:sha256(ROOT/p) for p in ['src/gallery/gallery.py','src/demo_workflow.py','scripts/validate_dynamic_enrollment.py','app.py']},
        input_sha256={p.relative_to(ROOT).as_posix():sha256(p) for p in [source/'combined_extraction.csv',ROOT/info['feature_cache'],decision_source/'predictions.csv']},
        timings_scope='Search class call includes matrix construction; excludes detection, embedding, UI and video. Insert timings include disk persistence.')
    (destination/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('COMPLETE',destination)


if __name__ == '__main__':
    main()
