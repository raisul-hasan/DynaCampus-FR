"""Independent reconstruction of augmented enrollment, growth and runtime tables.

No experiment transform, pooling, scoring, calibration or decision helpers imported.
Replays all 550 variant forward passes with real models and checks every decision.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def read(path):
    with path.open(encoding='utf-8',newline='') as stream: return list(csv.DictReader(stream))


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a,b,tol=2e-6):
    if not np.allclose(a,b,atol=tol,rtol=0): raise AssertionError(f'{a} != {b}')


def normalized_mean(vectors):
    mean=np.asarray(vectors,dtype=np.float64).mean(axis=0)
    return (mean/np.linalg.norm(mean)).astype(np.float32)


def decision(row,tau,margin):
    if row['status']!='ok': return 'CAPTURE_FAILURE'
    if float(row['top1_score'])<tau: return 'UNKNOWN'
    if float(row['top1_score'])-float(row['top2_score'])<margin: return 'AMBIGUOUS'
    return 'KNOWN'


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('run',type=Path)
    args=parser.parse_args(); dest=(ROOT/args.run).resolve()
    cfg=json.loads((dest/'config.json').read_text()); prep=json.loads((dest/'preparation.json').read_text())
    complete=json.loads((dest/'completion.json').read_text())
    for mapping,prefix in [(prep['source_sha256'],ROOT),(prep['input_sha256'],ROOT),
                           (complete['output_sha256'],dest),(prep['prepared_sha256'],dest)]:
        for name,digest in mapping.items(): assert sha(prefix/name)==digest,name
    cache=ROOT/complete['cache']
    for name,digest in complete['cache_sha256'].items(): assert sha(cache/name)==digest,name
    info=json.loads((ROOT/cfg['backbone_run']/'comparison.json').read_text())
    split=json.loads((dest/'split.json').read_text())['halfface']
    known,unknown=split['development']['known'],split['development']['unknown']
    reserved=set(sum(split['test_reserved'].values(),[]))
    rows=read(dest/'inputs.csv'); lookup={r['path']:r for r in rows}
    assert len(rows)==len(lookup)
    assert not {r['subject_id'] for r in rows}&reserved
    assert all(r['subject_id'] in known+unknown and r['dataset']=='halfface' for r in rows)
    for r in rows: assert sha(ROOT/r['path'])==r['source_sha256']
    select=json.loads((dest/'selection.json').read_text())
    assert select['core_ids']==known[:10]
    assert select['augmentation_probe_paths']==[r['path'] for r in rows if
        (r['subject_id'] in known and r['mask_status']=='masked') or r['subject_id'] in unknown]
    assert select['growth_probe_paths']==[r['path'] for r in rows if
        (r['subject_id'] in known[:10] and r['mask_status']=='masked') or r['subject_id'] in unknown]
    assert len(select['scenarios'])==7
    for scenario in select['scenarios']:
        expected=known[:10] if scenario['size']==10 else known if scenario['size']==55 else known[:10]+np.random.default_rng(
            scenario['seed']).permutation(known[10:]).tolist()[:15]
        assert scenario['ids']==sorted(expected)
        assert not set(scenario['ids'])&set(unknown)
    for sid in sorted(known):
        candidates=[r for r in rows if r['subject_id']==sid and r['mask_status']=='unmasked']
        candidates.sort(key=lambda r:({'front':0,'left':1,'right':2}[r['pose']],r['path']))
        assert select['one_photo'][sid]==[r['path'] for r in candidates if r['pose']=='front'][:1]
        assert select['few_photos'][sid]==[r['path'] for r in candidates[:3]]
    enrolled={p for ps in select['few_photos'].values() for p in ps}
    assert not enrolled&set(select['augmentation_probe_paths'])
    with np.load(ROOT/info['model']['features'],allow_pickle=False) as archive: base={k:archive[k] for k in cfg['models']}
    with np.load(cache/'augmentation_features.npz',allow_pickle=False) as archive: extra={k:archive[k] for k in cfg['models']}
    with np.load(cache/'templates.npz',allow_pickle=False) as archive: templates={k:archive[k] for k in archive.files}
    logs=read(dest/'augmentation_sources.csv')
    assert len(logs)==55*5
    assert [int(r['variant_index']) for r in logs]==list(range(275))
    from src.recognition import FaceEmbedder
    from src.recognition.adaface import AdaFaceEmbedder
    from src.detection import SCRFDDetector
    from src.alignment import FaceAligner
    cv2.setNumThreads(1); torch.set_num_threads(cfg['cpu_threads']); torch.set_num_interop_threads(1)
    ort.set_default_logger_severity(3)
    detector=SCRFDDetector(confidence_threshold=.5,input_size=(640,640),cpu_threads=cfg['cpu_threads'])
    models={'arcface':FaceEmbedder(cpu_threads=cfg['cpu_threads']),
            'adaface':AdaFaceEmbedder(ROOT/'data/models/adaface','cpu')}
    for m in cfg['models']: assert sha(models[m].model_path)==info['model'][m]['sha256']
    verified_forwards=0
    with threadpool_limits(limits=cfg['cpu_threads']):
        for si,(sid,paths) in enumerate(select['one_photo'].items()):
            path=paths[0]; image=cv2.imread(str(ROOT/path))
            py,px=(round(d*.5) for d in image.shape[:2])
            padded=cv2.copyMakeBorder(image,py,py,px,px,cv2.BORDER_CONSTANT,value=(127,127,127))
            faces=detector.detect(padded); assert len(faces)==1
            crop=FaceAligner().align(padded,faces[0]['kps'])
            assert hashlib.sha256(crop.tobytes()).hexdigest()==lookup[path]['aligned_crop_sha256']
            for row in [r for r in logs if r['subject_id']==sid]:
                name=row['variant']; out=crop.copy()
                if name=='blur_sigma1': out=cv2.GaussianBlur(out,(3,3),1.)
                elif name=='resize56': out=cv2.resize(cv2.resize(out,(56,56),interpolation=cv2.INTER_CUBIC),(112,112),interpolation=cv2.INTER_CUBIC)
                elif name=='jpeg50':
                    ok,data=cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,50]); assert ok
                    out=cv2.imdecode(data,cv2.IMREAD_COLOR)
                else:
                    assert name in ['gamma0.7','gamma1.4']
                    lut=np.round(np.power(np.arange(256,dtype=float)/255.,float(name[5:]))*255).clip(0,255).astype(np.uint8)
                    out=cv2.LUT(out,lut)
                assert hashlib.sha256(out.tobytes()).hexdigest()==row['variant_pixel_sha256']
                vi=int(row['variant_index'])
                for m in cfg['models']:
                    feature=models[m].extract_embedding(out)
                    close(feature,extra[m][vi],tol=1e-5)
                    close(float(row[m+'_original_cosine']),float(feature@base[m][int(lookup[path]['embedding_index'])]))
                    verified_forwards+=1
            if (si+1)%10==0 or si==54: print('Verified real variant replay',si+1,'/55',flush=True)
    ids=sorted(known)
    source_log=read(dest/'template_sources.csv')
    for m in cfg['models']:
        for policy in cfg['policies']:
            for index,sid in enumerate(ids):
                selected=select['few_photos'][sid] if policy=='up_to_three_photos' else select['one_photo'][sid]
                values=[base[m][int(lookup[p]['embedding_index'])] for p in selected if lookup[p]['status']=='ok']
                expected_sources={(p,'original',lookup[p]['status']=='ok') for p in selected}
                if policy=='augmented_one_photo':
                    for r in logs:
                        if r['subject_id']==sid:
                            values.append(extra[m][int(r['variant_index'])])
                            expected_sources.add((r['path'],r['variant'],True))
                actual={(r['path'],r['variant'],r['used']=='True') for r in source_log if
                        r['model']==m and r['policy']==policy and r['subject_id']==sid}
                assert actual==expected_sources
                close(templates[m+'_'+policy][index],normalized_mean(values))
    aug=read(dest/'augmentation_predictions.csv'); growth=read(dest/'growth_predictions.csv')
    thresholds=read(dest/'augmentation_thresholds.csv'); grids=read(dest/'augmentation_grid.csv')
    grid_groups=defaultdict(list)
    for r in grids: grid_groups[(r['model'],r['policy'],r['family'])].append(r)
    grouped=defaultdict(list)
    for r in aug:
        key=(r['model'],r['policy'],r['view'],r['family']); grouped[key].append(r)
    for r in growth:
        key=(r['model'],r['scenario'],r['family']); grouped[key].append(r)
    searches=0
    for key,predictions in grouped.items():
        m=key[0]
        if len(key)==4:
            _,policy,view,family=key; gallery=templates[m+'_'+policy]; gallery_ids=ids
            t=next(r for r in thresholds if (r['model'],r['policy'],r['view'],r['family'])==key)
            expected_probes=select['augmentation_probe_paths']
            if view=='fixed_one_photo':
                expected=next(r for r in info['summaries'] if r['model']==m and r['dataset']=='halfface'
                              and r['strategy']=='one_photo' and r['family']==family)
                for k in ['score_threshold','margin_threshold','quality_threshold']: assert float(t[k])==expected[k]
        else:
            _,scenario,family=key
            composition=next(s for s in select['scenarios'] if s['scenario']==scenario)
            gallery_ids=composition['ids']; gallery=np.stack([templates[m+'_up_to_three_photos'][ids.index(i)] for i in gallery_ids])
            t=next(r for r in info['summaries'] if r['model']==m and r['dataset']=='halfface'
                   and r['strategy']=='up_to_three_photos' and r['family']==family)
            expected_probes=select['growth_probe_paths']
        assert [r['path'] for r in predictions]==expected_probes
        for r in predictions:
            source=lookup[r['path']]
            assert r['status']==source['status'] and r['truth_status']==source['truth_status']
            if r['status']=='ok':
                scores=gallery@base[m][int(source['embedding_index'])]; order=np.argsort(scores)[::-1]
                assert r['top1_id']==gallery_ids[int(order[0])]
                close(float(r['top1_score']),scores[order[0]]); close(float(r['top2_score']),scores[order[1]])
                searches+=1
            state=decision(r,float(t['score_threshold']),float(t['margin_threshold']))
            assert r['decision']==state
            correct=state=='KNOWN' and r['top1_id']==r['subject_id'] and r['truth_status']=='known'
            assert (r['correct_known_accept']=='True')==correct
            assert (r['covered']=='True')==(state in ['KNOWN','UNKNOWN'])
        if len(key)==4 and key[2]=='recalibrated_development':
            valid=np.array([r['status']=='ok' for r in predictions]); is_known=np.array([r['truth_status']=='known' for r in predictions])
            scores=np.array([float(r['top1_score']) if r['status']=='ok' else -2 for r in predictions])
            margins=np.array([float(r['top1_score'])-float(r['top2_score']) if r['status']=='ok' else 0 for r in predictions])
            correct=np.array([r['top1_id']==r['subject_id'] and r['truth_status']=='known' for r in predictions])&valid
            expected_levels=np.unique(np.r_[-1.000001,np.unique(scores[valid]),np.nextafter(np.unique(scores[valid]),np.inf),1.000001])
            grid=grid_groups[(m,policy,family)]
            expected_margins=[0.] if family=='score_only' else cfg['margin_grid']
            assert len(grid)==len(expected_levels)*len(expected_margins)
            observed={(float(r['score_threshold']),float(r['margin_threshold'])) for r in grid}
            assert observed=={(float(tau),float(mm)) for tau in expected_levels for mm in expected_margins}
            best=None
            for g in grid:
                tau,mm=float(g['score_threshold']),float(g['margin_threshold'])
                accepted=valid&(scores>=tau)&(margins>=mm)
                covered=valid&((scores<tau)|(margins>=mm))
                ta=int((accepted&correct).sum()); fa=int((accepted&~is_known).sum())
                wrong=int((accepted&is_known&~correct).sum())
                kc=(covered&is_known).sum()/(valid&is_known).sum(); uc=(covered&~is_known).sum()/(valid&~is_known).sum()
                feasible=(fa/(valid&~is_known).sum()<=.01+1e-12 and kc>=.85 and uc>=.85)
                assert int(g['correct_known_accepts'])==ta and int(g['false_known_unknowns'])==fa
                assert int(g['wrong_known_identity_accepts'])==wrong and (g['feasible']=='True')==feasible
                close(float(g['known_valid_coverage']),kc); close(float(g['unknown_valid_coverage']),uc)
                if feasible:
                    rank=(-ta,fa,wrong,-int(covered.sum()),0.,mm,-tau)
                    if best is None or rank<best[0]: best=(rank,tau,mm)
            assert float(t['score_threshold'])==best[1] and float(t['margin_threshold'])==best[2]
    summaries=read(dest/'augmentation_summary.csv')+read(dest/'growth_summary.csv')
    for s in summaries:
        key=(s['model'],s['policy'],s['view'],s['family']) if 'policy' in s else (s['model'],s['scenario'],s['family'])
        data=grouped[key]; kn=[r for r in data if r['truth_status']=='known']; un=[r for r in data if r['truth_status']=='unknown']
        ok=[r for r in data if r['status']=='ok']; uok=[r for r in un if r['status']=='ok']
        covered=[r for r in data if r['covered']=='True']
        ta=sum(r['correct_known_accept']=='True' for r in kn); fa=sum(r['decision']=='KNOWN' for r in un)
        expected={'known_probe_attempts':len(kn),'unknown_probe_attempts':len(un),'known_with_embedding':sum(r['status']=='ok' for r in kn),
            'unknown_with_embedding':len(uok),'correct_known_accepts':ta,'false_known_unknowns':fa,
            'correct_known_accept_end_to_end':ta/len(kn),'fpir_conditional':fa/len(uok),
            'wrong_known_identity_accepts':sum(r['decision']=='KNOWN' and r['correct_known_accept']!='True' for r in kn),
            'valid_decision_coverage':len(covered)/len(ok),'end_to_end_decision_coverage':len(covered)/len(data),
            'rank1_correct':sum(r['rank1_correct']=='True' for r in kn)}
        for k,v in expected.items(): close(float(s[k]),v)
    for e in read(dest/'paired_effects.csv'):
        a=[r for r in grouped[(e['model'],'one_photo',e['view'],e['family'])] if r['truth_status']=='known']
        b={r['path']:r for r in grouped[(e['model'],e['policy'],e['view'],e['family'])]}
        labels=sorted(known)
        count=np.array([sum(r['subject_id']==sid for r in a) for sid in labels])
        d=np.array([sum((int(b[r['path']]['correct_known_accept']=='True')-int(r['correct_known_accept']=='True'))
                        for r in a if r['subject_id']==sid) for sid in labels])
        draws=np.random.default_rng(cfg['seed']).integers(0,55,(cfg['bootstrap_samples'],55))
        low,high=np.percentile(d[draws].sum(axis=1)/count[draws].sum(axis=1)*100,[2.5,97.5])
        close(float(e['delta_percentage_points']),100*d.sum()/count.sum())
        close(float(e['ci95_low_percentage_points']),low); close(float(e['ci95_high_percentage_points']),high)
    # A nested exact gallery can never reduce an unknown maximum score.
    gp={(r['model'],r['scenario'],r['family'],r['path']):r for r in growth}
    for m in cfg['models']:
        for path in select['growth_probe_paths']:
            if lookup[path]['status']!='ok': continue
            lo=float(gp[(m,'g10','score_only',path)]['top1_score']); hi=float(gp[(m,'g55','score_only',path)]['top1_score'])
            for scenario in select['scenarios']:
                value=float(gp[(m,scenario['scenario'],'score_only',path)]['top1_score'])
                assert lo-2e-6<=value<=hi+2e-6
    raw=read(dest/'runtime_samples.csv'); bygroup=defaultdict(list)
    for r in raw:
        value=float(r['milliseconds']); assert np.isfinite(value) and value>=0
        bygroup[(r['model'],r['phase'],r['stage'],r['gallery_size'],r['status'])].append(value)
    for r in read(dest/'runtime_summary.csv'):
        values=bygroup[(r['model'],r['phase'],r['stage'],r['gallery_size'],r['status'])]
        assert int(r['samples'])==len(values)
        close(float(r['median_ms']),np.median(values)); close(float(r['p95_ms']),np.percentile(values,95))
    image_timings=read(dest/'runtime_images.csv')
    for r in image_timings:
        assert float(r['wall_ms'])+1e-6>=float(r['sum_component_ms'])
        if r['status']=='ok': assert float(r['cache_max_difference'])<=1e-5
        assert r['status']==lookup[r['path']]['status']
    runtime_search=read(dest/'runtime_search.csv')
    for r in runtime_search:
        m=r['model']; n=int(r['gallery_size'])
        scenario=next(s for s in select['scenarios'] if s['size']==n and (n!=25 or s['seed']==42))
        gallery=np.stack([templates[m+'_up_to_three_photos'][ids.index(sid)] for sid in scenario['ids']])
        vector=base[m][int(lookup[r['path']]['embedding_index'])]
        scores=gallery@vector; order=np.argsort(scores)[::-1]
        assert r['top1_id']==scenario['ids'][int(order[0])]
        close(float(r['top1_score']),scores[order[0]]); close(float(r['top2_score']),scores[order[1]])
        t=next(x for x in info['summaries'] if x['model']==m and x['dataset']=='halfface'
               and x['strategy']=='up_to_three_photos' and x['family']=='score_margin')
        assert r['decision']==decision(dict(r,status='ok'),t['score_threshold'],t['margin_threshold'])
    report={'verified_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
        'verifier_sha256':sha(Path(__file__)),'completion_sha256':sha(dest/'completion.json'),
        'real_variant_forward_passes_replayed':verified_forwards,'predictions_checked':len(aug)+len(growth),
        'cosine_searches_checked':searches,'calibration_candidates_checked':len(grids),
        'summary_rows_checked':len(summaries),'runtime_searches_checked':len(runtime_search),
        'runtime_image_calls_checked':len(image_timings),'runtime_stage_samples_checked':len(raw),
        'reserved_id_overlap':0,'final_test_inference':False,'scope':cfg['scope']}
    (dest/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
