"""Prepare and run augmentation, fixed-probe gallery growth and runtime studies."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import onnxruntime as ort
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_arcface_baseline import read_csv,write_csv,sha256
from scripts.compare_enrollment import gallery_candidates,unit_mean,paired_bootstrap,reject_overlap
from scripts.compare_decisions import calibrate,apply_rule,summarize
from src.evaluation.enrollment_extensions import (enrollment_variant,gallery_scenarios,
    check_development_ids,probe_rows,score_rows)
from src.recognition import FaceEmbedder
from src.recognition.adaface import AdaFaceEmbedder
from src.detection import SCRFDDetector
from src.alignment import FaceAligner
from src.quality import FaceQualityEstimator


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def prepare(config):
    cfg=json.loads(config.read_text())
    if cfg['final_test_evaluation'] or cfg['training_performed']:
        raise ValueError('Only frozen-backbone development experiments supported')
    source=ROOT/cfg['backbone_run']
    info=json.loads((source/'comparison.json').read_text())
    split=json.loads((source/'split_snapshot.json').read_text())
    rows=[r for r in read_csv(source/'extraction.csv') if r['dataset']=='halfface']
    known,unknown=split['halfface']['development'].values()
    reserved=sum(split['halfface']['test_reserved'].values(),[])
    check_development_ids(rows,known,unknown,reserved)
    if info['final_test_evaluated'] or info['training_performed']:
        raise ValueError('Source evidence scope differs')
    core,scenarios=gallery_scenarios(known,cfg['growth_background_seeds'])
    gallery_one=gallery_candidates(rows,'halfface',known,'one_photo')
    gallery_few=gallery_candidates(rows,'halfface',known,'up_to_three_photos')
    probes=probe_rows(rows,known,unknown)
    growth=probe_rows(rows,core,unknown)
    all_enrollment=[r for v in gallery_few.values() for r in v]
    reject_overlap(all_enrollment,probes)
    if any(any(r['status']!='ok' for r in selected) for selected in gallery_one.values()):
        raise ValueError('One-photo enrollment failure needs a separate protocol')
    dest=ROOT/'results/enrollment_extensions'/('extensions_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    dest.mkdir(parents=True,exist_ok=False)
    save(dest/'config.json',cfg); save(dest/'split.json',split)
    write_csv(dest/'inputs.csv',rows)
    save(dest/'selection.json',{'core_ids':core,'scenarios':scenarios,
        'one_photo':{k:[r['path'] for r in v] for k,v in gallery_one.items()},
        'few_photos':{k:[r['path'] for r in v] for k,v in gallery_few.items()},
        'augmentation_probe_paths':[r['path'] for r in probes],
        'growth_probe_paths':[r['path'] for r in growth]})
    paths=[source/'comparison.json',source/'extraction.csv',source/'split_snapshot.json',ROOT/info['model']['features']]
    inputs={p.relative_to(ROOT).as_posix():sha256(p) for p in paths}
    for r in rows:
        if sha256(ROOT/r['path'])!=r['source_sha256']:
            raise ValueError('Source image changed')
    sources=['scripts/run_enrollment_extensions.py','scripts/measure_extension_runtime.py',
        'src/evaluation/enrollment_extensions.py','scripts/compare_decisions.py',
        'scripts/compare_enrollment.py','src/recognition/embedder.py','src/recognition/adaface.py',
        'src/detection/detector.py','src/alignment/aligner.py','src/quality/estimator.py',
        'src/open_set/decision.py','src/gallery/gallery.py']
    save(dest/'preparation.json',{'prepared_utc':datetime.now(timezone.utc).isoformat(),
        'scope':cfg['scope'],'source_sha256':{p:sha256(ROOT/p) for p in sources},'input_sha256':inputs,
        'prepared_sha256':{p.name:sha256(p) for p in dest.iterdir()},
        'counts':{'development_known':len(known),'development_unknown':len(unknown),
            'augmentation_known_probes':sum(r['truth_status']=='known' for r in probes),
            'augmentation_unknown_probes':sum(r['truth_status']=='unknown' for r in probes),
            'growth_known_probes':sum(r['truth_status']=='known' for r in growth),
            'growth_unknown_probes':sum(r['truth_status']=='unknown' for r in growth)}})
    print('Prepared:',dest.relative_to(ROOT),flush=True)


def load_models(cfg,info):
    cv2.setNumThreads(1); ort.set_default_logger_severity(3)
    torch.set_num_threads(cfg['cpu_threads']); torch.set_num_interop_threads(1)
    detector=SCRFDDetector(confidence_threshold=.5,input_size=(640,640),
                           providers=['CPUExecutionProvider'],cpu_threads=cfg['cpu_threads'])
    models={'arcface':FaceEmbedder(cpu_threads=cfg['cpu_threads']),
            'adaface':AdaFaceEmbedder(ROOT/'data/models/adaface','cpu')}
    for model in cfg['models']:
        if sha256(models[model].model_path)!=info['model'][model]['sha256']:
            raise ValueError('Checkpoint changed: '+model)
    if sha256(detector.model_path)!=info['model']['detector']['sha256']:
        raise ValueError('Detector changed')
    for _ in range(cfg['warmup_calls']):
        detector.detect(np.zeros((640,640,3),np.uint8))
        for model in models.values(): model.extract_embedding(np.zeros((112,112,3),np.uint8))
    return detector,models


def extract_variants(dest,cfg,rows,base,detector,models,selection):
    by_path={r['path']:r for r in rows}
    aligner=FaceAligner(); vectors={m:[] for m in cfg['models']}; log=[]
    cache=ROOT/'data/embeddings'/dest.name
    cache.mkdir(parents=True,exist_ok=False)
    sample_dir=ROOT/'data/defense'/dest.name
    sample_dir.mkdir(parents=True,exist_ok=True)
    for index,(sid,paths) in enumerate(selection['one_photo'].items()):
        path=paths[0]; record=by_path[path]
        image=cv2.imread(str(ROOT/path))
        if image is None: raise ValueError('Original cannot be decoded')
        py,px=(round(d*.5) for d in image.shape[:2])
        padded=cv2.copyMakeBorder(image,py,py,px,px,cv2.BORDER_CONSTANT,value=(127,127,127))
        faces=detector.detect(padded)
        if len(faces)!=1 or faces[0]['kps'] is None: raise ValueError('Original enrollment detection changed')
        crop=aligner.align(padded,faces[0]['kps'])
        if hashlib.sha256(crop.tobytes()).hexdigest()!=record['aligned_crop_sha256']:
            raise ValueError('Original crop differs from paired backbone study')
        for model in cfg['models']:
            current=models[model].extract_embedding(crop)
            error=float(np.max(np.abs(current-base[model][int(record['embedding_index'])])))
            if error>1e-5: raise ValueError('Original feature cache mismatch')
        example=[crop]
        for variant in cfg['variants']:
            image_variant=enrollment_variant(crop,variant); example.append(image_variant)
            row={'subject_id':sid,'path':path,'variant':variant,'variant_index':len(log),
                 'original_crop_sha256':record['aligned_crop_sha256'],
                 'variant_pixel_sha256':hashlib.sha256(image_variant.tobytes()).hexdigest()}
            for model in cfg['models']:
                feature=models[model].extract_embedding(image_variant)
                vectors[model].append(feature)
                row[model+'_original_cosine']=float(feature@base[model][int(record['embedding_index'])])
            log.append(row)
        # Inspection grid stays in ignored data/, outside public evidence.
        if index<3: cv2.imwrite(str(sample_dir/(sid+'_variant_grid.png')),np.hstack(example))
        if (index+1)%10==0 or index+1==len(selection['one_photo']):
            print(f'Enrollment variants: {index+1}/55 identities',flush=True)
    matrices={m:np.stack(v) for m,v in vectors.items()}
    np.savez_compressed(cache/'augmentation_features.npz',**matrices)
    write_csv(dest/'augmentation_sources.csv',log)
    return matrices,log,cache


def build_templates(cfg,rows,base,extra,log,selection,cache):
    by_path={r['path']:r for r in rows}; ids=sorted(selection['one_photo']); output={}; sources=[]
    for model in cfg['models']:
        for policy in cfg['policies']:
            templates=[]
            for sid in ids:
                paths=selection['few_photos'][sid] if policy=='up_to_three_photos' else selection['one_photo'][sid]
                usable=[by_path[p] for p in paths if by_path[p]['status']=='ok']
                vectors=[base[model][int(r['embedding_index'])] for r in usable]
                for row in [by_path[p] for p in paths]:
                    sources.append({'model':model,'policy':policy,'subject_id':sid,'path':row['path'],
                        'variant':'original','status':row['status'],'used':row['status']=='ok'})
                if policy=='augmented_one_photo':
                    for row in log:
                        if row['subject_id']==sid:
                            vectors.append(extra[model][int(row['variant_index'])])
                            sources.append({'model':model,'policy':policy,'subject_id':sid,'path':row['path'],
                                'variant':row['variant'],'status':'ok','used':True})
                templates.append(unit_mean(vectors))
            output[model+'_'+policy]=np.stack(templates)
    np.savez_compressed(cache/'templates.npz',**output)
    return ids,output,sources


def previous_thresholds(info,model,strategy,family):
    return next(r for r in info['summaries'] if r['model']==model and r['dataset']=='halfface'
                and r['strategy']==strategy and r['family']==family)


def experiments(dest,cfg,rows,base,templates,ids,selection,info):
    by_path={r['path']:r for r in rows}
    probes=[by_path[p] for p in selection['augmentation_probe_paths']]
    predictions=[]; summary=[]; thresholds=[]; gridlog=[]; effects=[]; paired={}
    for model in cfg['models']:
        for policy in cfg['policies']:
            scored=score_rows(probes,base[model],ids,templates[model+'_'+policy])
            for family in cfg['decision_families']:
                for view in cfg['threshold_views']:
                    if view=='fixed_one_photo':
                        threshold=previous_thresholds(info,model,'one_photo',family)
                    else:
                        threshold,grid=calibrate(scored,family,cfg)
                        gridlog.extend(dict(model=model,policy=policy,view=view,**r) for r in grid)
                    thresholds.append({'model':model,'policy':policy,'view':view,'family':family,
                        **{k:threshold[k] for k in ['score_threshold','margin_threshold','quality_threshold']}})
                    out=apply_rule(scored,threshold,family)
                    row=summarize(out,threshold,family)
                    summary.append(dict(model=model,policy=policy,view=view,**row,
                                        rank1_correct=sum(r['rank1_correct'] for r in out)))
                    predictions.extend(dict(model=model,policy=policy,view=view,**r) for r in out)
                    paired[(model,policy,view,family)]=out
                    print(model,policy,view,family,row['correct_known_accepts'],'/154',flush=True)
    known=json.loads((dest/'split.json').read_text())['halfface']['development']['known']
    for model in cfg['models']:
        for view in cfg['threshold_views']:
            for family in cfg['decision_families']:
                for policy in ['augmented_one_photo','up_to_three_photos']:
                    effect=paired_bootstrap(paired[(model,'one_photo',view,family)],paired[(model,policy,view,family)],
                        known,cfg['bootstrap_samples'],cfg['seed'],'correct_known_accept')
                    effects.append(dict(model=model,policy=policy,view=view,family=family,**effect))
    growth_predictions=[]; growth_summaries=[]
    growth_probes=[by_path[p] for p in selection['growth_probe_paths']]
    for model in cfg['models']:
        all_templates=templates[model+'_up_to_three_photos']
        for scenario in selection['scenarios']:
            selected=np.stack([all_templates[ids.index(sid)] for sid in scenario['ids']])
            scored=score_rows(growth_probes,base[model],scenario['ids'],selected)
            for family in cfg['decision_families']:
                threshold=previous_thresholds(info,model,'up_to_three_photos',family)
                out=apply_rule(scored,threshold,family); row=summarize(out,threshold,family)
                growth_summaries.append(dict(model=model,scenario=scenario['scenario'],gallery_size=scenario['size'],
                    **row,rank1_correct=sum(r['rank1_correct'] for r in out)))
                growth_predictions.extend(dict(model=model,scenario=scenario['scenario'],gallery_size=scenario['size'],**r) for r in out)
    for name,table in [('augmentation_predictions',predictions),('augmentation_summary',summary),
                       ('augmentation_thresholds',thresholds),('augmentation_grid',gridlog),('paired_effects',effects),
                       ('growth_predictions',growth_predictions),('growth_summary',growth_summaries)]:
        write_csv(dest/(name+'.csv'),[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in table])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true'); parser.add_argument('--config',default='configs/enrollment_extensions.json')
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.prepare: prepare(ROOT/args.config); return
    if args.run is None: parser.error('Use --prepare, then --run <printed path>')
    dest=(ROOT/args.run).resolve()
    if not dest.is_relative_to(ROOT/'results/enrollment_extensions') or (dest/'completion.json').exists():
        raise ValueError('Expected an unfinished prepared extension run')
    prep=json.loads((dest/'preparation.json').read_text())
    for mapping,prefix in [(prep['source_sha256'],ROOT),(prep['input_sha256'],ROOT),(prep['prepared_sha256'],dest)]:
        for name,digest in mapping.items():
            if sha256(prefix/name)!=digest: raise ValueError('Frozen input changed: '+name)
    cfg=json.loads((dest/'config.json').read_text()); rows=read_csv(dest/'inputs.csv')
    for r in rows:
        if sha256(ROOT/r['path'])!=r['source_sha256']: raise ValueError('Original image changed')
    selection=json.loads((dest/'selection.json').read_text())
    info=json.loads((ROOT/cfg['backbone_run']/'comparison.json').read_text())
    with np.load(ROOT/info['model']['features'],allow_pickle=False) as archive:
        base={m:archive[m] for m in cfg['models']}
    with threadpool_limits(limits=cfg['cpu_threads']):
        detector,models=load_models(cfg,info)
        extra,log,cache=extract_variants(dest,cfg,rows,base,detector,models,selection)
        ids,templates,sources=build_templates(cfg,rows,base,extra,log,selection,cache)
        write_csv(dest/'template_sources.csv',sources)
        experiments(dest,cfg,rows,base,templates,ids,selection,info)
        from scripts.measure_extension_runtime import measure_runtime
        measure_runtime(dest,cfg,rows,base,templates,ids,selection,info,detector,models,cache)
    save(dest/'completion.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),
        'scope':cfg['scope'],'training_performed':False,'final_test_evaluated':False,
        'cache':cache.relative_to(ROOT).as_posix(),
        'cache_sha256':{p.name:sha256(p) for p in cache.iterdir() if p.is_file()},
        'output_sha256':{p.name:sha256(p) for p in dest.iterdir() if p.is_file()}})
    print('Complete; independent audit required:',dest.relative_to(ROOT),flush=True)


if __name__=='__main__': main()
