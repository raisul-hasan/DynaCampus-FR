"""Real CPU stage and persistent-gallery timing on fixed development samples."""
import json
import os
import platform
import time
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort
import torch
from threadpoolctl import threadpool_info

from scripts.run_arcface_baseline import write_csv
from src.alignment import FaceAligner
from src.quality import FaceQualityEstimator
from src.gallery import FaceGallery
from src.open_set import OpenSetDecisionEngine


def timed(call):
    start=time.perf_counter_ns(); result=call()
    return result,(time.perf_counter_ns()-start)/1e6


def measure_runtime(dest,cfg,rows,base,templates,ids,selection,info,detector,models,cache):
    root=dest.parents[2]
    work=root/'data/runtime'/dest.name
    work.mkdir(parents=True,exist_ok=False)
    aligner,quality=FaceAligner(),FaceQualityEstimator()
    by_path={r['path']:r for r in rows}
    few={m:templates[m+'_up_to_three_photos'] for m in cfg['models']}
    stages=[]; searches=[]; image_log=[]
    galleries={}; engines={}
    thresholds={}
    scenarios={s['size']:s for s in selection['scenarios'] if s['size']!=25 or s['seed']==42}
    for m in cfg['models']:
        rule=next(r for r in info['summaries'] if r['model']==m and r['dataset']=='halfface'
                  and r['strategy']=='up_to_three_photos' and r['family']=='score_margin')
        thresholds[m]=rule
        engines[m]=OpenSetDecisionEngine(rule['score_threshold'],0.,rule['margin_threshold'])
        for n in cfg['growth_gallery_sizes']:
            directory=work/(m+'_'+str(n))
            gallery=FaceGallery(str(directory),min_quality=0.)
            for sid in scenarios[n]['ids']:
                assert gallery.enroll(sid,sid,few[m][ids.index(sid)],quality_score=1.)
            assert gallery.size==n
            galleries[(m,n)]=gallery
    valid_probes=[by_path[p] for p in selection['growth_probe_paths'] if by_path[p]['status']=='ok']
    for m in cfg['models']:
        engine=engines[m]
        for n in cfg['growth_gallery_sizes']:
            gallery=galleries[(m,n)]
            for index in range(30): gallery.search(base[m][int(valid_probes[index%len(valid_probes)]['embedding_index'])])
            for index in range(cfg['runtime_search_repetitions']):
                row=valid_probes[index%len(valid_probes)]; vector=base[m][int(row['embedding_index'])]
                result,search_ms=timed(lambda:gallery.search(vector,threshold=thresholds[m]['score_threshold']))
                decision,decision_ms=timed(lambda:engine.evaluate_scores(result['top1_id'],result['top1_score'],
                                          result['top2_score'],float(row['quality'])))
                searches.append({'model':m,'gallery_size':n,'repetition':index,'path':row['path'],
                    'top1_id':result['top1_id'],'top1_score':result['top1_score'],'top2_score':result['top2_score'],
                    'decision':decision[0],'search_ms':search_ms,'decision_ms':decision_ms})
                for stage,value in [('gallery_search',search_ms),('decision',decision_ms)]:
                    stages.append({'model':m,'phase':'microbenchmark','stage':stage,'gallery_size':n,
                                   'status':'ok','path':row['path'],'repetition':index,'milliseconds':value})
    enrollment=[by_path[selection['one_photo'][sid][0]] for sid in selection['core_ids'][:cfg['runtime_enrollment_subjects']]]
    probes=[]
    for truth in ['known','unknown']:
        probes.extend(sorted([by_path[p] for p in selection['growth_probe_paths'] if by_path[p]['truth_status']==truth],
                             key=lambda r:r['path'])[:cfg['runtime_probe_images_per_truth']])
    for phase,inputs in [('enrollment',enrollment),('recognition',probes)]:
        for repetition in range(cfg['runtime_image_repetitions']):
            for row in inputs:
                for m in cfg['models'] if repetition%2==0 else cfg['models'][::-1]:
                    gallery=galleries[(m,55)]; component={}; start=time.perf_counter_ns()
                    image,component['decode']=timed(lambda:cv2.imread(str(root/row['path'])))
                    if image is None: raise ValueError('Runtime source cannot be decoded')
                    py,px=(round(d*.5) for d in image.shape[:2])
                    padded,component['padding']=timed(lambda:cv2.copyMakeBorder(image,py,py,px,px,
                        cv2.BORDER_CONSTANT,value=(127,127,127)))
                    faces,component['detection']=timed(lambda:detector.detect(padded))
                    status='ok' if len(faces)==1 else 'no_face' if not faces else 'multiple_faces'
                    delta=None; outcome='CAPTURE_FAILURE'
                    if status=='ok':
                        face=faces[0]
                        if face['kps'] is None or not np.isfinite(face['kps']).all():
                            raise ValueError('Invalid real runtime landmarks')
                        crop,component['alignment']=timed(lambda:aligner.align(padded,face['kps']))
                        bbox=[max(0,face['bbox'][0]-px),max(0,face['bbox'][1]-py),
                              min(image.shape[1],face['bbox'][2]-px),min(image.shape[0],face['bbox'][3]-py)]
                        q,component['quality']=timed(lambda:quality.estimate_quality(crop,bbox=bbox,landmarks=face['kps']))
                        feature,component['embedding']=timed(lambda:models[m].extract_embedding(crop))
                        delta=float(np.max(np.abs(feature-base[m][int(row['embedding_index'])])))
                        if row['status']!='ok' or delta>1e-5: raise ValueError('Runtime feature differs from cache')
                        if phase=='enrollment':
                            template,component['template_normalization']=timed(lambda:(feature/np.linalg.norm(feature)).astype(np.float32))
                            temporary=f"bench_{m}_{row['subject_id']}_{repetition}"
                            inserted,component['persistent_insertion']=timed(lambda:gallery.enroll(
                                temporary,temporary,template,quality_score=q['overall_quality']))
                            assert inserted and gallery.size==56
                            outcome='INSERTED'
                        else:
                            result,component['gallery_search']=timed(lambda:gallery.search(feature,threshold=thresholds[m]['score_threshold']))
                            decision,component['decision']=timed(lambda:engines[m].evaluate_scores(
                                result['top1_id'],result['top1_score'],result['top2_score'],q['overall_quality']))
                            outcome=decision[0]
                    elapsed=(time.perf_counter_ns()-start)/1e6
                    total_stage=phase+'_end_to_end'
                    if phase=='enrollment' and status=='ok':
                        assert gallery.remove(temporary) and gallery.size==55
                    if status!=row['status']: raise ValueError('Runtime extraction outcome changed')
                    for stage,value in {**component,total_stage:elapsed}.items():
                        stages.append({'model':m,'phase':phase,'stage':stage,'gallery_size':55,
                                       'status':status,'path':row['path'],'repetition':repetition,'milliseconds':value})
                    image_log.append({'model':m,'phase':phase,'path':row['path'],'subject_id':row['subject_id'],
                        'truth_status':row['truth_status'],'repetition':repetition,'status':status,'outcome':outcome,
                        'cache_max_difference':delta if delta is not None else '',
                        'sum_component_ms':sum(component.values()),'wall_ms':elapsed})
            print('Runtime images:',phase,'repetition',repetition+1,flush=True)
    groups=defaultdict(list)
    for row in stages:
        groups[(row['model'],row['phase'],row['stage'],row['gallery_size'],row['status'])].append(row['milliseconds'])
    summary=[]
    for (m,phase,stage,n,status),values in sorted(groups.items()):
        summary.append({'model':m,'phase':phase,'stage':stage,'gallery_size':n,'status':status,
                        'samples':len(values),'median_ms':float(np.median(values)),
                        'p95_ms':float(np.percentile(values,95))})
    write_csv(dest/'runtime_samples.csv',stages); write_csv(dest/'runtime_summary.csv',summary)
    write_csv(dest/'runtime_search.csv',searches); write_csv(dest/'runtime_images.csv',image_log)
    hardware={'python':platform.python_version(),'platform':platform.platform(),'processor':platform.processor(),
        'logical_cpu_count':os.cpu_count(),'cpu_threads':cfg['cpu_threads'],'torch':torch.__version__,
        'numpy':np.__version__,'onnxruntime':ort.__version__,'opencv':cv2.__version__,
        'cuda_used':False,'onnx_providers':['CPUExecutionProvider'],'threadpools':threadpool_info(),
        'boundaries':{'enrollment':'direct wall from image decode through fresh persistent insertion into a 55-identity gallery (55 to 56); restoration excluded',
            'recognition':'direct wall from image decode through actual gallery search and four-state decision; capture failures retained separately',
            'search':'actual FaceGallery.search including normalization, rebuilding matrix, sorting and metadata lookup; not a cached matrix microkernel',
            'decision':'actual OpenSetDecisionEngine.evaluate_scores only; instance construction excluded',
            'excluded':'model loading, warmup, browser/UI transfer, webcam capture, network/RTSP and camera FPS'},
        'repetitions_note':'Repeated calls on fixed inputs measure latency, not independent accuracy samples; warm CPU and filesystem-cache conditions.'}
    (dest/'runtime_environment.json').write_text(json.dumps(hardware,indent=2)+'\n',encoding='utf-8')
