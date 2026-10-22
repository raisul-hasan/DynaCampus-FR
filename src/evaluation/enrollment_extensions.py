"""Enrollment-only transformations and controlled gallery-composition helpers."""
import cv2
import numpy as np


def enrollment_variant(crop, variant):
    if not isinstance(crop,np.ndarray) or crop.shape != (112,112,3) or crop.dtype != np.uint8:
        raise ValueError('Expected aligned 112x112 BGR uint8 enrollment crop')
    image=crop.copy()
    if variant=='blur_sigma1':
        return cv2.GaussianBlur(image,(3,3),1.)
    if variant=='resize56':
        return cv2.resize(cv2.resize(image,(56,56),interpolation=cv2.INTER_CUBIC),
                          (112,112),interpolation=cv2.INTER_CUBIC)
    if variant=='jpeg50':
        ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,50])
        if not ok: raise RuntimeError('JPEG encoding failed')
        return cv2.imdecode(encoded,cv2.IMREAD_COLOR)
    if variant in ['gamma0.7','gamma1.4']:
        gamma=float(variant[5:])
        lut=np.clip(np.rint((np.arange(256)/255.)**gamma*255),0,255).astype(np.uint8)
        return cv2.LUT(image,lut)
    raise ValueError('Unsupported enrollment variant')


def gallery_scenarios(known_ids,seeds):
    if len(known_ids)!=55 or len(set(known_ids))!=55:
        raise ValueError('Expected all 55 distinct development-known identities')
    core=list(known_ids[:10]); rest=list(known_ids[10:])
    output=[{'scenario':'g10','size':10,'seed':None,'ids':sorted(core)}]
    for seed in seeds:
        shuffled=np.random.default_rng(seed).permutation(rest).tolist()
        output.append({'scenario':f'g25_seed{seed}','size':25,'seed':seed,
                       'ids':sorted(core+shuffled[:15])})
    output.append({'scenario':'g55','size':55,'seed':None,'ids':sorted(known_ids)})
    return core,output


def check_development_ids(records,known,unknown,reserved):
    allowed=set(known)|set(unknown)
    if set(known)&set(unknown) or allowed&set(reserved):
        raise ValueError('Identity role overlap')
    if not records or any(r['dataset']!='halfface' or r['subject_id'] not in allowed for r in records):
        raise ValueError('Non-development inputs')


def probe_rows(records,known_ids,unknown_ids):
    known,unknown=set(known_ids),set(unknown_ids)
    return [r for r in records if (r['subject_id'] in known and r['mask_status']=='masked')
            or r['subject_id'] in unknown]


def score_rows(records,features,ids,templates):
    output=[]
    for record in records:
        row={k:record[k] for k in ['dataset','subject_id','path','truth_status','status','quality']}
        row.update(top1_id='',top1_score='',top2_score='',rank1_correct=False)
        if record['status']=='ok':
            scores=templates@features[int(record['embedding_index'])]
            order=np.argsort(scores)[::-1]
            row.update(top1_id=ids[int(order[0])],top1_score=float(scores[order[0]]),
                       top2_score=float(scores[order[1]]))
            row['rank1_correct']=row['truth_status']=='known' and row['top1_id']==row['subject_id']
        output.append(row)
    return output
