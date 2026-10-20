"""Shared still-image demo path matching the frozen original-photo protocol."""
import cv2
import numpy as np
from src.open_set import OpenSetDecisionEngine

SCORE_THRESHOLD = 0.3778589367866516
MARGIN_THRESHOLD = .15


def process_image(image, detector, aligner, embedder, quality_est, face_crop=False):
    if image is None or image.size == 0:
        raise ValueError('Image cannot be decoded')
    py,px = (round(d*.5) for d in image.shape[:2]) if face_crop else (0,0)
    padded = cv2.copyMakeBorder(image,py,py,px,px,cv2.BORDER_CONSTANT,value=(127,127,127)) if face_crop else image
    processed = []
    for face in detector.detect(padded):
        kps = face.get('kps')
        if kps is None or np.asarray(kps).shape != (5,2) or not np.isfinite(kps).all():
            continue
        crop = aligner.align(padded,kps)
        bbox = [max(0,face['bbox'][0]-px),max(0,face['bbox'][1]-py),
                min(image.shape[1],face['bbox'][2]-px),min(image.shape[0],face['bbox'][3]-py)]
        processed.append(dict(bbox=bbox,score=face['score'],kps=kps,aligned=crop,
            embedding=embedder.extract_embedding(crop),quality=quality_est.estimate_quality(crop,bbox=bbox,landmarks=kps)))
    return processed


def recognize(gallery, embedding, quality, score_threshold=SCORE_THRESHOLD, margin_threshold=MARGIN_THRESHOLD, quality_threshold=0.):
    result = gallery.search(embedding,threshold=score_threshold)
    engine = OpenSetDecisionEngine(base_threshold=score_threshold,margin_threshold=margin_threshold,quality_threshold=quality_threshold)
    state,identity,score = engine.evaluate_scores(result['top1_id'],result['top1_score'],
        result['top2_score'] if gallery.size > 1 else None,quality)
    return {**result,'decision':state,'accepted_id':identity,
            'accepted_name':gallery.get_name(identity) if identity else '', 'top1_score':score}
