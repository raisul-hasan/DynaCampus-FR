"""Deterministic operators for the corrected, exploratory ChokePoint study."""
import hashlib

import cv2
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

LEVELS = ['clean', 'mild', 'medium', 'severe']


def stable_seed(*parts):
    payload = '\x00'.join(str(p) for p in parts).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-10:
        raise ValueError('Invalid/zero embedding; no synthetic fallback')
    return (vector / norm).astype(np.float32)


def degrade(image, level, seed):
    if image.dtype != np.uint8 or image.shape != (112, 112, 3):
        raise ValueError('Corruptions require 112x112 BGR uint8 input')
    out = image.copy()
    if level == 'clean':
        return out
    if level not in LEVELS:
        raise ValueError('Unknown corruption level')
    kernel, sigma, jpeg = {'mild': (3, 1., 75), 'medium': (5, 2., 50),
                           'severe': (7, 3.5, 25)}[level]
    out = cv2.GaussianBlur(out, (kernel, kernel), sigma)
    if level != 'mild':
        side = 56 if level == 'medium' else 28
        out = cv2.resize(out, (side, side), interpolation=cv2.INTER_CUBIC)
        out = cv2.resize(out, (112, 112), interpolation=cv2.INTER_CUBIC)
    if level == 'severe':
        noise = np.random.default_rng(seed).normal(0., 15., out.shape)
        out = np.clip(np.rint(out.astype(np.float64) + noise), 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if out is None:
        raise RuntimeError('JPEG decoding failed')
    return out


def laplacian_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.clip(.6 * min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500, 1)
                         + .4 * min(gray.std() / 80, 1), 1e-4, 1))


def consecutive_tracks(records, length=5, max_gap=1, cap=15):
    """Partition one identity/sequence/camera into non-overlapping contiguous tracks."""
    if len({(r['identity'], r['sequence']) for r in records}) > 1:
        raise ValueError('Cross-identity/sequence track requested')
    ordered = sorted(records, key=lambda r: r['frame'])
    if len({r['frame'] for r in ordered}) != len(ordered):
        raise ValueError('Duplicate frame in sequence')
    tracks, pending, previous = [], [], None
    for row in ordered:
        if previous is not None and row['frame'] - previous > max_gap:
            pending = []
        pending.append(row)
        previous = row['frame']
        if len(pending) == length:
            tracks.append(pending)
            pending = []
            if len(tracks) == cap:
                break
    return tracks


def aggregate(embeddings, norms, laplacians, method, random_index=0):
    embs = np.asarray(embeddings)
    norms, laplacians = np.asarray(norms), np.asarray(laplacians)
    if method == 'first':
        return embs[0].copy()
    if method == 'random':
        return embs[random_index].copy()
    if method == 'best_norm':
        return embs[np.argmax(norms)].copy()
    if method == 'best_laplacian':
        return embs[np.argmax(laplacians)].copy()
    if method == 'mean':
        return unit(embs.mean(axis=0))
    if method.startswith('softmax_norm_'):
        beta = float(method.rsplit('_', 1)[1])
        logits = beta * (norms - norms.mean()) / (norms.std() + 1e-8)
        weights = np.exp(logits - logits.max())
        return unit((embs * (weights / weights.sum())[:, None]).sum(axis=0))
    raise ValueError('Unknown aggregation policy')


def accept(scores, quality, threshold, rule, margin=.08, floor=.2):
    scores, quality = np.asarray(scores), np.asarray(quality)
    if rule == 'fixed':
        return scores >= threshold
    if rule == 'aosm':
        return scores >= np.maximum(threshold - (1 - quality) * margin, floor)
    raise ValueError('Unknown decision rule')


def calibrate(scores, quality, rule, target=.01, margin=.08, floor=.2):
    """Use ONLY unknown calibration scores; choose lowest admissible threshold."""
    scores, quality = np.asarray(scores), np.asarray(quality)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Unknown calibration scores missing/invalid')
    shifted = scores if rule == 'fixed' else scores + (1 - quality) * margin
    candidates = np.unique(np.r_[-1., shifted, np.nextafter(shifted, np.inf), 1.1])
    budget = int(np.floor(target * len(scores) + 1e-12))
    for tau in candidates:
        if accept(scores, quality, tau, rule, margin, floor).sum() <= budget:
            return float(tau)
    raise RuntimeError('No admissible calibration threshold')


def verification_metrics(scores, truths):
    """Descriptive pairwise ROC; thresholds here are NOT deployment calibration."""
    scores, truths = np.asarray(scores), np.asarray(truths)
    labels = np.zeros(scores.shape, dtype=np.int8)
    known = truths >= 0
    labels[np.where(known)[0], truths[known]] = 1
    fpr, tpr, _ = roc_curve(labels.ravel(), scores.ravel(), drop_intermediate=False)
    idx = np.argmin(np.abs(fpr - (1 - tpr)))
    return {'genuine_pairs': int(labels.sum()), 'impostor_pairs': int(labels.size-labels.sum()),
            'tar_pair_far_1pct': float(tpr[fpr <= .01].max()),
            'tar_pair_far_0_1pct': float(tpr[fpr <= .001].max()),
            'eer': float((fpr[idx] + 1 - tpr[idx]) / 2),
            'roc_auc': float(roc_auc_score(labels.ravel(), scores.ravel()))}


def paired_identity_interval(values, identities, repeats=1000, seed=42):
    """Track-weighted statistic with paired identity-cluster resampling."""
    values, identities = np.asarray(values, dtype=float), np.asarray(identities)
    groups = sorted(set(identities))
    sums = np.array([values[identities == g].sum() for g in groups])
    counts = np.array([(identities == g).sum() for g in groups])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), (repeats, len(groups)))
    samples = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return [float(x) for x in np.quantile(samples, [.025, .975])]
