import numpy as np
import scipy.linalg
from scipy.optimize import linear_sum_assignment

def box_iou_matrix(boxes1, boxes2):
    """
    Computes IoU matrix between two sets of bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute intersections
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:4], boxes2[:, 2:4])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Compute union
    union = area1[:, None] + area2 - inter

    return inter / np.clip(union, a_min=1e-6, a_max=None)

class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed as bbox.
    Uses an 8-dimensional state space [cx, cy, a, h, vx, vy, va, vh].
    """
    count = 0
    def __init__(self, bbox):
        # [x1, y1, x2, y2] -> [cx, cy, a, h]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        a = w / float(h)
        
        # Initialize filter
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],  
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])
        
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = np.array([[x, y, a, h]]).T
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        a = w / float(h)
        self.kf.predict()
        self.kf.update(np.array([[x, y, a, h]]).T)

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.get_state())
        return self.history[-1]

    def get_state(self):
        # [cx, cy, a, h] -> [x1, y1, x2, y2]
        arr = self.kf.x.T[0]
        cx, cy, a, h = arr[:4]
        w = a * h
        x1 = cx - w / 2.
        y1 = cy - h / 2.
        x2 = cx + w / 2.
        y2 = cy + h / 2.
        return np.array([x1, y1, x2, y2])

class FaceTracker:
    """
    IoU based multi-object tracker for faces (SORT).
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        """
        Updates trackers.
        dets: numpy array of [x1, y1, x2, y2, score]
        Returns: numpy array of [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1
        
        # Predict locations for existing tracks
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create and initialize new trackers
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # Remove dead trackers
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def _associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
        if len(trackers) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
        iou_matrix = box_iou_matrix(detections, trackers)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                cost_matrix = 1.0 - iou_matrix
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                matched_indices = np.array(list(zip(row_ind, col_ind)))
        else:
            matched_indices = np.empty(shape=(0,2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if len(matched_indices) == 0 or d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # filter out matches with low IoU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
                
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
