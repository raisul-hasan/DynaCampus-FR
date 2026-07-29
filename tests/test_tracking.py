# pyrefly: ignore [missing-import]
import pytest
import numpy as np
from src.tracking.tracker import FaceTracker, KalmanBoxTracker, box_iou_matrix

def test_box_iou_matrix():
    boxes1 = np.array([[10, 10, 50, 50, 0.9]])
    boxes2 = np.array([[10, 10, 50, 50, 0.95]])
    iou = box_iou_matrix(boxes1, boxes2)
    assert iou.shape == (1, 1)
    assert np.isclose(iou[0, 0], 1.0)

def test_face_tracker_update():
    tracker = FaceTracker(max_age=5, min_hits=1, iou_threshold=0.3)
    dets = np.array([[10, 10, 50, 50, 0.9]])
    
    # Frame 1
    tracks1 = tracker.update(dets)
    assert len(tracks1) == 1
    assert tracks1[0, 4] == 1 # Track ID #1

    # Frame 2 (moving slightly)
    dets_f2 = np.array([[12, 12, 52, 52, 0.88]])
    tracks2 = tracker.update(dets_f2)
    assert len(tracks2) == 1
    assert tracks2[0, 4] == 1 # Retains Track ID #1

def test_face_tracker_empty_frame():
    tracker = FaceTracker(max_age=5, min_hits=1, iou_threshold=0.3)
    tracks = tracker.update(np.empty((0, 5)))
    assert len(tracks) == 0
