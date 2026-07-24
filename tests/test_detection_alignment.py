import os
import sys
import numpy as np
import pytest
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detection import SCRFDDetector
from src.alignment import FaceAligner

def test_detector_initialization():
    detector = SCRFDDetector(confidence_threshold=0.5)
    assert detector.conf_thresh == 0.5

def test_detector_empty_image():
    detector = SCRFDDetector(confidence_threshold=0.5)
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    results = detector.detect(empty_img)
    assert results == []

def test_aligner_with_dummy_landmarks():
    aligner = FaceAligner(crop_size=112)
    # Create a synthetic 200x200 image
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    # Draw simple facial features
    cv2.circle(img, (70, 80), 10, (255, 255, 255), -1) # left eye
    cv2.circle(img, (130, 80), 10, (255, 255, 255), -1) # right eye
    cv2.circle(img, (100, 110), 8, (255, 255, 255), -1) # nose
    cv2.circle(img, (75, 150), 8, (255, 255, 255), -1) # left mouth
    cv2.circle(img, (125, 150), 8, (255, 255, 255), -1) # right mouth

    landmarks = np.array([
        [70.0, 80.0],
        [130.0, 80.0],
        [100.0, 110.0],
        [75.0, 150.0],
        [125.0, 150.0]
    ], dtype=np.float32)

    aligned = aligner.align(img, landmarks)
    assert isinstance(aligned, np.ndarray)
    assert aligned.shape == (112, 112, 3)

def test_aligner_invalid_input():
    aligner = FaceAligner(crop_size=112)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    invalid_landmarks = np.array([[10, 10]], dtype=np.float32)

    with pytest.raises(ValueError):
        aligner.align(img, invalid_landmarks)
