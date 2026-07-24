import os
import sys
import numpy as np
import cv2
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.quality import FaceQualityEstimator

def test_estimator_initialization():
    estimator = FaceQualityEstimator()
    assert "blur" in estimator.weights
    assert "resolution" in estimator.weights
    assert "pose" in estimator.weights

def test_blur_score_sharp_vs_blurred():
    estimator = FaceQualityEstimator()
    # Create sharp image with high gradient variance
    sharp_img = np.zeros((112, 112, 3), dtype=np.uint8)
    cv2.rectangle(sharp_img, (10, 10), (100, 100), (255, 255, 255), -1)
    cv2.line(sharp_img, (0, 0), (112, 112), (0, 0, 0), 5)

    # Blur image with heavy Gaussian blur
    blurred_img = cv2.GaussianBlur(sharp_img, (25, 25), 10.0)

    score_sharp = estimator.estimate_blur_score(sharp_img)
    score_blurred = estimator.estimate_blur_score(blurred_img)

    assert score_sharp > score_blurred
    assert 0.0 <= score_sharp <= 1.0
    assert 0.0 <= score_blurred <= 1.0

def test_resolution_score():
    estimator = FaceQualityEstimator()
    # Tiny bbox (below min threshold)
    score_tiny = estimator.estimate_resolution_score(bbox=[0, 0, 15, 15])
    assert score_tiny == 0.0

    # Ideal bbox size (112x112)
    score_ideal = estimator.estimate_resolution_score(bbox=[0, 0, 112, 112])
    assert score_ideal == 1.0

    # Intermediate bbox
    score_mid = estimator.estimate_resolution_score(bbox=[0, 0, 66, 66])
    assert 0.0 < score_mid < 1.0

def test_pose_symmetry_score():
    estimator = FaceQualityEstimator()
    # Frontal symmetric landmarks
    frontal_kps = np.array([
        [35.0, 45.0],  # left eye
        [77.0, 45.0],  # right eye
        [56.0, 65.0],  # nose
        [40.0, 85.0],  # left mouth
        [72.0, 85.0]   # right mouth
    ], dtype=np.float32)

    score_frontal = estimator.estimate_pose_score(frontal_kps)
    assert score_frontal > 0.80

    # Asymmetric side-pose landmarks
    side_kps = np.array([
        [20.0, 45.0],  # left eye close to nose
        [80.0, 45.0],  # right eye far from nose
        [25.0, 65.0],  # nose shifted left
        [22.0, 85.0],
        [75.0, 85.0]
    ], dtype=np.float32)

    score_side = estimator.estimate_pose_score(side_kps)
    assert score_side < score_frontal

def test_full_quality_estimation():
    estimator = FaceQualityEstimator()
    img = np.full((112, 112, 3), 128, dtype=np.uint8)
    metrics = estimator.estimate_quality(img)

    assert "resolution_score" in metrics
    assert "blur_score" in metrics
    assert "pose_score" in metrics
    assert "overall_quality" in metrics
    assert 0.0 <= metrics["overall_quality"] <= 1.0

def test_visualize_quality():
    estimator = FaceQualityEstimator()
    img = np.full((112, 112, 3), 128, dtype=np.uint8)
    metrics = estimator.estimate_quality(img)

    annotated = estimator.visualize_quality(img, metrics)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == (112, 112, 3)
