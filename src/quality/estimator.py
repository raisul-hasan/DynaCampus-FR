import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class FaceQualityEstimator:
    """
    Face Quality Assessment (FQA) Estimator for Low-Quality Campus CCTV.
    Evaluates face resolution, motion blur (sharpness), and 5-point facial landmark pose symmetry
    to produce a composite quality score q in [0.0, 1.0].
    """
    def __init__(
        self,
        config_path: Union[str, Path] = "configs/models.yaml",
        weights: Optional[Dict[str, float]] = None
    ):
        if weights is None:
            # Default quality metric weighting (Blur 45%, Resolution 35%, Pose 20%)
            self.weights = {"blur": 0.45, "resolution": 0.35, "pose": 0.20}
        else:
            self.weights = weights

        # Ideal face crop dimensions for CCTV recognition
        self.ideal_crop_size = 112.0
        self.min_crop_size = 20.0

    def estimate_blur_score(self, image: np.ndarray) -> float:
        """
        Estimate sharpness/blur score using Laplacian variance of image gradients.

        Args:
            image: BGR face crop numpy array.

        Returns:
            Normalized sharpness score in range [0.0, 1.0] (higher = sharper).
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return 0.0

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Soft sigmoid-like mapping where var=150 gives ~0.77 quality score
        blur_score = 1.0 - np.exp(-laplacian_var / 100.0)
        return float(np.clip(blur_score, 0.0, 1.0))

    def estimate_resolution_score(
        self,
        bbox: Optional[List[float]] = None,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> float:
        """
        Estimate resolution score based on detected bounding box size or crop size.

        Args:
            bbox: Bounding box list [x1, y1, x2, y2].
            image_shape: Height and width tuple (H, W).

        Returns:
            Resolution score in range [0.0, 1.0].
        """
        if bbox is not None and len(bbox) >= 4:
            w = max(0.0, float(bbox[2] - bbox[0]))
            h = max(0.0, float(bbox[3] - bbox[1]))
            dim = min(w, h)
        elif image_shape is not None and len(image_shape) >= 2:
            dim = float(min(image_shape[0], image_shape[1]))
        else:
            return 0.50

        if dim <= self.min_crop_size:
            return 0.0

        res_score = (dim - self.min_crop_size) / (self.ideal_crop_size - self.min_crop_size)
        return float(np.clip(res_score, 0.0, 1.0))

    def estimate_pose_score(self, landmarks: Optional[np.ndarray]) -> float:
        """
        Estimate pose symmetry score from 5 facial landmarks (eyes, nose, mouth corners).

        Args:
            landmarks: Numpy array of shape (5, 2).

        Returns:
            Pose symmetry score in range [0.0, 1.0] (1.0 = perfect frontal symmetry).
        """
        if landmarks is None or not isinstance(landmarks, np.ndarray) or landmarks.shape != (5, 2):
            return 0.50

        left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]

        # Distance from eyes to nose
        dist_left = float(np.linalg.norm(nose - left_eye))
        dist_right = float(np.linalg.norm(nose - right_eye))

        if dist_left + dist_right < 1e-5:
            return 0.0

        # Asymmetry ratio (1.0 = perfect symmetry)
        ratio = min(dist_left, dist_right) / max(dist_left, dist_right)

        # Eye tilt angle
        eye_dx = right_eye[0] - left_eye[0]
        eye_dy = right_eye[1] - left_eye[1]
        angle = np.abs(np.arctan2(eye_dy, eye_dx)) # Radians
        angle_score = max(0.0, 1.0 - (angle / (np.pi / 4.0)))

        pose_score = 0.6 * ratio + 0.4 * angle_score
        return float(np.clip(pose_score, 0.0, 1.0))

    def estimate_quality(
        self,
        image: np.ndarray,
        bbox: Optional[List[float]] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute full quality score breakdown and composite quality score q in [0.0, 1.0].

        Args:
            image: Input face BGR numpy array.
            bbox: Optional bounding box [x1, y1, x2, y2].
            landmarks: Optional 5-point landmark keypoints (5, 2).

        Returns:
            Dictionary containing:
            - 'resolution_score': float [0.0, 1.0]
            - 'blur_score': float [0.0, 1.0]
            - 'pose_score': float [0.0, 1.0]
            - 'overall_quality': float [0.0, 1.0]
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {
                "resolution_score": 0.0,
                "blur_score": 0.0,
                "pose_score": 0.0,
                "overall_quality": 0.0
            }

        q_res = self.estimate_resolution_score(bbox=bbox, image_shape=image.shape[:2])
        q_blur = self.estimate_blur_score(image)
        q_pose = self.estimate_pose_score(landmarks)

        q_total = (
            self.weights["blur"] * q_blur +
            self.weights["resolution"] * q_res +
            self.weights["pose"] * q_pose
        )

        return {
            "resolution_score": float(np.round(q_res, 4)),
            "blur_score": float(np.round(q_blur, 4)),
            "pose_score": float(np.round(q_pose, 4)),
            "overall_quality": float(np.round(np.clip(q_total, 0.0, 1.0), 4))
        }

    def visualize_quality(
        self,
        image: np.ndarray,
        metrics: Dict[str, float],
        title: str = "Quality Overlay"
    ) -> np.ndarray:
        """
        Annotate face image with a visual quality badge overlay.

        Args:
            image: Input BGR numpy image array.
            metrics: Quality metrics dictionary returned by estimate_quality().
            title: Title text string.

        Returns:
            Annotated BGR numpy image array with graphical badges.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)

        annotated = image.copy()
        h, w = annotated.shape[:2]

        overall = metrics.get("overall_quality", 0.0)

        # Color coding: Green for high quality (>= 0.7), Yellow for medium (>= 0.4), Red for low
        if overall >= 0.70:
            color = (0, 220, 0) # Green
            status = "HIGH QUALITY"
        elif overall >= 0.40:
            color = (0, 220, 255) # Yellow
            status = "MEDIUM QUALITY"
        else:
            color = (0, 0, 255) # Red
            status = "LOW QUALITY"

        # Draw semi-transparent header bar
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(h * 0.30)), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

        # Draw border
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), color, 2)

        # Draw overlay text
        cv2.putText(
            annotated,
            f"Q: {overall:.2f} ({status[:4]})",
            (5, int(h * 0.12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated,
            f"B:{metrics.get('blur_score', 0):.2f} R:{metrics.get('resolution_score', 0):.2f}",
            (5, int(h * 0.24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (240, 240, 240),
            1,
            cv2.LINE_AA
        )

        return annotated
