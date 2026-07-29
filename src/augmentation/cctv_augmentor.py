"""
CCTV-Aware Data Augmentor — DynaCampus-FR Project
===================================================
Applies realistic surveillance domain transformations to single high-quality enrollment photos:
- Motion blur (simulates subject walking past camera)
- Downsampling / Low-resolution scaling (simulates distance from lens)
- JPEG/H.264 compression artifacts (simulates stream degradation)
- Lighting & gamma shadow shifts (simulates outdoor/indoor exposure)

Used during dynamic enrollment to create robust multi-style prototypes.
"""

import cv2
import numpy as np
from typing import List

class CCTVAugmentor:
    """
    Generates synthetic CCTV-style domain variations from a single face image.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def apply_motion_blur(self, image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """Simulate motion blur caused by moving targets."""
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        return cv2.filter2D(image, -1, kernel)

    def apply_downsample(self, image: np.ndarray, scale: float = 0.35) -> np.ndarray:
        """Simulate low-resolution surveillance crop."""
        h, w = image.shape[:2]
        low_h, low_w = max(int(h * scale), 16), max(int(w * scale), 16)
        small = cv2.resize(image, (low_w, low_h), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    def apply_jpeg_compression(self, image: np.ndarray, quality: int = 30) -> np.ndarray:
        """Simulate streaming compression noise."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, result = cv2.imencode('.jpg', image, encode_param)
        return cv2.imdecode(result, cv2.IMREAD_COLOR)

    def apply_lighting_shift(self, image: np.ndarray, gamma: float = 0.6) -> np.ndarray:
        """Simulate harsh lighting and shadow exposure."""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def generate_cctv_variants(self, aligned_face: np.ndarray) -> List[np.ndarray]:
        """
        Generate 4 CCTV domain variants from a single aligned face crop.
        Returns: [original, motion_blurred, low_res, compressed, shadow_shifted]
        """
        if aligned_face is None or aligned_face.size == 0:
            return []

        variants = [
            aligned_face,  # Original clean crop
            self.apply_motion_blur(aligned_face, kernel_size=7),
            self.apply_downsample(aligned_face, scale=0.35),
            self.apply_jpeg_compression(aligned_face, quality=25),
            self.apply_lighting_shift(aligned_face, gamma=0.6),
        ]
        return variants
