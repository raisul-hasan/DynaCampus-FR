import numpy as np
from typing import Optional
from insightface.utils import face_align

class FaceAligner:
    """
    5-Point Facial Landmark Aligner using InsightFace similarity transformation.
    Warps face crops into standardized aligned images (default: 112x112 pixels).
    """
    def __init__(self, crop_size: int = 112):
        self.crop_size = crop_size

    def align(
        self,
        image: np.ndarray,
        landmark: np.ndarray,
        crop_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Align face image given 5 facial landmarks (eyes, nose, mouth corners).

        Args:
            image: Input BGR numpy image (H, W, 3).
            landmark: 5 landmark coordinates numpy array of shape (5, 2).
            crop_size: Target output crop size (default: self.crop_size = 112).

        Returns:
            Aligned BGR face crop numpy array of shape (crop_size, crop_size, 3).
        """
        if crop_size is None:
            crop_size = self.crop_size

        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid image provided for alignment.")

        if landmark is None or not isinstance(landmark, np.ndarray) or landmark.shape != (5, 2):
            raise ValueError("Landmarks must be a numpy array of shape (5, 2).")

        aligned_crop = face_align.norm_crop(image, landmark, image_size=crop_size)
        return aligned_crop
