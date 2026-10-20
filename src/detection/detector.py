import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

class SCRFDDetector:
    """
    SCRFD Face Detector wrapper using InsightFace.
    Detects face bounding boxes, confidence scores, and 5 facial landmark keypoints.
    """
    def __init__(
        self,
        config_path: Union[str, Path] = "configs/models.yaml",
        confidence_threshold: Optional[float] = None,
        input_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None,
        model_path: Optional[Union[str, Path]] = None,
        cpu_threads: int = 2,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        
        # Load confidence threshold from config if available
        self.conf_thresh = confidence_threshold
        if self.conf_thresh is None and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                self.conf_thresh = cfg.get("detector", {}).get("confidence_threshold", 0.50)
                det_input_size = cfg.get("detector", {}).get("input_size", [640, 640])
                if isinstance(det_input_size, list) and len(det_input_size) == 2:
                    input_size = (det_input_size[0], det_input_size[1])

        if self.conf_thresh is None:
            self.conf_thresh = 0.50

        self.input_size = input_size
        self.providers = providers

        # Load only the detector, without initializing every buffalo_l model.
        import onnxruntime as ort
        from insightface.model_zoo import get_model
        self.model_path = Path(model_path).expanduser() if model_path else Path.home() / ".insightface/models/buffalo_l/det_10g.onnx"
        if not self.model_path.is_file():
            raise FileNotFoundError(f"SCRFD checkpoint not found: {self.model_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads
        options.inter_op_num_threads = 1
        self.model = get_model(str(self.model_path), providers=self.providers, sess_options=options, download=False)
        if self.model is None or getattr(self.model, "taskname", None) != "detection":
            raise RuntimeError("Checkpoint is not a face detection model.")
        self.model.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                           det_thresh=self.conf_thresh, input_size=self.input_size)

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a BGR numpy image.

        Args:
            image: BGR numpy image array (H, W, 3).

        Returns:
            List of detected faces, where each item is a dictionary containing:
            - 'bbox': List[float] [x1, y1, x2, y2]
            - 'score': float confidence score
            - 'kps': np.ndarray of shape (5, 2) containing landmark keypoints
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []

        boxes, landmarks = self.model.detect(image, max_num=0)
        results = []
        for index, box in enumerate(boxes):
            results.append({
                "bbox": box[:4].astype(float).tolist(),
                "score": float(box[4]),
                "kps": landmarks[index].astype(float) if landmarks is not None else None
            })
        return results
