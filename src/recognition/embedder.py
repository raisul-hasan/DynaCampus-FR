import numpy as np
import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Union

class FaceEmbedder:
    """
    ArcFace inference using an explicit local InsightFace ONNX checkpoint.
    Extracts 512-dimensional L2-normalized feature embeddings from 112x112 aligned face crops.
    """
    def __init__(
        self,
        config_path: Union[str, Path] = "configs/models.yaml",
        model_name: Optional[str] = None,
        providers: Optional[List[str]] = None,
        mock: bool = False,
        model_path: Optional[Union[str, Path]] = None,
        cpu_threads: int = 2,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.embedding_dim = 512
        self.providers = providers
        self.mock = mock

        # Load configuration if available
        if Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                rec_cfg = cfg.get("recognizer", {})
                if model_name is None:
                    model_name = rec_cfg.get("primary", "insightface_arcface")
                self.embedding_dim = rec_cfg.get("embedding_dim", 512)
                if model_path is None:
                    model_path = rec_cfg.get("model_path")

        self.model_name = model_name or "insightface_arcface"
        self.model = None
        self.model_path = None
        if self.embedding_dim != 512:
            raise ValueError("This ArcFace implementation requires 512-dimensional embeddings.")

        if not self.mock:
            if self.model_name not in {"insightface_arcface", "arcface", "buffalo_l"}:
                raise ValueError(f"Backbone {self.model_name!r} is not implemented. Select insightface_arcface.")
            self.model_name = "insightface_arcface"
            self.model_path = Path(model_path).expanduser() if model_path else Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"
            if not self.model_path.is_file():
                raise FileNotFoundError(f"ArcFace checkpoint not found: {self.model_path}. Supply model_path or install the official buffalo_l models.")
            try:
                import onnxruntime as ort
                from insightface.model_zoo import get_model
                options = ort.SessionOptions()
                options.intra_op_num_threads = cpu_threads
                options.inter_op_num_threads = 1
                self.model = get_model(str(self.model_path), providers=self.providers, sess_options=options, download=False)
                if self.model is None or getattr(self.model, "taskname", None) != "recognition":
                    raise ValueError("The checkpoint is not an ArcFace recognition model.")
                self.model.prepare(ctx_id=0 if "CUDAExecutionProvider" in self.providers else -1)
            except Exception as e:
                raise RuntimeError(f"Failed to load real ArcFace checkpoint {self.model_path}; synthetic fallback is disabled.") from e

    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 512-dimensional L2-normalized feature embedding(s) from aligned BGR crop(s).

        Args:
            image: BGR face crop numpy array of shape (112, 112, 3) or batch of shape (N, 112, 112, 3).

        Returns:
            Float32 numpy array of shape (512,) for a single image, or (N, 512) for a batch.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid image input for feature extraction.")

        is_batch = (image.ndim == 4)
        if not is_batch:
            if image.ndim != 3 or image.shape != (112, 112, 3):
                raise ValueError(f"Expected image crop shape (112, 112, 3), got {image.shape}.")
            batch = np.expand_dims(image, axis=0)
        else:
            if image.shape[1:] != (112, 112, 3) or image.shape[0] == 0:
                raise ValueError(f"Expected batch crop shape (N, 112, 112, 3), got {image.shape}.")
            batch = image

        num_samples = batch.shape[0]

        if not np.isfinite(batch).all():
            raise ValueError("Face crops contain non-finite values.")
        if self.mock:
            # Deterministic mock embedding generator based on image pixel stats for offline/test environments
            embeddings = []
            for i in range(num_samples):
                seed = int(np.sum(batch[i]) * 1000) % (2**31 - 1)
                rng = np.random.RandomState(seed)
                vec = rng.randn(self.embedding_dim).astype(np.float32)
                vec /= (np.linalg.norm(vec) + 1e-10)
                embeddings.append(vec)
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            if self.model is None:
                raise RuntimeError("Real recognition model is unavailable; synthetic fallback is disabled.")
            embeddings = []
            for i in range(num_samples):
                feat = self.model.get_feat(batch[i])
                if feat.ndim > 1:
                    feat = feat.flatten()
                norm = np.linalg.norm(feat)
                if norm > 1e-10:
                    feat = feat / norm
                embeddings.append(feat.astype(np.float32))
            embeddings = np.array(embeddings, dtype=np.float32)

        if embeddings.shape != (num_samples, 512) or not np.isfinite(embeddings).all():
            raise RuntimeError("Recognizer returned invalid embeddings.")
        if np.any(np.linalg.norm(embeddings, axis=1) < 1e-10):
            raise RuntimeError("Recognizer returned a zero-norm embedding.")
        # Re-normalize across feature dimension
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        embeddings = embeddings / norms

        if not is_batch:
            return embeddings[0]
        return embeddings

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute Cosine Similarity between two L2-normalized 512-d feature vectors.

        Args:
            emb1: Feature vector of shape (512,).
            emb2: Feature vector of shape (512,).

        Returns:
            Cosine similarity score in range [-1.0, 1.0].
        """
        if emb1 is None or emb2 is None:
            raise ValueError("Embeddings cannot be None.")
        emb1_flat = emb1.flatten()
        emb2_flat = emb2.flatten()
        if emb1_flat.shape[0] != emb2_flat.shape[0]:
            raise ValueError(f"Embedding dimension mismatch: {emb1_flat.shape[0]} vs {emb2_flat.shape[0]}.")
        norm1 = np.linalg.norm(emb1_flat)
        norm2 = np.linalg.norm(emb2_flat)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        dot = float(np.dot(emb1_flat / norm1, emb2_flat / norm2))
        return float(np.clip(dot, -1.0, 1.0))

    @staticmethod
    def compute_similarity_matrix(embs1: np.ndarray, embs2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Cosine Similarity Matrix between two sets of feature embeddings.

        Args:
            embs1: Array of shape (N, 512).
            embs2: Array of shape (M, 512).

        Returns:
            Similarity matrix of shape (N, M) with values in range [-1.0, 1.0].
        """
        if embs1 is None or embs2 is None:
            raise ValueError("Embedding sets cannot be None.")
        if embs1.ndim == 1:
            embs1 = np.expand_dims(embs1, axis=0)
        if embs2.ndim == 1:
            embs2 = np.expand_dims(embs2, axis=0)

        norm1 = np.linalg.norm(embs1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embs2, axis=1, keepdims=True)
        norm1[norm1 < 1e-10] = 1.0
        norm2[norm2 < 1e-10] = 1.0

        normalized1 = embs1 / norm1
        normalized2 = embs2 / norm2

        matrix = np.dot(normalized1, normalized2.T)
        return np.clip(matrix, -1.0, 1.0)
