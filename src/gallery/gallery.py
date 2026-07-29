"""
FaceGallery — Dynamic Face Gallery for Training-Free Enrollment
DynaCampus-FR Project

Stores identity prototypes (512-d embeddings), metadata, and face thumbnails.
Supports INSERT, SEARCH, REMOVE, and persistent disk storage.
Uses exact cosine matrix matching: gallery_matrix @ probe
"""

import numpy as np
import json
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FaceGallery:
    """
    Dynamic Face Gallery for training-free enrollment and cosine similarity search.

    Key operations:
    - enroll(): Add a new identity prototype to the gallery
    - enroll_multi(): Create quality-weighted prototype from multiple images
    - search(): Find the best matching identity using exact cosine similarity
    - remove(): Remove an identity from the gallery
    - list_identities(): List all enrolled identities

    Gallery data persists to disk as JSON metadata + NumPy embeddings + JPEG thumbnails.
    """

    def __init__(
        self,
        gallery_dir: str = "gallery_data",
        similarity_threshold: float = 0.40,
        min_quality: float = 0.30,
    ):
        self.gallery_dir = Path(gallery_dir)
        self.similarity_threshold = similarity_threshold
        self.min_quality = min_quality

        # Internal storage
        self._identities: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, np.ndarray] = {}

        # Create directories
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        (self.gallery_dir / "thumbnails").mkdir(exist_ok=True)

        # Load existing gallery from disk
        self._load()

    # ── Enrollment ──────────────────────────────────────────────────

    def enroll(
        self,
        subject_id: str,
        name: str,
        embedding: np.ndarray,
        face_thumbnail: Optional[np.ndarray] = None,
        quality_score: float = 0.0,
    ) -> bool:
        """
        Enroll a new identity or update an existing one.

        Args:
            subject_id: Unique identifier (e.g. "S001").
            name:       Display name.
            embedding:  L2-normalized 512-d feature vector.
            face_thumbnail: BGR face crop for display (optional).
            quality_score:  Composite quality score [0, 1].

        Returns:
            True if enrollment succeeded.
        """
        if embedding is None or not isinstance(embedding, np.ndarray) or embedding.size == 0:
            logger.warning("Enrollment failed: invalid embedding for %s", subject_id)
            return False

        # L2 normalize
        emb = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm < 1e-10:
            logger.warning("Enrollment failed: zero-norm embedding for %s", subject_id)
            return False
        emb = emb / norm

        self._embeddings[subject_id] = emb
        self._identities[subject_id] = {
            "name": name,
            "enrolled_at": datetime.now().isoformat(),
            "quality_score": round(float(quality_score), 4),
            "embedding_dim": int(emb.shape[0]),
        }

        # Save thumbnail
        if face_thumbnail is not None:
            thumb_path = self.gallery_dir / "thumbnails" / f"{subject_id}.jpg"
            cv2.imwrite(str(thumb_path), face_thumbnail)

        self._save()
        return True

    def enroll_multi(
        self,
        subject_id: str,
        name: str,
        embeddings: List[np.ndarray],
        qualities: List[float],
        face_thumbnail: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Enroll using quality-weighted prototype from multiple images.

        p = normalize( Σ q_k · z_k  /  Σ q_k )

        Args:
            subject_id: Unique identifier.
            name:       Display name.
            embeddings: List of 512-d feature vectors.
            qualities:  Quality score for each embedding.
            face_thumbnail: Best face crop for display.

        Returns:
            True if enrollment succeeded.
        """
        if not embeddings or not qualities or len(embeddings) != len(qualities):
            return False

        weighted_sum = np.zeros(embeddings[0].flatten().shape, dtype=np.float64)
        total_weight = 0.0
        for emb, q in zip(embeddings, qualities):
            w = max(q, 0.01)
            weighted_sum += w * emb.flatten().astype(np.float64)
            total_weight += w

        if total_weight < 1e-10:
            return False

        prototype = (weighted_sum / total_weight).astype(np.float32)
        norm = np.linalg.norm(prototype)
        if norm < 1e-10:
            return False
        prototype = prototype / norm

        avg_quality = float(np.mean(qualities))
        return self.enroll(subject_id, name, prototype, face_thumbnail, avg_quality)

    # ── Search ──────────────────────────────────────────────────────

    def search(
        self,
        probe_embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Search gallery using exact cosine similarity.

        ``scores = gallery_matrix @ probe``  (all vectors L2-normalized)

        Args:
            probe_embedding: 512-d feature vector of the probe face.
            threshold:       Minimum cosine similarity for KNOWN decision.

        Returns:
            Dictionary with keys: decision, top1_id, top1_name,
            top1_score, top2_score, margin.
        """
        if threshold is None:
            threshold = self.similarity_threshold

        empty_result: Dict[str, Any] = {
            "decision": "NOT_RECOGNIZED",
            "top1_id": None,
            "top1_name": None,
            "top1_score": 0.0,
            "top2_score": 0.0,
            "margin": 0.0,
        }

        if len(self._embeddings) == 0:
            empty_result["decision"] = "NO_GALLERY"
            return empty_result

        # Normalize probe
        probe = probe_embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(probe)
        if norm < 1e-10:
            return empty_result
        probe = probe / norm

        # Fast exact search using FAISS IndexFlatIP if available
        ids = list(self._embeddings.keys())
        try:
            import faiss
            gallery_matrix = np.stack([self._embeddings[sid] for sid in ids]).astype(np.float32)
            faiss_index = faiss.IndexFlatIP(512)
            faiss_index.add(gallery_matrix)
            k = min(2, len(ids))
            D, I = faiss_index.search(probe.reshape(1, -1), k)
            
            top1_idx = int(I[0][0])
            s1 = float(D[0][0])
            s2 = float(D[0][1]) if k > 1 else 0.0
        except Exception:
            # Fallback to NumPy matrix multiplication
            gallery_matrix = np.stack([self._embeddings[sid] for sid in ids])
            scores = gallery_matrix @ probe
            ranked = np.argsort(scores)[::-1]
            top1_idx = int(ranked[0])
            s1 = float(scores[top1_idx])
            s2 = float(scores[ranked[1]]) if len(ranked) > 1 else 0.0

        top1_id = ids[top1_idx]
        top1_name = self._identities[top1_id]["name"]

        decision = "KNOWN" if s1 >= threshold else "NOT_RECOGNIZED"

        return {
            "decision": decision,
            "top1_id": top1_id,
            "top1_name": top1_name,
            "top1_score": round(s1, 4),
            "top2_score": round(s2, 4),
            "margin": round(s1 - s2, 4),
            "search_backend": "FAISS IndexFlatIP",
        }

    # ── Management ──────────────────────────────────────────────────

    def list_pending(self) -> List[Dict[str, Any]]:
        """List all auto-enrolled identities waiting for approval."""
        result = []
        for sid, meta in self._identities.items():
            if sid.startswith("AutoID_"):
                entry = {"subject_id": sid, **meta}
                thumb = self.gallery_dir / "thumbnails" / f"{sid}.jpg"
                entry["has_thumbnail"] = thumb.exists()
                result.append(entry)
        return result

    def approve_pending(self, auto_id: str, new_name: str) -> bool:
        """Approve an auto-enrolled identity and give it a permanent display name."""
        if auto_id not in self._identities:
            return False
        
        self._identities[auto_id]["name"] = new_name
        self._identities[auto_id]["approved_at"] = datetime.now().isoformat()
        self._save()
        return True

    def remove(self, subject_id: str) -> bool:
        """Remove an identity from the gallery."""
        if subject_id not in self._identities:
            return False
        del self._identities[subject_id]
        del self._embeddings[subject_id]
        thumb = self.gallery_dir / "thumbnails" / f"{subject_id}.jpg"
        if thumb.exists():
            thumb.unlink()
        self._save()
        return True

    def list_identities(self) -> List[Dict[str, Any]]:
        """List all enrolled identities with metadata."""
        result = []
        for sid, meta in self._identities.items():
            entry = {"subject_id": sid, **meta}
            thumb = self.gallery_dir / "thumbnails" / f"{sid}.jpg"
            entry["has_thumbnail"] = thumb.exists()
            result.append(entry)
        return result

    def get_thumbnail(self, subject_id: str) -> Optional[np.ndarray]:
        """Get face thumbnail for a subject (BGR numpy array)."""
        thumb = self.gallery_dir / "thumbnails" / f"{subject_id}.jpg"
        if thumb.exists():
            return cv2.imread(str(thumb))
        return None

    def get_name(self, subject_id: str) -> str:
        """Get display name for a subject."""
        if subject_id in self._identities:
            return self._identities[subject_id]["name"]
        return "Unknown"

    @property
    def size(self) -> int:
        """Number of enrolled identities."""
        return len(self._identities)

    def next_id(self) -> str:
        """Generate the next available subject ID (S001, S002, ...)."""
        existing = set(self._identities.keys())
        idx = 1
        while f"S{idx:03d}" in existing:
            idx += 1
        return f"S{idx:03d}"

    # ── Persistence ─────────────────────────────────────────────────

    def _save(self):
        """Persist gallery to disk (JSON metadata + NumPy embeddings)."""
        meta_path = self.gallery_dir / "gallery_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._identities, f, indent=2, ensure_ascii=False)

        if self._embeddings:
            ids = list(self._embeddings.keys())
            matrix = np.stack([self._embeddings[sid] for sid in ids])
            np.savez(
                str(self.gallery_dir / "embeddings.npz"),
                ids=np.array(ids, dtype=object),
                embeddings=matrix,
            )
        else:
            emb_path = self.gallery_dir / "embeddings.npz"
            if emb_path.exists():
                emb_path.unlink()

    def _load(self):
        """Load gallery from disk."""
        meta_path = self.gallery_dir / "gallery_meta.json"
        emb_path = self.gallery_dir / "embeddings.npz"

        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._identities = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load gallery metadata: %s", e)
                self._identities = {}

        if emb_path.exists():
            try:
                data = np.load(str(emb_path), allow_pickle=True)
                ids = data["ids"]
                embeddings = data["embeddings"]
                for i, sid in enumerate(ids):
                    self._embeddings[str(sid)] = embeddings[i].astype(np.float32)
            except (IOError, KeyError) as e:
                logger.warning("Failed to load gallery embeddings: %s", e)
                self._embeddings = {}

        # Sync: remove embeddings for identities no longer in metadata and vice versa
        valid_ids = set(self._identities.keys()) & set(self._embeddings.keys())
        self._identities = {k: v for k, v in self._identities.items() if k in valid_ids}
        self._embeddings = {k: v for k, v in self._embeddings.items() if k in valid_ids}
