import os
import sys
import numpy as np
import pytest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.recognition import FaceEmbedder

def test_embedder_initialization():
    embedder = FaceEmbedder(mock=True)
    assert embedder.embedding_dim == 512
    assert embedder.mock is True

def test_single_embedding_extraction():
    embedder = FaceEmbedder(mock=True)
    crop = np.full((112, 112, 3), 128, dtype=np.uint8)
    emb = embedder.extract_embedding(crop)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)
    assert emb.dtype == np.float32

    # Check L2 normalization
    norm = np.linalg.norm(emb)
    assert pytest.approx(norm, abs=1e-5) == 1.0

def test_batch_embedding_extraction():
    embedder = FaceEmbedder(mock=True)
    crops = np.full((4, 112, 112, 3), 100, dtype=np.uint8)
    # Give different pixel content so mock generates different embeddings
    crops[1] += 10
    crops[2] += 20
    crops[3] += 30

    embs = embedder.extract_embedding(crops)

    assert isinstance(embs, np.ndarray)
    assert embs.shape == (4, 512)

    for i in range(4):
        norm = np.linalg.norm(embs[i])
        assert pytest.approx(norm, abs=1e-5) == 1.0

def test_invalid_crop_dimensions():
    embedder = FaceEmbedder(mock=True)
    invalid_crop = np.full((64, 64, 3), 128, dtype=np.uint8)

    with pytest.raises(ValueError):
        embedder.extract_embedding(invalid_crop)

def test_compute_similarity():
    embedder = FaceEmbedder(mock=True)
    v1 = np.ones(512, dtype=np.float32)
    v1 /= np.linalg.norm(v1)

    v2 = np.ones(512, dtype=np.float32)
    v2 /= np.linalg.norm(v2)

    # Identical vectors should have similarity 1.0
    sim_identical = embedder.compute_similarity(v1, v2)
    assert pytest.approx(sim_identical, abs=1e-5) == 1.0

    # Orthogonal vectors should have similarity 0.0
    v3 = np.zeros(512, dtype=np.float32)
    v3[:256] = 1.0
    v3 /= np.linalg.norm(v3)

    v4 = np.zeros(512, dtype=np.float32)
    v4[256:] = 1.0
    v4 /= np.linalg.norm(v4)

    sim_ortho = embedder.compute_similarity(v3, v4)
    assert pytest.approx(sim_ortho, abs=1e-5) == 0.0

def test_compute_similarity_matrix():
    embedder = FaceEmbedder(mock=True)
    set1 = np.random.randn(3, 512).astype(np.float32)
    set2 = np.random.randn(5, 512).astype(np.float32)

    matrix = embedder.compute_similarity_matrix(set1, set2)
    assert matrix.shape == (3, 5)
    assert np.all(matrix >= -1.0) and np.all(matrix <= 1.0)
