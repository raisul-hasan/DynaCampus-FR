# pyrefly: ignore [missing-import]
import pytest
import numpy as np
from src.aggregation.aggregator import FeatureAggregator

def test_quality_weighted_aggregation():
    aggregator = FeatureAggregator(strategy="quality_weighted", max_frames=10)
    
    vec1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
    vec2 = np.ones(512, dtype=np.float32) * 2.0 / np.sqrt(512)
    
    # Low quality for vec1 (0.2), High quality for vec2 (0.8)
    aggregator.update(track_id=1, feature=vec1, quality=0.2)
    aggregator.update(track_id=1, feature=vec2, quality=0.8)
    
    agg_feat = aggregator.aggregate(track_id=1)
    assert agg_feat is not None
    assert agg_feat.shape == (512,)

def test_aggregator_buffer_limit():
    aggregator = FeatureAggregator(strategy="mean", max_frames=5)
    for i in range(10):
        vec = np.random.randn(512).astype(np.float32)
        aggregator.update(track_id=1, feature=vec, quality=0.5)
        
    assert len(aggregator.track_buffers[1]["features"]) == 5
