import numpy as np

class FeatureAggregator:
    """
    Aggregates face embeddings for a single tracked identity across multiple frames.
    Implements QATA (Quality-Aware Temporal Aggregation) and baseline strategies.
    """
    def __init__(self, strategy="quality_weighted", max_frames=30):
        """
        strategies: 'mean', 'best', 'quality_weighted'
        """
        self.strategy = strategy
        self.max_frames = max_frames
        self.track_buffers = {} # track_id -> {'features': [], 'qualities': []}

    def update(self, track_id, feature, quality):
        """
        Add a new feature to the track buffer.
        """
        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = {'features': [], 'qualities': []}
            
        self.track_buffers[track_id]['features'].append(feature)
        self.track_buffers[track_id]['qualities'].append(quality)
        
        # enforce buffer size
        if len(self.track_buffers[track_id]['features']) > self.max_frames:
            self.track_buffers[track_id]['features'].pop(0)
            self.track_buffers[track_id]['qualities'].pop(0)

    def aggregate(self, track_id):
        """
        Compute the aggregated feature for a track_id.
        """
        if track_id not in self.track_buffers or len(self.track_buffers[track_id]['features']) == 0:
            return None
            
        features = np.array(self.track_buffers[track_id]['features'])
        qualities = np.array(self.track_buffers[track_id]['qualities'])
        
        if self.strategy == 'mean':
            agg_feat = np.mean(features, axis=0)
        elif self.strategy == 'best':
            best_idx = np.argmax(qualities)
            agg_feat = features[best_idx]
        elif self.strategy == 'quality_weighted':
            # Softmax-like temperature scaling on qualities to amplify good frames
            # QATA implementation
            q_norm = qualities / (np.sum(qualities) + 1e-8)
            agg_feat = np.sum(features * q_norm[:, None], axis=0)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        # L2 Normalize the final aggregated feature
        norm = np.linalg.norm(agg_feat)
        if norm > 0:
            agg_feat = agg_feat / norm
            
        return agg_feat

    def clear(self, track_id):
        """
        Clear buffer for a dead track
        """
        if track_id in self.track_buffers:
            del self.track_buffers[track_id]
