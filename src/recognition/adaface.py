"""Real AdaFace IR-50 inference using pinned official architecture and weights.

The official source is downloaded by scripts/fetch_adaface_assets.py. Existing
ArcFace/app code stays unchanged so the frozen evaluation remains auditable.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

OFFICIAL_COMMIT = 'c60eaa786a42c03444f3df7096dbaf9d57ae010d'


class _LegacyCheckpointMetadataKey:
    """Inert stand-in for the old Lightning callback class used as metadata key."""
    pass


def load_official_checkpoint(path):
    legacy = 'pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint'
    unexpected = set(torch.serialization.get_unsafe_globals_in_checkpoint(path)) - {legacy}
    if unexpected:
        raise ValueError(f'Unexpected checkpoint metadata globals: {sorted(unexpected)}')
    # The released checkpoint stores the callback CLASS as a dictionary key.
    # Map only that exact name to an inert local class; do not import Lightning
    # or enable unrestricted pickle loading. Backbone tensors load strictly below.
    with torch.serialization.safe_globals([(_LegacyCheckpointMetadataKey, legacy)]):
        return torch.load(path, map_location='cpu', weights_only=True)


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def bgr_tensor(crops):
    """Official (BGR / 255 - .5) / .5 preprocessing, without another color swap."""
    if not isinstance(crops, np.ndarray) or crops.dtype != np.uint8:
        raise ValueError('AdaFace expects uint8 BGR aligned crops')
    batch = crops[None] if crops.ndim == 3 else crops
    if batch.ndim != 4 or batch.shape[1:] != (112, 112, 3) or len(batch) == 0:
        raise ValueError('AdaFace expects (112,112,3) or (N,112,112,3) crops')
    # Match official NumPy float64 arithmetic followed by conversion to float32.
    normalized = ((batch / 255.) - .5) / .5
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(0, 3, 1, 2), dtype=np.float32))


class AdaFaceEmbedder:
    def __init__(self, model_dir, device='cpu'):
        self.model_dir = Path(model_dir)
        provenance_path = self.model_dir / 'provenance.json'
        if not provenance_path.is_file():
            raise FileNotFoundError('AdaFace assets missing; run scripts/fetch_adaface_assets.py')
        self.provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        if self.provenance['commit'] != OFFICIAL_COMMIT or self.provenance['architecture'] != 'ir_50':
            raise ValueError('Unexpected AdaFace architecture/source version')
        self.model_path = self.model_dir / 'adaface_ir50_ms1mv2.ckpt'
        if not self.model_path.is_file():
            raise FileNotFoundError(f'AdaFace checkpoint missing: {self.model_path}')
        if file_sha256(self.model_path) != self.provenance['checkpoint_sha256']:
            raise ValueError('AdaFace checkpoint hash differs from recorded provenance')
        source_path = self.model_dir / 'upstream/net.py'
        if file_sha256(source_path) != self.provenance['sources']['net.py']['sha256']:
            raise ValueError('AdaFace architecture source changed')
        spec = importlib.util.spec_from_file_location('_dynacampus_official_adaface_net', source_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.build_model('ir_50')
        # No arbitrary pickle execution or random-weight fallback.
        checkpoint = load_official_checkpoint(self.model_path)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('state_dict'), dict):
            raise ValueError('Expected an official AdaFace state_dict checkpoint')
        state = {k[6:]: v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')}
        if not state:
            raise ValueError('Official AdaFace backbone weights are absent')
        self.model.load_state_dict(state, strict=True)
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.mock = False
        self.model_name = 'adaface_ir50_ms1mv2'
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def extract_with_norm(self, crops):
        single = isinstance(crops, np.ndarray) and crops.ndim == 3
        tensor = bgr_tensor(crops).to(self.device)
        with torch.inference_mode():
            feature, norm = self.model(tensor)
        vectors = feature.cpu().numpy().astype(np.float32)
        norms = norm.cpu().numpy().reshape(-1).astype(np.float32)
        if vectors.shape != (len(tensor), 512) or norms.shape != (len(tensor),):
            raise RuntimeError('AdaFace returned invalid embedding dimensions')
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or np.any(lengths <= 1e-10) or np.any(norms <= 0):
            raise RuntimeError('AdaFace returned non-finite or zero features')
        vectors = vectors / lengths
        return (vectors[0], float(norms[0])) if single else (vectors, norms)

    def extract_embedding(self, crops):
        return self.extract_with_norm(crops)[0]
