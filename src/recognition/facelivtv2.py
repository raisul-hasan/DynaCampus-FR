"""Real FaceLiVTv2-S inference using pinned official architecture and weights.

The official source is downloaded by scripts/fetch_facelivtv2_assets.py.
Existing ArcFace/AdaFace/app code stays unchanged so the frozen evaluation
remains auditable. FaceLiVTv2 expects RGB input (unlike AdaFace's BGR),
so BGR-aligned crops from SCRFD are converted before inference.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

OFFICIAL_REPO = 'https://github.com/novendrastywn/FaceLiVT'
OFFICIAL_COMMIT = 'main'
EXPECTED_PARAM_COUNT = 4620000  # FaceLiVTv2-S: 4.62M params
EXPECTED_OUTPUT_DIM = 512


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def rgb_tensor(crops):
    """Official FaceLiVTv2 preprocessing: BGR->RGB, (x/255 - 0.5)/0.5.

    The official inference.py does:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)

    We accept uint8 BGR crops (same as AdaFace input convention) and convert
    to RGB before normalising, matching the official pipeline exactly.
    """
    if not isinstance(crops, np.ndarray) or crops.dtype != np.uint8:
        raise ValueError('FaceLiVTv2 expects uint8 BGR aligned crops')
    batch = crops[None] if crops.ndim == 3 else crops
    if batch.ndim != 4 or batch.shape[1:] != (112, 112, 3) or len(batch) == 0:
        raise ValueError('FaceLiVTv2 expects (112,112,3) or (N,112,112,3) crops')
    # BGR -> RGB, then (x/255 - 0.5)/0.5
    rgb = batch[:, :, :, ::-1].copy()  # BGR to RGB
    normalized = ((rgb / 255.) - .5) / .5
    return torch.from_numpy(np.ascontiguousarray(normalized.transpose(0, 3, 1, 2), dtype=np.float32))


class FaceLiVTv2Embedder:
    def __init__(self, model_dir, device='cpu'):
        self.model_dir = Path(model_dir)
        provenance_path = self.model_dir / 'provenance.json'
        if not provenance_path.is_file():
            raise FileNotFoundError('FaceLiVTv2 assets missing; run scripts/fetch_facelivtv2_assets.py')
        self.provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        if self.provenance['architecture'] != 'facelivtv2_s':
            raise ValueError('Unexpected FaceLiVTv2 architecture version')
        self.model_path = self.model_dir / 'facelivtv2-s.pt'
        if not self.model_path.is_file():
            raise FileNotFoundError(f'FaceLiVTv2 checkpoint missing: {self.model_path}')
        if file_sha256(self.model_path) != self.provenance['checkpoint_sha256']:
            raise ValueError('FaceLiVTv2 checkpoint hash differs from recorded provenance')
        source_path = self.model_dir / 'upstream' / 'facelivtv2.py'
        if file_sha256(source_path) != self.provenance['sources']['facelivtv2.py']['sha256']:
            raise ValueError('FaceLiVTv2 architecture source changed')
        # Load official architecture from pinned source
        import sys
        spec = importlib.util.spec_from_file_location('_dynacampus_official_facelivtv2', source_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules['_dynacampus_official_facelivtv2'] = module
        spec.loader.exec_module(module)
        self.model = module.facelivtv2_s(num_classes=512)
        # Load checkpoint as plain state_dict (OrderedDict)
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError('Expected an official FaceLiVTv2 state_dict checkpoint')
        self.model.load_state_dict(checkpoint, strict=True)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.device = torch.device(device)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.mock = False
        self.model_name = 'facelivtv2_s'
        self.pretraining_data = 'Glint360K'

    def extract_embedding(self, crops):
        """Extract L2-normalized 512-dim embedding(s) from BGR aligned crop(s)."""
        single = isinstance(crops, np.ndarray) and crops.ndim == 3
        tensor = rgb_tensor(crops).to(self.device)
        with torch.inference_mode():
            features = self.model(tensor)
        vectors = features.cpu().numpy().astype(np.float32)
        if vectors.shape != (len(tensor), 512):
            raise RuntimeError('FaceLiVTv2 returned invalid embedding dimensions')
        if not np.isfinite(vectors).all():
            raise RuntimeError('FaceLiVTv2 returned non-finite features')
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(lengths <= 1e-10):
            raise RuntimeError('FaceLiVTv2 returned zero-norm features')
        vectors = vectors / lengths
        return vectors[0] if single else vectors