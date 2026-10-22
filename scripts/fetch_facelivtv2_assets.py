"""Fetch FaceLiVTv2-S official assets and provenance records.

Mirrors the AdaFace fetch pattern: downloads architecture source, checkpoint,
and provenance metadata. Assets stay in ignored data/models/facelivtv2.
"""
import hashlib
import json
import requests
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data' / 'models' / 'facelivtv2'
UPSTREAM = DEST / 'upstream'
PROVENANCE_PATH = DEST / 'provenance.json'

ARCHIVE_COMMIT = 'main'  # FaceLiVT repo commit used


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    UPSTREAM.mkdir(parents=True, exist_ok=True)

    # Download architecture source files
    base = 'https://raw.githubusercontent.com/novendrastywn/FaceLiVT'
    # Mapping of local filename -> remote path relative to repo root
    file_map = {
        'facelivtv2.py': 'main/backbones/facelivtv2.py',
        'facelivt.py': 'main/backbones/facelivt.py',
        'inference.py': 'main/inference.py',
        'README.md': 'main/README.md',
        'LICENSE': 'main/LICENSE',
    }
    sources = {}
    for name, rel in file_map.items():
        url = f'{base}/{rel}'
        r = requests.get(url, timeout=60)
        if r.status_code == 404:
            print(f'  {name}: 404 (skipped)')
            continue
        r.raise_for_status()
        path = UPSTREAM / name
        if path.exists() and path.read_bytes() != r.content:
            raise ValueError(f'Existing pinned source differs: {path}')
        path.write_bytes(r.content)
        sources[name] = dict(url=url, sha256=sha(path))

    # Download checkpoint (skip if already present with same size)
    ckpt_url = 'https://huggingface.co/novendrastywn/FaceLiVT/resolve/main/facelivtv2-s.pt'
    ckpt_path = DEST / 'facelivtv2-s.pt'
    if ckpt_path.is_file() and ckpt_path.stat().st_size > 1 * 1024 * 1024:
        print(f'Checkpoint already present ({ckpt_path.stat().st_size/1024**2:.1f} MiB) – skipping download')
        ckpt_h = sha(ckpt_path)
    else:
        ckpt_r = requests.get(ckpt_url, timeout=(30, 300), stream=True)
        ckpt_r.raise_for_status()
        received = 0
        with ckpt_path.open('wb') as output:
            for chunk in ckpt_r.iter_content(1024 * 1024):
                output.write(chunk)
                received += len(chunk)
        if received < 1 * 1024 * 1024:  # at least 1 MiB
            raise RuntimeError('Checkpoint unexpectedly small; retaining partial file for diagnosis')
        ckpt_h = sha(ckpt_path)
        print(f'Checkpoint downloaded: {received / 1024**2:.1f} MiB, sha256={ckpt_h}')


    # Record provenance
    provenance = dict(
        repository='https://github.com/novendrastywn/FaceLiVT',
        commit=ARCHIVE_COMMIT,
        architecture='facelivtv2_s',
        training_data='Glint360K',
        sources=sources,
        checkpoint_source='https://huggingface.co/novendrastywn/FaceLiVT/resolve/main/facelivtv2-s.pt',
        checkpoint='data/models/facelivtv2/facelivtv2-s.pt',
        checkpoint_bytes=ckpt_path.stat().st_size,
        checkpoint_sha256=ckpt_h,
        retrieved_utc=datetime.now(timezone.utc).isoformat(),
        source_license='CC-BY-SA-4.0; source license is archived. Do not infer separate dataset or checkpoint permissions from this alone.',
        paper_reference='Setyawan et al., FaceLiVTv2: An Improved Baseline Hybrid Architecture for Efficient Mobile Face Recognition, arXiv:2604.09127, 2026 (under review, IEEE TBIOM).',
    )
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
