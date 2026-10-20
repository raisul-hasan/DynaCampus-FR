"""Download the official ICFD sample or its unsigned access agreement.

Downloads stay under ignored data/. This never signs or sends an agreement.
halfFace needs a manual browser download when its host presents a browser check.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = {
    'icfd-sample': ('https://www.cse.iitm.ac.in/~vplab/downloads/ICFD_Samples.zip', 'ICFD_Samples.zip', b'PK'),
    'icfd-agreement': ('https://drive.google.com/uc?export=download&id=1Xw4o5ZSihEixrTgXv3hA4bHgMpWp365R', 'ICFD_ACCESS_AGREEMENT.pdf', b'%PDF'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=DOWNLOADS, default='icfd-sample')
    args = parser.parse_args()
    url, name, signature = DOWNLOADS[args.dataset]
    folder = ROOT / 'data' / 'downloads'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    if path.exists():
        payload = path.read_bytes()
        print('Using existing file:', path, flush=True)
    else:
        response = requests.get(url, timeout=(15, 90))
        response.raise_for_status()
        payload = response.content
        if not payload.startswith(signature):
            raise ValueError('Host did not return the expected archive/document. Open the official page manually.')
        path.write_bytes(payload)
    if not payload.startswith(signature):
        raise ValueError(f'Invalid file signature: {path}')
    receipt = {
        'source_url': url, 'official_page': 'https://sites.google.com/view/icfd',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'filename': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
        'access': 'Public sample only; full dataset access is not granted by this download.'
    }
    sources = ROOT / 'data' / 'sources'
    sources.mkdir(parents=True, exist_ok=True)
    (sources / f'{args.dataset}_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
