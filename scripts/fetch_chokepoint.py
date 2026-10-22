"""Fetch official Portal-1 entering crops and ground truth, one file at a time."""
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'groundtruth.tar.xz': (456396, '5c9d4d38a1c614905fe48da48ecdf06c'),
    'P1E.tar.xz': (66406324, '07203fe0de5ebb1270742bd711f11dd8'),
}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    downloads = ROOT / 'data/downloads/chokepoint'
    sources = ROOT / 'data/sources/chokepoint'
    raw = ROOT / 'data/raw/chokepoint'
    for folder in [downloads, sources, raw]:
        folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    metadata = session.get('https://zenodo.org/api/records/815657', timeout=(15, 60))
    metadata.raise_for_status()
    info = metadata.json()
    (sources / 'zenodo_record.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    official = session.get('https://arma.sourceforge.net/chokepoint/', timeout=(15, 60))
    official.raise_for_status()
    (sources / 'official_page_and_license.html').write_bytes(official.content)
    receipts = []
    for name, (size, md5) in FILES.items():
        entry = next(f for f in info['files'] if f['key'] == name)
        if entry['size'] != size or entry['checksum'] != 'md5:' + md5:
            raise ValueError(f'Official metadata differs from reviewed version: {name}')
        target = downloads / name
        url = entry['links']['self']
        if not target.exists():
            partial = target.with_suffix(target.suffix + '.partial')
            print(f'Downloading {name}: {size / 1e6:.1f} MB', flush=True)
            with session.get(url, stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                with partial.open('wb') as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
            if partial.stat().st_size != size or digest(partial, 'md5') != md5:
                raise ValueError(f'Download integrity failure: {name}')
            partial.replace(target)
        if target.stat().st_size != size or digest(target, 'md5') != md5:
            raise ValueError(f'Archive integrity failure: {name}')
        marker = raw / (name + '.extracted.json')
        sha = digest(target)
        if not marker.exists():
            with tarfile.open(target, 'r:xz') as archive:
                # Python data filter rejects escaping paths, links and device files.
                archive.extractall(raw, filter='data')
            marker.write_text(json.dumps({'archive_sha256': sha}), encoding='utf-8')
        elif json.loads(marker.read_text())['archive_sha256'] != sha:
            raise ValueError('Extraction receipt differs from archive')
        receipts.append({'filename': name, 'bytes': size, 'md5': md5, 'sha256': sha,
                         'url': url, 'extraction_root': 'data/raw/chokepoint'})
        print('Verified and extracted:', name, flush=True)
    receipt = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
               'official_page': 'https://arma.sourceforge.net/chokepoint/',
               'record': 'https://zenodo.org/records/815657',
               'scope': 'Original supplied crops and ground truth, Portal 1 entering only',
               'license_copy': 'data/sources/chokepoint/official_page_and_license.html',
               'files': receipts}
    (sources / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print('Saved source receipt. Crops remain in ignored data/.', flush=True)


if __name__ == '__main__':
    main()
