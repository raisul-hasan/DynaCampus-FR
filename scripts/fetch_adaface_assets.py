"""Fetch pinned official AdaFace IR-50 source and its MS1MV2 checkpoint.

Artifacts stay in ignored data/models/adaface. No account or private data is used.
"""
import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'c60eaa786a42c03444f3df7096dbaf9d57ae010d'
DRIVE_ID = '1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI'


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form' and attrs.get('id') == 'download-form':
            self.action = attrs.get('action')
        if tag == 'input' and attrs.get('name'):
            self.fields[attrs['name']] = attrs.get('value', '')


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    destination = ROOT / 'data/models/adaface'
    upstream = destination / 'upstream'
    upstream.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in ['net.py', 'inference.py', 'README.md', 'LICENSE']:
        url = f'https://raw.githubusercontent.com/mk-minchul/AdaFace/{COMMIT}/{name}'
        response = requests.get(url, timeout=40)
        response.raise_for_status()
        path = upstream / name
        if path.exists() and path.read_bytes() != response.content:
            raise ValueError(f'Existing pinned source differs: {path}')
        path.write_bytes(response.content)
        sources[name] = dict(url=url, sha256=sha(path))
    readme = (upstream / 'README.md').read_text(encoding='utf-8')
    assert f'https://drive.google.com/file/d/{DRIVE_ID}/' in readme
    checkpoint = destination / 'adaface_ir50_ms1mv2.ckpt'
    if not checkpoint.exists():
        url = 'https://drive.usercontent.google.com/download'
        params = dict(id=DRIVE_ID, export='download', confirm='t')
        session = requests.Session()
        response = session.get(url, params=params, stream=True, timeout=(30, 90))
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', ''):
            form = DownloadForm()
            form.feed(response.text)
            if not form.action or not form.action.startswith('https://drive.usercontent.google.com/'):
                raise RuntimeError('Official Google Drive checkpoint download unavailable; no checkpoint saved')
            response.close()
            response = session.get(form.action, params=form.fields, stream=True, timeout=(30, 90))
            response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', ''):
            raise RuntimeError('Received HTML instead of official checkpoint')
        temporary = checkpoint.with_suffix('.ckpt.part')
        received, next_report = 0, 32 * 1024 * 1024
        with temporary.open('wb') as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
                received += len(chunk)
                if received >= next_report:
                    print(f'Checkpoint downloaded: {received / 1024**2:.1f} MiB', flush=True)
                    next_report += 32 * 1024 * 1024
        response.close()
        if received < 100 * 1024 * 1024:
            raise RuntimeError('Checkpoint is unexpectedly small; partial file retained for diagnosis')
        temporary.replace(checkpoint)
    record = dict(repository='https://github.com/mk-minchul/AdaFace', commit=COMMIT,
        architecture='ir_50', training_data='MS1MV2', sources=sources,
        checkpoint_source=f'https://drive.google.com/file/d/{DRIVE_ID}/view',
        checkpoint=checkpoint.relative_to(ROOT).as_posix(), checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=sha(checkpoint), retrieved_utc=datetime.now(timezone.utc).isoformat(),
        source_license='MIT; source license is archived. Do not infer separate dataset or checkpoint permissions from this alone.')
    (destination / 'provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
