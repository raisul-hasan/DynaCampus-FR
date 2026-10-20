"""Safely extract and inventory the official ICFD sample without changing pixels."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = ROOT / 'data' / 'downloads' / 'ICFD_Samples.zip'
    destination = (ROOT / 'data' / 'raw' / 'icfd_sample').resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename.replace('\\', '/')).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Unsafe archive path: {member.filename}')
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Archive contains symlink: {member.filename}')
            content = source.read(member)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f'Existing source differs; refusing to overwrite: {target}')
            if not target.exists():
                target.write_bytes(content)
    rows = []
    for path in sorted(destination.rglob('*')):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        parts = PurePosixPath(path.relative_to(destination).as_posix()).parts
        if len(parts) != 4 or parts[1] not in {'Gallery', 'Probe'}:
            raise ValueError(f'Unrecognized sample layout: {path}')
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        rows.append({
            'dataset': 'icfd_sample', 'subject_id': f'icfd_{parts[2]}',
            'source_role': parts[1].lower(), 'path': path.relative_to(ROOT).as_posix(),
            'width': width, 'height': height,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    if not rows:
        raise ValueError('No source images found')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['sha256']].append(row)
    cross_role_duplicates = [group for group in grouped.values() if len({r['source_role'] for r in group}) > 1]
    manifest_dir = ROOT / 'data' / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / 'icfd_sample.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'dataset': 'icfd_sample', 'access_status': 'sample_only',
        'images': len(rows), 'identities': len({r['subject_id'] for r in rows}),
        'roles': dict(Counter(r['source_role'] for r in rows)),
        'unique_image_hashes': len(grouped),
        'cross_role_duplicate_groups': len(cross_role_duplicates),
        'by_role': {
            role: {
                'images': len(subset), 'identities': len({r['subject_id'] for r in subset}),
                'width_range': [min(r['width'] for r in subset), max(r['width'] for r in subset)],
                'height_range': [min(r['height'] for r in subset), max(r['height'] for r in subset)],
            }
            for role in ('gallery', 'probe')
            for subset in [[r for r in rows if r['source_role'] == role]] if subset
        },
        'notes': 'Official gallery/probe roles preserved; no train/test protocol or accuracy is claimed.'
    }
    output_dir = ROOT / 'results' / 'tables' / 'data_readiness'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'icfd_sample_inventory.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output_dir / 'icfd_sample_counts.csv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(['subject_id', 'gallery_images', 'probe_images'])
        for subject in sorted({r['subject_id'] for r in rows}):
            writer.writerow([subject, *[sum(r['subject_id'] == subject and r['source_role'] == role for r in rows) for role in ('gallery', 'probe')]])
    print(json.dumps(summary, indent=2))
    if cross_role_duplicates:
        raise ValueError('Identical source images cross gallery/probe roles; resolve before evaluation.')


if __name__ == '__main__':
    main()
