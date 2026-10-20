"""Inventory the downloaded halfFace archive and extract original full-face images.

Derived upper-face and augmented collections stay archived. No train/test split
is made. All raw images and per-image metadata remain under ignored data/.
"""
import csv
import hashlib
import io
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_COLLECTION = 'Dataset1_Full Facial Image'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}
LABEL = re.compile(r'p(?P<subject>\d{3})_(?P<masked>[01])(?P<pose>[012])[mf]', re.I)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    download_dir = ROOT / 'data' / 'downloads' / 'halfface'
    candidates = sorted(download_dir.glob('*.zip'))
    if len(candidates) != 1:
        raise ValueError('Expected exactly one original halfFace Download All ZIP in data/downloads/halfface/')
    outer_path = candidates[0]
    with ZipFile(outer_path) as outer:
        nested = [member for member in outer.infolist() if member.filename.lower().endswith('.zip')]
        if len(nested) != 1:
            raise ValueError('Expected the official wrapper ZIP with one nested dataset ZIP')
        member = nested[0]
        inner_path = download_dir / 'unpacked' / PurePosixPath(member.filename).name
        inner_path.parent.mkdir(parents=True, exist_ok=True)
        if inner_path.exists():
            if inner_path.stat().st_size != member.file_size:
                raise ValueError('Existing inner archive is incomplete; refusing to overwrite it')
        else:
            with outer.open(member) as source, inner_path.open('xb') as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        readmes = [m for m in outer.infolist() if m.filename.lower().endswith('.txt')]
        source_dir = ROOT / 'data' / 'sources'
        source_dir.mkdir(parents=True, exist_ok=True)
        for index, readme in enumerate(readmes):
            (source_dir / f'halfface_readme_{index + 1}.txt').write_bytes(outer.read(readme))

    manifest_dir = ROOT / 'data' / 'manifests'
    destination = (ROOT / 'data' / 'raw' / 'halfface' / 'full_face').resolve()
    rows, invalid_images, label_review = [], [], []
    with ZipFile(inner_path) as inner:
        bad_member = inner.testzip()
        if bad_member is not None:
            raise ValueError(f'Archive CRC failure: {bad_member}')
        images = [m for m in inner.infolist() if PurePosixPath(m.filename).suffix.lower() in IMAGE_SUFFIXES]
        collection_counts = Counter(PurePosixPath(m.filename).parts[1] for m in images)
        for member in images:
            parts = PurePosixPath(member.filename).parts
            if len(parts) < 2 or parts[1] != ORIGINAL_COLLECTION:
                continue
            if len(parts) != 4 or not re.fullmatch(r'p\d{3}', parts[2], re.I):
                raise ValueError(f'Unrecognized original-image layout: {member.filename}')
            folder_id = parts[2][1:]
            target = (destination / parts[2] / parts[3]).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Unsafe image path: {member.filename}')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f'Symlink in source archive: {member.filename}')
            content = inner.read(member)
            sha256 = hashlib.sha256(content).hexdigest()
            try:
                with Image.open(io.BytesIO(content)) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
            except (OSError, ValueError) as error:
                invalid_images.append({'source_member': member.filename, 'reason': str(error)})
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest_file(target) != sha256:
                    raise ValueError(f'Existing original differs; refusing to overwrite: {target}')
            else:
                target.write_bytes(content)
            match = LABEL.fullmatch(PurePosixPath(member.filename).stem)
            if match and match['subject'] != folder_id:
                raise ValueError(f'Folder and filename identity conflict: {member.filename}')
            row = {
                'dataset': 'halfface', 'collection': 'original_full_face',
                'subject_id': f'halfface_{folder_id}', 'path': target.relative_to(ROOT).as_posix(),
                'source_member': member.filename, 'width': width, 'height': height, 'mode': mode,
                'mask_status': ('masked' if match['masked'] == '1' else 'unmasked') if match else 'unknown',
                'pose': {'0': 'front', '1': 'left', '2': 'right'}[match['pose']] if match else 'unknown',
                'condition_label_status': 'filename_documented' if match else 'needs_review',
                'sha256': sha256,
            }
            rows.append(row)
            if not match:
                label_review.append({k: row[k] for k in ('subject_id', 'path', 'condition_label_status')})

    if not rows:
        raise ValueError('No original full-face images found')
    hashes = defaultdict(list)
    for row in rows:
        hashes[row['sha256']].append(row)
    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    cross_identity_groups = [g for g in duplicate_groups if len({r['subject_id'] for r in g}) > 1]
    # Quarantine conflicting IDs rather than guessing which source identity is right.
    conflicting_ids = {r['subject_id'] for group in cross_identity_groups for r in group}
    excluded, usable = [], []
    for row in rows:
        reasons = []
        if row['subject_id'] in conflicting_ids:
            reasons.append('identity_has_exact_images_shared_with_another_source_id')
        if row['condition_label_status'] == 'needs_review':
            reasons.append('nonstandard_filename_condition_is_unknown')
        if reasons:
            excluded.append({'subject_id': row['subject_id'], 'path': row['path'], 'reason': ';'.join(reasons)})
        else:
            usable.append(row)
    if len({r['sha256'] for r in usable}) != len(usable):
        raise ValueError('Exact duplicates remain in usable images; resolve before evaluation')
    write_csv(manifest_dir / 'halfface_full_face.csv', rows, list(rows[0]))
    write_csv(manifest_dir / 'halfface_usable.csv', usable, list(rows[0]))
    write_csv(manifest_dir / 'halfface_exclusions.csv', excluded, ['subject_id', 'path', 'reason'])
    duplicate_rows = [
        {'duplicate_group': index, 'subject_id': row['subject_id'], 'path': row['path'], 'sha256': row['sha256']}
        for index, group in enumerate(duplicate_groups, start=1) for row in group
    ]
    write_csv(manifest_dir / 'halfface_duplicates.csv', duplicate_rows,
              ['duplicate_group', 'subject_id', 'path', 'sha256'])
    write_csv(manifest_dir / 'halfface_condition_review.csv', label_review,
              ['subject_id', 'path', 'condition_label_status'])
    identities = sorted({r['subject_id'] for r in rows})
    by_identity = []
    for subject in identities:
        subset = [r for r in rows if r['subject_id'] == subject]
        by_identity.append({
            'subject_id': subject, 'original_images': len(subset),
            'unmasked_images': sum(r['mask_status'] == 'unmasked' for r in subset),
            'masked_images': sum(r['mask_status'] == 'masked' for r in subset),
            'unknown_condition_images': sum(r['mask_status'] == 'unknown' for r in subset),
        })
    summary = {
        'dataset': 'halfface', 'source_doi': '10.17632/pk44mkx9vm.2',
        'original_archive': outer_path.relative_to(ROOT).as_posix(),
        'original_archive_sha256': digest_file(outer_path),
        'inner_archive_crc': 'passed', 'archive_image_counts': dict(collection_counts),
        'prepared_collection': ORIGINAL_COLLECTION, 'images': len(rows), 'identities': len(identities),
        'mask_status_counts': dict(Counter(r['mask_status'] for r in rows)),
        'pose_counts': dict(Counter(r['pose'] for r in rows)),
        'images_per_identity_distribution': dict(sorted(Counter(r['original_images'] for r in by_identity).items())),
        'unique_image_hashes': len(hashes), 'exact_duplicate_groups': len(duplicate_groups),
        'cross_identity_duplicate_groups': len(cross_identity_groups),
        'condition_labels_needing_review': len(label_review), 'invalid_image_count': len(invalid_images),
        'usable_images': len(usable), 'usable_identities': len({r['subject_id'] for r in usable}),
        'excluded_images': len(excluded), 'excluded_conflicting_identity_ids': sorted(conflicting_ids),
        'usable_mask_status_counts': dict(Counter(r['mask_status'] for r in usable)),
        'preparation_status': 'ready_with_documented_exclusions' if not invalid_images else 'invalid_images_require_review',
        'width_range': [min(r['width'] for r in rows), max(r['width'] for r in rows)],
        'height_range': [min(r['height'] for r in rows), max(r['height'] for r in rows)],
        'people_with_unmasked_and_masked_images': sum(r['unmasked_images'] > 0 and r['masked_images'] > 0 for r in by_identity),
        'maximum_documented_unmasked_images_per_identity': max(r['unmasked_images'] for r in by_identity),
        'notes': [
            'Only original full-face images were prepared; upper-face and augmented versions remain archived.',
            'Nonstandard source filenames are flagged rather than guessing their condition labels.',
            'All conflicting source identities and images with unknown condition labels are excluded from the usable manifest; original files remain intact.',
            'Identity comes from source folders; masks and pose come from documented filenames, not inferred appearance.',
            'No final evaluation split or accuracy result has been created. Exact hashes do not rule out near-duplicates.',
        ],
    }
    output_dir = ROOT / 'results' / 'tables' / 'data_readiness'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'halfface_inventory.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    write_csv(output_dir / 'halfface_counts.csv', by_identity, list(by_identity[0]))
    (manifest_dir / 'halfface_invalid_images.json').write_text(json.dumps(invalid_images, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    if invalid_images:
        raise ValueError('Invalid images found; inspect manifests before evaluation')


if __name__ == '__main__':
    main()
