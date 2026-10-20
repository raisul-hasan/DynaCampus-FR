"""Exercise unchanged app.py with AppTest and injected UploadedFile objects.

This verifies application behavior, not browser transport or screenshots.
Only the file-uploader boundary is substituted; inference and persistence are real.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from streamlit.testing.v1 import AppTest
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.proto.Common_pb2 import FileURLs
from src.defense_workflow import DefenseWorkflow
from src.gallery import FaceGallery
from scripts.run_arcface_baseline import sha256


def snapshot(directory):
    return {str(p.relative_to(directory)): sha256(p) for p in directory.rglob('*') if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    normal_before = snapshot(ROOT / 'gallery_data')
    workflow = DefenseWorkflow()
    fixture = workflow.fixture
    os.environ['DYNACAMPUS_GALLERY_DIR'] = str(workflow.gallery_path)
    uploads = {}
    events = []

    def uploader(label, *unused, **options):
        paths = uploads.get(label, [])
        files = [UploadedFile(UploadedFileRec(str(i), Path(p).name, 'image/jpeg', (ROOT/p).read_bytes()), FileURLs()) for i, p in enumerate(paths)]
        return files if options.get('accept_multiple_files') else (files[0] if files else None)

    def checked(app):
        assert not app.exception, [e.message for e in app.exception]
        assert not app.error, [e.value for e in app.error]
        return app

    with patch('streamlit.file_uploader', side_effect=uploader):
        app = checked(AppTest.from_file(str(ROOT/'app.py')).run(timeout=90))
        pages = list(app.sidebar.radio[0].options)
        for page in pages:
            checked(app.sidebar.radio[0].set_value(page).run(timeout=60))
        checked(app.sidebar.radio[1].set_value('Manual demo settings').run(timeout=60))
        assert len(app.sidebar.slider) == 3
        checked(app.sidebar.radio[1].set_value('Frozen research settings').run(timeout=60))
        checked(app.sidebar.checkbox[0].set_value(True).run(timeout=60))

        def page_containing(word):
            checked(app.sidebar.radio[0].set_value(next(p for p in pages if word in p)).run(timeout=60))

        def recognize(path, expected):
            uploads['Upload Image'] = [path]
            page_containing('Image')
            table = app.dataframe[-1].value
            row = table.iloc[0].to_dict()
            assert row['Decision'] == expected, row
            events.append(dict(action='recognize', path=path, **row))
            return row

        recognize(fixture['known_probe'], 'UNKNOWN')
        page_containing('Enroll')
        app.text_input[0].set_value(fixture['display_name'])
        app.text_input[1].set_value(fixture['subject_id'])
        uploads['Upload Photo(s)'] = fixture['enrollment_paths']
        checked(app.run(timeout=90))
        checked(next(b for b in app.button if 'Confirm Enrollment' in b.label).click().run(timeout=90))
        gallery = app.session_state['gallery']
        assert gallery.size == 55
        workflow.perform('enroll')  # independent extraction and template check
        # Both instances share the isolated folder; compare the app's in-memory vector.
        np.testing.assert_allclose(gallery._embeddings[fixture['subject_id']], workflow.gallery._embeddings[fixture['subject_id']], atol=1e-7)
        events.append(dict(action='enroll', gallery_size=gallery.size, photos=3, independent_template_matches=True))
        known = recognize(fixture['known_probe'], 'KNOWN')
        assert abs(known['Score'] - fixture['expected_after']['top1_score']) < 1e-6
        recognize(fixture['unknown_probe'], 'UNKNOWN')
        recognize(fixture['ambiguous_probe'], 'AMBIGUOUS')
        reloaded = FaceGallery(workflow.gallery_path)
        np.testing.assert_array_equal(reloaded._embeddings[fixture['subject_id']], gallery._embeddings[fixture['subject_id']])
        app.session_state['gallery'] = reloaded
        recognize(fixture['known_probe'], 'KNOWN')
        events.append(dict(action='reload', exact_template_preserved=True))
        page_containing('Gallery')
        checked(next(b for b in app.button if b.key == 'del_'+fixture['subject_id']).click().run(timeout=90))
        assert app.session_state['gallery'].size == 54
        assert fixture['subject_id'] not in FaceGallery(workflow.gallery_path)._embeddings
        events.append(dict(action='remove', gallery_size=54, target_absent_on_disk=True))
        recognize(fixture['known_probe'], 'UNKNOWN')
        uploads['Upload Image'] = [fixture['capture_failure_probe']]
        page_containing('Image')
        assert any('No faces detected' in w.value for w in app.warning)
        events.append(dict(action='capture_failure', reported=True))

    assert normal_before == snapshot(ROOT / 'gallery_data'), 'Normal user gallery changed'
    report = dict(utc=datetime.now(timezone.utc).isoformat(), passed=True,
                  mechanism='Streamlit AppTest; only file_uploader substituted with real UploadedFile bytes',
                  fresh_browser_capture=False, pages=pages, presets_checked=True,
                  real_detector_and_embedder=True, normal_user_gallery_unchanged=True,
                  gallery=str(workflow.gallery_path.relative_to(ROOT)), events=events,
                  app_sha256=sha256(ROOT/'app.py'), model_sha256=workflow.model_hashes,
                  models_unchanged=all(sha256(Path(p)) == h for p,h in workflow.model_hashes.items()))
    assert report['models_unchanged']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
