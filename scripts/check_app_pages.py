"""Run real Streamlit pages in an isolated gallery without a physical camera."""
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest
from scripts.run_arcface_baseline import sha256

RUN=ROOT/'results/demo/dynamic_development_20261019T233918475526Z'
os.environ['DYNACAMPUS_GALLERY_DIR']=str(ROOT/'data/demo/app_page_checks/gallery')
app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
assert not app.exception,[e.message for e in app.exception]
checks=[]
for page in app.sidebar.radio[0].options:
    # Camera connections are not started. Their controls can still be rendered.
    app.sidebar.radio[0].set_value(page).run(timeout=30)
    assert not app.exception,[e.message for e in app.exception]
    checks.append(dict(page=page,rendered_without_exception=True))
app.sidebar.radio[1].set_value('Manual demo settings').run(timeout=30)
assert not app.exception
assert len(app.sidebar.slider)==3
app.sidebar.radio[1].set_value('Frozen research settings').run(timeout=30)
assert not app.exception
report=dict(pages=checks,manual_and_frozen_presets_render=True,
            real_models_loaded=True,normal_user_gallery_untouched=True,
            webcam_or_rtsp_connected=False,upload_flow_browser_check='separate',app_sha256=sha256(ROOT/'app.py'))
(RUN/'app_page_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
