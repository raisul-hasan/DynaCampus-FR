"""Exercise the defense Streamlit UI with its app-test API, without a browser."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from streamlit.testing.v1 import AppTest
from scripts.run_arcface_baseline import sha256
from src.defense_workflow import default_run


def click(app, text):
    next(button for button in app.button if button.label == text).click().run(timeout=60)
    assert not app.exception, [e.message for e in app.exception]


def main():
    app = AppTest.from_file(str(ROOT/'defense_app.py')).run(timeout=60)
    assert not app.exception
    cases = len(app.selectbox[0].options)
    for option in app.selectbox[0].options:
        app.selectbox[0].set_value(option).run(timeout=30)
        assert not app.exception
    app.sidebar.radio[0].set_value('Study results').run(timeout=30)
    assert not app.exception
    app.sidebar.radio[0].set_value('Live gallery walkthrough').run(timeout=30)
    click(app, 'Start isolated walkthrough')
    states = []
    for _ in range(10):
        click(app, 'Run next walkthrough step')
        states.append(app.session_state['last_defense_event'].get('decision'))
    assert states == ['UNKNOWN', None, 'KNOWN', 'UNKNOWN', 'AMBIGUOUS', None, 'KNOWN', None, 'UNKNOWN', 'CAPTURE_FAILURE']
    assert not any(b.label == 'Run next walkthrough step' for b in app.button)
    workflow = app.session_state['defense_workflow']
    report = dict(verification='passed', case_views_checked=cases, real_walkthrough_steps=10, state_sequence=states,
        isolated_gallery_size_after=workflow.gallery.size, session_directory=workflow.directory.relative_to(ROOT).as_posix(),
        real_inference=True, app_test_api_used=True, actual_browser_interaction=False,
        source_sha256={p:sha256(ROOT/p) for p in ['defense_app.py','src/defense_workflow.py','scripts/check_defense_app.py']})
    (default_run()/'app_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
