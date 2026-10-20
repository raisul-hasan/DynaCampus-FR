"""Execute every real backend walkthrough step; this is not a browser recording."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.defense_workflow import DefenseWorkflow, STEPS
from scripts.run_arcface_baseline import sha256


def main():
    workflow = DefenseWorkflow()
    events = []
    for _ in STEPS:
        event = workflow.next_step()
        events.append(event)
        print(json.dumps(event), flush=True)
    expected = ['UNKNOWN', None, 'KNOWN', 'UNKNOWN', 'AMBIGUOUS', None, 'KNOWN', None, 'UNKNOWN', 'CAPTURE_FAILURE']
    assert [r.get('decision') for r in events] == expected
    assert [r['gallery_size'] for r in events] == [54, 55, 55, 55, 55, 55, 55, 54, 54, 54]
    assert all(r['backbone_checkpoints_unchanged'] for r in events)
    assert events[2]['accepted_id'] == events[6]['accepted_id'] == workflow.fixture['subject_id']
    assert workflow.gallery.size == 54
    receipt = dict(verification='passed', session_directory=workflow.directory.relative_to(ROOT).as_posix(),
        steps=len(events), real_images_reinferred=True, background_seed_templates=54,
        checkpoint_sha256=workflow.model_hashes, state_sequence=expected,
        browser_interaction_tested=False, screenshots_created=False, screen_recording_created=False,
        event_log_sha256=sha256(workflow.directory/'events.jsonl'),
        source_sha256={p: sha256(ROOT/p) for p in ['src/defense_workflow.py', 'scripts/rehearse_defense.py', 'src/demo_workflow.py', 'src/gallery/gallery.py']})
    (workflow.run/'backend_rehearsal.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
