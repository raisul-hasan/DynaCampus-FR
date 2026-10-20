"""Index verified defense cards, prior browser captures and the local backup."""
import json
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_arcface_baseline import sha256,write_csv
from src.defense_workflow import default_run


def main():
    run=default_run()
    manifest=json.loads((run/'case_manifest.json').read_text())
    verification=json.loads((run/'verification.json').read_text())
    backup=json.loads((run/'backup_video.json').read_text())
    assert verification['verification']=='passed'
    assert sha256(ROOT/backup['output'])==backup['video_sha256']
    public=ROOT/'paper_assets/development/defense_v1'
    prior=ROOT/'paper_assets/development/demo_v1'
    captures=[]
    for name in ['fig03_app_enrollment.png','fig04_app_known.png','fig05_app_unknown.png','fig06_app_persistence.png','fig07_app_after_removal.png']:
        target=public/('fig_prior_'+name[3:])
        shutil.copyfile(prior/name,target)
        captures.append(dict(source=(prior/name).relative_to(ROOT).as_posix(),destination=target.name,sha256=sha256(target),
            provenance='Genuine prior browser capture, 20 October 2026, single-identity gallery; not a fresh capture of defense_app.py.'))
    case_rows=[]
    for case in manifest['cases']:
        row={k:case[k] for k in ['case_id','model','scope','category','subject_id','truth_status','status','path']}
        row['enrollment_photo_count']=len(case['enrollment_paths'])
        for name in ['one','few']:
            source=case[name]
            for key in ['decision','accepted_id','top1_id','top1_score','top2_score','correct_known_accept']:
                row[name+'_'+key]=source[key]
            row[name+'_threshold']=case[name+'_threshold']
            row[name+'_margin_threshold']=case[name+'_margin']
        case_rows.append(row)
    write_csv(public/'case_summary.csv',case_rows)
    shutil.copyfile(ROOT/'docs/DEFENSE_CASE_ANALYSIS.md',public/'CASE_ANALYSIS.md')
    lines=['# Defense evidence pack', '', 'Prepared 21 October 2026. All case examples are development data. The frozen final evaluation remains unchanged.', '',
        '## Open the deliverables', '',
        '- Run the defense dashboard from the repository: `python -m streamlit run defense_app.py --server.address 127.0.0.1 --server.port 8521 --server.headless true`.',
        '- Open `http://127.0.0.1:8521`. The dashboard is local and reads the existing local dataset and checkpoints.',
        f"- Offline viewer with original photographs: `{manifest['private_directory']}/index.html`. This stays in ignored data/ and is not a public paper asset.",
        f"- Backup video: `{backup['output']}` ({backup['decoded_duration_seconds']:.0f} seconds, H.264, 1920 x 1080).",
        '- The backup is an assembled walkthrough of redacted evidence cards, result charts and prior genuine screenshots. It is not a fresh live screen recording.', '',
        '## Twelve disclosed cases', '',
        '[Visual analysis and decision explanations](CASE_ANALYSIS.md) | [Full-precision comparison table](case_summary.csv)', '',
        '| Case | Outcome | Model | Scope | Redacted evidence |', '|---|---|---|---|---|']
    for case in manifest['cases']:
        lines.append(f"| {case['case_id']} | {case['category'].replace('_',' ')} | {case['model']} | {case['scope']} | [Card](fig_case_{case['case_id']}.png) |")
    lines += ['', 'Cases were selected as the first lexicographic path in each category before inspection. '
        f"The complete eligible ledger contains {verification['candidate_records']} category records across the declared comparison scopes. "
        'Repeated probes across models/scopes are not independent samples. Fixed-rule examples use identical score/margin settings; separately calibrated examples explicitly display both operating points.', '',
        '## Repeatable live workflow', '',
        'Choose **Live gallery walkthrough**, select **Start isolated walkthrough**, then select **Run next walkthrough step** ten times. '
        'The sequence is: target unknown; enroll three original photos; target known; permanent unknown rejected; genuine ambiguity; reload; target still known; remove target; target unknown again; capture failure.', '',
        'Each session gets a new gallery under ignored data/defense. The 54 background identities are initialized from verified real development features. '
        'Every target enrollment photo and probe is freshly processed through SCRFD, alignment and ArcFace. '
        'Fresh successful features match the original cache exactly, and checkpoint hashes remain unchanged. '
        'The ambiguous search has two distinct enrolled candidates; no artificial duplicate is inserted.', '',
        'This uses the same core gallery and still-image recognition functions as app.py. It is a separate defense entry point; the original app and its evidence hashes remain intact.', '',
        '## Genuine browser screenshots already available', '',
        'These are reused, hash-checked captures from the earlier actual browser upload demonstration. '
        'That demonstration used one gallery identity, so it does not illustrate the new multi-identity ambiguity case. They are not synthetic UI mockups.', '',
        '- [Enrollment](fig_prior_03_app_enrollment.png)', '- [Known masked probe](fig_prior_04_app_known.png)',
        '- [Unknown probe](fig_prior_05_app_unknown.png)', '- [Persistence](fig_prior_06_app_persistence.png)',
        '- [After removal](fig_prior_07_app_after_removal.png)', '',
        '## Checks completed', '',
        '- Twelve selected cases, all exact source records, eligible first-path selection and original image hashes verified.',
        '- Ten real backend steps replayed and checked; target/probes freshly inferred; gallery insertion/reload/removal and unchanged models verified.',
        '- Streamlit AppTest checked every case view, results view and all ten live workflow controls with real inference.',
        '- Five new outcome/protocol tests pass.',
        '- Existing browser evidence independently rechecked: 1,824 gallery-stage attempts, uploaded template, displayed scores and five screenshot states.',
        '- Every backup-video chapter decodes successfully, with duration matched to the chapter manifest.', '',
        '## Remaining browser evidence', '',
        'The session currently exposes no connected browser. Fresh screenshot capture and a live screen recording of the new defense dashboard remain pending. '
        'Application-test checks are not described as browser capture. Connect the browser, open the local dashboard, then capture the ten steps. '
        'The backup video remains useful offline and is labelled accurately.', '',
        '## Verify and reproduce', '', '```powershell', 'python scripts/prepare_defense_cases.py',
        'python scripts/rehearse_defense.py','python scripts/check_defense_app.py','python -m unittest discover -s tests -p test_defense_cases.py -v',
        'python scripts/verify_defense_cases.py', '# Encoding helper stays inside ignored data/tools:',
        'python -m pip download imageio-ffmpeg --no-deps --dest data/tools',
        'python scripts/build_defense_backup.py','python scripts/finalize_defense_pack.py','```', '',
        f"Raw records: `{run.relative_to(ROOT).as_posix()}`. The source_run.json pointer selects that run for the dashboard and verifiers.", '',
        'Do not report selected examples as accuracy samples, infer causal explanations from appearance alone, or present these still-image tests as real CCTV validation.']
    (public/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt=dict(source_run=run.relative_to(ROOT).as_posix(),verification_sha256=sha256(run/'verification.json'),
        backup_video_receipt_sha256=sha256(run/'backup_video.json'),prior_browser_captures=captures,
        public_assets_have_no_photographs=True,new_browser_capture=False,
        asset_sha256={p.name:sha256(p) for p in public.iterdir() if p.is_file() and p.name!='asset_manifest.json'},
        script_sha256=sha256(Path(__file__)))
    (public/'asset_manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(dict(public_index=(public/'README.md').relative_to(ROOT).as_posix(),backup=backup['output'],
        photo_viewer=manifest['private_directory']+'/index.html',new_browser_capture=False),indent=2))


if __name__=='__main__':
    main()
