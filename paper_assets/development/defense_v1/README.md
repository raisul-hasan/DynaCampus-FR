# Defense evidence pack

Prepared 21 October 2026. All case examples are development data. The frozen final evaluation remains unchanged.

## Open the deliverables

- Run the defense dashboard from the repository: `python -m streamlit run defense_app.py --server.address 127.0.0.1 --server.port 8521 --server.headless true`.
- Open `http://127.0.0.1:8521`. The dashboard is local and reads the existing local dataset and checkpoints.
- Offline viewer with original photographs: `data/defense/defense_development_20261020T210142299692Z/index.html`. This stays in ignored data/ and is not a public paper asset.
- Backup video: `data/defense/defense_development_20261020T210142299692Z/defense_evidence_backup.mp4` (149 seconds, H.264, 1920 x 1080).
- The backup is an assembled walkthrough of redacted evidence cards, result charts and prior genuine screenshots. It is not a fresh live screen recording.

## Twelve disclosed cases

[Visual analysis and decision explanations](CASE_ANALYSIS.md) | [Full-precision comparison table](case_summary.csv)

| Case | Outcome | Model | Scope | Redacted evidence |
|---|---|---|---|---|
| C01 | helped known | arcface | fixed_rule_sensitivity | [Card](fig_case_C01.png) |
| C02 | harmed known | arcface | fixed_rule_sensitivity | [Card](fig_case_C02.png) |
| C03 | unchanged accept | arcface | fixed_rule_sensitivity | [Card](fig_case_C03.png) |
| C04 | unknown rejected | arcface | fixed_rule_sensitivity | [Card](fig_case_C04.png) |
| C05 | ambiguous unknown | arcface | fixed_rule_sensitivity | [Card](fig_case_C05.png) |
| C06 | known rejected | arcface | fixed_rule_sensitivity | [Card](fig_case_C06.png) |
| C07 | capture failure | arcface | fixed_rule_sensitivity | [Card](fig_case_C07.png) |
| C08 | unknown false accept | arcface | fixed_rule_sensitivity | [Card](fig_case_C08.png) |
| C09 | helped known | arcface | separately_calibrated_policies | [Card](fig_case_C09.png) |
| C10 | harmed known | arcface | separately_calibrated_policies | [Card](fig_case_C10.png) |
| C11 | helped known | adaface | separately_calibrated_policies | [Card](fig_case_C11.png) |
| C12 | harmed known | adaface | separately_calibrated_policies | [Card](fig_case_C12.png) |

Cases were selected as the first lexicographic path in each category before inspection. The complete eligible ledger contains 905 category records across the declared comparison scopes. Repeated probes across models/scopes are not independent samples. Fixed-rule examples use identical score/margin settings; separately calibrated examples explicitly display both operating points.

## Repeatable live workflow

Choose **Live gallery walkthrough**, select **Start isolated walkthrough**, then select **Run next walkthrough step** ten times. The sequence is: target unknown; enroll three original photos; target known; permanent unknown rejected; genuine ambiguity; reload; target still known; remove target; target unknown again; capture failure.

Each session gets a new gallery under ignored data/defense. The 54 background identities are initialized from verified real development features. Every target enrollment photo and probe is freshly processed through SCRFD, alignment and ArcFace. Fresh successful features match the original cache exactly, and checkpoint hashes remain unchanged. The ambiguous search has two distinct enrolled candidates; no artificial duplicate is inserted.

This uses the same core gallery and still-image recognition functions as app.py. It is a separate defense entry point; the original app and its evidence hashes remain intact.

## Genuine browser screenshots already available

These are reused, hash-checked captures from the earlier actual browser upload demonstration. That demonstration used one gallery identity, so it does not illustrate the new multi-identity ambiguity case. They are not synthetic UI mockups.

- [Enrollment](fig_prior_03_app_enrollment.png)
- [Known masked probe](fig_prior_04_app_known.png)
- [Unknown probe](fig_prior_05_app_unknown.png)
- [Persistence](fig_prior_06_app_persistence.png)
- [After removal](fig_prior_07_app_after_removal.png)

## Checks completed

- Twelve selected cases, all exact source records, eligible first-path selection and original image hashes verified.
- Ten real backend steps replayed and checked; target/probes freshly inferred; gallery insertion/reload/removal and unchanged models verified.
- Streamlit AppTest checked every case view, results view and all ten live workflow controls with real inference.
- Five new outcome/protocol tests pass.
- Existing browser evidence independently rechecked: 1,824 gallery-stage attempts, uploaded template, displayed scores and five screenshot states.
- Every backup-video chapter decodes successfully, with duration matched to the chapter manifest.

## Remaining browser evidence

The session currently exposes no connected browser. Fresh screenshot capture and a live screen recording of the new defense dashboard remain pending. Application-test checks are not described as browser capture. Connect the browser, open the local dashboard, then capture the ten steps. The backup video remains useful offline and is labelled accurately.

## Verify and reproduce

```powershell
python scripts/prepare_defense_cases.py
python scripts/rehearse_defense.py
python scripts/check_defense_app.py
python -m unittest discover -s tests -p test_defense_cases.py -v
python scripts/verify_defense_cases.py
# Encoding helper stays inside ignored data/tools:
python -m pip download imageio-ffmpeg --no-deps --dest data/tools
python scripts/build_defense_backup.py
python scripts/finalize_defense_pack.py
```

Raw records: `results/defense/defense_development_20261020T210142299692Z`. The source_run.json pointer selects that run for the dashboard and verifiers.

Do not report selected examples as accuracy samples, infer causal explanations from appearance alone, or present these still-image tests as real CCTV validation.
