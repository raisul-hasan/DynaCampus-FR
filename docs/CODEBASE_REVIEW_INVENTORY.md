# Codebase review inventory

Snapshot date: 23 October 2026. Companion: [previous report comparison](PREVIOUS_REPORT_VS_CURRENT_CODEBASE.md).

This is the pre-cleanup inventory. The temporary root scripts `temp.py`,
`temp_test.py` and `validate_facelivtv2.py` were subsequently removed during
GitHub preparation. Counts and file-status rows below preserve that historical
snapshot; they are not a new inventory of the cleaned checkout.

This inventory covers all Python files in the project root, `src/`, `scripts/` and `tests/`,
all files under `configs/`, the four main notebooks, dependency files, `environment.yml` and `pytest.ini`.
It excludes `.venv`, generated caches, downloaded third-party model source, binary model/media files and
the archived historical supplementary notebook from the first-party file counts. Data manifests, result
receipts, documentation and paper assets were additionally reviewed as evidence, not counted as source files.

**Counts:** 30 src files, 47 scripts, 21 test files, six root Python files = 104 Python files / 13,049 lines.
Adding 17 configs, four notebooks and six environment/dependency files gives 131 inventoried files.
AST syntax parsing succeeds for 102 Python files after decoding the inventory text; two scratch files fail.
This is not proof of runtime correctness. In particular, decoded AST acceptance is not a guarantee that a
UTF-16 scratch file is directly executable by Python.

Core method paths were reviewed in detail. Remaining utility, rendering and test files were inventoried
with imports, definitions and their evidence role. Automated tests and separate numerical verifiers add
behavioral checks; neither substitutes for a security audit or a complete live-camera benchmark.

## File-by-file role and status

| File | Lines | Static syntax | Role / research status |
|---|---:|---|---|
| [app.py](../app.py) | 1071 | OK | Streamlit enrollment, image/video, webcam/RTSP, pending review and gallery management; ArcFace path; live performance unvalidated. |
| [configs/adapter.yaml](../configs/adapter.yaml) | 11 | — | Configuration for adapter; presence alone does not establish a completed experiment. |
| [configs/arcface_baseline.yaml](../configs/arcface_baseline.yaml) | 24 | — | Configuration for arcface baseline; presence alone does not establish a completed experiment. |
| [configs/augmentation.yaml](../configs/augmentation.yaml) | 20 | — | Configuration for augmentation; presence alone does not establish a completed experiment. |
| [configs/backbone_comparison.yaml](../configs/backbone_comparison.yaml) | 22 | — | Configuration for backbone comparison; presence alone does not establish a completed experiment. |
| [configs/chokepoint_temporal.json](../configs/chokepoint_temporal.json) | 28 | — | Configuration for chokepoint temporal; presence alone does not establish a completed experiment. |
| [configs/collection.yaml](../configs/collection.yaml) | 10 | — | Planned campus collection/path settings; not proof of collected data. |
| [configs/decision_ablation.yaml](../configs/decision_ablation.yaml) | 23 | — | Configuration for decision ablation; presence alone does not establish a completed experiment. |
| [configs/enrollment_comparison.yaml](../configs/enrollment_comparison.yaml) | 12 | — | Configuration for enrollment comparison; presence alone does not establish a completed experiment. |
| [configs/enrollment_extensions.json](../configs/enrollment_extensions.json) | 32 | — | Configuration for enrollment extensions; presence alone does not establish a completed experiment. |
| [configs/enrollment_pooling.yaml](../configs/enrollment_pooling.yaml) | 9 | — | Configuration for enrollment pooling; presence alone does not establish a completed experiment. |
| [configs/evaluation.yaml](../configs/evaluation.yaml) | 18 | — | Configuration for evaluation; presence alone does not establish a completed experiment. |
| [configs/final_evaluation.yaml](../configs/final_evaluation.yaml) | 18 | — | Configuration for final evaluation; presence alone does not establish a completed experiment. |
| [configs/gallery.yaml](../configs/gallery.yaml) | 5 | — | Configuration for gallery; presence alone does not establish a completed experiment. |
| [configs/models.yaml](../configs/models.yaml) | 20 | — | Generic model settings; contains stale AdaFace/tracker/adapter labels; run-specific configs are authoritative. |
| [configs/paths.yaml](../configs/paths.yaml) | 12 | — | Planned campus collection/path settings; not proof of collected data. |
| [configs/robustness_development.yaml](../configs/robustness_development.yaml) | 25 | — | Configuration for robustness development; presence alone does not establish a completed experiment. |
| [configs/splits/arcface_baseline_v1.json](../configs/splits/arcface_baseline_v1.json) | 203 | — | Frozen dataset identity roles and manifest fingerprints. |
| [defense_app.py](../defense_app.py) | 82 | OK | Evidence viewer and isolated ten-step real-image gallery demonstration. |
| [environment.yml](../environment.yml) | 8 | — | Conda environment entry point. |
| [launch_dashboard.py](../launch_dashboard.py) | 60 | OK | Dependency/asset availability check and localhost Streamlit launcher. |
| [notebooks/00_environment_check.ipynb](../notebooks/00_environment_check.ipynb) | 78 | — | Early environment/synthetic teaching notebook; not the current quantitative research pipeline. |
| [notebooks/01_detection_and_alignment.ipynb](../notebooks/01_detection_and_alignment.ipynb) | 133 | — | Early environment/synthetic teaching notebook; not the current quantitative research pipeline. |
| [notebooks/02_recognition_embeddings.ipynb](../notebooks/02_recognition_embeddings.ipynb) | 151 | — | Early environment/synthetic teaching notebook; not the current quantitative research pipeline. |
| [notebooks/03_quality_assessment.ipynb](../notebooks/03_quality_assessment.ipynb) | 169 | — | Early environment/synthetic teaching notebook; not the current quantitative research pipeline. |
| [pytest.ini](../pytest.ini) | 2 | — | Limits pytest discovery to tests/, excluding broken root scratch files. |
| [requirements-dev.txt](../requirements-dev.txt) | 5 | — | Dependency declaration/snapshot; runtime and development groups differ. |
| [requirements-lock.txt](../requirements-lock.txt) | 89 | — | Dependency declaration/snapshot; runtime and development groups differ. |
| [requirements.in](../requirements.in) | 3 | — | Dependency declaration/snapshot; runtime and development groups differ. |
| [requirements.txt](../requirements.txt) | 16 | — | Dependency declaration/snapshot; runtime and development groups differ. |
| [scripts/browser_demo.py](../scripts/browser_demo.py) | 108 | OK | Browser automation helper used for evidence collection. |
| [scripts/build_defense_backup.py](../scripts/build_defense_backup.py) | 110 | OK | Assembles backup evidence video; not a new live screen recording. |
| [scripts/check_app_pages.py](../scripts/check_app_pages.py) | 31 | OK | Original Streamlit page smoke checks; not full live-stream verification. |
| [scripts/check_defense_app.py](../scripts/check_defense_app.py) | 44 | OK | Defense app smoke/workflow check. |
| [scripts/check_original_workflow.py](../scripts/check_original_workflow.py) | 120 | OK | Original app upload/enrollment/recognition workflow check using real image bytes. |
| [scripts/compare_backbones.py](../scripts/compare_backbones.py) | 343 | OK | Completed paired real ArcFace/AdaFace development experiment; not held-out. |
| [scripts/compare_decisions.py](../scripts/compare_decisions.py) | 189 | OK | Development score/margin/quality grid selection and refusal/coverage accounting. |
| [scripts/compare_enrollment.py](../scripts/compare_enrollment.py) | 216 | OK | Paired one/few-photo mean-template experiment and known-identity bootstrap. |
| [scripts/compare_enrollment_pooling.py](../scripts/compare_enrollment_pooling.py) | 255 | OK | Development uniform/quality/detector/best-quality pooling ablation. |
| [scripts/compare_transformers.py](../scripts/compare_transformers.py) | 137 | OK | Unfinished FaceLiVT comparison; imports after main invocation and incompatible model/detection APIs. |
| [scripts/evaluate.py](../scripts/evaluate.py) | 139 | OK | Legacy random-vector simulation, not a face-image benchmark or valid evidence for deployed accuracy. |
| [scripts/fetch_adaface_assets.py](../scripts/fetch_adaface_assets.py) | 96 | OK | Pinned official architecture/checkpoint provenance collection. |
| [scripts/fetch_chokepoint.py](../scripts/fetch_chokepoint.py) | 82 | OK | Official crop/annotation retrieval and archive integrity checks. |
| [scripts/fetch_facelivtv2_assets.py](../scripts/fetch_facelivtv2_assets.py) | 97 | OK | New transformer asset fetch; records mutable main ref; not experiment completion. |
| [scripts/fetch_research_data.py](../scripts/fetch_research_data.py) | 54 | OK | Public ICFD sample/access-form download; grants no full-data access. |
| [scripts/finalize_defense_pack.py](../scripts/finalize_defense_pack.py) | 99 | OK | Packages/logs defense evidence and artifact metadata. |
| [scripts/measure_extension_runtime.py](../scripts/measure_extension_runtime.py) | 140 | OK | Direct image-pipeline wall times and actual persistent-gallery/search measurements; excludes camera/UI. |
| [scripts/prepare_dataset_inventory.py](../scripts/prepare_dataset_inventory.py) | 93 | OK | Safe ICFD public-sample extraction and gallery/probe inventory. |
| [scripts/prepare_defense_cases.py](../scripts/prepare_defense_cases.py) | 227 | OK | Paired helped/harmed/failure case selection and separate demo fixtures. |
| [scripts/prepare_halfface_inventory.py](../scripts/prepare_halfface_inventory.py) | 200 | OK | Original/derived separation, hashes, duplicate-ID quarantine and label exclusions. |
| [scripts/rehearse_defense.py](../scripts/rehearse_defense.py) | 35 | OK | Executes isolated defense workflow for demonstration evidence. |
| [scripts/render_backbone_evidence.py](../scripts/render_backbone_evidence.py) | 177 | OK | Generates figures, tables or narrative from saved results; not a new inference experiment. |
| [scripts/render_chokepoint_evidence.py](../scripts/render_chokepoint_evidence.py) | 467 | OK | Generates figures, tables or narrative from saved results; not a new inference experiment. |
| [scripts/render_decision_evidence.py](../scripts/render_decision_evidence.py) | 229 | OK | Generates figures, tables or narrative from saved results; not a new inference experiment. |
| [scripts/render_demo_evidence.py](../scripts/render_demo_evidence.py) | 243 | OK | Generates figures, tables or narrative from saved results; not a new inference experiment. |
| [scripts/render_enrollment_evidence.py](../scripts/render_enrollment_evidence.py) | 219 | OK | Generates figures, tables or narrative from saved results; not a new inference experiment. |
| [scripts/render_enrollment_extensions.py](../scripts/render_enrollment_extensions.py) | 381 | OK | Generates figures, tables or narrative from saved results; not a new inference experiment. |
| [scripts/render_final_evidence.py](../scripts/render_final_evidence.py) | 298 | OK | Generates figures, tables or narrative from saved results; not a new inference experiment. |
| [scripts/render_robustness_evidence.py](../scripts/render_robustness_evidence.py) | 259 | OK | Generates figures, tables or narrative from saved results; not a new inference experiment. |
| [scripts/run_arcface_baseline.py](../scripts/run_arcface_baseline.py) | 313 | OK | Frozen identity partition, real extraction, explicit failures, development score calibration. |
| [scripts/run_chokepoint_temporal.py](../scripts/run_chokepoint_temporal.py) | 366 | OK | Sequence-separated exploratory five-frame study on supplied aligned crops; no live tracker. |
| [scripts/run_enrollment_extensions.py](../scripts/run_enrollment_extensions.py) | 263 | OK | Development-only augmentation, fixed-probe growth and runtime orchestration. |
| [scripts/run_final_evaluation.py](../scripts/run_final_evaluation.py) | 246 | OK | Pre-inference freeze and exclusive one-time reserved-original evaluation; do not rerun for new methods. |
| [scripts/run_robustness.py](../scripts/run_robustness.py) | 266 | OK | Development fixed-threshold image degradation stress study. |
| [scripts/summarize_arcface_baseline.py](../scripts/summarize_arcface_baseline.py) | 93 | OK | Summarizes recorded development baseline; no new accuracy experiment. |
| [scripts/train_adapter.py](../scripts/train_adapter.py) | 101 | OK | Toy noisy random-vector paired training, not real image/domain adaptation evidence. |
| [scripts/validate_dynamic_enrollment.py](../scripts/validate_dynamic_enrollment.py) | 148 | OK | Real-image isolated gallery insertion/reload/removal validation. |
| [scripts/verify_backbone_comparison.py](../scripts/verify_backbone_comparison.py) | 190 | OK | Separate evidence verifier: source/cache/score/decision/summary checks for its named study; inspect receipt scope. |
| [scripts/verify_chokepoint_temporal.py](../scripts/verify_chokepoint_temporal.py) | 338 | OK | Separate evidence verifier: source/cache/score/decision/summary checks for its named study; inspect receipt scope. |
| [scripts/verify_decision_run.py](../scripts/verify_decision_run.py) | 81 | OK | Separate evidence verifier: source/cache/score/decision/summary checks for its named study; inspect receipt scope. |
| [scripts/verify_defense_cases.py](../scripts/verify_defense_cases.py) | 99 | OK | Separate evidence verifier: source/cache/score/decision/summary checks for its named study; inspect receipt scope. |
| [scripts/verify_demo_evidence.py](../scripts/verify_demo_evidence.py) | 93 | OK | Separate evidence verifier: source/cache/score/decision/summary checks for its named study; inspect receipt scope. |
| [scripts/verify_enrollment_extensions.py](../scripts/verify_enrollment_extensions.py) | 285 | OK | Separate evidence verifier: source/cache/score/decision/summary checks for its named study; inspect receipt scope. |
| [scripts/verify_enrollment_pooling.py](../scripts/verify_enrollment_pooling.py) | 151 | OK | Separate evidence verifier: source/cache/score/decision/summary checks for its named study; inspect receipt scope. |
| [scripts/verify_final_evaluation.py](../scripts/verify_final_evaluation.py) | 165 | OK | Separate evidence verifier: source/cache/score/decision/summary checks for its named study; inspect receipt scope. |
| [scripts/verify_robustness_run.py](../scripts/verify_robustness_run.py) | 125 | OK | Separate evidence verifier: source/cache/score/decision/summary checks for its named study; inspect receipt scope. |
| [scripts/verify_setup.py](../scripts/verify_setup.py) | 72 | OK | Dependency/module/device availability checks; not dataset or recognition certification. |
| [src/__init__.py](../src/__init__.py) | 3 | OK | Package initializer/export surface. |
| [src/adapter/__init__.py](../src/adapter/__init__.py) | 3 | OK | Package initializer/export surface. |
| [src/adapter/domain_adapter.py](../src/adapter/domain_adapter.py) | 42 | OK | Residual MLP prototype; no real-domain validation. |
| [src/aggregation/__init__.py](../src/aggregation/__init__.py) | 3 | OK | Package initializer/export surface. |
| [src/aggregation/aggregator.py](../src/aggregation/aggregator.py) | 66 | OK | Per-track mean, best or linear quality-weighted buffering; not norm-softmax experiment. |
| [src/alignment/__init__.py](../src/alignment/__init__.py) | 3 | OK | Package initializer/export surface. |
| [src/alignment/aligner.py](../src/alignment/aligner.py) | 40 | OK | Five-point 112x112 similarity alignment. |
| [src/augmentation/__init__.py](../src/augmentation/__init__.py) | 3 | OK | Package initializer/export surface. |
| [src/augmentation/cctv_augmentor.py](../src/augmentation/cctv_augmentor.py) | 66 | OK | Earlier image-transform utility; distinct from the evaluated enrollment-extension recipe. |
| [src/defense_workflow.py](../src/defense_workflow.py) | 157 | OK | Isolated gallery lifecycle, fresh development inference, cache agreement and checkpoint-hash checks. |
| [src/demo_workflow.py](../src/demo_workflow.py) | 34 | OK | Shared real still-image processing and frozen research decision defaults. |
| [src/detection/__init__.py](../src/detection/__init__.py) | 3 | OK | Package initializer/export surface. |
| [src/detection/detector.py](../src/detection/detector.py) | 79 | OK | SCRFD ONNX boxes, scores and five landmarks; explicit local checkpoint. |
| [src/evaluation/__init__.py](../src/evaluation/__init__.py) | 1 | OK | Package initializer/export surface. |
| [src/evaluation/chokepoint.py](../src/evaluation/chokepoint.py) | 146 | OK | Deterministic degradation, track construction, mean/norm-softmax pooling, calibration and paired intervals. |
| [src/evaluation/enrollment_extensions.py](../src/evaluation/enrollment_extensions.py) | 65 | OK | Fixed development inputs, deterministic enrollment transforms and fixed-probe gallery composition. |
| [src/gallery/__init__.py](../src/gallery/__init__.py) | 3 | OK | Package initializer/export surface. |
| [src/gallery/gallery.py](../src/gallery/gallery.py) | 362 | OK | One/few-photo template persistence, exact NumPy search, pending approval and removal; no SQL backend. |
| [src/open_set/__init__.py](../src/open_set/__init__.py) | 3 | OK | Package initializer/export surface. |
| [src/open_set/decision.py](../src/open_set/decision.py) | 58 | OK | Fixed score, top-two gap and quality gate; four states; main quality gate selected as zero. |
| [src/quality/__init__.py](../src/quality/__init__.py) | 3 | OK | Package initializer/export surface. |
| [src/quality/estimator.py](../src/quality/estimator.py) | 228 | OK | Laplacian blur, native face size and landmark-symmetry heuristics; not learned FQA. |
| [src/recognition/__init__.py](../src/recognition/__init__.py) | 3 | OK | Package initializer/export surface. |
| [src/recognition/adaface.py](../src/recognition/adaface.py) | 105 | OK | Pinned official AdaFace IR-50/MS1MV2; strict checkpoint load and BGR preprocessing; validated development use. |
| [src/recognition/embedder.py](../src/recognition/embedder.py) | 181 | OK | Real 512-D ArcFace ONNX inference; missing-model failure; explicit mock only for selected tests. |
| [src/recognition/facelivtv2.py](../src/recognition/facelivtv2.py) | 104 | OK | FaceLiVTv2 wrapper and RGB conversion; unfinished experimental integration; mutable upstream ref and unrestricted checkpoint loading. |
| [src/tracking/__init__.py](../src/tracking/__init__.py) | 3 | OK | Package initializer/export surface. |
| [src/tracking/tracker.py](../src/tracking/tracker.py) | 206 | OK | Kalman/IoU/Hungarian SORT-like prototype; not ByteTrack; no tracking benchmark. |
| [src/utils/__init__.py](../src/utils/__init__.py) | 1 | OK | Package initializer/export surface. |
| [src/utils/logger.py](../src/utils/logger.py) | 20 | OK | Console logging helper; not a persistent recognition database. |
| [temp.py](../temp.py) | 81 | OK | UTF-16 scratch copy of asset-fetch work; not a research result. |
| [temp_test.py](../temp_test.py) | 1 | ERROR | Transformer scratch probe; syntax error from literal escaped newlines. |
| [tests/test_adaface.py](../tests/test_adaface.py) | 58 | OK | Automated behavior/protocol tests for adaface. |
| [tests/test_adapter.py](../tests/test_adapter.py) | 12 | OK | Automated behavior/protocol tests for adapter. |
| [tests/test_aggregation.py](../tests/test_aggregation.py) | 26 | OK | Automated behavior/protocol tests for aggregation. |
| [tests/test_augmentation.py](../tests/test_augmentation.py) | 12 | OK | Automated behavior/protocol tests for augmentation. |
| [tests/test_backbone_comparison.py](../tests/test_backbone_comparison.py) | 46 | OK | Automated behavior/protocol tests for backbone comparison. |
| [tests/test_baseline_protocol.py](../tests/test_baseline_protocol.py) | 43 | OK | Automated behavior/protocol tests for baseline protocol. |
| [tests/test_chokepoint_protocol.py](../tests/test_chokepoint_protocol.py) | 81 | OK | Automated behavior/protocol tests for chokepoint protocol. |
| [tests/test_decision_ablation.py](../tests/test_decision_ablation.py) | 64 | OK | Automated behavior/protocol tests for decision ablation. |
| [tests/test_defense_cases.py](../tests/test_defense_cases.py) | 35 | OK | Automated behavior/protocol tests for defense cases. |
| [tests/test_detection_alignment.py](../tests/test_detection_alignment.py) | 52 | OK | Automated behavior/protocol tests for detection alignment. |
| [tests/test_dynamic_gallery.py](../tests/test_dynamic_gallery.py) | 72 | OK | Automated behavior/protocol tests for dynamic gallery. |
| [tests/test_embedder_safety.py](../tests/test_embedder_safety.py) | 47 | OK | Automated behavior/protocol tests for embedder safety. |
| [tests/test_enrollment_comparison.py](../tests/test_enrollment_comparison.py) | 43 | OK | Automated behavior/protocol tests for enrollment comparison. |
| [tests/test_enrollment_extensions.py](../tests/test_enrollment_extensions.py) | 58 | OK | Automated behavior/protocol tests for enrollment extensions. |
| [tests/test_enrollment_pooling.py](../tests/test_enrollment_pooling.py) | 67 | OK | Automated behavior/protocol tests for enrollment pooling. |
| [tests/test_final_protocol.py](../tests/test_final_protocol.py) | 76 | OK | Automated behavior/protocol tests for final protocol. |
| [tests/test_open_set.py](../tests/test_open_set.py) | 20 | OK | Automated behavior/protocol tests for open set. |
| [tests/test_quality.py](../tests/test_quality.py) | 93 | OK | Automated behavior/protocol tests for quality. |
| [tests/test_recognition.py](../tests/test_recognition.py) | 84 | OK | Automated behavior/protocol tests for recognition. |
| [tests/test_robustness_protocol.py](../tests/test_robustness_protocol.py) | 54 | OK | Automated behavior/protocol tests for robustness protocol. |
| [tests/test_tracking.py](../tests/test_tracking.py) | 31 | OK | Automated behavior/protocol tests for tracking. |
| [validate_facelivtv2.py](../validate_facelivtv2.py) | 203 | ERROR | Unfinished transformer validation draft; syntax and API/protocol problems; no completed evidence. |

## Reproducibility snapshot

The full path/encoding/size/line/SHA-256 inventory is saved in
`F:/Thesis Final/audit_work/previous_vs_current/code_inventory.csv`.
The JSON companion also records imports and definitions. The extraction, selected PDF page renders,
contact sheets and audit receipt are in the same audit directory. These hashes describe this local
working-tree snapshot, including uncommitted transformer work; they are not represented as a clean commit.

No code repairs were made by this comparison. The report explains which findings affect paper claims.
