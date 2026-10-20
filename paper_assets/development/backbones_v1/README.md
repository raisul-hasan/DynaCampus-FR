# Paired AdaFace and ArcFace development evidence

These comparisons use the same original images, detection outcomes and aligned crops.
No held-out inference, recognition training, domain adapter or final-method replacement occurred.

| Model | Enrollment | Decision | Correct known / 154 | False known / extracted unknown |
|---|---|---|---:|---:|
| arcface | one_photo | score_only | 129/154 | 1/149 |
| arcface | one_photo | score_margin | 139/154 | 1/149 |
| arcface | up_to_three_photos | score_only | 138/154 | 1/149 |
| arcface | up_to_three_photos | score_margin | 139/154 | 1/149 |
| adaface | one_photo | score_only | 112/154 | 1/149 |
| adaface | one_photo | score_margin | 120/154 | 1/149 |
| adaface | up_to_three_photos | score_only | 122/154 | 1/149 |
| adaface | up_to_three_photos | score_margin | 133/154 | 0/149 |

## Main finding

With score-only decisions, few-photo enrollment changes correct known acceptance from 129 to 138 for ArcFace and from 112 to 122 for AdaFace. With separately calibrated score-plus-margin rules, ArcFace changes from 139 to 139, while AdaFace changes from 120 to 133. The enrollment benefit must be assessed jointly with the checkpoint and decision rule.

The few-photo margin comparison has 139/154 correct known acceptances and 1/149 false unknown accepts for ArcFace, versus 133/154 and 0/149 for AdaFace. Extracted-probe decision coverage is 93.27% for AdaFace and 99.33% for ArcFace. Both obey the same empirical FPIR limit, but their observed FPIR and coverage are not identical. Do not interpret the acceptance counts alone as an overall superiority test.

The original frozen final evaluation remains separate. These new development results do not replace it, and the newly calibrated one-photo margin policy was not one of its predeclared final methods. It must not be retrospectively described as a held-out method.

## What was measured

Paired extraction covers 571 development images, with 7 shared capture failures. ArcFace features reproduce 564 original cached embeddings; maximum difference 0.

Each halfFace comparison includes 55 enrolled identities, 154 masked known attempts and 150 unknown attempts. Six known and one unknown capture failures stay in attempted denominators. Unknown FPIR uses all 149 successfully extracted unknown probes, including refused inputs.

Each model/enrollment/decision family is calibrated independently using the same declared objective and margin grid. Threshold selection and evaluation reuse development data, so these results are descriptive rather than independent generalization estimates.

ICFD is reported separately in summary.csv: five development identities and 100 closed-set public-sample probes; no unknown people.

## Models and attribution

[Official AdaFace source and checkpoints](https://github.com/mk-minchul/AdaFace): IR-50 pretrained on MS1MV2. Source commit c60eaa786a42c03444f3df7096dbaf9d57ae010d is pinned and archived locally with its MIT source license. The checkpoint has SHA-256 234da6ce931f821e0a8e920063dc2091d8122ea46285c159d2f25d3f762fecb3.

[Official InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo): buffalo_l ResNet50 pretrained on WebFace600K. The model is for non-commercial research. The two checkpoints have different pretraining datasets and runtime engines; this does not isolate the effect of the AdaFace training loss.

Both use the existing SCRFD five-point alignment for a controlled crop comparison. AdaFace receives uint8 BGR crops normalized using (value/255 - 0.5)/0.5; ArcFace uses its existing InsightFace preprocessing. No flip fusion or test-time augmentation is applied.

Real-crop preprocessing and output match the official AdaFace reference. Strict state loading is used. A legacy Lightning callback metadata key is mapped to an inert local class while keeping weights-only loading; no Lightning training code runs.

## Figures and captions

1. **[Development comparison](fig01_backbone_comparison.png)** ([SVG](fig01_backbone_comparison.svg)). Correct known acceptance and false-known counts for the same probes, with failures retained. Each checkpoint and policy has its own development-calibrated thresholds. This is not a held-out result or evidence of an isolated loss-function advantage.

2. **[CPU timing](fig02_backbone_timing.png)** ([SVG](fig02_backbone_timing.svg)). Median bars and p95 markers for successful halfFace extraction, using two CPU threads and five warm-up calls. The composed latency sums measured decode, padding, detection, alignment, quality and embedding stages; it excludes search, gallery persistence, UI, model loading and source-hash checks. It is not live-video FPS. Source hashing precedes decoding, so decode timings likely benefit from a warm file cache. The local processor was identified as AMD Ryzen 5 3600; timings reflect ordinary background load.

## Verification and reproducibility

The independent cached audit checks 2832 prediction attempts, 2776 cosine searches and 19072 calibration candidates. It reconstructs templates and decisions without importing the experiment implementation.

Six AdaFace preprocessing/loading safeguards and three paired-protocol tests pass. Biometric feature arrays, raw images and checkpoint binaries remain in ignored data folders.

```powershell
python scripts/fetch_adaface_assets.py
python scripts/compare_backbones.py
# Audit/re-render this completed run without new image inference:
python scripts/verify_backbone_comparison.py results/backbones/backbones_development_20261020T200004286474Z
python scripts/render_backbone_evidence.py results/backbones/backbones_development_20261020T200004286474Z
```

Raw run: `results/backbones/backbones_development_20261020T200004286474Z`. Open REPORT.md, comparison.json, predictions.csv, calibration_grid.csv, enrollment_sources.csv, paired_cases.csv, failure_cases.csv, real_model_smoke.json and verification.json there.

## Interpretation limits

- Calibration and evaluation reuse development identities/images; descriptive comparison, not held-out generalization.
- Pretraining data and runtime engines differ between public checkpoints; no isolated AdaFace loss-function superiority claim.
- Both models use the same SCRFD five-point crops. This is not the official AdaFace MTCNN benchmark protocol.
- ICFD is a five-identity public-sample closed-set engineering check, without unknown probes.
- Paired known-identity intervals condition on the selected thresholds and omit calibration and unknown-sample uncertainty.
- Timings use alternating CPU forwards. Composed image-to-embedding sums measured stages; it is not camera FPS or complete UI latency.
- No model retraining, adapter, augmentation, reserved-test inference or final-method replacement was performed.
