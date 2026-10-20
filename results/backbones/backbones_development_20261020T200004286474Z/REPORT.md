# AdaFace and ArcFace: development comparison

| Model | Enrollment | Rule | Correct known / attempts | False known / extracted unknown |
|---|---|---|---:|---:|
| arcface | one_photo | score_only | 129/154 | 1/149 |
| arcface | one_photo | score_margin | 139/154 | 1/149 |
| arcface | up_to_three_photos | score_only | 138/154 | 1/149 |
| arcface | up_to_three_photos | score_margin | 139/154 | 1/149 |
| adaface | one_photo | score_only | 112/154 | 1/149 |
| adaface | one_photo | score_margin | 120/154 | 1/149 |
| adaface | up_to_three_photos | score_only | 122/154 | 1/149 |
| adaface | up_to_three_photos | score_margin | 133/154 | 0/149 |

## Scope and limitations

- Calibration and evaluation reuse development identities/images; descriptive comparison, not held-out generalization.
- Pretraining data and runtime engines differ between public checkpoints; no isolated AdaFace loss-function superiority claim.
- Both models use the same SCRFD five-point crops. This is not the official AdaFace MTCNN benchmark protocol.
- ICFD is a five-identity public-sample closed-set engineering check, without unknown probes.
- Paired known-identity intervals condition on the selected thresholds and omit calibration and unknown-sample uncertainty.
- Timings use alternating CPU forwards. Composed image-to-embedding sums measured stages; it is not camera FPS or complete UI latency.
- No model retraining, adapter, augmentation, reserved-test inference or final-method replacement was performed.

Official source: https://github.com/mk-minchul/AdaFace

Raw per-probe predictions, all calibration candidates, selected thresholds, paired effects, timing stages and failure cases are saved beside this report.
