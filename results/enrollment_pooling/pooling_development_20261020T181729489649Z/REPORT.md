# Enrollment pooling: descriptive development evidence

Same real ArcFace features, enrollment candidates and probes. No training or reserved-test access.

| Dataset | Pooling | Decision rule | Correct known / attempts | False known / extracted unknown |
|---|---|---|---:|---:|
| halfface | one_photo | recalibrated_score | 129/154 | 1/149 |
| halfface | one_photo | frozen_margin_sensitivity | 131/154 | 0/149 |
| halfface | uniform_mean | recalibrated_score | 138/154 | 1/149 |
| halfface | uniform_mean | frozen_margin_sensitivity | 139/154 | 1/149 |
| halfface | heuristic_weighted | recalibrated_score | 138/154 | 1/149 |
| halfface | heuristic_weighted | frozen_margin_sensitivity | 138/154 | 1/149 |
| halfface | detector_weighted | recalibrated_score | 139/154 | 1/149 |
| halfface | detector_weighted | frozen_margin_sensitivity | 138/154 | 1/149 |
| halfface | best_quality | recalibrated_score | 129/154 | 1/149 |
| halfface | best_quality | frozen_margin_sensitivity | 131/154 | 0/149 |
| icfd_sample | one_photo | closed_set_engineering | 100/100 retrieval | not available |
| icfd_sample | uniform_mean | closed_set_engineering | 100/100 retrieval | not available |
| icfd_sample | heuristic_weighted | closed_set_engineering | 100/100 retrieval | not available |
| icfd_sample | detector_weighted | closed_set_engineering | 100/100 retrieval | not available |
| icfd_sample | best_quality | closed_set_engineering | 100/100 retrieval | not available |

## Interpretation limits

- Descriptive development comparison: calibration and evaluation reuse identities and images; not unseen-test performance.
- Frozen-margin sensitivity applies the existing uniform-mean settings to other pooling policies; it is not equal-FPIR calibration.
- ICFD public sample is closed-set engineering evidence with five development identities, not an open-set benchmark.
- Paired intervals condition on the calibrated thresholds and omit calibration and unknown-sample uncertainty.
- Heuristic quality and detector confidence are existing proxies, not new learned face-quality methods.
- No real CCTV, temporal, population-generalization or method-novelty claim follows from these experiments.

Baseline reproduction: 808 original probe/method attempts.
Per-probe scores and decisions, weights, paired effects, input hashes and configuration are saved beside this report.
