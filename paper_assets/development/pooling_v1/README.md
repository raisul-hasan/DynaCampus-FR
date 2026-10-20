# Enrollment pooling: development ablation

Added 21 October 2026. This is descriptive development evidence, separate from
the already completed final evaluation. No new face images, model inference,
training or reserved-test feature caches are used.

Raw run: [report](../../../results/enrollment_pooling/pooling_development_20261020T181729489649Z/REPORT.md),
[summary](../../../results/enrollment_pooling/pooling_development_20261020T181729489649Z/summary.csv),
[per-probe predictions](../../../results/enrollment_pooling/pooling_development_20261020T181729489649Z/predictions.csv),
[enrollment weights](../../../results/enrollment_pooling/pooling_development_20261020T181729489649Z/enrollment_weights.csv),
[paired effects](../../../results/enrollment_pooling/pooling_development_20261020T181729489649Z/paired_effects.csv),
[verification receipt](../../../results/enrollment_pooling/pooling_development_20261020T181729489649Z/verification.json).
Run identifiers retain execution-host UTC timestamps; the date above follows the
user's Asia/Dhaka session calendar.

## Same-probe comparison

The halfFace experiment uses the existing 55 development gallery identities,
154 masked known probes and 150 unknown attempts. There are 148 extracted known
probes and 149 extracted unknown probes. Failures remain in attempted counts.
The few-photo policies use the same original unmasked enrollment candidates;
one-photo enrollment uses the frontal image. Best-quality selection chooses one
of the same candidates. No image can occur in both enrollment and the probe set.

| Enrollment policy | Correct known / 154, recalibrated score | False known / 149 extracted unknown | Correct known / 154, fixed previous score+margin |
|---|---:|---:|---:|
| One frontal photo | 129 | 1 | 131 |
| Uniform mean, up to three photos | 138 | 1 | 139 |
| Heuristic-quality weighted mean | 138 | 1 | 138 |
| Detector-confidence weighted mean | 139 | 1 | 138 |
| Highest heuristic-quality photo | 129 | 1 | 131 |

The fixed-rule column uses threshold 0.3778589367866516, margin 0.15 and no
quality rejection. It is a sensitivity check, not an equal-FPIR comparison:
the few-photo policies falsely accept 1/149 unknown probes, while one-photo and
best-quality selection falsely accept 0/149 under that rule.

Recalibrated score comparisons separately select a threshold from these same
development unknown probes at an empirical FPIR limit of 1%. Evaluation reuses
that development data; these numbers cannot be presented as held-out accuracy.
The detector-weighted gain over the mean is one probe, with a descriptive paired
known-identity interval of 0.00 to 2.04 percentage points. With the fixed prior
score/margin rule, detector weighting instead loses one acceptance. There is no
consistent weighting advantage in these tests. The proxies are not learned
recognition-quality measures, and these results do not establish that all quality
weighting is ineffective. Many enrollment images have similarly high proxy values.

The separate five-person ICFD development public sample gives 100/100 closed-set
retrieval for every policy. It has no unknown probes and a ceiling result, so it
does not discriminate between these policies or establish open-set performance.

## Verification

- Six new safeguards pass: normalized/weighted behavior, scale invariance,
  explicit zero-weight fallback, actual-sample selection, invalid-weight and
  reserved-identity rejection, and retained failure/error denominators.
- The one-photo and uniform baselines reproduce all 808 original dataset/probe/
  method attempts, including failed extractions and full-precision top-two scores.
- A separate verifier reconstructs templates without importing the pooling
  implementation. It confirms all 3,540 probe/method/rule attempts, 3,470
  successful searches, all 15 summary rows, weights, decisions and input hashes.
- The existing frozen final-evaluation audit still passes: 1,133 attempts,
  1,126 searches, 585 unchanged source images and zero development-ID overlap.

```powershell
python scripts/compare_enrollment_pooling.py
# Audit this completed run without new model inference:
python scripts/verify_enrollment_pooling.py results/enrollment_pooling/pooling_development_20261020T181729489649Z
python -m unittest discover -s tests -p test_enrollment_pooling.py -v
```

Do not use this ablation to change the frozen final method. These are existing
pooling policies being evaluated, not a proposed new aggregation algorithm.
