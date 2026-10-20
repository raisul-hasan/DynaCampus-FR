# Held-out evaluation: frozen final evidence

The primary method (up to three enrollment views + score + margin) correctly
accepts **151/160 known probes (94.38%)**,
with **0/151 observed false known acceptances**.
This is the first reserved-image inference for this frozen protocol. Settings
were selected from development evidence and were not changed after this test.

## halfFace results

| Method | Enrollment photos | Correct top match | Correct known acceptance | False known / extracted unknown |
|---|---:|---:|---:|---:|
| One photo + score | 55 | 159/160 (99.38%) | 130/160 (81.25%) | 1/151 |
| Up to three photos + score | 159 | 159/160 (99.38%) | 150/160 (93.75%) | 0/151 |
| Up to three photos + score + margin | 159 | 159/160 (99.38%) | 151/160 (94.38%) | 0/151 |

All three comparisons use the same 55 held-out enrolled identities, 27 held-out
unknown identities, 160 masked known probes and 151 mixed-mask unknown probes.
All unknown probes yield embeddings. One known probe has a capture failure and
stays in the denominator. No wrong enrolled identity is accepted by any method.
Every intended identity has a usable enrollment template; none is dropped.

The primary method returns 151 known decisions, 158 unknown decisions, one
ambiguous decision and one capture failure. Its 151 accepted known identities
are correct; 7 other known probes are rejected as unknown,
1 is ambiguous and 1 has a capture failure.
All 151 unknown probes are rejected as unknown. Decision coverage is
309/310 (99.68%) after successful extraction and 309/311 (99.36%) over all attempts.

The 95% known-identity bootstrap interval for primary correct known acceptance
is **88.89% to 98.74%**.

| Paired change in correct known acceptance | Difference (percentage points) | Descriptive 95% interval |
|---|---:|---:|
| few_photo_score_minus_one_photo_score | +12.50 | +6.92 to +18.75 |
| few_photo_margin_minus_few_photo_score | +0.62 | -1.28 to +2.55 |
| few_photo_margin_minus_one_photo_score | +13.12 | +7.45 to +19.62 |

Few-photo enrollment provides the larger observed acceptance gain. Adding the
margin rule to few-photo enrollment helps 2 known probes and hurts 1, for a net change of one probe;
its paired interval includes zero, so no established superiority claim is made.
The comparisons use separately frozen development-calibrated score thresholds;
they compare calibrated methods, rather than an isolated gate at one common score.
The few-photo margin method was designated primary before the test, regardless
of which method would obtain the highest test rate.

## Separate ICFD public-sample check

| ICFD sample method | Probe attempts | Successful extractions | Correct top match |
|---|---:|---:|---:|
| one_photo | 100 | 98 | 98/100 (98.00%) |
| up_to_three_photos | 100 | 98 | 98/100 (98.00%) |

Five reserved sample identities are different from the five used in the earlier
engineering run. Two probe extractions fail and remain included. This is a
closed-set engineering check of the public sample: there are no unknown people
in these probes, so false-positive identification rate cannot be estimated.
It is not a result on the full ICFD dataset or a full CCTV benchmark.

## Frozen protocol and provenance

- Frozen at `2026-10-19T23:15:12.874441+00:00`, before the start receipt and first reserved
  image inference. The receipt includes hashes of source code, manifests, split,
  configuration, development results and evaluation selection.
- Models: unchanged local real SCRFD and ArcFace ONNX checkpoints; same 50%
  neutral source border, five-point alignment and normalized 512-D features.
- Same original inventory exclusions; no new test-driven exclusion or image
  quality filtering. Known enrollment photos are original unmasked views.
- One-photo threshold: 0.3887378871440888. Few-photo score-only threshold:
  0.3978477716445923. Primary threshold: 0.3778589367866516; margin: 0.15.
- Quality threshold stays zero, as selected during development. No learned
  adapter, AdaFace, fine-tuning or temporal aggregation is claimed.
- Frozen methods and test selection: `results/final/heldout_20261019T231512869782Z/freeze.json` and
  `evaluation_manifest.csv`. Start receipt: `evaluation_started.json`.
- Seven protocol tests pass. Independent verification recomputes all
  **1126 successful per-method cosine searches** and
  **1133 probe/method attempts**, checks the frozen gallery
  prototypes, decisions, summary counts, split separation and unchanged hashes
  for all **585 extracted source images**.
- The earlier incomplete metadata-only freeze attempt is marked aborted before
  inference. It did not open test images; this run is the completed frozen evaluation.
- Original images and biometric features remain outside the paper asset folder.
  The evaluation script refuses a second inference start for this freeze.

## Figure captions

1. **[Held-out protocol](fig00_heldout_protocol.png)** ([SVG](fig00_heldout_protocol.svg)).
   Development and test identities are disjoint within halfFace. Models,
   enrollment policy, primary method and thresholds are frozen before reserved
   image inference. The ICFD public-sample check is reported separately.
2. **[Held-out comparison](fig01_heldout_comparison.png)** ([SVG](fig01_heldout_comparison.svg)).
   End-to-end correct known acceptance for three methods on the same 160 masked
   known probes (left), with descriptive 95% known-identity bootstrap intervals,
   and false known acceptances among the same 151 unknown probes (right). One
   known capture failure remains included. Thresholds are development-selected.
3. **[Primary decision states](fig02_heldout_states.png)** ([SVG](fig02_heldout_states.svg)).
   Counts and within-truth-group percentages for known, unknown, ambiguous,
   insufficient-quality and capture-failure outcomes of the primary method.
   Capture failure is an extraction outcome rather than a fifth engine state.
   Quality filtering is disabled. No accepted known probe has a wrong identity.

Numeric sources: [summary](summary.csv), [decision states](decision_state_counts.csv),
[ICFD engineering table](icfd_summary.csv), [paired effects](paired_effects.csv),
[LaTeX result table](heldout_table.tex), [asset/source hashes](asset_manifest.json).

## Suggested paper paragraph

Using thresholds and a primary method frozen on identity-disjoint development
data, few-photo enrollment with score and margin decisions correctly accepted
151/160 masked known probes (94.38%) on held-out halfFace identities, compared
with 130/160 (81.25%) for one-photo enrollment with a calibrated score threshold.
No false known acceptance was observed among 151 held-out unknown probes for
the primary method, compared with one for the one-photo method. Few-photo
score-only decisions accepted 150/160 known probes (93.75%); the additional
margin rule helped two known probes and hurt one, for a net gain of one, with
one known probe marked ambiguous. All 151 unknown probes were rejected as
unknown. One known capture failure was retained in all denominators.
The descriptive known-identity bootstrap interval for primary acceptance was
88.89% to 98.74%, while the paired margin-versus-score interval included zero.
On five separately reserved ICFD public-sample identities, one-photo and
three-photo closed-set retrieval both reached 98/100, including two extraction
failures. These experiments support a limited public-dataset evaluation rather
than a claim of deployment performance.

## Interpretation limits

- **Zero observed false accepts does not establish zero population error.**
  There are only 27 unknown halfFace identities and correlated images. The CSV's
  percentile cluster bootstrap interval becomes [0,0] when every observed
  unknown cluster has zero errors; it cannot create unobserved events and is
  **not a population risk bound**. Report the sample count and this limitation.
- Intervals use 2,000 identity-cluster resamples, seed 42. They condition on the
  frozen calibration and omit calibration, pretraining and acquisition-domain
  uncertainty. No multiple-comparison significance claim is made.
- Identity separation is verified within the supplied manifests; overlap with
  pretrained ArcFace data, near duplicates or residual label errors is unknown.
- Known halfFace probes are masked and unknown probes mix mask conditions.
  Source IDs are used as supplied. This is not a Bangladesh-wide population
  sample or a live campus/CCTV deployment study.
- The earlier blur/resolution/JPEG results remain **development stress tests**;
  they were not repeated or tuned on these held-out identities.
- The reserved test is now consumed for final reporting. Do not adjust the
  method using these results and present a rerun as a fresh held-out test.

## Reproduction and cached verification

```powershell
# Creates a new pre-inference freeze for a clean reproduction:
python scripts/run_final_evaluation.py --freeze
# Use its printed freeze path, then evaluate once. Existing completed freeze refuses reruns.
# For this existing run, use saved real features without new image inference:
python scripts/verify_final_evaluation.py results/final/heldout_20261019T231512869782Z
python scripts/render_final_evidence.py results/final/heldout_20261019T231512869782Z
python -m unittest discover -s tests -p test_final_protocol.py -v
```

Reproductions after these published results are known are reproductions, not a
new unseen-test claim. Figures are 300 dpi PNG and editable SVG with no portraits.
