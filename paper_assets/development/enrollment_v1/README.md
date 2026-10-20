# Enrollment comparison: development evidence

These results are development observations, not final held-out performance.

| Method | Enrollment images | Correct top match | Correct known acceptance | False known / extracted unknown |
|---|---:|---:|---:|---:|
| One photo | 55 | 146/154 (94.81%) | 129/154 (83.77%) | 1/149 |
| Up to three photos | 152 | 146/154 (94.81%) | 138/154 (89.61%) | 1/149 |

The few-photo method helped 10 known probes and hurt 1, a net gain of 9. Top-match correctness did not change.
The paired identity-bootstrap difference in correct acceptance was 5.84 percentage points (descriptive 95% interval 1.31 to 11.61). This conditions on development-calibrated thresholds and excludes calibration/unknown-sample uncertainty.

## Protocol and limitations

- Both methods use the same 55 enrolled identities, 154 masked known probes and 150 unknown probe attempts. One unknown detection fails, leaving 149 scores for empirical calibration.
- One-photo enrollment uses the original frontal unmasked photo. Few-photo enrollment averages up to three original unmasked views and L2-normalizes the mean. Actual enrollment counts are 47 identities with three photos, three with two and five with one.
- Both methods have six known-probe detection failures; these remain errors in the reported rates.
- Extra enrollment photos are excluded from probes for both methods. The earlier mixed-mask baseline used 251 known probes; its 96.81% is not directly comparable with this masked-only 154-probe experiment.
- Each method is separately calibrated to at most 1% empirical conditional FPIR using the same development unknown probes. Both accept one of 149; this is not a guaranteed real-world false-identification rate.
- A separate ICFD engineering check uses five sample identities and the same 100 classroom probes: one-photo and three-photo methods both retrieve 100/100 correctly. This ceiling result is not a full ICFD benchmark.
- No model retraining, adaptation, augmentation or quality gate was used. Reserved test identities remain untouched. Near-duplicate and pretraining overlap limitations remain.

## Figure captions

**Figure 00. Data preparation and identity split.** The original halfFace full-face collection is filtered using documented duplicate-identity and label checks. Development and reserved-test source identities are disjoint. Only development data enter the current comparison.

**Figure 01. Enrollment comparison pipeline.** Original one-photo or up-to-three-photo enrollment media pass through the same frozen detection, alignment and ArcFace encoder. Normalized mean identity templates are matched against a common probe set without updating model weights.

**Figure 02. Enrollment results on masked development probes.** Bars compare correct top-match retrieval and correct acceptance after a score threshold. Denominators include six failed detections. Each method accepts one of 149 successfully embedded development unknown probes.

**Figure 03. Development open-set trade-off.** Curves show correct known acceptance including failed detections versus false-positive identification among successfully embedded unknowns. Markers indicate the separately calibrated operating points; the right panel enlarges the low-FPIR range.

## Suggested paper wording

In a development-only enrollment ablation, one frontal unmasked image was compared with a normalized mean template formed from up to three unmasked views. Both strategies were evaluated on the same 154 masked known probes against 55 enrolled identities, with 150 probe attempts from 27 unenrolled identities. Correct top-match retrieval remained 146/154 (94.81%). After separate development calibration to a target empirical FPIR of 1%, correct known acceptance increased from 129/154 (83.77%) to 138/154 (89.61%); both methods falsely accepted one of 149 successfully embedded unknown probes. These observations require confirmation on the reserved test identities.

## Files and reproduction

Source run: results/enrollment/enrollment_development_20261019T222121709545Z
Run comparison: python scripts/compare_enrollment.py
Render this evidence: python scripts/render_enrollment_evidence.py results/enrollment/enrollment_development_20261019T222121709545Z
Figures are supplied as 300-dpi PNG and editable vector SVG. Tables preserve unrounded numeric data in CSV. Raw predictions and enrollment-source records remain in the source run directory.
