# Controlled robustness: development evidence

The same **304 probes** were evaluated under **nine isolated synthetic conditions**,
plus an original-image control. The gallery, models and thresholds stayed fixed.
The original control exactly reproduces the preceding decision comparison.

## Main results: score + margin

| Condition | Correct known / 154 | Extracted / 304 | False known / extracted unknown | Wrong known identity |
|---|---:|---:|---:|---:|
| original | 139 (90.26%) | 297 | 1/149 | 0 |
| blur_003 | 141 (91.56%) | 300 | 1/150 | 0 |
| blur_006 | 140 (90.91%) | 302 | 1/150 | 0 |
| blur_012 | 132 (85.71%) | 299 | 1/149 | 0 |
| resolution_128 | 140 (90.91%) | 301 | 1/149 | 0 |
| resolution_64 | 134 (87.01%) | 301 | 1/150 | 0 |
| resolution_32 | 104 (67.53%) | 290 | 0/144 | 0 |
| jpeg_70 | 140 (90.91%) | 298 | 1/149 | 0 |
| jpeg_30 | 136 (88.31%) | 297 | 1/149 | 0 |
| jpeg_10 | 119 (77.27%) | 275 | 0/145 | 0 |

Across the nine transformed conditions, the margin rule accepts more known probes than score only in 6 conditions, the same number in 2, and fewer in 1. These are comparisons of the two fixed calibrated rules; neither is assumed to dominate every condition.

Correct known acceptance includes all 154 masked known attempts, including
extraction failures. Unknown inputs are the same 150 mixed-mask source probes in
every condition. Conditional FPIR divides false known acceptances by successfully
extracted unknown probes; its denominator changes when detection fails. The CSV
also reports false acceptance over all 150 attempted unknown probes, extraction
counts, rank-1 results, wrong enrolled identities, quality and decision coverage.
The original-development 1% calibration target is not guaranteed under these
transformations; report observed error counts even when that target is exceeded.

## Exact experiment

- 55 enrolled identities, 152 original enrollment images; 27 unknown identities.
- Nine conditions chosen before degraded-image inference; transformations are
  separate, never combined. All probes receive the same condition in each run.
- **Blur:** Gaussian sigma = 0.003, 0.006 or 0.012 times each source image's short
  edge, with reflected borders. This is relative to crop size, not a claimed
  camera motion model. It is Gaussian blur, not physical motion blur.
- **Resolution:** downsample the long edge to at most 128, 64 or 32 pixels using
  INTER_AREA, preserve aspect ratio, and never upscale a smaller source. The
  detector's normal resize still runs afterward; no lost detail is restored.
- **JPEG:** re-encode each decoded source at quality 70, 30 or 10, then decode once.
  Sources may already be JPEG. These settings are encoder parameters, not known
  camera bitrate or PSNR levels.
- Transformations run **before** the same neutral padding, SCRFD, alignment,
  ArcFace and exact cosine search. New embeddings are real model outputs.
- Score-only threshold 0.3978477716; score-plus-margin threshold 0.3778589368 and
  margin 0.15. No degradation-specific calibration. Quality gate stays disabled
  as selected in the earlier development experiment.
- No transformed face images are written. Reproduction uses original checksums,
  configuration, decoded-pixel checksums, source versions and ignored caches.
- Nine times 304 = **2,736 new image attempts**. The clean control uses cached
  original features; 10 conditions x 304 probes x 2 rules = 6,080 saved decisions.

## Figure captions

1. **[Protocol diagram](fig00_stress_protocol.png)** ([SVG](fig00_stress_protocol.svg)).
   Isolated synthetic degradation of the same development probes, followed by
   the unchanged recognition pipeline, clean gallery and fixed decision rules.
   Original data are preserved and reserved test identities are excluded.
2. **[Recognition graph](fig01_stress_recognition.png)** ([SVG](fig01_stress_recognition.svg)).
   End-to-end correct known acceptance under three blur, resolution and JPEG
   settings, with the same original control repeated for context. Each point
   uses the same 154 masked known attempts; failed extractions are unsuccessful
   recognition. The rule thresholds are fixed from original development data.
3. **[Extraction and false acceptance](fig02_stress_failures.png)** ([SVG](fig02_stress_failures.svg)).
   Extraction success across all 304 probe attempts (top), with successful
   extraction counts labelled, and false known acceptances across the same 150
   attempted unknown probes (bottom). Extraction is shared by both rules. These
   counts keep failure effects visible when conditional FPIR denominators change.

Numeric graph sources: [summary.csv](summary.csv). [LaTeX table](robustness_table.tex).
Paired degradation-minus-original uncertainty: [paired_effects.csv](paired_effects.csv).
All uncertainty intervals resample **known identities**, not synthetic copies as
independent images (2,000 resamples, seed 42). They condition on fixed thresholds
and omit threshold-selection, unknown-sample and model-pretraining uncertainty.
They are descriptive; no multiple-comparison significance claim is made.

## Suggested paper paragraph

With enrollment templates and decision thresholds fixed, the score-plus-margin rule correctly accepted 139/154 masked known probes (90.26%) on original development images. Across nine isolated synthetic degradation conditions, correct known acceptance ranged from 104/154 (67.53%) to 141/154 (91.56%). The lowest result occurred under resolution_32, with 290/304 successful probe extractions. Detection failures were retained in the end-to-end denominator. These paired synthetic development stress tests do not measure deployment performance on actual CCTV footage.

## Limits and verification

These are development stress tests of public face crops; the same development
identities supplied earlier calibration. A synthetic condition can occasionally
improve a particular probe through changed detection/alignment. Report observed
values without assuming degradation is strictly monotonic or choosing new
thresholds to improve them. Derived images do not increase the number of people.
No Bangladesh-wide or real CCTV generalization claim is supported. Reserved test
identities remain untouched; no model has been trained or adapted in this step.

Independent verification recomputes every successful cosine search from saved
real embeddings, every decision and all summary counts, checks fixed prototypes
and thresholds, and hashes all original probe files again. The five targeted
protocol tests cover transforms, source preservation and failure denominators.
No participant portraits or biometric features are copied into this asset folder.

Raw evidence: `results/robustness/robustness_development_20261019T230159679928Z`. [Source and asset hashes](asset_manifest.json).

```powershell
python scripts/run_robustness.py
python scripts/verify_robustness_run.py results/robustness/robustness_development_20261019T230159679928Z
python scripts/render_robustness_evidence.py results/robustness/robustness_development_20261019T230159679928Z
python -m unittest discover -s tests -p test_robustness_protocol.py -v
```

Figures are 300 dpi PNG plus editable SVG. Full per-probe predictions, extraction
outcomes, source checksums, model hashes and configuration snapshots stay in the
raw run. Embeddings remain under ignored `data/embeddings/`.
