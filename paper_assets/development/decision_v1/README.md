# Decision comparison: development evidence

The margin rule correctly accepts **139/154 known probes (90.26%)**, versus
**138/154 (89.61%)** for score only. Both falsely accept **1/149 extracted
unknown probes (0.67%)**. The quality option selects **zero**, so it adds no
benefit under the stated objective and coverage constraint.

| Method | Correct known acceptance | False known / extracted unknown | Decision coverage / extracted probes |
|---|---:|---:|---:|
| Score only | 138/154 (89.61%) | 1/149 | 297/297 (100%) |
| Score + margin | 139/154 (90.26%) | 1/149 | 295/297 (99.33%) |
| Score + margin + quality option | 139/154 (90.26%) | 1/149 | 295/297 (99.33%) |

The margin methods refuse two unknown probes as ambiguous. Seven capture failures
(six known, one unknown) remain included in attempted-probe counts. End-to-end
decision coverage is 297/304 for score only and 295/304 for the margin methods.
No accepted known probe receives the wrong enrolled identity in this run.

## Protocol and selected thresholds

- Same frozen ArcFace features, 55 enrolled identities, 152 enrollment images,
  154 masked known probes and 150 mixed-mask unknown probes as the few-photo run.
- No additional model inference or reserved-test image access was needed.
- Jointly search score, margin and (where enabled) quality thresholds on the same
  development data. The 42,316 grid configurations are saved with feasibility.
- Require empirical unknown FPIR <= 1% over **all 149 extracted unknowns**, including
  those refused by quality or ambiguity. Require at least 85% decision coverage
  separately for extracted known and unknown probes.
- Maximize correct known acceptances; ties prefer fewer false unknown acceptances,
  fewer wrong known identities, greater coverage, lower quality/margin gates,
  then a higher similarity threshold. Zero gates are permitted.
- Score only: score 0.3978477716, margin 0, quality 0.
- Margin methods: score 0.3778589368, margin 0.15, quality 0.
- Thresholds above are research settings; dashboard sliders remain manual settings.
- Quality is the existing heuristic: 45% Laplacian sharpness, 35% native face size,
  20% landmark symmetry/tilt. It is not a learned quality model or calibrated
  probability. Sharpness is measured on the aligned crop; size uses the native bbox.

The margin method changes the calibrated score threshold as well as the margin
gate. Its gain is a comparison of calibrated rule families, not an isolated causal
effect of adding a gate at an unchanged score threshold. The earlier enrollment
step used a different threshold selection tie rule; its few-photo acceptances
match this score-only result.

## Figure captions

1. **[Decision rule](fig01_decision_rule.png)** ([editable SVG](fig01_decision_rule.svg)).
   Implemented quality, similarity and top-two margin rule. Quality failure returns
   insufficient quality; low similarity returns unknown; a small gap between the
   two best identity scores returns ambiguous; otherwise the top identity is
   accepted. Capture failure is recorded separately. The optimized development
   quality gate is disabled; the diagram shows the general supported rule.
2. **[Decision ablation](fig02_decision_ablation.png)** ([editable SVG](fig02_decision_ablation.svg)).
   Correct known acceptance and extracted-probe decision coverage on the same
   halfFace development probes. Margin calibration recovers one additional known
   probe while refusing two unknown probes. All three families have one false
   known acceptance. The quality-enabled family selects a zero quality threshold.
3. **[Quality sensitivity](fig03_quality_sensitivity.png)** ([editable SVG](fig03_quality_sensitivity.svg)).
   Coverage and end-to-end correct known acceptance as the quality threshold
   varies, keeping score 0.3779 and margin 0.15 fixed. Filtering can preserve or
   reduce these counts; this graph does not establish that the quality estimate
   improves recognition. Numeric points: [CSV](quality_sensitivity.csv).

## Suggested paper text and limits

On halfFace development probes, jointly calibrated score and margin decisions
increased correct known acceptance from 138/154 (89.61%) to 139/154 (90.26%), with
one false known acceptance among 149 extracted unknown probes in both cases.
Decision coverage decreased from 100% to 99.33% among extracted probes because two
unknown probes were marked ambiguous. Enabling quality-threshold selection yielded
a zero threshold and the same decisions, providing no incremental benefit for the
current heuristic under the chosen development objective.

The paired known-identity bootstrap difference is +0.65 percentage points, with a
descriptive 95% interval of 0.00 to 1.97 points (2,000 resamples, fixed thresholds).
This interval includes zero and excludes threshold-selection and unknown-sample
uncertainty. Calibration and evaluation reuse development data; these figures
must not be described as final held-out performance or statistically established
improvements. The small masked-face sample does not establish CCTV robustness or
Bangladesh-wide generalization. Reserved test identities remain untouched.

## Files and reproduction

- [Numeric ablation table](ablation.csv), [LaTeX table](decision_table.tex),
  [paired uncertainty](paired_effects.csv), [source manifest](asset_manifest.json).
- Raw run: `results/decisions/decisions_development_20261019T222927022126Z` (input scores, predictions, entire calibration grid,
  configuration, split, source hashes and independent verification).
- Eight decision tests cover ambiguity, boundaries, invalid inputs, refusal
  denominators, single-candidate galleries and calibration behavior. Independent
  verification checks all 912 saved probe/method attempts and grid selection.
- Dashboard wiring receives both identity scores, with explicit ambiguity and
  insufficient-quality states. Syntax is checked; interactive UI testing is pending.

```powershell
python scripts/compare_decisions.py
python scripts/verify_decision_run.py results/decisions/decisions_development_20261019T222927022126Z
python scripts/render_decision_evidence.py results/decisions/decisions_development_20261019T222927022126Z
python -m unittest discover -s tests -p test_decision_ablation.py -v
```

Figures are 300 dpi PNG and editable SVG. They contain no participant portraits.
