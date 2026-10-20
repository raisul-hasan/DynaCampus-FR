# Paper evidence index

Updated: 20 October 2026. The frozen held-out halfFace evaluation is complete.
The primary method accepts 151/160 known probes (94.38%), with 0/151 observed
false known acceptances. Earlier development evidence is retained separately.
The reserved ICFD public-sample check is closed-set engineering evidence.

Historical development captions describing the test set as untouched refer to
the time of those development runs. The test set is now consumed for final
reporting; changing settings using its results would invalidate a fresh-test claim.

| Work completed | Evidence | Paper use |
|---|---|---|
| **Final held-out comparison at frozen settings** | [Results, captions and limits](heldout/final_v1/README.md), [numeric table](heldout/final_v1/summary.csv), [LaTeX table](heldout/final_v1/heldout_table.tex) | Primary final result; identity-disjoint within halfFace |
| Frozen protocol and split diagram | [PNG](heldout/final_v1/fig00_heldout_protocol.png), [SVG](heldout/final_v1/fig00_heldout_protocol.svg) | Development selection followed by reserved-image evaluation |
| Held-out result graph with uncertainty | [PNG](heldout/final_v1/fig01_heldout_comparison.png), [SVG](heldout/final_v1/fig01_heldout_comparison.svg) | Correct known acceptance and observed false-known counts |
| Primary held-out decision states | [PNG](heldout/final_v1/fig02_heldout_states.png), [SVG](heldout/final_v1/fig02_heldout_states.svg) | Counts and percentages with capture failures retained |
| Reserved ICFD public-sample check | [Engineering table](heldout/final_v1/icfd_summary.csv) | 98/100 closed-set retrieval for both enrollment policies; unknown FPIR not estimable |
| Dynamic enrollment and real application flow | [Results, screenshots and captions](development/demo_v1/README.md), [stage counts](development/demo_v1/stage_summary.csv) | Development engineering validation; 55 insertions without backbone changes |
| Dynamic workflow and gallery growth | [Workflow](development/demo_v1/fig00_dynamic_workflow.png), [growth graph](development/demo_v1/fig01_dynamic_growth.png) | Persistent insertion, reload and removal; changing enrolled-known denominators |
| Local search-call latency | [Graph](development/demo_v1/fig02_search_latency.png), [numeric table](development/demo_v1/stage_summary.csv) | Component timings only; no video FPS claim |
| Actual app screenshots with photo previews off | [Enrollment](development/demo_v1/fig03_app_enrollment.png), [known](development/demo_v1/fig04_app_known.png), [unknown](development/demo_v1/fig05_app_unknown.png), [reload](development/demo_v1/fig06_app_persistence.png), [removal](development/demo_v1/fig07_app_after_removal.png) | Disclosed development examples; not additional accuracy samples |
| Dataset integrity and original/derived separation | [Dataset status](../docs/DATASET_START.md), [halfFace inventory](../results/tables/data_readiness/halfface_inventory.json) | Data preparation and limitations |
| Real-model baseline and identity-disjoint reservation | [Baseline report](../results/baselines/arcface_development_20261019T221350351017Z/BASELINE_REPORT.md), [score distribution](../results/baselines/arcface_development_20261019T221350351017Z/development_scores.png) | Implemented baseline; development calibration |
| One-photo versus few-photo enrollment | [Enrollment evidence and captions](development/enrollment_v1/README.md), [numeric table](development/enrollment_v1/comparison.csv), [LaTeX table](development/enrollment_v1/enrollment_table.tex) | Development enrollment ablation |
| Dataset selection and split diagram | [PNG](development/enrollment_v1/fig00_data_protocol.png), [SVG](development/enrollment_v1/fig00_data_protocol.svg) | Dataset/protocol figure |
| Actual enrollment pipeline diagram | [PNG](development/enrollment_v1/fig01_enrollment_method.png), [SVG](development/enrollment_v1/fig01_enrollment_method.svg) | Implemented-method figure |
| Enrollment comparison graph | [PNG](development/enrollment_v1/fig02_enrollment_comparison.png), [SVG](development/enrollment_v1/fig02_enrollment_comparison.svg) | Development results figure |
| Open-set trade-off graph | [PNG](development/enrollment_v1/fig03_open_set_tradeoff.png), [SVG](development/enrollment_v1/fig03_open_set_tradeoff.svg) | Development score trade-off; not held-out ROC performance |
| Score, margin and quality decision comparison | [Results and captions](development/decision_v1/README.md), [numeric table](development/decision_v1/ablation.csv), [LaTeX table](development/decision_v1/decision_table.tex) | Development decision ablation, including no added quality benefit |
| Implemented decision rule | [PNG](development/decision_v1/fig01_decision_rule.png), [SVG](development/decision_v1/fig01_decision_rule.svg) | Four-state method diagram; selected quality gate disabled |
| Decision comparison graph | [PNG](development/decision_v1/fig02_decision_ablation.png), [SVG](development/decision_v1/fig02_decision_ablation.svg) | Known acceptance and coverage with explicit denominators |
| Quality filtering sensitivity | [PNG](development/decision_v1/fig03_quality_sensitivity.png), [SVG](development/decision_v1/fig03_quality_sensitivity.svg) | Coverage trade-off at fixed score and margin thresholds |
| Controlled blur, resolution and JPEG stress tests | [Results and captions](development/robustness_v1/README.md), [numeric table](development/robustness_v1/summary.csv), [LaTeX table](development/robustness_v1/robustness_table.tex) | Nine synthetic development conditions; models and thresholds fixed |
| Stress-test protocol diagram | [PNG](development/robustness_v1/fig00_stress_protocol.png), [SVG](development/robustness_v1/fig00_stress_protocol.svg) | Probe transformations and unchanged enrollment gallery |
| Recognition under degradation | [PNG](development/robustness_v1/fig01_stress_recognition.png), [SVG](development/robustness_v1/fig01_stress_recognition.svg) | End-to-end correct known acceptance, including extraction failures |
| Extraction and false acceptance under degradation | [PNG](development/robustness_v1/fig02_stress_failures.png), [SVG](development/robustness_v1/fig02_stress_failures.svg) | Separate capture failures and false-known counts; 304 paired source probes |

## Evidence standards for remaining steps

- Save raw per-probe predictions, configurations, source/model hashes and split
  identifiers before writing a result into the paper.
- Label development and held-out results explicitly. Use the final held-out
  evidence for final claims while keeping synthetic stress tests labelled development.
- Zero false accepts among 151 held-out probes does not establish zero population
  error. The margin gain is one net probe and its paired interval includes zero;
  the larger observed gain comes from few-photo enrollment.
- Keep the same probes for paired comparisons and record exclusions and failures.
- Report uncertainty and sample counts; avoid treating adjacent images as fully
  independent evidence.
- Keep diagrams aligned with the implementation. AdaFace, adapter training and
  temporal tracking have not been evaluated in the evidence linked above.
  Margin and heuristic quality gates have development evidence; the selected
  quality gate is zero, so do not claim an incremental quality-gating benefit.
- The dynamic-gallery app flow is validated with development examples, cached
  real features and actual browser uploads. Rendering camera controls is not a
  live camera, RTSP or tracking validation. Search-call timing is not pipeline FPS.
- Label synthetic robustness results as controlled stress tests. They do not
  establish performance on real CCTV footage, and transformed versions of one
  source image are not independent people or an expanded original dataset.
- Add a caption and numeric source table for each new graph where practical.
- Keep raw identifiable face images and biometric embeddings out of the paper
  asset folder and Git. Current diagrams contain no participant portraits.

## Dataset attribution to retain in the paper

- halfFace, version 2, DOI: 10.17632/pk44mkx9vm.2.
  Source: https://data.mendeley.com/datasets/pk44mkx9vm/2
- ICFD source: https://sites.google.com/view/icfd
  Cite Bhattacharjee and Das, D2SC-GAN, IEEE TBIOM 2(3), 223-234, 2020,
  DOI: 10.1109/TBIOM.2020.2983524. Only the public sample has been used.

Use the original sources' attribution and access terms. Full ICFD access remains
pending; South Asian evidence does not establish Bangladesh-wide performance.
