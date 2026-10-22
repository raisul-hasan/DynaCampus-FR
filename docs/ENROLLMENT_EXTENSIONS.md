# Enrollment, gallery-growth and runtime results

Completed 23 October 2026. These descriptive development experiments use both
ArcFace and AdaFace, with no reserved-test inference or model training.

- [Evidence report, four figures, tables and reproduction](../paper_assets/development/extensions_v1/README.md).
- [Standalone results section](../paper_assets/development/extensions_v1/thesis_section.md).
- [Integrated thesis content draft](THESIS_DRAFT.md).

The tested five-variant enrollment profile does not consistently improve on one
original; additional original photos and the selected decision rule interact.
Growth keeps the same 28 known and 150 unknown attempts across 10/25/55-person
galleries, exposing composition and ambiguity effects. Runtime separates actual
image processing and persistent insertion from small-gallery search.

Separate verification replays all 550 variant model forwards, checks 12,280
predictions and 28,608 calibration candidates, and recomputes the timing
summaries from 5,220 samples. All 104 tests pass. Existing frozen halfFace evidence
retains its original settings and results.
