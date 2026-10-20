# Completion plan: submission on 30 October 2026

Prepared 21 October 2026, Asia/Dhaka. The user accepts an experimental contribution
and a working dashboard; a newly invented method is not required. There are nine
calendar days until submission. Dates below are work targets, not promises that
blocked downloads or external approvals will finish on schedule.

## Scope and acceptance criteria

Deliver a defensible still-image experimental thesis: traceable data/protocol,
calibrated ArcFace comparisons, enrollment and decision ablations, degradation
limits, raw predictions with independent verification, an executable gallery demo,
runtime/failure analysis, paper, slides and defense evidence. Add the planned
AdaFace comparison (now complete) with checkpoint/pretraining confounds labelled.
No new video collection is assumed. A toy adapter, unvalidated tracking and a
new transformer comparison are not requirements for this scoped submission.

The frozen final evaluation remains unchanged. Additional experiments use
development inputs; the already examined held-out identities cannot become a new
unseen test merely by rerunning or repartitioning them.

| Target | Work | Definition of done |
|---|---|---|
| 21 October | Contribution audit, research questions, evidence map and pooling ablation | Literature overlap recorded; experiment verified; claims and remaining gaps explicit |
| 22 October | Augmented one-photo enrollment development experiment; writing in parallel | One original vs original plus controlled variants vs three originals, both backbones; fixed declared protocol, raw decisions and independent verification, including negative results |
| 23 October | Gallery-growth development experiment; writing in parallel | Fixed core known/unknown probes and thresholds at 10/25/55 identities, gallery-composition effects disclosed; no fresh held-out claim |
| 24 October | Complete runtime and contribution comparison | Median/p95 enrollment, detection, embedding, search and decision timings with explicit boundaries; precise comparison to closest prior work; no camera FPS claim |
| 25 October | Thesis draft, cards and offline setup review | Local dependency/test cleanup is already complete; finish documentation and an offline rehearsal, recording any clean-install limitations |
| 26 October | Evidence freeze and complete first thesis draft | Every reported number traces to a result file; implemented method and negative findings match text |
| 27 October | Revise paper and assemble defense slides | Related work, methods, results, uncertainty, limits, practical relevance and references covered |
| 28 October | Full defense rehearsal and backup capture | Local models load, isolated gallery demonstration executes, verification works, genuine screen recording saved |
| 29 October | Final checks and submission package | Offline artifact copy, required formatting, citations, source permissions and final file checks complete |
| 30 October | Submit | Required institutional submission completed by the user or an explicitly authorized channel |

Writing and evidence organization should proceed while experiments run. There
is no reason to wait for a new method before drafting data/protocol/methods.
The user asked to remember the augmented-enrollment, gallery-growth, runtime,
contribution and defense priorities on 21 October. Budget at most two days for
the two added development experiments; they are planned, not completed. If either
blocks progress, record that and continue the verified study and submission work.
Do not spend the final buffer inventing a method whose generalization cannot be
tested. See [the detailed handoff](NEXT_SESSION.md).

## Current state

- **Complete:** real ArcFace development baseline, enrollment/decision ablations,
  blur/resolution/JPEG stress tests, frozen final evaluation, original dynamic
  gallery demonstration and genuine browser evidence.
- **Complete this session:** five pooling policies compared on development data;
  3,540 attempts and 15 summary rows independently verified; six new safeguards
  and 16 existing baseline/final/robustness protocol tests pass. The final cached
  audit still passes with all 585 source hashes unchanged.
- **Also complete:** official AdaFace integration and the paired development
  comparison; 2,832 decisions and 19,072 calibration candidates independently
  verified. Image-to-embedding CPU timings, operational failure categories and
  paired helped/harmed cases are saved. The selected ArcFace margin policies tie
  for one and few photos, so enrollment benefits must be qualified by decision rule.
- **Also complete:** 12 inspected first-path cases, private offline photo viewer,
  redacted case cards, failure explanations, isolated defense dashboard, ten real
  inference/gallery lifecycle steps and AppTest checks. A labelled 149-second
  backup video combines cards/charts and genuine prior screenshots.
- **Remaining:** fresh browser captures and a live recording of the defense
  dashboard (browser connection pending), search/decision and UI timing beyond
  image-to-embedding stages, paper/slides, complete cards and an end-user offline
  rehearsal on the defense laptop. Clean-machine installation remains untested.
- **Also complete:** original app lifecycle checks with real uploaded-image bytes
  supplied through AppTest, all 89 tests, runtime/dev requirements, setup guide
  and double-click launchers. These checks are not fresh browser interaction.
- **Newly planned:** augmented one-photo enrollment for both backbones and gallery
  growth at 10/25/55 identities, using development data and fixed declared protocols.
- **Outside validated scope:** learned adapter, local campus dataset, temporal
  tracking performance, live CCTV accuracy, deployment or population guarantees.

See [contribution audit](NOVELTY_AUDIT.md), [defense evidence](DEFENSE_EVIDENCE.md)
and [pooling experiment](../paper_assets/development/pooling_v1/README.md).
