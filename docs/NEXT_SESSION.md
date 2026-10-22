# Latest checkpoint - enrollment, growth, runtime and thesis, 23 October 2026

The user authorized all four next steps. They are now complete:
[evidence pack](ENROLLMENT_EXTENSIONS.md), [integrated content draft](THESIS_DRAFT.md).

- Run: `results/enrollment_extensions/extensions_20261022T193101706934Z`.
  Prepared protocol/config/source/input hashes precede inference. Inputs are
  halfFace development only; both checkpoint caches and original source crops
  are verified. Learned adapters, final-test inference and app changes are absent.
- Augmentation: one frontal original vs that original plus five mild variants
  vs up to three original views, both ArcFace/AdaFace. The same 154 known and 150
  unknown attempts are retained, including seven extraction failures. Fixed
  one-photo rules and separate development calibration are both saved.
- Recalibrated score-only acceptance: ArcFace 129 -> 128 -> 138/154; AdaFace
  112 -> 111 -> 122/154. Synthetic variants do not consistently replace originals.
  Margin results differ: ArcFace 139 -> 135 -> 139; AdaFace 120 -> 123 -> 133.
  Report unknown accepts and coverage with these counts; fixed-rule gains can
  accompany more unknown accepts. Paired known-identity intervals are saved.
- Growth: ten fixed core identities, 28 known and 150 unknown attempts at every
  size. One 10-person, five seeded 25-person and one 55-person gallery. Prior
  55-person few-photo thresholds stay fixed. Rank-1 is 26/28 for both models
  throughout; score/margin unknown acceptance varies non-monotonically because
  new competitors can cause ambiguity. Composition repetitions are not new people.
- Runtime: 180 real image calls, 1,800 actual gallery searches, 5,220 stage samples.
  ArcFace median/p95 image-to-decision 143.81/175.57 ms; AdaFace 207.03/241.83 ms.
  ArcFace 55-person search microbenchmark 0.0637/0.0708 ms. Two CPU inference
  threads per engine; full pool/environment details and timing boundaries saved.
  Browser transfer, camera/RTSP and cold loading are excluded; do not claim FPS.
- Separate verifier replays all 550 synthetic variant forwards and checks 12,280
  predictions, 12,056 searches, 28,608 calibration candidates and all summaries.
  All 104 tests pass (two upstream warnings). The final audit still verifies
  all 585 input hashes, frozen thresholds and original final predictions.
- Four new PNG/SVG figure pairs and a LaTeX table are in
  `paper_assets/development/extensions_v1`. The report and thesis section link
  raw predictions, candidate grids, compositions and measured runtime samples.
- The integrated thesis draft has abstract, research questions, literature
  boundaries, methods, data integrity/splits, all results, uncertainty, limits,
  reproducibility and checked primary-source references. It is not an
  institution-formatted/submitted thesis. Next submission work: supervisor review,
  formatting, slides and an offline defense rehearsal with fresh live capture.

# Previous checkpoint - ChokePoint, 23 October 2026

The user authorized turning the supplied ChokePoint notebook/manuscript into
verified supplementary experiments. This is complete:
[results and thesis section](CHOKEPOINT_RESULTS.md).

- Official P1E crops (66.4 MB) and ground truth downloaded sequentially, published
  MD5s verified, license/source receipts retained under ignored data/.
- Frozen custom protocol: one-photo gallery P1E_S1_C1; calibration P1E_S2_C2;
  evaluation P1E_S3_C3/P1E_S4_C1. Five consecutive frames per track for both known
  and unknown queries; 143/66 calibration and 303/143 evaluation known/unknown.
- Run: `results/chokepoint/temporal_20261022T185858644292Z`. Real ArcFace raw norms
  and unit features stay under ignored data/embeddings/chokepoint. All nine
  predeclared policies, four tiers plus mixed, fixed/AOSM, and two paired
  contamination controls are computed. The supplied notebook remains unedited.
- Separate verifier passes: 41,265 predictions, 701,505 gallery scores, 13,168
  corrupted arrays, 4,776 original crops, 42 paired intervals and eight repeated
  real inference checks. The verifier materializes compressed arrays once.
- Severe Rank-1: first 295/303 (97.36%), mean 302/303 (99.67%), paired improvement
  +2.31 points, 95% identity-cluster interval +0.96 to +3.75. No mixed-track Rank-1
  gain from norm-softmax beta 1 over the mean. Mean remains 303/303 under both
  contamination controls while best-norm selection drops substantially.
- Calibration transfer limitation: mixed mean accepts 303/303 known but also
  12/143 unknown tracks (8.39% FPIR) at the frozen fixed threshold. Do not claim
  that a <=1% calibration target achieved 1% evaluation or population FPIR.
- Five portrait-free PNG/SVG figures, a LaTeX table and draft `thesis_section.md`
  are in paper_assets/supplementary/chokepoint_v1. This is a corrected exploratory
  analysis, not replication of the PDF's numbers, actual tracker validation,
  campus deployment or a fresh unseen-identity test. Same identities recur across
  sequence roles, and the supplied pilot already examined P1E data.
- All 98 tests pass. `pytest.ini` restricts discovery to tests/ so the ignored
  checkpoint checkout no longer causes duplicate-module collection errors.
  Validation receipt: results/validation/chokepoint_protocol_checks.json.
- Existing halfFace final audit still passes with all 585 input hashes and
  frozen settings intact. Dashboard defaults and recognition modules were not
  changed. Augmented enrollment/gallery growth below remain separate priorities.

# Previous priorities - 21 October 2026

The user supplied a 30 October submission deadline and confirmed that an
experimental contribution with a working dashboard is acceptable. A new method
is not required. Paper writing can now proceed alongside the remaining work.

## Latest saved priorities - user asked to remember these

On 21 October the user asked to retain the following improvement plan and push
the current work to GitHub. These experiments are **planned, not completed**.

1. **Augmented one-photo enrollment:** compare one original photograph, that
   photograph plus controlled degraded variants, and three original photographs,
   using both ArcFace and AdaFace. This directly tests the unfinished augmented
   enrollment question. Fix the protocol before execution and retain negative
   findings; no improvement is promised.
2. **Gallery growth:** test 10, 25 and 55 enrolled identities with fixed
   thresholds and the same probe set. Keep the probed known identities enrolled
   at every size and add background identities. Account for gallery composition;
   repeated subsets are not new independent people. Report correct acceptance,
   false-known decisions, ambiguity and capture failures together.
3. **Complete runtime:** separately measure enrollment, detection, embedding,
   search and decision; report median and p95 with hardware and timing boundaries.
4. **Contribution argument:** compare our exact protocol/findings with the closest
   prior work in a clear table; do not claim existing components as inventions.
5. **Submission and defense:** write the thesis while experiments proceed; finish
   references, slides, traceable result tables, offline rehearsal and genuine
   screenshots/recording when a browser is available.

Recommended experiment budget is at most two days before concentrating on the
submission. All additions use development inputs; preserve the frozen final
evaluation. The original final identities are already examined and cannot become
a fresh unseen test by rerunning or repartitioning them. Defer the trained
adapter, transformer and full video pipeline unless the core submission is
finished early and valid evaluation is feasible. No new video collection is assumed.

The research story to develop is: **How should people be enrolled when photos
are limited, and how does reliability change as the gallery grows?** This is a
proposed experimental direction, not an established new method or campus impact
result. The [original-PDF audit](ORIGINAL_PDF_PROGRESS_AUDIT.md) found 5 met,
11 partial and 10 not-demonstrated checklist items; its approximately 40% score
is a planning convention, not a grade or percentage of effort.

## Verified checkpoint

Read [completion plan](COMPLETION_PLAN.md), [contribution assessment](NOVELTY_AUDIT.md)
and [defense evidence](DEFENSE_EVIDENCE.md). Do not claim a new recognition
algorithm or actual CCTV validation. The main observed benefit is few-photo
enrollment; quality and margin effects must be reported with their limits.

Completed this session: five enrollment pooling policies evaluated using only
real development caches. Independent verification checks 3,540 decisions and
15 summaries; six new safeguards pass. Heuristic weighting does not consistently
outperform the mean. See [new evidence](../paper_assets/development/pooling_v1/README.md).
The frozen final audit still passes; no final-test image inference was rerun.

Also complete: genuine official AdaFace IR-50 integration and a paired ArcFace
comparison on all 571 original development images. All 564 ArcFace features match
the cache exactly; seven capture failures are shared. The independent verifier
checks 2,832 prediction attempts and 19,072 calibration candidates. Nine new
AdaFace/paired-protocol tests pass. See [backbone evidence](../paper_assets/development/backbones_v1/README.md).

The new results refine the claim: score-only few-photo enrollment improves both
checkpoints, but separately calibrated ArcFace score/margin rules tie at 139/154
for one and few photos. AdaFace improves from 120 to 133/154 under its margin
rules. This is descriptive development evidence, with different pretraining and
runtime engines. It does not replace the original frozen final evaluation.

Defense cases are now prepared and visually inspected: 12 first-path examples
from 905 eligible category records, with helped/harmed/unchanged, unknown,
ambiguous, rejected, capture-failure and false-accept outcomes. See
[case analysis](DEFENSE_CASE_ANALYSIS.md) and [the pack](../paper_assets/development/defense_v1/README.md).
Photographic cards and the offline viewer stay in ignored data/defense; public
cards contain no portraits. Fixed-rule examples and separately calibrated
examples are explicitly separated.

`defense_app.py` provides case views and a ten-step isolated real-inference
walkthrough. Target halfface_013 is unknown in a seeded 54-identity gallery,
becomes known after fresh three-photo enrollment, persists across reload and
becomes unknown after removal. A genuine unknown probe produces AMBIGUOUS with
two different candidates in the 55-identity gallery. All ten steps pass, and
fresh successful features match cached features exactly. The Streamlit AppTest
also checks all 12 views and ten controls. Five new outcome/protocol tests pass.

A 149-second H.264 backup video is saved under the local defense run. It is an
assembled walkthrough of redacted cards, charts and five genuine prior browser
screenshots, NOT a fresh screen recording. All chapters decode; prior browser
evidence is independently verified. No connected browser is currently exposed;
fresh screenshots and a live recording remain pending. The user asked how to
connect: official guidance is Settings > Computer Use > Chrome > Install, follow
extension setup, confirm Manage and toggle on, then mention @Chrome. Instructions
were given while independent work continued. Do not claim fresh capture occurred.

The local defense server was started on http://127.0.0.1:8521, exec session 57422.
Keep it available while the user connects a browser. Existing app.py and frozen
research code remain unchanged. Dependency/setup cleanup and all 89 tests are now
complete locally. Follow the latest priorities above; clean-machine installation,
fresh browser evidence, remaining runtime work and thesis/slides are still pending.

# Previous session checkpoint - 20 October 2026

Work is paused for the day at the user's request. Paper writing is deferred.

## Verified work saved

- Real SCRFD detection, alignment and frozen ArcFace features.
- One-photo versus up-to-three-photo enrollment and calibrated open-set decisions.
- Controlled development tests for blur, resolution and JPEG compression.
- Frozen held-out halfFace result: 151/160 correct known acceptances (94.38%),
  with 0/151 observed false known acceptances. This is a limited public-dataset
  result, not a population error guarantee.
- Reserved ICFD public-sample engineering check: 98/100 closed-set retrieval.
- Persistent dynamic gallery and actual browser enrollment/recognition evidence.
- Results, figures, captions and screenshots indexed in `paper_assets/README.md`.

Raw face images, biometric feature arrays and model weights stay local under
the existing ignore rules. Reproduction still requires those local inputs.

## Next work after dashboard setup

The user cannot obtain video recordings in the remaining time. Continue with
still-image experiments; do not make real CCTV or temporal-performance claims.
AdaFace integration, backbone comparison, pooling comparison and defense cases
are complete as described above. Do not repeat them as pending work.

The original dashboard still exists unchanged. Its eight pages, both settings
presets, real enrollment, known/unknown/ambiguous decisions, reload, removal and
capture failure pass an isolated AppTest workflow. UploadedFile bytes are
injected at the uploader boundary; this is not new browser evidence. All 89
tests pass. See [setup and launch instructions](DASHBOARD_SETUP.md) and
`results/validation/setup_20261021/`. Runtime requirements, test requirements,
Conda configuration and double-click launchers now match the verified local
environment. A clean-machine installation has not been tested.

Next: follow the augmented-enrollment and gallery-growth plan above while writing
the thesis results/discussion and contribution statement, assembling defense
slides, and practicing the live demonstration.
Capture fresh browser screenshots and a real screen recording once a browser
is connected. Keep the assembled backup video labelled as assembled evidence.
Additional lighting/noise/occlusion experiments are optional and incomplete;
prioritize completing the thesis before 30 October.

Preserve the current held-out settings and predictions. That test set has already
been examined; further model comparisons on it must be labelled exploratory.
No domain adapter training, local CampusFace-BD dataset, validated video tracking
or transformer comparison has been completed.
