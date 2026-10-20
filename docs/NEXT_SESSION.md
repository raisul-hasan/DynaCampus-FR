# Current priorities - 21 October 2026

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
