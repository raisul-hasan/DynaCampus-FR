# Original PDF plan compared with verified work

Assessment: 21 October 2026, Asia/Dhaka. This is a workspace evidence audit,
not confirmation of supervisor approval or work held outside this checkout.

## Conclusion

The original campus-CCTV research plan is **not complete**. The current work is
a substantial, reproducible **public-image experimental study with a working
dynamic-gallery demonstration**. It does not deliver the original local dataset,
trained domain adapter or evaluated temporal CCTV pipeline.

Using the complete-context PDF's 26-item master checklist, this audit finds
**5 met, 11 partial, and 10 not demonstrated**. An explicit equal-item scoring
convention (met = 1, partial = 0.5, not demonstrated = 0) gives
`(5 + 11 * 0.5) / 26 = 40.4%`. Rounded: **about 40% of that original checklist**.
This is a rough planning index, not an objective percentage of research effort,
scientific value, marks, or time remaining. Major missing experiments may require
more effort than several completed checklist items combined. No percentage for
the smaller submission scope is assigned without an agreed deliverable weighting
and a review of the actual thesis draft.

The user subsequently accepted an experimental contribution and dashboard,
without requiring a new method, and reported that new video collection is not
feasible before the 30 October deadline. That supports a narrower submission
plan; it does not mean institutional approval of the revised scope is evidenced.

## Sources and how they were read

The two planning PDFs listed below are retained locally and excluded from the
current Git tree, together with their text extracts. This dated audit preserves
the findings from their review.

- `Face Identification Research.pdf`: 49 pages. Its five contributions are on
  page 6; research questions on page 7; enrollment, adapter and temporal plans
  on pages 27-34; paper outline on pages 47-49.
- `DynaCampus_FR_Complete_Project_Context.pdf`: 52 pages. Fixed choices and
  objectives on pages 7-9; definition of done on page 10; data/splits on pages
  22-27; experiments/metrics on pages 36-39; deliverables on pages 46-48.
- Both PDFs were text-extracted. The original contribution page and the
  complete-context experiment matrix and master checklist were also rendered
  and visually inspected to verify headings, diagram labels and checklist items.
- The complete-context PDF clarifies exact cosine search as primary and FAISS
  as optional. It explicitly permits FaceLiVTv2 to be transparently omitted.
  These choices avoid counting optional software as missing core research.
- Research-question numbering differs between the PDFs. The question table
  below uses the complete-context PDF, rather than silently combining IDs.

The earlier `F:/Thesis Final/DynaCampus-FR_Project_Progress_Review.md` describes
an older checkout. Its statements that real experimental results and genuine
AdaFace inference are missing are no longer current. It is preserved as a
historical review; this document assesses the present state.

## Main deliverables

| Original area | Current verified evidence | Assessment against original plan |
|---|---|---|
| SCRFD, alignment, embeddings | Real checkpoints; shared preprocessing; ArcFace and genuine official AdaFace inference | Functional foundation complete; local detector precision/recall/AP by face size not established |
| Primary AdaFace system | Paired ArcFace/AdaFace development comparison using the same crops and original photos | Partial: dashboards and frozen final result use ArcFace; no complete adapted AdaFace CCTV pipeline |
| Dynamic enrollment | Real original-photo enrollment, template insertion, disk reload and removal; unchanged model hashes | Core training-free operation demonstrated; full audit/deactivation/withdrawal governance not demonstrated |
| CampusFace-BD | Collection guide and configuration | Not demonstrated: no verified consent-based 60-person, two-camera, two-CCTV-session dataset or CVAT evaluation |
| Public surveillance benchmark | halfFace image study and separate ICFD public-sample engineering check | Substitute evidence; not QMUL-SurvFace or the full planned surveillance benchmark |
| CCTV Domain Adapter | Residual MLP prototype and toy random-vector training script | Not a trained or evaluated research contribution; no valid with/without-adapter result |
| Augmented one-shot enrollment | Augmentor prototype; robustness tests degrade probes | Proposed enrollment experiment not completed: degrading probes is different from constructing augmented enrollment templates |
| Enrollment comparisons | One original photo vs up to three originals; five development pooling policies | Partial: useful evidence, but not the planned five-photo, augmented-photo and enrollment-video matrix |
| Tracking and temporal aggregation | SORT-style IoU/Kalman tracker; mean/best/weighted aggregation prototype; unit tests | No verified ByteTrack or real-video single/best/mean/Top-K/weighted experiment. Multiple enrollment photos are not temporal probe aggregation |
| Four-state decision | Implemented score, top-two margin and quality engine; calibrated development comparisons; frozen final thresholds | Strong still-image evidence. Selected quality gate is disabled; no established temporal or universal quality-rejection benefit |
| Robustness and conditions | Nine isolated Gaussian-blur, resolution and JPEG conditions plus control | Partial: not motion-blur validation, low-light/noise/pose/combined-degradation matrix or camera/distance/session study |
| Evaluation and statistics | Identity-disjoint development/final identities; frozen predictions, retained failures, paired cluster-bootstrap intervals | Partial against original: one frozen public-image evaluation rather than three rotating CCTV splits; no adapter seed study or full planned biometric metric suite |
| Efficiency | CPU detector/embedding stage timings, checkpoint details, gallery-operation timing evidence | Partial: no validated camera FPS, complete UI/search/decision latency study, peak-memory or gallery-scaling benchmark |
| Model comparison | Real ArcFace vs AdaFace development results | Two-checkpoint comparison complete for stated protocol; no transformer result. Omission is allowed if transparent |
| Dashboard and defense evidence | Original app retained; separate evidence app; real lifecycle rehearsal; prior genuine browser screenshots; redacted cases and assembled backup video | Still-image demonstration ready; fresh defense-browser capture/live recording and video-tracking demonstration remain absent |
| Tests and reproducibility | 89 passing tests; corrected runtime/dev requirements and launchers; independent result verifiers | Local checks pass; clean-machine install and end-user offline rehearsal are not demonstrated |
| Thesis and submission | Figures, CSV/LaTeX result tables, methods/provenance notes and contribution audit | Supporting assets exist; no complete final thesis, bibliography or defense slide deck found in the reviewed workspace |

## The five contributions in the first PDF

1. **CampusFace-BD:** not delivered in the available evidence.
2. **CCTV Domain Adapter:** architecture only; no valid measured adapter contribution.
3. **CCTV-augmented dynamic enrollment:** ordinary training-free enrollment is
   demonstrated, but the proposed augmented one-shot method is not evaluated.
4. **Quality-aware multi-frame aggregation:** prototype only; no real-track result.
5. **Four-state open-set decision:** implemented and evaluated on images, with
   the quality gate disabled by development selection. The margin's additional
   held-out gain is small and uncertain.

Thus the original five-part novelty package has not been achieved. The revised
contribution is the controlled experiment and its findings, plus executable
integration. Neither the dashboard nor using AdaFace establishes a new algorithm.

## Research questions from the complete-context PDF

| ID | Original question, shortened | Present answer |
|---|---|---|
| RQ1 | Individual/joint CCTV degradations | Partially answered: isolated Gaussian blur, image resolution and JPEG on development crops; no real CCTV condition matrix |
| RQ2 | Adapter generalization to unseen identities | Unanswered: no valid trained adapter experiment |
| RQ3 | Augmented one-shot enrollment benefit | Unanswered as written. One vs three real photos and weighting are different experiments |
| RQ4 | Quality-aware temporal aggregation | Unanswered: no verified temporal probe experiment |
| RQ5 | Score/margin/quality calibration | Substantially addressed for the still-image protocol, with weak margin benefit and a disabled quality gate reported honestly |
| RQ6 | AdaFace vs ArcFace vs hybrid transformer | Partially answered: two pretrained checkpoints on development data; no transformer or identical real CCTV evaluation |

The first PDF's RQ6 instead asks about adding identities without retraining.
That behavior is demonstrated, with reliability limited to the measured image
protocol. Its transformer comparison is RQ5, which remains incomplete.

## Exact 26-item master-checklist assessment

"Not demonstrated" includes external approvals and artifacts that may exist
elsewhere but were not available for this audit. Partial does not mean half the
labor is finished; half credit is only the stated planning convention.

| # | PDF checklist item | Status | Reason |
|---:|---|---|---|
| 1 | Repository created and private | Partial | Local repository exists; current remote privacy not independently verified |
| 2 | Working Python environment locked | Partial | Working versions and direct requirements recorded; no clean transitive environment reproduction |
| 3 | SCRFD detection verified | Met | Real local checkpoint and repeated actual image inference; benchmark accuracy is a separate unfinished requirement |
| 4 | Alignment and embeddings verified | Met | Real aligned inference, reference preprocessing checks and feature-cache comparisons |
| 5 | Dynamic gallery demonstrated | Met | Actual insertion, recognition, persistence and removal with unchanged models |
| 6 | Research questions approved | Not demonstrated | No approval record reviewed |
| 7 | Permission and consent approved | Not demonstrated | No local-study approval/consent evidence reviewed |
| 8 | Pilot completed and reviewed | Not demonstrated | Public-image engineering checks do not replace the planned campus camera pilot |
| 9 | Main dataset collected and backed up | Not demonstrated | No verified CampusFace-BD collection |
| 10 | Metadata and CVAT verification complete | Not demonstrated | Public image inventories exist; required video/track/CVAT evidence does not |
| 11 | Three identity-disjoint splits frozen | Partial | A fixed public-image development/final partition exists, not the three rotating planned splits |
| 12 | ArcFace and AdaFace baselines complete | Partial | Genuine paired image-development baselines; not the full original surveillance evaluation |
| 13 | Augmentation profiles validated | Partial | Three families of isolated probe stress tests; not complete augmentation/enrollment validation |
| 14 | Domain Adapter trained on development identities only | Not demonstrated | Existing random-vector toy training is not this experiment |
| 15 | Enrollment ablation complete | Partial | One/three-photo and pooling evidence; planned five-shot/augmented/video modes missing |
| 16 | Temporal aggregation ablation complete | Not demonstrated | No verified real-video temporal ablation |
| 17 | Open-set thresholds frozen from development data | Met | Pre-inference freeze and independent provenance verification |
| 18 | Three-split final evaluation complete | Partial | One valid frozen public-image evaluation, not three CCTV splits |
| 19 | Runtime and condition-wise results complete | Partial | Image stage timings and limited stress conditions; full runtime/condition matrix incomplete |
| 20 | FaceLiVTv2 complete or transparently omitted | Met | Omitted explicitly under the accepted smaller scope; no transformer performance claimed |
| 21 | Confidence intervals and failure analysis complete | Partial | Strong image-study intervals and case analysis; original camera/track and three-split scope absent |
| 22 | Paper tables match raw results | Partial | Verified result CSV/LaTeX/figures; no complete final manuscript to cross-check |
| 23 | Dataset/model cards complete | Not demonstrated | Provenance notes exist; no complete dedicated final card package found |
| 24 | Clean-environment rerun successful | Not demonstrated | Local venv reuses installed system packages |
| 25 | Minimal demo complete after research | Partial | Strong image demonstration; planned CCTV-video track display not verified |
| 26 | Privacy review and final deletion/retention plan complete | Not demonstrated | Planning documentation exists; no completed final governance review evidenced |

Counts: **5 met, 11 partial, 10 not demonstrated**. The optional transformer
omission receives credit because that option is explicit in the PDF checklist;
it does not imply that a third backbone was implemented.

## What the completed work establishes

- **Frozen halfFace:** one-photo score-only 130/160 correct known acceptances;
  few-photo score-only 150/160; few-photo score-plus-margin 151/160. The primary
  method has 0/151 observed false known accepts, across 27 unknown identities.
  These are calibrated-method comparisons, with separately frozen thresholds.
- **Uncertainty:** few-photo enrollment provides the larger observed gain.
  Margin over few-photo score-only adds one net acceptance; its paired interval
  includes zero. Zero observed false accepts is not a population zero-risk bound.
- **Paired development comparison:** official AdaFace and ArcFace use matched
  crops and enrollment inputs. Enrollment effects depend on checkpoint and rule;
  quality weighting does not consistently improve over simple averaging.
- **Stress tests:** fixed models/gallery/thresholds expose limitations under
  Gaussian blur, downsampling and JPEG. These are development image tests.
- **Executable mechanism:** people can be added/removed by changing templates,
  with model files unchanged. Unknown/ambiguous outcomes can be demonstrated.
- **Validation:** 89 tests pass, real original-app lifecycle checks pass, and
  prior browser evidence and frozen result provenance remain verifiable.

This supports practical feasibility for updating a gallery without backbone
retraining and studying refusal decisions. It does not establish campus security
improvement, real CCTV accuracy, reduced human workload, demographic fairness,
or successful deployment.

## Recommended route to 30 October

Finish the narrower study and describe it consistently throughout the title,
research questions, methods, results and claims. Use a title such as **Few-Photo
Enrollment for Open-Set Face Recognition under Masking and Image Degradation:
An Experimental Study**. Confirm the revised scope through the user's normal
supervisor process; this audit does not contact anyone or assume approval.

Priority remaining deliverables are the complete thesis and references,
defense slides, a precise prior-work/contribution comparison, final dataset/model
documentation, complete timing boundaries, clean-environment/offline checks where
feasible, and genuine fresh demo screenshots/recording. Preserve the frozen test
and report later development comparisons separately. Adapter, local campus data
and temporal CCTV research belong in clearly stated limitations/future work if
they cannot be completed and independently evaluated.

The existing final test is consumed. Adding a method and reusing those examined
identities does not create fresh held-out validation. None of the missing work
should be filled by relabelling toy training or image pooling as CCTV research.

## Evidence paths

- Frozen results: `paper_assets/heldout/final_v1/README.md` and
  `results/final/heldout_20261019T231512869782Z/verification.json`.
- Backbone comparison: `paper_assets/development/backbones_v1/README.md` and
  `results/backbones/backbones_development_20261020T200004286474Z/REPORT.md`.
- Enrollment pooling: `paper_assets/development/pooling_v1/README.md`.
- Robustness: `paper_assets/development/robustness_v1/README.md`.
- Decision ablation: `paper_assets/development/decision_v1/README.md`.
- Application: `paper_assets/development/demo_v1/README.md`.
- Defense: `docs/DEFENSE_EVIDENCE.md`, `docs/DEFENSE_CASE_ANALYSIS.md` and
  `paper_assets/development/defense_v1/README.md`.
- Current tests/setup: `results/validation/setup_20261021/setup_verification.json`,
  `pytest.xml` and `original_workflow.json` in that directory.
- Prototype limitations: `scripts/train_adapter.py`, `src/adapter/domain_adapter.py`,
  `src/tracking/tracker.py` and `src/aggregation/aggregator.py`.
- Current scope/prior-work boundary: `docs/COMPLETION_PLAN.md`,
  `docs/NOVELTY_AUDIT.md` and `docs/DASHBOARD_SETUP.md`.
