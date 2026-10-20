# Contribution assessment and prior-work boundary

Assessment date: 21 October 2026 (Asia/Dhaka). Submission deadline supplied by
the user: 30 October 2026. An experimental contribution with a working dashboard
is acceptable; a new recognition architecture is not required.

## Assessment

The project has a reproducible experimental contribution and a demonstrated
engineering prototype. It has not established a new face-recognition algorithm.
The current evidence does not support calling the backbone, embedding averaging,
quality weighting, adaptive thresholds or dynamic enrollment novel inventions.
High accuracy and a working dashboard establish measured behavior and feasibility;
neither, by itself, establishes novelty relative to published research.

The strongest existing result is the enrollment-policy comparison: on the same
held-out halfFace known probes, one-photo score decisions accept 130/160 correctly,
few-photo score decisions accept 150/160, and few-photo score-plus-margin decisions
accept 151/160. This separates the large observed enrollment benefit from the
small, uncertain additional margin benefit. The primary method observes no false
known accepts among 151 unknown probes; that is not a zero-risk guarantee.

The defensible contribution is an original, auditable experimental investigation
of enrollment and rejection choices under the specified masked-probe and synthetic
degradation protocol, plus a persistent gallery implementation demonstrating the
evaluated workflow without backbone retraining. Do not claim the study is the
first of its kind without a more exhaustive comparison of protocols and sources.
The degree/supervisor determines whether this contribution meets its standard.

The completed AdaFace comparison adds an important qualification: score-only
few-photo enrollment improves development acceptance for both checkpoints, but
the separately calibrated ArcFace one-photo and few-photo margin rules tie at
139/154. AdaFace margin acceptance changes from 120 to 133/154. This supports an
investigation of interactions between enrollment, checkpoint and rejection rule;
it does not establish a universal benefit from adding photos. These new policies
were evaluated on development data and cannot be retroactively added to the
predeclared final comparison. See [the evidence](../paper_assets/development/backbones_v1/README.md).

Suggested research question:

> Under a fixed pretrained recognizer, how do enrollment policy and open-set
> rejection choices affect correct identification, unknown acceptance and refusal
> on masked face images and controlled image degradations?

Suggested title for discussion:

> Few-Photo Enrollment for Open-Set Face Recognition under Masking and Image
> Degradation: An Experimental Study

This title better matches the validated work than claims of learned domain
adaptation, campus deployment or evaluated temporal CCTV recognition.

## Closest work checked

Six successful Exa searches returned 36 search-result entries, including duplicates.
This was a targeted first-pass review, not a systematic review of 36 unique papers.
Primary-source pages were checked for the specific overlaps below. A search miss
is not evidence that an idea has never been published.

| Prior work | Established overlap | Boundary for our contribution |
|---|---|---|
| [Quality Aware Network, CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Liu_Quality_Aware_Network_CVPR_2017_paper.html) | Learned quality-aware aggregation for sets | Our simple quality weighting is a baseline/ablation, not a new principle |
| [Crystal Loss and Quality Pooling](https://arxiv.org/abs/1804.01159) | Detector-score quality pooling and template recognition | Detector-confidence weighting must be attributed as an existing idea |
| [Data-specific Adaptive Threshold for Face Recognition and Authentication, 2018](https://arxiv.org/abs/1810.11160) | Threshold updates and online registration with a changing database | Inserting identities without retraining or adjusting thresholds is not inherently new |
| [SER-FIQ, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Terhorst_SER-FIQ_Unsupervised_Estimation_of_Face_Image_Quality_Based_on_Stochastic_CVPR_2020_paper.html) | Recognition quality from stochastic embedding robustness | Embedding stability is not an unexplored quality principle |
| [Gallery-Aware Uncertainty Estimation for Open-Set Face Recognition, 2024](https://arxiv.org/abs/2408.14229) | Joint treatment of gallery ambiguity and embedding uncertainty | Combining quality with gallery ambiguity needs a precise distinction, not a broad novelty claim |
| [LocalScore, 2026 preprint](https://arxiv.org/abs/2602.01012) | Local-density scoring for multi-sample galleries and open-set biometrics | Multi-sample gallery scoring is an active, established research area |
| [AdaFace official implementation](https://github.com/mk-minchul/AdaFace) | Existing quality-adaptive recognition backbone | Integrating AdaFace supplies a comparative baseline, not our algorithmic novelty |

The relevant difference we can currently defend is the specific controlled
experiment and resulting evidence: fixed real preprocessing, paired enrollment
policies, development-only selection, identity-disjoint final evaluation, recorded
failures, uncertainty, controlled degradations and a matching still-image demo.
This is an experimental contribution claim. It is not a claim that each component
or their broad combination is unprecedented.

## Contribution statements and evidence

1. **Quantify few-photo enrollment benefit.** The final score-only comparison
   changes correct known acceptance from 130/160 to 150/160, with false unknown
   acceptance counts changing from 1/151 to 0/151. Methods have separately frozen
   development-calibrated thresholds; this is a comparison of calibrated methods.
   [Final evidence](../paper_assets/heldout/final_v1/README.md).
2. **Separate useful changes from weak ones.** Margin decisions add one net known
   acceptance over few-photo score-only decisions; its paired interval includes
   zero. The development-selected quality gate is disabled. The new development
   pooling comparison finds no consistent benefit of the tested heuristic weights
   over the mean. [Pooling evidence](../paper_assets/development/pooling_v1/README.md).
3. **Characterize limits under controlled degradation.** Blur, reduced resolution
   and JPEG compression alter capture success and rejection behavior. These tests
   reuse development source images and do not create independent new subjects.
   [Robustness evidence](../paper_assets/development/robustness_v1/README.md).
4. **Demonstrate the evaluated enrollment workflow.** The gallery accepts new
   templates without changing pretrained checkpoints, persists them, supports
   removal and reproduces the recorded decisions. This is an engineering
   contribution, not a deployment accuracy result.
   [Demo evidence](../paper_assets/development/demo_v1/README.md).

## Rules for the remaining work

- Keep the existing held-out evaluation as the final result for its frozen methods.
- AdaFace and later pooling comparisons start on development data. Any later use
  of the already examined final identities is exploratory or a reproduction, not
  a fresh unseen-test claim.
- New independent evaluation requires data with an audited absence of prior use,
  not another random split of identities whose results have already been examined.
- If AdaFace is compared, attribute checkpoint/pretraining differences; a pair of
  public checkpoints does not isolate the effect of their loss functions.
- Do not train the current toy adapter and label it validated domain adaptation.
- Prefer an honest negative result to selecting a method because its name sounds
  novel. No improvement is promised before an experiment is run.
- Show the literature boundary and the evidence together at defense. A dashboard
  proves executable functionality; the paired experiments support the contribution.
