# Supplementary study: temporal aggregation under controlled degradation

## Objective and evidence scope

This supplementary experiment examines whether combining multiple face frames
improves frozen recognition under controlled surveillance-style degradation,
and whether quality weighting adds value over uniform averaging. It extends
the main still-image enrollment and decision study with probe-side aggregation.
It is an exploratory, sequence-separated analysis of ChokePoint rather than a
fresh test of previously unexamined identities, a replication of the supplied
pilot scores, or validation of a deployed campus surveillance system.

## Experimental method

The official Portal-1 entering face crops and ground truth were obtained from
the original ChokePoint distribution [1]. Crop labels were checked against the
frame-level identity annotations. The 25 identities were deterministically
partitioned into 17 gallery-known and eight unknown identities using seed 42.
One first-available crop from sequence P1E_S1_C1 enrolled each known identity;
this rule did not select gallery images using recognition outcomes or assert
that each selected image was frontal. P1E_S2_C2 supplied calibration tracks,
and P1E_S3_C3 and P1E_S4_C1 supplied evaluation tracks. There were 143 known and
66 unknown calibration tracks and 303 known and 143 unknown evaluation tracks.
Identities were reused across roles, while recording sequences and source files
were disjoint. There were no identical source hashes between these roles.

Each query comprised five consecutive, non-overlapping frames of the same
annotated identity and camera sequence. The same construction was applied to
known and unknown queries. The supplied eye-aligned 96x96 grayscale crops were
decoded as BGR and resized to 112x112 using cubic interpolation before applying
the declared corruptions. Clean, mild, medium and severe tiers combined Gaussian
blur, spatial downsampling, JPEG compression and, at severe level, independent
Gaussian noise. Mixed tracks used per-frame tier probabilities 0.2, 0.3, 0.3 and
0.2. Stable seeds derived from SHA-256 made the corrupted pixels identical across
methods and repeatable across Python sessions.

All methods used the same frozen ArcFace ResNet-50 checkpoint and cosine gallery
search. The predeclared comparisons comprised first-frame and seeded random-frame
selection, best-frame selection using Laplacian/contrast quality or raw feature
norm, uniform averaging, and norm-based softmax weighting at beta values 0.5, 1,
2 and 5. Pooled vectors were unit-normalized after combining unit frame vectors.
Raw feature norms were preserved separately for quality estimation. No detector,
learned adapter or real tracker was evaluated.

Two evaluation protocols were kept separate. Pairwise verification generated
ROC, EER and TAR at stated pairwise FAR targets as descriptive evaluation-set
summaries. Open-set identification used independently frozen thresholds selected
from unknown calibration scores with at most 1% observed FPIR. This required
zero observed false accepts among 66 calibration unknown tracks. Fixed and AOSM
rules were calibrated separately; AOSM used margin 0.08, threshold floor 0.20 and
fixed feature-norm bounds 16-22. A shared calibration target does not imply equal
FPIR on evaluation data.

## Results

Under severe corruption, first-frame known-track Rank-1 was
295/303 (97.36%), compared with
302/303 (99.67%) for uniform pooling.
The paired mean-minus-first difference was +2.31
percentage points, with a 95% identity-cluster bootstrap interval of
[+0.96, +3.75] points.
The intervals used 1,000 paired identity resamples, retaining each sampled
identity's tracks and both evaluation sequences.

On mixed-quality tracks, uniform pooling achieved
303/303 (100.00%) Rank-1, while
norm-softmax pooling at beta 1 achieved
303/303 (100.00%). The corresponding
softmax-minus-mean difference was +0.00 points
(95% paired interval [+0.00,
+0.00]). All nine declared policies and paired
helped/harmed counts are retained in the accompanying report, including negative
or unchanged outcomes. Quality-proxy correlations are diagnostic and do not
by themselves establish a recognition benefit from weighting.

With mean pooling on mixed tracks, the calibration-frozen fixed rule correctly
accepted 303/303 known tracks and falsely accepted
12/143 unknown tracks. The separately calibrated
floored AOSM rule correctly accepted 303/303 known
tracks and falsely accepted 13/143 unknown tracks.
These are observed evaluation counts at calibration-selected operating points,
not an assertion of population error rates or exactly matched evaluation FPIR.
In particular, the fixed rule's mixed-track evaluation FPIR is
8.39% despite zero calibration false accepts. The
chosen calibration operating point therefore did not establish the intended
1% evaluation FPIR. Strong known-identity retrieval and reliable unknown
rejection must be assessed separately; evaluation thresholds were not retuned
after observing this limitation.

The contamination ablation replaced the last frame of each known track with
one frame from another known identity. Each method received the same intruder
image for the same track. Matched-tier and mild-tier intruder controls were both
retained; unknown tracks remained unchanged. This experiment measures sensitivity
to a simulated identity substitution, not the error rate of an actual tracker.

## Limitations and reproducibility

The study uses one small public portal dataset and synthetic independent noise,
which can favor temporal averaging. The calibration and evaluation populations
share identities; recording sequence and camera also change together. Only eight
unknown identities and 17 known identity clusters are available. Bootstrap
intervals therefore do not capture variation across new cameras, sites or
populations. Multiple beta settings are descriptive comparisons, without a claim
from unadjusted significance tests. Norm bounds were inherited from the supplied
pilot, whose P1E data had already been examined. No campus CCTV generalization,
dynamic enrollment benefit or trained domain adaptation follows from this study.

The corrected experiment differs from the supplied notebook in its portable
seeds, contiguous tracks, paired contamination inputs, preprocessing order,
unknown-track aggregation and floored threshold rule. Its numerical results must
therefore be attributed to the corrected protocol. Raw predictions, source and
model hashes, track manifests, selection accounting, threshold choices and an
independent verification receipt accompany the figures and numeric tables.
The original halfFace held-out evaluation and application defaults are preserved.

Figures: [degradation](fig01_degradation.png), [mixed pooling](fig02_mixed_pooling.png),
[quality](fig03_quality.png), [contamination](fig04_contamination.png),
[thresholds](fig05_thresholds.png). [Complete tables and reproduction](README.md).

[1] NICTA; Y. Wong, S. Chen, S. Mau, C. Sanderson and B. C. Lovell,
Patch-based Probabilistic Image Quality Assessment for Face Selection and Improved
Video-based Face Recognition, CVPR Workshops, 2011. Dataset source:
[official ChokePoint page](https://arma.sourceforge.net/chokepoint/) and
[original archive record](https://zenodo.org/records/815657).
