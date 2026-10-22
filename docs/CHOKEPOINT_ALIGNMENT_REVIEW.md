# ChokePoint supplementary study: alignment and integration

Reviewed 23 October 2026 (Asia/Dhaka). The supplied PDF and notebook are
preserved under [supplementary/chokepoint](supplementary/chokepoint/README.md).
Their instructions and code were assessed as source material, not executed.

**Update, 23 October:** the corrected, sequence-separated experiment has now
been run on the official P1E crops and verified by a separate recomputation
script. See [new results and thesis section](CHOKEPOINT_RESULTS.md). The source
notebook remains unedited and unexecuted; its original scores are not reproduced.
The intake findings below are retained to explain the protocol corrections.

## Decision

Yes: add this as a **supplementary exploratory study of temporal aggregation
and quality-dependent decisions**. It fits DynaCampus-FR's low-quality face
recognition motivation and supplies experiments missing from the still-image
study. Negative results are useful experimental contributions too.

The paper's numerical claims are **reported, not independently reproduced**.
They must not be presented as new verified project results. At intake, the
inspected input folders contained halfFace and ICFD, but no ChokePoint dataset.
There is no editable thesis manuscript in this checkout; this addition supplies
an integration review, source archive and draft thesis wording.

## Alignment

| Research area | Supplied work | Appropriate use |
|---|---|---|
| Low-quality surveillance | Four synthetic blur/resolution/noise/JPEG tiers | Supplement existing stress tests; not campus deployment validation |
| Frozen recognition | ArcFace ResNet-50 and cosine matching | Compatible baseline; not evidence for a trained domain adapter |
| Temporal aggregation | Five-frame mean, best-frame and norm-softmax pooling | New probe-side comparison; existing enrollment pooling averages gallery photos |
| Quality estimation | Laplacian/contrast versus raw embedding norm | Test proxy usefulness and noise sensitivity |
| Open-set decisions | Fixed versus quality-adaptive thresholds | Negative ablation; retain current calibrated fixed score/margin app settings |
| Tracking failure | One intruder frame per five-frame track | Simulated contamination, not actual SORT/ByteTrack evaluation |
| Dynamic enrollment | Static 17-identity gallery | Does not demonstrate insertion, persistence, removal or gallery-growth behavior |
| Backbone comparison | ArcFace in the manuscript | Does not automatically extend the separate AdaFace experiment |

Describe the contribution as empirical analysis. Do not claim temporal averaging,
quality weighting or threshold relaxation as newly invented recognition methods.

## Reported findings worth retaining

These values are transcribed from the paper, not recalculated. Their denominators
and protocols differ from the frozen halfFace evaluation.

| Source | Reported result | Interpretation |
|---|---|---|
| Table I, severe | Single-frame Rank-1 86.3%, TAR at pairwise FAR 1% 70.2%, EER 8.5%; five-frame mean 93.3%, 83.1%, 4.6% | Reported temporal-pooling benefit under synthetic degradation |
| Table II | Mean Laplacian quality 0.1746 at medium and 0.3124 at severe; norm/similarity within-tier Spearman correlations about 0.20-0.33 | Motivation to audit noise sensitivity; nominal frame-level p-values assume independence |
| Table IV | Uniform and norm-softmax pooling both reach 100.0% Rank-1 on mixed tracks | No demonstrated weighting advantage; ceiling and missing intervals limit conclusions |
| Table V, 20% contamination | Uniform Rank-1 99.2%, norm-best 53.3%, norm-softmax beta 2 at 55.7% | Stress-test observation requiring a paired rerun |
| Section V-F | At unknown-query false-alarm target 1%: fixed DIR 96.5%, AOSM DIR 91.8% | Negative adaptive-threshold result in the supplied exploratory protocol |

Do not merge these denominators with halfFace results or retune its consumed
held-out set. No publication venue, peer review or independent reproduction is
established by the supplied local draft.

## Corrections needed before numerical integration

Cell numbers below are zero-based notebook indices.

1. **Execution evidence:** only cells 0 and 1 have saved execution counts.
   Cell 1 ends with `NameError: name 'os' is not defined`; the other 47 cells
   have no saved outputs/counts. Several later variants depend on earlier state.
2. **Stable seeds:** cells 40, 43, 44 and 48 derive noise seeds from `hash(path)`.
   String hashes can vary between processes, and absolute paths vary by machine.
   Use a stable digest of relative path, condition and master seed; save seeds
   and corrupted-image hashes. A fixed NumPy seed alone does not fix this.
3. **Paired contamination:** cell 48 creates one RNG outside the method loop,
   then samples intruders inside it. Methods can receive different intruder
   images for the same track. Precompute one contamination manifest for all
   methods and include a same-quality intruder control as well as mild intruders.
4. **Track construction:** cells 3/47 group by parent directory name; later
   cells sort all images per identity and chunk them. Both P1E and P1L archives
   appear in the code, without an explicit P1E-only filter, camera/sequence
   boundary check or numerical temporal-continuity check. Audit labels against
   ground truth and save portal, sequence, camera, identity and frame IDs.
5. **Gallery selection:** first sorted image in cell 48 is not an independently
   verified frontal P1E crop. Save and inspect the enrollment manifest.
6. **Resize mismatch:** the paper degrades 112x112 crops using bicubic resizing
   to 28x28 at severe level. Later notebook cells degrade the original input
   before recognizer resizing and use OpenCV's default interpolation. Official
   crops are 96x96, so unmodified crops reach 24x24, not 28x28. Record actual
   dimensions and reconcile preprocessing order and interpolation.
7. **AOSM floor:** paper Eq. (6) uses
   `max(tau_base - (1-q)*0.08, 0.20)`; cell 48 uses additive score shifting
   without the floor across thresholds 0.10-0.85. They agree only when the floor
   is inactive. Reconcile the equation/code and check the reported operating points.
8. **Asymmetric query units:** known queries are pooled five-frame tracks;
   unknown queries are single frames. Disclose this in the original study.
   Also evaluate unknown tracks under the same aggregation and length policy.
9. **Calibration:** late cells sweep thresholds on the evaluated queries,
   without a separate calibration set. These curves describe exploratory
   performance, not an independently tested deployment threshold. Select norm
   bounds, beta, margin and thresholds on development sequences, then freeze them.
10. **Metric denominators:** pairwise FAR uses non-mated comparisons; open-set
    false alarms use each unknown query's gallery maximum (FPIR). Rank-1 is
    retrieval before rejection. The claimed 8,160 impostor pairs are consistent
    with `255*16 + 240*17`; this arithmetic does not verify actual data membership.
    With 240 unknown queries, FPIR steps are about 0.417 percentage points, and
    a 1% target permits at most two observed false accepts. Pairwise FAR 0.1%
    rests on about eight of 8,160 comparisons, which are not independent.
    Report counts and paired identity/sequence-aware uncertainty intervals.
11. **Table reconciliation:** Table I severe mean is 93.3% Rank-1, 83.1%
    TAR@1%, 4.6% EER; Table III all-severe uniform is 94.1%, 84.3%, 4.3%.
    Trace each to a run/seed. Table I's rounded 86.3 to 93.3 change is 7.0 points;
    verify its stated +7.1 from raw counts. Cell 43's `Random Frame` chooses
    index zero: label it first-frame or save a seeded random frame choice.
12. **Implementation distinction:** the app's `FeatureAggregator` uses linear
    normalized quality weights, whereas the manuscript uses standardized softmax
    weights. The current fixed-threshold decision engine does not implement AOSM.
    Shared names do not establish equivalent algorithms.
13. **Loader cleanup:** cell 43 stops extracting later archives once any PGM
    exists; cell 47 lists ZIPs but only extracts TARs. Use explicit archives,
    checksums and a verified inventory. Keep images, embeddings and weights local.

The [official ChokePoint source](https://arma.sourceforge.net/chokepoint/)
confirms the surveillance setting, 25 Portal-1 subjects, 96x96 supplied crops
and separate development/evaluation sequence groups. Its dataset reference lists
pages 81-88; the supplied PDF lists 74-81. Verify the bibliography before submission.
This source check does not validate the notebook's derived split or scores.

## Thesis addition

Place this after the main results as **Supplementary exploratory study: temporal
aggregation under controlled degradation**, with separate data, protocol and limits.

Suggested research question:

> Under controlled surveillance-style degradation, when does temporal mean
> pooling outperform frame selection or quality weighting, and does a
> quality-dependent threshold improve identification at a matched FPIR?

Draft discussion text for the current evidence status:

> A supplementary ChokePoint pilot manuscript examines frozen ArcFace
> representations under synthetic blur, resolution loss, noise and compression.
> It reports that five-frame uniform pooling improves recognition under severe
> degradation, while quality weighting provides no demonstrated advantage over
> uniform pooling and quality-adaptive threshold relaxation reduces DIR at the
> examined matched false-alarm rate. These observations motivate simple
> aggregation baselines and reporting negative ablations. The supplied notebook
> has not yet been independently reproduced, and its seeding, track construction,
> contamination pairing and preprocessing require reconciliation. These pilot
> observations therefore remain separate from the verified still-image results
> and do not establish campus CCTV or real-tracker performance.

## Path to verified evidence

1. Obtain exact ChokePoint archives/ground truth and record sources, terms,
   checksums, recognizer hash, versions and actual execution provider.
2. Audit identities, frames, enrollment, sequences and cameras. Preserve the
   supplied split as exploratory; define separate calibration/evaluation sequences
   and disclose that this changes the original same-sequence protocol.
3. Consolidate one runner with stable seeds, identical method inputs and saved
   contamination manifests. Save raw norms before unit normalization; normalized
   cached embeddings alone cannot recover the norm quality signal.
4. Evaluate first/random/best, mean and declared softmax settings on paired
   known/unknown tracks. Keep the original asymmetric protocol only as a
   separately labelled reproduction.
5. Save per-query scores, decisions, quality, failures, exclusions and seeds;
   independently recompute metrics and intervals and link every table to a run.
6. Promote only verified rerun results into the thesis evidence index. Corrected
   experiments may yield different numbers. Retain the existing frozen halfFace
   methods and dashboard defaults until separate evidence supports a change.
