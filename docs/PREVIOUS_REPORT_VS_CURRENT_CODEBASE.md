# Previous FYDP report compared with the current codebase

Review date: 23 October 2026. Project reviewed: `F:/Thesis Final/DynaCampus-FR`.

The campus face-recognition motivation remains, but the implementation and the
research contribution have changed substantially. The previous report describes
a seven-class CNN training project with a desktop/dashboard/database application.
The current validated work evaluates enrollment and open-set rejection around
frozen pretrained face embeddings, with a persistent gallery and a Streamlit
prototype. The new report needs rewritten methods and results, not just extra
model names added to the old chapters.

## 1. Sources and review boundaries

Historical source: `C:/Users/msi/Downloads/FYDP_II_Report__CCTV_Based_Face_Detection_and_Recognition.pdf`.
All 56 PDF pages were extracted and reviewed. Four contact sheets covering every
page and selected full-page renders were inspected, including the numerical
tables and confusion matrix. Page references below mean physical PDF page;
printed report page numbers are six lower in the main chapters.

The current review covers the first-party application, all `src/` modules,
experiment and verification script families, configuration files, the four main
notebooks, test modules, result receipts and paper assets. The inventory contains
104 Python files / 13,049 lines, 17 configuration files, four notebooks and six
environment/dependency/test-configuration files: 131 files altogether.
See [the file inventory and component assessment](CODEBASE_REVIEW_INVENTORY.md).
Core inference, enrollment, decision, final-evaluation, backbone and supplementary
experiment paths received detailed source review; utility/rendering/test files
were also inventoried and examined for their role and evidence boundaries.
This is a research/codebase comparison, not a security certification or an
exhaustive defect audit of every possible execution path.

The old application source and old raw training/prediction artifacts were not
provided with the PDF and were not identified in this current checkout. Therefore
the historical side below means **reported in the PDF**, not independently
reproduced old behavior. The PDF's embedded requirements were treated as
historical material, not instructions to change or execute the current system.
The review did not train models, alter application code or repeat reserved-test
image inference. Existing uncommitted work was preserved.

## 2. What changed

| Area | Previous report | Current implementation/evidence | Writing consequence |
|---|---|---|---|
| Research emphasis | Real-time educational CCTV monitoring; compare PCA-KNN with a trained CNN | Few-photo enrollment, unknown rejection, ambiguity, degradation, checkpoint comparisons and reproducibility | Keep campus motivation; state the narrower evaluated scope |
| Main task | Seven-class identity classification plus a confidence-based known/unknown test | Open-set identification: retrieve a gallery identity, then accept or refuse it | Define identification, retrieval and rejection separately |
| Main dataset | LFW: 1,288 images of seven people; 1,030 training / 258 test (PDF pp. 27, 36) | halfFace v2 originals: 926 usable images / 164 IDs after documented exclusions | Replace dataset and split sections |
| Additional data | 400 Olivetti unknown images (pp. 29–30, 41–42) | ICFD public sample; separate ChokePoint temporal supplement | Keep each dataset's role and limits separate |
| Face detection | Viola–Jones Haar cascade (pp. 27, 37) | Real local SCRFD ONNX detector, threshold 0.50, 640×640 detector input | Replace Haar methodology and diagram |
| Preprocessing | Gaussian/Sobel/Canny enhancement, ROI extraction and 40×40 input | Five landmarks and similarity alignment to 112×112; fixed neutral padding for tight benchmark crops | Do not carry over edge-enhancement claims |
| Main recognizer | Custom Conv32 → pool → Conv64 → pool → dense128 → seven-way softmax | Frozen ArcFace checkpoint producing 512-dimensional normalized embeddings | Replace custom CNN architecture and training description |
| Model training | Adam, categorical cross-entropy, 50 epochs; TensorFlow/Keras | Main experiments perform inference, template construction and threshold calibration; no backbone training | Explain pretrained training vs local enrollment/calibration |
| Adding a person | Seven-class output ties the reported model to its trained identities | Add/update a gallery template without modifying backbone weights | Dynamic enrollment is a real implemented addition |
| Multiple enrollment images | No paired one-photo/few-photo experiment reported | Normalized mean of up to three original unmasked views, compared against one-photo enrollment | This is the strongest primary experimental result |
| Matching | CNN class confidence; PCA-KNN comparator | Exact NumPy cosine search across unit templates | Cosine similarity is not a class probability |
| Unknown rule | Confidence ≥0.85 accepted, otherwise rejected | Development-calibrated score and top-two gap; explicit ambiguity and quality refusal | The old 0.85 threshold has no numerical correspondence to current thresholds |
| Decision outcomes | Known / Not Recognized | KNOWN, UNKNOWN, AMBIGUOUS, INSUFFICIENT_QUALITY; capture failure recorded separately | Report refusals and failures alongside acceptance |
| Quality handling | Image enhancement and general low-quality motivation | Blur/resolution/landmark-symmetry heuristic; main selected quality gate is disabled | Do not claim validated learned quality estimation or a quality-gating gain |
| Temporal processing | Tracking/aggregation mainly in conceptual objectives | SORT-like tracker and linear quality pooling in live prototype; separate evaluated five-frame ChokePoint study | Distinguish live implementation from crop-based experimental evidence |
| Application | Reported desktop application, security dashboard and PostgreSQL logs (pp. 30–32, 40–44) | Streamlit original dashboard plus separate defense evidence app | Replace interface screenshots and deployment architecture |
| Persistence | PostgreSQL Users and Recognition Logs tables | Gallery JSON metadata, NPZ embeddings and optional JPEG thumbnails | Old SQL schema is no longer an implementation diagram |
| Logs/notifications | Persistent recognition history, daily statistics and alerts reported | Live recent-event lists; saved experiment files; defense JSONL events | These do not establish a production recognition-event database or alert service |
| Experimental rigor | Image-level 80:20 split; a fixed confidence cutoff | Frozen identity-disjoint development/final roles; paired inputs; hashes; raw scores; independent recomputation scripts | Add protocol, leakage controls, uncertainty and reproduction sections |
| Deployment evidence | Webcam/real-time operation claimed, but final limitations say no deployment | Still-image workflow and persistence demonstrated; CPU image timings measured; live camera/RTSP not benchmarked | Use “research prototype,” not “validated campus deployment” |

Relevant code: [detector](../src/detection/detector.py),
[aligner](../src/alignment/aligner.py), [ArcFace](../src/recognition/embedder.py),
[gallery](../src/gallery/gallery.py), [decision engine](../src/open_set/decision.py),
[shared image workflow](../src/demo_workflow.py), [original app](../app.py),
[defense app](../defense_app.py).

## 3. Dataset and protocol changes that matter

### Main halfFace study

The original full-face collection contains 941 files / 166 source identities.
Six exact image pairs occur under conflicting IDs p039 and p090. Both identities
(12 files) were excluded, along with three files whose condition labels were
ambiguous. The resulting usable manifest has 926 images / 164 IDs. Supplied
upper-face crops and augmentations are not treated as independent new subjects.
See [the inventory](../results/tables/data_readiness/halfface_inventory.json)
and [preparation code](../scripts/prepare_halfface_inventory.py).

The frozen identity roles are 55 development-known, 27 development-unknown,
55 final-known and 27 final-unknown. Known probes are masked originals; enrollment
uses original unmasked photos. Unknown probes include their available original
conditions. This is substantially different from training and testing a classifier
on images of the same seven LFW identities.

“Training-free enrollment” means no local backbone retraining when a new identity
is inserted. It does not mean that ArcFace or AdaFace was never trained. Threshold
selection uses development labels. Possible overlap with external pretraining
data and near duplicates is not ruled out by the local identity split.

### ICFD

The available public sample contains ten IDs / 400 images. Five IDs support
development engineering checks and five support a reserved closed-set check.
The final check retrieves 98/100 correctly for both enrollment policies, with two
capture failures. There are no unknown probes in this check, so it cannot support
an ICFD open-set false-acceptance claim. The full ICFD release is not available
merely because the public sample was downloaded.

### ChokePoint

The separate supplement uses supplied 96×96 eye-aligned face crops, resized to
112×112 before controlled corruptions. It does **not** evaluate SCRFD or the live
tracker. One gallery sequence, one calibration sequence and two evaluation
sequences are separate in source files; identities recur between roles.
There are 17 enrolled and eight unknown IDs; evaluation contains 303 known and
143 unknown five-frame tracks. Because the supplied pilot had already examined
this material, the corrected study is exploratory supplementary evidence.

The earlier research notebook/PDF is archived under `docs/supplementary/chokepoint/`.
Its figures are not substituted for the corrected experiment's outputs.
See [the alignment review](CHOKEPOINT_ALIGNMENT_REVIEW.md) and
[results](CHOKEPOINT_RESULTS.md).

## 4. Current results that can go into the paperwork

### Primary result: frozen ArcFace evaluation

All methods below use the same final 55-person gallery identities, 160 masked
known-probe attempts and 151 unknown probes. Their operating thresholds were
selected on development data and frozen before final inference.

| Method | Enrollment photos actually used | Correct identity accepted / all known attempts | Unknown falsely accepted |
|---|---:|---:|---:|
| One-photo, score only | 55 | 130/160 = 81.25% | 1/151 = 0.66% |
| Up to three photos, score only | 159 | 150/160 = 93.75% | 0/151 observed |
| Up to three photos, score + margin (primary) | 159 | 151/160 = 94.38% | 0/151 observed |

The primary known-acceptance identity-bootstrap interval is 88.89%–98.74%.
The score-only enrollment comparison improves by 12.50 percentage points
(paired interval 6.92–18.75). The additional margin method improves by one net
probe, 0.625 percentage points (interval −1.28 to 2.55), which does not establish
a clear additional benefit. Score thresholds also differ between calibrated
methods, so the comparison is not a fixed-threshold isolation of the margin.

For the primary method, 151 known probes are correctly accepted, seven known
probes are rejected, one known probe is ambiguous, and one fails capture.
All 151 unknown probes receive UNKNOWN. There are no wrong-identity accepted
known probes. Overall decision coverage is 309/311. Rank-1 retrieval is
159/160 = 99.38%, which is **not** the same as 94.38% correct acceptance.

Primary settings: cosine threshold `0.3778589367866516`, minimum top-two gap
`0.15`, quality gate `0.0`. Zero observed unknown errors does not prove a
population false-acceptance rate of zero. A resampled all-zero error interval
also does not establish a zero-risk bound.

Sources: [final summary](../results/final/heldout_20261019T231512869782Z/summary.csv),
[paired effects](../results/final/heldout_20261019T231512869782Z/paired_effects.csv),
[freeze receipt](../results/final/heldout_20261019T231512869782Z/freeze.json),
[paper figures and LaTeX table](../paper_assets/heldout/final_v1/README.md).

### Additional findings and their evidence status

| Study | Finding | Status for the report |
|---|---|---|
| AdaFace versus ArcFace | Few-photo score/margin acceptance: AdaFace 133/154 with 0/149 false unknown accepts; ArcFace 139/154 with 1/149 | Completed development comparison; not a new held-out result or a controlled comparison of loss functions |
| One versus few photos with AdaFace | Score-only 112→122/154; score/margin 120→133/154 | Descriptive development finding |
| One versus few photos with ArcFace in that development study | Score-only 129→138/154; separately calibrated score/margin rules tie at 139/154 | Enrollment gains depend on the rule and selected operating point |
| Enrollment pooling | Tested uniform mean, heuristic weights, detector-confidence weights and best-quality selection | No consistent quality-weighting gain established; retain the simple mean baseline |
| Synthetic enrollment | One original plus five mild variants; recalibrated score-only ArcFace 129→128/154 and AdaFace 112→111/154 | Completed negative/mixed development result; not a replacement for independent original views |
| Gallery growth | Fixed probes and existing thresholds at 10, 25 and 55 gallery IDs; five 25-ID background compositions | Small descriptive composition study; unknown acceptance and ambiguity need reporting together; not large-scale deployment validation |
| Image robustness | Blur, resolution reduction and JPEG stress tests with fixed galleries/thresholds | Development stress tests; transformed versions do not add independent subjects |
| Temporal supplement | Severe-tier Rank-1: first frame 295/303 (97.36%), mean 302/303 (99.67%) | Exploratory five-frame crop result under specified synthetic degradation |
| Temporal threshold transfer | Mixed mean accepts 303/303 known but falsely accepts 12/143 unknown tracks (8.39%) after calibration targeted ≤1% | Important limitation; do not present calibration target as achieved evaluation FPIR |
| Dynamic gallery demo | Insert, recognize, reload, remove; separate known, unknown, ambiguous and capture-failure cases | Working engineering evidence; no deployment accuracy inference |

AdaFace uses official IR-50/MS1MV2 assets with pinned source, strict loading and
preprocessing checked against the archived implementation. Both backbones use
the same SCRFD-aligned crops; this is not AdaFace's official MTCNN benchmark.
Checkpoint pretraining and inference engines differ. The application still uses
ArcFace; AdaFace is a separate comparative research path.

Evidence packs: [backbones](../paper_assets/development/backbones_v1/README.md),
[pooling](../paper_assets/development/pooling_v1/README.md),
[robustness](../paper_assets/development/robustness_v1/README.md),
[extensions](../paper_assets/development/extensions_v1/README.md),
[ChokePoint](../paper_assets/supplementary/chokepoint_v1/README.md),
[defense cases](../paper_assets/development/defense_v1/README.md).

### Runtime is now measured, within explicit boundaries

| CPU path | ArcFace median / p95 | AdaFace median / p95 |
|---|---:|---:|
| One-image enrollment including persistent insertion | 136.97 / 173.15 ms | 199.88 / 221.32 ms |
| Image decode through recognition decision | 143.81 / 175.57 ms | 207.03 / 241.83 ms |

These are repeated measurements on fixed development samples, two CPU threads,
with 30 enrollment and 60 recognition image calls per model. The search
microbenchmark uses the actual `FaceGallery.search`, including matrix assembly
and sorting: at 55 IDs, ArcFace median search is approximately 0.064 ms.
These measurements exclude model loading, browser transport, camera capture,
RTSP/network delays and live tracking. Do not convert the search time, or the
reciprocal of median image latency, into a claimed measured camera FPS.
See [runtime boundaries](../results/enrollment_extensions/extensions_20261022T193101706934Z/runtime_environment.json)
and [runtime table](../results/enrollment_extensions/extensions_20261022T193101706934Z/runtime_summary.csv).

## 5. What remains a prototype or future work

| Component | What exists | What cannot yet be claimed |
|---|---|---|
| FaceLiVT / transformer comparison | `facelivtv2.py` wrapper, fetch script, local source and ~18.97 MB checkpoint, provenance JSON and comparison drafts | No completed calibrated paired transformer experiment or verified result pack was found |
| Live webcam / RTSP | Streamlit pages, camera-opening loops, tracker, aggregation and manual review queue | Validated continuous CCTV throughput, tracker identity accuracy, multi-camera deployment or representative campus recognition |
| Residual domain adapter | MLP module and synthetic-vector training/evaluation scripts | Trained and validated adaptation between real enrollment and CCTV domains |
| Quality estimator | Heuristic blur, resolution and landmark symmetry scores | Learned FQA, demographic fairness, or a demonstrated gain from the main quality gate |
| PostgreSQL / institutional backend | Historical report describes one; current gallery uses local files | Current SQL user/log database, login/roles, protected event history, attendance records, external alerts or access-control actuation |
| CampusFaceBD collection | Planning/configuration material | Collected local participants, measured Bangladesh-wide performance or local demographic fairness |
| Other old future work | Design aspirations for cloud/mobile/multiple cameras/anti-spoofing | Completed cloud service, mobile app, liveness or spoof resistance |

### FaceLiVT status found during this review

There has been progress beyond a configuration entry: local model assets and a
wrapper are present. However, the current comparison runner is not a completed
experiment. Static review found:

- `scripts/compare_transformers.py` calls `main()` before importing
  `FaceEmbedder`, `FaceLiVTv2Embedder` and the detector/aligner classes, so direct
  script execution reaches unresolved names.
- It subsequently treats ArcFace's ONNX wrapper as a PyTorch module with
  `.parameters()`, and uses detector/alignment interfaces inconsistent with the
  actual list-of-dictionaries / crop-only APIs.
- `validate_facelivtv2.py` has a syntax error at line 165 and further escaped
  expressions; `temp_test.py` contains literal `\n` escapes and does not parse.
- The fetcher/provenance records `main`, not an immutable upstream commit.
  Local hashes are present, but the wrapper's “pinned” wording is too strong.
- The wrapper uses unrestricted `torch.load(..., weights_only=False)`, unlike
  the stricter AdaFace loading path. This needs review before relying on it.
- No transformer result directory, calibration grid, paired prediction records
  or corresponding tests were found. Parameter constants and normalized vector
  norms are not recognition-accuracy results.

Keep FaceLiVT out of completed-result tables until a repaired, development-only
comparison produces verifiable evidence. This review does not establish that
the downloaded checkpoint cannot work; it establishes that the current files do
not demonstrate a completed research comparison.

Other stale labels should also be corrected before submission: `models.yaml`
still calls AdaFace future work and names ByteTrack, while AdaFace has a completed
development study and the implemented tracker is SORT-like. The generic
`adapter.enabled: true` does not make the adapter part of the evaluated app.
The original aggregator's comment says “softmax-like,” but its implemented
weights are linear normalized quality scores. The ChokePoint helper implements
the separate norm-softmax experiment. Treat executed code and verified run
settings as evidence rather than these old labels.

## 6. Problems in the old report to resolve before reusing it

These are document inconsistencies, not allegations about how the old work was
performed. Original logs/code are needed to resolve them reliably.

| Location | Observation | Required treatment |
|---|---|---|
| Abstract, PDF p. 2; organization p. 11 | Calls the work conceptual FYDP 1 and says implementation belongs to later phases; Chapters 3–4 describe implemented systems/results | Rewrite abstract and tense for the actual current submission stage |
| PCA, pp. 29 and 37 versus p. 45 | 50-component accuracy is 51.55% in early tables, 54.65% in Table 4.5 | Do not select whichever number looks better; recover the original experiment if retaining this history |
| CNN, pp. 29, 38, 45 and 51 | Narratives mix 100% training and 82–86% validation with Table 4.6's 99.42% peak training, 91.09% peak validation and **46.12% final test** | Training/validation/test are different measures; the current study has no equivalent new CNN training curve |
| Discussion pp. 44–45 | Broad CNN-superiority statement despite Table 4.6's 46.12% test versus Table 4.5's 54.65% PCA result | A blanket conclusion does not follow from those listed test values |
| Figure 4.4, PDF p. 39 | It is a 2×2 known/unknown matrix, not a seven-class identity confusion matrix. Visible cells are 226, 32 / 400, 0; with its displayed labels, 400 is in the unknown→known cell | Figure conflicts with the zero-false-acceptance narrative and Table 4.7; inspect raw predictions before reusing it |
| Table 4.7, PDF p. 46 | Listed TP=226, FN=32, TN=400, FP=0 imply binary accuracy `(226+400)/658 = 95.14%`, not 96.23% | Correct only after confirming the underlying counts |
| Same table | Listed binary precision is 0.36, but the stated TP/FP imply `226/(226+0) = 1.00`; recall `226/258 = 0.876` does round to 0.88 | Definitions and denominators must be explicit |
| Discussion p. 44 | Calls 400 Olivetti images “400 entirely unseen individuals” | Images and unique people are different units; verify the actual identity count |
| Preprocessing p. 37 | Includes an unfilled insertion placeholder; figure numbering is reused for CNN architecture | Replace with current pipeline figures and regenerate numbering |
| Literature pp. 16, 20–21 | Unresolved `[?]` citations and Unknown rows; bibliographic entries need normalization | Audit references individually before reuse; this comparison did not revalidate all cited papers |
| Standards/cost pp. 48–49 | Access protection, energy efficiency and total zero-cost claims lack corresponding current measured evidence | Separate design intentions, implemented controls and measured costs |

The previous report's 91.09% validation, 46.12% identity test, reported 96.23%
binary accuracy and current 94.38% correct identification-and-acceptance are
different quantities. **Do not write “accuracy improved from X to 94.38%” across
these studies.** Datasets, identities, preprocessing, models, splits and outcome
definitions changed. The valid numerical improvement claim is the paired
one-photo versus few-photo comparison within the current frozen experiment.

## 7. Progress against the original six objectives

The motivation still aligns with the old objectives on PDF p. 9, but not every
deployment objective has been completed.

| Original objective | Current position |
|---|---|
| Surveillance-realistic conditions | Partly addressed by masked originals, controlled stress tests and sequence-separated crops; no campus height/distance/crowd-density study |
| Benchmark-to-CCTV gap | Multiple public data sources exist, but tasks/protocols differ; this is not a matched same-population measurement of a benchmark-to-campus gap |
| Continuous complete analytics pipeline | Modules/live pages exist; strongest validated evidence is still-image workflow and separate crop aggregation; no full detector–tracker–logging deployment benchmark |
| Edge-first and accuracy-first configurations | ArcFace/AdaFace compared; FaceLiVT work started. No validated lightweight edge deployment or controlled edge-versus-server comparison |
| FPS, latency, memory and utilization | Image and component CPU latency measured; camera FPS, memory footprint and CPU/GPU utilization under realistic stream load remain unmeasured |
| Privacy and operational safeguards | Local data separation, hidden photo previews and pending identity approval exist; institutional permissions, authentication, encryption, retention policy and operational deployment controls are not demonstrated |

The old future-work item “use ArcFace” is now implemented and evaluated. AdaFace
is an additional completed development comparison. “Vision Transformers” has
moved to an unfinished integration. Multi-camera, anti-spoofing, cloud and mobile
items remain future work. Larger public identity coverage has been added, while
large-scale local deployment and fairness evaluation remain open.

## 8. How to start the new paperwork

Use [THESIS_DRAFT.md](THESIS_DRAFT.md) as the current content source and the old
report only as a formatting/chapter scaffold. The draft already integrates the
main result, AdaFace, enrollment extensions, temporal supplement and runtime.
It is not yet an institution-formatted final thesis.

| Existing chapter | Keep | Rewrite/add |
|---|---|---|
| Front matter and abstract | Institution's required structure; confirm author/supervisor information | Actual submission stage, present methods, main fractions and limited scope; remove FYDP1 conceptual-only wording |
| 1. Introduction | Educational motivation, difficult imaging conditions, need to reject unknowns | Few-photo enrollment/open-set research questions; distinguish campus motivation from evaluated data; revise objectives to match evidence |
| 2. Background | Relevant verified recognition, surveillance and governance context | ArcFace/AdaFace/SCRFD, template pooling, open-set identification, calibration and domain shift; repair all unresolved citations; FaceLiVT as ongoing work if mentioned |
| 3. Project design/methodology | Modular-system rationale | Dataset integrity and role diagrams; actual image pipeline; mean template; exact search; four-state rule; calibration grids; failure accounting; file persistence; implemented app architecture |
| 4. Implementation and results | Broad implementation/results organization | Actual environment; frozen final table first; separate development ablations; exploratory temporal results; runtime; demonstrated UI; limitations beside each result |
| 5. Standards/design constraints | Budget, resource and privacy concerns as design considerations | Actual inference setup and timing boundaries; data handling; dependency/model terms; measured costs if collected; avoid claiming unimplemented access controls/compliance |
| 6. Conclusion/future work | Motivation and unresolved deployment challenges | Evidence-based contributions and negative findings; FaceLiVT completion, independent evaluation, representative live cameras, real adapter learning, anti-spoofing and institutional backend as future work |
| Appendices | Required institutional forms | Configuration/source index, dataset exclusion summary, reproduction commands, verifier receipts, additional results and demo instructions |

Recommended writing order:

1. Freeze the scope statement and write Chapter 3 from the implemented method.
2. Write Chapter 4 using the final evidence pack first; label subsequent studies
   development-only or exploratory. Preserve exact denominators.
3. Rewrite objectives, contribution statement, conclusion and abstract to match.
4. Update the literature and all architecture diagrams/screenshots.
5. Apply institutional formatting, references, captions and supervisor revisions.

Suggested scope paragraph:

> This project investigates few-photo enrollment and open-set face identification
> using frozen pretrained recognition models. It evaluates enrollment and
> rejection choices on masked public face images, examines controlled degradation
> and temporal aggregation in a separate supplementary study, and implements a
> persistent gallery prototype. Its contribution is reproducible experimental
> evidence and system integration; operational campus deployment is future work.

Defensible contribution statements are the paired enrollment comparison,
explicit rejection/ambiguity/failure accounting, checkpoint and pooling ablations,
negative/mixed synthetic-enrollment evidence, controlled temporal findings and
measured gallery/runtime behavior. A new backbone, new training loss, universally
superior quality weighting or demonstrated campus security improvement has not
been established. See [the contribution assessment](NOVELTY_AUDIT.md).

## 9. Verification performed for this comparison

- Full software suite: **104 passed**. Two upstream warnings and a pytest cache
  permission warning occurred; the process exited successfully. Tests do not
  cover the new FaceLiVT comparison and do not prove live camera performance.
- Final verifier rerun: **1,133 prediction attempts, 1,126 searches, 585 original
  image hashes, no development identity overlap, frozen settings intact**.
- Backbone verifier rerun: **2,832 attempts, 2,776 searches, 19,072 calibration
  grid entries; 571 shared source images; ArcFace cache difference zero**.
- Existing ChokePoint and enrollment-extension verification receipts were
  inspected. Their prior real-inference replay counts are historical checks,
  not new inference performed by this review.
- Static source parsing found the two new scratch-script syntax errors described
  above. Passing `pytest` is compatible with these errors because `pytest.ini`
  limits collection to `tests/`.

Detailed source hashes, inventory and report extraction are preserved in
`F:/Thesis Final/audit_work/previous_vs_current/`. No new held-out image inference
or model training was performed for this review.
