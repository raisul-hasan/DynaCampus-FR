# Dataset preparation - 20 October 2026

**Current stage:** the frozen held-out evaluation is complete. Primary halfFace
correct known acceptance is 151/160 (94.38%), with 0/151 observed false known
acceptances. See `paper_assets/heldout/final_v1/README.md` for final figures and
limits. Earlier sections are a chronological record; statements that the reserved
test was untouched describe the time before this final evaluation.

## Ready now

Application evidence is also complete for the still-image enrollment flow;
see `paper_assets/development/demo_v1/README.md`. This uses development examples
and leaves the frozen held-out evaluation unchanged.

The official ICFD public sample is downloaded and inventoried:

| Item | Verified count |
|---|---:|
| People | 10 |
| Enrollment/gallery images | 200 |
| Classroom/probe images | 200 |
| Total images | 400 |
| Unique SHA-256 image hashes | 400 |
| Identical images shared between gallery and probe | 0 |

Each person has 20 gallery and 20 probe images. Gallery widths range from 189 to
457 pixels; probe widths range from 28 to 136 pixels. Original images and source
roles are preserved. This is the sample, not the full 84-person ICFD release.
These checks do not establish recognition accuracy or rule out near-duplicate frames.

Files:

- Original archive: `data/downloads/ICFD_Samples.zip`
- Original extracted images: `data/raw/icfd_sample/ICFD_Samples/`
- Image manifest: `data/manifests/icfd_sample.csv`
- Per-person counts: `results/tables/data_readiness/icfd_sample_counts.csv`
- Summary: `results/tables/data_readiness/icfd_sample_inventory.json`
- Unsigned agreement: `data/downloads/ICFD_ACCESS_AGREEMENT.pdf`

## halfFace: downloaded and checked

You downloaded the official archive manually after its host blocked automated
requests with a browser check. The archive's internal CRC checks pass, and every
original full-face image decodes successfully.

| Collection | Images |
|---|---:|
| Dataset-1: original full-face images | 941 |
| Dataset-2: derived upper-face images | 933 |
| Dataset-3: upper-face/augmented collection | 3,741 |

Dataset-1 contains 166 source person IDs. Only this original collection is extracted
for the experiment; the other collections remain archived.

Six image pairs are byte-for-byte identical under source IDs p039 and p090. Both
IDs (12 files) are excluded from the usable manifest rather than guessing the
correct identity. Three additional filenames have ambiguous condition labels and
are excluded from condition-based experiments. No original data is deleted.

**Usable collection: 926 images from 164 source identities.** This does not rule
out near-duplicate images or other identity-label issues; final protocol checks
still apply.

Files:

- Original images: `data/raw/halfface/full_face/`
- Complete manifest: `data/manifests/halfface_full_face.csv`
- Usable manifest: `data/manifests/halfface_usable.csv`
- Exclusions: `data/manifests/halfface_exclusions.csv`
- Duplicate groups: `data/manifests/halfface_duplicates.csv`
- Ambiguous labels: `data/manifests/halfface_condition_review.csv`
- Summary: `results/tables/data_readiness/halfface_inventory.json`

The original collection provides at most three documented unmasked images per
person. One-shot versus up-to-three-shot unmasked enrollment is feasible here;
five-shot unmasked enrollment is not. ICFD can support the five-shot comparison.
Do not combine derived or augmented versions with originals as independent test
evidence. Keep the datasets' experiments and identities separately namespaced.

Source: https://data.mendeley.com/datasets/pk44mkx9vm/2
DOI: 10.17632/pk44mkx9vm.2. Source naming instructions are saved under `data/sources/`.

## Full ICFD access

Official page: https://sites.google.com/view/icfd

The downloaded agreement requires a signatory authorized to sign on behalf of the
institution and an office seal. It says agreements signed by someone without that
authority will be discarded. Contact your department's authorized representative;
the document does not limit this role to your supervisor.

The agreement says to send the signed scan to `sdas@iitm.ac.in`. Nothing has been
signed or sent. Full access is pending; no approval or response time is assumed.

Until access is resolved, the 400-image public sample is available for engineering
checks. Any paper use must retain the authors' citation and follow their terms.
Keep restricted data private; `data/` is already ignored by Git.

## Reproduce this step

```powershell
python scripts/fetch_research_data.py --dataset icfd-sample
python scripts/fetch_research_data.py --dataset icfd-agreement
python scripts/prepare_dataset_inventory.py
python scripts/prepare_halfface_inventory.py
```

## Step 2 completed: real ArcFace development baseline

- Real local SCRFD and ArcFace checkpoints load without automatic downloads.
- Missing or failed recognition models raise errors; the dashboard cannot silently
  replace them with synthetic vectors.
- Development/test identities are frozen in `configs/splits/arcface_baseline_v1.json`.
- halfFace development: 55 enrolled identities and 27 unenrolled identities.
- halfFace reserved test: 55 enrolled and 27 unenrolled identities, untouched.
- ICFD engineering run: 5 sample identities; the other 5 are reserved.
- Source face crops receive a fixed 50% neutral border before detection; the setting
  was established using development images and is recorded in the configuration.
- All 561 attempted images are recorded; 554 yield real normalized embeddings.

The halfFace development rank-1 result is 243/251 (96.81%) including detection
failures. At its development-calibrated similarity threshold, correct known
acceptance is 225/251 (89.64%), with one false known acceptance among 149
successfully embedded unknown probes. These are development results, not final
held-out performance.

Report: `results/baselines/arcface_development_20261019T221350351017Z/BASELINE_REPORT.md`

Raw predictions, extraction outcomes, threshold, configuration snapshot, model
hashes, dependency versions and score plot are saved beside the report. Embeddings
stay in ignored `data/embeddings/`.

```powershell
python -m unittest discover -s tests -p test_embedder_safety.py -v
python -m unittest discover -s tests -p test_baseline_protocol.py -v
python scripts/run_arcface_baseline.py
# Use the newly printed run directory to regenerate its readable report:
python scripts/summarize_arcface_baseline.py results/baselines/arcface_development_20261019T221350351017Z
```

Ten targeted safety/protocol tests pass. The existing pytest suite has not been
run because pytest is unavailable in this runtime. The dashboard changes have
syntax validation; no interactive UI validation is claimed. Full ICFD access is
still pending. No model training or final-test evaluation has been performed.

## Step 3 completed: enrollment comparison and paper evidence

The same 154 masked halfFace known probes were compared with one frontal
enrollment photo versus up to three unmasked enrollment views. Extra enrollment
images are excluded from both methods' probe sets. Unknown probes are unchanged.

| Method | Correct top match | Correct acceptance after calibrated threshold |
|---|---:|---:|
| One photo | 146/154 (94.81%) | 129/154 (83.77%) |
| Up to three photos | 146/154 (94.81%) | 138/154 (89.61%) |

Both methods falsely accept one of 149 successfully embedded unknown probes.
Few-photo enrollment helps 10 known probes and hurts one, for a net gain of nine.
The known probe set here is masked-only; do not directly compare its rates with
the earlier 251-probe mixed-mask baseline. Six detection failures remain included.

The ICFD engineering sample uses the same 100 probes with one or three enrollment
images per person; both methods retrieve all 100 correctly. This is a small sample
ceiling result, not a full-dataset benchmark.

Paper figures, captions, CSV/LaTeX tables and source records are indexed in
`paper_assets/README.md`. Five enrollment safety tests pass. Reserved test data
remain untouched; no model has been retrained.

## Step 4 completed: score, margin and quality decisions

The same few-photo gallery and development probes were reused without new model
inference. Three rule families were calibrated with an empirical unknown FPIR
limit of 1% and at least 85% decision coverage for each extracted probe group.

| Rule family | Correct known acceptance | False known / extracted unknown | Extracted-probe decision coverage |
|---|---:|---:|---:|
| Score only | 138/154 (89.61%) | 1/149 | 297/297 (100%) |
| Score + margin | 139/154 (90.26%) | 1/149 | 295/297 (99.33%) |
| Score + margin + quality option | 139/154 (90.26%) | 1/149 | 295/297 (99.33%) |

The selected margin is 0.15 and score threshold is 0.3778589368. The selected
quality threshold is **zero (disabled)**: this heuristic adds no benefit under
the development objective. Margin calibration recovers one known probe and
marks two unknown probes ambiguous. The paired gain is small (+0.65 percentage
points; descriptive fixed-threshold 95% interval 0.00 to 1.97).

The decision engine now uses the difference between the two best identity scores;
refusal states return no accepted identity. Dashboard search passes both scores.
Dashboard thresholds remain manual, and interactive UI validation is pending.

Eight new decision tests pass. Independent verification checks all 912 saved
probe/method attempts and all 42,316 grid entries for selection consistency.
The reserved test remains untouched. All results here reuse development data for
calibration and evaluation and are not final accuracy claims.

Paper diagrams, graphs, captions, numeric sources and LaTeX table:
`paper_assets/development/decision_v1/README.md`.
Raw run: `results/decisions/decisions_development_20261019T222927022126Z`.

```powershell
python scripts/compare_decisions.py
python scripts/verify_decision_run.py results/decisions/decisions_development_20261019T222927022126Z
python scripts/render_decision_evidence.py results/decisions/decisions_development_20261019T222927022126Z
python -m unittest discover -s tests -p test_decision_ablation.py -v
```

## Step 5 completed: controlled image-quality stress tests

The same 304 halfFace development probes were tested under three blur levels,
three lower-resolution sizes and three JPEG qualities. Each change was applied
separately before detection. Models, the original enrollment gallery and the
previously selected thresholds stayed fixed. Original files were never modified.

Selected score-plus-margin results (the strongest tested setting of each type):

| Condition | Correct known / 154 attempts | Extracted / 304 attempts |
|---|---:|---:|
| Original control | 139 (90.26%) | 297 |
| Gaussian blur, sigma = 0.012 x source short edge | 132 (85.71%) | 299 |
| Long edge reduced to 32 pixels | 104 (67.53%) | 290 |
| JPEG quality 10 | 119 (77.27%) | 275 |

All ten conditions and both fixed decision rules are in the saved tables and
graphs. Some mild changes improve a few probes through altered detection and
alignment. The margin rule accepts more known probes in six transformed
conditions, the same number in two and fewer in one; no universal improvement
claim is supported. Under strong changes, lower false acceptance can occur
alongside more missed known probes, so report both outcomes.

This step adds 2,736 real-model image attempts, not new independent people.
The original control reuses saved real embeddings and exactly reproduces step 4.
Independent verification confirms 6,080 per-rule decisions and 2,960 successful
cosine searches across original and transformed conditions, fixed templates and
thresholds, unchanged hashes for all 304 original probes, and no reserved IDs.
Five new transform/protocol tests pass. No threshold tuning or training occurred.

These are synthetic development stress tests, not actual CCTV deployment results.
Conditional unknown error rates use condition-specific extraction denominators;
the attempted-unknown denominator is also saved. Identity-cluster bootstrap
intervals describe paired changes with fixed thresholds, not final-test accuracy.

Evidence and captions: `paper_assets/development/robustness_v1/README.md`.
Raw run: `results/robustness/robustness_development_20261019T230159679928Z`.
Reserved test identities remain untouched. The next evaluation step is to freeze
the method and run the reserved test with no test-driven threshold adjustment.

```powershell
python scripts/run_robustness.py
python scripts/verify_robustness_run.py results/robustness/robustness_development_20261019T230159679928Z
python scripts/render_robustness_evidence.py results/robustness/robustness_development_20261019T230159679928Z
python -m unittest discover -s tests -p test_robustness_protocol.py -v
```

## Step 6 completed: frozen held-out evaluation

Source code, model identifiers, input manifests, identity split, three comparison
methods and all thresholds were frozen before reading the reserved images.
The primary method was few-photo enrollment with score and margin decisions.
No test-dependent tuning, exclusion, training or method selection occurred.

The halfFace test uses 82 identities distinct from development: 55 enrolled and
27 unknown, with 160 masked known probes and 151 mixed-mask unknown probes.

| Frozen method | Correct known / 160 attempts | False known / 151 unknown attempts |
|---|---:|---:|
| One photo + score | 130 (81.25%) | 1 |
| Up to three photos + score | 150 (93.75%) | 0 |
| Up to three photos + score + margin (primary) | 151 (94.38%) | 0 |

All methods retrieve the correct top identity for 159/160 known attempts before
thresholding; one capture failure stays included. No wrong known identity is
accepted. All enrolled identities have usable templates. Few-photo methods use
159 original enrollment images, compared with 55 for the one-photo method.

For the primary method, seven known probes are rejected as unknown, one is
ambiguous and one has a capture failure. All 151 unknown probes are rejected as
unknown. The known-identity bootstrap 95% interval is 88.89% to 98.74%. Margin
decisions help two known probes and hurt one (net +0.625 percentage points); its
paired interval is -1.28 to +2.55 points, so no established margin advantage is
claimed. The larger observed gain comes from few-photo enrollment.

Zero observed false accepts is a small-sample result, not a population risk
bound. Percentile bootstrap intervals at zero errors cannot account for unseen
error events. Source identity labels and the original split are retained; model
pretraining overlap, near duplicates and Bangladesh-wide generalization remain
unresolved limitations.

The five reserved ICFD public-sample identities provide 100 separate engineering
probes: both one-photo and three-photo retrieval score 98/100, including two
extraction failures. No unknown identities are present, so this is a closed-set
sample check and cannot establish open-set FPIR or full-dataset performance.

Seven new protocol tests pass. Independent cached verification confirms all
1,133 probe/method attempts, 1,126 successful cosine searches, unchanged hashes
for all 585 extracted originals, the pre-inference freeze and no development-ID
overlap. The image extraction ran once. The earlier metadata-only freeze attempt
was aborted before any image inference and is explicitly marked as such.

Raw final run: `results/final/heldout_20261019T231512869782Z`.
Final figures, captions, tables and paper text: `paper_assets/heldout/final_v1/README.md`.
The reserved test is now consumed; preserve these settings and results for
reporting. Further application validation should use development/demo data.

```powershell
# Audit/re-render the completed run using caches; no new image inference:
python scripts/verify_final_evaluation.py results/final/heldout_20261019T231512869782Z
python scripts/render_final_evidence.py results/final/heldout_20261019T231512869782Z
python -m unittest discover -s tests -p test_final_protocol.py -v
```

## Step 7 completed: dynamic gallery and application demonstration

- Inserted 55 development identities in stages 0, 10, 20, 30, 40 and 55. Existing
  template arrays and both real-model checkpoint hashes stayed unchanged.
- Independently verified 1,824 probe/stage decisions. All 304 full-gallery
  decisions match the earlier development margin result exactly.
- Verified on-disk reload, removal and reinsertion. Five actual development
  images produce the same embeddings and quality as the saved research cache.
- Corrected the app to use original-photo normalized means, full-precision
  scores, the four-state rule and frozen settings. Synthetic enrollment variants
  are no longer added automatically. Pending entries are excluded from active
  recognition until approved, and gallery counts refresh after mutations.
- Eight gallery regression tests pass. All eight application pages render with
  real models; the browser uploaded three photos, enrolled one example identity,
  recognized a separate masked probe, rejected an unknown probe, reloaded the
  saved gallery and removed the identity. The removed identity is then unknown.
- Browser photo previews are off in all five saved screenshots. The fixture is
  explicitly selected from known successful development examples, not a random
  accuracy sample. Its one-identity gallery bypasses the top-two margin check;
  multi-identity behavior is covered by the full-gallery comparison and tests.
- Local search-only timing at 55 identities has median 0.0631 ms and p95 0.0791 ms
  over 300 calls. This excludes detection, embedding, disk loading and UI and
  cannot be reported as end-to-end video throughput.

Raw run: `results/demo/dynamic_development_20261019T233918475526Z`.
Figures, source tables, captions, actual browser captures and paper text:
`paper_assets/development/demo_v1/README.md`.
No reserved test image was used in this stage. The frozen final-evaluation audit
still passes. Webcam/RTSP connections, temporal tracking, AdaFace and adapter
training remain outside the validated evidence.

The local test browser and server were stopped after capture. To run the app,
use `python -m streamlit run app.py` and select **Input is a tight face crop** for
dataset-style face crops. An isolated gallery can be selected through
`DYNACAMPUS_GALLERY_DIR`. Existing user gallery files are preserved.

Next: prepare the paper outline and evidence-backed claims, then draft sections
using the frozen results and the recorded limitations.
