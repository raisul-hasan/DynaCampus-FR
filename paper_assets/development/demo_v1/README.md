# Dynamic enrollment and application evidence

**Verified:** 55 identity insertions without backbone changes, preservation of
existing templates, exact reload, removal/reinsertion, agreement with all 304
saved full-gallery development decisions, and a real browser upload flow.
This is development engineering evidence. The final held-out results are unchanged.

## What was checked

- Add identities in stages 0, 10, 20, 30, 40 and 55, using the original few-photo
  candidates and real cached ArcFace features. Existing template arrays are
  checked after every insertion; the detector/recognizer checkpoint hashes are
  checked before and after. No training function is called.
- All 1,824 probe/stage decisions are independently recomputed. At 55 identities,
  all 304 decisions and successful-search scores match the earlier development
  margin run. This reproduces a prior result rather than adding independent data.
- Reload the on-disk 55-identity gallery with exact template/search equality;
  remove an identity, verify the saved size is 54, and insert its template again.
- Process five actual development images through the shared app path, checking
  their real features and quality against the existing cache (max feature
  difference below 1e-6). These include three enrollment photos and known/unknown
  probes. The face-crop mode applies the same fixed neutral padding.
- Browser: upload three original photos, confirm enrollment, recognize a separate
  masked photo, reject an unknown photo, reload the application and remove the
  identity. The previously recognized probe then returns unknown with an empty
  gallery. The uploaded gallery template exactly equals the cached original-photo
  mean; displayed known/unknown scores match independent reference searches.
- Eight gallery regression tests pass, including invalid vectors, exact score
  precision, near-tie ambiguity, reload/removal and pending approval. All
  8 Streamlit pages render without exceptions, including manual/frozen presets.
  Camera connections and real video tracking are not started or validated.

## Engineering stage counts

| Active identities | Enrolled-known attempts | Correct known accepts | Not-yet/permanently unenrolled attempts | False known among unenrolled |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 304 | 0 |
| 10 | 24 | 23 | 280 | 0 |
| 20 | 50 | 47 | 254 | 0 |
| 30 | 80 | 72 | 224 | 1 |
| 40 | 110 | 100 | 194 | 1 |
| 55 | 154 | 139 | 150 | 1 |

The same 304 source probes are reused at every stage. A person who has not yet
been inserted is unenrolled at that stage. This differs from the fixed known/
unknown labels of the recognition benchmark; growing denominators must not be
interpreted as improving model accuracy. Capture failures remain in attempted
counts. The final stage has 154 enrolled-known and 150 unknown attempts.

## Timing scope

At 55 identities, the local search median is **0.0631 ms** and
the 95th percentile is **0.0791 ms**, from 300 repeated calls
following 20 warm-up calls. The class rebuilds the small identity matrix on each
search. It uses exact NumPy cosine matching. Template insertion has median
**1.044 ms** over 55 operations, including disk persistence,
using cached features. Insertion timing excludes image decoding, detection and
feature extraction; it is not total photo-to-enrollment time.

These are local component measurements under ordinary background load, without
thread pinning or a controlled hardware benchmark. Environment details stay in
the raw run; the optional CPU model query was unavailable. Do not invert search
time into a system FPS claim. No end-to-end camera throughput is established.

## Application corrections

- Keep raw score precision for threshold decisions and report the actual NumPy
  backend. The former code rounded scores and always labelled search as FAISS.
- Use original-photo, unweighted normalized means for the evaluated enrollment
  path. Automatic synthetic enrollment variants have been removed from this path.
- Use the same four-state rule and frozen score/margin settings in still-image
  recognition; show an accepted name only for a known decision. Manual settings
  remain available for demonstrations and do not change paper results.
- Require exactly one detected face per enrollment photo and at most three
  photos. Offer tight-face-crop preprocessing explicitly for dataset-style crops.
- Keep automatically queued entries outside active search until approval; approved
  entries leave the pending list and retain approval when updated. Automatic
  capture is opt-in. The live capture mechanism itself is not a validated study.
- Update gallery counts immediately after changes. Photo previews can be turned
  off; the saved browser screenshots contain no participant portraits.
- Use an isolated gallery under ignored `data/demo/` during checks. The normal
  user gallery and original datasets are not used as mutable test fixtures.

The offline held-out evaluator does not use the app/gallery module. Its frozen
source hashes and saved predictions were re-audited successfully after these
changes, without new held-out image inference or parameter changes.

## Figures and screenshots: captions

1. **[Dynamic workflow](fig00_dynamic_workflow.png)** ([SVG](fig00_dynamic_workflow.svg)).
   Real features from original enrollment photos are averaged and normalized,
   then inserted into a persistent exact-search gallery. The recognizer checkpoint
   remains fixed; enrollment still requires feature extraction, rather than training.
2. **[Gallery growth](fig01_dynamic_growth.png)** ([SVG](fig01_dynamic_growth.svg)).
   Currently enrolled-known attempts and correct known acceptances as development
   identities are inserted. The same source probes are reused and the denominator
   grows with enrollment. The figure demonstrates gallery operations, not a new
   recognition-accuracy improvement.
3. **[Search latency](fig02_search_latency.png)** ([SVG](fig02_search_latency.svg)).
   Median and 95th-percentile gallery search-call latency over 300 repetitions at
   each size, including matrix construction. These are local component timings
   and exclude real-model inference and frame processing.
4. **[Enrollment screen](fig03_app_enrollment.png)**. Actual local browser confirmation
   after three original development photos are processed and an identity template
   is inserted. Photo previews are disabled and a pseudonymous display name is used.
5. **[Known probe](fig04_app_known.png)**. The separate masked development image is
   accepted as the enrolled demonstration identity, with cosine score 0.4977.
6. **[Unknown probe](fig05_app_unknown.png)**. A different development identity is
   rejected as unknown, score 0.0208, with no accepted identity name.
7. **[Persisted gallery](fig06_app_persistence.png)**. The enrolled identity remains
   available after reloading the application. On-disk template equality is checked.
8. **[After removal](fig07_app_after_removal.png)**. Removing the demonstration
   identity leaves an empty gallery; the formerly accepted probe returns unknown.

Browser fixtures were selected from previously successful development examples
to demonstrate the flow. They are not a random accuracy sample. The screenshot
gallery has one active identity, so the two-candidate margin test is bypassed;
the displayed single-candidate gap is a UI convention. Multi-identity margin
behavior is checked by regression tests and full-gallery development agreement.

Graphs/diagrams are 300 dpi PNG plus editable SVG. Screenshots are genuine native
browser PNG captures, not generated mockups; photo visibility is disabled in the
actual app. These screenshots are illustrative engineering evidence only.

## Suggested paper text

Dynamic enrollment was verified by inserting 55 development identities into a
persistent exact-cosine gallery while retaining the pretrained recognition
checkpoints. Existing prototypes were unchanged after each insertion, and all
304 full-gallery decisions matched the saved development reference. Reload and
removal operations were verified, and the browser demonstrated original-photo
enrollment, known/unknown recognition and persistence with a disclosed development
fixture. These checks establish prototype functionality rather than live campus
deployment or tracking performance. Search-only timings are reported separately
from model inference and video throughput.

## Reproduction and evidence

Raw run: `results/demo/dynamic_development_20261019T233918475526Z`. Numeric sources:
[stage counts](stage_summary.csv), [insertions](insertions.csv),
[real-image equivalence](image_equivalence.csv), [asset/source hashes](asset_manifest.json).
Raw evidence also includes individual timings, all 1,824 decisions, browser
observations, page checks and independent verification. Embeddings and thumbnails
remain under ignored `data/`.

```powershell
python -m unittest discover -s tests -p test_dynamic_gallery.py -v
python scripts/validate_dynamic_enrollment.py
python scripts/check_app_pages.py
python scripts/verify_demo_evidence.py
python scripts/render_demo_evidence.py
```

The UI audit uses the named isolated browser gallery and the saved browser events.
Reproducing the browser flow requires local Chrome/Edge and a fresh isolated
gallery. For a normal local demo, run `python -m streamlit run app.py`; select
**Input is a tight face crop** for these dataset crops and use the frozen preset.
Existing galleries are preserved and are not automatically rebuilt.

This stage used the [computer-use skill](C:/Users/msi/.codex/plugins/cache/openai-bundled/computer-use/26.1013.40934/skills/computer-use/SKILL.md)
for native capability inspection. Its native connection was unavailable; actual
browser checks used local headless Chrome's protocol, and page smoke checks used
Streamlit AppTest. No physical camera or RTSP source was connected.
