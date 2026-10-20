"""Render development engineering figures and document real browser captures."""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.render_enrollment_evidence import arrow,box,save_figure
from scripts.run_arcface_baseline import read_csv,sha256

RUN=ROOT/'results/demo/dynamic_development_20261019T233918475526Z'
DEST=ROOT/'paper_assets/development/demo_v1'
DEST.mkdir(parents=True,exist_ok=True)
info=json.loads((RUN/'comparison.json').read_text())
verification=json.loads((RUN/'verification.json').read_text())
pages=json.loads((RUN/'app_page_checks.json').read_text())
summaries=info['summaries']
fig,ax=plt.subplots(figsize=(7.8,4.7))
ax.set(xlim=(0,7.8),ylim=(0,4.7))
ax.axis('off')
ax.text(3.9,4.43,'Training-free dynamic enrollment: implemented workflow',ha='center',fontsize=12)
xs=[1.,2.93,4.86,6.8]
for x,text in zip(xs,['Original enrollment\nphotos (up to 3)','SCRFD + alignment\nFrozen ArcFace','Normalize mean\nof real features','Insert identity\ninto disk gallery']):
    box(ax,x,3.19,text,width=1.7,height=.88)
for left,right in zip(xs[:-1],xs[1:]):
    arrow(ax,(left+.9,3.19),(right-.9,3.19))
box(ax,2.,1.64,'New probe → same frozen feature extractor',width=3.6,height=.67,fill='#F5F5F3')
box(ax,6.,1.64,'Exact cosine search + fixed decisions',width=3.1,height=.67,fill='#F5F5F3')
arrow(ax,(3.87,1.64),(4.37,1.64))
arrow(ax,(6.8,2.7),(6.,2.03))
ax.text(3.9,.69,'55 identity insertions; existing templates preserved; checkpoint hashes unchanged.',ha='center',fontsize=9.3)
ax.text(3.9,.27,'Reload and removal verified. Development engineering evidence; no new held-out tuning.',ha='center',fontsize=9)
fig.tight_layout(pad=.3)
save_figure(fig,DEST,'fig00_dynamic_workflow')

x=[r['gallery_identities'] for r in summaries]
fig,ax=plt.subplots(figsize=(7.1,4.7))
ax.plot(x,[r['active_known_attempts'] for r in summaries],'--o',color='#677D68',label='Probes whose identity is enrolled')
ax.plot(x,[r['correct_known_accepts'] for r in summaries],'-s',color='#2878A0',label='Correct known acceptances')
for r in summaries:
    if not r['gallery_identities']:
        continue
    label=f'{r["correct_known_accepts"]}/{r["active_known_attempts"]}'
    ax.annotate(label,(r['gallery_identities'],r['correct_known_accepts']),
        xytext=(0,-17),textcoords='offset points',ha='center',fontsize=9)
ax.set(xlabel='Number of identities inserted into the gallery',ylabel='Probe count',ylim=(-9,170),xticks=x,
       title='Gallery growth without recognizer training')
ax.grid(alpha=.2)
ax.spines[['top','right']].set_visible(False)
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.10),ncol=1,frameon=False,fontsize=9)
fig.text(.5,.02,'Same 304 development probes at each stage. The enrolled-known denominator grows with the gallery.\nAnnotations are accepted / currently enrolled-known attempts; this is not an accuracy trend.',ha='center',fontsize=8.5)
fig.subplots_adjust(left=.11,right=.97,top=.88,bottom=.32)
save_figure(fig,DEST,'fig01_dynamic_growth')

fig,ax=plt.subplots(figsize=(7.1,4.6))
ax.plot(x,[r['search_median_ms'] for r in summaries],'-o',color='#2878A0',label='Median')
ax.plot(x,[r['search_p95_ms'] for r in summaries],'--s',color='#D18432',label='95th percentile')
ax.set(xlabel='Active gallery identities',ylabel='Gallery.search call (ms)',xticks=x,ylim=(0,None),
       title='Local search latency: 300 calls per gallery size')
ax.grid(alpha=.2)
ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False)
fig.text(.5,.025,'Includes matrix construction and exact cosine matching; excludes detection, features, disk loading and UI.\nZero identities returns immediately. These timings do not measure end-to-end video FPS.',ha='center',fontsize=8.5)
fig.subplots_adjust(left=.13,right=.97,top=.88,bottom=.25)
save_figure(fig,DEST,'fig02_search_latency')

for name in ['stage_summary.csv','insertions.csv','image_equivalence.csv']:
    shutil.copyfile(RUN/name,DEST/name)
last=summaries[-1]
insertion_ms=[float(r['milliseconds']) for r in read_csv(RUN/'insertions.csv')]
table=['| Active identities | Enrolled-known attempts | Correct known accepts | Not-yet/permanently unenrolled attempts | False known among unenrolled |',
       '|---:|---:|---:|---:|---:|']
for r in summaries:
    table.append(f'| {r["gallery_identities"]} | {r["active_known_attempts"]} | {r["correct_known_accepts"]} | {r["not_enrolled_attempts"]} | {r["false_known_not_enrolled"]} |')
report=f'''# Dynamic enrollment and application evidence

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
  {len(pages['pages'])} Streamlit pages render without exceptions, including manual/frozen presets.
  Camera connections and real video tracking are not started or validated.

## Engineering stage counts

{chr(10).join(table)}

The same 304 source probes are reused at every stage. A person who has not yet
been inserted is unenrolled at that stage. This differs from the fixed known/
unknown labels of the recognition benchmark; growing denominators must not be
interpreted as improving model accuracy. Capture failures remain in attempted
counts. The final stage has 154 enrolled-known and 150 unknown attempts.

## Timing scope

At 55 identities, the local search median is **{last['search_median_ms']:.4f} ms** and
the 95th percentile is **{last['search_p95_ms']:.4f} ms**, from 300 repeated calls
following 20 warm-up calls. The class rebuilds the small identity matrix on each
search. It uses exact NumPy cosine matching. Template insertion has median
**{np.median(insertion_ms):.3f} ms** over 55 operations, including disk persistence,
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

Raw run: `{RUN.relative_to(ROOT).as_posix()}`. Numeric sources:
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
'''
(DEST/'README.md').write_text(report,encoding='utf-8')
manifest=dict(source_run=RUN.relative_to(ROOT).as_posix(),scope='development_engineering_only',
    source_sha256={name:sha256(RUN/name) for name in ['comparison.json','verification.json','app_page_checks.json','browser_events.jsonl','cpu_environment.json']},
    renderer_sha256=sha256(__file__),assets_sha256={p.name:sha256(p) for p in DEST.iterdir() if p.is_file() and p.name!='asset_manifest.json'})
(DEST/'asset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('PAPER ASSETS',DEST)
