"""Render held-out paper figures from frozen evaluation artifacts, without inference."""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.render_enrollment_evidence import arrow,box,save_figure
from scripts.run_arcface_baseline import read_csv,sha256,write_csv

LABELS = {'one_photo_score':'One photo\n+ score', 'few_photo_score':'Up to three photos\n+ score',
          'few_photo_margin':'Up to three photos\n+ score + margin'}
COLORS = ['#2878A0','#677D68','#D18432']


def freeze_diagram(destination,info,freeze):
    primary = next(r for r in info['summaries'] if r['primary'])
    fig,ax = plt.subplots(figsize=(8.3,5.0))
    ax.set(xlim=(0,8.3),ylim=(0,5.0))
    ax.axis('off')
    ax.text(4.15,4.71,'Identity-disjoint held-out evaluation',ha='center',fontsize=13)
    box(ax,1.4,3.53,'Development: 82 IDs\n55 known + 27 unknown\nCalibration and method choice',width=2.35,height=1.0)
    box(ax,4.15,3.53,'Freeze settings + hashes\nBefore test-image inference\nNo test-driven changes',width=2.35,height=1.0,fill='#E4EFE6')
    box(ax,6.9,3.53,'Held out: 82 different IDs\n55 known + 27 unknown\nFixed settings evaluated once',width=2.35,height=1.0)
    arrow(ax,(2.64,3.53),(2.91,3.53))
    arrow(ax,(5.39,3.53),(5.66,3.53))
    box(ax,6.9,1.81,f'{primary["gallery_images"]} enrollment images\n{primary["known_probe_attempts"]} masked known probes\n{primary["unknown_probe_attempts"]} mixed-mask unknown probes',width=2.35,height=.96,fill='#F5F5F3')
    arrow(ax,(6.9,2.98),(6.9,2.34))
    box(ax,2.72,1.81,'Same SCRFD, alignment and frozen ArcFace\nSame few-photo template construction\nPrimary: score 0.3779, margin 0.15, quality gate off',width=4.83,height=.96,fill='#F5F5F3')
    ax.text(4.15,.70,'ICFD public sample: five other reserved IDs, 100 probes; closed-set engineering check only.',ha='center',fontsize=9)
    ax.text(4.15,.28,'Within-dataset identity separation does not rule out overlap with model pretraining data.',ha='center',fontsize=9,color='#59636C')
    fig.tight_layout(pad=.35)
    save_figure(fig,destination,'fig00_heldout_protocol')


def comparison_plot(destination,summaries):
    fig,axes = plt.subplots(1,2,figsize=(9,5.1))
    x = np.arange(len(summaries))
    values = np.array([100*r['correct_known_accept_end_to_end'] for r in summaries])
    low = np.array([100*r['known_accept_ci95_low'] for r in summaries])
    high = np.array([100*r['known_accept_ci95_high'] for r in summaries])
    axes[0].bar(x,values,color=COLORS,width=.64,yerr=np.array([values-low,high-values]),capsize=4,error_kw={'linewidth':1})
    for index,(r,value,upper) in enumerate(zip(summaries,values,high)):
        axes[0].text(index,upper+1.8,f'{value:.2f}%\n{r["correct_known_accepts"]}/{r["known_probe_attempts"]}',ha='center',fontsize=9)
    axes[0].set(ylabel='Correct known acceptance (%)',ylim=(0,116),title='Known probes: all 160 attempts')
    axes[0].set_yticks(range(0,101,20))
    false = [r['false_known_unknowns'] for r in summaries]
    axes[1].bar(x,false,color=COLORS,width=.64)
    for index,r in enumerate(summaries):
        axes[1].text(index,r['false_known_unknowns']+.08,f'{r["false_known_unknowns"]}/{r["unknown_with_embedding"]}',ha='center',fontsize=10)
    axes[1].set(ylabel='False known count',ylim=(0,max(2,max(false)+1)),title='Unknown probes: 151 extracted / attempted')
    axes[1].set_yticks(range(max(2,max(false)+1)+1))
    for ax in axes:
        ax.set_xticks(x,[LABELS[r['method']] for r in summaries])
        ax.tick_params(axis='x',labelsize=8.5)
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('halfFace held-out results at frozen development thresholds',fontsize=13)
    fig.text(.5,.035,'Left: 95% descriptive known-identity bootstrap intervals; one capture failure stays in the denominator.\nRight: zero observed false accepts is a sample result, not a population error bound. Primary method: few-photo + margin.',ha='center',fontsize=8.7)
    fig.subplots_adjust(left=.08,right=.99,top=.83,bottom=.26,wspace=.30)
    save_figure(fig,destination,'fig01_heldout_comparison')


def states_plot(destination,rows,primary):
    selected = [r for r in rows if r['method'] == primary]
    states = ['KNOWN','UNKNOWN','AMBIGUOUS','INSUFFICIENT_QUALITY','CAPTURE_FAILURE']
    counts = np.array([[sum(r['truth_status'] == truth and r['decision'] == state for r in selected) for state in states] for truth in ['known','unknown']])
    rates = 100*counts/counts.sum(axis=1,keepdims=True)
    fig,ax = plt.subplots(figsize=(8,3.7))
    heatmap = ax.imshow(rates,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    for i in range(2):
        for j in range(5):
            ax.text(j,i,f'{counts[i,j]}\n{rates[i,j]:.2f}%',ha='center',va='center',fontsize=10,
                    color='white' if rates[i,j] > 60 else '#18242D')
    ax.set_xticks(range(5),['Known','Unknown','Ambiguous','Insufficient\nquality','Capture\nfailure'])
    ax.set_yticks([0,1],[f'Known probes\n(n={counts[0].sum()})',f'Unknown probes\n(n={counts[1].sum()})'])
    ax.set_title('Primary method: observed decision states on held-out probes',pad=14,fontsize=12)
    bar = fig.colorbar(heatmap,ax=ax,fraction=.03,pad=.025)
    bar.set_label('Within each truth group (%)',fontsize=9)
    fig.text(.5,.025,'Each source probe contributes once. Accepted known identities are correct; no wrong enrolled identity was accepted.\nQuality gate is disabled. Capture failure is separate from the four decision states.',ha='center',fontsize=8.7)
    fig.subplots_adjust(left=.15,right=.94,top=.77,bottom=.29)
    save_figure(fig,destination,'fig02_heldout_states')
    write_csv(destination/'decision_state_counts.csv',[dict(truth_status=truth,decision=state,count=int(counts[i,j]),group_total=int(counts[i].sum()),percentage=float(rates[i,j]))
              for i,truth in enumerate(['known','unknown']) for j,state in enumerate(states)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--output',default='paper_assets/heldout/final_v1')
    args = parser.parse_args()
    run,destination = ROOT/args.run,ROOT/args.output
    info = json.loads((run/'comparison.json').read_text())
    freeze = json.loads((run/'freeze.json').read_text())
    verification = json.loads((run/'verification.json').read_text())
    if not info['final_test_evaluated'] or info['thresholds_retuned'] or not verification['frozen_before_inference']:
        raise ValueError('Expected verified frozen held-out evaluation')
    destination.mkdir(parents=True,exist_ok=True)
    summaries = info['summaries']
    primary = next(r for r in summaries if r['primary'])
    freeze_diagram(destination,info,freeze)
    comparison_plot(destination,summaries)
    predictions = read_csv(run/'predictions.csv')
    states_plot(destination,predictions,info['primary_method'])
    primary_states = Counter((r['truth_status'],r['decision']) for r in predictions if r['method'] == info['primary_method'])
    score_known = {r['path']:r for r in predictions if r['method'] == 'few_photo_score' and r['truth_status'] == 'known'}
    margin_known = {r['path']:r for r in predictions if r['method'] == 'few_photo_margin' and r['truth_status'] == 'known'}
    helped = sum(score_known[p]['correct_known_accept'] == 'False' and margin_known[p]['correct_known_accept'] == 'True' for p in score_known)
    hurt = sum(score_known[p]['correct_known_accept'] == 'True' and margin_known[p]['correct_known_accept'] == 'False' for p in score_known)
    for name in ['summary.csv','icfd_summary.csv','paired_effects.csv']:
        shutil.copyfile(run/name,destination/name)
    table = ['| Method | Enrollment photos | Correct top match | Correct known acceptance | False known / extracted unknown |',
             '|---|---:|---:|---:|---:|']
    latex = [r'\begin{tabular}{lrrrr}',r'\hline',r'Method & Enrollment photos & Rank-1 /160 & Correct known /160 & False known /151 \\',r'\hline']
    for r in summaries:
        label = LABELS[r['method']].replace('\n',' ')
        table.append(f'| {label} | {r["gallery_images"]} | {r["rank1_correct"]}/{r["known_probe_attempts"]} ({100*r["rank1_end_to_end"]:.2f}%) | '
                     f'{r["correct_known_accepts"]}/{r["known_probe_attempts"]} ({100*r["correct_known_accept_end_to_end"]:.2f}%) | {r["false_known_unknowns"]}/{r["unknown_with_embedding"]} |')
        latex.append(f'{label} & {r["gallery_images"]} & {r["rank1_correct"]} & {r["correct_known_accepts"]} & {r["false_known_unknowns"]} ' + r'\\')
    latex += [r'\hline',r'\end{tabular}','% Same 55 enrolled identities and 27 unknown identities. Thresholds frozen from development.',
              '% Known probes: 160 masked originals; unknown probes: 151 mixed-mask originals. One known capture failure included.']
    (destination/'heldout_table.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    effects = info['paired_effects']
    uncertainty = ['| Paired change in correct known acceptance | Difference (percentage points) | Descriptive 95% interval |',
                   '|---|---:|---:|']
    for effect in effects:
        uncertainty.append(f'| {effect["comparison"]} | {effect["delta_percentage_points"]:+.2f} | {effect["ci95_low_percentage_points"]:+.2f} to {effect["ci95_high_percentage_points"]:+.2f} |')
    icfd_table = ['| ICFD sample method | Probe attempts | Successful extractions | Correct top match |',
                  '|---|---:|---:|---:|']
    for r in info['icfd_summaries']:
        icfd_table.append(f'| {r["strategy"]} | {r["probe_attempts"]} | {r["successful_extractions"]} | {r["rank1_correct"]}/{r["probe_attempts"]} ({100*r["rank1_end_to_end"]:.2f}%) |')
    report = f'''# Held-out evaluation: frozen final evidence

The primary method (up to three enrollment views + score + margin) correctly
accepts **{primary['correct_known_accepts']}/{primary['known_probe_attempts']} known probes ({100*primary['correct_known_accept_end_to_end']:.2f}%)**,
with **{primary['false_known_unknowns']}/{primary['unknown_with_embedding']} observed false known acceptances**.
This is the first reserved-image inference for this frozen protocol. Settings
were selected from development evidence and were not changed after this test.

## halfFace results

{chr(10).join(table)}

All three comparisons use the same 55 held-out enrolled identities, 27 held-out
unknown identities, 160 masked known probes and 151 mixed-mask unknown probes.
All unknown probes yield embeddings. One known probe has a capture failure and
stays in the denominator. No wrong enrolled identity is accepted by any method.
Every intended identity has a usable enrollment template; none is dropped.

The primary method returns 151 known decisions, 158 unknown decisions, one
ambiguous decision and one capture failure. Its 151 accepted known identities
are correct; {primary_states[('known','UNKNOWN')]} other known probes are rejected as unknown,
{primary_states[('known','AMBIGUOUS')]} is ambiguous and {primary_states[('known','CAPTURE_FAILURE')]} has a capture failure.
All {primary_states[('unknown','UNKNOWN')]} unknown probes are rejected as unknown. Decision coverage is
309/310 (99.68%) after successful extraction and 309/311 (99.36%) over all attempts.

The 95% known-identity bootstrap interval for primary correct known acceptance
is **{100*primary['known_accept_ci95_low']:.2f}% to {100*primary['known_accept_ci95_high']:.2f}%**.

{chr(10).join(uncertainty)}

Few-photo enrollment provides the larger observed acceptance gain. Adding the
margin rule to few-photo enrollment helps {helped} known probes and hurts {hurt}, for a net change of one probe;
its paired interval includes zero, so no established superiority claim is made.
The comparisons use separately frozen development-calibrated score thresholds;
they compare calibrated methods, rather than an isolated gate at one common score.
The few-photo margin method was designated primary before the test, regardless
of which method would obtain the highest test rate.

## Separate ICFD public-sample check

{chr(10).join(icfd_table)}

Five reserved sample identities are different from the five used in the earlier
engineering run. Two probe extractions fail and remain included. This is a
closed-set engineering check of the public sample: there are no unknown people
in these probes, so false-positive identification rate cannot be estimated.
It is not a result on the full ICFD dataset or a full CCTV benchmark.

## Frozen protocol and provenance

- Frozen at `{freeze['frozen_utc']}`, before the start receipt and first reserved
  image inference. The receipt includes hashes of source code, manifests, split,
  configuration, development results and evaluation selection.
- Models: unchanged local real SCRFD and ArcFace ONNX checkpoints; same 50%
  neutral source border, five-point alignment and normalized 512-D features.
- Same original inventory exclusions; no new test-driven exclusion or image
  quality filtering. Known enrollment photos are original unmasked views.
- One-photo threshold: 0.3887378871440888. Few-photo score-only threshold:
  0.3978477716445923. Primary threshold: 0.3778589367866516; margin: 0.15.
- Quality threshold stays zero, as selected during development. No learned
  adapter, AdaFace, fine-tuning or temporal aggregation is claimed.
- Frozen methods and test selection: `{args.run}/freeze.json` and
  `evaluation_manifest.csv`. Start receipt: `evaluation_started.json`.
- Seven protocol tests pass. Independent verification recomputes all
  **{verification['verified_cosine_searches']} successful per-method cosine searches** and
  **{verification['verified_prediction_attempts']} probe/method attempts**, checks the frozen gallery
  prototypes, decisions, summary counts, split separation and unchanged hashes
  for all **{verification['source_images_unchanged']} extracted source images**.
- The earlier incomplete metadata-only freeze attempt is marked aborted before
  inference. It did not open test images; this run is the completed frozen evaluation.
- Original images and biometric features remain outside the paper asset folder.
  The evaluation script refuses a second inference start for this freeze.

## Figure captions

1. **[Held-out protocol](fig00_heldout_protocol.png)** ([SVG](fig00_heldout_protocol.svg)).
   Development and test identities are disjoint within halfFace. Models,
   enrollment policy, primary method and thresholds are frozen before reserved
   image inference. The ICFD public-sample check is reported separately.
2. **[Held-out comparison](fig01_heldout_comparison.png)** ([SVG](fig01_heldout_comparison.svg)).
   End-to-end correct known acceptance for three methods on the same 160 masked
   known probes (left), with descriptive 95% known-identity bootstrap intervals,
   and false known acceptances among the same 151 unknown probes (right). One
   known capture failure remains included. Thresholds are development-selected.
3. **[Primary decision states](fig02_heldout_states.png)** ([SVG](fig02_heldout_states.svg)).
   Counts and within-truth-group percentages for known, unknown, ambiguous,
   insufficient-quality and capture-failure outcomes of the primary method.
   Capture failure is an extraction outcome rather than a fifth engine state.
   Quality filtering is disabled. No accepted known probe has a wrong identity.

Numeric sources: [summary](summary.csv), [decision states](decision_state_counts.csv),
[ICFD engineering table](icfd_summary.csv), [paired effects](paired_effects.csv),
[LaTeX result table](heldout_table.tex), [asset/source hashes](asset_manifest.json).

## Suggested paper paragraph

Using thresholds and a primary method frozen on identity-disjoint development
data, few-photo enrollment with score and margin decisions correctly accepted
151/160 masked known probes (94.38%) on held-out halfFace identities, compared
with 130/160 (81.25%) for one-photo enrollment with a calibrated score threshold.
No false known acceptance was observed among 151 held-out unknown probes for
the primary method, compared with one for the one-photo method. Few-photo
score-only decisions accepted 150/160 known probes (93.75%); the additional
margin rule helped two known probes and hurt one, for a net gain of one, with
one known probe marked ambiguous. All 151 unknown probes were rejected as
unknown. One known capture failure was retained in all denominators.
The descriptive known-identity bootstrap interval for primary acceptance was
88.89% to 98.74%, while the paired margin-versus-score interval included zero.
On five separately reserved ICFD public-sample identities, one-photo and
three-photo closed-set retrieval both reached 98/100, including two extraction
failures. These experiments support a limited public-dataset evaluation rather
than a claim of deployment performance.

## Interpretation limits

- **Zero observed false accepts does not establish zero population error.**
  There are only 27 unknown halfFace identities and correlated images. The CSV's
  percentile cluster bootstrap interval becomes [0,0] when every observed
  unknown cluster has zero errors; it cannot create unobserved events and is
  **not a population risk bound**. Report the sample count and this limitation.
- Intervals use 2,000 identity-cluster resamples, seed 42. They condition on the
  frozen calibration and omit calibration, pretraining and acquisition-domain
  uncertainty. No multiple-comparison significance claim is made.
- Identity separation is verified within the supplied manifests; overlap with
  pretrained ArcFace data, near duplicates or residual label errors is unknown.
- Known halfFace probes are masked and unknown probes mix mask conditions.
  Source IDs are used as supplied. This is not a Bangladesh-wide population
  sample or a live campus/CCTV deployment study.
- The earlier blur/resolution/JPEG results remain **development stress tests**;
  they were not repeated or tuned on these held-out identities.
- The reserved test is now consumed for final reporting. Do not adjust the
  method using these results and present a rerun as a fresh held-out test.

## Reproduction and cached verification

```powershell
# Creates a new pre-inference freeze for a clean reproduction:
python scripts/run_final_evaluation.py --freeze
# Use its printed freeze path, then evaluate once. Existing completed freeze refuses reruns.
# For this existing run, use saved real features without new image inference:
python scripts/verify_final_evaluation.py {args.run}
python scripts/render_final_evidence.py {args.run}
python -m unittest discover -s tests -p test_final_protocol.py -v
```

Reproductions after these published results are known are reproductions, not a
new unseen-test claim. Figures are 300 dpi PNG and editable SVG with no portraits.
'''
    (destination/'README.md').write_text(report,encoding='utf-8')
    manifest = dict(source_run=args.run,final_test_evaluated=True,thresholds_retuned=False,
        source_sha256={p:sha256(run/p) for p in ['freeze.json','evaluation_started.json','comparison.json','verification.json','summary.csv','icfd_summary.csv','paired_effects.csv']},
        renderer_sha256=sha256(__file__),assets_sha256={p.name:sha256(p) for p in destination.iterdir() if p.is_file() and p.name != 'asset_manifest.json'})
    (destination/'asset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('PAPER ASSETS',destination)


if __name__ == '__main__':
    main()
