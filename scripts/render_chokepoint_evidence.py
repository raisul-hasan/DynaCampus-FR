"""Render verified ChokePoint tables, portrait-free figures and thesis text."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'first':'First frame','random':'Seeded random frame','best_laplacian':'Best Laplacian',
          'best_norm':'Best feature norm','mean':'Uniform mean','softmax_norm_0.5':'Norm softmax beta=0.5',
          'softmax_norm_1':'Norm softmax beta=1','softmax_norm_2':'Norm softmax beta=2',
          'softmax_norm_5':'Norm softmax beta=5'}


def load(path):
    with path.open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    verified = json.loads((run/'verification.json').read_text())
    if verified['status'] != 'passed' or verified['completion_sha256'] != digest(run/'completion.json'):
        raise ValueError('Verified completion receipt required')
    completion = json.loads((run/'completion.json').read_text())
    for name,expected in completion['output_sha256'].items():
        if digest(run/name) != expected:
            raise ValueError('Verified input changed: '+name)
    cfg = json.loads((run/'config.json').read_text())
    prep = json.loads((run/'preparation.json').read_text())
    summaries, pairs, delta = load(run/'summary.csv'),load(run/'pairwise_verification.csv'),load(run/'paired_intervals.csv')
    quality = load(run/'quality_diagnostics.csv')
    dest = ROOT/'paper_assets/supplementary/chokepoint_v1'
    dest.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
    def s(condition,method,rule='fixed',contamination='none'):
        return next(r for r in summaries if (r['condition'],r['method'],r['rule'],r['contamination'],r['role'])
                    == (condition,method,rule,contamination,'evaluation'))
    def p(condition,method,contamination='none'):
        return next(r for r in pairs if (r['condition'],r['method'],r['contamination'])==(condition,method,contamination))
    def finish(fig,name,caption):
        fig.text(.01,.015,caption,fontsize=8,color='#475569',va='bottom')
        fig.tight_layout(rect=(0,.08,1,.97))
        fig.savefig(dest/(name+'.png'),dpi=180)
        fig.savefig(dest/(name+'.svg'))
        plt.close(fig)
    levels=['clean','mild','medium','severe']
    colors=['#64748b','#0f766e']
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for method,color in zip(['first','mean'],colors):
        axes[0].plot(levels,[100*float(s(c,method)['rank1']) for c in levels],'-o',color=color,label=LABELS[method])
        axes[1].plot(levels,[100*float(p(c,method)['tar_pair_far_1pct']) for c in levels],'-o',color=color)
        axes[2].plot(levels,[100*float(p(c,method)['eer']) for c in levels],'-o',color=color)
    for ax,title in zip(axes,['Known-track Rank-1 (%)','Pairwise TAR at FAR <= 1% (%)','Pairwise EER (%)']):
        ax.set_title(title,fontsize=11); ax.grid(alpha=.18)
    axes[0].legend(frameon=False)
    finish(fig,'fig01_degradation','Paired evaluation tracks; supplied eye-aligned crops with synthetic corruption. Pairwise ROC summaries are descriptive.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    methods=cfg['methods']
    positions=np.arange(len(methods))
    for ax,metric,title in zip(axes,['rank1','dir'],['Mixed-quality Rank-1 (%)','Mixed-quality DIR at frozen thresholds (%)']):
        values=[100*float(s('mixed',m)[metric]) for m in methods]
        ax.barh(positions,values,color=['#0f766e' if m=='mean' else '#64748b' for m in methods])
        ax.set_yticks(positions,[LABELS[m] for m in methods] if ax is axes[0] else [])
        ax.invert_yaxis(); ax.set_xlim(0,105); ax.set_title(title,fontsize=11)
        for y,v in enumerate(values): ax.text(min(v+1,100),y,f'{v:.1f}',va='center',fontsize=8)
    finish(fig,'fig02_mixed_pooling','All nine predeclared methods use identical five-frame known and unknown tracks. Thresholds use calibration data only.')
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    axes[0].bar(levels,[float(r['mean_laplacian']) for r in quality],color='#64748b')
    axes[1].bar(levels,[float(r['mean_norm']) for r in quality],color='#0f766e')
    axes[0].set_title('Mean Laplacian/contrast quality')
    axes[1].set_title('Mean raw ArcFace feature norm')
    axes[1].set_ylim(0,max(float(r['mean_norm']) for r in quality)*1.15)
    finish(fig,'fig03_quality','One record per selected known evaluation frame and tier. These are heuristic quality signals, not calibrated probabilities.')
    fig,ax=plt.subplots(figsize=(10,4.5))
    controls=['none','matched','mild']; x=np.arange(3)
    for j,(method,color) in enumerate(zip(['mean','best_norm','softmax_norm_1'],['#0f766e','#64748b','#b45309'])):
        ax.bar(x+(j-1)*.24,[100*float(s('mixed',method,contamination=c)['rank1']) for c in controls],.24,
               label=LABELS[method],color=color)
    ax.set_xticks(x,['No contamination','One intruder: matched tier','One intruder: mild tier'])
    ax.set_ylim(0,105); ax.set_ylabel('Known-track Rank-1 (%)'); ax.set_title('Paired simulated identity contamination',pad=35)
    ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=3,fontsize=9)
    finish(fig,'fig04_contamination','Same intruder path per track across all methods; last frame replaced in known tracks only. No real tracker was run.')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    cs=['clean','mild','medium','severe','mixed']; x=np.arange(len(cs))
    for j,(rule,color) in enumerate(zip(['fixed','aosm'],['#0f766e','#b45309'])):
        for ax,metric in zip(axes,['dir','fpir']):
            ax.bar(x+(j-.5)*.34,[100*float(s(c,'mean',rule)[metric]) for c in cs],.34,
                   label='Fixed' if rule=='fixed' else 'Floored AOSM',color=color)
    axes[0].set_title('Uniform mean: evaluation DIR (%)',pad=35)
    axes[1].set_title('Uniform mean: evaluation FPIR (%)',pad=35)
    for ax in axes:
        ax.set_xticks(x,cs)
        ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,fontsize=9)
    finish(fig,'fig05_thresholds','Each rule targets <=1% empirical calibration FPIR (66 unknown tracks). Evaluation FPIR is measured, not forced to match.')
    main_rows=[]
    for condition in cfg['conditions']:
        for method in ['first','mean']:
            r=s(condition,method); v=p(condition,method)
            main_rows.append(f"| {condition.title()} | {LABELS[method]} | {r['rank1_correct']}/{r['known_tracks']} ({100*float(r['rank1']):.2f}%) | {100*float(v['tar_pair_far_1pct']):.2f}% | {100*float(v['eer']):.2f}% | {r['correct_known_accepted']}/{r['known_tracks']} | {r['false_known_accepted']}/{r['unknown_tracks']} |")
    mixed_rows=[]
    for method in methods:
        r=s('mixed',method)
        mixed_rows.append(f"| {LABELS[method]} | {100*float(r['rank1']):.2f}% | {100*float(r['dir']):.2f}% | {r['false_known_accepted']}/{r['unknown_tracks']} |")
    delta_rows=[]
    for condition in ['severe','mixed']:
        for row in delta:
            if row['condition']==condition and row['contamination']=='none' and row['metric']=='rank1':
                delta_rows.append(f"| {condition.title()} | {LABELS[row['comparison']]} minus {LABELS[row['reference']]} | {100*float(row['delta']):+.2f} | [{100*float(row['ci_low']):+.2f}, {100*float(row['ci_high']):+.2f}] | {row['helped']} / {row['harmed']} |")
    quality_rows=[f"| {r['level'].title()} | {float(r['mean_laplacian']):.4f} | {float(r['mean_norm']):.2f} | {float(r['norm_spearman_r']):+.3f} | {float(r['laplacian_spearman_r']):+.3f} |" for r in quality]
    threshold_rows=[]
    for condition in ['severe','mixed']:
        for rule in ['fixed','aosm']:
            r=s(condition,'mean',rule)
            threshold_rows.append(f"| {condition.title()} | {rule} | {r['correct_known_accepted']}/{r['known_tracks']} ({100*float(r['dir']):.2f}%) | {r['false_known_accepted']}/{r['unknown_tracks']} | {float(r['threshold']):.6f} |")
    contam_rows=[]
    for control in controls:
        for method in ['mean','best_norm','softmax_norm_1']:
            r=s('mixed',method,contamination=control)
            contam_rows.append(f"| {control} | {LABELS[method]} | {r['rank1_correct']}/{r['known_tracks']} ({100*float(r['rank1']):.2f}%) |")
    severe_first,severe_mean=s('severe','first'),s('severe','mean')
    mixed_mean=s('mixed','mean')
    difference=100*(float(severe_mean['rank1'])-float(severe_first['rank1']))
    counts=prep['counts']
    rel=run.relative_to(ROOT).as_posix()
    source_link='../../../'+rel
    report=f'''# Corrected ChokePoint temporal study

Completed 23 October 2026 (Asia/Dhaka). **Verified exploratory supplementary
evidence.** This is a corrected sequence-separated experiment, not a reproduction
of the supplied paper's numbers or a fresh unseen-identity test. It supplements
the existing still-image research; it does not establish campus CCTV performance.

Run: `{rel}`. [Independent verification]({source_link}/verification.json).
The verifier checked {verified['query_predictions_checked']:,} query predictions,
{verified['gallery_scores_checked']:,} gallery scores, {verified['corrupted_pixels_checked']:,}
corrupted pixel arrays and {verified['repeated_real_inference_checks']} repeated real-model samples.

## Protocol and research question

Does temporal averaging help frozen recognition under synthetic degradation,
does quality weighting improve on that baseline, and how do quality thresholds
and simulated identity contamination affect recognition?

- Original ChokePoint P1E crop and ground-truth archives, verified against
  official published MD5s; source/terms receipts stay under ignored data/.
- 25 identities split deterministically into 17 gallery-known and 8 unknown.
  One first-available numeric crop per known identity from P1E_S1_C1; no claim
  that this crop was selected as frontal or optimal quality.
- Calibration: P1E_S2_C2, {counts['calibration_known']} known and
  {counts['calibration_unknown']} unknown tracks. Evaluation: P1E_S3_C3 and
  P1E_S4_C1, {counts['evaluation_known']} known and {counts['evaluation_unknown']} unknown tracks.
  Roles have disjoint files/sequences; their identity populations are shared.
- Five numerically consecutive frames per track, no overlap or crossing camera,
  sequence or identity boundaries. Earliest eligible tracks up to 15 per identity
  per sequence; unused-frame accounting is saved, not hidden.
- Supplied eye-aligned 96x96 grayscale crops decoded as BGR, resized to 112x112
  using cubic interpolation BEFORE corruption. Four fixed tiers and seeded mixed
  tracks; severe cubic downsampling to 28x28 and independent noise sigma 15.
  No detector, landmark estimator or real tracker is evaluated.
- Real frozen ArcFace ResNet-50, CPU ONNX inference; raw feature norms retained
  separately from unit embeddings. Gallery templates and checkpoint stay fixed.
- Nine predeclared aggregation policies use exactly the same pixels. Unknown
  queries are also five-frame tracks. Beta settings are diagnostic comparisons,
  not selected using evaluation outcomes.
- Each method/condition's fixed and floored-AOSM thresholds are frozen from
  unknown calibration scores at <=1% empirical FPIR. With 66 unknown calibration
  tracks, this requires zero observed false accepts. This does not guarantee
  1% population FPIR. AOSM uses margin 0.08, floor 0.20 and norm bounds 16-22.
- Paired 95% percentile intervals use 1,000 identity-cluster resamples, retaining
  both evaluation sequences and adjacent tracks within the sampled identity.
  These do not measure uncertainty over independently sampled cameras/sites.

## First frame versus temporal mean

The useful findings are separate: temporal mean pooling improves severe-tier
retrieval, while calibration does not reliably transfer at the intended false
acceptance target. For mixed tracks, uniform mean pooling accepts
{mixed_mean['correct_known_accepted']}/{mixed_mean['known_tracks']} known queries but
also accepts {mixed_mean['false_known_accepted']}/{mixed_mean['unknown_tracks']}
unknown queries ({100*float(mixed_mean['fpir']):.2f}% observed FPIR), despite zero
calibration false accepts. High Rank-1 therefore does not establish safe unknown
rejection, and the frozen thresholds have not demonstrated 1% evaluation FPIR.

| Condition | Method | Rank-1 known tracks | Pairwise TAR at FAR <=1% | Pairwise EER | Correct known acceptance | False unknown acceptance |
|---|---|---|---|---|---|---|
{chr(10).join(main_rows)}

Under severe corruption, the observed mean-minus-first Rank-1 difference is
**{difference:+.2f} percentage points**. Pairwise ROC operating points above are
descriptive verification statistics computed on evaluation comparisons. They
must not be confused with independently calibrated open-set acceptance rates.
The last two columns instead use calibration-frozen fixed thresholds.

![Degradation comparison](fig01_degradation.png)

## Quality weighting on mixed tracks

| Method | Rank-1 | Frozen-threshold DIR | False unknown acceptance |
|---|---|---|---|
{chr(10).join(mixed_rows)}

![Mixed aggregation](fig02_mixed_pooling.png)

The paired differences below address whether weighting helps over the mean.
An interval containing zero does not establish a general improvement or
equivalence. These predeclared comparisons are descriptive, without a claim
of significance from multiple unadjusted tests.

| Condition | Comparison | Rank-1 change (points) | Paired 95% interval (points) | Helped / harmed tracks |
|---|---|---|---|---|
{chr(10).join(delta_rows)}

## Quality-proxy diagnostic

| Tier | Mean Laplacian/contrast | Mean raw norm | Norm/similarity Spearman r | Laplacian/similarity Spearman r |
|---|---|---|---|---|
{chr(10).join(quality_rows)}

![Quality proxies](fig03_quality.png)

These correlations use selected known evaluation frames. Frame dependence and
17 identity clusters limit inference. Nominal p-values in the numeric source
are not evidence that frames are independent. Quality-proxy correlation does
not itself establish that quality weighting improves recognition.

## Paired simulated contamination

| Intruder control | Method | Known-track Rank-1 |
|---|---|---|
{chr(10).join(contam_rows)}

![Contamination comparison](fig04_contamination.png)

Exactly one last-frame substitution per known track (20%); the same foreign
identity/image is used for every method. Matched-tier and mild-tier controls
use the same intruder source. Unknown tracks remain unchanged in this
known-track sensitivity analysis. This is not measured tracker performance.

## Fixed versus floored quality-adaptive thresholds

Uniform mean pooling, separate calibration for each rule:

| Condition | Rule | Correct known acceptance / DIR | False unknown acceptance | Frozen base threshold |
|---|---|---|---|---|
{chr(10).join(threshold_rows)}

![Frozen-threshold comparison](fig05_thresholds.png)

Both rules target the same CALIBRATION FPIR budget; evaluation FPIR is measured
and may differ. Therefore this table does not claim an evaluation comparison at
exactly matched FPIR. Pairwise FAR and unknown-query FPIR have different denominators.

## Limits and relationship to the supplied manuscript

Stable SHA-256-derived seeds, sequence boundaries, paired intruders, crop resize
order and the AOSM floor correct issues in the source notebook. Consequently
the supplied scores are not substituted into these tables. The original paper
already inspected P1E data, so sequence separation here is a protocol improvement,
not a claim that these identities or recordings were previously unexamined.

This is one small portal, one public checkpoint and synthetic independent noise.
Camera and recording sequence change together; their effects are not isolated.
Calibration and evaluation reuse identities, and only eight unknown identities
are represented. Rejection counts and identity-aware intervals must accompany
rates. Norm bounds came from the supplied pilot, so this is not a new independent
quality-proxy validation. The study does not train a domain adapter, test dynamic
gallery growth, evaluate actual tracking or establish Bangladesh campus accuracy.

## Reproduce and inspect

Original sources: [official ChokePoint page](https://arma.sourceforge.net/chokepoint/),
[original archive record](https://zenodo.org/records/815657).
Attribution: NICTA; Wong et al., Patch-based Probabilistic Image Quality Assessment
for Face Selection and Improved Video-based Face Recognition, CVPR Workshops 2011.
The synthetic images are derivatives of supplied crops, not original recordings;
raw/derived media and embeddings remain local under ignored data/ with the source
license retained. Public figures contain no portraits.

From the repository root, with the verified local environment and model:

```powershell
.\\.venv\\Scripts\\python.exe -u scripts/fetch_chokepoint.py
.\\.venv\\Scripts\\python.exe -u scripts/run_chokepoint_temporal.py --prepare
# Use the NEW run path printed by --prepare for fresh inference:
.\\.venv\\Scripts\\python.exe -u scripts/run_chokepoint_temporal.py --run <new-run-path>
# Audit this recorded run without rerunning the original experiment:
.\\.venv\\Scripts\\python.exe scripts/verify_chokepoint_temporal.py {rel} --real-sample
.\\.venv\\Scripts\\python.exe scripts/render_chokepoint_evidence.py {rel}
```

Numeric sources: [all summaries]({source_link}/summary.csv),
[all predictions]({source_link}/predictions.csv),
[pairwise verification]({source_link}/pairwise_verification.csv),
[paired intervals]({source_link}/paired_intervals.csv),
[calibration thresholds]({source_link}/calibration.csv),
[quality diagnostics]({source_link}/quality_diagnostics.csv),
[track manifest]({source_link}/tracks.json),
[selection accounting]({source_link}/selection_accounting.csv).
'''
    (dest/'README.md').write_text(report,encoding='utf-8')
    severe_delta=next(r for r in delta if (r['condition'],r['contamination'],r['reference'],r['comparison'],r['metric'])
                      == ('severe','none','first','mean','rank1'))
    mixed_delta=next(r for r in delta if (r['condition'],r['contamination'],r['reference'],r['comparison'],r['metric'])
                     == ('mixed','none','mean','softmax_norm_1','rank1'))
    mean_mixed=s('mixed','mean'); norm_mixed=s('mixed','softmax_norm_1')
    fixed_mixed=s('mixed','mean','fixed'); aosm_mixed=s('mixed','mean','aosm')
    chapter=f'''# Supplementary study: temporal aggregation under controlled degradation

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
{severe_first['rank1_correct']}/303 ({100*float(severe_first['rank1']):.2f}%), compared with
{severe_mean['rank1_correct']}/303 ({100*float(severe_mean['rank1']):.2f}%) for uniform pooling.
The paired mean-minus-first difference was {100*float(severe_delta['delta']):+.2f}
percentage points, with a 95% identity-cluster bootstrap interval of
[{100*float(severe_delta['ci_low']):+.2f}, {100*float(severe_delta['ci_high']):+.2f}] points.
The intervals used 1,000 paired identity resamples, retaining each sampled
identity's tracks and both evaluation sequences.

On mixed-quality tracks, uniform pooling achieved
{mean_mixed['rank1_correct']}/303 ({100*float(mean_mixed['rank1']):.2f}%) Rank-1, while
norm-softmax pooling at beta 1 achieved
{norm_mixed['rank1_correct']}/303 ({100*float(norm_mixed['rank1']):.2f}%). The corresponding
softmax-minus-mean difference was {100*float(mixed_delta['delta']):+.2f} points
(95% paired interval [{100*float(mixed_delta['ci_low']):+.2f},
{100*float(mixed_delta['ci_high']):+.2f}]). All nine declared policies and paired
helped/harmed counts are retained in the accompanying report, including negative
or unchanged outcomes. Quality-proxy correlations are diagnostic and do not
by themselves establish a recognition benefit from weighting.

With mean pooling on mixed tracks, the calibration-frozen fixed rule correctly
accepted {fixed_mixed['correct_known_accepted']}/303 known tracks and falsely accepted
{fixed_mixed['false_known_accepted']}/143 unknown tracks. The separately calibrated
floored AOSM rule correctly accepted {aosm_mixed['correct_known_accepted']}/303 known
tracks and falsely accepted {aosm_mixed['false_known_accepted']}/143 unknown tracks.
These are observed evaluation counts at calibration-selected operating points,
not an assertion of population error rates or exactly matched evaluation FPIR.
In particular, the fixed rule's mixed-track evaluation FPIR is
{100*float(fixed_mixed['fpir']):.2f}% despite zero calibration false accepts. The
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
'''
    (dest/'thesis_section.md').write_text(chapter,encoding='utf-8')
    (ROOT/'docs/CHOKEPOINT_RESULTS.md').write_text('# ChokePoint supplementary results\n\n'
        'Verified corrected exploratory study completed 23 October 2026.\n\n'
        '[Full report, figures, tables, limits and reproduction](../paper_assets/supplementary/chokepoint_v1/README.md).\n\n'
        'This supplements the still-image experiments; the original supplied scores remain unreproduced.\n',encoding='utf-8')
    # A compact paper table uses the verified counts/rates, never the supplied PDF numbers.
    latex=['\\begin{tabular}{llrrr}','\\hline','Condition & Method & Rank-1 (\\%) & DIR (\\%) & False accepts \\\\', '\\hline']
    for c in cfg['conditions']:
        for m in ['first','mean']:
            r=s(c,m)
            latex.append(f"{c.title()} & {LABELS[m]} & {100*float(r['rank1']):.2f} & {100*float(r['dir']):.2f} & {r['false_known_accepted']}/{r['unknown_tracks']} \\\\")
    latex += ['\\hline','\\end{tabular}']
    (dest/'temporal_table.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    receipt={'run':rel,'verification_sha256':digest(run/'verification.json'),
        'renderer_sha256':digest(Path(__file__)), 'files':{p.name:digest(p) for p in dest.iterdir() if p.is_file() and p.name!='render_receipt.json'}}
    (dest/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('Evidence:',dest.relative_to(ROOT))
    print('Severe Rank-1 mean-minus-first:',f'{difference:+.2f} percentage points')


if __name__=='__main__':
    main()
