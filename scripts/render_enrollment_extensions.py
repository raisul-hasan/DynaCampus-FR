"""Create figures, paper tables and a results section from verified extensions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
POLICIES={'one_photo':'One original','augmented_one_photo':'One + five variants','up_to_three_photos':'Up to three originals'}
MODEL_LABELS={'arcface':'ArcFace','adaface':'AdaFace'}


def read(p):
    with p.open(encoding='utf-8',newline='') as stream: return list(csv.DictReader(stream))


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('run',type=Path)
    args=parser.parse_args(); run=(ROOT/args.run).resolve()
    verify=json.loads((run/'verification.json').read_text())
    assert verify['status']=='passed' and verify['completion_sha256']==sha(run/'completion.json')
    complete=json.loads((run/'completion.json').read_text())
    for name,digest in complete['output_sha256'].items(): assert sha(run/name)==digest,name
    cfg=json.loads((run/'config.json').read_text()); aug=read(run/'augmentation_summary.csv')
    growth=read(run/'growth_summary.csv'); runtime=read(run/'runtime_summary.csv'); effects=read(run/'paired_effects.csv')
    dest=ROOT/'paper_assets/development/extensions_v1'; dest.mkdir(parents=True,exist_ok=True)
    def a(model,policy,family,view='recalibrated_development'):
        return next(r for r in aug if (r['model'],r['policy'],r['family'],r['view'])==(model,policy,family,view))
    def rt(model,phase,stage,n=55):
        return next(r for r in runtime if (r['model'],r['phase'],r['stage'],r['gallery_size'],r['status'])
                    == (model,phase,stage,str(n),'ok'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
    def finish(fig,name,caption,top=.94):
        fig.text(.01,.02,caption,fontsize=8,color='#475569',va='bottom')
        fig.tight_layout(rect=(0,.10,1,top))
        fig.savefig(dest/(name+'.png'),dpi=180); fig.savefig(dest/(name+'.svg')); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    x=np.arange(3); policies=cfg['policies']
    for ax,model in zip(axes,cfg['models']):
        for j,(family,label,color) in enumerate([('score_only','Score only','#0f766e'),('score_margin','Score + margin','#64748b')]):
            values=[int(a(model,p,family)['correct_known_accepts']) for p in policies]
            ax.bar(x+(j-.5)*.34,values,.34,color=color,label=label)
            for xx,policy,value in zip(x+(j-.5)*.34,policies,values):
                false=a(model,policy,family)['false_known_unknowns']
                ax.text(xx,value+1,f'{value}\nFA {false}',ha='center',fontsize=8)
        ax.set_xticks(x,['One original','One + five\nvariants','Up to three\noriginals'])
        ax.set_ylim(0,160); ax.set_title(MODEL_LABELS[model],pad=35); ax.set_ylabel('Correct known accepts / 154 attempts')
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,fontsize=9)
    finish(fig,'fig01_augmented_enrollment','Descriptive development recalibration on the same probes; FA labels count false accepts among 149 extracted unknowns.',top=.88)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    handles=[]
    for model,color in [('arcface','#0f766e'),('adaface','#64748b')]:
        for family,marker in [('score_only','o'),('score_margin','^')]:
            subset=[r for r in growth if r['model']==model and r['family']==family]
            for ax,key in zip(axes,['correct_known_accepts','false_known_unknowns']):
                means=[]; lows=[]; highs=[]
                for n in [10,25,55]:
                    values=[int(r[key]) for r in subset if int(r['gallery_size'])==n]
                    means.append(np.mean(values)); lows.append(min(values)); highs.append(max(values))
                means=np.array(means)
                h=ax.errorbar([10,25,55],means,yerr=[means-np.array(lows),np.array(highs)-means],
                    marker=marker,color=color,linestyle='-' if family=='score_only' else '--',capsize=4,
                    label=MODEL_LABELS[model]+(' score' if family=='score_only' else ' score+margin'))
                if ax is axes[0]: handles.append(h)
    axes[0].set_title('Correct known accepts / 28 fixed attempts'); axes[0].set_ylim(0,30)
    axes[1].set_title('False accepts / 149 extracted unknowns'); axes[1].set_ylim(-.1,2.8)
    for ax in axes: ax.set_xticks([10,25,55]); ax.set_xlabel('Enrolled identities'); ax.grid(alpha=.15)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=4,frameon=False,fontsize=9)
    finish(fig,'fig02_gallery_growth','Fixed probes/thresholds. At 25 identities: mean of five background compositions, with MIN/MAX whiskers (not confidence intervals).',top=.88)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    stage_names=[('recognition_end_to_end','Image to decision'),('detection','Detection'),('embedding','Embedding'),
                 ('gallery_search','Gallery search'),('decision','Decision')]
    for ax,model in zip(axes,cfg['models']):
        values=[float(rt(model,'recognition',stage)['median_ms']) for stage,_ in stage_names]
        p95=[float(rt(model,'recognition',stage)['p95_ms']) for stage,_ in stage_names]
        y=np.arange(len(values)); ax.scatter(values,y,color='#0f766e',marker='o',s=35,label='Median',zorder=3)
        ax.hlines(y,values,p95,color='#94a3b8',linewidth=2)
        ax.scatter(p95,y,color='#b45309',marker='D',s=25,label='p95',zorder=3)
        ax.set_yticks(y,[label for _,label in stage_names]); ax.invert_yaxis(); ax.set_xscale('log')
        ax.set_xlim(.001,350)
        ax.set_xlabel('Milliseconds, logarithmic scale'); ax.set_title(MODEL_LABELS[model],pad=35); ax.grid(axis='x',alpha=.15)
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False,fontsize=9)
    finish(fig,'fig03_runtime','60 warm real-image recognition calls per model; CPU inference threads=2, 55-identity gallery. No browser, camera capture or network timing.',top=.88)
    fig,ax=plt.subplots(figsize=(8,4.4))
    for model,color in [('arcface','#0f766e'),('adaface','#64748b')]:
        med=[1000*float(rt(model,'microbenchmark','gallery_search',n)['median_ms']) for n in [10,25,55]]
        high=[1000*float(rt(model,'microbenchmark','gallery_search',n)['p95_ms']) for n in [10,25,55]]
        ax.plot([10,25,55],med,'-o',color=color,label=MODEL_LABELS[model]+' median')
        ax.plot([10,25,55],high,'--',color=color,alpha=.8,label=MODEL_LABELS[model]+' p95')
    ax.set_xticks([10,25,55]); ax.set_ylim(bottom=0); ax.set_xlabel('Enrolled identities')
    ax.set_ylabel('Search time (microseconds)'); ax.set_title('Actual persistent-gallery search call')
    ax.legend(frameon=False,loc='upper left',fontsize=9); ax.grid(alpha=.15)
    finish(fig,'fig04_search_scaling','300 calls per model/size; includes probe validation, rebuilding the matrix, cosine scores, sorting and metadata lookup.')
    aug_lines=[]
    for model in cfg['models']:
        for family in cfg['decision_families']:
            for policy in policies:
                r=a(model,policy,family)
                aug_lines.append(f"| {model} | {family} | {POLICIES[policy]} | {r['correct_known_accepts']}/154 | {r['false_known_unknowns']}/149 | {100*float(r['valid_decision_coverage']):.2f}% |")
    fixed_lines=[]
    for model in cfg['models']:
        for policy in policies:
            r=a(model,policy,'score_margin','fixed_one_photo')
            fixed_lines.append(f"| {model} | {POLICIES[policy]} | {r['correct_known_accepts']}/154 | {r['false_known_unknowns']}/149 |")
    growth_lines=[]
    def span(values):
        lo,hi=min(values),max(values)
        return str(lo) if lo==hi else f'{lo}-{hi}'
    for model in cfg['models']:
        for family in cfg['decision_families']:
            for n in [10,25,55]:
                data=[r for r in growth if r['model']==model and r['family']==family and int(r['gallery_size'])==n]
                growth_lines.append(f"| {model} | {family} | {n} | {span([int(r['correct_known_accepts']) for r in data])}/28 | {span([int(r['false_known_unknowns']) for r in data])}/149 |")
    timing_lines=[]
    requested=[('enrollment','enrollment_end_to_end'),('enrollment','persistent_insertion'),
        ('recognition','recognition_end_to_end'),('recognition','detection'),('recognition','embedding'),
        ('recognition','gallery_search'),('recognition','decision')]
    for model in cfg['models']:
        for phase,stage in requested:
            r=rt(model,phase,stage)
            timing_lines.append(f"| {model} | {phase}: {stage} | {r['samples']} | {float(r['median_ms']):.4f} | {float(r['p95_ms']):.4f} |")
    effect_lines=[]
    for r in effects:
        if r['view']=='recalibrated_development':
            effect_lines.append(f"| {r['model']} | {POLICIES[r['policy']]} minus one original | {r['family']} | {float(r['delta_percentage_points']):+.2f} | [{float(r['ci95_low_percentage_points']):+.2f}, {float(r['ci95_high_percentage_points']):+.2f}] |")
    rel=run.relative_to(ROOT).as_posix(); link='../../../'+rel
    report=f'''# Augmented enrollment, gallery growth and runtime

Completed 23 October 2026 (Asia/Dhaka). **Verified descriptive development
evidence**, separate from the frozen halfFace final evaluation. No training,
reserved-test image inference or application-default change was performed.

Run: `{rel}`. [Separate verification]({link}/verification.json) replayed all
550 synthetic-variant model forwards and checked {verify['predictions_checked']:,}
predictions, {verify['calibration_candidates_checked']:,} calibration candidates,
1,800 runtime searches and 180 real-image timing calls. The full suite passes
104 tests. [Integrated thesis draft](../../../docs/THESIS_DRAFT.md).

## Augmented one-photo enrollment

Same 55 development gallery identities, 154 masked known attempts and 150
unknown attempts; 148 known and 149 unknown probes yield embeddings. All seven
capture failures remain in attempted denominators. Existing original-photo
features are reused for both public checkpoints. Fresh detection/alignment of
each frontal enrollment source reproduces its original crop and feature cache.

Three policies: one original frontal photo; that same aligned crop plus five
equally weighted variants (Gaussian blur sigma 1, cubic down/up to 56 pixels,
JPEG quality 50, and gamma 0.7/1.4); and up to three original unmasked views.
Templates are normalized means. The variants do not add independent people,
views or test samples, and do not simulate masking. Probes are unchanged.
Augmented crops are not redetected. The original has weight 1/6 in its six-vector
template; this is one declared recipe, not an exhaustive augmentation search.

### Separately calibrated development comparison

| Checkpoint | Rule | Enrollment | Correct known / attempts | False unknown acceptance | Valid decision coverage |
|---|---|---|---|---|---|
{chr(10).join(aug_lines)}

Calibration and evaluation reuse these same development probes. Score/margin
selection maximizes correct known acceptance subject to <=1% conditional FPIR
and at least 85% valid known/unknown decision coverage. The full candidate grid
and deterministic tie-break are saved. These are optimistic descriptive results,
not a held-out estimate or evidence of 1% population FPIR.

![Augmentation comparison](fig01_augmented_enrollment.png)

The tested synthetic enrollment profile does not improve score-only acceptance
on either checkpoint: ArcFace changes 129 to 128 and AdaFace 112 to 111 out of
154 attempts. Margin decisions interact with the checkpoint: ArcFace changes
139 to 135 while AdaFace changes 120 to 123 after separate calibration.
The latter small gain has a paired interval touching zero. Three originals
increase score-only acceptance to 138 for ArcFace and 122 for AdaFace. These
findings support reporting the exact policy/checkpoint/rule, not a universal
claim that augmentation helps or never helps.

| Checkpoint | Paired comparison | Rule | Change (points) | Descriptive 95% interval (points) |
|---|---|---|---|---|
{chr(10).join(effect_lines)}

Intervals resample 55 known identities 2,000 times with fixed selected thresholds;
they omit calibration and unknown-sample uncertainty. Multiple ablations are
descriptive, without an unadjusted significance claim.

### Fixed previous one-photo score/margin rule

| Checkpoint | Enrollment | Correct known / attempts | False unknown acceptance |
|---|---|---|---|
{chr(10).join(fixed_lines)}

This view holds each checkpoint's previous one-photo rule fixed across policies.
It is not an equal-FPIR comparison. For example, ArcFace's three-original
acceptance of 143 accompanies four false accepts, versus 139 and one for one
original. Those rates must be interpreted together. Both threshold views are
retained rather than selecting whichever favors a policy.

## Fixed-probe gallery growth

The first ten IDs in the original frozen development-known order form the core.
They contribute 28 masked known attempts (26 extracted). The same 150 unknown
attempts (149 extracted) are used at every size. The other 45 known development
identities are background gallery entries and never enter these probes. Unknown
identities are never enrolled. All templates use up to three original photos.

The ten-core and full 55-person galleries are each evaluated once. Five seeded
25-person galleries add 15 different background selections to the same core.
Each checkpoint's existing 55-person score and score/margin thresholds stay
fixed for all seven compositions. Thus neither probes nor thresholds change
when the gallery grows.

| Checkpoint | Rule | Gallery identities | Correct known / 28 | False unknown acceptance / 149 |
|---|---|---|---|---|
{chr(10).join(growth_lines)}

Ranges at 25 identities describe composition sensitivity, not confidence
intervals or independent new participants. Rank-1 is unchanged at 26/28 for both
checkpoints across these compositions. This small core does not establish
accuracy at arbitrary scale. Score-only unknown maximum similarity is
nondecreasing in nested galleries, but score/margin false accepts need not be:
added competitors can also trigger ambiguity. AdaFace illustrates this with
two false accepts at ten identities and zero at 55 under its fixed margin rule.

![Gallery growth](fig02_gallery_growth.png)

## Actual CPU runtime

Environment: Windows 11, AMD64 Family 23 Model 113 Stepping 0 processor,
12 logical CPUs, two inference threads per engine, ONNX Runtime CPU for ArcFace
and PyTorch CPU for AdaFace. NumPy BLAS pool is two threads; complete loaded-pool
details, including other libraries, are saved in the environment receipt.
Both pretrained checkpoints and all source images are hashed.

| Checkpoint | Timed boundary | Calls | Median ms | p95 ms |
|---|---|---|---|---|
{chr(10).join(timing_lines)}

Enrollment measures direct wall time from image-file decoding through a fresh
persistent identity insertion into a 55-entry isolated gallery (55 to 56 entries),
then restores the benchmark gallery outside the timed window. Ten fixed original
enrollment inputs are repeated three times per checkpoint. Recognition uses ten
known and ten unknown original probes, repeated three times, from decoding
through actual gallery search and four-state decision. All 180 sampled calls
produce valid features; the sampler is not a new accuracy study.

![Runtime stages](fig03_runtime.png)

An additional 300 real search/decision calls per model at each gallery size
measure the actual FaceGallery implementation: validation, matrix reconstruction,
cosine scores, sorting and metadata lookup. ArcFace search medians are about
0.030, 0.041 and 0.064 ms at 10, 25 and 55 identities. Full ArcFace image-to-decision
median is about 143.81 ms, substantially larger than search alone.

![Search scaling](fig04_search_scaling.png)

Repeated calls on fixed inputs measure warm laptop latency, not independent
accuracy samples. Model load/warmup, browser/UI transfer, camera capture and
network/RTSP are excluded. End-to-end values are directly measured wall times;
they are not sums of component medians and are not camera FPS. Checkpoint
pretraining, architecture details and runtime engines differ, so the ArcFace/
AdaFace timing gap does not isolate either training loss.

## Reproduce and inspect

```powershell
.\\.venv\\Scripts\\python.exe -u scripts/run_enrollment_extensions.py --prepare
.\\.venv\\Scripts\\python.exe -u scripts/run_enrollment_extensions.py --run <new-prepared-run>
.\\.venv\\Scripts\\python.exe scripts/verify_enrollment_extensions.py {rel}
.\\.venv\\Scripts\\python.exe scripts/render_enrollment_extensions.py {rel}
```

Sources: [augmentation summary]({link}/augmentation_summary.csv),
[all augmentation predictions]({link}/augmentation_predictions.csv),
[thresholds]({link}/augmentation_thresholds.csv), [candidate grid]({link}/augmentation_grid.csv),
[paired effects]({link}/paired_effects.csv), [growth summary]({link}/growth_summary.csv),
[growth predictions]({link}/growth_predictions.csv), [selection manifest]({link}/selection.json),
[runtime samples]({link}/runtime_samples.csv), [runtime summary]({link}/runtime_summary.csv),
[timing boundaries and hardware]({link}/runtime_environment.json).
Face images, derived inspection grids and embedding arrays remain under ignored
data/. These public figures contain no participant portraits.
'''
    (dest/'README.md').write_text(report,encoding='utf-8')
    chapter=f'''# Additional development experiments

## Synthetic one-photo enrollment

To distinguish additional image information from synthetic variation, the same
55-person development gallery was constructed from one frontal original,
that original plus five deterministic mild variants, or up to three original
unmasked views. Both pretrained checkpoints used identical source crops and
unchanged probes. There were 154 known masked attempts and 150 unknown attempts;
six known and one unknown extraction failure remained in the attempted counts.
Templates averaged unit embeddings and were normalized again after pooling.

The five variants comprised blur sigma 1, cubic reduction to 56x56 and restoration
to 112x112, JPEG quality 50, and gamma values 0.7 and 1.4. Each variant and the
original had equal weight. These operations change capture quality rather than
creating new pose information or simulating masks. Policies were evaluated both
with the existing one-photo thresholds fixed and with separate development
calibration. The latter reuses development probes and is descriptive.

With separately calibrated score-only rules, augmentation changed ArcFace correct
acceptance from 129 to 128 of 154 and AdaFace from 112 to 111. Up to three original
views instead achieved 138 and 122. With score/margin rules, augmentation changed
ArcFace from 139 to 135 and AdaFace from 120 to 123. The small AdaFace gain has
a descriptive paired identity-bootstrap interval that touches zero. The tested
augmentation recipe therefore provides no consistent replacement for additional
original views. This does not establish that all augmentation recipes are ineffective.

## Gallery growth under fixed probes and thresholds

A fixed ten-identity core supplied 28 masked known attempts, while the same 150
unknown attempts were retained at all gallery sizes. The two known and one
unknown extraction failures were retained. Additional enrolled identities served
only as background candidates. The ten-identity and full 55-identity galleries
were each scored once; five seeded 25-identity compositions sampled 15 background
identities from the remaining 45 development-known identities. All used the same
few-photo templates and previous 55-identity thresholds.

Both checkpoints retained 26/28 known Rank-1 matches across all compositions.
ArcFace correct acceptance stayed 23/28 for both rules. AdaFace's margin rule
changed from 24/28 at ten identities to 23/28 at 25 and 55 identities. Unknown
false-accept counts depended on the rule and composition. ArcFace score-only
false accepts changed from zero at ten identities to one at 55. AdaFace's fixed
margin rule changed from two to zero because additional gallery competitors can
increase ambiguity. Five composition variations do not supply five independent
participant samples; these results describe this small fixed core only.

## Runtime and engineering relevance

Warm CPU timings used two inference threads per engine and the actual persistent
gallery implementation. There were 30 end-to-end enrollment calls and 60
image-to-decision calls per checkpoint, plus 300 search/decision calls at each of
three gallery sizes. Enrollment included file decode, detection, alignment,
quality estimation, embedding, template normalization and persistent insertion
into a 55-entry isolated gallery. Recognition included file decode through the
four-state decision. The restoration of the enrollment benchmark gallery was
outside its timed interval.

ArcFace's median/p95 enrollment time was {float(rt('arcface','enrollment','enrollment_end_to_end')['median_ms']):.2f}/
{float(rt('arcface','enrollment','enrollment_end_to_end')['p95_ms']):.2f} ms, and image-to-decision time was
{float(rt('arcface','recognition','recognition_end_to_end')['median_ms']):.2f}/
{float(rt('arcface','recognition','recognition_end_to_end')['p95_ms']):.2f} ms. AdaFace's corresponding recognition
time was {float(rt('adaface','recognition','recognition_end_to_end')['median_ms']):.2f}/
{float(rt('adaface','recognition','recognition_end_to_end')['p95_ms']):.2f} ms. Actual ArcFace gallery search at 55
identities had median/p95 {float(rt('arcface','microbenchmark','gallery_search')['median_ms']):.4f}/
{float(rt('arcface','microbenchmark','gallery_search')['p95_ms']):.4f} ms. This separates small-gallery search cost
from image processing, which dominates the measured path.

The timing comparison includes different pretrained checkpoints and different
runtime engines; it does not isolate a loss function. Model loading, UI transfer,
camera capture and network transport were excluded, so these values are neither
browser latency nor validated camera throughput. All repetitions are latency
measurements on fixed images, not extra accuracy observations.

All policies, conditions, outcomes and negative findings are retained in the
[full evidence report](README.md), alongside raw scores and separate verification.
'''
    (dest/'thesis_section.md').write_text(chapter,encoding='utf-8')
    latex=['\\begin{tabular}{llrr}','\\hline','Checkpoint & Enrollment & Correct known & False unknown accepts \\\\', '\\hline']
    for model in cfg['models']:
        for policy in policies:
            r=a(model,policy,'score_only')
            latex.append(f"{model} & {POLICIES[policy]} & {r['correct_known_accepts']}/154 & {r['false_known_unknowns']}/149 \\\\")
    latex+=['\\hline','\\end{tabular}']
    (dest/'augmentation_table.tex').write_text('\n'.join(latex)+'\n',encoding='utf-8')
    (dest/'render_receipt.json').write_text(json.dumps({'run':rel,'verification_sha256':sha(run/'verification.json'),
        'renderer_sha256':sha(Path(__file__)),'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='render_receipt.json'}},indent=2)+'\n',encoding='utf-8')
    print('Created evidence pack:',dest.relative_to(ROOT))


if __name__=='__main__': main()
