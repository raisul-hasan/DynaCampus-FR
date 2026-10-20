"""Create static paper figures and captions from a completed development stress run."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.render_enrollment_evidence import arrow, box, save_figure
from scripts.run_arcface_baseline import read_csv, sha256

FAMILIES = ['score_only', 'score_margin']
LABELS = {'score_only':'Score only', 'score_margin':'Score + margin'}
COLORS = {'score_only':'#2878A0', 'score_margin':'#D18432'}
GROUPS = [('blur', 'Gaussian blur', ['Original', '0.003', '0.006', '0.012']),
          ('resolution', 'Lower resolution', ['Original', '128', '64', '32']),
          ('jpeg', 'JPEG recompression', ['Original', '70', '30', '10'])]
XLABELS = ['Sigma / original short edge', 'Maximum long edge (pixels)', 'JPEG encoder quality']


def ordered(summaries, kind, family):
    return [next(r for r in summaries if r['degradation'] == 'original' and r['family'] == family)] + [r for r in summaries if r['degradation'] == kind and r['family'] == family]


def method_plot(destination):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set(xlim=(0, 7.4), ylim=(0, 4.6))
    ax.axis('off')
    ax.text(3.7, 4.33, 'Controlled robustness protocol: development probes only', ha='center', fontsize=12)
    xs = [1., 2.8, 4.6, 6.4]
    texts = ['Same 304 probes\n154 masked known\n150 mixed unknown', 'One change at a time\nBlur OR resolution\nOR JPEG quality', 'Same SCRFD\nSame alignment\nFrozen ArcFace', 'Fixed score / margin\nNo retuning\nSaved decisions']
    for x, text in zip(xs, texts):
        box(ax, x, 2.86, text, width=1.62, height=1.02)
    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left+.86, 2.86), (right-.86, 2.86))
    box(ax, 4.6, 1.24, 'Unchanged gallery\n55 identities / 152 images', width=2.7, height=.75, fill='#E4EFE6')
    arrow(ax, (6.0, 1.24), (6.4, 2.31))
    ax.text(1.8, 1.24, '3 levels per degradation\n9 transformed conditions\n+ original-image control', ha='center', va='center', fontsize=10, linespacing=1.5)
    ax.text(3.7, .25, 'Transforms run before detection. Originals are preserved. Reserved test identities are excluded.', ha='center', fontsize=9)
    fig.tight_layout(pad=.3)
    save_figure(fig, destination, 'fig00_stress_protocol')


def recognition_plot(destination, summaries):
    fig, axes = plt.subplots(1, 3, figsize=(10.3, 4.7))
    for ax, (kind, title, ticks), xlabel in zip(axes, GROUPS, XLABELS):
        for family in FAMILIES:
            rows = ordered(summaries, kind, family)
            ax.plot(range(4), [100*r['correct_known_accept_end_to_end'] for r in rows],
                    linestyle='-' if family == 'score_only' else '--', marker='o' if family == 'score_only' else 's',
                    color=COLORS[family], label=LABELS[family], linewidth=1.8, markersize=5)
        ax.set(xticks=range(4), xticklabels=ticks, ylim=(0, 103), title=title, xlabel=xlabel)
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x', labelsize=9)
    axes[0].set_ylabel('Correct known acceptance (%)')
    fig.suptitle('Recognition under isolated synthetic degradations', fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5,.105), ncol=2, frameon=False)
    fig.text(.5, .025, 'Fixed original-development thresholds and clean gallery. Detection failures count as unsuccessful recognition.\nSame 154 known probes in each condition; these are paired versions, not additional independent samples.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.07, right=.99, top=.83, bottom=.30, wspace=.28)
    save_figure(fig, destination, 'fig01_stress_recognition')


def failures_plot(destination, summaries):
    fig, axes = plt.subplots(2, 3, figsize=(10.3, 7.2))
    max_false = max(r['false_known_unknowns'] for r in summaries)
    for column, ((kind, title, ticks), xlabel) in enumerate(zip(GROUPS, XLABELS)):
        top, bottom = axes[:, column]
        rows = ordered(summaries, kind, 'score_margin')
        top.plot(range(4), [100*r['extraction_success_rate'] for r in rows], '-o', color='#677D68', linewidth=1.8)
        for x, row in enumerate(rows):
            top.annotate(str(row['extraction_successes']), (x, 100*row['extraction_success_rate']), xytext=(0, -16), textcoords='offset points', ha='center', fontsize=8)
        top.set(title=title, ylim=(0, 105))
        for family in FAMILIES:
            rows = ordered(summaries, kind, family)
            bottom.plot(range(4), [r['false_known_unknowns'] for r in rows],
                        linestyle='-' if family == 'score_only' else '--', marker='o' if family == 'score_only' else 's',
                        color=COLORS[family], label=LABELS[family], linewidth=1.8, markersize=5)
        bottom.set(ylim=(-.2, max(2, max_false+1)), xlabel=xlabel)
        bottom.yaxis.set_major_locator(MaxNLocator(integer=True))
        for ax in [top, bottom]:
            ax.set(xticks=range(4), xticklabels=ticks)
            ax.tick_params(axis='x', labelsize=9)
            ax.grid(alpha=.2)
            ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('Extraction success (%)')
    axes[1,0].set_ylabel('False known count')
    fig.suptitle('Extraction failures and false acceptance under degradation', fontsize=13)
    fig.legend(*axes[1,0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5,.075), ncol=2, frameon=False)
    fig.text(.5, .02, 'Top: labels are successful extraction counts / 304; shared by both decision rules.\nBottom: absolute false-known counts over the same 150 attempted unknown probes. Conditional FPIR is saved in the table.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.075, right=.99, top=.88, bottom=.20, hspace=.30, wspace=.28)
    save_figure(fig, destination, 'fig02_stress_failures')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--output', default='paper_assets/development/robustness_v1')
    args = parser.parse_args()
    run, destination = ROOT/args.run, ROOT/args.output
    info = json.loads((run/'comparison.json').read_text())
    if info['final_test_evaluated'] or info['thresholds_retuned']:
        raise ValueError('Expected fixed-setting development results')
    destination.mkdir(parents=True, exist_ok=True)
    summaries = info['summaries']
    method_plot(destination)
    recognition_plot(destination, summaries)
    failures_plot(destination, summaries)
    for name in ['summary.csv', 'paired_effects.csv']:
        shutil.copyfile(run/name, destination/name)
    selected = [r for r in summaries if r['family'] == 'score_margin']
    method_deltas = [r['correct_known_accepts'] - next(s['correct_known_accepts'] for s in summaries
                     if s['family'] == 'score_only' and s['condition'] == r['condition']) for r in selected[1:]]
    relative_note = (f'Across the nine transformed conditions, the margin rule accepts more known probes than score only '
                     f'in {sum(d > 0 for d in method_deltas)} conditions, the same number in '
                     f'{sum(d == 0 for d in method_deltas)}, and fewer in {sum(d < 0 for d in method_deltas)}. '
                     f'These are comparisons of the two fixed calibrated rules; neither is assumed to dominate every condition.')
    lines = ['| Condition | Correct known / 154 | Extracted / 304 | False known / extracted unknown | Wrong known identity |',
             '|---|---:|---:|---:|---:|']
    for r in selected:
        lines.append(f'| {r["condition"]} | {r["correct_known_accepts"]} ({100*r["correct_known_accept_end_to_end"]:.2f}%) | '
                     f'{r["extraction_successes"]} | {r["false_known_unknowns"]}/{r["unknown_with_embedding"]} | {r["wrong_known_identity_accepts"]} |')
    latex = [r'\begin{tabular}{lrrrr}', r'\hline', r'Condition & Correct known /154 & Extraction /304 & False known / extracted unknown & Wrong known ID \\', r'\hline']
    for r in selected:
        label = r['condition'].replace('_', r'\_')
        latex.append(f'{label} & {r["correct_known_accepts"]} & {r["extraction_successes"]} & {r["false_known_unknowns"]}/{r["unknown_with_embedding"]} & {r["wrong_known_identity_accepts"]} ' + r'\\')
    latex += [r'\hline', r'\end{tabular}', '% Score + margin; fixed original-development thresholds; synthetic development stress test.']
    (destination/'robustness_table.tex').write_text('\n'.join(latex)+'\n', encoding='utf-8')
    original = selected[0]
    worst = min(selected[1:], key=lambda r:r['correct_known_accepts'])
    best = max(selected[1:], key=lambda r:r['correct_known_accepts'])
    paragraph = (f'With enrollment templates and decision thresholds fixed, the score-plus-margin rule correctly accepted '
                 f'{original["correct_known_accepts"]}/154 masked known probes ({100*original["correct_known_accept_end_to_end"]:.2f}%) '
                 f'on original development images. Across nine isolated synthetic degradation conditions, correct known acceptance '
                 f'ranged from {worst["correct_known_accepts"]}/154 ({100*worst["correct_known_accept_end_to_end"]:.2f}%) '
                 f'to {best["correct_known_accepts"]}/154 ({100*best["correct_known_accept_end_to_end"]:.2f}%). '
                 f'The lowest result occurred under {worst["condition"]}, with {worst["extraction_successes"]}/304 successful '
                 f'probe extractions. Detection failures were retained in the end-to-end denominator. These paired synthetic '
                 f'development stress tests do not measure deployment performance on actual CCTV footage.')
    report = f'''# Controlled robustness: development evidence

The same **304 probes** were evaluated under **nine isolated synthetic conditions**,
plus an original-image control. The gallery, models and thresholds stayed fixed.
The original control exactly reproduces the preceding decision comparison.

## Main results: score + margin

{chr(10).join(lines)}

{relative_note}

Correct known acceptance includes all 154 masked known attempts, including
extraction failures. Unknown inputs are the same 150 mixed-mask source probes in
every condition. Conditional FPIR divides false known acceptances by successfully
extracted unknown probes; its denominator changes when detection fails. The CSV
also reports false acceptance over all 150 attempted unknown probes, extraction
counts, rank-1 results, wrong enrolled identities, quality and decision coverage.
The original-development 1% calibration target is not guaranteed under these
transformations; report observed error counts even when that target is exceeded.

## Exact experiment

- 55 enrolled identities, 152 original enrollment images; 27 unknown identities.
- Nine conditions chosen before degraded-image inference; transformations are
  separate, never combined. All probes receive the same condition in each run.
- **Blur:** Gaussian sigma = 0.003, 0.006 or 0.012 times each source image's short
  edge, with reflected borders. This is relative to crop size, not a claimed
  camera motion model. It is Gaussian blur, not physical motion blur.
- **Resolution:** downsample the long edge to at most 128, 64 or 32 pixels using
  INTER_AREA, preserve aspect ratio, and never upscale a smaller source. The
  detector's normal resize still runs afterward; no lost detail is restored.
- **JPEG:** re-encode each decoded source at quality 70, 30 or 10, then decode once.
  Sources may already be JPEG. These settings are encoder parameters, not known
  camera bitrate or PSNR levels.
- Transformations run **before** the same neutral padding, SCRFD, alignment,
  ArcFace and exact cosine search. New embeddings are real model outputs.
- Score-only threshold 0.3978477716; score-plus-margin threshold 0.3778589368 and
  margin 0.15. No degradation-specific calibration. Quality gate stays disabled
  as selected in the earlier development experiment.
- No transformed face images are written. Reproduction uses original checksums,
  configuration, decoded-pixel checksums, source versions and ignored caches.
- Nine times 304 = **2,736 new image attempts**. The clean control uses cached
  original features; 10 conditions x 304 probes x 2 rules = 6,080 saved decisions.

## Figure captions

1. **[Protocol diagram](fig00_stress_protocol.png)** ([SVG](fig00_stress_protocol.svg)).
   Isolated synthetic degradation of the same development probes, followed by
   the unchanged recognition pipeline, clean gallery and fixed decision rules.
   Original data are preserved and reserved test identities are excluded.
2. **[Recognition graph](fig01_stress_recognition.png)** ([SVG](fig01_stress_recognition.svg)).
   End-to-end correct known acceptance under three blur, resolution and JPEG
   settings, with the same original control repeated for context. Each point
   uses the same 154 masked known attempts; failed extractions are unsuccessful
   recognition. The rule thresholds are fixed from original development data.
3. **[Extraction and false acceptance](fig02_stress_failures.png)** ([SVG](fig02_stress_failures.svg)).
   Extraction success across all 304 probe attempts (top), with successful
   extraction counts labelled, and false known acceptances across the same 150
   attempted unknown probes (bottom). Extraction is shared by both rules. These
   counts keep failure effects visible when conditional FPIR denominators change.

Numeric graph sources: [summary.csv](summary.csv). [LaTeX table](robustness_table.tex).
Paired degradation-minus-original uncertainty: [paired_effects.csv](paired_effects.csv).
All uncertainty intervals resample **known identities**, not synthetic copies as
independent images (2,000 resamples, seed 42). They condition on fixed thresholds
and omit threshold-selection, unknown-sample and model-pretraining uncertainty.
They are descriptive; no multiple-comparison significance claim is made.

## Suggested paper paragraph

{paragraph}

## Limits and verification

These are development stress tests of public face crops; the same development
identities supplied earlier calibration. A synthetic condition can occasionally
improve a particular probe through changed detection/alignment. Report observed
values without assuming degradation is strictly monotonic or choosing new
thresholds to improve them. Derived images do not increase the number of people.
No Bangladesh-wide or real CCTV generalization claim is supported. Reserved test
identities remain untouched; no model has been trained or adapted in this step.

Independent verification recomputes every successful cosine search from saved
real embeddings, every decision and all summary counts, checks fixed prototypes
and thresholds, and hashes all original probe files again. The five targeted
protocol tests cover transforms, source preservation and failure denominators.
No participant portraits or biometric features are copied into this asset folder.

Raw evidence: `{args.run}`. [Source and asset hashes](asset_manifest.json).

```powershell
python scripts/run_robustness.py
python scripts/verify_robustness_run.py {args.run}
python scripts/render_robustness_evidence.py {args.run}
python -m unittest discover -s tests -p test_robustness_protocol.py -v
```

Figures are 300 dpi PNG plus editable SVG. Full per-probe predictions, extraction
outcomes, source checksums, model hashes and configuration snapshots stay in the
raw run. Embeddings remain under ignored `data/embeddings/`.
'''
    (destination/'README.md').write_text(report, encoding='utf-8')
    manifest = dict(source_run=args.run, final_test_evaluated=False, thresholds_retuned=False,
                    source_sha256={p:sha256(run/p) for p in ['comparison.json','summary.csv','paired_effects.csv','verification.json']},
                    renderer_sha256=sha256(__file__),
                    assets_sha256={p.name:sha256(p) for p in destination.iterdir() if p.is_file() and p.name != 'asset_manifest.json'})
    (destination/'asset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('PAPER ASSETS', destination)


if __name__ == '__main__':
    main()
