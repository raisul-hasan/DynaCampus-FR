"""Generate paper figures, captions and tables from an existing enrollment run.

No inference, threshold tuning or final-test access is performed by this script.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_arcface_baseline import read_csv, sha256, write_csv

COLORS = {'one_photo': '#2878A0', 'up_to_three_photos': '#D18432'}
LABELS = {'one_photo': 'One photo', 'up_to_three_photos': 'Up to three photos'}


def save_figure(fig, directory, stem):
    fig.savefig(directory / f'{stem}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(directory / f'{stem}.svg', bbox_inches='tight', facecolor='white', metadata={'Date': None})
    plt.close(fig)


def box(ax, x, y, text, width=1.45, height=0.80, fill='#EDF3F7'):
    ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                 boxstyle='round,pad=0.04,rounding_size=0.04', linewidth=0.8,
                 edgecolor='#59636C', facecolor=fill))
    ax.text(x, y, text, ha='center', va='center', fontsize=9.5, color='#18242D', linespacing=1.4)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=12,
                               color='#4F5962', linewidth=1))


def method_diagram(directory):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set(xlim=(0, 7.2), ylim=(0, 4.4))
    ax.axis('off')
    ax.text(3.6, 4.18, 'Enrollment comparison: same frozen model, same probes', ha='center', fontsize=12)
    xs = [0.9, 2.7, 4.5, 6.3]
    box(ax, xs[0], 3.05, 'One frontal photo\nOR up to three\nunmasked photos')
    box(ax, xs[1], 3.05, 'SCRFD + alignment\nFrozen ArcFace\n512-D features')
    box(ax, xs[2], 3.05, 'Normalized mean\nof enrollment\nfeatures')
    ax.text(xs[3], 3.05, '55 enrolled IDs\n27 unknown IDs\nDevelopment only', ha='center', va='center', fontsize=9.5, linespacing=1.5)
    arrow(ax, (1.68, 3.05), (1.92, 3.05))
    arrow(ax, (3.48, 3.05), (3.72, 3.05))
    box(ax, xs[0], 1.35, 'Same 154 masked\nknown probes +\n150 unknown probes', fill='#F5F5F3')
    box(ax, xs[1], 1.35, 'Same preprocessing\nand frozen\nArcFace model', fill='#F5F5F3')
    box(ax, xs[2], 1.35, 'Exact cosine search\nagainst 55\nidentity templates', fill='#F5F5F3')
    box(ax, xs[3], 1.35, 'Score threshold\nKnown / unknown\nSaved predictions', fill='#F5F5F3')
    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left + 0.78, 1.35), (right - 0.78, 1.35))
    arrow(ax, (4.5, 2.60), (4.5, 1.80))
    ax.text(4.65, 2.20, 'Gallery', fontsize=8.5, va='center')
    ax.text(3.6, 0.42, 'No retraining. Enrollment images never enter the common probe set.', ha='center', fontsize=9.5)
    ax.text(3.6, 0.12, '82 reserved halfFace test identities remain untouched.', ha='center', fontsize=9, color='#59636C')
    fig.tight_layout(pad=0.4)
    save_figure(fig, directory, 'fig01_enrollment_method')


def split_diagram(directory):
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.set(xlim=(0, 7.2), ylim=(0, 4.9))
    ax.axis('off')
    ax.text(3.6, 4.65, 'halfFace data preparation and frozen identity split', ha='center', fontsize=12)
    box(ax, 3.6, 3.85, 'Original full-face collection: 941 images / 166 source IDs', width=5.7, height=.48)
    box(ax, 3.6, 2.86, 'Exclude 12 files under conflicting IDs + 3 unclear-label files\nUsable collection: 926 images / 164 source IDs', width=5.7, height=.62)
    arrow(ax, (3.6, 3.55), (3.6, 3.23))
    box(ax, 1.95, 1.72, 'Development: 82 IDs\n55 enrolled + 27 unenrolled', width=2.8, height=.72)
    box(ax, 5.25, 1.72, 'Reserved test: 82 IDs\n55 enrolled + 27 unenrolled\nNot evaluated', width=2.8, height=.72, fill='#F5F5F3')
    arrow(ax, (2.6, 2.5), (1.95, 2.13))
    arrow(ax, (4.6, 2.5), (5.25, 2.13))
    box(ax, 1.95, .55, '154 masked known probes\n150 mixed-mask unknown probes', width=2.8, height=.66)
    arrow(ax, (1.95, 1.3), (1.95, .94))
    ax.text(5.25, .55, 'Originals preserved.\nNo augmented copies used.\nSource IDs stay disjoint.', ha='center', va='center', fontsize=9.5, linespacing=1.5)
    fig.tight_layout(pad=.4)
    save_figure(fig, directory, 'fig00_data_protocol')


def comparison_plot(summaries, directory):
    hf = [r for r in summaries if r['dataset'] == 'halfface']
    fig, ax = plt.subplots(figsize=(7.2, 5.1))
    x = np.arange(2)
    for offset, row in zip([-.18, .18], hf):
        values = [100 * row['rank1_end_to_end'], 100 * row['correct_known_accept_end_to_end']]
        counts = [row['rank1_correct'], row['correct_known_accepts']]
        bars = ax.bar(x + offset, values, .34, label=LABELS[row['strategy']], color=COLORS[row['strategy']])
        for bar, value, count in zip(bars, values, counts):
            ax.text(bar.get_x() + bar.get_width()/2, value + 1.2, f'{value:.2f}%\n({count}/154)',
                    ha='center', va='bottom', fontsize=9)
    ax.set(ylim=(0, 114), ylabel='Known probes (%)', xticks=x,
           xticklabels=['Correct top match', 'Correct known acceptance\nafter score threshold'],
           title='Enrollment comparison on halfFace development probes')
    ax.set_yticks(np.arange(0, 101, 20))
    ax.spines[['top', 'right']].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, .12),
               ncol=2, frameon=False, fontsize=9.5)
    fig.text(.5, .025, 'Same 154 masked known probes. Each method: 1 false known / 149 extracted unknown probes.\nSix known-probe detection failures remain in the denominator.',
             ha='center', va='bottom', fontsize=8.5)
    fig.subplots_adjust(left=.10, right=.98, top=.88, bottom=.31)
    save_figure(fig, directory, 'fig02_enrollment_comparison')


def open_set_plot(predictions, summaries, directory):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.1))
    curves = []
    for strategy in COLORS:
        rows = [r for r in predictions if r['dataset'] == 'halfface' and r['strategy'] == strategy]
        unknown = np.array([float(r['top1_score']) for r in rows if r['truth_status'] == 'unknown' and r['status'] == 'ok'])
        known = [r for r in rows if r['truth_status'] == 'known']
        scores = np.array([float(r['top1_score']) if r['status'] == 'ok' else -np.inf for r in known])
        correct = np.array([r['rank1_correct'] == 'True' for r in known])
        levels = np.unique(np.concatenate([unknown, scores[np.isfinite(scores)]]))
        thresholds = np.r_[np.nextafter(levels[-1], np.inf), levels[::-1], np.nextafter(levels[0], -np.inf)]
        fpir = np.array([np.mean(unknown >= t) for t in thresholds])
        tpir = np.array([np.sum(correct & (scores >= t)) / len(known) for t in thresholds])
        for t, f, p in zip(thresholds, fpir, tpir):
            curves.append({'strategy': strategy, 'threshold': float(t), 'fpir_conditional': float(f), 'correct_known_accept_end_to_end': float(p)})
        summary = next(r for r in summaries if r['dataset'] == 'halfface' and r['strategy'] == strategy)
        for ax in axes:
            ax.step(fpir*100, tpir*100, where='post', label=LABELS[strategy], color=COLORS[strategy], linewidth=1.7)
            ax.scatter(summary['fpir_conditional']*100, summary['correct_known_accept_end_to_end']*100,
                       color=COLORS[strategy], marker='o' if strategy == 'one_photo' else 's', s=35, zorder=5)
    for ax, xmax, title in zip(axes, [100, 10], ['Full range', 'Low false-identification range']):
        ax.set(xlim=(0, xmax), ylim=(0, 100), xlabel='FPIR among extracted unknown probes (%)', title=title)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    axes[0].set_ylabel('Correct known acceptance, including failures (%)')
    axes[0].legend(loc='lower right', frameon=False, fontsize=9)
    axes[1].axvline(1, linestyle=':', color='#555555', linewidth=1)
    fig.suptitle('Open-set trade-off on development data', fontsize=12)
    fig.text(.5, .01, 'Markers show development-calibrated operating points; no reserved test identities were evaluated.', ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .05, 1, .98))
    save_figure(fig, directory, 'fig03_open_set_tradeoff')
    write_csv(directory / 'open_set_curve_points.csv', curves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    run = (ROOT / args.run_directory).resolve()
    data = json.loads((run / 'comparison.json').read_text())
    if data['final_test_evaluated']:
        raise ValueError('This renderer is labeled for development evidence only')
    summaries = data['summaries']
    predictions = read_csv(run / 'predictions.csv')
    directory = ROOT / 'paper_assets' / 'development' / 'enrollment_v1'
    directory.mkdir(parents=True, exist_ok=True)
    previous = directory / 'asset_manifest.json'
    if previous.exists() and json.loads(previous.read_text())['source_run'] != data['run_directory']:
        raise ValueError('Use a new asset version for a different experiment run')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': 'dynacampus-enrollment-v1'})
    split_diagram(directory)
    method_diagram(directory)
    comparison_plot(summaries, directory)
    open_set_plot(predictions, summaries, directory)
    for filename in ['comparison.csv', 'paired_bootstrap.csv']:
        (directory / filename).write_bytes((run / filename).read_bytes())
    known = {strategy: {r['path']: r for r in predictions if r['dataset'] == 'halfface' and r['strategy'] == strategy and r['truth_status'] == 'known'} for strategy in COLORS}
    one, few = known['one_photo'], known['up_to_three_photos']
    gains = sum(one[p]['correct_known_accept'] == 'False' and few[p]['correct_known_accept'] == 'True' for p in one)
    losses = sum(one[p]['correct_known_accept'] == 'True' and few[p]['correct_known_accept'] == 'False' for p in one)
    hf = [s for s in summaries if s['dataset'] == 'halfface']
    effect = next(e for e in data['paired_effects'] if e['dataset'] == 'halfface' and e['metric'] == 'correct_known_accept')
    report = ['# Enrollment comparison: development evidence', '',
        'These results are development observations, not final held-out performance.', '',
        '| Method | Enrollment images | Correct top match | Correct known acceptance | False known / extracted unknown |',
        '|---|---:|---:|---:|---:|']
    for row in hf:
        report.append(f'| {LABELS[row["strategy"]]} | {row["gallery_images"]} | {row["rank1_correct"]}/154 ({100*row["rank1_end_to_end"]:.2f}%) | {row["correct_known_accepts"]}/154 ({100*row["correct_known_accept_end_to_end"]:.2f}%) | {row["false_known_count"]}/149 |')
    report += ['', f'The few-photo method helped {gains} known probes and hurt {losses}, a net gain of {gains-losses}. Top-match correctness did not change.',
        f'The paired identity-bootstrap difference in correct acceptance was {effect["delta_percentage_points"]:.2f} percentage points (descriptive 95% interval {effect["ci95_low_percentage_points"]:.2f} to {effect["ci95_high_percentage_points"]:.2f}). This conditions on development-calibrated thresholds and excludes calibration/unknown-sample uncertainty.', '',
        '## Protocol and limitations', '',
        '- Both methods use the same 55 enrolled identities, 154 masked known probes and 150 unknown probe attempts. One unknown detection fails, leaving 149 scores for empirical calibration.',
        '- One-photo enrollment uses the original frontal unmasked photo. Few-photo enrollment averages up to three original unmasked views and L2-normalizes the mean. Actual enrollment counts are 47 identities with three photos, three with two and five with one.',
        '- Both methods have six known-probe detection failures; these remain errors in the reported rates.',
        '- Extra enrollment photos are excluded from probes for both methods. The earlier mixed-mask baseline used 251 known probes; its 96.81% is not directly comparable with this masked-only 154-probe experiment.',
        '- Each method is separately calibrated to at most 1% empirical conditional FPIR using the same development unknown probes. Both accept one of 149; this is not a guaranteed real-world false-identification rate.',
        '- A separate ICFD engineering check uses five sample identities and the same 100 classroom probes: one-photo and three-photo methods both retrieve 100/100 correctly. This ceiling result is not a full ICFD benchmark.',
        '- No model retraining, adaptation, augmentation or quality gate was used. Reserved test identities remain untouched. Near-duplicate and pretraining overlap limitations remain.', '',
        '## Figure captions', '',
        '**Figure 00. Data preparation and identity split.** The original halfFace full-face collection is filtered using documented duplicate-identity and label checks. Development and reserved-test source identities are disjoint. Only development data enter the current comparison.', '',
        '**Figure 01. Enrollment comparison pipeline.** Original one-photo or up-to-three-photo enrollment media pass through the same frozen detection, alignment and ArcFace encoder. Normalized mean identity templates are matched against a common probe set without updating model weights.', '',
        '**Figure 02. Enrollment results on masked development probes.** Bars compare correct top-match retrieval and correct acceptance after a score threshold. Denominators include six failed detections. Each method accepts one of 149 successfully embedded development unknown probes.', '',
        '**Figure 03. Development open-set trade-off.** Curves show correct known acceptance including failed detections versus false-positive identification among successfully embedded unknowns. Markers indicate the separately calibrated operating points; the right panel enlarges the low-FPIR range.', '',
        '## Suggested paper wording', '',
        'In a development-only enrollment ablation, one frontal unmasked image was compared with a normalized mean template formed from up to three unmasked views. Both strategies were evaluated on the same 154 masked known probes against 55 enrolled identities, with 150 probe attempts from 27 unenrolled identities. Correct top-match retrieval remained 146/154 (94.81%). After separate development calibration to a target empirical FPIR of 1%, correct known acceptance increased from 129/154 (83.77%) to 138/154 (89.61%); both methods falsely accepted one of 149 successfully embedded unknown probes. These observations require confirmation on the reserved test identities.', '',
        '## Files and reproduction', '',
        f'Source run: {data["run_directory"]}',
        f'Run comparison: python scripts/compare_enrollment.py',
        f'Render this evidence: python scripts/render_enrollment_evidence.py {data["run_directory"]}',
        'Figures are supplied as 300-dpi PNG and editable vector SVG. Tables preserve unrounded numeric data in CSV. Raw predictions and enrollment-source records remain in the source run directory.', '']
    (directory / 'README.md').write_text('\n'.join(report), encoding='utf-8')
    latex = ['% Development-only halfFace enrollment ablation. Not final held-out performance.',
             r'\begin{tabular}{lrrrr}', r'\hline',
             r'Enrollment & Images & Rank-1 (\%) & Correct acceptance (\%) & FP / N \\', r'\hline']
    for row in hf:
        latex.append(f'{LABELS[row["strategy"]]} & {row["gallery_images"]} & {100*row["rank1_end_to_end"]:.2f} & {100*row["correct_known_accept_end_to_end"]:.2f} & 1/149 ' + r'\\')
    latex += [r'\hline', r'\end{tabular}', '% Known denominator: 154 masked development probes; identical in both methods.']
    (directory / 'enrollment_table.tex').write_text('\n'.join(latex) + '\n', encoding='utf-8')
    manifest = {'source_run': data['run_directory'], 'scope': 'development_only',
                'source_predictions_sha256': sha256(run / 'predictions.csv'),
                'comparison_json_sha256': sha256(run / 'comparison.json'), 'renderer_sha256': sha256(__file__),
                'assets': {p.name: sha256(p) for p in sorted(directory.iterdir()) if p.is_file() and p.name != 'asset_manifest.json'}}
    (directory / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(directory)
    print('\n'.join(report[:12]))


if __name__ == '__main__':
    main()
