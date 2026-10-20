"""Render publication figures and captions from a verified development run."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_arcface_baseline import read_csv, sha256
from scripts.render_enrollment_evidence import save_figure

COLORS = {'arcface': '#2878A0', 'adaface': '#D18432'}
NAMES = {'arcface': 'ArcFace / buffalo_l', 'adaface': 'AdaFace IR-50 / MS1MV2'}
POLICIES = [('one_photo', 'score_only'), ('up_to_three_photos', 'score_only'),
            ('one_photo', 'score_margin'), ('up_to_three_photos', 'score_margin')]
LABELS = ['One photo\nScore', 'Up to three\nScore', 'One photo\nScore + margin', 'Up to three\nScore + margin']


def comparison(destination, rows):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.1), gridspec_kw={'width_ratios': [1.35, 1]})
    x, width = np.arange(len(POLICIES)), .35
    for number, model in enumerate(COLORS):
        selected = [next(r for r in rows if (r['model'], r['strategy'], r['family']) == (model, strategy, family)) for strategy, family in POLICIES]
        at = x + (number - .5) * width
        rates = [100 * r['correct_known_accepts'] / r['known_probe_attempts'] for r in selected]
        false = [r['false_known_unknowns'] for r in selected]
        axes[0].bar(at, rates, width=width, color=COLORS[model], label=NAMES[model])
        axes[1].bar(at, false, width=width, color=COLORS[model])
        for pos, value, row in zip(at, rates, selected):
            axes[0].text(pos, value + 1.6, f"{row['correct_known_accepts']}/{row['known_probe_attempts']}", ha='center', fontsize=8, rotation=90)
        for pos, value, row in zip(at, false, selected):
            axes[1].text(pos, value + .06, str(value), ha='center', fontsize=9)
    axes[0].set(ylabel='Correct known acceptance (%)', ylim=(0, 116), title='Same 154 known attempts; failures retained')
    axes[0].set_yticks(range(0, 101, 20))
    axes[1].set(ylabel='False known acceptances (count)', ylim=(0, 2.5), title='Same 149 extracted unknown probes')
    axes[1].set_yticks([0, 1, 2])
    for ax in axes:
        ax.set_xticks(x, LABELS, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
    fig.suptitle('Paired backbone comparison: halfFace development data', fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .93), ncol=2, frameon=False, fontsize=9)
    fig.text(.5, .025, 'Calibration and evaluation reuse development images. Each method has its own calibrated thresholds.\nPublic checkpoints differ in pretraining; these are descriptive results, not held-out or isolated loss-function comparisons.', ha='center', fontsize=8.5)
    fig.subplots_adjust(left=.065, right=.99, top=.77, bottom=.24, wspace=.28)
    save_figure(fig, destination, 'fig01_backbone_comparison')


def timings(destination, rows):
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.5))
    for ax, stage, title in zip(axes, ['recognizer_ms', 'composed_image_to_embedding_ms'],
                                ['Recognizer call', 'Composed image-to-embedding stages']):
        selected = [next(r for r in rows if r['model'] == model and r['dataset'] == 'halfface' and r['stage'] == stage) for model in COLORS]
        medians = [float(r['median_ms']) for r in selected]
        p95 = [float(r['p95_ms']) for r in selected]
        ax.bar(range(2), medians, color=list(COLORS.values()), width=.58, label='Median')
        ax.scatter(range(2), p95, color='#18242D', marker='D', s=25, label='95th percentile', zorder=3)
        for i, (median, upper) in enumerate(zip(medians, p95)):
            ax.text(i, upper + max(p95) * .05, f'{median:.1f} / {upper:.1f} ms', ha='center', fontsize=9)
        ax.set_xticks(range(2), ['ArcFace\nONNX Runtime', 'AdaFace\nPyTorch'])
        ax.set(ylabel='Milliseconds', title=title, ylim=(0, max(p95) * 1.3))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.16)
        ax.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('Local CPU timing on successfully extracted halfFace development images', fontsize=12)
    fig.text(.5, .025, 'Two CPU threads per model; five warm-up calls; alternating model order. Pretraining and runtime engines differ.\nComposed stages include decode, padding, detection, alignment, quality and embedding; exclude search, UI, disk hashes and model loading.', ha='center', fontsize=8.2)
    fig.subplots_adjust(left=.075, right=.98, top=.79, bottom=.25, wspace=.28)
    save_figure(fig, destination, 'fig02_backbone_timing')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--output', default='paper_assets/development/backbones_v1')
    args = parser.parse_args()
    run = ROOT / args.run
    info = json.loads((run / 'comparison.json').read_text(encoding='utf-8'))
    verification = json.loads((run / 'verification.json').read_text(encoding='utf-8'))
    if verification['verification'] != 'passed':
        raise ValueError('Verify the run before rendering')
    for name, digest in verification['artifact_sha256'].items():
        if sha256(run / name) != digest:
            raise ValueError(f'Verified source artifact changed: {name}')
    destination = ROOT / args.output
    destination.mkdir(parents=True, exist_ok=True)
    selected = [r for r in info['summaries'] if r['dataset'] == 'halfface']
    comparison(destination, selected)
    timings(destination, read_csv(run / 'timing_summary.csv'))
    for name in ['summary.csv', 'selected_thresholds.csv', 'paired_effects.csv', 'timing_summary.csv']:
        shutil.copyfile(run / name, destination / name)
    lines = ['# Paired AdaFace and ArcFace development evidence', '',
        'These comparisons use the same original images, detection outcomes and aligned crops.',
        'No held-out inference, recognition training, domain adapter or final-method replacement occurred.', '',
        '| Model | Enrollment | Decision | Correct known / 154 | False known / extracted unknown |',
        '|---|---|---|---:|---:|']
    for row in selected:
        lines.append(f"| {row['model']} | {row['strategy']} | {row['family']} | {row['correct_known_accepts']}/{row['known_probe_attempts']} | {row['false_known_unknowns']}/{row['unknown_with_embedding']} |")
    lookup = {(r['model'], r['strategy'], r['family']): r for r in selected}
    count = lambda model, strategy, family: lookup[(model, strategy, family)]['correct_known_accepts']
    arc = lookup[('arcface', 'up_to_three_photos', 'score_margin')]
    ada = lookup[('adaface', 'up_to_three_photos', 'score_margin')]
    lines += ['', '## Main finding', '',
        'With score-only decisions, few-photo enrollment changes correct known acceptance '
        f"from {count('arcface', 'one_photo', 'score_only')} to {count('arcface', 'up_to_three_photos', 'score_only')} for ArcFace "
        f"and from {count('adaface', 'one_photo', 'score_only')} to {count('adaface', 'up_to_three_photos', 'score_only')} for AdaFace. "
        'With separately calibrated score-plus-margin rules, ArcFace changes from '
        f"{count('arcface', 'one_photo', 'score_margin')} to {arc['correct_known_accepts']}, while AdaFace changes from "
        f"{count('adaface', 'one_photo', 'score_margin')} to {ada['correct_known_accepts']}. "
        'The enrollment benefit must be assessed jointly with the checkpoint and decision rule.', '',
        f"The few-photo margin comparison has {arc['correct_known_accepts']}/{arc['known_probe_attempts']} correct known acceptances and "
        f"{arc['false_known_unknowns']}/{arc['unknown_with_embedding']} false unknown accepts for ArcFace, versus "
        f"{ada['correct_known_accepts']}/{ada['known_probe_attempts']} and {ada['false_known_unknowns']}/{ada['unknown_with_embedding']} for AdaFace. "
        f"Extracted-probe decision coverage is {100*ada['valid_decision_coverage']:.2f}% for AdaFace and {100*arc['valid_decision_coverage']:.2f}% for ArcFace. "
        'Both obey the same empirical FPIR limit, but their observed FPIR and coverage are not '
        'identical. Do not interpret the acceptance counts alone as an overall superiority test.', '',
        'The original frozen final evaluation remains separate. These new development results '
        'do not replace it, and the newly calibrated one-photo margin policy was not one of its '
        'predeclared final methods. It must not be retrospectively described as a held-out method.']
    lines += ['', '## What was measured', '',
        f"Paired extraction covers {info['extraction_attempts']} development images, with {info['shared_extraction_failures']} shared capture failures. "
        f"ArcFace features reproduce {info['model']['arcface_reproduced_images']} original cached embeddings; maximum difference {info['model']['arcface_max_cache_difference']:.3g}.", '',
        'Each halfFace comparison includes 55 enrolled identities, 154 masked known attempts and 150 unknown attempts. '
        'Six known and one unknown capture failures stay in attempted denominators. '
        'Unknown FPIR uses all 149 successfully extracted unknown probes, including refused inputs.', '',
        'Each model/enrollment/decision family is calibrated independently using the same declared objective and margin grid. '
        'Threshold selection and evaluation reuse development data, so these results are descriptive rather than independent generalization estimates.', '',
        'ICFD is reported separately in summary.csv: five development identities and 100 closed-set public-sample probes; no unknown people.', '',
        '## Models and attribution', '',
        '[Official AdaFace source and checkpoints](https://github.com/mk-minchul/AdaFace): IR-50 pretrained on MS1MV2. '
        'Source commit c60eaa786a42c03444f3df7096dbaf9d57ae010d is pinned and archived locally with its MIT source license. '
        'The checkpoint has SHA-256 ' + info['model']['adaface']['sha256'] + '.', '',
        '[Official InsightFace model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo): '
        'buffalo_l ResNet50 pretrained on WebFace600K. The model is for non-commercial research. '
        'The two checkpoints have different pretraining datasets and runtime engines; this does not isolate the effect of the AdaFace training loss.', '',
        'Both use the existing SCRFD five-point alignment for a controlled crop comparison. '
        'AdaFace receives uint8 BGR crops normalized using (value/255 - 0.5)/0.5; '
        'ArcFace uses its existing InsightFace preprocessing. No flip fusion or test-time augmentation is applied.', '',
        'Real-crop preprocessing and output match the official AdaFace reference. Strict state loading is used. '
        'A legacy Lightning callback metadata key is mapped to an inert local class while keeping weights-only loading; no Lightning training code runs.', '',
        '## Figures and captions', '',
        '1. **[Development comparison](fig01_backbone_comparison.png)** ([SVG](fig01_backbone_comparison.svg)). '
        'Correct known acceptance and false-known counts for the same probes, with failures retained. '
        'Each checkpoint and policy has its own development-calibrated thresholds. '
        'This is not a held-out result or evidence of an isolated loss-function advantage.', '',
        '2. **[CPU timing](fig02_backbone_timing.png)** ([SVG](fig02_backbone_timing.svg)). '
        'Median bars and p95 markers for successful halfFace extraction, using two CPU threads and five warm-up calls. '
        'The composed latency sums measured decode, padding, detection, alignment, quality and embedding stages; '
        'it excludes search, gallery persistence, UI, model loading and source-hash checks. It is not live-video FPS. '
        'Source hashing precedes decoding, so decode timings likely benefit from a warm file cache. '
        'The local processor was identified as AMD Ryzen 5 3600; timings reflect ordinary background load.', '',
        '## Verification and reproducibility', '',
        f"The independent cached audit checks {verification['prediction_attempts']} prediction attempts, "
        f"{verification['cosine_searches']} cosine searches and {verification['calibration_grid_entries']} calibration candidates. "
        'It reconstructs templates and decisions without importing the experiment implementation.', '',
        'Six AdaFace preprocessing/loading safeguards and three paired-protocol tests pass. '
        'Biometric feature arrays, raw images and checkpoint binaries remain in ignored data folders.', '',
        '```powershell', 'python scripts/fetch_adaface_assets.py', 'python scripts/compare_backbones.py',
        '# Audit/re-render this completed run without new image inference:',
        f'python scripts/verify_backbone_comparison.py {args.run}', f'python scripts/render_backbone_evidence.py {args.run}', '```', '',
        f'Raw run: `{args.run}`. Open REPORT.md, comparison.json, predictions.csv, calibration_grid.csv, '
        'enrollment_sources.csv, paired_cases.csv, failure_cases.csv, real_model_smoke.json and verification.json there.', '',
        '## Interpretation limits', ''] + [f'- {item}' for item in info['limitations']]
    (destination / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    manifest = dict(source_run=args.run, source_verification_sha256=sha256(run / 'verification.json'),
                    renderer_sha256=sha256(Path(__file__)), asset_sha256={p.name: sha256(p) for p in destination.iterdir() if p.is_file() and p.name != 'asset_manifest.json'})
    (destination / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(destination.relative_to(ROOT).as_posix())


if __name__ == '__main__':
    main()
