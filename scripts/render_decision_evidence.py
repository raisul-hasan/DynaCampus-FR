"""Render paper artifacts from saved development decisions, without inference."""
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
from scripts.render_enrollment_evidence import arrow, box, save_figure
from scripts.run_arcface_baseline import read_csv, sha256

LABELS = ['Score only', 'Score + margin', 'Score + margin\n+ quality option']
COLORS = ['#2878A0', '#D18432', '#677D68']


def decision_diagram(destination, selected):
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    ax.set(xlim=(0, 7.2), ylim=(0, 7.1))
    ax.axis('off')
    ax.text(3.6, 6.88, 'Implemented four-state decision rule', ha='center', fontsize=13)
    steps = [(6.05, 'Detection + alignment\nReal ArcFace feature', 'Failure', 'CAPTURE FAILURE'),
             (4.90, 'Quality q below threshold?', 'Yes', 'INSUFFICIENT\nQUALITY'),
             (3.75, 'Top score s1 below threshold?', 'Yes', 'UNKNOWN'),
             (2.60, 'Top-two gap s1 - s2\nbelow margin threshold?', 'Yes', 'AMBIGUOUS')]
    for y, text, branch, state in steps:
        box(ax, 2.45, y, text, width=3.1, height=.66)
        box(ax, 5.9, y, state, width=1.95, height=.66, fill='#F5F5F3')
        arrow(ax, (4.07, y), (4.86, y))
        ax.text(4.46, y+.12, branch, ha='center', fontsize=8.5)
    for upper, lower in zip(steps[:-1], steps[1:]):
        arrow(ax, (2.45, upper[0]-.38), (2.45, lower[0]+.38))
        ax.text(2.61, (upper[0]+lower[0])/2, 'Success' if upper is steps[0] else 'No', fontsize=8.5, va='center')
    box(ax, 2.45, 1.45, 'KNOWN: accept top identity', width=3.1, height=.66, fill='#E4EFE6')
    arrow(ax, (2.45, 2.22), (2.45, 1.83))
    ax.text(2.61, 2.02, 'No', fontsize=8.5, va='center')
    ax.text(3.6, .64, f'Selected development rule: score = {selected["score_threshold"]:.4f}, margin = {selected["margin_threshold"]:.2f}, quality = 0 (disabled).',
            ha='center', fontsize=9)
    ax.text(3.6, .24, 'Capture failure is recorded separately. Refusal states never return an accepted identity.', ha='center', fontsize=9)
    fig.tight_layout(pad=.3)
    save_figure(fig, destination, 'fig01_decision_rule')


def ablation_plot(destination, summaries):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.7))
    metrics = [('correct_known_accept_end_to_end', 'Correct known acceptance', 154),
               ('valid_decision_coverage', 'Decision coverage after extraction', 297)]
    for ax, (metric, title, denominator) in zip(axes, metrics):
        values = [100*r[metric] for r in summaries]
        bars = ax.bar(np.arange(3), values, color=COLORS, width=.68)
        for bar, value, row in zip(bars, values, summaries):
            count = row['correct_known_accepts'] if denominator == 154 else sum(row['decision_counts'].get(s, 0) for s in ['KNOWN', 'UNKNOWN'])
            ax.text(bar.get_x()+bar.get_width()/2, value+1.7, f'{value:.2f}%\n{count}/{denominator}', ha='center', fontsize=9)
        ax.set(xticks=np.arange(3), xticklabels=LABELS, ylim=(0, 115), ylabel='Probes (%)', title=title)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.tick_params(axis='x', labelsize=8.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('halfFace development decision ablation', fontsize=13)
    fig.text(.5, .03, 'All methods: 1 false known / 149 extracted unknown probes. Quality option selects a zero gate.\nSame gallery and probes; seven capture failures remain outside extracted-probe coverage.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.07, right=.99, bottom=.24, top=.82, wspace=.28)
    save_figure(fig, destination, 'fig02_decision_ablation')


def sensitivity_plot(destination, rows):
    q = [float(r['quality_threshold']) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4))
    for ax, metric, title, color in zip(axes,
            ['valid_decision_coverage', 'correct_known_accept_end_to_end'],
            ['Decision coverage (297 extracted probes)', 'Correct acceptance (154 known attempts)'], COLORS):
        ax.plot(q, [100*float(r[metric]) for r in rows], color=color, linewidth=2, marker='.', markersize=4)
        ax.axvline(0, color='#677D68', linestyle=':', linewidth=1.3)
        ax.set(xlabel='Minimum quality threshold', ylabel='Probes (%)', xlim=(-.015, 1.015), ylim=(0, 105), title=title)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_title(title, fontsize=10)
    fig.suptitle('Quality filtering reduces coverage on this development sample', fontsize=12)
    fig.text(.5, .035, 'Fixed score threshold 0.3779 and margin 0.15. Quality threshold varies; no score retuning.\nSelected quality gate is zero. This is a sensitivity analysis, not a held-out result.', ha='center', fontsize=9)
    fig.subplots_adjust(left=.08, right=.99, top=.81, bottom=.27, wspace=.3)
    save_figure(fig, destination, 'fig03_quality_sensitivity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('--output', default='paper_assets/development/decision_v1')
    args = parser.parse_args()
    run, destination = ROOT / args.run, ROOT / args.output
    info = json.loads((run / 'comparison.json').read_text())
    if info['final_test_evaluated']:
        raise ValueError('This renderer labels development evidence only')
    destination.mkdir(parents=True, exist_ok=True)
    summaries = info['summaries']
    selected = info['selected_thresholds']['score_margin_quality']
    decision_diagram(destination, selected)
    ablation_plot(destination, summaries)
    sensitivity_plot(destination, read_csv(run / 'quality_sensitivity.csv'))
    for name in ['ablation.csv', 'paired_effects.csv', 'quality_sensitivity.csv']:
        shutil.copyfile(run / name, destination / name)
    latex = [r'\begin{tabular}{lrrrrrr}', r'\hline',
             r'Method & $\tau_s$ & $\tau_m$ & $\tau_q$ & Correct known (\%) & FPIR (\%) & Coverage (\%) \\', r'\hline']
    for label, r in zip(['Score only', 'Score + margin', 'Score + margin + quality option'], summaries):
        latex.append(f'{label} & {r["score_threshold"]:.4f} & {r["margin_threshold"]:.2f} & {r["quality_threshold"]:.2f} & '
                     f'{100*r["correct_known_accept_end_to_end"]:.2f} & {100*r["fpir_conditional"]:.2f} & {100*r["valid_decision_coverage"]:.2f} ' + r'\\')
    latex.extend([r'\hline', r'\end{tabular}',
                  '% Development only. Correct known: /154 attempts; FPIR: /149 extracted unknowns;',
                  '% Coverage: /297 extracted probes. Zero quality gate means disabled.'])
    (destination / 'decision_table.tex').write_text('\n'.join(latex)+'\n', encoding='utf-8')
    report = f'''# Decision comparison: development evidence

The margin rule correctly accepts **139/154 known probes (90.26%)**, versus
**138/154 (89.61%)** for score only. Both falsely accept **1/149 extracted
unknown probes (0.67%)**. The quality option selects **zero**, so it adds no
benefit under the stated objective and coverage constraint.

| Method | Correct known acceptance | False known / extracted unknown | Decision coverage / extracted probes |
|---|---:|---:|---:|
| Score only | 138/154 (89.61%) | 1/149 | 297/297 (100%) |
| Score + margin | 139/154 (90.26%) | 1/149 | 295/297 (99.33%) |
| Score + margin + quality option | 139/154 (90.26%) | 1/149 | 295/297 (99.33%) |

The margin methods refuse two unknown probes as ambiguous. Seven capture failures
(six known, one unknown) remain included in attempted-probe counts. End-to-end
decision coverage is 297/304 for score only and 295/304 for the margin methods.
No accepted known probe receives the wrong enrolled identity in this run.

## Protocol and selected thresholds

- Same frozen ArcFace features, 55 enrolled identities, 152 enrollment images,
  154 masked known probes and 150 mixed-mask unknown probes as the few-photo run.
- No additional model inference or reserved-test image access was needed.
- Jointly search score, margin and (where enabled) quality thresholds on the same
  development data. The 42,316 grid configurations are saved with feasibility.
- Require empirical unknown FPIR <= 1% over **all 149 extracted unknowns**, including
  those refused by quality or ambiguity. Require at least 85% decision coverage
  separately for extracted known and unknown probes.
- Maximize correct known acceptances; ties prefer fewer false unknown acceptances,
  fewer wrong known identities, greater coverage, lower quality/margin gates,
  then a higher similarity threshold. Zero gates are permitted.
- Score only: score 0.3978477716, margin 0, quality 0.
- Margin methods: score 0.3778589368, margin 0.15, quality 0.
- Thresholds above are research settings; dashboard sliders remain manual settings.
- Quality is the existing heuristic: 45% Laplacian sharpness, 35% native face size,
  20% landmark symmetry/tilt. It is not a learned quality model or calibrated
  probability. Sharpness is measured on the aligned crop; size uses the native bbox.

The margin method changes the calibrated score threshold as well as the margin
gate. Its gain is a comparison of calibrated rule families, not an isolated causal
effect of adding a gate at an unchanged score threshold. The earlier enrollment
step used a different threshold selection tie rule; its few-photo acceptances
match this score-only result.

## Figure captions

1. **[Decision rule](fig01_decision_rule.png)** ([editable SVG](fig01_decision_rule.svg)).
   Implemented quality, similarity and top-two margin rule. Quality failure returns
   insufficient quality; low similarity returns unknown; a small gap between the
   two best identity scores returns ambiguous; otherwise the top identity is
   accepted. Capture failure is recorded separately. The optimized development
   quality gate is disabled; the diagram shows the general supported rule.
2. **[Decision ablation](fig02_decision_ablation.png)** ([editable SVG](fig02_decision_ablation.svg)).
   Correct known acceptance and extracted-probe decision coverage on the same
   halfFace development probes. Margin calibration recovers one additional known
   probe while refusing two unknown probes. All three families have one false
   known acceptance. The quality-enabled family selects a zero quality threshold.
3. **[Quality sensitivity](fig03_quality_sensitivity.png)** ([editable SVG](fig03_quality_sensitivity.svg)).
   Coverage and end-to-end correct known acceptance as the quality threshold
   varies, keeping score 0.3779 and margin 0.15 fixed. Filtering can preserve or
   reduce these counts; this graph does not establish that the quality estimate
   improves recognition. Numeric points: [CSV](quality_sensitivity.csv).

## Suggested paper text and limits

On halfFace development probes, jointly calibrated score and margin decisions
increased correct known acceptance from 138/154 (89.61%) to 139/154 (90.26%), with
one false known acceptance among 149 extracted unknown probes in both cases.
Decision coverage decreased from 100% to 99.33% among extracted probes because two
unknown probes were marked ambiguous. Enabling quality-threshold selection yielded
a zero threshold and the same decisions, providing no incremental benefit for the
current heuristic under the chosen development objective.

The paired known-identity bootstrap difference is +0.65 percentage points, with a
descriptive 95% interval of 0.00 to 1.97 points (2,000 resamples, fixed thresholds).
This interval includes zero and excludes threshold-selection and unknown-sample
uncertainty. Calibration and evaluation reuse development data; these figures
must not be described as final held-out performance or statistically established
improvements. The small masked-face sample does not establish CCTV robustness or
Bangladesh-wide generalization. Reserved test identities remain untouched.

## Files and reproduction

- [Numeric ablation table](ablation.csv), [LaTeX table](decision_table.tex),
  [paired uncertainty](paired_effects.csv), [source manifest](asset_manifest.json).
- Raw run: `{args.run}` (input scores, predictions, entire calibration grid,
  configuration, split, source hashes and independent verification).
- Eight decision tests cover ambiguity, boundaries, invalid inputs, refusal
  denominators, single-candidate galleries and calibration behavior. Independent
  verification checks all 912 saved probe/method attempts and grid selection.
- Dashboard wiring receives both identity scores, with explicit ambiguity and
  insufficient-quality states. Syntax is checked; interactive UI testing is pending.

```powershell
python scripts/compare_decisions.py
python scripts/verify_decision_run.py {args.run}
python scripts/render_decision_evidence.py {args.run}
python -m unittest discover -s tests -p test_decision_ablation.py -v
```

Figures are 300 dpi PNG and editable SVG. They contain no participant portraits.
'''
    (destination / 'README.md').write_text(report, encoding='utf-8')
    manifest = {'source_run': args.run, 'final_test_evaluated': False,
                'source_sha256': {name: sha256(run / name) for name in
                                  ['comparison.json', 'ablation.csv', 'quality_sensitivity.csv', 'paired_effects.csv', 'verification.json']},
                'renderer_sha256': sha256(__file__),
                'assets_sha256': {p.name: sha256(p) for p in destination.iterdir() if p.is_file() and p.name != 'asset_manifest.json'}}
    (destination / 'asset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('PAPER ASSETS', destination)


if __name__ == '__main__':
    main()
