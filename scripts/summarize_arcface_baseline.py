"""Create readable tables and a development-score plot from saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_arcface_baseline import ROOT, read_csv, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    run = Path(args.run_directory).resolve()
    info = json.loads((run / 'run_info.json').read_text(encoding='utf-8'))
    extraction = read_csv(run / 'extraction.csv')
    predictions = read_csv(run / 'predictions.csv')
    split = json.loads((run / 'split_snapshot.json').read_text(encoding='utf-8'))
    percentages = lambda value: f'{value * 100:.2f}%' if value is not None else 'not evaluated'
    table = []
    lines = ['# First ArcFace baseline', '',
             '**Development / engineering results only. Final-test identities remain untouched.**', '',
             f'Run: `{info["run_id"]}`', '',
             'Real SCRFD detection, five-point alignment, ArcFace 512-dimensional normalized embeddings and exact cosine search.',
             'One original unmasked frontal image enrolls each halfFace known identity. No adapter, augmentation or quality gate is applied.', '',
             '| Dataset | Enrollment succeeded | Known probe attempts | Rank-1, including pipeline failures | Rank-1, successful embeddings with gallery mate |',
             '|---|---:|---:|---:|---:|']
    for summary in info['summaries']:
        lines.append(f'| {summary["dataset"]} | {summary["gallery_successes"]}/{summary["gallery_attempts"]} | {summary["known_probe_attempts"]} | {percentages(summary["rank1_end_to_end"])} | {percentages(summary["rank1_conditional"])} |')
        table.append({k: v for k, v in summary.items() if not isinstance(v, dict)})
    write_csv(run / 'baseline_table.csv', table)
    lines += ['', '## halfFace open-set development calibration', '']
    hf = next(s for s in info['summaries'] if s['dataset'] == 'halfface')
    lines += [f'- Development identities: {len(split["halfface"]["development"]["known"])} enrolled and {len(split["halfface"]["development"]["unknown"])} unenrolled.',
              f'- Reserved test identities: {len(split["halfface"]["test_reserved"]["known"])} enrolled and {len(split["halfface"]["test_reserved"]["unknown"])} unenrolled; none evaluated.',
              f'- Non-mated probe attempts: {hf["unknown_probe_attempts"]}; successful embeddings: {hf["unknown_with_embedding"]}.',
              f'- Similarity threshold: {hf["threshold"]:.10f}, calibrated for at most 1% empirical FPIR among successfully embedded development unknowns.',
              f'- Development conditional FPIR: {percentages(hf["fpir_conditional"])}.',
              f'- Correct known acceptance including pipeline failures: {percentages(hf["correct_known_accept_end_to_end"])}.',
              '- These calibration results are not held-out accuracy and do not establish a guaranteed population FPIR.', '',
              '## halfFace known probes by mask condition', '',
              '| Condition | Attempts | Successful embeddings | Correct rank-1 | Rank-1 including failures |',
              '|---|---:|---:|---:|---:|']
    conditions = []
    for mask in ('unmasked', 'masked'):
        subset = [r for r in predictions if r['dataset'] == 'halfface' and r['truth_status'] == 'known' and r['mask_status'] == mask]
        correct = sum(r['rank1_correct'] == 'True' for r in subset)
        row = {'condition': mask, 'attempts': len(subset), 'successful_embeddings': sum(r['status'] == 'ok' for r in subset),
               'rank1_correct': correct, 'rank1_end_to_end': correct / len(subset) if subset else None}
        conditions.append(row)
        lines.append(f'| {mask} | {len(subset)} | {row["successful_embeddings"]} | {correct} | {percentages(row["rank1_end_to_end"])} |')
    write_csv(run / 'mask_conditions.csv', conditions)
    lines += ['', '## Timing and failures', '']
    for dataset in ('halfface', 'icfd_sample'):
        subset = [r for r in extraction if r['dataset'] == dataset]
        success = [r for r in subset if r['status'] == 'ok']
        detection_ms = np.mean([float(r['detector_ms']) for r in subset])
        embedding_ms = np.mean([float(r['recognizer_ms']) for r in success]) if success else float('nan')
        lines.append(f'- {dataset}: {len(success)}/{len(subset)} successful embeddings. Mean detector time {detection_ms:.1f} ms/source crop; mean recognizer time {embedding_ms:.1f} ms/aligned face.')
    lines += ['- CPU ONNX Runtime was used; these timings are image-inference measurements, not live-video FPS.', '',
              '## Limits and reproducibility', '',
              '- ICFD uses five development people from the public sample; full ICFD access remains pending.',
              '- A fixed neutral border is applied before detection because the datasets provide tight face crops. The initial unpadded engineering smoke run failed detection; preprocessing was revised using development images only.',
              '- All original-file SHA-256 values are verified before inference. Source exclusions and manifests were prepared before recognition.',
              '- Failed detection or ambiguous multiple-face input is retained as a pipeline failure rather than silently resized or assigned a synthetic embedding.',
              '- No claim is made about video tracking, domain adaptation, Bangladesh-wide performance, independent capture sessions, or absence of pretraining overlap.',
              '- Exact duplicate filtering does not establish that all remaining frames/images are independent.',
              '- See config.yaml, split_snapshot.json, extraction.csv, predictions.csv, threshold.json and run_info.json for the evidence.', '']
    (run / 'BASELINE_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for status, label, color in [('known', 'Enrolled probes: best gallery score', '#2471a3'), ('unknown', 'Unenrolled probes: best gallery score', '#c0392b')]:
        scores = [float(r['top1_score']) for r in predictions if r['dataset'] == 'halfface' and r['truth_status'] == status and r['status'] == 'ok']
        ax.hist(scores, bins=np.linspace(-0.1, 1, 34), alpha=0.55, label=label, color=color)
    ax.axvline(hf['threshold'], color='black', linestyle='--', label='Development-calibrated threshold')
    ax.set(xlabel='Cosine similarity', ylabel='Probe count', title='halfFace: ArcFace development baseline (not final test)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run / 'development_scores.png', dpi=160)
    plt.close(fig)
    sources = ['src/recognition/embedder.py', 'src/detection/detector.py', 'src/alignment/aligner.py',
               'src/quality/estimator.py', 'scripts/run_arcface_baseline.py', 'configs/arcface_baseline.yaml']
    (run / 'source_fingerprints.json').write_text(json.dumps({p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}, indent=2), encoding='utf-8')
    print(run / 'BASELINE_REPORT.md')
    print('\n'.join(lines[:30]))


if __name__ == '__main__':
    main()
