"""Select disclosed development examples without changing research decisions.

Creates a complete candidate ledger, fixed-rule and separately calibrated cases,
an offline local photo viewer, redacted evidence, and a real-demo fixture.
"""
import base64
import html
import io
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_arcface_baseline import read_csv, write_csv, sha256
from scripts.compare_enrollment import gallery_candidates, unit_mean
from src.open_set import OpenSetDecisionEngine

POOL = ROOT / 'results/enrollment_pooling/pooling_development_20261020T181729489649Z'
BACKBONES = ROOT / 'results/backbones/backbones_development_20261020T200004286474Z'
ENROLLMENT = ROOT / 'results/enrollment/enrollment_development_20261019T222121709545Z'
SCORE, MARGIN = .3778589367866516, .15


def truth(value):
    return value is True or value == 'True'


def category(first, few):
    if few['status'] != 'ok':
        return 'capture_failure' if few['truth_status'] == 'known' else None
    if few['truth_status'] == 'unknown':
        return {'UNKNOWN': 'unknown_rejected', 'AMBIGUOUS': 'ambiguous_unknown', 'KNOWN': 'unknown_false_accept'}.get(few['decision'])
    a, b = truth(first['correct_known_accept']), truth(few['correct_known_accept'])
    if not a and b:
        return 'helped_known'
    if a and not b:
        return 'harmed_known'
    if a and b:
        return 'unchanged_accept'
    if few['decision'] == 'UNKNOWN':
        return 'known_rejected'
    return None


def compare_pair(one, few, scope, model, thresholds):
    first, second = {r['path']: r for r in one}, {r['path']: r for r in few}
    if first.keys() != second.keys():
        raise ValueError('Different probes in paired example source')
    pairs = []
    for path in sorted(first):
        a, b = first[path], second[path]
        label = category(a, b)
        if label:
            pairs.append(dict(scope=scope, model=model, category=label, path=path, subject_id=b['subject_id'],
                truth_status=b['truth_status'], status=b['status'], one=a, few=b,
                one_threshold=thresholds[0][0], one_margin=thresholds[0][1],
                few_threshold=thresholds[1][0], few_margin=thresholds[1][1]))
    return pairs


def font(size):
    for path in ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/segoeui.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_case(case, private=True):
    canvas = Image.new('RGB', (1360, 780), '#f4f7fa')
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), f"{case['case_id']}  |  {case['category'].replace('_', ' ')}", fill='#143047', font=font(27))
    draw.text((30, 65), f"{case['model']}  |  {case['scope']}  |  ground truth: {case['subject_id']} ({case['truth_status']})", fill='#30475c', font=font(18))
    draw.text((30, 96), 'Selected development example: illustrative, not an additional accuracy sample.', fill='#7a4321', font=font(18))
    tiles = [('Probe', case['path'])] + [(f'Enrollment {i+1}', p) for i, p in enumerate(case['enrollment_paths'])]
    for i, (label, path) in enumerate(tiles):
        x = 30 + 330 * i
        draw.rounded_rectangle((x, 145, x+305, 425), radius=8, fill='white', outline='#d5e0e8')
        draw.text((x+12, 153), label, fill='#143047', font=font(18))
        if private:
            image = Image.open(ROOT / path).convert('RGB')
            image.thumbnail((275, 220))
            canvas.paste(image, (x+15+(275-image.width)//2, 190+(220-image.height)//2))
        else:
            draw.text((x+35, 260), 'Photo kept local', fill='#6a7c8c', font=font(20))
    draw.text((30, 448), f"Enrollment views belong to {case['enrollment_subject']}; original files are unchanged.", fill='#30475c', font=font(17))
    for number, name in enumerate(['one', 'few']):
        row = case[name]
        y = 485 + number*92
        threshold = case[f'{name}_threshold']
        margin = case[f'{name}_margin']
        score = float(row['top1_score']) if row['status'] == 'ok' else None
        gap = float(row['top1_score'])-float(row['top2_score']) if score is not None else None
        draw.text((30, y), f"{name.upper()}: {row['decision']}  |  accepted: {row.get('accepted_id') or 'none'}  |  closest: {row.get('top1_id') or 'unavailable'}", fill='#143047', font=font(22))
        values = 'No embedding: detector/capture stage failed.' if score is None else f'score {score:.6f} / threshold {threshold:.6f}    gap {gap:.6f} / minimum {margin:.6f}'
        draw.text((30, y+35), values, fill='#30475c', font=font(20))
    explanation = ('Same fixed thresholds isolate template-policy effects in this sensitivity example.' if case['scope'] == 'fixed_rule_sensitivity' else
                   'Thresholds differ: the example compares calibrated policies, not enrollment alone.')
    draw.text((30, 690), explanation, fill='#7a4321', font=font(18))
    draw.text((30, 730), 'DynaCampus-FR | Source paths, hashes and all eligible examples are recorded in the case manifest.', fill='#536a7b', font=font(16))
    return canvas


def evaluate(matrix, ids, vector):
    values = matrix @ vector
    order = np.argsort(values)[::-1]
    state, identity, _ = OpenSetDecisionEngine(base_threshold=SCORE, margin_threshold=MARGIN, quality_threshold=0.).evaluate_scores(
        ids[order[0]], float(values[order[0]]), float(values[order[1]]), 1.)
    return dict(decision=state, accepted_id=identity or '', top1_id=ids[order[0]], top1_score=float(values[order[0]]),
                top2_id=ids[order[1]], top2_score=float(values[order[1]]))


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run = ROOT / 'results/defense' / f'defense_development_{stamp}'
    private = ROOT / 'data/defense' / run.name
    public = ROOT / 'paper_assets/development/defense_v1'
    for path in [run, private / 'cards', public]:
        path.mkdir(parents=True, exist_ok=False if path == run else True)
    extraction = read_csv(ENROLLMENT / 'combined_extraction.csv')
    by_path = {r['path']: r for r in extraction}
    split = json.loads((ENROLLMENT / 'split_snapshot.json').read_text())
    allowed = set(sum(split['halfface']['development'].values(), []))
    known = sorted(split['halfface']['development']['known'])
    info = json.loads((ENROLLMENT / 'comparison.json').read_text())
    features = np.load(ROOT / info['feature_cache'], allow_pickle=False)['embeddings']
    candidates = gallery_candidates(extraction, 'halfface', known, 'up_to_three_photos')
    source_rows = read_csv(POOL / 'predictions.csv')
    fixed = [r for r in source_rows if r['dataset'] == 'halfface' and r['rule'] == 'frozen_margin_sensitivity']
    pairs = compare_pair([r for r in fixed if r['method'] == 'one_photo'], [r for r in fixed if r['method'] == 'uniform_mean'],
                         'fixed_rule_sensitivity', 'arcface', [(SCORE, MARGIN)]*2)
    backbone_rows = read_csv(BACKBONES / 'predictions.csv')
    thresholds = read_csv(BACKBONES / 'selected_thresholds.csv')
    for model in ['arcface', 'adaface']:
        rows = [r for r in backbone_rows if r['model'] == model and r['dataset'] == 'halfface' and r['family'] == 'score_margin']
        settings = [next(r for r in thresholds if r['model'] == model and r['strategy'] == strategy and r['family'] == 'score_margin')
                    for strategy in ['one_photo', 'up_to_three_photos']]
        pairs += compare_pair([r for r in rows if r['strategy'] == 'one_photo'], [r for r in rows if r['strategy'] == 'up_to_three_photos'],
            'separately_calibrated_policies', model, [(float(r['score_threshold']), float(r['margin_threshold'])) for r in settings])
    requested = [('fixed_rule_sensitivity', 'arcface', name) for name in ['helped_known', 'harmed_known', 'unchanged_accept',
        'unknown_rejected', 'ambiguous_unknown', 'known_rejected', 'capture_failure', 'unknown_false_accept']]
    requested += [('separately_calibrated_policies', model, name) for model in ['arcface', 'adaface'] for name in ['helped_known', 'harmed_known']]
    selected = []
    for scope, model, label in requested:
        pool = [r for r in pairs if (r['scope'], r['model'], r['category']) == (scope, model, label)]
        if not pool:
            raise ValueError(f'Requested category has no example: {scope}, {model}, {label}')
        case = dict(sorted(pool, key=lambda r: r['path'])[0])
        case['case_id'] = f'C{len(selected)+1:02d}'
        if case['subject_id'] not in allowed:
            raise ValueError('Non-development example')
        sid = case['subject_id'] if case['truth_status'] == 'known' else case['few']['top1_id']
        case['enrollment_subject'] = sid
        case['enrollment_paths'] = [r['path'] for r in candidates[sid] if r['status'] == 'ok']
        paths = [case['path']] + case['enrollment_paths']
        if len(paths) != len(set(paths)):
            raise ValueError('Selected example reuses an enrollment image as probe')
        case['source_sha256'] = {p: sha256(ROOT / p) for p in paths}
        selected.append(case)
        render_case(case).save(private / 'cards' / f"{case['case_id']}.png")
        render_case(case, private=False).save(public / f"fig_case_{case['case_id']}.png")
    # Select a real membership-change fixture with 54 existing templates and a
    # previously unenrolled target. All 55 identities are actual development IDs.
    prototypes = {sid: unit_mean([features[int(r['embedding_index'])] for r in candidates[sid] if r['status'] == 'ok']) for sid in known}
    matrix = np.stack([prototypes[sid] for sid in known])
    fixture = None
    for pair in sorted([r for r in pairs if r['scope'] == 'fixed_rule_sensitivity' and r['category'] == 'helped_known'], key=lambda r: r['path']):
        sid = pair['subject_id']
        if len(candidates[sid]) != 3 or any(r['status'] != 'ok' for r in candidates[sid]):
            continue
        others = [s for s in known if s != sid]
        before = evaluate(np.stack([prototypes[s] for s in others]), others, features[int(by_path[pair['path']]['embedding_index'])])
        after = evaluate(matrix, known, features[int(by_path[pair['path']]['embedding_index'])])
        if before['decision'] == 'UNKNOWN' and after['decision'] == 'KNOWN' and after['accepted_id'] == sid:
            fixture = dict(subject_id=sid, display_name='Development target', enrollment_paths=[r['path'] for r in candidates[sid]],
                known_probe=pair['path'], expected_before=before, expected_after=after, seed_ids=others)
            break
    if fixture is None:
        raise ValueError('No genuine membership-change fixture satisfies the declared conditions')
    for label, category_name in [('unknown_probe', 'unknown_rejected'), ('ambiguous_probe', 'ambiguous_unknown'), ('capture_failure_probe', 'capture_failure')]:
        fixture[label] = next(r['path'] for r in selected if r['scope'] == 'fixed_rule_sensitivity' and r['category'] == category_name)
    fixture.update(gallery_size_before=54, gallery_size_after=55, score_threshold=SCORE, margin_threshold=MARGIN,
        note='Selected reproducible development examples, not a random accuracy sample. 54 seeds use cached real features; target photos and probes are to be freshly inferred.',
        source_sha256={p: sha256(ROOT / p) for p in fixture['enrollment_paths'] + [fixture[k] for k in ['known_probe', 'unknown_probe', 'ambiguous_probe', 'capture_failure_probe']]})
    fixture['expected_unknown'] = evaluate(matrix, known, features[int(by_path[fixture['unknown_probe']]['embedding_index'])])
    fixture['expected_ambiguous'] = evaluate(matrix, known, features[int(by_path[fixture['ambiguous_probe']]['embedding_index'])])
    (run / 'demo_fixture.json').write_text(json.dumps(fixture, indent=2), encoding='utf-8')
    ledger = [{k: r[k] for k in ['scope', 'model', 'category', 'path', 'subject_id', 'truth_status', 'status']} for r in pairs]
    write_csv(run / 'candidate_ledger.csv', ledger)
    counts = Counter((r['scope'], r['model'], r['category']) for r in pairs)
    write_csv(run / 'candidate_counts.csv', [dict(scope=k[0], model=k[1], category=k[2], count=v) for k, v in sorted(counts.items())])
    manifest = dict(run_directory=run.relative_to(ROOT).as_posix(), private_directory=private.relative_to(ROOT).as_posix(),
        selection_rule='First lexicographic source path in every declared category, before visual inspection; no replacements to improve appearance.',
        cases=selected, requested_categories=requested, candidate_count=len(pairs),
        input_sha256={p.relative_to(ROOT).as_posix(): sha256(p) for p in [POOL/'predictions.csv', POOL/'verification.json', BACKBONES/'predictions.csv',
            BACKBONES/'selected_thresholds.csv', BACKBONES/'verification.json', ENROLLMENT/'combined_extraction.csv', ENROLLMENT/'split_snapshot.json',
            ENROLLMENT/'comparison.json', ROOT/info['feature_cache']]},
        final_test_used=False, new_model_inference=False, script_sha256=sha256(Path(__file__)),
        caution='Operational failure labels describe outcomes. Visual observations cannot establish the cause of a recognition failure.')
    (run / 'case_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (private / 'case_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # A portable offline viewer with photographs stays under ignored data/.
    sections = []
    for case in selected:
        data = base64.b64encode((private / 'cards' / f"{case['case_id']}.png").read_bytes()).decode()
        sections.append(f'<section id="{case["case_id"]}"><img src="data:image/png;base64,{data}" alt="{html.escape(case["category"])}"></section>')
    nav = ''.join(f'<a href="#{r["case_id"]}">{r["case_id"]}: {html.escape(r["category"].replace("_", " "))}</a>' for r in selected)
    viewer = '<!doctype html><meta charset="utf-8"><title>DynaCampus defense cases: local development evidence</title><style>body{margin:0;background:#e8eef4;font:16px Arial}nav{position:sticky;top:0;background:#123047;padding:12px;display:flex;flex-wrap:wrap;gap:12px}a{color:white}section{max-width:1360px;margin:25px auto}img{width:100%;height:auto}</style><nav>'+nav+'</nav>'+''.join(sections)
    (private / 'index.html').write_text(viewer, encoding='utf-8')
    # A compact contact sheet is for local visual inspection, not publication.
    thumb = Image.new('RGB', (1360, 780*3), 'white')
    for group in range(0, len(selected), 3):
        for offset, case in enumerate(selected[group:group+3]):
            thumb.paste(render_case(case), (0, offset*780))
        thumb.save(private / f'contact_{group//3+1:02d}.png')
    (public / 'source_run.json').write_text(json.dumps(dict(run=manifest['run_directory'], case_manifest_sha256=sha256(run/'case_manifest.json'),
        photographs_in_public_assets=False, selection_rule=manifest['selection_rule']), indent=2), encoding='utf-8')
    print(json.dumps(dict(run_directory=manifest['run_directory'], private_viewer=str(private/'index.html'), cases=len(selected),
        fixture_target=fixture['subject_id'], fixture_known=fixture['known_probe'], expected_ambiguous=fixture['expected_ambiguous']), indent=2))


if __name__ == '__main__':
    main()
