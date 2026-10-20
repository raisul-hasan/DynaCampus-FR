"""Isolated, real-image demonstration of the evaluated ArcFace gallery workflow."""
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from scripts.run_arcface_baseline import read_csv, sha256
from scripts.compare_enrollment import gallery_candidates, unit_mean
from src.gallery import FaceGallery
from src.recognition import FaceEmbedder
from src.detection import SCRFDDetector
from src.alignment import FaceAligner
from src.quality import FaceQualityEstimator
from src.demo_workflow import process_image, recognize

ROOT = Path(__file__).resolve().parents[1]
STEPS = [('recognize_target', 'Before enrollment: target is unknown'),
         ('enroll', 'Enroll target from three original photos'),
         ('recognize_target', 'After enrollment: masked target is known'),
         ('recognize_unknown', 'A different, unenrolled person is unknown'),
         ('recognize_ambiguous', 'Two competing identities produce an ambiguous decision'),
         ('reload', 'Reload the gallery from disk'),
         ('recognize_target', 'Recognition survives reload'),
         ('remove', 'Remove only the demonstration target'),
         ('recognize_target', 'After removal: target is unknown'),
         ('recognize_capture_failure', 'Show a genuine capture failure')]


def default_run():
    return ROOT / json.loads((ROOT / 'paper_assets/development/defense_v1/source_run.json').read_text())['run']


class DefenseWorkflow:
    def __init__(self, run=None):
        self.run = Path(run) if run else default_run()
        self.manifest = json.loads((self.run / 'case_manifest.json').read_text(encoding='utf-8'))
        self.fixture = json.loads((self.run / 'demo_fixture.json').read_text(encoding='utf-8'))
        for path, expected in self.manifest['input_sha256'].items():
            if sha256(ROOT / path) != expected:
                raise ValueError(f'Research source changed: {path}')
        for path, expected in self.fixture['source_sha256'].items():
            if sha256(ROOT / path) != expected:
                raise ValueError(f'Demo source image changed: {path}')
        self.directory = ROOT / 'data/defense' / self.run.name / 'sessions' / uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=False)
        self.gallery_path = self.directory / 'gallery'
        if not self.gallery_path.resolve().is_relative_to((ROOT / 'data/defense').resolve()):
            raise ValueError('Demo gallery escaped isolated data directory')
        self.gallery = FaceGallery(self.gallery_path)
        source = ROOT / 'results/enrollment/enrollment_development_20261019T222121709545Z'
        info = json.loads((source / 'comparison.json').read_text())
        self.records = {r['path']: r for r in read_csv(source / 'combined_extraction.csv')}
        self.features = np.load(ROOT / info['feature_cache'], allow_pickle=False)['embeddings']
        candidates = gallery_candidates(list(self.records.values()), 'halfface', self.fixture['seed_ids'], 'up_to_three_photos')
        for sid, selected in candidates.items():
            usable = [r for r in selected if r['status'] == 'ok']
            if not self.gallery.enroll_multi(sid, 'Development '+sid[-3:], [self.features[int(r['embedding_index'])] for r in usable], [float(r['quality']) for r in usable]):
                raise ValueError('Unable to initialize verified background gallery')
        if self.gallery.size != self.fixture['gallery_size_before']:
            raise ValueError('Incorrect background gallery size')
        self.detector = SCRFDDetector(confidence_threshold=.5, providers=['CPUExecutionProvider'], cpu_threads=2)
        self.embedder = FaceEmbedder(model_name='insightface_arcface', providers=['CPUExecutionProvider'], cpu_threads=2)
        self.aligner, self.quality = FaceAligner(), FaceQualityEstimator()
        self.model_hashes = {str(p): sha256(p) for p in [self.detector.model_path, self.embedder.model_path]}
        self.events = []
        self.step = 0
        self.append(dict(action='initialize', gallery_size=self.gallery.size, target_enrolled=False,
                         seed_source='cached real development embeddings', target_and_probe_source='fresh original-image inference'))

    def append(self, event):
        event.update(sequence=len(self.events), utc=datetime.now(timezone.utc).isoformat())
        self.events.append(event)
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps(event)+'\n')
        return event

    def infer(self, path):
        if path not in self.fixture['source_sha256'] or sha256(ROOT / path) != self.fixture['source_sha256'][path]:
            raise ValueError('Unexpected or changed fixture image')
        faces = process_image(cv2.imread(str(ROOT/path)), self.detector, self.aligner, self.embedder, self.quality, face_crop=True)
        delta = None
        record = self.records[path]
        if record['status'] == 'ok':
            if len(faces) != 1:
                raise ValueError('Fresh capture differs from saved detector outcome')
            delta = float(np.max(np.abs(faces[0]['embedding']-self.features[int(record['embedding_index'])])))
            if delta > 1e-6:
                raise ValueError('Fresh image inference differs from saved real features')
        elif faces:
            raise ValueError('Saved capture failure no longer reproduces')
        return faces, delta

    def perform(self, action):
        before = self.gallery.size
        start = time.perf_counter_ns()
        sid = self.fixture['subject_id']
        event = dict(action=action, gallery_size_before=before)
        if action.startswith('recognize_'):
            key = {'recognize_target': 'known_probe', 'recognize_unknown': 'unknown_probe',
                   'recognize_ambiguous': 'ambiguous_probe', 'recognize_capture_failure': 'capture_failure_probe'}[action]
            path = self.fixture[key]
            faces, delta = self.infer(path)
            if not faces:
                event.update(decision='CAPTURE_FAILURE', accepted_id='', face_count=0)
            else:
                result = recognize(self.gallery, faces[0]['embedding'], faces[0]['quality']['overall_quality'])
                event.update({k: result[k] for k in ['decision', 'top1_id', 'top1_score', 'top2_score']})
                ids = list(self.gallery._embeddings)
                scores = np.stack([self.gallery._embeddings[s] for s in ids]) @ faces[0]['embedding']
                ranked = np.argsort(scores)[::-1]
                event['top2_id'] = ids[int(ranked[1])]
                event.update(accepted_id=result['accepted_id'] or '', face_count=1,
                             margin=result['top1_score']-result['top2_score'], quality=faces[0]['quality']['overall_quality'])
            event.update(source_path=path, source='fresh original-image inference', feature_cache_max_difference=delta)
        elif action == 'enroll':
            vectors, qualities, deltas = [], [], []
            for path in self.fixture['enrollment_paths']:
                faces, delta = self.infer(path)
                vectors.append(faces[0]['embedding'])
                qualities.append(faces[0]['quality']['overall_quality'])
                deltas.append(delta)
            if not self.gallery.enroll_multi(sid, self.fixture['display_name'], vectors, qualities):
                raise ValueError('Target enrollment failed')
            expected = unit_mean([self.features[int(self.records[p]['embedding_index'])] for p in self.fixture['enrollment_paths']])
            np.testing.assert_allclose(self.gallery._embeddings[sid], expected, atol=1e-7)
            event.update(enrolled_id=sid, photos=len(vectors), source='fresh original-image inference', feature_cache_max_difference=max(deltas))
        elif action == 'reload':
            snapshot = {s: v.copy() for s, v in self.gallery._embeddings.items()}
            self.gallery = FaceGallery(self.gallery_path)
            if snapshot.keys() != self.gallery._embeddings.keys() or any(not np.array_equal(v, self.gallery._embeddings[s]) for s, v in snapshot.items()):
                raise ValueError('Persistent gallery differs after reload')
            event.update(persistence_exact=True)
        elif action == 'remove':
            if sid not in self.gallery._embeddings or not self.gallery.remove(sid):
                raise ValueError('Demo target is not currently enrolled')
            event.update(removed_id=sid)
        else:
            raise ValueError('Unknown defense action')
        event.update(gallery_size=self.gallery.size, target_enrolled=sid in self.gallery._embeddings,
                     elapsed_ms=(time.perf_counter_ns()-start)/1e6, score_threshold=self.fixture['score_threshold'],
                     margin_threshold=self.fixture['margin_threshold'])
        for path, expected in self.model_hashes.items():
            if sha256(Path(path)) != expected:
                raise ValueError('Checkpoint changed during demo')
        event['backbone_checkpoints_unchanged'] = True
        return self.append(event)

    def next_step(self):
        if self.step >= len(STEPS):
            raise ValueError('Walkthrough complete; start a new isolated session to repeat it')
        action, title = STEPS[self.step]
        result = self.perform(action)
        result['step_title'] = title
        self.step += 1
        return result
