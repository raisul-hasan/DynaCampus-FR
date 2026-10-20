"""Launch either dashboard from the project directory with this interpreter."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dashboard', choices=['main', 'defense'], nargs='?', default='main')
    parser.add_argument('--port', type=int)
    parser.add_argument('--check', action='store_true', help='Check local dependencies and assets without starting a server')
    args = parser.parse_args()
    port = args.port or (8501 if args.dashboard == 'main' else 8521)
    if not 1 <= port <= 65535:
        parser.error('Port must be between 1 and 65535')
    packages = {'streamlit': 'streamlit', 'cv2': 'opencv-python', 'numpy': 'numpy',
                'pandas': 'pandas', 'scipy': 'scipy', 'yaml': 'pyyaml',
                'insightface': 'insightface', 'onnxruntime': 'onnxruntime', 'filterpy': 'filterpy'}
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    paths = [ROOT/'configs/models.yaml', Path.home()/'.insightface/models/buffalo_l/det_10g.onnx',
             Path.home()/'.insightface/models/buffalo_l/w600k_r50.onnx']
    if args.dashboard == 'defense':
        paths.append(ROOT/'paper_assets/development/defense_v1/source_run.json')
        if paths[-1].is_file():
            run = ROOT/json.loads(paths[-1].read_text())['run']
            paths.extend([run/'case_manifest.json', run/'demo_fixture.json'])
            if (run/'demo_fixture.json').is_file():
                fixture = json.loads((run/'demo_fixture.json').read_text())
                paths.extend(ROOT/p for p in fixture['source_sha256'])
    missing_paths = [str(p) for p in paths if not p.is_file()]
    if missing or missing_paths:
        print(json.dumps(dict(missing_modules=missing, missing_files=missing_paths), indent=2))
        print('See README.md for setup; models and research data are local assets.')
        return 1
    if args.check:
        print(json.dumps(dict(dashboard=args.dashboard, python=sys.version.split()[0],
                             interpreter=sys.executable, port=port, passed=True,
                             versions={dist: importlib.metadata.version(dist) for dist in packages.values()},
                             check_scope='module availability and required local asset paths; use workflow scripts for inference'), indent=2))
        return 0
    env = dict(os.environ, STREAMLIT_BROWSER_GATHER_USAGE_STATS='false')
    print(f'Starting {args.dashboard} dashboard at http://127.0.0.1:{port}', flush=True)
    try:
        return subprocess.call([sys.executable, '-m', 'streamlit', 'run',
                                str(ROOT/('app.py' if args.dashboard == 'main' else 'defense_app.py')),
                                '--server.address', '127.0.0.1', '--server.port', str(port),
                                '--server.headless', 'true', '--browser.gatherUsageStats', 'false'], cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
