"""Save dependency, launcher and server-health receipts without browser automation."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from packaging.requirements import Requirement
from scripts.run_arcface_baseline import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ports', type=int, nargs='+', default=[8501, 8521])
    args = parser.parse_args()
    directory = args.output.parent
    directory.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ['requirements.txt', 'requirements-dev.txt']:
        for line in (ROOT/name).read_text().splitlines():
            if not line or line.startswith(('#', '-r')):
                continue
            req = Requirement(line)
            version = importlib.metadata.version(req.name)
            assert req.specifier.contains(version), (str(req), version)
            versions[req.name] = version
    checks = {}
    for label, command in {
        'pip_check': [sys.executable, '-m', 'pip', 'check'],
        'dependency_resolution': [sys.executable, '-m', 'pip', 'install', '--dry-run', '--no-index', '-r', 'requirements-dev.txt'],
        'main_launcher_preflight': [sys.executable, 'launch_dashboard.py', 'main', '--check'],
        'defense_launcher_preflight': [sys.executable, 'launch_dashboard.py', 'defense', '--check'],
    }.items():
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (directory/(label+'.txt')).write_text(result.stdout+result.stderr, encoding='utf-8')
        assert result.returncode == 0, (label, result.stderr)
        checks[label] = 'passed'
    servers = {}
    for port in args.ports:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/_stcore/health', timeout=5) as response:
            status = response.read().decode()
            assert status == 'ok', (port, status)
            servers[str(port)] = status
    test_suite = ET.parse(directory/'pytest.xml').getroot().find('testsuite')
    tests = {k: int(test_suite.attrib[k]) for k in ['tests', 'errors', 'failures', 'skipped']}
    assert tests == dict(tests=89, errors=0, failures=0, skipped=0), tests
    workflow = json.loads((directory/'original_workflow.json').read_text())
    assert workflow['passed'] and workflow['models_unchanged'] and workflow['normal_user_gallery_unchanged']
    files = ['README.md', 'docs/DASHBOARD_SETUP.md', 'requirements.txt', 'requirements-dev.txt',
             'requirements.in', 'environment.yml', 'launch_dashboard.py', 'Start Original Dashboard.cmd',
             'Start Defense Dashboard.cmd', 'scripts/check_original_workflow.py', 'scripts/verify_setup.py',
             'app.py', 'defense_app.py']
    report = dict(utc=datetime.now(timezone.utc).isoformat(), passed=True, python=platform.python_version(),
                  executable=sys.executable, local_environment_reuses_system_packages=True,
                  clean_machine_installation_tested=False, versions=versions, checks=checks,
                  server_health=servers, full_test_suite=tests, original_application_flow='passed',
                  fresh_browser_capture=False, source_sha256={p: sha256(ROOT/p) for p in files},
                  receipt_sha256={p.name:sha256(p) for p in directory.iterdir() if p.is_file() and p != args.output})
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
