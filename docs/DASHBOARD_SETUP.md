# Dashboard setup and verification

The original research dashboard is `app.py`; the separate defense evidence
dashboard is `defense_app.py`. Both use ArcFace. The defense dashboard adds saved
comparisons and an isolated development walkthrough; it does not replace the
original dashboard or provide a new recognition algorithm.

## Launch on this laptop

Double-click **Start Original Dashboard.cmd** or **Start Defense Dashboard.cmd**
in the project folder. Keep the server window open and visit:

- Original dashboard: <http://127.0.0.1:8501>
- Defense dashboard: <http://127.0.0.1:8521>

The launchers select `.venv\Scripts\python.exe` when available. They check local
dependencies and model files before starting. They do not download assets or
automatically open a browser. Stop a server with Ctrl+C in its terminal. If its
port is already occupied, use the existing server or supply another port:

```powershell
.\.venv\Scripts\python.exe launch_dashboard.py main --port 8502
.\.venv\Scripts\python.exe launch_dashboard.py defense --port 8522
```

Check setup without starting a server:

```powershell
.\.venv\Scripts\python.exe launch_dashboard.py main --check
.\.venv\Scripts\python.exe launch_dashboard.py defense --check
```

The original app normally uses `gallery_data/`. The defense walkthrough and
verification script use separate galleries under ignored `data/defense/`.
For a separate manual practice gallery, set an environment override first:

```powershell
$env:DYNACAMPUS_GALLERY_DIR = "$PWD\data\practice_gallery"
.\.venv\Scripts\python.exe launch_dashboard.py main
```

For the halfFace demonstration crops, enable **Input is a tight face crop**.
Keep **Frozen research settings** selected. Photo previews are off by default.

## Installation on another machine

The verified machine uses Windows and Python **3.13.3**. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For AdaFace research scripts and all tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

`requirements.txt` pins direct runtime dependencies observed working here.
`requirements-dev.txt` adds PyTorch and pytest. These are not a complete
transitive lock or proof of a successful installation on a clean machine.
The current local `.venv` reuses existing system packages and adds pytest and
FilterPy locally. `pip check` passes. A clean installation remains untested.
The existing CUDA PyTorch build is `2.9.0+cu126`; the dashboards run ONNX inference
on CPU and do not require CUDA. Avoid installing a second OpenCV distribution
alongside `opencv-python` in the same environment. The old
`requirements-lock.txt` is a historical snapshot and is not the installation
entry point. `requirements.in` forwards to the current runtime list.

Conda is an alternative: run `conda env create -f environment.yml` from the
repository root. That installation path has not been freshly tested.

Model files must already exist at:

```text
~/.insightface/models/buffalo_l/det_10g.onnx
~/.insightface/models/buffalo_l/w600k_r50.onnx
```

The app loads those files explicitly and fails if they are missing. It does not
automatically download the model bundle. Preserve the verified model files on
this laptop. For another machine, obtain the corresponding official InsightFace
assets and compare hashes with the saved verification report. AdaFace research
assets have a separate [setup and provenance guide](ADAFACE_INTEGRATION.md).
The defense dashboard also needs the local `data/` research inputs and saved
`results/` artifacts; a source-only clone does not contain the private local
feature caches, images or model weights.

## What is implemented and validated

The tested still-image pipeline is SCRFD detection, five-point alignment,
frozen ArcFace embeddings, normalized original-photo template averaging,
exact cosine search, and calibrated score/margin decisions. Enrollment, gallery
persistence, removal, known/unknown decisions, ambiguity and capture failure
are checked. AdaFace is a separate development comparison.

The adapter is a prototype module, not a trained or validated contribution.
The current tracker is a SORT-style IoU/Kalman implementation, not ByteTrack.
Video, webcam, RTSP and temporal aggregation controls exist, but their presence
does not establish real CCTV accuracy, temporal performance or deployment impact.

The defensible contribution is experimental evidence about enrollment and
decision choices with a working demonstration. The thesis must distinguish
the original CCTV motivation from what was actually evaluated on public images.
See [novelty audit](NOVELTY_AUDIT.md), [evidence index](../paper_assets/README.md)
and [defense evidence](DEFENSE_EVIDENCE.md).

## Reproduce local checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/check_original_workflow.py --output results/validation/manual_original_workflow.json
.\.venv\Scripts\python.exe scripts/check_defense_app.py
.\.venv\Scripts\python.exe scripts/verify_final_evaluation.py results/final/heldout_20261019T231512869782Z
```

The original workflow check runs the unchanged `app.py` through Streamlit
AppTest with real uploaded-image bytes supplied at the file-uploader boundary.
Detection, embeddings, decision logic, enrollment and persistence execute for
real. This is application testing; it is not a browser upload test or a fresh
screen recording. Genuine earlier browser evidence remains separately indexed
under `paper_assets/development/demo_v1/`.

Current validation receipts are in `results/validation/setup_20261021/`.
The full suite passed **89 tests**; dependency checks and UI flow checks have
their own evidence scopes. The frozen evaluation has not been rerun to obtain
new predictions.
