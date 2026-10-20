# DynaCampus-FR

**Domain-Adaptive, Quality-Aware Open-Set Face Recognition with Training-Free Dynamic Enrollment for Low-Quality Campus CCTV**

Department of Computer Science and Engineering  
United International University

---

## Current verified stage

The implemented experiments use **ArcFace** with real SCRFD detection, alignment
and exact cosine search on halfFace development identities and the ICFD public
engineering sample. Saved comparisons cover one-photo versus up-to-three-photo
enrollment and calibrated score, top-two margin and heuristic quality decisions.
The selected development quality gate is disabled because it added no benefit.
Controlled development stress tests also cover Gaussian blur, lower resolution
and JPEG recompression with fixed models, gallery and decision thresholds.
Missing model files cause explicit errors; synthetic embeddings are restricted to
explicit offline test mode.

See [project evidence](paper_assets/README.md) and [dataset and experiment status](docs/DATASET_START.md)
for figures, captions, tables and reproduction commands. A frozen evaluation on
reserved halfFace identities is complete: the primary few-photo margin method
correctly accepts 151/160 known probes (94.38%), with 0/151 observed false known
acceptances. These counts apply to this public-dataset test; they are not a
population error guarantee. The separate reserved ICFD sample scores 98/100 in
closed-set retrieval. See the [held-out evidence](paper_assets/heldout/final_v1/README.md).
The still-image app flow is checked with real development uploads, including
original-photo enrollment, known/unknown recognition, reload and removal.
See [application evidence](paper_assets/development/demo_v1/README.md) for actual
screenshots and reproduction. Camera/RTSP and temporal performance are unvalidated.
AdaFace, the domain adapter
and the full original architecture below are planned work, not validated results.

---

## 📌 Overview
DynaCampus-FR is a four-month research project developing an open-set campus CCTV face recognition framework. It addresses the domain gap between clean enrollment photographs and degraded CCTV probe tracks without retraining the core recognition backbone when new identities enroll.

### Key Architecture
- **Detector**: SCRFD (5-point face landmarks)
- **Primary Backbone**: AdaFace IR-50 (pretrained & frozen)
- **Baseline Backbone**: ArcFace / InsightFace
- **Domain Adapter**: Lightweight Residual MLP ($512 \to 256 \to 512$)
- **Tracking**: ByteTrack face tracking
- **Aggregation**: Quality-weighted multi-frame track aggregation
- **Gallery Search**: Exact Cosine Matrix Matching
- **Decision Engine**: Calibrated 4-State Open-Set Decision (`KNOWN`, `UNKNOWN`, `AMBIGUOUS`, `INSUFFICIENT_QUALITY`)

---

## 📁 Repository Structure
```
DynaCampus-FR/
├── configs/          # YAML configuration files
├── notebooks/        # Exploration and verification notebooks
├── src/              # Core modular library
│   ├── detection/
│   ├── alignment/
│   ├── recognition/
│   ├── augmentation/
│   ├── tracking/
│   ├── quality/
│   ├── adapter/
│   ├── gallery/
│   ├── aggregation/
│   ├── open_set/
│   ├── evaluation/
│   └── utils/
├── scripts/          # Pipeline execution and evaluation scripts
├── results/          # Raw predictions, tables, figures, checkpoints, logs
├── docs/             # Protocols, consent forms, cards, notes
└── tests/            # Unit and integration tests
```

---

## 🖥️ Supervisor Dashboard

A fully functional Streamlit dashboard is included to manage the face gallery, test the recognition pipeline, and evaluate live webcam and video feeds.

### Features
- **Dynamic Enrollment:** Enroll new subjects directly from the UI without retraining the underlying model. Extracts 512-d embeddings and saves them to a local JSON/NPZ gallery.
- **Image & Video Recognition:** Upload files for automated bounding-box annotation and similarity scoring.
- **Live Webcam:** Real-time face detection and recognition using your local camera.

---

## 🚀 Setup Guide

### 1. Requirements
- Python 3.10+
- A working C++ compiler (for InsightFace)
- Webcam (optional, for live feed)

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/raisul-hasan/DynaCampus-FR.git
cd DynaCampus-FR
python -m venv .venv
```

Activate the environment:
- **Windows:** `.venv\Scripts\activate`
- **Linux/Mac:** `source .venv/bin/activate`

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Model Downloads
The system uses `insightface`. The first time you run the application, it will attempt to download the `buffalo_l` model bundle (~300MB) to `~/.insightface/models/buffalo_l`.

### 4. Running the Dashboard
To start the Streamlit web application:

```bash
python -m streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.
