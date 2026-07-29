# DynaCampus-FR

**Domain-Adaptive, Quality-Aware Open-Set Face Recognition with Training-Free Dynamic Enrollment for Low-Quality Campus CCTV**

Department of Computer Science and Engineering  
United International University

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
