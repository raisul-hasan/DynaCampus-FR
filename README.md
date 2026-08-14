# DynaCampus-FR

**Domain-Adaptive, Quality-Aware Open-Set Face Recognition with Training-Free Dynamic Enrollment for Low-Quality Campus CCTV**

Department of Computer Science and Engineering  
United International University

---

## 📌 Overview
DynaCampus-FR is a research project developing an open-set campus CCTV face recognition framework. It addresses the domain gap between clean enrollment photographs and degraded CCTV probe tracks without retraining the core recognition backbone when new identities enroll.

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
