# 🎓 DynaCampus-FR: Complete System Visual Blueprint & Architecture

This visual guide demonstrates **everything the final product does**, showing how all 9 AI/CV components connect into the end-to-end open-set surveillance recognition pipeline.

---

## 1. 🏗️ Complete End-to-End System Architecture

```mermaid
flowchart TD
    subgraph Input ["1. Input Media Streams"]
        A1["📹 Live Webcam Feed"]
        A2["📡 IP CCTV Stream (RTSP)"]
        A3["🎬 Uploaded Video File"]
        A4["➕ ID Photo Enrollment"]
    end

    subgraph Preprocessing ["2. Face Detection & Alignment"]
        B1["SCRFD Multi-Scale Face Detector"]
        B2["5-Point Facial Landmark Aligner (112x112)"]
        B3["Face Quality Estimator (Pose, Blur, Res, Exposure)"]
    end

    subgraph Enhancement ["3. Domain Adaptation & Augmentation"]
        C1["CCTV-Aware Data Augmentor (4 Synthetic Variants)"]
        C2["Pretrained ArcFace ResNet50 Backbone (512-d)"]
        C3["Residual Domain Adapter (MLP Mapping)"]
    end

    subgraph Pipeline ["4. Temporal Tracking & Aggregation"]
        D1["SORT Multi-Person Kalman Filter Tracker"]
        D2["Quality-Aware Temporal Aggregator (QATA)"]
    end

    subgraph Matching ["5. Scalable Vector Gallery"]
        E1["FAISS IndexFlatIP Inner Product Search"]
        E2["Disk Gallery Storage (meta.json, npz, thumbnails)"]
    end

    subgraph Decision ["6. Open-Set Decision Engine (AOSM)"]
        F1["Adaptive Threshold Calculator (AOSM Margin)"]
        F2{"4-State Classification"}
        G1["✅ KNOWN (Green Box)"]
        G2["❌ UNKNOWN (Red Box)"]
        G3["⚠️ AMBIGUOUS (Yellow Box)"]
        G4["⚪ INSUFFICIENT QUALITY (Gray Box)"]
    end

    subgraph AutoEnroll ["7. Zero-Shot Auto-Enrollment Queue"]
        H1["High-Quality Unknown Face Trigger (q > 0.70)"]
        H2["⏳ Pending Approvals Queue"]
        H3["Admin One-Click Name & Approve Panel"]
    end

    A1 & A2 & A3 --> B1 --> B2 --> B3
    A4 --> C1 --> C2 --> E2
    B2 --> D1 --> D2 --> C3 --> E1
    E1 --> F1 --> F2
    F2 --> G1 & G2 & G3 & G4
    G2 --> H1 --> H2 --> H3 --> E2
```

---

## 2. 🎬 Visual UI Map of Dashboard Pages (`app.py`)

```
===================================================================================
                       🎓 DynaCampus-FR RESEARCH DASHBOARD                       
===================================================================================
┌───────────────────────┬─────────────────────────────────────────────────────────┐
│ NAVIGATION            │ 📊 DASHBOARD OVERVIEW                                  │
│                       │ ------------------------------------------------------- │
│ 🔹 Dashboard         │ [ Identities: 5 ]  [ System FPS: 30.4 ]  [ Accuracy: 100% ]│
│ 🔹 Enroll Person      │                                                         │
│ 🔹 Image Recognition  ├─────────────────────────────────────────────────────────┤
│ 🔹 Video Recognition  │ 📹 LIVE WEBCAM & RTSP CCTV RECOGNITION                  │
│ 🔹 Live Webcam        │ ------------------------------------------------------- │
│ 🔹 Live CCTV (RTSP)   │  ┌───────────────────────────────────────────────────┐  │
│ 🔹 Pending Approvals  │  │  [  Track #1: Raisul (0.94)  ] (Green Box)       │  │
│ 🔹 Gallery Management │  │  [  Track #2: UNKNOWN (0.12) ] (Red Box)         │  │
│                       │  └───────────────────────────────────────────────────┘  │
│                       │  ⚡ Tracking: SORT  |  Aggregation: QATA  | Engine: AOSM   │
│                       ├─────────────────────────────────────────────────────────┤
│ RECOGNITION THRESHOLD │ ⏳ PENDING APPROVALS QUEUE (Auto-Captured Unknowns)      │
│ [━━━━━━━●━━━━━] 0.40  │ ------------------------------------------------------- │
│                       │  [ 👤 AutoID_3 ] -> Quality: 0.828 -> [ Type Name ] [Approve]│
└───────────────────────┴─────────────────────────────────────────────────────────┘
```

---

## 3. 🔬 Core Novelties Implemented in the Product

### 🌟 1. Quality-Aware Temporal Aggregation (QATA)
Rather than making unstable frame-by-frame decisions, the system buffers the last 15 frames for each person. It computes a weighted average embedding:
$$\mathbf{z}_{\text{QATA}} = \text{normalize}\left(\frac{\sum q_k \cdot \mathbf{z}_k}{\sum q_k}\right)$$

### 🌟 2. Adaptive Open-Set Margin (AOSM)
The decision threshold dynamically softens for lower quality frames to eliminate false rejections:
$$\tau_{\text{adaptive}} = \max\left(\tau_{\text{base}} - (1.0 - q) \cdot 0.10, \; 0.25\right)$$

### 🌟 3. Residual Domain Adapter
A lightweight PyTorch MLP maps blurry CCTV probe vectors closer to clean ID card prototypes:
$$\mathbf{z}_{\text{adapted}} = \text{normalize}(\mathbf{z} + f_{\theta}(\mathbf{z}))$$

### 🌟 4. CCTV-Aware Data Augmentation
Single enrollment photos generate 4 synthetic surveillance variants (motion blur, low-res downsample, JPEG compression noise, shadow shifts) to make gallery templates natively resilient to CCTV noise.

### 🌟 5. Industrial FAISS Vector Indexing
Replaced naive loops with `faiss.IndexFlatIP(512)` for sub-millisecond similarity search across thousands of face vectors.
