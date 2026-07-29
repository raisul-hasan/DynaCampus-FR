# 📸 CampusFace-BD Dataset Collection Protocol & Guidelines

This document outlines the exact specification for collecting, naming, protecting, and splitting the **`CampusFace-BD`** local campus research dataset according to **Part III** of the DynaCampus-FR blueprint.

---

## 1. 📊 Target Scale & Participants

| Parameter | Target Requirement |
| :--- | :--- |
| **Total Participants** | **60 consenting adults** (students, faculty, staff) |
| **Enrollment Photos** | **15 photos** per participant |
| **Enrollment Video** | ~15 seconds per participant (slow head movements) |
| **CCTV Sessions** | **2 separate sessions on different days** ($C_1$ and $C_2$) |
| **Camera Angles** | **2 fixed cameras** simultaneously ($CAM_A$ frontal, $CAM_B$ diagonal) |
| **Distances** | Near (~2 meters), Medium (~5 meters), Far (~8 meters) |
| **Frame Sampling** | 5 FPS for research evaluation frames |
| **Expected Face Crops** | ~8,000 – 20,000 processed face crops |

---

## 2. 🎥 Physical Camera & Environment Setup

- **Location:** A controlled corridor or room where non-consenting bystanders can be excluded.
- **Camera Heights:** Mount cameras at **2.3 to 2.8 meters** height (typical CCTV height).
- **Camera A ($CAM_A$):** Frontal or near-frontal corridor view.
- **Camera B ($CAM_B$):** Diagonal / side profile view.

```
                  [ CAM_A (Frontal, Height: 2.5m) ]
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Corridor Walk Path                          │
│   (Far: 8m)  ───────>  (Medium: 5m)  ───────>  (Near: 2m)     │
└─────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
                  [ CAM_B (Diagonal, Height: 2.5m) ]
```

---

## 3. 📸 Session Specifications

### A. Enrollment Session (Single High-Quality Photo Capture)
Each participant provides:
- 5 Frontal photographs
- 3 Slight left pose photos
- 3 Slight right pose photos
- 2 Slight upward pose photos
- 2 Slight downward pose photos
- 1 Video (~15 seconds) moving head slowly

### B. CCTV Session 1 ($C_1$ - Baseline Day)
- Daylight / normal indoor lighting.
- Walk toward camera, away from camera, left-to-right, right-to-left.
- Pause briefly at near (2m), medium (5m), and far (8m) positions.

### C. CCTV Session 2 ($C_2$ - Different Day & Variation)
- Recorded on a **different day** to introduce appearance variation (different clothes, hair).
- Lower illumination / shadow conditions.
- Controlled occlusions: glasses, face mask (with consent), small group walking together.

---

## 4. 📁 File Naming & Folder Conventions

Biometric raw files must never be uploaded to GitHub. Store them in `CampusFaceBD_Data/` (added to `.gitignore`).

```
CampusFaceBD_Data/
├── private/
│   ├── identity_map_encrypted.csv
│   └── consent_forms/
├── raw/
│   ├── enrollment/
│   │   ├── S001_E_IMG_001.jpg
│   │   └── S001_E_VIDEO_001.mp4
│   ├── cctv_session_1/
│   │   ├── S001_C1_CAMA_001.mp4
│   │   └── S001_C1_CAMB_001.mp4
│   └── cctv_session_2/
│       ├── S001_C2_CAMA_001.mp4
│       └── S001_C2_CAMB_001.mp4
├── metadata/
│   ├── participants.csv   (subject_id, consent_date, notes)
│   ├── videos.csv         (video_id, subject_id, camera_id, fps, path)
│   ├── frames.csv         (frame_id, video_id, timestamp, path)
│   └── quality.csv        (face_id, blur_score, pose_score, final_quality)
└── splits/
    ├── dev_identities.json  (30 development subjects for Domain Adapter)
    └── test_identities.json (30 evaluation subjects for Open-Set benchmark)
```

---

## 5. 📜 Ethics & Written Consent Form Rules

> [!IMPORTANT]
> **Mandatory Precondition:** No participant recording can begin before written consent is signed and supervisor approval is granted.

1. **Anonymous Subject IDs:** Participants are assigned IDs like `S001` through `S060`. Real names are stored in an encrypted file separate from images.
2. **Right to Withdraw:** Participants can withdraw at any time, and their raw videos/embeddings will be deleted.
3. **Restricted Access:** Raw videos are kept strictly on restricted research storage and never uploaded to public cloud repositories.
