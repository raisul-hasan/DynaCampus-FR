#!/usr/bin/env python
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
def main():
    ROOT = Path(__file__).resolve().parents[1]
    print("Loading models...")
    arc = FaceEmbedder(model_name="insightface_arcface", providers=["CPUExecutionProvider"])
    fl = FaceLiVTv2Embedder("data/models/facelivtv2", device="cpu")
    print(f"ArcFace params: {sum(p.numel() for p in arc.model.parameters()):,}")
    print(f"FaceLiVTv2-S params: {sum(p.numel() for p in fl.model.parameters()):,}")

    # Validate preprocessing matches official reference
    print("Validating preprocessing...")
    test_crop = np.zeros((112, 112, 3), dtype=np.uint8)
    test_crop[:] = [100, 150, 200]  # BGR
    tensor = rgb_tensor(test_crop)
    rgb = test_crop[:, :, ::-1].copy()
    expected = ((rgb / 255.0) - 0.5) / 0.5
    expected = torch.from_numpy(np.ascontiguousarray(expected.transpose(2, 0, 1)[None], dtype=np.float32))
    assert torch.allclose(tensor, expected, atol=1e-6), "Preprocessing mismatch"
    print("Preprocessing validation: PASS")

    # Setup detector and aligner
    detector = SCRFDDetector(confidence_threshold=0.5, input_size=(640, 640), providers=["CPUExecutionProvider"])
    aligner = FaceAligner()

    # Load a tiny subset of development data
    manifest = ROOT / "data" / "manifests" / "halfface_usable.csv"
    import csv
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    seen = set()
    selected = []
    for r in rows:
        if len(seen) >= 5:
            break
        if r["subject_id"] not in seen:
            seen.add(r["subject_id"])
            selected.append(r)
    print(f"Evaluating {len(selected)} images from {len(seen)} identities")

    # Extract embeddings
    arc_embs = []
    fl_embs = []
    statuses = []
    for r in selected:
        img_path = ROOT / r["path"]
        if not img_path.is_file():
            arc_embs.append(None)
            fl_embs.append(None)
            statuses.append("missing")
            continue
        img = cv2.imread(str(img_path))
        det = detector.detect(img)
        if det is None:
            arc_embs.append(None)
            fl_embs.append(None)
            statuses.append("detect_fail")
            continue
        aligned, _ = aligner.align(det.detected_face, det.landmarks)
        arc_emb = arc.extract_embedding(aligned)
        fl_emb = fl.extract_embedding(aligned)
        arc_embs.append(arc_emb)
        fl_embs.append(fl_emb)
        statuses.append("ok")

    ok = [s == "ok" for s in statuses]
    arc_valid = np.stack([e for e, ok in zip(arc_embs, ok) if ok])
    fl_valid = np.stack([e for e, ok in zip(fl_embs, ok) if ok])
    print(f"Successful extractions: {np.sum(ok)}/{len(statuses)}")
    print(f"ArcFace embeddings shape: {arc_valid.shape}")
    print(f"FaceLiVTv2 embeddings shape: {fl_valid.shape}")

    # Save outputs
    out_dir = ROOT / "results" / "transformers" / f"dev_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    out_dir.mkdir(parents=True)
    np.save(out_dir / "arcface_embeddings.npy", arc_valid)
    np.save(out_dir / "facelivtv2_embeddings.npy", fl_valid)
    meta = {
        "num_images": len(selected),
        "num_successful": int(np.sum(ok)),
        "arcface_mean_norm": float(np.mean(np.linalg.norm(arc_valid, axis=1))),
        "facelivtv2_mean_norm": float(np.mean(np.linalg.norm(fl_valid, axis=1))),
        "arcface_params": sum(p.numel() for p in arc.model.parameters()),
        "facelivtv2_params": sum(p.numel() for p in fl.model.parameters()),
        "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved to {out_dir}")

    # Report
    report = [
        "# FaceLiVTv2-S vs ArcFace Development Comparison",
        "",
        f"Images processed: {len(selected)}",
        f"Successful extractions: {np.sum(ok)}/{len(statuses)}",
        "",
        f"ArcFace parameters: {meta['arcface_params']:,}",
        f"FaceLiVTv2-S parameters: {meta['facelivtv2_params']:,}",
        f"ArcFace mean embedding norm: {meta['arcface_mean_norm']:.6f}",
        f"FaceLiVTv2-S mean embedding norm: {meta['facelivtv2_mean_norm']:.6f}",
        "",
        "Preprocessing validated against official FaceLiVTv2 inference.py reference.",
        "",
        "This is a lightweight development comparison demonstrating:",
        "- Real FaceLiVTv2-S inference with official weights",
        "- Correct RGB preprocessing (vs ArcFace BGR)",
        "- Identical SCRFD alignment and cropping",
        "- Embedding extraction and L2 normalization",
        "",
        "For full paired evaluation with enrollment strategies, see the comprehensive comparison script.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()

from src.recognition import FaceEmbedder
from src.recognition.facelivtv2 import FaceLiVTv2Embedder, rgb_tensor
from src.detection import SCRFDDetector
from src.alignment import FaceAligner
"""FaceLiVTv2-S vs ArcFace development comparison.

Validates preprocessing, loads models, extracts embeddings on subset of development data.
Saves raw predictions and a brief report.
"""
