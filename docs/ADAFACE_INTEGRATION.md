# AdaFace development integration

This research path compares genuine AdaFace and ArcFace inference while preserving
the original frozen ArcFace evaluator and the existing application model choice.
The Streamlit dashboard still uses ArcFace; the new AdaFace backend is used by
`scripts/compare_backbones.py` and is not silently substituted into the app.

## Official assets

- Source: [mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace).
- Pinned commit: `c60eaa786a42c03444f3df7096dbaf9d57ae010d`.
- Architecture: official `ir_50`, 112 x 112 aligned input, 512-dimensional output.
- Checkpoint: the official R50/MS1MV2 link in the pinned README.
- Checkpoint SHA-256: `234da6ce931f821e0a8e920063dc2091d8122ea46285c159d2f25d3f762fecb3`.
- Source, checkpoint, license and retrieval hashes: local ignored
  `data/models/adaface/provenance.json`; copied into completed run metadata.

The source is MIT-licensed. Preserve the model/dataset attribution and review
their applicable terms rather than assuming the source license grants every use.
The existing InsightFace model zoo specifies non-commercial research use.

## Inference contract and compatibility

`src/recognition/adaface.py` accepts uint8 BGR arrays shaped `(112,112,3)` or
`(N,112,112,3)`. It preserves BGR channel order and applies the official
`(value / 255 - 0.5) / 0.5` normalization, then returns normalized features.
The pre-normalization feature norm is also recorded; it is not calibrated or
used as a quality gate in this comparison.

Source and checkpoint hashes are checked before loading. Every expected backbone
tensor must load with `strict=True`; model errors cannot return random features.
The released checkpoint contains an old Lightning callback class as a metadata
key. Restricted PyTorch loading maps that exact name to an inert local class and
rejects any additional unknown globals. `weights_only=True` remains enabled.
The integration does not import or train PyTorch Lightning.

The comparison uses the existing SCRFD five-point alignment for both models.
This differs from the authors' sample MTCNN alignment workflow and must be
disclosed when describing the protocol. It is a controlled shared-crop comparison,
not a reproduction of an official AdaFace benchmark.

## Reproduce

The tested host uses Python 3.13.3, PyTorch 2.9.0+cu126 with CPU inference,
the existing InsightFace/ONNX Runtime dependencies, NumPy, OpenCV, PyYAML and
requests. The experiment uses two CPU threads per recognition backend. This
does not certify a fresh environment installation; dependency cleanup remains.

```powershell
# Requires network; assets are downloaded into the project, not a user cache.
python scripts/fetch_adaface_assets.py
python -m unittest discover -s tests -p test_adaface.py -v
python -m unittest discover -s tests -p test_backbone_comparison.py -v
python scripts/compare_backbones.py
# Use the completed directory printed by the command:
python scripts/verify_backbone_comparison.py results/backbones/<completed-run>
python scripts/render_backbone_evidence.py results/backbones/<completed-run>
```

The existing development manifests, enrollment extraction and feature cache are
required. Both recognizers run on exactly the same selected images/crops, and
ArcFace features must reproduce the existing cache within 1e-6. Source hashes and
detector outcomes must match. The run stops on incompatibility instead of silently
changing which samples are compared. Images and biometric arrays remain private
under existing ignore rules.

## Interpretation

One and up-to-three-photo enrollment use the same original candidates. Score-only
and score-plus-margin rules use the same development objective and declared grid,
with separately selected operating points for each model/policy. All capture
failures remain in attempted denominators. The reserved final identities are
excluded before extraction, and no final feature cache is opened.

Threshold calibration and evaluation reuse development data. Therefore the new
comparison is descriptive. AdaFace/MS1MV2 and buffalo_l/WebFace600K also have
different pretraining and runtime engines. Results compare those released
checkpoints and do not isolate the effect of the AdaFace loss function.

Timing logs retain decode, padding, detection, alignment, quality and recognition
stages. The composed image-to-embedding value sums those measured stages and
excludes search, UI, model loading and integrity hashing. Do not report it as
end-to-end attendance latency or camera FPS.
Source-hash checks precede image decoding, so the decode measurements are likely
to benefit from a warm operating-system file cache. They are not cold-storage
read benchmarks.
