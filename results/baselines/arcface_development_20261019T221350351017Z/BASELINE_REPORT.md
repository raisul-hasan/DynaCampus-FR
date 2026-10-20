# First ArcFace baseline

**Development / engineering results only. Final-test identities remain untouched.**

Run: `arcface_development_20261019T221350351017Z`

Real SCRFD detection, five-point alignment, ArcFace 512-dimensional normalized embeddings and exact cosine search.
One original unmasked frontal image enrolls each halfFace known identity. No adapter, augmentation or quality gate is applied.

| Dataset | Enrollment succeeded | Known probe attempts | Rank-1, including pipeline failures | Rank-1, successful embeddings with gallery mate |
|---|---:|---:|---:|---:|
| halfface | 55/55 | 251 | 96.81% | 99.18% |
| icfd_sample | 5/5 | 100 | 100.00% | 100.00% |

## halfFace open-set development calibration

- Development identities: 55 enrolled and 27 unenrolled.
- Reserved test identities: 55 enrolled and 27 unenrolled; none evaluated.
- Non-mated probe attempts: 150; successful embeddings: 149.
- Similarity threshold: 0.3887378871, calibrated for at most 1% empirical FPIR among successfully embedded development unknowns.
- Development conditional FPIR: 0.67%.
- Correct known acceptance including pipeline failures: 89.64%.
- These calibration results are not held-out accuracy and do not establish a guaranteed population FPIR.

## halfFace known probes by mask condition

| Condition | Attempts | Successful embeddings | Correct rank-1 | Rank-1 including failures |
|---|---:|---:|---:|---:|
| unmasked | 97 | 97 | 97 | 100.00% |
| masked | 154 | 148 | 146 | 94.81% |

## Timing and failures

- halfface: 449/456 successful embeddings. Mean detector time 96.2 ms/source crop; mean recognizer time 49.2 ms/aligned face.
- icfd_sample: 105/105 successful embeddings. Mean detector time 96.1 ms/source crop; mean recognizer time 48.6 ms/aligned face.
- CPU ONNX Runtime was used; these timings are image-inference measurements, not live-video FPS.

## Limits and reproducibility

- ICFD uses five development people from the public sample; full ICFD access remains pending.
- A fixed neutral border is applied before detection because the datasets provide tight face crops. The initial unpadded engineering smoke run failed detection; preprocessing was revised using development images only.
- All original-file SHA-256 values are verified before inference. Source exclusions and manifests were prepared before recognition.
- Failed detection or ambiguous multiple-face input is retained as a pipeline failure rather than silently resized or assigned a synthetic embedding.
- No claim is made about video tracking, domain adaptation, Bangladesh-wide performance, independent capture sessions, or absence of pretraining overlap.
- Exact duplicate filtering does not establish that all remaining frames/images are independent.
- See config.yaml, split_snapshot.json, extraction.csv, predictions.csv, threshold.json and run_info.json for the evidence.
