# Session checkpoint - 20 October 2026

Work is paused for the day at the user's request. Paper writing is deferred.

## Verified work saved

- Real SCRFD detection, alignment and frozen ArcFace features.
- One-photo versus up-to-three-photo enrollment and calibrated open-set decisions.
- Controlled development tests for blur, resolution and JPEG compression.
- Frozen held-out halfFace result: 151/160 correct known acceptances (94.38%),
  with 0/151 observed false known acceptances. This is a limited public-dataset
  result, not a population error guarantee.
- Reserved ICFD public-sample engineering check: 98/100 closed-set retrieval.
- Persistent dynamic gallery and actual browser enrollment/recognition evidence.
- Results, figures, captions and screenshots indexed in `paper_assets/README.md`.

Raw face images, biometric feature arrays and model weights stay local under
the existing ignore rules. Reproduction still requires those local inputs.

## Starting point for tomorrow

The user cannot obtain video recordings in the remaining time. Continue with
still-image experiments; do not make real CCTV or temporal-performance claims.
The next proposed task is to check official AdaFace pretrained weights and
compatibility, then compare against ArcFace on development data if feasible.
AdaFace has not yet been downloaded, integrated or evaluated in this session.

Other proposed image-only work: enrollment averaging versus quality weighting,
lighting/noise/occlusion stress tests, failure analysis, and complete image
processing timing. These are options, not completed experiments.

Preserve the current held-out settings and predictions. That test set has already
been examined; further model comparisons on it must be labelled exploratory.
No domain adapter training, local CampusFace-BD dataset, validated video tracking
or transformer comparison has been completed.
