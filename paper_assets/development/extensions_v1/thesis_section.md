# Additional development experiments

## Synthetic one-photo enrollment

To distinguish additional image information from synthetic variation, the same
55-person development gallery was constructed from one frontal original,
that original plus five deterministic mild variants, or up to three original
unmasked views. Both pretrained checkpoints used identical source crops and
unchanged probes. There were 154 known masked attempts and 150 unknown attempts;
six known and one unknown extraction failure remained in the attempted counts.
Templates averaged unit embeddings and were normalized again after pooling.

The five variants comprised blur sigma 1, cubic reduction to 56x56 and restoration
to 112x112, JPEG quality 50, and gamma values 0.7 and 1.4. Each variant and the
original had equal weight. These operations change capture quality rather than
creating new pose information or simulating masks. Policies were evaluated both
with the existing one-photo thresholds fixed and with separate development
calibration. The latter reuses development probes and is descriptive.

With separately calibrated score-only rules, augmentation changed ArcFace correct
acceptance from 129 to 128 of 154 and AdaFace from 112 to 111. Up to three original
views instead achieved 138 and 122. With score/margin rules, augmentation changed
ArcFace from 139 to 135 and AdaFace from 120 to 123. The small AdaFace gain has
a descriptive paired identity-bootstrap interval that touches zero. The tested
augmentation recipe therefore provides no consistent replacement for additional
original views. This does not establish that all augmentation recipes are ineffective.

## Gallery growth under fixed probes and thresholds

A fixed ten-identity core supplied 28 masked known attempts, while the same 150
unknown attempts were retained at all gallery sizes. The two known and one
unknown extraction failures were retained. Additional enrolled identities served
only as background candidates. The ten-identity and full 55-identity galleries
were each scored once; five seeded 25-identity compositions sampled 15 background
identities from the remaining 45 development-known identities. All used the same
few-photo templates and previous 55-identity thresholds.

Both checkpoints retained 26/28 known Rank-1 matches across all compositions.
ArcFace correct acceptance stayed 23/28 for both rules. AdaFace's margin rule
changed from 24/28 at ten identities to 23/28 at 25 and 55 identities. Unknown
false-accept counts depended on the rule and composition. ArcFace score-only
false accepts changed from zero at ten identities to one at 55. AdaFace's fixed
margin rule changed from two to zero because additional gallery competitors can
increase ambiguity. Five composition variations do not supply five independent
participant samples; these results describe this small fixed core only.

## Runtime and engineering relevance

Warm CPU timings used two inference threads per engine and the actual persistent
gallery implementation. There were 30 end-to-end enrollment calls and 60
image-to-decision calls per checkpoint, plus 300 search/decision calls at each of
three gallery sizes. Enrollment included file decode, detection, alignment,
quality estimation, embedding, template normalization and persistent insertion
into a 55-entry isolated gallery. Recognition included file decode through the
four-state decision. The restoration of the enrollment benchmark gallery was
outside its timed interval.

ArcFace's median/p95 enrollment time was 136.97/
173.15 ms, and image-to-decision time was
143.81/
175.57 ms. AdaFace's corresponding recognition
time was 207.03/
241.83 ms. Actual ArcFace gallery search at 55
identities had median/p95 0.0637/
0.0708 ms. This separates small-gallery search cost
from image processing, which dominates the measured path.

The timing comparison includes different pretrained checkpoints and different
runtime engines; it does not isolate a loss function. Model loading, UI transfer,
camera capture and network transport were excluded, so these values are neither
browser latency nor validated camera throughput. All repetitions are latency
measurements on fixed images, not extra accuracy observations.

All policies, conditions, outcomes and negative findings are retained in the
[full evidence report](README.md), alongside raw scores and separate verification.
