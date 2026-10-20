# Inspected development cases and defensible explanations

Prepared 21 October 2026. All 12 examples were selected before visual inspection:
the first lexicographic source path in each declared category. None was replaced
because it was visually inconvenient. The full eligible candidate ledger, source
hashes and exact scores are saved in the defense run. These are illustrations of
previously measured outcomes, not 12 additional independent accuracy samples.

Local photo viewer:
`data/defense/defense_development_20261020T210142299692Z/index.html`.
Public cards in `paper_assets/development/defense_v1/` omit photographs.

## Observed examples

| Case | Visual observation | Measured decision mechanism | What can be said |
|---|---|---|---|
| C01: ArcFace helps, fixed rule | Masked, angled probe; enrollment includes frontal and two angled views | Same score threshold 0.377859 and margin 0.15. Top score changes 0.320618 to 0.442239; gap changes 0.125249 to 0.263663. UNKNOWN becomes correct KNOWN | This example supports a template-policy benefit under identical decision settings. Multiple views are a plausible explanation, not an experimentally isolated cause |
| C02: ArcFace harms, fixed rule | Masked probe; two usable enrollment views, one frontal and one profile | Score falls 0.429137 to 0.376741, below the same 0.377859 threshold. Correct KNOWN becomes UNKNOWN | Averaging additional available views can hurt. Do not describe every few-photo template as a three-photo template |
| C03: accepted under both, fixed rule | Masked probe and three unmasked views | Both policies correctly accept; score falls slightly from 0.519131 to 0.497705 while remaining above the threshold | A lower similarity need not worsen the final decision; show unchanged cases as well as improvements |
| C04: genuine unknown rejection | Unmasked probe; displayed candidate belongs to a different source ID | Closest score 0.264252 is below threshold in both policies; no identity is accepted | Nearest neighbor is not the same as accepted identity. This is a selected rejection example, not an FPIR estimate |
| C05: genuine ambiguity | Angled unknown probe; the displayed closest candidate has multiple enrollment views | Few-photo top score 0.419889 exceeds the score threshold, but the top-two gap is 0.123053, below 0.15. State is AMBIGUOUS and accepted ID is empty | The margin rule can refuse a high-scoring search. The 55-identity live fixture also shows the second candidate; no duplicated identity or fabricated near-tie is needed |
| C06: enrolled person rejected | Masked side-view probe and unmasked enrollment views | The supplied ground-truth ID is the closest candidate, yet the few-photo score is 0.321359, below threshold. UNKNOWN is returned | UNKNOWN is a rejection state; it does not prove the person is actually unenrolled |
| C07: capture failure | A masked face is visibly present in the source image | SCRFD returns no usable detection at the fixed operating point; both recorded policies have no embedding or match score. Fresh inference reproduces zero faces | The failure happens before recognition. It remains in attempted-probe counts; do not claim the recognizer misidentified it or assert a visual cause without a controlled test |
| C08: false acceptance, fixed rule | Masked, angled unknown probe; candidate photos have a different supplied source ID | Score changes 0.357947 to 0.379359 and gap to 0.164272. Both gates now pass, so a wrong known ID is accepted | This is an important counterexample: an increased similarity can be harmful. A high-looking quality proxy is not a correctness guarantee |
| C09: ArcFace helps, calibrated policies | Masked angled probe; three enrollment views | Score rises 0.301220 to 0.397848. Both score and margin settings also change between policies | This illustrates a calibrated-policy improvement, not an isolated enrollment-only effect |
| C10: ArcFace harms, calibrated policies | Masked angled probe; three enrollment views | Score rises slightly, 0.341000 to 0.346112, but the few-photo threshold is stricter (0.377859 versus 0.320618) and margin requirement rises from 0.10 to 0.15. KNOWN becomes UNKNOWN | Better raw similarity does not necessarily improve a decision when its operating point also changes. This is essential context for the tied aggregate acceptance counts |
| C11: AdaFace helps, calibrated policies | The same selected masked probe as C01, evaluated by the other checkpoint | Score changes 0.275396 to 0.419932. Thresholds differ; UNKNOWN becomes correct KNOWN | This is paired evidence on the same source probe, not another person or independent sample |
| C12: AdaFace harms, calibrated policies | Masked probe; frontal and angled enrollment views | Score falls 0.424907 to 0.326829; the few-photo threshold is 0.337226. Correct KNOWN becomes UNKNOWN | Additional enrollment does not help every probe. Preserve this counterexample in the defense |

These observations describe visible capture conditions and recorded behavior.
They do not infer identity from appearance, attributes, relatedness or demographic
groups. Ground-truth IDs are the source dataset labels. Source label quality and
pretraining overlap remain limitations. No visual observation establishes which
factor caused a detector or embedding failure.

## Aggregate context for the selected examples

At the identical frozen score/margin settings, the ArcFace sensitivity comparison
has **9 helped and 1 harmed known probe**, changing 131/154 correct known accepts
to 139/154. It also changes unknown false acceptance from 0/149 to 1/149. This is
development sensitivity evidence and must be shown with both outcomes.

With separately calibrated policies, ArcFace has **2 helped and 2 harmed known
probes**, giving 139/154 for both enrollment policies. AdaFace has **18 helped and
5 harmed known probes**, changing 120/154 to 133/154. Thresholds and observed
unknown acceptance/coverage differ; these are descriptive development results.

The frozen final comparison remains separate: score-only one-photo 130/160,
score-only few-photo 150/160, and primary few-photo margin 151/160. The new
one-photo margin policy was not predeclared for that final comparison and must not
be described as if it had been tested there.

## Practical implications to explain

- Several enrollment views can improve some masked-probe decisions without
  backbone retraining, but other probes can be harmed. Enrollment policy should
  be evaluated rather than assumed beneficial.
- A successful nearest-neighbor search is not sufficient for acceptance: the
  score and ambiguity checks can refuse it.
- A refusal can require review or recapture in a future operational design. This
  study does not validate a real attendance or access-control outcome.
- Detection failure and identity-decision failure belong to different stages and
  should not be hidden by reporting only successfully extracted probes.
- Show the false acceptance and uncertainty alongside successful examples. The
  study does not establish zero deployment error or real CCTV performance.
