# Defense evidence and demonstration plan

Prepared 21 October 2026. Target submission: 30 October 2026, Asia/Dhaka.

## The claim to defend

> We experimentally evaluated how original-photo enrollment and open-set
> decision choices affect masked face identification using a fixed pretrained
> recognizer. Few-photo enrollment provides the main observed benefit. We also
> measured limitations under controlled degradation and implemented the evaluated
> still-image workflow with persistent, training-free gallery updates.

Our algorithms use established components. The contribution is the controlled
study, attributable findings and reproducible implementation. The literature
comparison explains overlap; it cannot prove universal absence of prior work.
See [the contribution audit](NOVELTY_AUDIT.md).

## Physical evidence to bring

Bring the laptop with local checkpoints, permitted development images, all saved
results and an offline copy of the repository. Test the exact defense setup before
the event. Keep a slide/PDF copy of existing charts and genuine app screenshots,
plus a short local screen recording as a backup; the backup recording is still
to be prepared. A live demonstration and backup should use an isolated gallery.
Check source terms before displaying or distributing source face images; the
existing saved evidence screenshots have participant photo previews disabled.

The minimum evidence packet is:

| Question the examiner can ask | Concrete artifact to open | What it establishes |
|---|---|---|
| Where did the data come from? | `docs/DATASET_START.md`, local source/license records and manifests | Original versus derived data, exclusions, counts and provenance |
| Were identities separated? | `configs/splits/arcface_baseline_v1.json`, final `freeze.json` and `evaluation_manifest.csv` | Within-dataset development/test separation and known/unknown roles |
| What actually changed? | Final comparison table and paired-effects table | Enrollment policy and decision-rule differences with identical probes |
| What is your measured benefit? | `paper_assets/heldout/final_v1/README.md` and `summary.csv` | One-photo 130/160, few-photo score 150/160, primary 151/160 correct known acceptance |
| Did you test unknown people? | Final unknown predictions and state counts | Primary 0/151 observed false known accepts, with explicit sample limits |
| Is every extra component helpful? | Decision and pooling ablations | Small uncertain margin gain; no selected quality gate; no consistent tested weighting gain |
| What happens when images worsen? | `paper_assets/development/robustness_v1/README.md` and source tables | Blur, resolution and JPEG effects under fixed settings; synthetic tests only |
| Does it really execute? | Existing Streamlit still-image enrollment/recognition screens | Actual model inference, gallery search and decisions |
| Does enrollment retrain the backbone? | Gallery insertion logs and before/after checkpoint hashes | New template insertion while recognition checkpoints stay unchanged |
| Can someone verify your numbers? | Final and pooling verifier output, per-probe CSVs and configuration/hash receipts | Recomputable scores and decisions tied to source artifacts |
| What is original relative to papers? | `docs/NOVELTY_AUDIT.md` | Explicit prior-work boundary and the experimental contribution being claimed |

The final run is `results/final/heldout_20261019T231512869782Z`.
The canonical app evidence is indexed in `paper_assets/development/demo_v1/README.md`.
The new pooling run is linked from `paper_assets/development/pooling_v1/README.md`.
The genuine AdaFace/ArcFace comparison, separately calibrated policies, CPU stage
timings and complete development failure lists are now available in
`paper_assets/development/backbones_v1/README.md`. Its results are development
evidence; the application still uses ArcFace.

## An eight-minute demonstration

This is a script for the defense, not a record that every step below has already
been rehearsed. Use development/demo inputs; never tune thresholds on final data.

| Time | Action | Evidence and explanation |
|---|---|---|
| 0:00-0:45 | State the research question and show the actual implemented pipeline | SCRFD, five-point alignment, frozen ArcFace, original-photo template, cosine matching and rejection |
| 0:45-2:00 | Open the final comparison graph and numeric table | Same 160 known and 151 unknown attempts; show 130, 150 and 151 correct known acceptances |
| 2:00-3:30 | Enroll an example identity and upload a distinct masked development probe | Actual recognition; show scores and accepted identity. Label it a selected example, not an accuracy estimate |
| 3:30-4:15 | Upload a development unknown probe | Show refusal and no accepted identity; one example does not estimate FPIR |
| 4:15-5:00 | Reload, remove the enrolled identity, and retry the earlier known probe | Persistence, removal and the resulting change to unenrolled status |
| 5:00-6:00 | Show one-photo versus few-photo and one helped/one harmed case from the defense pack | Twelve cases have disclosed selection, source hashes and full-precision values; distinguish fixed rules from separately calibrated policies |
| 6:00-7:00 | Show degradation results and a failure example | Explain why reduced resolution or compression can cause rejection or detection failure |
| 7:00-8:00 | Run a cached verifier and show the prior-work comparison | Trace the displayed result to raw predictions; explain the contribution boundary |

The original app supports enrollment with up to three original photos. The new
separate `defense_app.py` entry point provides paired evidence cards and an
isolated real-inference walkthrough. Its case view reads saved research evidence;
its gallery walkthrough runs fresh target/probe inference. The original screenshot example
uses one gallery identity and therefore cannot demonstrate two-candidate ambiguity.
To show AMBIGUOUS, prepare a genuine multi-identity development case. To show
INSUFFICIENT_QUALITY, explicitly label a manual quality-gate illustration: the
selected research gate is disabled. A capture failure is an extraction outcome,
not a fifth decision-engine state.

Start the app with an isolated gallery in PowerShell:

```powershell
$env:DYNACAMPUS_GALLERY_DIR = 'F:\Thesis Final\DynaCampus-FR\data\demo\defense_gallery'
python -m streamlit run app.py
```

Choose the frozen research preset. Enable tight-face-crop preprocessing for
dataset-style cropped faces. Run model loading, enrollment and reload without
internet before relying on offline operation; a clean-install offline deployment
has not yet been certified.

For a quick numeric evidence check, without new image inference:

```powershell
python scripts/verify_enrollment_pooling.py results/enrollment_pooling/pooling_development_20261020T181729489649Z
# The final audit also reads original bytes to check their hashes:
python scripts/verify_final_evaluation.py results/final/heldout_20261019T231512869782Z
```

## Real-life relevance and the level of evidence

| Practical situation | Result demonstrated here | Remaining gap before a deployment claim |
|---|---|---|
| An organization enrolls a new consenting member | Template insertion, persistence and removal without backbone retraining | Governance, representative evaluation and operational maintenance |
| An enrolled person supplies a masked photograph | Held-out masked-probe comparison supports a benefit from multiple enrollment views | New cameras, capture conditions and target-population validation |
| An unenrolled person submits a photograph | Unknown rejection is measured, with 151 held-out unknown probes | More independent unknown identities and uncertainty suitable for the intended operating point |
| A low-resolution or compressed image arrives | Controlled stress tests expose capture and recognition failures | Actual acquisition-domain images; synthetic degradation is a proxy |
| Several people look similar | The top-two margin can refuse close candidates; its net test benefit is small | More hard impostors and representative ambiguity cases |
| A camera processes people continuously | No validated temporal result | Consented video, tracking evaluation, end-to-end latency and sustained throughput |

For defense, describe these as operational motivations and prototype behavior.
Do not extrapolate 94.38% correct known acceptance to attendance accuracy,
automatic door access, safety, camera FPS or Bangladesh-wide reliability.
The data do not establish those outcomes.

## Honest answers to likely questions

**What is novel?** Our contribution is the controlled experimental study and
its reproducible evidence about enrollment and rejection choices. We use existing
recognition and aggregation methods and explicitly compare them with prior work.
We do not claim a new backbone or the invention of dynamic enrollment.

**What made the difference?** With calibrated methods on the same held-out
probes, the few-photo score policy accepts 20 more known probes than one-photo
score decisions. Margin decisions add one net acceptance; the interval includes
zero. That makes enrollment the stronger supported finding.

**Why not say 100% security?** Zero observed false accepts among 151 probes
cannot establish zero future error. The unknown images come from 27 source
identities and are not 151 independent people.

**Why no campus CCTV results?** Recording was unavailable. The completed scope
uses public still images, masked probes, controlled degradation and a working
still-image prototype. CCTV and temporal performance remain future validation.

**Why keep negative findings?** They identify added complexity that was not
consistently beneficial under this protocol. They also prevent attributing the
enrollment gain to a quality gate or adapter that did not produce it.

## Prepared defense materials

The [defense pack](../paper_assets/development/defense_v1/README.md) now contains
12 disclosed, inspected examples and full-precision values. The local photo
viewer and 149-second backup video are under ignored data/defense. The backup
uses cards/charts and five genuine previous browser screenshots; it is labelled
as an assembled evidence video, not a new live recording.

Run `python -m streamlit run defense_app.py --server.address 127.0.0.1 --server.port 8521`.
The separate defense entry point has an evidence-case view and a real-inference
gallery walkthrough, preserving the original app.py and its recorded source
hashes. The 54 cached background identities and fresh target/probe inference are
explicitly labelled. It demonstrates a real two-candidate ambiguity in a
55-identity gallery, exact persistence, removal and capture failure. Backend and
AppTest checks pass all ten steps. Fresh browser capture remains pending because
this session has no browser connection.

The case analysis distinguishes score changes from decision changes and shows
a false acceptance as well as successful rejections. See
[the inspected explanations](DEFENSE_CASE_ANALYSIS.md).

## Still to produce

- Search/decision and UI timing beyond the newly measured image-to-embedding
  stages, if required for the final claim. Do not combine separately measured
  stage medians and call their sum a measured end-to-end median.
- A concise paper and defense deck tied to these exact results and limitations.
- Fresh browser screenshots and a live screen recording of the new walkthrough,
  followed by an end-user offline rehearsal on the defense laptop.
