# Draft sections for Manuscript V4 / CCF v2

Two additions and three edits. Terminology follows the manuscript: *Real Data*,
*Full Synthetic*, *Hybrid (Real Eyes)*, *Identity A / B*, and the four models
iTracker, MobileNet-V3, AFFNet, MGazeNet. Every figure cited is in
`results/paper/figures/`; every number is in `numbers.json`.

The privacy measurements are made on the **images**, with a pretrained face
recogniser, so they do not involve the gaze models and do not depend on which
four architectures the manuscript benchmarks.

---

## ADDITION 1 — Methods, new subsection after *Privacy Preserving Data Synthesis*

### Privacy Evaluation Protocol

The synthesis pipeline above is intended to remove identity. Whether it does is
an empirical question, and one that a utility measurement cannot answer. We
therefore evaluate the released artefacts directly, with an attacker model, a
pretrained recogniser, and a control that establishes what "no identity signal"
scores on this dataset.

**What is released.** The shareable artefact is the set of cropped streams the
models consume — face ROI, left and right eye ROIs, and the face grid — not the
source video. All attacks operate on exactly those crops.

**Attacker models.** Two attackers are distinguished, because they yield
different answers on identical data:

- **A1 — enrolment photographs.** The attacker holds ordinary photographs of the
  candidate individuals (a gallery of the original crops) and asks which
  enrolled person a released crop belongs to. This is the classical
  re-identification threat.
- **A2 — release only.** The attacker holds nothing but the shared dataset.
  They cannot name anyone unaided, but they can ask whether the release
  partitions into individuals, and a single labelled example then names a whole
  cluster. For a dataset intended for public release, A2 is the operative
  attacker.

We additionally report an **adaptive** attacker (A1 with knowledge of the
anonymisation method, able to run it on their own data) in §Results.

**Recogniser.** InsightFace `buffalo_l` (ArcFace, w600k_r50), used without any
adaptation to this dataset. Cosine similarity against per-subject gallery
templates. No component is trained on CognitiveGaze.

**Metrics.** True-accept rate at a fixed false-accept rate of 1e-3
(TAR@FAR=1e-3) and the signal-detection separation d′ are primary. Rank-1
accuracy is reported for completeness only: on a single-session dataset it is
inflated by session structure (below). For linkage we report the adjusted Rand
index (ARI) of an unsupervised partition against the true subject labels, and
the number of clusters the procedure selects on its own.

**The session-nuisance floor.** Each participant was recorded in one session, so
illumination, seating and background are constant within a participant and differ
between participants. Any identification metric on such data therefore has a
non-zero baseline that owes nothing to the face. We measure it by running every
attack unchanged on a 120 px face-free patch of the same frames. The floor is
**Rank-1 50.5%** (chance 5.6%), **TAR@FAR=1e-3 1.6%**, and **ARI 0.089** with
2 clusters selected rather than 18. Closed-set Rank-1 is thus not evidence of
de-identification on this kind of data; verification at a strict operating point
and linkage are.

**Alignment.** Face detection and alignment are taken from the *original* frame
and applied to the anonymised one, so an operator is never credited for defeating
the detector — a failure to detect is not privacy, and it would also break the
gaze pipeline. Where an attack is run with the detector applied to the released
image instead ("self-aligned"), it is labelled as such and the number of frames
surviving detection is reported alongside.

**Uncertainty.** Frames of one participant are not independent. All intervals are
95% subject-level cluster bootstrap (2,000 replicates, resampling participants
with replacement, gallery held fixed).

---

## ADDITION 2 — Results, new subsection before *The Privacy-Utility Trade-off*

### Identity Leakage under Anonymisation

**The eye region alone identifies the participant.** Before any operator is
considered: an unmodified eye ROI, 120 px, given to a pretrained recogniser with
no adaptation, verifies identity at **66.6%** (left) and **62.2%** (right)
TAR@FAR=1e-3 — against a 1.6% floor (Figure 1). Rank-1 is 91–98%. The eye ROI is
not a de-identified crop of the face; it is the part of the face that identifies
best per pixel.

**Consequence for the Hybrid protocol.** The Hybrid strategy releases the
original, unaltered eye ROIs by design — that is what preserves the gaze signal.
Those ROIs are the crops measured above. On the identity axis, therefore, Hybrid
inherits the leakage of unprocessed data: **62–67% TAR@FAR=1e-3, roughly forty
times the floor**, before the synthetic face is considered at all.

**Full Synthesis does remove the correspondence to the real face.** With the eye
region synthesised as well, the same attack falls to **0.2% / 1.8%**
TAR@FAR=1e-3, at or below the floor; the face ROI falls from 100% to **22.0%**
[5.2, 44.5] for Identity A and **23.3%** [8.2, 41.8] for Identity B. The two
templates agree, so this is a property of the method rather than of a particular
source face — the same conclusion Table 6 reaches on the utility axis, reached
independently on the privacy axis.

**But the release remains perfectly linkable.** Under attacker A2, an
unsupervised partition of the released corpus recovers the participants exactly:
**ARI 1.000 at the correct k = 18**, for both templates and both alignment
regimes, against a floor of 0.089 at k = 2 (Figure 5). Identity swapping is
*per-participant consistent*: each participant is mapped to the same synthetic
appearance every time, which destroys the correspondence to their real face while
leaving the released set internally self-consistent. The same eye ROIs that score
0.2% against attacker A1 score **42.8% / 62.5%** against A2 (Figure 3).

**The linkage is not an artefact of what the face crop happens to contain.** A
face ROI retains channels the synthesis never edits — hair at the crop border,
spectacles, skin tone, head pose and face outline — and the session-nuisance
floor does not control for them, because it is a patch of the *room* and those
channels are attributes of the *person*. We therefore repeat the linkage attack
on the 120 px eye ROI, which excludes hair, ears, neck, clothing and the face
outline, and whose pixels are themselves synthetic in the Full Synthetic arm
(Figure 5b). Linkage survives: **ARI 0.68–0.79** at k = 18 for the two synthetic
identities, against **0.84–0.85** for unmodified eye ROIs and a floor of 0.089,
with 79–86% of frames assigned to the correct participant. These eye-ROI figures
assume an attacker who knows the cohort size — on the eye ROI the cluster count
is not recovered unaided, whereas on the face ROI it is (k = 18 selected by
silhouette). For a public release the cohort size is stated in the documentation,
so the assumption is mild, but it is an assumption and the face ROI does not need
it. Identity synthesis
thus removes almost none of the within-corpus separability, and what remains
cannot be attributed to context outside the edited region. The residual channels
— illumination, skin tone, periocular geometry and crop scale — are the same
signals the gaze models depend on, which is why the axis trades: linkability and
gaze utility share a representation.

**Realism buys detectability, not unlinkability.** 1073–1080 of 1078 frames
survive self-aligned face detection under synthesis, against 119 for strong blur,
so the released corpus is maximally easy to align. This is a property of the
synthesis and it is what keeps the gaze pipeline working; it is not, however,
what makes the corpus linkable. All participants are mapped onto the same
synthetic appearance, so the swapped identity carries no discriminative signal,
and the eye-ROI control above reproduces the linkage with the crop border and the
face outline removed. The correct statement is the structural one: face swapping
operates on inner-face texture, and identity in a released corpus is carried by
channels outside that edit domain. The conclusion generalises to any method with
the same edit domain, independently of how well it is implemented.

**Metadata leaks before any pixel is read.** The released crop dimensions are
constant per participant. Face ROI size alone yields an expected Rank-1 of 50.0%;
face size, eye size and inter-ocular distance together yield **94.4%** (4.06 of
4.17 bits) without reading a single pixel. A release must normalise crop
geometry; ours does, and all figures above are pixel-based.

**An adaptive attacker gains little.** An attacker who knows the operator and
holds paired (original, synthesised) imagery of *other* participants can fit a
de-anonymisation map and apply it to a participant they could not train on. Across
mean-shift, orthogonal Procrustes and ridge maps, fitted leave-one-participant-out,
the best gain is **+3.3 points** on the face ROI (19.9% → 23.2%) and nothing on the
eye ROIs. The same maps reach Rank-1 100% when fitted *including* the target — they
memorise identities rather than invert the operator. The protection is a
generalisation barrier rather than an information-theoretic one: the residual
identity is present and linearly extractable given the target's own paired data,
and what fails is transfer to an unseen person.

#### Table N — Identity leakage across release strategies
Attacker A1 (enrolment photographs), TAR@FAR=1e-3 with 95% subject bootstrap CI.

| Released stream | Real Data | Full Synthetic (A) | Full Synthetic (B) | Hybrid (Real Eyes) |
|---|---|---|---|---|
| Face ROI | 100% | 22.0% [5.2, 44.5] | 23.3% [8.2, 41.8] | 22.0% (face is synthetic) |
| Left eye ROI | 66.6% [53.6, 79.6] | 0.2% [0.0, 3.0] | 0.6% [0.0, 3.1] | **66.6%** (unaltered) |
| Right eye ROI | 62.2% [49.0, 84.6] | 1.8% [0.1, 5.7] | 1.1% [0.0, 4.8] | **62.2%** (unaltered) |
| *Session-nuisance floor* | *1.6%* | *1.6%* | *1.6%* | *1.6%* |
| Linkage, face ROI, ARI @ k=18 (A2) | 1.000 | 1.000 | 1.000 | 1.000 |
| Linkage, left eye ROI, ARI @ k=18 (A2) | 0.844 | 0.684 | 0.737 | 0.844 (unaltered) |
| Linkage, right eye ROI, ARI @ k=18 (A2) | 0.853 | 0.789 | 0.761 | 0.853 (unaltered) |
| *Linkage floor* | *0.089* | *0.089* | *0.089* | *0.089* |

---

## EDIT 1 — Contributions bullet 4

The current bullet claims identity removal and public shareability. Suggested
replacement:

> **Privacy Preservation Data Synthesis, and its measured limits.** We synthesise
> anonymised datasets with a GAN framework (SimSwap++) and quantify the
> privacy–utility trade-off on *both* axes rather than assuming the privacy side.
> Using a pretrained face recogniser and a session-nuisance control, we show that
> full facial synthesis removes the correspondence to the participant's real face
> (TAR@FAR=1e-3 falls from 100% to 22%, and to the 1.6% floor on the eye region)
> but costs 1.35–3.26 cm of gaze accuracy, while the Hybrid strategy preserves
> accuracy (−0.03 to +0.69 cm on three of four architectures) precisely because it
> retains the unaltered eye region — which is itself sufficient to identify the
> participant at 62–67%. We further show that both strategies leave the released
> corpus exactly linkable by participant. We therefore report the conditions a
> shareable clinical gaze dataset must satisfy, rather than claiming they are met.

## EDIT 2 — the Hybrid claim in *Privacy Preserving Data Synthesis*

> ~~This framework potentially establishes a paradigm that preserves clinical
> accuracy with identity protection, thereby offering a viable pathway for sharing
> clinical datasets publicly.~~

Replace with:

> This protocol tests whether the "Context" (Face) can be anonymised while the
> "Signal" (Eyes) remains genuine. Section *Identity Leakage under Anonymisation*
> evaluates what each protocol protects; as shown there, retaining genuine eye
> ROIs retains the identifying information they carry, so the Hybrid strategy is a
> utility-preserving protocol rather than a privacy-preserving one, and is not by
> itself sufficient for public release.

## EDIT 3 — Limitations and Conclusion

Add:

> Neither synthesis protocol produces a corpus that can be released publicly
> without further control. Full synthesis removes the link to the real face but at
> a gaze cost of 1.35–3.26 cm; the Hybrid protocol preserves gaze at the price of
> releasing genuine ocular biometrics; and both leave the corpus linkable by
> participant, so a single labelled example compromises a whole cluster. The
> linkage is structural rather than incidental: it survives on a 120 px ocular
> crop containing none of the contextual cues a face crop carries, and the
> channels that carry it — illumination, skin tone, periocular geometry, crop
> scale — are the channels gaze estimation reads, so any method whose edit domain
> is inner-face texture inherits this limit. A
> deployable route would need to break the per-participant consistency of the
> transformation, or to protect the ocular region while preserving the gaze
> geometry within it — an axis we examined and found to trade linearly, with no
> setting that was both private and accuracy-preserving. Controlled-access release
> under an agreement remains available and is what we recommend for this dataset.

---

## ADDITION 3 — Ethical Considerations

Placed before Limitations. IMWUT expects this section to be specific to the study
rather than generic; ours can be, because the release decision was made from
measurements reported in this paper.

### Ethical Considerations

**Approval and consent.** The study was approved by the University of Warwick
Biomedical and Scientific Research Ethics Committee (BSREC 08/22-23). All
participants gave written informed consent. The consent form states that raw
images will not be published unless the participant is asked and agrees; no
participant has been approached for such a release, and none of the material
described below is raw imagery.

**Anonymisation was evaluated, not assumed.** The privacy claims in this paper
are measurements on the released artefacts, made with a pretrained recogniser
that was never trained on this dataset, and reported against a control that
establishes what "no identity signal" scores on single-session data. We report
the attacks that succeed as prominently as the ones that fail, including on our
own preferred protocol.

**What the measurements imply for release.** The two protocols fail in different
ways and neither yields an anonymous corpus.

- The *Hybrid* protocol releases unaltered ocular imagery, which verifies
  identity at 62–67% TAR@FAR=1e-3. These are photographs of the participant's
  eye region and are recognised as such. Ocular imagery is biometric data under
  UK GDPR Article 9 when used for identification, and it is demonstrably usable
  for identification here.
- The *Full Synthetic* protocol removes the correspondence to the real face — to
  the control floor on the eye region — but leaves the corpus partitionable by
  participant at ARI 1.000 on the face ROI and 0.68–0.79 on the eye ROI. A
  corpus that can be partitioned by individual is pseudonymised, not anonymised,
  and remains personal data. One labelled example names a whole cluster.

We therefore describe the Full Synthetic corpus as **pseudonymised, materially
de-identified imagery with a measured residual**, and we release it on terms
matched to that description rather than on terms that presume anonymity.

**Risk to participants.** The residual re-identification risk is that someone
holding an ordinary photograph of a participant verifies them from the synthetic
face ROI at 22.0% TAR@FAR=1e-3, roughly fourteen times the control floor. This is
the risk the access controls in *Data Availability* are sized against. Attribute
inference and adaptive attacks were also measured and are reported in §Results.

**Participant burden and compensation.** [Retain the manuscript's existing text;
add reimbursement details if the original protocol specified any.]

**Researcher access.** Source recordings remain on University-managed
infrastructure and are not copied to personal storage. Pixel-level processing for
this paper was performed on that infrastructure. The linkage between participant
identifiers and personal details is held separately under the principal
investigator's control.

**Dual use.** The attacks in this paper use only published, pretrained models and
standard clustering, applied to our own participants' data with their consent and
with committee approval. We report no new attack technique. The measurements are
intended to let dataset authors evaluate their own releases before publishing
them, and we note that the same evaluation applied to previously released
face-swapped corpora would likely return similar results.

---

## ADDITION 4 — Data Availability

Placed after Ethical Considerations, or wherever the venue positions it.

### Data Availability

Release is tiered by measured risk. The tier boundaries follow the identity
measurements in §*Identity Leakage under Anonymisation*, not an assumption about
what synthesis removes.

**Open.** Per-frame gaze estimates and errors, all attack scores and summary
tables, experiment configuration, and the full analysis and evaluation code are
deposited in the University of Warwick research data repository under an open
licence with a DOI. Everything is keyed to a pseudonymous participant number.
This tier contains **no imagery of any kind and no face-recognition embeddings** —
an embedding is biometric data and is matchable even though it is not a picture —
and no per-participant attributes. It is sufficient to reproduce every numerical
claim in this paper.

**Controlled access.** The Full Synthetic imagery, in which both the facial and
the ocular pixels are replaced by a synthetic identity, is available to named
academic researchers on written request under a data access agreement signed by
the requester and their institution. The agreement prohibits redistribution, any
attempt at re-identification including matching against external or web imagery,
and any use beyond the stated research purpose. Requests are decided by the
principal investigator and access is logged. It is not a download link, because
the material is measurably re-identifiable at fourteen times the control floor
and exactly linkable by participant.

**Not released.** The source recordings and the original crops; **any arm that
retains original participant pixels**, which includes the Hybrid arm and the
low-strength blur and pixelation arms; any face-recognition embedding computed
from participant imagery; and the identification key. The Hybrid arm is our
best-utility configuration and is excluded on privacy grounds alone: its eye ROIs
are unaltered photographs of the participant.

No participant imagery appears in any figure in this paper.

---

## Figures to include

| Figure | File | Shows |
|---|---|---|
| Identity leakage by stream | `fig1_periocular_leakage.pdf` | eye ROI vs face ROI vs floor, with CIs |
| Two attackers | `fig3_threat_models.pdf` | same crops, A1 vs A2, 0.2% → 43% |
| Linkage, two crops | `fig5_linkage_release_only.pdf` | (a) face ROI, ARI with frames-detected; (b) eye ROI control, ARI 0.68–0.85 |

## Not included, and why

- The non-inferiority analysis and the enrolment-condition experiments use a
  different set of backbones and a per-participant calibration protocol that this
  manuscript does not describe. They are a separate contribution and would need
  their own methods text.
- Conventional operators (blur, pixelation, occlusion) are measured on the privacy
  axis only. They are dominated there — with eye ROIs preserved they leave
  identification at 95–99% — so no utility measurement is needed to rule them out.
