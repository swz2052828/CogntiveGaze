# CCF draft — text for the four blank sections and the privacy axis

Target: `ccf_manuscript_v2.docx`. The blanks are Abstract, Introduction, Related
Work, and Limitations and Conclusion; the privacy axis does not yet exist and is
added as one Methods subsection plus one Results subsection.

Every number here comes from `runs/anon_swap_utility` (utility) or
`results/anon_*` (privacy), via `results/paper/numbers.json`. Nothing is carried
over from the earlier pipeline.

Terminology follows the draft: *Real Data*, *Full Synthetic*, *Hybrid (Real
Eyes)*, *Identity A / B*.

---

## A — Abstract (266 words)

Trim target if the venue caps at 250: the CognitiveGaze sentence can lose "seven
oculomotor tasks" and the final sentence can drop the tiered-release clause.

> Smartphone eye tracking moves oculomotor assessment out of the clinic, but what
> it records is video of the participant's face — which is what blocks the
> dataset sharing the field depends on. GAN face replacement is the common
> answer, and recent work reports that it hides identity effectively: swap the
> face, keep the gaze. We test that claim on both axes.
>
> We introduce CognitiveGaze: 18 participants, seven oculomotor tasks, recorded
> simultaneously by a smartphone and an EyeLink 1000, so every smartphone frame
> carries a physiological ground-truth gaze coordinate rather than a
> screen-target label. Utility is measured by retraining four gaze architectures
> on de-identified data. Privacy is measured on the released crops themselves,
> with pretrained recognisers, two attacker models, and a control establishing
> what "no identity signal" scores on single-session data.
>
> The axes do not trade as assumed. Retaining the genuine ocular region preserves
> accuracy in every architecture — within 0.09–1.32 cm of baseline, against
> 2.16–6.27 cm for full synthesis — yet those same crops verify identity at
> 62–80% TAR at FAR 1e-3 against a measured 1.6% floor, under two unrelated
> recognisers. The pixels that make the protocol work are the pixels that
> identify. Full synthesis does break the link to the real face, but an attacker
> holding only the release still recovers exactly the right 18 people, and does
> so from a 120 px ocular crop containing no hair, ears, clothing or face
> outline. De-identification confined to inner-face texture therefore cannot make
> a gaze corpus releasable. We state what such a release would require, and
> report what our own tiered release leaks when measured rather than assumed.

**Notes.** (i) The abstract deliberately does not use "anonymised" for the
output; everything measured here is pseudonymised. (ii) "recent work reports that
it hides identity effectively" is a deliberate forward reference to Muştu &
Ekenel (2025) — we are contradicting a published result and should say so in the
first paragraph rather than bury it. (iii) The EyeLink is kept as the instrument
claim: the point is not "we also validated against a gold standard" but that the
utility axis is measured against physiological ground truth, which is what makes
a 0.5 cm difference interpretable at all.

---

## B — Introduction (~30 lines)

The risk with this section is writing a dataset paper with anonymisation
attached. Invert it: open on the sharing problem, introduce the dataset and the
four-architecture benchmark as the *instrument*, and make the privacy finding the
result.

### B0 — drafted prose

> Oculomotor function is a sensitive early marker of neurological and cognitive
> change, and the instruments that measure it well — infrared trackers such as
> the EyeLink 1000 — are confined to the clinic. Smartphone gaze estimation moves
> the measurement to where the patient already is [CITE: krafka16,
> valliappan20], which is why it has become an active route to accessible
> screening. What it does not move is the data. A smartphone eye tracker records
> video of the participant's face, and a face video is not a shareable research
> artefact. The result is a field whose progress depends on shared corpora and
> whose corpora cannot be shared.
>
> The common answer is generative de-identification: replace the face, keep the
> behaviour. Face swapping is attractive here precisely because it preserves the
> geometry — head pose, eyelid aperture, iris position — that a gaze model reads,
> where methods that inpaint the ocular region destroy it. It has been proposed
> for clinical video [CITE: disguises, digitalmask], and a recent evaluation
> concludes that face swapping hides identity effectively enough to serve as a
> video anonymiser [CITE: mustu25].
>
> We find that it does not, for gaze data, and that the reason is structural
> rather than a shortcoming of any particular generator. Two observations drive
> the paper. First, the ocular region is not a de-identified crop of the face: a
> 120 px eye ROI handed to a pretrained recogniser verifies identity at 62–80%
> at a false-accept rate of 1e-3, against a measured floor of 1.6%. The pixels
> that carry the gaze signal are the pixels that carry the identity, so a
> protocol that preserves accuracy by retaining genuine ocular imagery preserves
> the biometric with it. Second, replacing the whole face — ocular region
> included — does remove the correspondence to the participant's real face, but
> leaves the released corpus exactly partitionable by individual. Identity
> replacement is per-participant consistent, and the channels that remain are
> illumination, skin tone, periocular geometry and crop scale: attributes of the
> person that lie outside the inner-face region any swapping method edits.
>
> These are claims about measurement, so the paper is built around an instrument.
> We introduce **CognitiveGaze**: 18 participants performing seven oculomotor
> tasks, recorded simultaneously by a smartphone and an EyeLink 1000 and
> synchronised to the frame, so that each smartphone frame carries a
> physiological ground-truth gaze coordinate rather than a screen-target label.
> That is what makes a sub-centimetre difference on the utility axis
> interpretable. On the privacy axis we adopt the discipline of the security
> literature: a stated attacker, a pretrained recogniser never trained on our
> data, a second recogniser family as a cross-check, and a session-nuisance
> control that establishes what "no identity signal" scores on single-session
> recordings — without which a 62% verification rate cannot be distinguished from
> the fact that each participant was recorded once, in one chair, under one lamp.

### B1 — contributions

> **A privacy evaluation protocol for gaze corpora.** We evaluate
> de-identification the way the security literature evaluates it — with a stated
> attacker, a pretrained recogniser that never sees our data, and a
> session-nuisance control — rather than asserting it from the generator's
> objective. We distinguish an attacker holding enrolment photographs from one
> holding only the released corpus, and show the two differ by up to 60 points on
> identical pixels, which makes an unqualified privacy number uninterpretable.
>
> **Gaze utility and linkability share a representation.** The ocular region is
> simultaneously the signal gaze estimation needs and the region that identifies
> best per pixel. We show that the protocol which preserves accuracy does so by
> releasing genuine ocular biometrics — recovering the synthesis penalty to
> within 0.09–1.32 cm in *every* architecture we test, because in every case what
> it restores is the participant's own eye pixels — and that full synthesis,
> which does remove the correspondence to the real face, still leaves the corpus
> exactly partitionable by individual. Because the residual channels —
> illumination, skin tone, periocular geometry, crop scale — lie outside the edit
> domain of inner-face replacement, the limit is structural rather than a defect
> of one generator. We put it as a falsifiable claim: a transformation's
> protection against a release-only attacker is bounded by how far it breaks
> per-participant consistency, independently of how well it removes identity
> inside its own edit domain.
>
> **CognitiveGaze, and a four-architecture utility benchmark under
> de-identification.** 18 participants, smartphone and EyeLink 1000 recorded
> simultaneously across seven oculomotor tasks. We report the utility cost of
> both de-identification protocols across four architectures, paired at
> participant level, including a face-only positive control that isolates whether
> a model's ocular stream is active at all.

**Note on the ordering.** The dataset is third deliberately. It is a genuine
contribution, but as written in the current draft it is *first*, which frames the
paper as a dataset paper with de-identification attached and invites the reviewer
question "why is the privacy section here?". Led by the evaluation protocol and
the structural claim, the dataset reads as what it is: the instrument that makes
those claims measurable.

**On the title.** *Recovering Gaze Utility under Face De-identification* now
emphasises the half that turns out to be easy — utility recovery is uniform
across all four architectures. The paper's result is the other half. Two options
that fit the finding:
> *The Eyes You Keep: Gaze Utility and Identity Leakage Are the Same Pixels*
> *What Face Swapping Does Not Anonymise: Measuring De-identification for Gaze Datasets*

Also: "High-Precision Benchmark" invites a reviewer to check it against 4.9–5.9 cm
baselines. If the phrase stays, make explicit that it refers to the EyeLink
ground truth, not to the smartphone estimates.

---

## C — Related Work (~20 lines, four blocks)

This field is more crowded than the current draft implies, and it contains a
**directly competing published result**. Positioning against it is what turns our
finding from "another evaluation" into a contribution.

### C1 — Smartphone gaze estimation (~4 lines)

iTracker/GazeCapture [Krafka 2016], Valliappan et al. 2020 (Nature
Communications), AFFNet, MGazeNet. Already cited in the Methods; compress here to
the point that all of these consume face and eye ROIs, so all of them inherit the
release problem.

### C2 — Face de-identification, and the replacement/removal split (~6 lines)

The methods designed *for anonymisation* optimise identity **removal** — the
k-same family, DeepPrivacy and DeepPrivacy2 (conditional U-Net inpainting),
CIAGAN (conditional identity anonymisation preserving pose and background), and
the recent training-free / diffusion-based identity editors. Face *swapping*
methods such as SimSwap++ instead optimise identity **replacement**. The
distinction matters for gaze because replacement is what preserves the geometry
the task needs; removal methods that inpaint the eye region destroy the signal
outright. **Our result is that the objective difference does not rescue either
family, because the leak we measure lies outside the edit domain both share.**

### C3 — Face swapping as a privacy mechanism: the competing claim (~6 lines)

**This is the block that cannot be omitted.**

- **Muştu & Ekenel (2025), *Assessing the Use of Face Swapping Methods as Face
  Anonymizers in Videos*** (arXiv 2505.20985, IEEE) evaluate temporal
  consistency, anonymity strength and visual fidelity, and conclude that face
  swapping "can produce consistent facial transitions and effectively hide
  identities", recommending it for privacy-preserving video. **We reach the
  opposite conclusion and must explain the difference:** anonymity strength as
  they measure it is an A1 quantity — can the swapped face be matched to the
  person's real face — and on that axis we agree with them (22% versus 100%).
  They do not evaluate whether the released corpus can be partitioned by
  individual without any external reference, which is the question a public
  release actually poses.
- **Yang et al., *Practical Digital Disguises* (arXiv 2204.03559)** is the nearest
  neighbour in the medical domain: a face-swap pipeline for autism-assessment
  videos, self-described as the first methodology for assessing face-swap
  privacy–utility trade-offs for patient privacy, and it explicitly measures how
  well gaze and expression survive the swap relative to blurring. Our
  differentiation is the A2 attacker, the periocular control, and the finding
  that the retained ocular region is itself the biometric.
- **Nature Medicine's "digital mask" (2022)** and the **published critique of it**
  are worth one sentence each: they establish that whether generative
  de-identification is sufficient for clinical release is contested, not settled.
- **Explicit gaze constraints in face swapping** (arXiv 2305.16138) is the
  gaze-aware swapping line and belongs here.

### C4 — Eye movement as a biometric, and why our open tier is what it is (~4 lines)

Eye movement is an established biometric: the oculomotor-plant and scanpath line
(Rigas & Komogortsev and successors; the BioEye/EMVIC competitions), and in HCI
the demonstration that identity, gender and age are recoverable from gaze
features alone. The eye-tracking privacy literature is mature — differential
privacy for gaze streams with temporal-correlation handling, RL-based
obfuscation, privacy-preserving scanpath comparison, a decade review, and
specifically **Alsakar, Alotaibi, Khamis & Stumpf (ACM TOPS 2025) on handheld
mobile devices**, which is our exact device class.

**This block is why our open tier is measured rather than assumed.** We first
intended to defend it with a resolution argument — our instrument is 30 Hz at
~5 cm (≈2.4–3.6° at 80–120 cm), far coarser than the operating points usually
quoted. **That argument does not survive the literature and must not be used:**
one line of work reports 0.1° / 30 Hz as sufficient for oculomotor-plant
biometrics, degradation with sampling rate is gentle (EER 0.073 → 0.082 → 0.090
at 1000 / 250 / 125 Hz), and identification remains above chance as spatial noise
is raised to 0.5°. So we ran the attack instead; see §E.

**Reviewer risk:** this community will ask whether the non-pixel tier leaks. An
answer with a number is a strength; silence reads as not knowing the field.

---

**Reviewer risk if C2–C4 are thin:** a reviewer who works on face privacy or gaze
privacy will read a thin Related Work as evidence that the authors do not know the
field they claim to contribute to. With a competing published claim in C3, thinness
also looks like avoidance. This is the highest-leverage 20 lines in the paper.

**To verify before submission:** whether *Practical Digital Disguises* evaluates
any release-only/linkage attack. The abstract suggests not, but this should be
confirmed from the paper itself, because if it does, our C3 differentiation needs
rewording.

---

### C — drafted prose

Citation markers are `[CITE: short-key]`; substitute EndNote entries. Roughly 20
lines set, as budgeted.

> **Smartphone gaze estimation.** Appearance-based gaze estimation on commodity
> phones was established by iTracker and the GazeCapture corpus [CITE: krafka16],
> and brought to research-grade accuracy with per-participant calibration by
> Valliappan et al. [CITE: valliappan20]. Subsequent architectures refine how the
> ocular stream is conditioned on facial context: AFFNet modulates eye features
> through adaptive group normalisation driven by a face-and-grid guidance factor
> [CITE: affnet], and MGazeNet extends the idea with linear adaptive batch
> normalisation [CITE: zhu24]. All of them consume the same released artefacts —
> a face ROI, two eye ROIs and a face grid — so all of them inherit the same
> data-sharing problem.
>
> **Face de-identification.** Methods built for anonymisation optimise identity
> *removal*. The k-Same family guarantees that a released face cannot be matched
> to fewer than k enrolled identities by replacing faces with averages
> [CITE: newton05], and its neural successors extend the guarantee to generated
> faces [CITE: ksamenet]. DeepPrivacy and DeepPrivacy2 inpaint the facial region
> with a conditional generator, preserving pose and background
> [CITE: deepprivacy, deepprivacy2]; CIAGAN conditions the generation so that
> expression and pose survive while identity does not [CITE: ciagan]; more recent
> work removes identity in diffusion latent space, including training-free
> variants [CITE: fluid, apl]. Face *swapping* methods such as SimSwap and
> SimSwap++ instead optimise identity *replacement* [CITE: simswap, simswappp].
> The distinction matters for gaze: removal methods that inpaint the ocular
> region destroy the signal the task depends on, whereas replacement preserves
> the underlying geometry. Our finding is that the difference in objective does
> not decide the outcome, because the leakage we measure lies outside the edit
> domain that both families share.
>
> **Face swapping as a privacy mechanism.** Swapping has accordingly been
> proposed for exactly our setting. Yang et al. build a face-swap pipeline for
> recorded clinical assessments and give the first privacy–utility methodology
> for patient-privacy face swaps, measuring how far gaze and expression survive
> relative to blurring [CITE: disguises]. A digital-mask approach was reported in
> the clinical literature [CITE: digitalmask] and promptly contested
> [CITE: maskconcerns], so the sufficiency of generative de-identification for
> clinical release is an open question rather than a settled one. Most directly,
> Muştu and Ekenel evaluate face swapping as a video anonymiser on temporal
> consistency, anonymity strength and visual fidelity, and conclude that it
> hides identity effectively [CITE: mustu25]. **We reach the opposite conclusion,
> and the difference is the attacker.** Anonymity strength as measured there asks
> whether a swapped face can be matched to the person's real face; on that
> question our results agree with theirs. It does not ask whether the released
> corpus can be partitioned by individual with no external reference, which is
> the question a public release actually poses, and which we show survives the
> swap almost intact.
>
> **Eye movement as a biometric.** Identity is recoverable from eye movement
> itself, through oculomotor-plant characteristics and scanpath statistics
> [CITE: rigas, holland], as formalised by the BioEye and EMVIC benchmarks
> [CITE: bioeye]. The eye-tracking privacy literature that follows has developed
> differential-privacy mechanisms that account for temporal correlation
> [CITE: bozkir, steil], reinforcement-learning obfuscation [CITE: fuhl], and
> privacy-preserving scanpath comparison [CITE: scanpath]; the area now has a
> decade review [CITE: etprivreview] and a treatment specific to handheld mobile
> devices [CITE: alsakar25]. This line is why we do not treat a pixel-free
> release tier as risk-free by construction, and instead measure it (§E).

**Note on one temptation.** It is tempting to dismiss gaze-signal leakage on
resolution grounds. Do not: 0.1° / 30 Hz has been reported as sufficient for
oculomotor-plant biometrics, and identification remains above chance at 0.5° of
added spatial noise. Our own measurement — ARI 0.033 against a ~0.000
permutation null — is the defensible statement.

---

## D — Methods, new subsection after *GAN-based Privacy De-identification*

### Threat Model and Privacy Evaluation

**What is released.** The shareable artefact is the set of cropped streams the
models consume — the face ROI, the left and right eye ROIs, and the face grid —
not the source video. All attacks operate on exactly those crops.

**Attackers.** Two, because they give different answers on identical data.

- **A1, enrolment photographs.** The attacker holds ordinary photographs of the
  candidate individuals and asks which enrolled person a released crop is. This
  is the classical re-identification threat.
- **A2, release only.** The attacker holds nothing but the shared dataset. They
  cannot name anyone unaided, but they can ask whether the release partitions
  into individuals; a single labelled example then names an entire cluster. For
  a corpus intended for public release, A2 is the operative attacker, because
  the recipient of a public dataset is by definition A2.

**Recognisers.** InsightFace `buffalo_l` (ArcFace, w600k_r50) throughout, used
without any adaptation to this dataset; cosine similarity against per-participant
templates. No component is trained on CognitiveGaze. Because a privacy figure
should not be one model's opinion, the load-bearing periocular measurements are
repeated with FaceNet (Inception-ResNet-v1, VGGFace2, triplet loss) — a different
backbone trained on a different corpus with a different objective. Each
recogniser is quoted against **its own** floor: FaceNet's is 3.52%, more than
twice ArcFace's 1.57%, so the two must not be mixed.

**Metrics.** True-accept rate at a fixed false-accept rate of 1e-3
(TAR@FAR=1e-3) is primary for A1; closed-set Rank-1 is reported for completeness
only, because on single-session data it is inflated by session structure (below).
For A2 we report the adjusted Rand index (ARI) of an unsupervised partition
against the true participant labels, together with the number of clusters the
procedure selects unaided.

**The session-nuisance floor.** Each participant was recorded in a single
session, so illumination, seating and background are constant within a
participant and differ between participants. Any identification metric on such
data therefore has a non-zero baseline owing nothing to the face. We measure it
by running every attack unchanged on a 120 px face-free patch of the same frames.
The floor is **Rank-1 50.5%** (chance 5.6%), **TAR@FAR=1e-3 1.6%**, and **ARI
0.089** with 2 clusters selected rather than 18. Closed-set Rank-1 is therefore
not evidence of de-identification on this kind of data; verification at a strict
operating point, and linkage, are.

**Alignment.** Detection and alignment are taken from the *original* frame and
applied to the de-identified one, so an operator is never credited for defeating
the detector — detector failure is not privacy, and it would also break the gaze
pipeline. Where an attack instead detects on the released image ("self-aligned"),
it is labelled as such and the number of frames surviving detection is reported
beside it.

**Uncertainty.** Frames of one participant are not independent. Privacy intervals
are 95% subject-level cluster bootstrap (2,000 replicates, resampling
participants with replacement, gallery held fixed).

---

## E — Results, new subsection before *Privacy-Utility Experiments*

### Identity Leakage under De-identification

**The ocular region identifies the participant.** Before any operator is
considered: an unmodified eye ROI, 120 px, given to a pretrained recogniser with
no adaptation, verifies identity at **66.6%** (left) and **62.2%** (right)
TAR@FAR=1e-3, against a 1.6% floor. Rank-1 is 91–98%. The eye ROI is not a
de-identified crop of the face; per pixel it is the part of the face that
identifies best.

**A second recogniser agrees, in both threat models.** Every measurement above was
repeated with FaceNet (Inception-ResNet-v1, VGGFace2, triplet loss) — a different
backbone, corpus and objective from ArcFace. Under A1 it gives **69.1% / 81.2%**
on unmodified ocular ROIs and **0.00–1.48%** on synthesised ones, reproducing both
halves of the ArcFace result. Under A2 it gives **67.1% / 80.5%** unmodified
against its own 3.52% floor (19–23× floor, where ArcFace gives 40×), and
42.8% / 55.8% (Identity A) and 34.4% / 51.0% (Identity B) on synthesised ROIs,
again far above floor. Each recogniser is quoted against its own floor; FaceNet's
is more than twice ArcFace's, so the two must not be mixed. The agreement is
evidence about the imagery rather than about one model's inductive bias.

**This is what the Hybrid protocol releases.** Hybrid preserves gaze accuracy by
retaining the original, unaltered eye ROIs — the crops measured above. On the
identity axis it therefore inherits the leakage of unprocessed data: **62–67%
TAR@FAR=1e-3, roughly forty times the floor**, before the synthetic face is
considered at all. The mechanism that preserves the utility is the mechanism that
leaks the identity.

**Full synthesis does remove the correspondence to the real face.** With the
ocular region synthesised as well, the same attack falls to **0.2% / 1.8%**
TAR@FAR=1e-3, at or below the floor (FaceNet: 0.00% / 1.48%); the face ROI falls
from 100% to **22.0%** [5.2, 44.5] for Identity A and **23.3%** [8.2, 41.8] for
Identity B. The two templates agree, and so do the two recognisers, so this is a
property of the method rather than of a particular source face or a particular
recogniser.

**But the release stays linkable.** Under A2, an unsupervised partition of the
released corpus recovers the participants exactly: **ARI 1.000 at the correct
k = 18**, for both templates, against a floor of 0.089 at k = 2. Identity
replacement is per-participant consistent — each participant is mapped to the
same synthetic appearance every time — which destroys the correspondence to their
real face while leaving the released set internally self-consistent. The same eye
ROIs that score 0.2% against A1 score **42.8% / 62.5%** against A2.

**The linkage is not an artefact of surrounding context.** A face ROI retains
channels the synthesis never edits — hair at the crop border, spectacles, skin
tone, head pose, face outline — and the floor does not control for them, because
it is a patch of the *room* whereas those are attributes of the *person*. We
therefore repeat the linkage attack on the 120 px eye ROI, which excludes hair,
ears, neck, clothing and the face outline, and whose pixels are themselves
synthetic in the Full Synthetic arm. Linkage survives: **ARI 0.68–0.79** at
k = 18 for the two synthetic identities, against **0.84–0.85** for unmodified eye
ROIs and a floor of 0.089, with 79–86% of frames assigned to the correct
participant. The residual channels — illumination, skin tone, periocular
geometry, crop scale — are the signals gaze estimation reads, which is why the
axis trades at all.

These eye-ROI figures assume an attacker who knows the cohort size: on the eye
ROI the cluster count is not recovered unaided, whereas on the face ROI it is
(k = 18 selected by silhouette). For a public release the cohort size is stated
in the documentation, so the assumption is mild, but it is an assumption and the
face ROI does not need it. Breaking unsupervised *discoverability* is in fact the
one place replacement helps: on the right eye, k selected falls from 26 (ARI
0.674) to 2 (ARI 0.011). It breaks discovery of the partition, not its
separability.

**The open tier leaks too, by two orders of magnitude less.** Our release plan
puts per-frame gaze estimates and oculomotor measures in an open tier because it
contains no imagery and no face embeddings. Since eye movement is itself a
biometric, we tested that rather than asserting it. Treating the released gaze
series as the attacker's only input, we cut each recording into 5 s fragments
(150 frames), described each with fifteen dispersion, velocity and
saccade/fixation statistics, and clustered. The result is **ARI 0.033 at the
correct k, purity 0.157 against a chance purity of 0.059**, versus a
label-permutation null of −0.0001 (max 0.0015 over 20 permutations). The signal
is therefore real but small: two orders of magnitude below the pixel tiers
(0.68–1.000). We report the open tier as low — not zero — residual risk, measured
rather than assumed.

**The floor is now eye-crop native.** The 1.6% / ARI 0.089 floor quoted above was
originally measured by running the *face-crop* pipeline on a face-free patch,
which controls for room and session but is not strictly an eye-crop quantity.
Running the periocular pipeline itself on a 120 px face-free patch of the same
frames — identical frame selection, identical downstream steps, only the pixels
differ — gives **TAR@FAR=1e-3 = 1.57%**, Rank-1 50.6%, d′ 0.458, agreeing with the
borrowed figure to three significant figures.

**Metadata leaks before a pixel is read.** Released crop dimensions are constant
per participant. Face ROI size alone yields an expected Rank-1 of 50.0%; face
size, eye size and inter-ocular distance together yield **94.4%** (4.06 of 4.17
bits) without reading any pixels. A release must normalise crop geometry; ours
does, and all figures above are pixel-based.

**An adaptive attacker gains little.** An attacker who knows the operator and
holds paired (original, synthesised) imagery of *other* participants can fit a
de-anonymisation map and apply it to a participant they could not train on.
Across mean-shift, orthogonal Procrustes and ridge maps, fitted
leave-one-participant-out, the best gain is **+3.3 points** on the face ROI
(19.9% → 23.2%) and nothing on the eye ROIs. The same maps reach Rank-1 100% when
fitted *including* the target — they memorise identities rather than invert the
operator. The protection is a generalisation barrier, not an information-theoretic
one.

#### Table P — Identity leakage across release strategies

A1 = attacker holds enrolment photographs; A2 = attacker holds only the release.
TAR at FAR = 1e-3 with 95% subject-level bootstrap CI.

| Released stream | Real Data | Full Synthetic (A) | Full Synthetic (B) | Hybrid (Real Eyes) |
|---|---|---|---|---|
| Face ROI, TAR (A1) | 100% | 22.0% [5.2, 44.5] | 23.3% [8.2, 41.8] | 22.0% (face is synthetic) |
| Left eye ROI, TAR (A1) | 66.6% [53.6, 79.6] | 0.2% [0.0, 3.0] | 0.6% [0.0, 3.1] | **66.6%** (unaltered) |
| Right eye ROI, TAR (A1) | 62.2% [49.0, 84.6] | 1.8% [0.1, 5.7] | 1.1% [0.0, 4.8] | **62.2%** (unaltered) |
| *Session-nuisance floor* | *1.6%* | *1.6%* | *1.6%* | *1.6%* |
| Face ROI, ARI @ k=18 (A2) | 1.000 | 1.000 | 1.000 | 1.000 |
| Left eye ROI, ARI @ k=18 (A2) | 0.844 | 0.684 | 0.737 | 0.844 (unaltered) |
| Right eye ROI, ARI @ k=18 (A2) | 0.853 | 0.789 | 0.761 | 0.853 (unaltered) |
| *Linkage floor* | *0.089* | *0.089* | *0.089* | *0.089* |

---

## F — Edits to the existing text, by location

Six edits. The first four are deletions of claims the current pipeline
contradicts or that are simply wrong; they are **not** softenings. The last two
are consistency fixes a reviewer will otherwise catch.

### F0 — checklist

| # | Section | What | Action |
|---|---|---|---|
| 1 | Privacy-Utility Experiments, final para | "iTracker and AFFNet consistently failed the Hybrid test…" + the "fundamental architectural incompatibility" conclusion | **Delete** |
| 2 | Contributions (V4) / GAN-based Privacy De-identification | the word "encrypted" | **Delete** |
| 3 | Privacy-Utility Experiments, para 1 end | "viable pathway for sharing clinical datasets publicly" | **Replace** (F1) |
| 4 | GAN-based Privacy De-identification, opening | "eliminating the patient's biometric identity" | **Replace** (F2) |
| 5 | Table 6 | "(Failure)" / "(Stable Failure)" labels | **Restructure** (F4) |
| 6 | Dataset and Recording Protocol | "initially comprised 23 volunteers" vs N=18 everywhere | **Explain** (F5) |

### F1 — the "public sharing" claim

**F1 — *Privacy-Utility Experiments*, final sentence of paragraph 1.** Currently:

> ~~This framework potentially preserves clinical accuracy with identity
> protection, thereby offering a viable pathway for sharing clinical datasets
> publicly.~~

Replace with:

> This protocol tests whether the "Context" (Face) can be de-identified while the
> "Signal" (Eyes) remains genuine. Section *Identity Leakage under
> De-identification* measures what each protocol actually protects; as shown
> there, retaining genuine ocular ROIs retains the identifying information they
> carry, so Hybrid is a utility-preserving protocol rather than a
> privacy-preserving one, and is not by itself sufficient for public release.

**F2 — *GAN-based Privacy De-identification*, opening.** The phrase
"eliminating the patient's biometric identity" is not supported for either arm.
Replace with "replacing the facial identity", and defer the measurement to §E.
The figure caption for the Hybrid pipeline carries the same phrase ("eliminating
the patient's biometric identity") and needs the same fix.

**F3 — the word "encrypted".** Nothing in the pipeline is encrypted. The word is
a category error and a reviewer will read it as evidence the authors do not
distinguish encryption from de-identification. Delete wherever it appears.

**F4 — Table 6's "Deviation" column.** It currently conflates two different
things: how well an arm performs in absolute terms, and how stable it is across
templates. That produces incoherent labels — an *improvement* of −0.90 cm marked
"(Failure)", and "−0.17 cm (Stable Failure)". Report cross-template deviation
alone in the column; whether an arm performs poorly is a sentence in the text,
not a parenthetical in a stability column. Note also that Table 3 and Table 6
disagree on iTracker (9.36/9.35 versus 9.35/9.36); both are superseded by §H.

**F5 — the cohort count.** *Dataset and Recording Protocol* says the study
"initially comprised 23 volunteers (13 males, 10 females)", while every result
uses 18 (17 in the participant-level statistics). "Initially" implies attrition
that is never explained. State how many were excluded, why, and against what
criterion decided when — this is a standard reviewer question and, in a privacy
paper, a sensitive one if exclusion correlated with appearance. Report cohort
demographics (age range, and enough to let a reader judge homogeneity), since
face recognisers have known demographic bias and the privacy figures'
generalisability depends on it.

---

## G — Limitations and Conclusion (~20 lines)

> Neither protocol yields a corpus that can be released without further control.
> Full synthesis removes the link to the real face at a gaze cost of 2.16–6.27 cm;
> Hybrid preserves gaze — to within 0.09–1.32 cm across every architecture we
> tested — at the price of releasing genuine ocular biometrics; and
> both leave the corpus linkable by participant, so one labelled example
> compromises a cluster. The linkage is structural rather than incidental: it
> survives on a 120 px ocular crop containing none of the contextual cues a face
> crop carries, and the channels that carry it are the channels gaze estimation
> reads, so any method whose edit domain is inner-face texture inherits the
> limit.
>
> Four constraints bound these conclusions. Participants were recorded in a
> single session, so the session-nuisance floor controls for room and background
> but not for per-session imaging conditions, and the longitudinal tracking that
> makes linkage consequential is precisely the cross-session case we cannot test.
> The cohort is 18 healthy adults, so we make no claim about clinical
> populations and describe the contribution as an evaluation of dataset release
> rather than of home screening. Privacy is measured with one recogniser family.
> And the structural claim is demonstrated on one replacement method with two
> templates; methods with a wider edit domain remain to be tested, and are the
> obvious next experiment.
>
> A deployable route would have to break the per-participant consistency of the
> transformation — for instance by resampling the template per recording session
> rather than per participant, which our single-session design cannot evaluate —
> or protect the ocular region while preserving the gaze geometry inside it, an
> axis we examined and found to trade linearly, with no setting that was both
> private and accuracy-preserving. Until then, controlled access under an
> enforceable agreement, not open deposit, is what the measurements support.

---

## H — Table 3 (utility), rebuilt

Source: `runs/anon_swap_utility`, job 1283656 (mobilenet_v3, affnet, mgazenet) and
1280374 (itracker and the controls). Subject-wise 5-fold CV, K=72 clean
enrolment, no per-participant calibration (the `base` column), 15/15 tasks
completed with no failures. **These values supersede the earlier Table 3
entirely**; see `results/ITRACKER_HYBRID_CHECK.md` for why.

#### Table 3 — Utility cost of de-identification

Mean Euclidean error in cm, mean ± SD over 5 folds. Impact is relative to that
model's own Real Data baseline.

| Model | Real Data | Full Synthetic (A) | Hybrid (A) | Full Synthetic (B) | Hybrid (B) |
|---|---|---|---|---|---|
| iTracker | 5.930 ± 2.487 | 9.078 (+3.15) | **6.432 (+0.50)** | 9.520 (+3.59) | **6.039 (+0.11)** |
| MobileNet-V3 Large | 5.038 ± 1.596 | 8.633 (+3.60) | **5.433 (+0.40)** | 11.312 (+6.27) | **6.361 (+1.32)** |
| AFFNet | 5.791 ± 2.008 | 7.950 (+2.16) | **5.983 (+0.19)** | 9.250 (+3.46) | **5.882 (+0.09)** |
| MGazeNet | 4.947 ± 2.254 | 7.835 (+2.89) | **5.281 (+0.33)** | 10.383 (+5.44) | **5.484 (+0.54)** |
| *Face-only control* | *5.430 ± 1.918* | *11.426 (+6.00)* | *11.426 (+6.00)* | *18.372 (+12.94)* | *18.372 (+12.94)* |

#### Table 4 — Paired effect of restoring the genuine ocular ROIs

Hybrid minus Full Synthetic, paired within fold. Negative means the real eye
crops recover accuracy. 95% CI from the t distribution on 5 folds.

| Model | Identity A | Identity B |
|---|---|---|
| iTracker | −2.646 ± 2.163 [−5.33, +0.04] | −3.482 ± 1.445 [−5.28, −1.69] |
| MobileNet-V3 Large | −3.200 ± 1.170 [−4.65, −1.75] | −4.950 ± 1.627 [−6.97, −2.93] |
| AFFNet | −1.967 ± 1.126 [−3.36, −0.57] | −3.368 ± 1.910 [−5.74, −1.00] |
| MGazeNet | −2.554 ± 1.058 [−3.87, −1.24] | −4.899 ± 0.783 [−5.87, −3.93] |
| *Face-only control* | *+0.000 ± 0.000* | *+0.000 ± 0.001* |

**The face-only control is not decoration.** It consumes the face ROI only and
therefore cannot see the eye crops; its two arms must be identical, and they are,
to three decimals. It establishes that the harness distinguishes the arms when a
model uses the ocular stream and reports exactly zero when it does not — which is
what licenses reading every other row as a real effect.

### Text to accompany the tables

> Restoring the genuine ocular ROIs recovers most of the synthesis penalty in
> every architecture tested. Full synthesis costs **+2.16 to +6.27 cm** depending
> on model and template; Hybrid costs **+0.09 to +1.32 cm**. The paired
> within-fold difference is negative for all four architectures and both
> templates, and its 95% interval excludes zero in seven of eight cases — the
> exception, iTracker with Identity A, is driven by a single fold in which the
> Real Data baseline itself collapsed to 10.2 cm.
>
> We find no evidence of an architecture that fails to exploit the restored
> ocular signal. In particular the two guidance-factor architectures, whose eye
> features are modulated by parameters regressed from the face, do not behave
> differently from the late-fusion ones: AFFNet preserves best of the four
> (+0.19 / +0.09 cm). The distinguishing factor for the Hybrid protocol is not
> the fusion topology but simply whether the model reads the eye crops at all, as
> the face-only control makes explicit.

**Why this matters for the paper's argument.** Hybrid preserving utility
*uniformly* is what makes the privacy finding sharp rather than incidental: the
protocol works because it releases the genuine ocular pixels, and those pixels
verify identity at 62–67%. Utility and leakage are not two quantities to be
traded off against one another here — they are the same pixels.

### Claims that must be withdrawn

The draft states that iTracker and AFFNet "consistently failed the Hybrid test
regardless of the identity used (errors remaining >8.4 cm)" and concludes that
"'Context-Signal Mismatch' is a fundamental architectural incompatibility". In
this pipeline **neither model fails and AFFNet preserves best of the four**. The
sentence and the conclusion must be removed, not softened.

The sensitivity-analysis table's "Deviation" column also conflates two things —
how well an arm performs in absolute terms, and how stable it is across templates
— and labels an *improvement* of −0.90 cm as "(Failure)" and −0.17 cm as "(Stable
Failure)". Report cross-template deviation alone in that column; whether an arm
performs poorly is a separate statement in the text.

## Figures

One privacy figure fits the page budget: `fig3_threat_models.pdf` (same crops,
A1 vs A2). It carries the whole "a privacy number without its threat model is
uninterpretable" point in a single column. `fig5_linkage_release_only.pdf` (two
panels, face crop and eye crop) is the journal-version figure.

## Page budget

To make room: compress the four per-architecture descriptions (eqs. 2–3 and the
iTracker three-stream prose) into one comparison table plus two sentences — these
are published architectures; and collapse the seven task descriptions to the two
that later sections use, plus one sentence for the rest.
