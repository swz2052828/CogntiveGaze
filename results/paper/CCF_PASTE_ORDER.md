# CCF manuscript — walkthrough in document order

Work top to bottom through `ccf_manuscript_v2.docx`. Each entry names the
section, the action, and the text.

- **INSERT** — new text, nothing to remove
- **REPLACE** — strike the quoted sentence, paste the replacement
- **DELETE** — remove, do not soften
- **CHECK** — a factual or consistency fix you need to supply

Rationale, alternatives and the reviewer's-eye reasoning live in
`CCF_SECTIONS.md`; this file is the paste copy. Numbers come from
`runs/anon_swap_utility` + `runs/initvar_utility` (utility, condition (ii)),
`results/anon_dp2/util_table.json` (DeepPrivacy2, condition (ii)),
`results/anon_train/table3_three_conditions.json` (condition (i), retrained) and
`results/anon_*` (privacy).
Citation markers are `[CITE: key]` — substitute EndNote entries.

**Terminology:** *Real Data*, *Full Synthetic*, *Hybrid (Real Eyes)*,
*Identity A / B*. Never "anonymised" for the output — everything here is
pseudonymised.

---

## 0. Title — CHECK

The current title, *Recovering Gaze Utility under Face De-identification: A
Smartphone Dataset and High-Precision Benchmark*, emphasises the half that turns
out to be easy: utility recovery is uniform across all four architectures. The
result is the other half. Two options that fit the finding:

- *The Eyes You Keep: Gaze Utility and Identity Leakage Are the Same Pixels*
- *What Face Swapping Does Not Anonymise: Measuring De-identification for Gaze Datasets*

If "High-Precision Benchmark" stays, say explicitly that it refers to the EyeLink
ground truth, not to the smartphone estimates — otherwise a reviewer checks it
against the 4.9–5.9 cm baselines in Table 3.

---

## 1. Abstract — INSERT (~285 words)

> Smartphone eye tracking moves oculomotor assessment out of the clinic, but what
> it records is video of the participant's face — which is what blocks the
> dataset sharing the field depends on. GAN face replacement is the common
> answer, and recent work reports that it hides identity effectively: swap the
> face, keep the gaze. We test that claim on both axes.
>
> We introduce CognitiveGaze: 18 participants, seven oculomotor tasks, recorded
> simultaneously by a smartphone and an EyeLink 1000, so every smartphone frame
> carries a physiological ground-truth gaze coordinate rather than a
> screen-target label. Utility is measured by deploying four gaze architectures
> on de-identified crops and by retraining three of them on the de-identified
> release. Privacy is measured on the released crops themselves, with pretrained
> recognisers, two attacker models, and a control establishing what "no identity
> signal" scores on single-session data.
>
> The axes do not trade as assumed. Retaining the genuine ocular region recovers
> 85–101% of the synthesis penalty in every architecture, and 86–108% when the
> models are retrained on the release — yet those same crops verify identity at
> 62–80% TAR at FAR 1e-3 against a measured 1.6% floor, under two unrelated
> recognisers. The pixels that make the protocol work are the pixels that
> identify. Full synthesis does break the link to the real face, but costs
> 2.2–5.4 cm even after retraining, and an attacker holding only the release
> still recovers exactly the right 18 people — from a 120 px ocular crop
> containing no hair, ears, clothing or face outline. Removal by inpainting fares
> worse: a model retrained on it is no better than predicting the screen centre.
> Across a replacement and a removal method, the one configuration that breaks
> linkage — repainting the whole person with a fresh identity every frame — is
> one that discards the eyes. We state what a releasable gaze corpus would
> require, and report what our own tiered release leaks when measured rather than
> assumed.

If the venue caps at 250: drop "seven oculomotor tasks", the sentence on removal
by inpainting, and the final clause about the tiered release.

---

## 2. Introduction — INSERT (~30 lines)

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
> protocol that recovers accuracy by retaining genuine ocular imagery retains
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

### Contributions

Order matters: the dataset is **third**. Leading with it frames the paper as a
dataset paper with de-identification attached.

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
> best per pixel. We show that the protocol which recovers accuracy does so by
> releasing genuine ocular biometrics — recovering 85–101% of the synthesis
> penalty in *every* architecture we test, and 86–108% when the models are
> retrained on the release, because in every case what it restores is the
> participant's own eye pixels — and that full synthesis,
> which does remove the correspondence to the real face, still leaves the corpus
> exactly partitionable by individual. Because the residual channels —
> illumination, skin tone, periocular geometry, crop scale — lie outside the edit
> domain of inner-face replacement, the limit is structural rather than a defect
> of one generator: we confirm it on a second family — DeepPrivacy2, which
> *removes* identity by inpainting rather than replacing it, and in its face mode
> touches 40% more of the crop — and the released face crop is still exactly
> partitionable (ARI 1.000); so is it, near enough (0.982), when DeepPrivacy2
> repaints the whole person but keeps each participant's synthetic identity fixed.
>
> **A tested, two-part condition for breaking release-only linkage.** Rather than
> stopping at a negative result, we state the requirement as a falsifiable claim
> and test it by construction. Release-only linkage falls only when **both** the
> transformation breaks per-participant consistency **and** the released crop
> contains nothing outside the transformation's edit domain. We test it as a 2×2
> — face-only vs full-body DeepPrivacy2, fixed vs per-frame identity — and only
> the cell that meets both halves breaks: face-crop linkage is 1.000, 0.982 and
> 0.982 in the other three cells and **0.297** in that one, where an attacker who
> does not know the cohort size recovers no structure at all (ARI 0.000). On the
> eye ROI, which contains only edited pixels, per-frame identity alone suffices:
> verification 63.2% → 2.04% and linkage 0.805 [0.63, 0.94] → 0.148
> [0.10, 0.27], against floors of 1.57% and 0.089. The one configuration that
> breaks linkage repaints the eyes, and a gaze model retrained on repainted eyes
> does no better than predicting the screen centre. This tells a dataset author
> what to do, and what it costs, not merely what fails.
>
> **CognitiveGaze, and a four-architecture utility benchmark under
> de-identification.** 18 participants, smartphone and EyeLink 1000 recorded
> simultaneously across seven oculomotor tasks. We report the utility cost of
> both de-identification protocols, for two de-identification families, under
> three conditions — trained on real data and deployed on the de-identified
> crops, and retrained on the de-identified release then tested on real or on
> de-identified faces — against a measured initialisation noise floor and a
> predict-the-mean reference that marks where a model has learned no gaze at
> all, including a face-only positive control that isolates whether a model's
> ocular stream is active at all.

---

## 3. Related Work — INSERT (~20 lines)

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
> faces [CITE: ksamenet]. DeepPrivacy inpaints the facial region with a
> conditional generator, preserving pose and background [CITE: deepprivacy];
> DeepPrivacy2 extends inpainting to the whole body, on the argument that face
> anonymisation leaves identifiers outside the face, and evaluates it by
> re-identifying anonymised pedestrians from original images [CITE: deepprivacy2]; CIAGAN conditions the generation so that
> expression and pose survive while identity does not [CITE: ciagan]; more recent
> work removes identity in diffusion latent space, including training-free
> variants [CITE: fluid, apl]. Face *swapping* methods such as SimSwap and
> SimSwap++ instead optimise identity *replacement* [CITE: simswap, simswappp].
> The distinction is expected to matter for gaze: removal methods that inpaint
> the ocular region should destroy the signal the task depends on, whereas
> replacement is meant to preserve the underlying geometry. We measure both
> expectations. The first holds outright — a gaze model retrained on inpainted
> eyes learns nothing — but the second only partly: synthetic eyes keep enough
> geometry to cost 2–5 cm rather than everything. On the privacy axis the
> difference in objective does not decide the outcome, because the leakage we
> measure lies outside the edit domain that both families share.
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
> release tier as risk-free by construction, and instead measure it.

**CHECK before submission:** whether *Practical Digital Disguises* evaluates any
release-only / linkage attack. The abstract suggests not; confirm from the paper,
because if it does, the third paragraph's differentiation needs rewording.

---

## 4. Dataset and Recording Protocol — RESTORE (text exists in Manuscript V4)

The CCF draft cut the demographics and the attrition explanation; V4 has both.
Restore them. Without them the draft says the cohort "initially comprised 23
volunteers" and then reports 18 with no account of the difference — a standard
reviewer question, and a sensitive one in a privacy paper if exclusion could have
correlated with appearance.

**INSERT after the recruitment sentence:**

> Five participants were excluded before analysis, for calibration failure or
> incomplete data acquisition, leaving a final cohort of N = 18. Inclusion
> required no history of photosensitive migraine, epilepsy, motion sickness, or
> substance use affecting oculomotor function.
>
> The final cohort spanned 20 to 54 years (median 29) and comprised
> White/Caucasian (n = 12), East Asian (n = 2), Black/African (n = 1), Middle
> Eastern (n = 1), Asian-Indian (n = 1) and Mixed background (n = 1)
> participants. Three wore glasses. Unlike many open gaze corpora we did not
> exclude spectacle wearers, because vision correction is frequently a
> prerequisite for completing a cognitive protocol and excluding it would
> compromise clinical generalisability.

**Why this matters for the privacy axis, and a sentence to add there.** Face
recognisers carry known demographic bias, so the generalisability of every
identity figure depends on the cohort composition. Ours is moderately
homogeneous — two thirds White/Caucasian, recruited from a single engineering
school, median age 29 — and this belongs in Limitations:

> The cohort is moderately homogeneous in ethnicity and age and was recruited
> from a single institution. Face recognition accuracy is known to vary across
> demographic groups, so the absolute identity rates reported here should not be
> read as cohort-independent; the comparisons between arms, which hold the cohort
> fixed, are unaffected.

**A connection worth making explicit.** Three participants wear glasses, and
spectacle frames are retained verbatim by inner-face replacement — they sit
outside its edit domain, like hair at the crop border. This is a concrete,
visible instance of the structural claim and can be cited as such in §8.

---

## 5. Evaluation of Gaze Models — COMPRESS

Keep the synchronisation, calibration and cross-validation text. To make room for
the privacy axis, compress the four per-architecture descriptions (eqs. 2–3 and
the iTracker three-stream prose) into one comparison table plus two sentences —
these are published architectures. Keep one of Figure 3 / Figure 4, not both.

Also collapse the seven task descriptions to the two that later sections use,
plus one sentence for the rest.

---

## 6. GAN-based Privacy De-identification — REPLACE ×2, DELETE ×1

**REPLACE, opening paragraph.** "eliminating the patient's biometric identity" is
not supported for either arm. Use "replacing the facial identity", and defer the
measurement to the new results subsection. The Hybrid pipeline figure caption
carries the same phrase and needs the same fix.

**DELETE, anywhere it appears: the word "encrypted".** Nothing in the pipeline is
encrypted. It is a category error and reads as not distinguishing encryption from
de-identification.

**REPLACE, end of the Privacy-Utility Experiments lead paragraph.**

> ~~This framework potentially preserves clinical accuracy with identity
> protection, thereby offering a viable pathway for sharing clinical datasets
> publicly.~~

with

> This protocol tests whether the "Context" (Face) can be de-identified while the
> "Signal" (Eyes) remains genuine. The following section measures what each
> protocol actually protects; as shown there, retaining genuine ocular ROIs
> retains the identifying information they carry, so Hybrid is a
> utility-preserving protocol rather than a privacy-preserving one, and is not by
> itself sufficient for public release.

---

## 7. Threat Model and Privacy Evaluation — INSERT (new Methods subsection)

Place immediately after *GAN-based Privacy De-identification*.

> **What is released.** The shareable artefact is the set of cropped streams the
> models consume — the face ROI, the left and right eye ROIs, and the face grid —
> not the source video. All attacks operate on exactly those crops.
>
> **Attackers.** Two, because they give different answers on identical data.
> *A1, enrolment photographs:* the attacker holds ordinary photographs of the
> candidate individuals and asks which enrolled person a released crop is — the
> classical re-identification threat. *A2, release only:* the attacker holds
> nothing but the shared dataset; they cannot name anyone unaided, but they can
> ask whether the release partitions into individuals, and a single labelled
> example then names an entire cluster. For a corpus intended for public release
> A2 is the operative attacker, because the recipient of a public dataset is by
> definition A2.
>
> **Recognisers.** InsightFace `buffalo_l` (ArcFace, w600k_r50) throughout, used
> without any adaptation to this dataset; cosine similarity against
> per-participant templates. No component is trained on CognitiveGaze. Because a
> privacy figure should not be one model's opinion, the load-bearing periocular
> measurements are repeated with FaceNet (Inception-ResNet-v1, VGGFace2, triplet
> loss) — a different backbone trained on a different corpus with a different
> objective. Each recogniser is quoted against its own floor: FaceNet's is 3.52%,
> more than twice ArcFace's 1.57%, so the two are never mixed.
>
> **Metrics.** True-accept rate at a fixed false-accept rate of 1e-3
> (TAR@FAR=1e-3) is primary for A1; closed-set Rank-1 is reported for
> completeness only, because on single-session data it is inflated by session
> structure. For A2 we report the adjusted Rand index (ARI) of an unsupervised
> partition against the true participant labels, together with the number of
> clusters the procedure selects unaided.
>
> **The session-nuisance floor.** Each participant was recorded in a single
> session, so illumination, seating and background are constant within a
> participant and differ between participants. Any identification metric on such
> data therefore has a non-zero baseline owing nothing to the face. We measure it
> by running every attack unchanged on a 120 px face-free patch of the same
> frames. The floor is **Rank-1 50.5%** (chance 5.6%), **TAR@FAR=1e-3 1.6%**, and
> **ARI 0.089** with 2 clusters selected rather than 18. Closed-set Rank-1 is
> therefore not evidence of de-identification on this kind of data; verification
> at a strict operating point, and linkage, are.
>
> **Alignment.** Detection and alignment are taken from the *original* frame and
> applied to the de-identified one, so an operator is never credited for
> defeating the detector — detector failure is not privacy, and it would also
> break the gaze pipeline. Where an attack instead detects on the released image
> ("self-aligned") it is labelled as such, and the number of frames surviving
> detection is reported beside it.
>
> **Uncertainty.** Frames of one participant are not independent. Privacy
> intervals are 95% subject-level cluster bootstrap (2,000 replicates, resampling
> participants with replacement, gallery held fixed). Utility differences are
> paired within participant (n = 17).

---

## 8. Identity Leakage under De-identification — INSERT (new Results subsection)

Place **before** *Privacy-Utility Experiments*, so the reader meets the privacy
axis before the utility table that it reframes.

> **The ocular region identifies the participant.** Before any operator is
> considered: an unmodified eye ROI, 120 px, given to a pretrained recogniser
> with no adaptation, verifies identity at **66.6%** (left) and **62.2%** (right)
> TAR@FAR=1e-3, against a 1.6% floor. Rank-1 is 91–98%. The eye ROI is not a
> de-identified crop of the face; per pixel it is the part of the face that
> identifies best.
>
> **A second recogniser agrees, in both threat models.** Every measurement above
> was repeated with FaceNet. Under A1 it gives **69.1% / 81.2%** on unmodified
> ocular ROIs and **0.00–1.48%** on synthesised ones, reproducing both halves of
> the ArcFace result. Under A2 it gives **67.1% / 80.5%** unmodified against its
> own 3.52% floor (19–23× floor, where ArcFace gives 40×), and 42.8% / 55.8%
> (Identity A) and 34.4% / 51.0% (Identity B) on synthesised ROIs, again far
> above floor. The agreement is evidence about the imagery rather than about one
> model's inductive bias.
>
> **This is what the Hybrid protocol releases.** Hybrid recovers gaze accuracy
> by retaining the original, unaltered eye ROIs — the crops measured above. On
> the identity axis it therefore inherits the leakage of unprocessed data:
> **62–67% TAR@FAR=1e-3, roughly forty times the floor**, before the synthetic
> face is considered at all. The mechanism that recovers the utility is the
> mechanism that leaks the identity.
>
> **Full synthesis does remove the correspondence to the real face.** With the
> ocular region synthesised as well, the same attack falls to **0.2% / 1.8%**
> TAR@FAR=1e-3, at or below the floor (FaceNet: 0.00% / 1.48%); the face ROI
> falls from 100% to **22.0%** [5.2, 44.5] for Identity A and **23.3%**
> [8.2, 41.8] for Identity B. The two templates agree, and so do the two
> recognisers, so this is a property of the method rather than of a particular
> source face or a particular recogniser.
>
> The face-ROI intervals are wide — [5.2, 44.5] for Identity A — because the
> sampling unit is 18 participants. The direction is secure: the lower bound is
> more than three times the 1.6% floor, and both templates and both recognisers
> agree. The magnitude is not: these data support "well above floor, an order of
> magnitude below unprotected imagery", and not a finer grading than that. The
> release decisions in §Data Availability rest on the direction, not on the point
> estimate.
>
> **But the release stays linkable.** Under A2, an unsupervised partition of the
> released corpus recovers the participants exactly: **ARI 1.000 at the correct
> k = 18**, for both templates, against a floor of 0.089 at k = 2. Identity
> replacement is per-participant consistent — each participant is mapped to the
> same synthetic appearance every time — which destroys the correspondence to
> their real face while leaving the released set internally self-consistent. The
> same eye ROIs that score 0.2% against A1 score **42.8% / 62.5%** against A2.
>
> **The linkage is not an artefact of surrounding context.** A face ROI retains
> channels the synthesis never edits — hair at the crop border, spectacles, skin
> tone, head pose, face outline — and the floor does not control for them,
> because it is a patch of the *room* whereas those are attributes of the
> *person*. We therefore repeat the linkage attack on the 120 px eye ROI, which
> excludes hair, ears, neck, clothing and the face outline, and whose pixels are
> themselves synthetic in the Full Synthetic arm. Linkage survives: **ARI
> 0.68–0.79** at k = 18 for the two synthetic identities, against **0.84–0.85**
> for unmodified eye ROIs and a floor of 0.089, with 79–86% of frames assigned to
> the correct participant. The subject-level 95% intervals of the synthetic and
> unmodified arms overlap substantially (e.g. [0.59, 0.86] against [0.70, 0.96]
> on the left eye), so replacement cannot be said to reduce eye-ROI linkage at
> all; clustering variability across twenty KMeans initialisations is small
> (SD 0.015–0.037), so single-fit point estimates are representative.
>
> **Nor is it an illumination artefact.** Single-session recording leaves the
> imaging conditions — exposure, white balance, focus distance, the angle of the
> light on the cornea — constant within a participant, and the face-free floor
> does not control for them because they are properties of the imaging of the
> *person* rather than of the room. We therefore attack that channel directly.
> Removing per-crop brightness and contrast entirely leaves the linkage intact
> (ARI 0.684 → 0.742 on the swapped left eye; verification 62.2% → 73.0% on the
> right), local histogram equalisation costs 0.05–0.14 without collapsing it, and
> a large within-session illumination change — gallery from the dark-screen
> stimulus blocks, probe from the natural-image blocks — leaves it intact and
> raises it to 0.918 / 0.931 on unmodified crops. The recogniser is therefore not
> relying on photometry.
>
> **Photometry is nonetheless a second, independent linkage channel.** Twenty-one
> summary statistics per crop — per-channel means, percentiles, white-balance
> ratios, dynamic range, no structure at all — recover the participants at **ARI
> 0.87–0.90** with no face model whatsoever, and identity replacement does not
> touch this channel: 0.898 / 0.901 on swapped crops against 0.870 / 0.902 on
> originals. A release that does not normalise crop photometry is linkable
> without any recogniser at all.
>
> **A second family, and its strongest mode.** DeepPrivacy2 removes identity by
> inpainting rather than replacing it. Its face mode repaints the detected face
> box and touches 40% more of the crop than SimSwap; its full-body mode — the
> paper's main contribution, motivated by exactly our concern that face
> anonymisation leaves identifiers outside the face [CITE: deepprivacy2] —
> repaints the whole person, face included. Crossing edit domain with identity
> regime gives the two-part condition as a factorial result (face-crop T1 ARI at
> k = 18, floor 0.089):
>
> | Edit domain | Fixed identity per participant | Fresh identity per frame |
> |---|---|---|
> | DeepPrivacy2, face | 1.000 | 0.982 |
> | DeepPrivacy2, full body | 0.982 | **0.297** |
>
> Widening the edit domain alone changes nothing, and neither does resampling the
> identity alone; only both together break the partition, and then an attacker
> who must discover the cohort size finds none (silhouette selects k = 2,
> ARI 0.000; pair AUC 0.78). The full-body generator works at 288×160, so its
> faces are visibly low-resolution — but the fixed-identity full-body arm carries
> the same degradation and still links at 0.982, so the collapse is due to the
> identity regime, not to image quality. The residual 0.297 is well above floor:
> what the whole-person generator leaves untouched — background, pose, crop
> geometry, illumination — still carries part of the partition. Against A1 every
> DeepPrivacy2 arm is at or below floor (≤ 1.2%), consistent with DeepPrivacy2's
> own re-identification evaluation, which, like A1, matches enrolment images to
> the release and does not test release-only linkage.
>
> The residual channels the recogniser does exploit — skin tone, periocular
> geometry, crop scale — are signals gaze estimation reads, which is why the axis
> trades at all.
>
> These eye-ROI figures assume an attacker who knows the cohort size: on the eye
> ROI the cluster count is not recovered unaided, whereas on the face ROI it is
> (k = 18 selected by silhouette). For a public release the cohort size is stated
> in the documentation, so the assumption is mild, but it is an assumption and
> the face ROI does not need it. Breaking unsupervised *discoverability* is in
> fact the one place replacement helps: on the right eye, k selected falls from
> 26 (ARI 0.674) to 2 (ARI 0.011). It breaks discovery of the partition, not its
> separability.
>
> **The floor is eye-crop native.** Running the periocular pipeline itself on a
> 120 px face-free patch of the same frames — identical frame selection,
> identical downstream steps, only the pixels differ — gives **TAR@FAR=1e-3 =
> 1.57%**, Rank-1 50.6%, d′ 0.458, agreeing with the face-crop floor to three
> significant figures.
>
> **The open tier leaks too, by two orders of magnitude less.** Our release plan
> puts per-frame gaze estimates and oculomotor measures in an open tier because
> it contains no imagery and no face embeddings. Since eye movement is itself a
> biometric, we tested that rather than asserting it. Treating the released gaze
> series as the attacker's only input, we cut each recording into 5 s fragments
> (150 frames), described each with fifteen dispersion, velocity and
> saccade/fixation statistics, and clustered. The result is **ARI 0.033 at the
> correct k, purity 0.157 against a chance purity of 0.059**, versus a
> label-permutation null of −0.0001 (max 0.0015 over 20 permutations). The signal
> is real but small — two orders of magnitude below the pixel tiers (0.68–1.000).
> We report the open tier as low, not zero, residual risk.
>
> **Metadata leaks before a pixel is read.** Released crop dimensions are
> constant per participant. Face ROI size alone yields an expected Rank-1 of
> 50.0%; face size, eye size and inter-ocular distance together yield **94.4%**
> (4.06 of 4.17 bits) without reading any pixels. A release must normalise crop
> geometry; ours does, and all figures above are pixel-based.
>
> **An adaptive attacker gains little.** An attacker who knows the operator and
> holds paired (original, synthesised) imagery of *other* participants can fit a
> de-anonymisation map and apply it to a participant they could not train on.
> Across mean-shift, orthogonal Procrustes and ridge maps, fitted
> leave-one-participant-out, the best gain is **+3.3 points** on the face ROI
> (19.9% → 23.2%) and nothing on the eye ROIs. The same maps reach Rank-1 100%
> when fitted *including* the target — they memorise identities rather than
> invert the operator. The protection is a generalisation barrier, not an
> information-theoretic one.

### Table P — Identity leakage across release strategies

A1 = attacker holds enrolment photographs; A2 = attacker holds only the release.
TAR at FAR = 1e-3, ArcFace, with 95% subject-level bootstrap CI.

| Released stream | Real Data | Full Synthetic (A) | Full Synthetic (B) | Hybrid (Real Eyes) |
|---|---|---|---|---|
| Face ROI, TAR (A1) | 100% | 22.0% [5.2, 44.5] | 23.3% [8.2, 41.8] | 22.0% (face is synthetic) |
| Left eye ROI, TAR (A1) | 66.6% [53.6, 79.6] | 0.2% [0.0, 3.0] | 0.6% [0.0, 3.1] | **66.6%** (unaltered) |
| Right eye ROI, TAR (A1) | 62.2% [49.0, 84.6] | 1.8% [0.1, 5.7] | 1.1% [0.0, 4.8] | **62.2%** (unaltered) |
| *Session-nuisance floor* | *1.6%* | *1.6%* | *1.6%* | *1.6%* |
| Face ROI, ARI @ k=18 (A2) | 1.000 | 1.000 | 1.000 | 1.000 |
| Left eye ROI, ARI @ k=18 (A2) | 0.844 [0.70, 0.96] | 0.684 [0.59, 0.86] | 0.737 [0.60, 0.88] | 0.844 (unaltered) |
| Right eye ROI, ARI @ k=18 (A2) | 0.853 [0.73, 0.96] | 0.789 [0.64, 0.92] | 0.761 [0.62, 0.95] | 0.853 (unaltered) |
| *Linkage floor* | *0.089* | *0.089* | *0.089* | *0.089* |

---

## 9. Privacy-Utility Experiments — REPLACE Table 3, ADD Table 4, DELETE a claim

### 9-pre. Utility protocol — INSERT at the head of the section

> **Three conditions.** A utility number answers a question only once it says who
> trained the model on what. We report three. In **condition (ii)**, models
> trained on real data are deployed on de-identified crops: the cost to a
> pipeline that receives de-identified input. In **condition (i)**, models are
> retrained on the de-identified release, with the same recipe, participant
> partition and hyperparameters as the real-data models, and tested either on
> **real faces** — a third party training on the released corpus and deploying
> the model on its own users — or on **de-identified faces** — benchmarking on
> the release itself. Condition (ii) is the only one that can separate
> information loss from domain shift when read alongside (i): a penalty that
> retraining removes was shift; one that survives retraining was information.
>
> Two references bound every number. The **initialisation noise floor** is the
> spread of the real-data model over five independent initialisations,
> 0.04–0.20 cm. The **no-information reference** is the error of predicting the
> training participants' mean gaze point for every frame: **9.37 cm**, a penalty
> of +3.70 to +4.41 cm depending on the backbone's own baseline. A model at that
> level has learned no gaze.

### 9a. Table 3 — REPLACE entirely (condition (ii))

Condition (ii): trained on real data, deployed on de-identified crops. Mean over
5 folds, then **mean ± SD over 5 independent initialisations**. Impact
is relative to that model's own Real Data baseline. Replication design and the
three checked timeouts are in `results/INITVAR_VALIDATION.md`.

| Model | Real Data | Full Synthetic (A) | Hybrid (A) | Full Synthetic (B) | Hybrid (B) |
|---|---|---|---|---|---|
| iTracker | 5.85 ± 0.14 | 8.85 ± 0.17 (+3.00) | **6.16 ± 0.20 (+0.31)** | 9.48 ± 0.21 (+3.62) | **5.82 ± 0.20 (−0.03)** |
| MobileNet-V3 Large | 4.98 ± 0.13 | 8.65 ± 0.18 (+3.66) | **5.17 ± 0.16 (+0.18)** | 12.03 ± 0.53 (+7.04) | **6.03 ± 0.25 (+1.04)** |
| AFFNet | 5.67 ± 0.20 | 8.07 ± 0.25 (+2.41) | **5.85 ± 0.24 (+0.19)** | 9.64 ± 0.53 (+3.98) | **5.74 ± 0.25 (+0.08)** |
| MGazeNet | 4.96 ± 0.04 | 7.74 ± 0.20 (+2.78) | **5.21 ± 0.21 (+0.24)** | 9.72 ± 0.40 (+4.75) | **5.45 ± 0.25 (+0.48)** |

The ± is the spread over initialisations, which is also the noise floor of the
whole comparison: **0.04–0.20 cm on the baseline**. Any claimed difference
smaller than that cannot be supported in either direction. (The current draft
reports ±0.03 cm differences as "Preserved"; they are an order of magnitude below
this floor.)

### 9a-bis. DeepPrivacy2 under condition (ii) — ADD as extra columns of Table 3, or as Table 3b

Same models, same five initialisations, same harness. Before any DeepPrivacy2
number was written, the real-data arm was re-run through the same job and
reproduced Table 3 to within 0.0012 cm on all 100 folds.

| Model | Full Synthetic (fixed) | Hybrid (fixed) | Full Synthetic (per frame) | Hybrid (per frame) |
|---|---|---|---|---|
| iTracker | 16.44 ± 0.24 (+10.58) | **5.75 ± 0.11 (−0.10)** | 17.50 ± 0.25 (+11.65) | **5.83 ± 0.06 (−0.02)** |
| MobileNet-V3 Large | 15.88 ± 0.37 (+10.90) | **6.02 ± 0.54 (+1.03)** | 16.24 ± 0.23 (+11.25) | **5.95 ± 0.45 (+0.97)** |
| AFFNet | 19.33 ± 0.26 (+13.66) | **5.74 ± 0.24 (+0.07)** | 20.52 ± 0.22 (+14.85) | **5.68 ± 0.22 (+0.01)** |
| MGazeNet | 17.06 ± 0.39 (+12.09) | **5.43 ± 0.38 (+0.47)** | 17.04 ± 0.53 (+12.08) | **5.37 ± 0.32 (+0.40)** |

"Fixed" and "per frame" are the identity regimes of §8 (DeepPrivacy2 face mode,
truncation 1.0). Hybrid recovers 91–101% of the penalty, as with SimSwap. The
Full Synthetic errors of 15.9–20.5 cm are **worse than predicting the mean**
(9.37 cm): a real-data model given repainted eyes is confidently wrong. Read
these columns with Table 5: most of this penalty is domain shift, and what is
left after retraining is the no-information level.

### 9b. Table 4 — ADD (new)

Hybrid minus Full Synthetic, **paired within initialisation**, 5 draws.
Negative means the genuine ocular crops recover accuracy.

| Model | Identity A | Identity B | Penalty recovered |
|---|---|---|---|
| iTracker | −2.693 ± 0.097 | −3.656 ± 0.320 | 90% / 101% |
| MobileNet-V3 Large | −3.481 ± 0.274 | −5.999 ± 0.655 | 95% / 85% |
| AFFNet | −2.221 ± 0.236 | −3.897 ± 0.428 | 92% / 98% |
| MGazeNet | −2.535 ± 0.119 | −4.273 ± 0.389 | 91% / 90% |
| *Face-only control* | *+0.000 ± 0.000* | *+0.000 ± 0.001* | *0%* |

All 40 draw-level values are negative. The across-draw SD is an order of
magnitude below the effect in every cell.

### 9c. Accompanying text — INSERT

> Restoring the genuine ocular ROIs recovers most of the synthesis penalty in
> every architecture tested. Full synthesis costs **+2.41 to +7.04 cm** depending
> on model and template; Hybrid recovers **85–101%** of that, leaving a residual
> of **+0.08 to +1.04 cm**. Across five independent initialisations the paired
> difference is negative in all four architectures, both templates and every
> draw, with a between-draw spread an order of magnitude smaller than the effect.
>
> The residual is small but mostly real: tested against each model's own
> baseline, six of the eight Hybrid arms are detectably worse than real data and
> two are indistinguishable from it. We therefore describe Hybrid as *recovering*
> the synthesis penalty rather than *preserving* accuracy — a distinction the
> measured initialisation noise floor (0.04–0.20 cm) is fine enough to make.
>
> The face-only control consumes the face ROI only and therefore cannot see the
> eye crops; its two arms must be identical, and they are, to three decimals. It
> establishes that the comparison distinguishes the arms when a model uses the
> ocular stream and reports exactly zero when it does not, which is what licenses
> reading every other row as a real effect.
>
> We find no evidence of an architecture that fails to exploit the restored
> ocular signal. In particular the two guidance-factor architectures, whose eye
> features are modulated by parameters regressed from the face, do not behave
> differently from the late-fusion ones: AFFNet recovers 92% and 98% of the
> penalty. What distinguishes the Hybrid protocol is not fusion topology but
> whether the model reads the eye crops at all, as the face-only control makes
> explicit.

### 9c-ter. Table 5 and text — ADD (training on the release, condition (i))

Penalty in cm against each model's own real-data baseline: mean over MobileNet-V3
Large, AFFNet and MGazeNet, [min, max] across the three. Condition (ii) is
recomputed over the same three backbones so the columns are comparable.
Condition (i) is one initialisation per cell (iTracker, ~8.5 h per fold, was not
retrained). Same recipe and participant partition as the real-data models.

| Arm | (ii) train real, test de-id | (i) train de-id, test real | (i) train de-id, test de-id |
|---|---|---|---|
| SimSwap A, Full Synthetic | +2.95 [+2.41, +3.66] | +3.80 [+2.19, +4.61] | +2.82 [+2.27, +3.39] |
| SimSwap A, Hybrid | **+0.20** [+0.18, +0.24] | **+0.12** [−0.02, +0.31] | **+0.06** [−0.01, +0.16] |
| SimSwap B, Full Synthetic | +5.26 [+3.98, +7.04] | +4.38 [+3.39, +5.42] | +2.52 [+2.22, +2.90] |
| SimSwap B, Hybrid | **+0.53** [+0.08, +1.04] | **+0.11** [−0.16, +0.31] | **−0.02** [−0.23, +0.16] |
| DeepPrivacy2 fixed, Full Synthetic | +12.22 [+10.90, +13.66] | +4.64 [+3.99, +5.03] | +4.51 [+3.96, +4.86] |
| DeepPrivacy2 fixed, Hybrid | **+0.52** [+0.07, +1.03] | **−0.11** [−0.21, −0.03] | **−0.15** [−0.20, −0.07] |
| DeepPrivacy2 per frame, Full Synthetic | +12.73 [+11.25, +14.85] | +4.43 [+3.77, +4.92] | +4.37 [+3.79, +4.74] |
| DeepPrivacy2 per frame, Hybrid | **+0.46** [+0.01, +0.97] | **+0.01** [−0.07, +0.06] | **−0.02** [−0.10, +0.03] |
| *No-information reference (predict the mean)* | *+4.17 [+3.70, +4.41]* | | |

> **Retraining does not rescue full synthesis.** For SimSwap, the penalty
> survives retraining: +2.2 to +3.4 cm when the retrained model is tested on the
> release, and +2.2 to +5.4 cm when it is deployed on real faces — where two of
> the three architectures end up at or beyond the no-information reference. The
> synthesis penalty in Table 3 is therefore mostly lost information, not domain
> shift: a model cannot learn from synthetic eyes what synthetic eyes do not
> carry. A Full Synthetic release is of little use to a third party who wants a
> gaze model for real users.
>
> **Removal by inpainting leaves no gaze signal at all.** For DeepPrivacy2 the
> picture is the reverse. Most of the condition (ii) penalty *is* domain shift —
> retraining removes about two thirds of it — but what remains is the
> no-information level. No retrained model, in any fold, under either identity
> regime or on either test set, does better than predicting the mean (60 of 60
> fold-level errors at or above it, by 0.0 to 1.7 cm); 18 of the 30 training runs
> reach their best validation error at the first epoch and learn nothing
> thereafter, and per-participant calibration does not help. The repainted eyes
> carry no recoverable gaze information. This replaces the qualitative
> observation that the generator "alters gaze direction" with a measurement.
>
> **Hybrid recovers the penalty in every condition.** Restoring the genuine
> ocular ROIs recovers 86–104% of the penalty when the retrained model is
> deployed on real faces and 93–108% on the release, for both families and both
> identity regimes; every Hybrid cell in condition (i) lies within ±0.31 cm of
> baseline, most inside the initialisation noise floor. The central utility
> finding of Table 4 therefore does not depend on how the model was trained.
>
> **The identity regime costs little; the edit domain costs everything.** Once
> models are trained on the release, resampling the synthetic identity every
> frame changes utility by at most 0.31 cm against a fixed identity, in either
> direction, for both Full Synthetic and Hybrid (in condition (ii) the Full
> Synthetic arm moves by up to 1.2 cm, but there both regimes are already worse
> than predicting the mean). What destroys utility is repainting the eyes — and
> repainting everything the crop contains, eyes included, is what §8 found
> necessary to break face-crop linkage.

Caveats to keep beside Table 5: one initialisation per cell, so differences under
about 0.4 cm (twice the noise floor) are not findings, and Hybrid's negative
values read as "indistinguishable from baseline". Fold 2 is hard for every model,
real-data included (7.3–8.5 cm), which compresses all penalties there; tested
on the release, the SimSwap Full Synthetic penalty is positive in 29 of 30
fold-level cells (2 templates × 3 backbones × 5 folds, against the 5-draw
real-data fold mean).

### 9c-bis. What the cost means — INSERT after 9c

Review point M2: a centimetre figure on its own does not say whether a protocol
is usable. Anchor it to what each biomarker needs to resolve.

**Which number to anchor on.** Table 3 reports the uncalibrated `base` error,
which is the right quantity for comparing *arms* — holding calibration out
isolates the effect of the imagery. It is the wrong quantity for asking whether
the instrument is clinically usable, because a deployed screening tool calibrates
each user. With per-participant calibration the same four architectures reach
**3.70–5.11 cm on real data**, which at the 80–120 cm viewing distance of our
protocol is **1.8–3.7°**.

| Biomarker | What it must resolve | Requirement | Verdict at 1.8–3.7° |
|---|---|---|---|
| Anti-saccade directional error | which hemifield the gaze went to | coarse: roughly half the stimulus eccentricity | **usable** |
| Saccade latency | movement onset | one frame; 100°/s detection threshold [CITE: saccadethresh] | **usable** — measured bias −16.8 ms, under one 33 ms frame |
| Search-task dispersion | spread over tens of cm | coarse | **usable** |
| Fixation stability (BCEA) | healthy median 0.75–1.07 deg², a sub-degree ellipse [CITE: bcea] | < 0.5° | **below our noise floor** |
| Peak saccade velocity | a 30–80 ms movement | ≥ 250 Hz sampling | **undersampled at 30 Hz** |

> Two of the five biomarkers are beyond this instrument before de-identification
> is considered, for reasons de-identification cannot change: fixation stability
> lives at a sub-degree scale our error cannot reach, and peak saccade velocity
> needs a sampling rate an order above 30 Hz. The three that do work are the
> coarse ones — hemifield discrimination, movement onset, and large-scale
> dispersion.

Calibrated, per arm:

| Arm | cm | angular (120 → 80 cm) |
|---|---|---|
| Real Data | 3.70–5.11 | 1.8–3.7° |
| Hybrid (A / B) | 4.42–5.65 / 5.10–5.48 | 2.1–4.0° / 2.4–3.9° |
| Full Synthetic (A / B) | 7.28–8.97 / 8.69–11.21 | 3.5–6.4° / 4.1–8.0° |

> This is what makes the utility axis interpretable. Hybrid leaves the instrument
> in the same regime (2.1–4.0° against a 1.8–3.7° baseline), so the biomarkers
> that worked still work. Full synthesis moves it to 3.5–8.0°, which is not a
> proportionate cost along a continuum but a move out of the regime: at 8° the
> hemifield discrimination that anti-saccade error rate depends on is no longer
> comfortably resolved. **The protocol that keeps the measurement clinically
> usable is precisely the one that releases the participant's own ocular
> biometrics.**

**On comparing with the literature.** Webcam-based eye tracking validated on
clinically relevant saccade and free-viewing paradigms reports about 1.42° mean
error [CITE: webcamval]. That figure is obtained *after* careful per-user
calibration, so the like-for-like comparison is against our calibrated 1.8–3.7°,
not against the uncalibrated numbers in Table 3. On that basis the two sit in the
same regime, with ours somewhat coarser. Quoting 1.42° beside an uncalibrated
figure would overstate the gap.

**To check with a clinician before submission:** the requirement column is
derived from published normative ranges, not from a target diagnostic sensitivity
for a named condition. If the intended screening application has an established
minimum detectable effect, that is the better anchor and should replace this.

### 9d. DELETE

> ~~iTracker and AFFNet consistently failed the Hybrid test regardless of the
> identity used (errors remaining >8.4 cm). This corroborates the hypothesis that
> the 'Context-Signal Mismatch' is a fundamental architectural incompatibility,
> rather than an artifact of a specific face template.~~

In this pipeline neither model fails and AFFNet preserves best of the four.
Remove the sentence and the conclusion — do not soften them.

### 9e. Table 6 — RESTRUCTURE

The "Deviation" column conflates absolute performance with cross-template
stability, which produces incoherent labels: an *improvement* of −0.90 cm marked
"(Failure)", and "−0.17 cm (Stable Failure)". Report cross-template deviation
alone in the column; whether an arm performs poorly is a sentence in the text.

Note also that the current Table 3 and Table 6 disagree on iTracker
(9.36 / 9.35 versus 9.35 / 9.36). Both are superseded by §9a.

---

## 10. Limitations and Conclusion — INSERT (~20 lines)

> Neither protocol yields a corpus that can be released without further control.
> Full synthesis removes the link to the real face at a gaze cost of 2.41–7.04
> cm, a cost that retraining on the release does not remove (+2.2 to +5.4 cm);
> inpainting the face removes the gaze signal outright, leaving retrained models
> no better than predicting the mean. Hybrid recovers 85–101% of the penalty —
> 86–108% after retraining — but only at the price of releasing genuine ocular
> biometrics; and both protocols leave the corpus linkable by participant, so one
> labelled example compromises a cluster. The linkage is structural rather than incidental: it survives on a
> 120 px ocular crop containing none of the contextual cues a face crop carries,
> and the channels that carry it are the channels gaze estimation reads, so any
> method whose edit domain is inner-face texture inherits the limit.
>
> Five constraints bound these conclusions. Participants were recorded in a
> single session. We attacked the resulting confound directly — photometric
> normalisation and a large within-session illumination change both leave the
> linkage intact — so the recogniser is not exploiting imaging conditions. But
> the dissociation is partial: changing the screen does not change the ambient
> lamp, the camera position or the seating, and "this participant's skin tone
> under this lamp" cannot be separated from "this recording's exposure" without a
> second session. That also bounds the photometric attack we report, which is
> high precisely because the two are confounded. The longitudinal tracking that
> makes linkage consequential is itself the cross-session case, so a second
> recording remains the decisive experiment and we did not run one.
> The cohort is 18 healthy adults, moderately homogeneous in ethnicity and age
> and recruited from a single institution, so we make no claim about clinical
> populations and describe the contribution as an evaluation of dataset release
> rather than of home screening. Privacy is measured with two recogniser
> families, both trained on web-scale face corpora. The structural claim is
> tested on one replacement method (two templates) and one removal method in two
> modes; diffusion-based de-identification is untested. Full-body DeepPrivacy2
> ran without its generator for persons that DensePose fails to map (the weights
> are no longer distributed); no such person occurred in our frames, but on other
> footage that path would leave people unedited. Retraining on the release was
> run for three of the four architectures and one initialisation per cell, so
> it supports the direction and rough size of each penalty, not differences
> below about 0.4 cm; and the utility of full-body DeepPrivacy2 was not
> measured — it repaints the eyes at lower resolution than the face mode whose
> eyes already carry no gaze signal, so we treat the no-information level as its
> lower bound rather than report a number.
>
> A deployable route must satisfy both halves of the condition we establish:
> resample the synthetic identity per recording session, *and* ensure the release
> contains nothing the transformation does not edit — by cropping to the edit
> domain, or by widening the edit domain to the whole person, which is the one
> configuration we found that breaks face-crop linkage (0.297), and which also
> repaints the eyes. Our per-frame experiment
> demonstrates the mechanism but is not itself deployable — it destroys the
> temporal coherence a gaze pipeline needs — and per-session resampling is
> precisely what a single-session corpus cannot evaluate. The alternative of
> protecting the ocular region while preserving the gaze geometry inside it is an
> axis we examined and found to trade linearly, with no setting that was both
> private and accuracy-preserving. Until a corpus exists on which the two-part
> condition can be met and verified, controlled access under an enforceable
> agreement, not open deposit, is what the measurements support.

---

## 11. Figures

All in `results/paper/figs_core/` (PNG + PDF), generated from result files by
`scripts/anon/make_core_figs.py`; no number is typed in by hand.

| Figure | File | Shows | Placement |
|---|---|---|---|
| Privacy–utility plane | `fig_core3_tradeoff.pdf` | every arm on one plane; the "private and useful" corner is empty | **main text, the one privacy figure** (§9, after Table 5) |
| Release-only linkage | `fig_core1_privacy.pdf` | (a) the 2×2 on the face crop; (b) eye-ROI linkage with CIs | main text if a second figure fits (§8); otherwise supplement |
| Utility under three conditions | `fig_core2_utility.pdf` | Full Synthetic vs Hybrid penalty, (ii) / (i)-real / (i)-de-id | supplement; Table 5 carries the numbers |
| Two attackers | `fig3_threat_models.pdf` | same crops, A1 vs A2, 0.2% → 43% | supplement (was the main privacy figure) |

The plane replaces the threat-model figure in the main text: it carries the
paper's conclusion in one panel, where the threat-model figure carries a
methodological point that §7 already makes in prose.
`fig5_linkage_release_only.pdf` and `fig1_periocular_leakage.pdf` remain
journal-version figures; fig 5b's caption should read the measured eye-native
floor (1.57%), not "borrowed from panel a".

**Caption, privacy–utility plane.**

> **Privacy against utility for every protocol tested.** Horizontal: gaze-error
> penalty of a model retrained on the release and deployed on real faces (mean
> of three architectures). Vertical: release-only linkage, the ARI at k = 18 of
> the most linkable crop the release contains (face or eye ROI). Circles: Full
> Synthetic; squares: Hybrid (original eyes); star: unprocessed data. Dashed
> red: no-information reference (predicting the mean gaze point); dashed grey:
> linkage floor. The shaded corner — private and useful — contains no protocol.
> Hybrid sits at the utility of real data and the linkage of real data; Full
> Synthetic loses most of the gaze signal and none of the linkage. The single
> point that leaves the top row (diamond: DeepPrivacy2 full body with a fresh
> identity per frame, ARI 0.297) repaints the eyes; its utility was not measured
> and is drawn at the no-information level as a lower bound.

**Caption, release-only linkage.**

> **(a)** Face-crop linkage (ARI at k = 18) for DeepPrivacy2, crossing edit
> domain (face box vs whole person) with identity regime (fixed per participant
> vs fresh per frame). Only the cell that meets both halves of the condition
> falls (0.297); unprocessed and SimSwap face crops are at 1.000. **(b)** Eye-ROI
> linkage for every arm, with 95% subject-level bootstrap intervals; dashed: the
> eye-native floor (0.089). Every arm with a fixed identity per participant
> overlaps unprocessed eyes; only per-frame identity approaches the floor.

**Caption, utility under three conditions.**

> **(a)** Full Synthetic and **(b)** Hybrid penalty against each model's
> real-data baseline, for models trained on real data and tested on
> de-identified crops (grey), and models retrained on the release and tested on
> real (green) or de-identified (purple) faces. Bars: mean of three
> architectures; dots: each architecture. Red band: no-information reference;
> grey band: initialisation noise floor. Retraining removes most of the
> DeepPrivacy2 penalty but stops at the no-information level; it does not remove
> SimSwap's; Hybrid is at baseline in every condition.

---

## Open items

1. **CHECK** whether *Practical Digital Disguises* runs a linkage attack (§3).
2. **CHECK** the cohort count and demographics (§4).
3. **DECIDE** the title (§0).
4. ~~**WAIT** for the initialisation-variance replication before pasting Table 3.~~
   Done (2026-09-20); Table 3 is final.
5. ~~**CHECK** the DeepPrivacy2 quotations and numbers against the WACV PDF.~~
   Done 2026-10-08 against the CVF open-access version and its supplement: the
   quotation (p. 1, "does not prevent recognition through identifiers outside
   the face, including both primary (e.g. ears, gait) and secondary (e.g.
   gender) identifiers"), Table 1 (R1 94.4 → 44.7, mask-out 45.5, mAP 82.5 →
   8.5) and the attack setup (Appendix C: original query images against the
   anonymised test images, i.e. our A1) all match.
6. **DECIDE** whether to retrain iTracker under condition (i) (~8.5 h per fold ×
   5 folds × 8 arms) so Table 5 covers all four architectures.
