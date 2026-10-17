# Keep Your Eyes On It: Measuring Gaze Utility and Re-identification Risk in De-identified Smartphone Gaze Data

## ABSTRACT

Smartphone eye tracking moves oculomotor assessment out of the clinic, but what
it records is video of the participant's face — which is what blocks the
dataset sharing the field depends on. Face swapping has been proposed to make
such video shareable, and has been evaluated by whether the swapped face still
matches the real one and how much gaze or behaviour survives. Neither
evaluation asks whether the release can be linked to itself without any
outside reference, or what a gaze model trained on the release is worth. We
measure both.

We introduce CognitiveGaze: 18 participants, seven oculomotor tasks, recorded
simultaneously by a smartphone and an EyeLink 1000, so every smartphone frame
carries a physiological ground-truth gaze coordinate rather than a
screen-target label. Utility is measured by deploying four gaze architectures
on de-identified crops and by retraining three of them on the de-identified
release. Privacy is measured on the released crops themselves, with pretrained
recognisers, two attacker models, and a control establishing what "no identity
signal" scores on single-session data.

The axes do not trade as assumed. Retaining the genuine ocular region recovers
85–101% of the synthesis penalty in every architecture, and 91–106% when the
models are retrained on the release — yet those same crops verify identity at
62–80% TAR at FAR 1e-3 against a measured 1.6% floor, under two unrelated
recognisers. The eye pixels that make gaze estimation work are the pixels that
identify. Full synthesis breaks the link between the released eye crops and the
real face, but the synthetic face crop still verifies against the participant's
own photographs (18–24% TAR under ArcFace), retraining leaves a 2.2–6.2 cm cost,
and the release still singles out every participant: an unsupervised partition
recovers exactly the 18 individuals, even from a 120 px ocular crop containing
no hair, ears, clothing or face outline. Removal by inpainting fares worse: a
model retrained on it is no better than predicting the screen centre. A
diffusion anonymiser yields models that transfer to real faces far better
(0.4–1.3 cm against 2.5–6.2 cm), but not because it keeps more gaze: its pupils
follow true gaze less closely than SimSwap's, and stronger settings trade pupil
signal for identity protection without reducing linkability. Releasing eye
landmarks instead of pixels links too and carries almost no gaze signal. A
usable gaze corpus must keep each participant's frames together, so it is
linkable by construction; within the content, the one configuration that breaks
linkage — repainting the whole person with a fresh identity every frame — is one
on which a retrained gaze model learns nothing. We state what a releasable gaze
corpus would require, and report what our own tiered release leaks when measured
rather than assumed.

**CCS Concepts:** • **Security and privacy** → **Pseudonymity, anonymity and untraceability**; • **Human-centered computing** → **Empirical studies in ubiquitous and mobile computing**; • **Computing methodologies** → *Biometrics*.

**Additional Key Words and Phrases:** smartphone gaze estimation, face de-identification, re-identification, linkage attack, privacy–utility trade-off, ocular biometrics, dataset release, privacy auditing

# 1 INTRODUCTION

Oculomotor function is a sensitive early marker of neurological and cognitive
change, and the instruments that measure it well — infrared trackers such as
the EyeLink 1000 — are confined to the clinic. Smartphone gaze estimation moves
the measurement to where the patient already is [krafka16,
valliappan20], which is why it has become an active route to accessible
screening. What it does not move is the data. A smartphone eye tracker records
video of the participant's face, and a face video is not a shareable research
artefact. The result is a field whose progress depends on shared corpora and
whose corpora cannot be shared.

The consequence is a cycle that reproduces its own scarcity (Figure 1).

![Figure 1](figs_core/fig_cycle.pdf)

**Figure 1.** The data-scarcity cycle in smartphone gaze research. Few public corpora record the face under conditions a study can use (1), so study designs converge on the same corpora and inherit their protocol (2). Groups that record data for a clinical question (3) cannot release it, because face video is personal and health data (4), and the standard exits do not yield a usable public corpus: blurring or masking removes the gaze geometry, computation on data that stays with its holder produces no shared benchmark, and controlled access restricts who can use it (5). The corpus stays unreleased or restricted, and the next group faces the same scarcity (6). A de-identification that removes identity and keeps gaze would break the cycle at step 5; this paper measures whether any current method does.

Researchers who want to study gaze on smartphones find few public corpora that
record the face under conditions they can use, so they build on the same few —
GazeCapture above all [krafka16] — and their study designs inherit those
corpora's assumptions: a hand-held phone, a small screen with the camera above
it, a viewing distance of a few tens of centimetres, and the position of an
on-screen target standing in for measured gaze. Newer datasets adopt the same
protocol to remain comparable [zjugaze, dynamicgaze], so the assumptions
harden into the benchmark. Clinical oculomotor protocols break most of them —
stimuli span a large display viewed from a metre away, and the quantity of
interest is the gaze trajectory rather than the location of a target — and
whether models developed under the hand-held protocol transfer to them is
largely untested. Clinical groups that do record such data hold it in
separate silos, so there is no common reference against which analysis choices
can be checked, and those choices multiply [simmons11, carp12]. A group
that records the data a clinical question needs then wants to release it, and
meets the privacy problem. Face video of a study participant is personal data,
and in a clinical study also health data, a special category under the GDPR
[gdpr]; full-face photographs and comparable images are among the
identifiers HIPAA requires to be removed [hipaa]. None of the standard
exits yields a corpus the next group can use. Blurring or masking the face
removes the geometry a gaze model reads [blurmask, disguises].
Computation on data that stays with its holder — homomorphic encryption,
federated learning — leaves the data where it is and produces no shared
benchmark, at a computational cost that is prohibitive for video
[acar18he, kairouz21fl]. Controlled access, the usual compromise, admits
only those who can apply to and sign an agreement with the holding institution.
The corpus stays unreleased or restricted, and the next group looking for
usable public data finds the same few corpora. The cycle closes.

It has one step at which it could break without giving up release: a
transformation that removes identity and keeps the gaze signal would let a
clinical gaze corpus be shared openly. Generative de-identification is the
proposed candidate: replace the face, keep the
behaviour. Face swapping is attractive here because it is meant to preserve the
geometry — head pose, eyelid aperture, iris position — that a gaze model reads,
where methods that inpaint the ocular region destroy it. It has been applied to
clinical video of movement disorders, autism assessments and epileptic
seizures [zhu20deepfakes, disguises, li24seizure]; gaze is known to
degrade under it unless the generator is explicitly constrained to keep it
[wilson23gaze]; and a recent evaluation concludes that face swapping
hides identity effectively enough to serve as a video anonymiser
[mustu25] — measured, as in the work before it, by whether the swapped
face can be matched to the person's real face.

We find that it does not, for gaze data, and that the reason is structural
rather than a shortcoming of any particular generator. Two observations drive
the paper. First, the ocular region is not a de-identified crop of the face: a
120 px eye ROI handed to a pretrained recogniser verifies identity at 62–80%
at a false-accept rate of 1e-3, against a measured floor of 1.6%. The pixels
that carry the gaze signal are the pixels that carry the identity, so a
protocol that recovers accuracy by retaining genuine ocular imagery retains
the biometric with it. Second, replacing the whole face — ocular region
included — removes the correspondence between the released eye crops and the
participant's real face, but not between the synthetic face crop and the real
face, and it leaves the released corpus exactly partitionable by individual —
which, under the European test for anonymisation, is enough for the data to
remain personal data [wp29anon]. Identity
replacement is per-participant consistent, and the channels that remain are
illumination, skin tone, periocular geometry and crop scale: attributes of the
person that lie outside the inner-face region any swapping method edits.

These are claims about measurement, so the paper is built around an instrument.
We introduce **CognitiveGaze**: 18 participants performing seven oculomotor
tasks, recorded simultaneously by a smartphone and an EyeLink 1000 and
synchronised to the frame, so that each smartphone frame carries a
physiological ground-truth gaze coordinate rather than a screen-target label.
That is what makes a sub-centimetre difference on the utility axis
interpretable. The protocol is deliberately not GazeCapture's: stimuli appear
on a 53 × 30 cm display viewed from 80–120 cm, not on the phone's own screen,
and they are clinical oculomotor tasks — the kind of corpus the cycle above
keeps unreleased. On the privacy axis we adopt the discipline of the security
literature: a stated attacker, a pretrained recogniser never trained on our
data, a second recogniser family as a cross-check, and a session-nuisance
control that establishes what "no identity signal" scores on single-session
recordings — without which a 62% verification rate cannot be distinguished from
the fact that each participant was recorded once, in one chair, under one lamp.

This paper makes four contributions.

**A privacy evaluation protocol for gaze corpora.** We evaluate
de-identification the way the security literature evaluates it — with a stated
attacker, a pretrained recogniser that never sees our data, and a
session-nuisance control — rather than asserting it from the generator's
objective. We distinguish re-identification by an attacker holding the
participants' photographs (A1) from a test of whether the released content
singles out individuals with no external reference (A2, the linkability
criterion of the European anonymisation standard), and show the two differ by up
to 60 points on identical pixels, which makes an unqualified privacy number
uninterpretable.
We release the protocol as an audit tool and a fifteen-item release checklist
for gaze corpora, distilled into seven design guidelines (§10)
[privaudit-repo]; run on this study's saved embeddings it
reproduces every eye-ROI verification and linkage figure exactly, and end to end,
re-embedding the crops, to within GPU inference noise.

**Gaze utility and linkability share a representation.** The ocular region is
simultaneously the signal gaze estimation needs and the region that identifies
best per pixel. We show that the protocol which recovers accuracy does so by
releasing genuine ocular biometrics — recovering 85–101% of the synthesis
penalty in *every* architecture we test, and 91–106% when the models are
retrained on the release, because in every case what it restores is the
participant's own eye pixels. Releasing landmark geometry instead of pixels does
not escape this: four iris coordinates per frame still link the release (ARI
0.27, verification 10% against a 1.6% floor) while predicting gaze barely better
than the mean. And full synthesis,
which does remove the correspondence to the real face, still leaves the corpus
exactly partitionable by individual. Because the residual channels —
illumination, skin tone, periocular geometry, crop scale — lie outside the edit
domain of inner-face replacement, the limit is structural rather than a defect
of one generator: we confirm it on a second family — DeepPrivacy2, which
*removes* identity by inpainting rather than replacing it, and in its face mode
touches 40% more of the crop — and the released face crop is still exactly
partitionable (ARI 1.000); so is it, near enough (0.982), when DeepPrivacy2
repaints the whole person but keeps each participant's synthetic identity fixed.

**A tested, two-part condition for breaking content-based linkage.** Rather
than stopping at a negative result, we state the requirement as a falsifiable
claim and test it by construction. It concerns releases that strip participant
identifiers; one that keeps them, as a usable gaze corpus must, is linkable by
construction. Linkage from the released content falls only when **both** the
transformation breaks per-participant consistency **and** the released crop
contains nothing outside the transformation's edit domain. We test it as a 2×2
— face-only vs full-body DeepPrivacy2, fixed vs per-frame identity — and only
the cell that meets both halves breaks: face-crop linkage is 1.000, 0.982 and
0.982 in the other three cells and **0.297** in that one, where an attacker who
does not know the cohort size recovers no structure at all (ARI 0.000). It
breaks *discovery*, not *verification*: an attacker holding a few labelled
frames per person still verifies 35.7% [27, 45] of that cell's face crops,
against 90–100% in the other three and a 1.6% floor. On the eye ROI, which
contains only edited pixels, per-frame identity alone suffices: verification
63.2% → 2.04% and linkage 0.805 [0.63, 0.94] → 0.148 [0.10, 0.27], against
floors of 1.57% and 0.074. The one configuration that breaks linkage repaints
the eyes, and a gaze model retrained on repainted eyes
does no better than predicting the screen centre. Each half of the condition is
attainable on its own — a diffusion anonymiser meets the first, full-body
inpainting meets both while destroying the gaze signal —
which locates what is missing precisely: a gaze-preserving generator whose edit
domain covers the whole released crop. This tells a dataset author what to do,
and what it costs, not merely what fails.

**CognitiveGaze, and a four-architecture utility benchmark under
de-identification.** 18 participants, smartphone and EyeLink 1000 recorded
simultaneously across seven oculomotor tasks. We report the utility cost of
both de-identification protocols, for two de-identification families, under
three conditions — trained on real data and deployed on the de-identified
crops, and retrained on the de-identified release then tested on real or on
de-identified faces — against a measured initialisation noise floor and a
predict-the-mean reference that marks where a model has learned no gaze at
all, including a face-only positive control that isolates whether a model's
ocular stream is active at all.

# 2 RELATED WORK

**Smartphone gaze estimation.** Appearance-based gaze estimation on commodity
phones was established by iTracker and the GazeCapture corpus [krafka16],
and brought to research-grade accuracy with per-participant calibration by
Valliappan et al. [valliappan20]. Subsequent architectures refine how the
ocular stream is conditioned on facial context: AFFNet modulates eye features
through adaptive group normalisation driven by a face-and-grid guidance factor
[affnet], and MGazeNet extends the idea with linear adaptive batch
normalisation [zhu24]. All of them consume the same released artefacts —
a face ROI, two eye ROIs and a face grid — so all of them inherit the same
data-sharing problem.

**Face de-identification.** Methods built for anonymisation optimise identity
*removal*. The k-Same family guarantees that a released face cannot be matched
to fewer than k enrolled identities by replacing faces with averages
[newton05], and its neural successors extend the guarantee to generated
faces [ksamenet]. DeepPrivacy inpaints the facial region with a
conditional generator, preserving pose and background [deepprivacy];
DeepPrivacy2 extends inpainting to the whole body, on the argument that face
anonymisation leaves identifiers outside the face, and evaluates it by
re-identifying anonymised pedestrians from original images [deepprivacy2]; CIAGAN conditions the generation so that
expression and pose survive while identity does not [ciagan]; more recent
work removes identity with diffusion models, conditioning a generator on the
source face so that attributes survive [fams], or editing in diffusion
latent space, including training-free variants [fluid, apl]. Face *swapping* methods such as SimSwap and
SimSwap++ instead optimise identity *replacement* [simswap, simswappp].
The distinction is expected to matter for gaze: removal methods that inpaint
the ocular region should destroy the signal the task depends on, whereas
replacement is meant to preserve the underlying geometry. We measure both
expectations. The first holds outright — a gaze model retrained on inpainted
eyes learns nothing — but the second only partly: synthetic eyes keep enough
geometry to cost 2–5 cm rather than everything. On the privacy axis the
difference in objective does not decide the outcome, because the leakage we
measure lies outside the edit domain that both families share.

**Face swapping as a privacy mechanism.** Swapping has accordingly been
proposed for exactly our setting. Zhu et al. swap faces in clinical movement
videos and show that body keypoints needed for diagnosis survive
[zhu20deepfakes]; Li et al. compare three swapping models, SimSwap among
them, on seizure videos, rated subjectively on retained semiology
[li24seizure]. Wilson et al. build a face-swap pipeline for
recorded clinical assessments and give the first privacy–utility methodology
for patient-privacy face swaps, measuring how far gaze and expression survive
relative to blurring [disguises]; they then add a gaze-prediction loss to
the swap model, because full-face losses miss the small eye region and gaze is
not otherwise preserved [wilson23gaze]. A digital-mask approach was reported in
the clinical literature [digitalmask] and promptly contested
[maskconcerns], so the sufficiency of generative de-identification for
clinical release is an open question rather than a settled one. Most directly,
Muştu and Ekenel evaluate face swapping as a video anonymiser on temporal
consistency, anonymity strength and visual fidelity, and conclude that it
hides identity effectively [mustu25]. **We reach the opposite conclusion,
and the difference is the test.** Anonymity strength as measured there asks
whether a swapped face can be matched to the person's real face. On that
question our results agree for the synthesised eye crops, which fall to the
floor, but not for the synthetic face crop, which still verifies against the
participant's photographs at a strict operating point. Nor does it ask whether
the released corpus singles out individuals with no external reference — the
linkability criterion under which European regulators judge whether data are
anonymous [wp29anon] — and that property survives the swap almost intact.

**Eye movement as a biometric.** Identity is recoverable from eye movement
itself, through oculomotor-plant characteristics and scanpath statistics
[rigas, holland], as formalised by the BioEye and EMVIC benchmarks
[bioeye]. The eye-tracking privacy literature that follows has developed
differential-privacy mechanisms that account for temporal correlation
[bozkir, steil], reinforcement-learning obfuscation [fuhl], and
privacy-preserving scanpath comparison [scanpath]; the area now has a
decade review [etprivreview] and a treatment specific to handheld mobile
devices [alsakar25]. This line is why we do not treat a pixel-free
release tier as risk-free by construction, and instead measure it.

# 3 THE COGNITIVEGAZE DATASET

CognitiveGaze was recorded to benchmark smartphone gaze estimation against a physiological ground truth rather than against the position of an on-screen target.

**Participants.** Twenty-three volunteers (13 male, 10 female) were recruited. Five were excluded before analysis for calibration failure or incomplete data acquisition, leaving a final cohort of N = 18. Inclusion required no history of photosensitive migraine, epilepsy, motion sickness, or substance use affecting oculomotor function. The final cohort spanned 20 to 54 years (median 29) and comprised White/Caucasian (n = 12), East Asian (n = 2), Black/African (n = 1), Middle Eastern (n = 1), Asian-Indian (n = 1) and mixed-background (n = 1) participants. Three wore glasses. Unlike many open gaze corpora, we did not exclude spectacle wearers, because vision correction is frequently a prerequisite for completing a cognitive protocol and excluding it would compromise clinical generalisability. One participant has no usable ground-truth gaze recording; utility analyses therefore use 17 participants and privacy analyses all 18. The protocol was approved by the institutional research ethics committee (reference withheld for review), and written informed consent was obtained after a briefing on the protocol and its risks.

**Apparatus.** Participants sat 80–120 cm from a 53.3 × 29.6 cm display (1920 × 1080 px) without head restraint (Figure 2). Ground-truth gaze was recorded at 1000 Hz by an EyeLink 1000, calibrated with a standard nine-point sequence; a target sticker on the forehead supported head tracking. A smartphone (Xiaomi Mi 8) was mounted directly below the display, and its rear camera (12 MP, 30 fps) recorded the participant's face. Stimuli were programmed in PsychoPy. Blinks and tracking losses appear as missing samples in the ground truth; unlike the GazeCapture protocol, blink frames were retained and annotated by a blink detector, because blink behaviour is itself a cognitive marker.

**Tasks.** The battery was designed with rehabilitation specialists and informed by established smartphone eye-tracking protocols [valliappan20]. After calibration, participants completed seven oculomotor tasks: *fixation stability* on a central cross; *prosaccades* to targets at the four screen corners and then five random locations; *anti-saccades*, in which participants suppressed the reflexive saccade to a red target at one of eight peripheral positions and looked instead to its mirror location; *smooth pursuit* of a target moving horizontally at constant speed along a triangular waveform; *optokinetic nystagmus*, tracking one stripe of a drifting vertical grating until it left the screen; *free viewing* of 30 natural scenes from the OSIE dataset [osie]; and *visual search* for a named target in 12 of those scenes. Each block was preceded by a 10–20 s instruction screen, rest breaks were interspersed between blocks, and participants were familiarised with the instructions beforehand.

**Synchronisation.** Participant and investigator kept silent during trials, because sound is the synchronisation channel. A 0.6 s tone preceded each task block; it was recorded by the smartphone microphone and timestamped by the experiment software. Detecting its onset in the smartphone audio gives a shared anchor t₀, against which video frame indices are mapped to EyeLink time. The 1000 Hz ground truth is then resampled to the 30 Hz video, so that every smartphone frame carries a synchronised ground-truth gaze coordinate (x, y).

![Figure 2](FIGURE: original setup figure)

**Figure 2.** Data acquisition setup. (A) Participant station: the participant sits 80–120 cm from the stimulus display without head restraint; the smartphone is mounted below the display and the EyeLink 1000 records ground-truth gaze. (B) Control workstation: the investigator monitors the live gaze trace and calibration accuracy.

# 4 GAZE ESTIMATION MODELS

**Inputs.** Every model consumes the same four streams extracted from each smartphone frame: the left-eye, right-eye and face ROIs, resized to 224 × 224 × 3, and a 25 × 25 binary face grid encoding the face's position in the frame. These cropped streams, not the video, are what a release would contain (§6).

**Architectures.** We evaluate four published architectures (Table 1, Figure 3). iTracker and MobileNet-V3 fuse the streams late, concatenating the per-stream embeddings before a fully connected regression head; AFFNet and MGazeNet instead regress a guidance factor from the face and use it to modulate the normalisation of the eye features. We adapted MGazeNet to 224 px input, with an additional squeeze-and-excitation and LABN block in its fusion stage.

**Table 1.** The four gaze architectures. All regress the gaze point (x, y) on the display.

| Model | Eye streams | Face and grid streams | Fusion and normalisation |
|---|---|---|---|
| iTracker [krafka16] | weight-shared CNN, 128-d | face CNN, 64-d; grid MLP, 128-d | late concatenation |
| MobileNet-V3 Large [mobilenetv3] | MobileNet-V3 backbones | MobileNet-V3 face backbone; grid encoding as iTracker | late concatenation |
| AFFNet [affnet] | CNN with SE blocks | face features regress a guidance factor | adaptive group normalisation of eye features, conditioned on the factor |
| MGazeNet [zhu24] | CNN with SE blocks | face features regress a guidance factor | linear adaptive batch normalisation (LABN) of eye features; SE + LABN fusion block |

![Figure 3](FIGURE: original architecture comparison figure)

**Figure 3.** System architecture and processing workflow. (A) Layer-level comparison of the four gaze architectures. (B) Gaze estimation workflow, from ROI extraction from the smartphone video through multi-stream feature fusion to supervised training against the EyeLink 1000 ground truth.

**Training and evaluation.** Models are trained with a mean-squared-error loss on the gaze point in normalised screen coordinates and evaluated by the mean Euclidean distance between predicted and ground-truth gaze, converted to centimetres on the display. To measure generalisation to unseen people rather than memorisation of their appearance, we use participant-wise five-fold cross-validation [kfold]: the data are partitioned by recording rather than by frame, weights are re-initialised for every fold, and the checkpoint with the lowest validation error is kept. Every configuration is trained from five independent initialisations, whose spread gives the noise floor of all utility comparisons. Arms are compared on the uncalibrated model, which isolates the effect of the imagery; for the clinical-usability analysis (§8.4), each model's regression head is additionally fine-tuned per participant on frames from the calibration sequence recorded before the tasks. A face-only model, which receives the face ROI and grid but no eye crops, serves as a null control: it cannot distinguish two releases that differ only in their eye crops.

# 5 DE-IDENTIFICATION METHODS

We test three families of generative de-identification, chosen because they differ in what they edit and how they treat identity. Each operator is run on the original full frame, and the released crops are re-extracted from the processed frame with the stored per-frame geometry, so that crops are pixel-comparable across every arm.

**Replacement: SimSwap++.** An adapted SimSwap++ [simswappp] transfers the identity of a source template onto the participant's face while preserving the target's geometric attributes — gaze direction, head pose and expression. Its conditional dynamic convolution and morphable knowledge distillation give high-resolution synthesis at low cost (about 2.5 GFLOPs). Each participant is mapped to the same synthetic appearance in every frame. To separate the method from the choice of face, every experiment is run with two source templates of different appearance, Identity A and Identity B (Figure 4). We refer to this family as SimSwap below.

**Removal: DeepPrivacy2.** DeepPrivacy2 [deepprivacy2] removes identity by inpainting rather than replacing it. Its *face* mode repaints the detected face box; its *full-body* mode repaints the whole person, guided by a dense surface map. We run both, each under two identity regimes: a latent code held fixed per participant (*fixed*), or drawn afresh for every frame (*per frame*), with truncation 1.0 so that the regimes differ.

**Diffusion: FAMS.** Face Anonymization Made Simple [fams] conditions a diffusion model on the source face and pastes an aligned square covering face, forehead and some hair back into the frame. We use the authors' anonymisation setting (anonymisation degree 1.25, 25 denoising steps, guidance 4.0), with the noise seed either fixed per participant or drawn afresh per frame. Diffusion runs at about 6 s per frame; the utility experiments use all 210,924 task frames. To see what a stronger setting buys, we also sweep the degree from 1.35 to 3.0 on the privacy sample (§7).

**Release protocols.** Each family yields two candidate releases (Figure 5). **Full Synthetic** takes every crop from the de-identified frame. **Hybrid** combines the de-identified face crop with the participant's original, unaltered eye crops, fused at the data-loading level. Hybrid tests whether the context (the face) can be de-identified while the signal (the eyes) remains genuine. §7 measures what each protocol protects: as shown there, retaining genuine ocular ROIs retains the identifying information they carry, so Hybrid is a utility-preserving protocol rather than a privacy-preserving one, and is not by itself sufficient for public release.

![Figure 4](FIGURE: original face-swap examples figure)

**Figure 4.** Face replacement with SimSwap++. The face in each target frame is replaced by the source template's facial identity, while the participant's pose, expression and surroundings are preserved. Rows show the effect of two source templates (Identity A and Identity B) on different participants.

![Figure 5](FIGURE: original Hybrid pipeline figure)

**Figure 5.** The two release protocols. (A) Full Synthetic: every crop is taken from the de-identified frame, replacing the facial identity, ocular region included. (B) Hybrid: the de-identified face crop is combined with the original, unaltered eye crops, which carry the gaze geometry. The streams are fused at the data-loading level, so the model learns to predict gaze from real ocular features within a synthetic facial context.

# 6 THREAT MODEL AND PRIVACY EVALUATION

**What is released.** The shareable artefact is the set of cropped streams the
models consume — the face ROI, the left and right eye ROIs, and the face grid —
not the source video. All attacks operate on exactly those crops.

**Attackers and tests.** *A1, re-identification from photographs,* is the
primary threat: the attacker holds ordinary photographs of candidate
individuals — for example, people known to belong to the recruitment pool — and
asks which of them a released crop shows. *A2, linkability,* is a property test
rather than a separate adversary: whether the released content alone lets
frames be grouped by individual, with no external reference. It operationalises
two of the three criteria by which the Article 29 Working Party judges
anonymisation, singling out and linkability [wp29anon]; data whose
records can be isolated per person remain personal data even when no name is
attached [gdpr]. A usable gaze corpus keeps each participant's frames
together, because participant-wise splits and per-user calibration require it,
so a released corpus is linkable by construction; A2 asks whether the content
would restore that grouping if the identifiers were stripped. The two tests can
disagree by up to 60 points on identical crops (Figure S2).

**Recognisers.** InsightFace `buffalo_l` (ArcFace, w600k_r50) throughout, used
without any adaptation to this dataset; cosine similarity against
per-participant templates. No component is trained on CognitiveGaze. Because a
privacy figure should not be one model's opinion, the load-bearing periocular
measurements are repeated with FaceNet (Inception-ResNet-v1, VGGFace2, triplet
loss) — a different backbone trained on a different corpus with a different
objective. Each recogniser is quoted against its own floor: FaceNet's is 3.52%,
more than twice ArcFace's 1.57%, so the two are never mixed.

**Metrics.** True-accept rate at a fixed false-accept rate of 1e-3
(TAR@FAR=1e-3) is primary for A1; closed-set Rank-1 is reported for
completeness only, because on single-session data it is inflated by session
structure. For A2 we report the adjusted Rand index (ARI) of an unsupervised
partition against the true participant labels, together with the number of
clusters the procedure selects unaided.

**The session-nuisance floor.** Each participant was recorded in a single
session, so illumination, seating and background are constant within a
participant and differ between participants. Any identification metric on such
data therefore has a non-zero baseline owing nothing to the face. We measure it
by running every attack unchanged on a 120 px face-free patch of the same
frames. The floor is **Rank-1 50.5%** (chance 5.6%), **TAR@FAR=1e-3 1.6%**, and
**ARI 0.089** with 2 clusters selected rather than 18. Closed-set Rank-1 is
therefore not evidence of de-identification on this kind of data; verification
at a strict operating point, and linkage, are.

**Alignment.** Detection and alignment are taken from the *original* frame and
applied to the de-identified one, so an operator is never credited for
defeating the detector — detector failure is not privacy, and it would also
break the gaze pipeline. Where an attack instead detects on the released image
("self-aligned") it is labelled as such, and the number of frames surviving
detection is reported beside it.

**Uncertainty.** Frames of one participant are not independent, so every
privacy interval is a 95% participant-level cluster bootstrap. Verification
intervals hold the operating threshold at its full-data value (the attacker
calibrates once) and resample participants' genuine scores (2,000 replicates);
re-estimating a 1-in-1,000 impostor quantile inside each resample biases the
interval upward, by 10 points on one face ROI. Linkage intervals resample
participants, keep a duplicated participant's label, and refit (1,000
replicates).

**Tooling.** Every attack, floor and interval above is packaged as `privaudit`,
which takes a candidate release in GazeCapture layout and maps each result onto a
release checklist. Fed this study's saved embeddings it reproduces all 32
eye-ROI verification and linkage values exactly; re-embedding the crops from
scratch, all 46 audited values agree within GPU inference noise (embeddings
differ by ~1e-4 between runs; ARI within 0.015, TAR within one probe). Utility differences are
paired within participant (n = 17).

# 7 IDENTITY LEAKAGE UNDER DE-IDENTIFICATION

Table 4 summarises the leakage of each SimSwap release under both attackers; Table 2 gives face-crop re-identification for every family, and Figure 6 linkability.

**The ocular region identifies the participant.** Before any operator is
considered: an unmodified eye ROI, 120 px, given to a pretrained recogniser
with no adaptation, verifies identity at **66.6%** (left) and **62.2%** (right)
TAR@FAR=1e-3, against a 1.6% floor. Rank-1 is 91–98%. The eye ROI is not a
de-identified crop of the face; per pixel it is the part of the face that
identifies best.

**A second recogniser agrees, in both threat models.** Every measurement above
was repeated with FaceNet. Under A1 it gives **69.1% / 81.2%** on unmodified
ocular ROIs and **0.00–1.48%** on synthesised ones, reproducing both halves of
the ArcFace result. Under A2 it gives **67.1% / 80.5%** unmodified against its
own 3.52% floor (19–23× floor, where ArcFace gives 40×), and 42.8% / 55.8%
(Identity A) and 34.4% / 51.0% (Identity B) on synthesised ROIs, again far
above floor. The agreement is evidence about the imagery rather than about one
model's inductive bias.

**This is what the Hybrid protocol releases.** Hybrid recovers gaze accuracy
by retaining the original, unaltered eye ROIs — the crops measured above. On
the identity axis it therefore inherits the leakage of unprocessed data:
**62–67% TAR@FAR=1e-3, roughly forty times the floor**, before the synthetic
face is considered at all. The mechanism that recovers the utility is the
mechanism that leaks the identity.

**Full synthesis removes the correspondence for the eye crops, not for the face
crop.** With the ocular region synthesised as well, the same attack on the eye
ROIs falls to **0.2% / 1.8%** TAR@FAR=1e-3, at or below the floor (FaceNet:
0.00% / 1.48%); the two templates agree, and so do the two recognisers. The
synthetic face crop does not fall that far. Matched against the participant's
own photographs it still verifies at **18.4%** [5.1, 35.4] for Identity A and
**22.0%** [6.8, 40.7] for Identity B under ArcFace (Table 2), against 100% for
unprocessed faces and a 1.6% floor.

The face-ROI intervals are wide because the sampling unit is 18 participants.
For Identity A the direction is secure: the lower bound is three times the
floor, and FaceNet agrees (14.4% against its own 3.52% floor). For Identity B
FaceNet does not confirm it (0.2%), so the residual depends on the template and
the recogniser. These data support "above floor, an order of magnitude below
unprotected imagery" for the face crop, and not a finer grading than that. The
release decisions in the Data Availability statement rest on the direction, not
on the point estimate.

**But the release stays linkable.** Under A2, an unsupervised partition of the
released corpus recovers the participants exactly: **ARI 1.000 at the correct
k = 18**, for both templates, against a floor of 0.089 at k = 2. Identity
replacement is per-participant consistent — each participant is mapped to the
same synthetic appearance every time — which destroys the correspondence to
their real face while leaving the released set internally self-consistent. The
same eye ROIs that score 0.2% against A1 score **42.8% / 62.5%** against A2.

**The linkage is not an artefact of surrounding context.** A face ROI retains
channels the synthesis never edits — hair at the crop border, spectacles, skin
tone, head pose, face outline — and the floor does not control for them,
because it is a patch of the *room* whereas those are attributes of the
*person*. We therefore repeat the linkage attack on the 120 px eye ROI, which
excludes hair, ears, neck, clothing and the face outline, and whose pixels are
themselves synthetic in the Full Synthetic arm. Linkage survives: **ARI
0.68–0.79** at k = 18 for the two synthetic identities, against **0.84–0.85**
for unmodified eye ROIs and a floor of 0.089, with 79–86% of frames assigned to
the correct participant. The subject-level 95% intervals of the synthetic and
unmodified arms overlap substantially (e.g. [0.59, 0.86] against [0.70, 0.96]
on the left eye), so replacement cannot be said to reduce eye-ROI linkage at
all; clustering variability across twenty KMeans initialisations is small
(SD 0.015–0.037), so single-fit point estimates are representative.

**Nor is it an illumination artefact.** Single-session recording leaves the
imaging conditions — exposure, white balance, focus distance, the angle of the
light on the cornea — constant within a participant, and the face-free floor
does not control for them because they are properties of the imaging of the
*person* rather than of the room. We therefore attack that channel directly.
Removing per-crop brightness and contrast entirely leaves the linkage intact
(ARI 0.684 → 0.742 on the swapped left eye; verification 62.2% → 73.0% on the
right), local histogram equalisation costs 0.05–0.14 without collapsing it, and
a large within-session illumination change — gallery from the dark-screen
stimulus blocks, probe from the natural-image blocks — leaves it intact and
raises it to 0.918 / 0.931 on unmodified crops. The recogniser is therefore not
relying on photometry.

**Photometry is nonetheless a second, independent linkage channel.** Twenty-one
summary statistics per crop — per-channel means, percentiles, white-balance
ratios, dynamic range, no structure at all — recover the participants at **ARI
0.87–0.90** with no face model whatsoever, and identity replacement does not
touch this channel: 0.898 / 0.901 on swapped crops against 0.870 / 0.902 on
originals. A release that does not normalise crop photometry is linkable
without any recogniser at all.

**A second family, and its strongest mode.** DeepPrivacy2 removes identity by
inpainting rather than replacing it. Its face mode repaints the detected face
box and touches 40% more of the crop than SimSwap; its full-body mode — the
main contribution of the DeepPrivacy2 paper, motivated by exactly our concern that face
anonymisation leaves identifiers outside the face [deepprivacy2] —
repaints the whole person, face included. Crossing edit domain with identity
regime gives the two-part condition as a factorial result, on the face crop,
for both release-only attacks (linkage: ARI at k = 18, floor 0.089;
verification: TAR at FAR 1e-3 with 95% subject-level interval, floor 1.6%; Figure 6a). Face-crop linkage is 1.000 with face-mode editing and a fixed identity, 0.982 when either the edit domain is widened to the whole person or the identity is resampled every frame, and 0.297 only when both are; verification falls from 89.7–100% in the first three cells to 35.7% [27, 45] in the fourth.

Widening the edit domain alone changes nothing, and neither does resampling the
identity alone; only both together break the partition, and then an attacker
who must discover the cohort size finds none (silhouette selects k = 2,
ARI 0.000; pair AUC 0.78). The full-body generator works at 288×160, so its
faces are visibly low-resolution — but the fixed-identity full-body arm carries
the same degradation and still links at 0.982, so the collapse is due to the
identity regime, not to image quality. Both residuals are well above floor:
what the whole-person generator leaves untouched — background, pose, crop
geometry, illumination — still carries part of the partition, and enough
identity for an attacker with templates to verify a third of the frames. The
condition blocks *discovery* of who is who; it does not stop *verification* by
someone who already holds a few labelled frames. Against A1 every
DeepPrivacy2 arm is at or below floor, on the eye crops (≤ 1.2%) and on the face
crop (≤ 1.6%; Table 2), consistent with DeepPrivacy2's
own re-identification evaluation, which, like A1, matches enrolment images to
the release and does not test release-only linkage.

**A diffusion method meets the condition's first half by itself, and fails on
the second.** Face Anonymization Made Simple [fams] conditions a diffusion
model on the source face and pastes an FFHQ-aligned square — face, forehead,
some hair — back into the frame. Holding its noise seed fixed per participant
does not hold the synthetic person fixed: the output identity follows the source
frame, and the fixed-seed and fresh-seed regimes are indistinguishable on every
measure. It therefore breaks per-participant consistency by construction, and
eye-ROI linkage roughly halves — verification 5–8% against 43–63% for SimSwap,
ARI 0.29–0.33 [lower bounds 0.19–0.26] against 0.68–0.80 — though every interval
stays above the eye-native floor (0.074). The face crop, whose border, head pose
and geometry the aligned square leaves alone, stays linkable at ARI 0.85–0.94.
The two-part condition was derived from SimSwap and DeepPrivacy2; this is its
prediction holding on a family it was not derived from.

**The diffusion face still resembles the participant.** Matched against the
participant's own photographs, the FAMS face crop verifies at **24.1%**
[15.3, 33.6] under ArcFace — no lower than SimSwap — although FaceNet places it
near its own floor (5.9% [2.2, 10.5] against 3.52%; Table 2). FAMS conditions
on the source face to keep pose and expression, and part of the identity comes
with them. Closed-set Rank-1 is high for both families (57–61%) but is not
evidence here: on single-session data its floor is 50.5%.

**Table 2.** Face-crop A1 for every family: released face crops (second half of
each recording) matched against the participant's original face crops (first
half). TAR at FAR = 1e-3 with 95% participant-level intervals, each recogniser
against its own floor (ArcFace 1.6%, FaceNet 3.52%).

| Release | ArcFace TAR | FaceNet TAR |
|---|---|---|
| SimSwap, Identity A | 18.4% [5.1, 35.4] | 14.4% [0.7, 30.5] |
| SimSwap, Identity B | 22.0% [6.8, 40.7] | 0.2% [0.0, 0.6] |
| DeepPrivacy2 face, fixed | 1.6% [0.0, 4.6] | 0.7% [0.0, 1.6] |
| DeepPrivacy2 face, per frame | 1.4% [0.5, 2.5] | 2.8% [0.8, 5.5] |
| DeepPrivacy2 full body, fixed | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| DeepPrivacy2 full body, per frame | 0.3% [0.0, 0.7] | 0.6% [0.0, 1.4] |
| FAMS, fixed seed | 22.4% [14.3, 31.9] | 3.2% [1.6, 5.1] |
| FAMS, per frame | 24.1% [15.3, 33.6] | 5.9% [2.2, 10.5] |

**Strength trades pupil signal for identity, not for linkability.** The FAMS
anonymisation degree extrapolates the generator away from the source identity.
Raising it from the authors' 1.25 to 1.5 — the strongest setting that still
yields a face in more than 90% of frames; beyond it the generator degrades
rather than anonymises, with 19–94% of crops containing no detectable face —
lowers ArcFace face verification from 24.1% to 10.1% [3.0, 19.3], still above
the floor (FaceNet reaches its floor from 1.35). It leaves face-crop linkage
where it was (ARI 0.81–0.90) and costs pupil signal: the within-participant
correlation between the synthetic pupil and true horizontal gaze falls from
0.68 to 0.53 (Table 3; SimSwap 0.79, unprocessed 0.87). Each step that lowers
re-identification removes gaze information; none removes linkability.

**Table 3.** FAMS anonymisation-strength sweep on the privacy sample (per-frame
noise). Settings producing more than 10% non-face outputs are shown but not used
for privacy conclusions. Pupil r: within-participant correlation of the synthetic
pupil's horizontal position with EyeLink gaze (§8.1).

| Degree | Non-face outputs | Face TAR, ArcFace | Face TAR, FaceNet | Face linkage ARI | Eye linkage ARI | Pupil r |
|---|---|---|---|---|---|---|
| 1.25 (authors) | 0.0% | 24.1% [15.3, 33.6] | 5.9% | 0.853 | 0.292 | 0.68 |
| 1.35 | 0.1% | 14.1% [5.9, 23.5] | 2.6% | 0.810 | 0.305 | 0.62 |
| 1.5 | 6.5% | 10.1% [3.0, 19.3] | 2.2% | 0.899 | 0.283 | 0.53 |
| 1.6 | 18.7% | 5.9% [0.9, 12.0] | 1.7% | 0.792 | 0.302 | 0.41 |
| 1.75 | 36.0% | 3.7% [0.3, 9.4] | 1.9% | 0.784 | 0.259 | 0.23 |
| 2.0 | 93.9% | — | — | — | — | — |

The residual channels the recogniser does exploit — skin tone, periocular
geometry, crop scale — are signals gaze estimation reads, which is why the axis
trades at all.

These eye-ROI figures assume an attacker who knows the cohort size: on the eye
ROI the cluster count is not recovered unaided, whereas on the face ROI it is
(k = 18 selected by silhouette). For a public release the cohort size is stated
in the documentation, so the assumption is mild, but it is an assumption and
the face ROI does not need it. Breaking unsupervised *discoverability* is in
fact the one place replacement helps: on the right eye, k selected falls from
26 (ARI 0.674) to 2 (ARI 0.011). It breaks discovery of the partition, not its
separability.

**The floor is eye-crop native.** Running the periocular pipeline itself on a
120 px face-free patch of the same frames — identical frame selection,
identical downstream steps, only the pixels differ — gives **TAR@FAR=1e-3 =
1.57%**, Rank-1 50.6%, d′ 0.458, agreeing with the face-crop floor to three
significant figures; its linkage floor is **ARI 0.074**, against 0.089 for the
face pipeline. Eye-ROI linkage is quoted against the former.

**The open tier leaks too, by two orders of magnitude less.** Our release plan
puts per-frame gaze estimates and oculomotor measures in an open tier because
it contains no imagery and no face embeddings. Since eye movement is itself a
biometric, we tested that rather than asserting it. Treating the released gaze
series as the attacker's only input, we cut each recording into 5 s fragments
(150 frames), described each with fifteen dispersion, velocity and
saccade/fixation statistics, and clustered. The result is **ARI 0.033 at the
correct k, purity 0.157 against a chance purity of 0.059**, versus a
label-permutation null of −0.0001 (max 0.0015 over 20 permutations). The signal
is real but small — two orders of magnitude below the pixel tiers (0.68–1.000).
We report the open tier as low, not zero, residual risk.

**Metadata leaks before a pixel is read.** Released crop dimensions are
constant per participant. Face ROI size alone yields an expected Rank-1 of
50.0%; face size, eye size and inter-ocular distance together yield **94.4%**
(4.06 of 4.17 bits) without reading any pixels. A release must normalise crop
geometry; ours does, and all figures above are pixel-based.

**An adaptive attacker gains little.** An attacker who knows the operator and
holds paired (original, synthesised) imagery of *other* participants can fit a
de-anonymisation map and apply it to a participant they could not train on.
Across mean-shift, orthogonal Procrustes and ridge maps, fitted
leave-one-participant-out, the best gain is **+3.3 points** on the face ROI
(19.9% → 23.2%) and nothing on the eye ROIs. The same maps reach Rank-1 100%
when fitted *including* the target — they memorise identities rather than
invert the operator. The protection is a generalisation barrier, not an
information-theoretic one.

**Table 4.** Identity leakage across release strategies. A1: the attacker holds enrolment photographs; A2: the attacker holds only the release. TAR at FAR = 1e-3 (ArcFace) and ARI at k = 18, with 95% participant-level bootstrap intervals.

| Released stream | Real Data | Full Synthetic (A) | Full Synthetic (B) | Hybrid (Real Eyes) |
|---|---|---|---|---|
| Face ROI, TAR (A1) | 100% | 18.4% [5.1, 35.4] | 22.0% [6.8, 40.7] | 18.4% (face is synthetic) |
| Left eye ROI, TAR (A1) | 66.6% [53.1, 78.7] | 0.2% [0.0, 0.5] | 0.6% [0.0, 1.6] | **66.6%** (unaltered) |
| Right eye ROI, TAR (A1) | 62.2% [49.7, 73.3] | 1.8% [0.1, 4.4] | 1.1% [0.0, 2.8] | **62.2%** (unaltered) |
| *Session-nuisance floor* | *1.6%* | *1.6%* | *1.6%* | *1.6%* |
| Face ROI, ARI @ k=18 (A2) | 1.000 | 1.000 | 1.000 | 1.000 |
| Left eye ROI, ARI @ k=18 (A2) | 0.844 [0.70, 0.96] | 0.684 [0.59, 0.86] | 0.737 [0.60, 0.88] | 0.844 (unaltered) |
| Right eye ROI, ARI @ k=18 (A2) | 0.853 [0.73, 0.96] | 0.789 [0.64, 0.92] | 0.761 [0.62, 0.95] | 0.853 (unaltered) |
| *Linkage floor* | *0.089* | *0.089* | *0.089* | *0.089* |

![Figure 6](figs_core/fig_core1_privacy.pdf)

**Figure 6.** **Release-only linkage.** **(a)** Face crop, DeepPrivacy2, crossing edit domain (face box vs whole person) with identity regime (fixed per participant vs fresh per frame): linkage (ARI at k = 18) and verification (TAR at FAR 1e-3, 95% subject-level interval). Only the cell that meets both halves of the condition falls (ARI 0.297), and verification there is still 36%. Unprocessed and SimSwap face crops are at 1.000; FAMS at 0.939 / 0.853. **(b)** Eye-ROI linkage for every arm, with 95% subject-level bootstrap intervals; dashed: the eye-native floor (0.074). SimSwap and DeepPrivacy2 with a fixed identity per participant overlap unprocessed eyes. A fresh DeepPrivacy2 identity per frame brings the eye ROI close to the floor (0.12–0.17), and FAMS sits between (0.29–0.33) under either seed regime, because a fixed seed does not fix its output identity. Eye-ROI linkage assumes the attacker knows the cohort size (k = 18).

# 8 GAZE UTILITY UNDER DE-IDENTIFICATION

## 8.1 Utility protocol

**Three conditions.** A utility number answers a question only once it says who
trained the model on what. We report three. In **condition (ii)**, models
trained on real data are deployed on de-identified crops: the cost to a
pipeline that receives de-identified input. In **condition (i)**, models are
retrained on the de-identified release, with the same recipe, participant
partition and hyperparameters as the real-data models, and tested either on
**real faces** — a third party training on the released corpus and deploying
the model on its own users — or on **de-identified faces** — benchmarking on
the release itself. Condition (ii) is the only one that can separate
information loss from domain shift when read alongside (i): a penalty that
retraining removes was shift; one that survives retraining was information.

Two references bound every number. The **initialisation noise floor** is the
spread of the real-data model over five independent initialisations,
0.04–0.20 cm. The **no-information reference** is the error of predicting the
training participants' mean gaze point for every frame: **9.37 cm**, a penalty
of +3.70 to +4.41 cm depending on the backbone's own baseline. A model at that
level has learned no gaze.

**A model-free check on the pupil.** End-to-end error mixes the information a
release carries with how well a model trained on it transfers. We therefore
also measure the pupil directly. FaceMesh [facemesh] locates both iris
centres on the released face crop and on the original crop of the same frame,
each expressed in its eye's own corner-to-corner frame, which removes head pose
and crop placement. We report the within-participant correlation of the
released pupil position with the original one and with the EyeLink gaze
coordinate, on 4,680 frames identical across all releases.

## 8.2 Models trained on real data (condition (ii))

Table 5 reports SimSwap under condition (ii); Table 6 reports DeepPrivacy2 and FAMS under the same models, initialisations and harness. Before any DeepPrivacy2 or FAMS number was computed, the real-data arm was re-run through the same pipeline and reproduced Table 5 to within 0.0012 cm on all 100 folds.

**Table 5.** Gaze error (cm) under condition (ii) for SimSwap: trained on real data, deployed on de-identified crops. Mean over five folds, then mean ± SD over five independent initialisations; in parentheses, the change against the same model's real-data baseline. The SD is the noise floor of the comparison (0.04–0.20 cm on the baseline): no difference smaller than it can be supported in either direction.

| Model | Real Data | Full Synthetic (A) | Hybrid (A) | Full Synthetic (B) | Hybrid (B) |
|---|---|---|---|---|---|
| iTracker | 5.85 ± 0.14 | 8.85 ± 0.17 (+3.00) | **6.16 ± 0.20 (+0.31)** | 9.48 ± 0.21 (+3.62) | **5.82 ± 0.20 (−0.03)** |
| MobileNet-V3 Large | 4.98 ± 0.13 | 8.65 ± 0.18 (+3.66) | **5.17 ± 0.16 (+0.18)** | 12.03 ± 0.53 (+7.04) | **6.03 ± 0.25 (+1.04)** |
| AFFNet | 5.67 ± 0.20 | 8.07 ± 0.25 (+2.41) | **5.85 ± 0.24 (+0.19)** | 9.64 ± 0.53 (+3.98) | **5.74 ± 0.25 (+0.08)** |
| MGazeNet | 4.96 ± 0.04 | 7.74 ± 0.20 (+2.78) | **5.21 ± 0.21 (+0.24)** | 9.72 ± 0.40 (+4.75) | **5.45 ± 0.25 (+0.48)** |

**Table 6.** Condition (ii) for DeepPrivacy2 (face and full-body modes, fixed and per-frame identity) and FAMS (fresh noise per frame, all 210,924 task frames). Mean ± SD over five initialisations; in parentheses, or as listed for FAMS, the change against the real-data baseline in Table 5.

*(a) DeepPrivacy2, face mode*

| Model | Full Synthetic (fixed) | Hybrid (fixed) | Full Synthetic (per frame) | Hybrid (per frame) |
|---|---|---|---|---|
| iTracker | 16.44 ± 0.24 (+10.58) | **5.75 ± 0.11 (−0.10)** | 17.50 ± 0.25 (+11.65) | **5.83 ± 0.06 (−0.02)** |
| MobileNet-V3 Large | 15.88 ± 0.37 (+10.90) | **6.02 ± 0.54 (+1.03)** | 16.24 ± 0.23 (+11.25) | **5.95 ± 0.45 (+0.97)** |
| AFFNet | 19.33 ± 0.26 (+13.66) | **5.74 ± 0.24 (+0.07)** | 20.52 ± 0.22 (+14.85) | **5.68 ± 0.22 (+0.01)** |
| MGazeNet | 17.06 ± 0.39 (+12.09) | **5.43 ± 0.38 (+0.47)** | 17.04 ± 0.53 (+12.08) | **5.37 ± 0.32 (+0.40)** |

*(b) DeepPrivacy2, full-body mode*

| Model | Full Synthetic (fixed) | Hybrid (fixed) | Full Synthetic (per frame) | Hybrid (per frame) |
|---|---|---|---|---|
| iTracker | 15.92 ± 0.29 (+10.06) | **6.01 ± 0.11 (+0.16)** | 17.12 ± 0.35 (+11.27) | **5.96 ± 0.10 (+0.11)** |
| MobileNet-V3 Large | 14.63 ± 0.81 (+9.65) | **5.60 ± 0.48 (+0.61)** | 14.65 ± 0.76 (+9.67) | **5.61 ± 0.45 (+0.63)** |
| AFFNet | 17.72 ± 0.42 (+12.05) | **5.78 ± 0.23 (+0.11)** | 17.82 ± 0.47 (+12.16) | **5.80 ± 0.23 (+0.13)** |
| MGazeNet | 15.28 ± 0.29 (+10.31) | **5.22 ± 0.20 (+0.26)** | 15.59 ± 0.19 (+10.63) | **5.35 ± 0.20 (+0.39)** |

*(c) FAMS*

| Model | Real Data | Full Synthetic | Hybrid |
|---|---|---|---|
| iTracker | 5.85 ± 0.14 | +5.06 ± 0.17 | **+0.02 ± 0.07** |
| MobileNet-V3 Large | 4.98 ± 0.13 | +4.06 ± 0.24 | **+0.27 ± 0.20** |
| AFFNet | 5.67 ± 0.20 | +4.91 ± 0.20 | **+0.19 ± 0.10** |
| MGazeNet | 4.96 ± 0.04 | +4.94 ± 0.17 | **+0.35 ± 0.21** |

Deployed on DeepPrivacy2 crops, real-data models are worse than predicting the mean: 15.9–20.5 cm in face mode and 14.6–17.8 cm in full-body mode, against 9.37 cm. A model given repainted eyes is confidently wrong. Hybrid recovers 91–101% of the face-mode penalty and 93–99% of the full-body one, as it does for SimSwap. Every one of the 421,848 full-body frames had the participant detected and repainted. In 2,861 of them (1.4%) the detector also returned a second person region without a surface map, which the pipeline leaves unedited; the participant's released crops in those frames are changed as much as in all others, so no original pixels reach the release. Under deployment alone FAMS looks worse than SimSwap (+4.1 to +5.1 cm, against +2.4 to +3.7 cm for SimSwap A), and Hybrid recovers 93–100% of its penalty. Section 8.3 shows that most of the DeepPrivacy2 and FAMS penalty under condition (ii) is domain shift.

**Table 7.** Hybrid minus Full Synthetic (cm), paired within initialisation, mean ± SD over five draws; negative means the genuine ocular crops recover accuracy. All 40 draw-level values are negative, and the across-draw SD is an order of magnitude below the effect in every cell.

| Model | Identity A | Identity B | Penalty recovered |
|---|---|---|---|
| iTracker | −2.693 ± 0.097 | −3.656 ± 0.320 | 90% / 101% |
| MobileNet-V3 Large | −3.481 ± 0.274 | −5.999 ± 0.655 | 95% / 85% |
| AFFNet | −2.221 ± 0.236 | −3.897 ± 0.428 | 92% / 98% |
| MGazeNet | −2.535 ± 0.119 | −4.273 ± 0.389 | 91% / 90% |
| *Face-only control* | *+0.000 ± 0.000* | *+0.000 ± 0.001* | *0%* |

Restoring the genuine ocular ROIs recovers most of the synthesis penalty in
every architecture tested. Full synthesis costs **+2.41 to +7.04 cm** depending
on model and template; Hybrid recovers **85–101%** of that, leaving a residual
of **+0.08 to +1.04 cm**. Across five independent initialisations the paired
difference is negative in all four architectures, both templates and every
draw, with a between-draw spread an order of magnitude smaller than the effect.

The residual is small but mostly real: tested against each model's own
baseline, six of the eight Hybrid arms are detectably worse than real data and
two are indistinguishable from it. We therefore describe Hybrid as *recovering*
the synthesis penalty rather than *preserving* accuracy — a distinction the
measured initialisation noise floor (0.04–0.20 cm) is fine enough to make.

The face-only control consumes the face ROI only and therefore cannot see the
eye crops; its two arms must be identical, and they are, to three decimals. It
establishes that the comparison distinguishes the arms when a model uses the
ocular stream and reports exactly zero when it does not, which is what licenses
reading every other row as a real effect.

We find no evidence of an architecture that fails to exploit the restored
ocular signal. In particular the two guidance-factor architectures, whose eye
features are modulated by parameters regressed from the face, do not behave
differently from the late-fusion ones: AFFNet recovers 92% and 98% of the
penalty. What distinguishes the Hybrid protocol is not fusion topology but
whether the model reads the eye crops at all, as the face-only control makes
explicit.

Changing the source template from Identity A to Identity B changes the Full Synthetic error by +0.63 to +3.38 cm, but the Hybrid error by only −0.34 to +0.86 cm, and Hybrid recovers most of the penalty under both templates (Table 7). The synthesis penalty depends strongly on the template; the Hybrid error changes much less.

## 8.3 Models retrained on the release (condition (i))

**Table 8.** Penalty (cm) against each model's own real-data baseline, under the three conditions: mean over MobileNet-V3 Large, AFFNet and MGazeNet, with [min, max] across the three. Condition (i) is the mean over five independent initialisations per cell (per-draw SDs 0.01–0.83 cm), against the five-draw real-data mean; condition (ii) is recomputed over the same three architectures so the columns are comparable. Retraining uses the same recipe and participant partition as the real-data models. iTracker (about 9 h per fold) is reported separately in Table 12. Negative Hybrid values read as indistinguishable from baseline, not as gains.

| Arm | (ii) train real, test de-id | (i) train de-id, test real | (i) train de-id, test de-id |
|---|---|---|---|
| SimSwap A, Full Synthetic | +2.95 [+2.41, +3.66] | +3.83 [+2.47, +4.56] | +2.76 [+2.50, +3.06] |
| SimSwap A, Hybrid | **+0.20** [+0.18, +0.24] | **+0.09** [+0.01, +0.23] | **−0.04** [−0.14, +0.04] |
| SimSwap B, Full Synthetic | +5.26 [+3.98, +7.04] | +4.81 [+3.16, +6.21] | +2.54 [+2.17, +3.02] |
| SimSwap B, Hybrid | **+0.53** [+0.08, +1.04] | **+0.04** [−0.10, +0.14] | **−0.08** [−0.19, +0.04] |
| DeepPrivacy2 fixed, Full Synthetic | +12.22 [+10.90, +13.66] | +4.51 [+3.94, +4.81] | +4.50 [+3.93, +4.81] |
| DeepPrivacy2 fixed, Hybrid | **+0.52** [+0.07, +1.03] | **−0.08** [−0.14, −0.02] | **−0.12** [−0.17, −0.09] |
| DeepPrivacy2 per frame, Full Synthetic | +12.73 [+11.25, +14.85] | +4.40 [+3.79, +4.77] | +4.34 [+3.79, +4.68] |
| DeepPrivacy2 per frame, Hybrid | **+0.46** [+0.01, +0.97] | **+0.02** [−0.09, +0.10] | **+0.02** [−0.10, +0.08] |
| DeepPrivacy2 full body fixed, Full Synthetic | +10.67 [+9.65, +12.05] | +4.72 [+4.03, +5.21] | +4.70 [+4.25, +5.02] |
| DeepPrivacy2 full body fixed, Hybrid | **+0.33** [+0.11, +0.61] | **+0.01** [−0.08, +0.15] | **−0.02** [−0.14, +0.10] |
| DeepPrivacy2 full body per frame, Full Synthetic | +10.82 [+9.67, +12.16] | +4.34 [+3.91, +4.58] | +4.34 [+3.91, +4.59] |
| DeepPrivacy2 full body per frame, Hybrid | **+0.38** [+0.13, +0.63] | **−0.05** [−0.15, +0.05] | **−0.05** [−0.14, +0.05] |
| FAMS (diffusion), Full Synthetic | +4.64 [+4.06, +4.94] | +0.94 [+0.39, +1.29] | +2.15 [+1.56, +2.57] |
| FAMS (diffusion), Hybrid | **+0.27** [+0.19, +0.35] | **−0.01** [−0.13, +0.07] | **−0.02** [−0.14, +0.06] |
| *No-information reference (predict the mean)* | *+4.17 [+3.70, +4.41]* | | |

FAMS was retrained on every task frame with the same five initialisations as every other arm. On a 10% frame subset, against a real-data model retrained on the same subset, the FAMS penalty on real faces was +0.32 cm, against +3.26 cm for SimSwap A and +3.46 cm for DeepPrivacy2 retrained on exactly those frames. The subset understated the FAMS penalty (+0.94 cm on all frames) but not its ranking: FAMS is better than SimSwap A and DeepPrivacy2 in all 15 paired draws (architecture × initialisation), on the subset and again on all frames.

**Retraining does not rescue full synthesis.** For SimSwap, the penalty
survives retraining: +2.2 to +3.1 cm when the retrained model is tested on the
release, and +2.5 to +6.2 cm when it is deployed on real faces — where two of
the three architectures end up at or beyond the no-information reference. The
synthesis penalty in Table 5 is therefore mostly lost information, not domain
shift: a model cannot learn from synthetic eyes what synthetic eyes do not
carry. A Full Synthetic release is of little use to a third party who wants a
gaze model for real users.

**Removal by inpainting leaves no gaze signal at all.** For DeepPrivacy2 the
picture is the reverse. More than half of the condition (ii) penalty *is*
domain shift — retraining removes 56–65% of it — but what remains is the
no-information level, in face mode and in full-body mode alike. Across five
initialisations, both modes, both identity regimes and both test sets, 580 of
600 fold-level errors are at or above predicting the mean, and none is more
than 0.17 cm below it; 232 of the 300 training runs reach their best validation
error at the first epoch and learn nothing thereafter, and per-participant
calibration does not help. The repainted eyes
carry no recoverable gaze information. This replaces the qualitative
observation that the generator "alters gaze direction" with a measurement.

**Hybrid recovers the penalty in every condition.** Restoring the genuine
ocular ROIs recovers 91–103% of the penalty when the retrained model is
deployed on real faces and 98–106% on the release, for both families, both
DeepPrivacy2 modes and both identity regimes; every Hybrid cell in condition
(i) lies between −0.19 and +0.23 cm of baseline, inside the initialisation noise
floor or at its edge. FAMS falls in the same band (−0.14 to +0.07 cm); its
percentages (82–111% and 96–106%) are noisier because its Full Synthetic
penalty on real faces is small, +0.39 cm for AFFNet. The central utility finding
of Table 7 therefore does not depend on how the model was trained.

**A diffusion anonymiser transfers better; it does not keep more gaze.** FAMS
looks, under condition (ii), like the others: +4.1 to +5.1 cm. Retrained on its
release and deployed on real faces, models lose only +0.94 cm on average —
+1.15 ± 0.03 (MobileNet-V3), +0.39 ± 0.21 (AFFNet), +1.29 ± 0.06 (MGazeNet) —
against +3.83 for SimSwap A and +4.51 for DeepPrivacy2, and FAMS is better than
both in every one of 15 paired draws. Two measurements show that this is not
because its eyes carry more gaze. The model-free pupil test (Table 9) finds
the opposite: SimSwap's synthetic pupils follow the participant's true
horizontal gaze more closely than FAMS's in all 17 participants with ground
truth (r = 0.79–0.83 against 0.68–0.69; unprocessed 0.87). And models that see
only the eye crops (Table 10) reproduce the advantage on real faces — +0.96 to
+1.16 cm for FAMS against +2.71 to +6.77 for SimSwap — but not inside the
release, where the two are comparable (+2.04 to +2.72 against +1.98 to +2.95,
each method better for one eyes-only architecture in every draw). The difference
is transfer: models trained on SimSwap eyes fail on real eyes, models trained on
FAMS eyes do not. A plausible reading, which we did not test, is diversity:
SimSwap maps every participant onto one template's eyes, whereas FAMS draws a
new identity every frame. Tested on its own release, a FAMS-trained model is
less accurate than on real faces (+1.56 to +2.57 cm), the reverse of SimSwap.
FAMS is therefore the more useful release for training models deployed on real
users, for reasons that have nothing to do with keeping more of the pupil; on
the privacy axis it breaks per-participant consistency without being asked to,
yet its face crop still verifies against the participant's photographs and
stays linkable (§7).

**Table 9.** Pupil-position preservation, model-free (§8.1): within-participant
correlation of the released pupil position with the original one (horizontal,
vertical) and with EyeLink gaze, mean over participants; 4,680 identical frames.

| Release | r with original, horizontal | r with original, vertical | r with gaze x | r with gaze y |
|---|---|---|---|---|
| Unprocessed | 1 | 1 | 0.87 | 0.56 |
| SimSwap, Identity A | 0.70 | 0.44 | 0.79 | 0.29 |
| SimSwap, Identity B | 0.70 | 0.44 | 0.83 | 0.28 |
| FAMS | 0.61 | 0.35 | 0.69 | 0.21 |
| DeepPrivacy2 (face and full body) | ≈ 0 | ≈ 0 | 0.06 | 0.05–0.07 |

**Table 10.** Stream isolation: eyes-only and face-only models retrained on each
release, five initialisations; penalty (cm, mean ± SD) against the same model
trained on real data, paired within initialisation.

| Model (real-data error) | Tested on real faces: SimSwap A / B / FAMS | Tested on the release: SimSwap A / B / FAMS |
|---|---|---|
| Eyes-only MobileNet-V3 (4.95) | +2.71 ± 0.93 / +3.56 ± 0.39 / **+1.16 ± 0.10** | **+2.44 ± 0.36 / +1.98 ± 0.22** / +2.72 ± 0.21 |
| Eyes-only MGazeNet (5.14) | +4.79 ± 0.94 / +6.77 ± 0.56 / **+0.96 ± 0.27** | +2.83 ± 0.38 / +2.95 ± 0.12 / **+2.04 ± 0.17** |
| Face-only MobileViT (5.29) | +4.51 ± 0.40 / +3.87 ± 0.59 / **+1.44 ± 0.26** | +3.67 ± 0.16 / +2.98 ± 0.33 / **+2.24 ± 0.29** |

**Releasing eye geometry instead of pixels does not escape the trade-off.** The
obvious alternative to de-identified imagery is to release no imagery: FaceMesh
landmarks, from which gaze could in principle be regressed. We tested three
tiers, each a superset of the last, on 1,200 frames per participant spread over
the recording, with the same participant split and the same attacks as the
pixel releases (verification TAR at FAR 1e-3; linkage ARI at k = 18; floors 1.6%
and 0.074):

**Table 11.** Releasing FaceMesh eye geometry instead of pixels: gaze error of gradient-boosted regressors (no per-participant calibration) and release-only privacy of the released features, with 95% participant-level intervals.

| Released per frame | Dims | Gaze error, uncalibrated | Verification | Linkage, frame | Linkage, 10-frame window |
|---|---|---|---|---|---|
| Iris position in each eye's corner frame | 4 | 9.02 cm | 10.2% [5.7, 15.2] | 0.267 [0.22, 0.48] | 0.441 [0.35, 0.68] |
| + face-box position and size | 7 | 9.27 cm | 62.2% [50.9, 72.2] | 0.665 [0.54, 0.84] | 0.843 [0.71, 0.98] |
| + both eyelid contours | 71 | 8.31 cm | 74.3% [64.0, 82.4] | 0.761 [0.63, 0.89] | 0.914 [0.79, 0.99] |

Predicting the mean scores 9.53 cm on these frames; the pixel models score
5.0–5.9 cm. Four iris coordinates per frame carry almost no gaze signal across
participants at smartphone resolution — the iris moves a dozen pixels across the
whole screen — yet still link the release well above floor, more so once an
attacker pools neighbouring frames. Head position and eyelid shape are what make
the geometry useful, and they are constant within a session and biometric
respectively, so they make it linkable. Per-participant calibration lifts an
iris mapping to about 5.2 cm on these data, but that requires every
end user to enrol, which a released training corpus cannot assume.

**The identity regime costs nothing; the edit domain costs everything.** Once
models are trained on the release, resampling the synthetic identity every
frame changes utility by −0.63 to +0.19 cm against a fixed identity, across
both DeepPrivacy2 modes, both arms and both test sets — never worse by more
than 0.19 cm. In condition (ii) the Full
Synthetic arm moves by up to 1.2 cm, but there both regimes are already worse
than predicting the mean. What destroys utility is repainting the eyes — and
repainting everything the crop contains, eyes included, is what §7 found
necessary to break face-crop linkage.

Figure 7 places every protocol on one plane, utility against release-only linkage; Figure S1 shows the per-architecture penalties under all three conditions.

![Figure 7](figs_core/fig_core3_tradeoff.pdf)

**Figure 7.** **Privacy against utility for every protocol tested.** Horizontal: gaze-error penalty of a model retrained on the release and deployed on real faces (mean of three architectures). Vertical: release-only linkage, the ARI at k = 18 of the most linkable crop the release contains (face or eye ROI). Circles: Full Synthetic; squares: Hybrid (original eyes); star: unprocessed data. Dashed red: no-information reference (predicting the mean gaze point); dashed grey: linkage floor. The shaded corner — private and useful — contains no protocol. Hybrid sits at the utility of real data and the linkage of real data; SimSwap and DeepPrivacy2 Full Synthetic lose most of the gaze signal and none of the linkage. The single point that leaves the top row (DeepPrivacy2 full body with a fresh identity per frame, ARI 0.297) is one on which a retrained gaze model learns nothing (+4.34 cm, at the no-information reference). Restoring the original eyes to it (square 13) recovers the utility and, with it, the linkage (0.849). The diffusion anonymiser (circle 14) is the one full-synthesis release near the utility of real data (+0.94 cm, against +3.83 for SimSwap A and +4.51 for DeepPrivacy2) — because models trained on it transfer to real eyes, not because its pupils carry more gaze (§8.3) — and its face crop keeps it in the linkable region (0.853). Diamonds: releasing FaceMesh eye geometry instead of pixels — the least linkable tier (four iris coordinates, ARI 0.267) is also near the no-information level.

**Table 12.** iTracker under the three conditions, one initialisation in condition (i). Penalty (cm) against its five-draw real-data baseline of 5.85 cm; predicting the mean corresponds to +3.52 cm.

| Arm | (ii) | (i) test real | (i) test de-id |
|---|---|---|---|
| SimSwap A, Full Synthetic / Hybrid | +3.00 / **+0.31** | +7.50 / **−0.10** | +3.01 / **+0.00** |
| SimSwap B, Full Synthetic / Hybrid | +3.62 / **−0.03** | +12.05 / **+0.62** | +4.38 / **+0.43** |
| DeepPrivacy2 fixed, Full Synthetic / Hybrid | +10.58 / **−0.10** | +3.77 / **−0.14** | +3.78 / **−0.07** |
| DeepPrivacy2 per frame, Full Synthetic / Hybrid | +11.65 / **−0.02** | +3.61 / **−0.29** | +3.62 / **−0.28** |
| DeepPrivacy2 full body fixed, Full Synthetic / Hybrid | +10.06 / **+0.16** | +3.77 / **−0.22** | +4.07 / **−0.25** |
| DeepPrivacy2 full body per frame, Full Synthetic / Hybrid | +11.27 / **+0.11** | +3.65 / **−0.16** | +3.71 / **−0.17** |

The fourth architecture agrees with the other three, and sharpens one point.
DeepPrivacy2 again retrains to the no-information level (+3.6 to +4.1 against
+3.52) and Hybrid again sits at baseline (−0.29 to +0.62). Trained on SimSwap and
deployed on real faces, iTracker is *worse than predicting the mean* — +7.5 and
+12.1 cm, every fold between 10.6 and 29.6 cm against 9.4 — while it fits the
synthetic training set normally: it learns the synthetic faces and fails to
transfer to real ones. A Full Synthetic SimSwap release can therefore be worse
than useless to a third party, depending on their architecture.

Fold 2 is hard for every model, real data included (7.3–8.5 cm), which compresses all penalties in that fold. Tested on the release, the SimSwap Full Synthetic penalty is positive in 148 of 150 fold-level cells (two templates × three architectures × five folds × five initialisations).

## 8.4 What the cost means for clinical measurement

A centimetre figure on its own does not say whether a protocol is usable; it has to be set against what each biomarker needs to resolve. Tables 5–8 report the uncalibrated error, which is the right quantity for comparing arms, because holding calibration out isolates the effect of the imagery. It is the wrong quantity for asking whether the instrument is clinically usable, because a deployed screening tool calibrates each user. With per-participant calibration the same four architectures reach 3.70–5.11 cm on real data, which at the 80–120 cm viewing distance of our protocol is 1.8–3.7°.

**Table 13.** What each oculomotor biomarker must resolve, against the calibrated error of 1.8–3.7°. Requirements are derived from published normative ranges.

| Biomarker | What it must resolve | Requirement | Verdict at 1.8–3.7° |
|---|---|---|---|
| Anti-saccade directional error | which hemifield the gaze went to | coarse: roughly half the stimulus eccentricity | **usable** |
| Saccade latency | movement onset | one frame; 100°/s detection threshold [saccadethresh] | **usable** — measured bias −16.8 ms, under one 33 ms frame |
| Search-task dispersion | spread over tens of cm | coarse | **usable** |
| Fixation stability (BCEA) | healthy median 0.75–1.07 deg², a sub-degree ellipse [bcea] | < 0.5° | **below our noise floor** |
| Peak saccade velocity | a 30–80 ms movement | ≥ 250 Hz sampling | **undersampled at 30 Hz** |

Two of the five biomarkers are beyond this instrument before de-identification
is considered, for reasons de-identification cannot change: fixation stability
lives at a sub-degree scale our error cannot reach, and peak saccade velocity
needs a sampling rate an order above 30 Hz. The three that do work are the
coarse ones — hemifield discrimination, movement onset, and large-scale
dispersion.

**Table 14.** Calibrated error per release protocol, in centimetres and in degrees of visual angle at 120 and 80 cm, as ranges over the four architectures.

| Arm | cm | angular (120 → 80 cm) |
|---|---|---|
| Real Data | 3.70–5.11 | 1.8–3.7° |
| Hybrid (A / B) | 4.42–5.65 / 5.10–5.48 | 2.1–4.0° / 2.4–3.9° |
| Full Synthetic (A / B) | 7.28–8.97 / 8.69–11.21 | 3.5–6.4° / 4.1–8.0° |

This is what makes the utility axis interpretable. Hybrid leaves the instrument
in the same regime (2.1–4.0° against a 1.8–3.7° baseline), so the biomarkers
that worked still work. Full synthesis moves it to 3.5–8.0°, which is not a
proportionate cost along a continuum but a move out of the regime: at 8° the
hemifield discrimination that anti-saccade error rate depends on is no longer
comfortably resolved. **The protocol that keeps the measurement clinically
usable is precisely the one that releases the participant's own ocular
biometrics.**

Webcam-based eye tracking validated on clinically relevant saccade and free-viewing paradigms reports about 1.42° mean error [webcamval]. That figure is obtained after careful per-user calibration, so the like-for-like comparison is with our calibrated 1.8–3.7°, not with the uncalibrated numbers in Table 5; on that basis the two sit in the same regime, with ours somewhat coarser.

# 9 DISCUSSION

Every claim in this section concerns appearance-based gaze data — face and eye crops released for training gaze models. We make no claim about other sensing modalities; the evidence does not reach them.

**The pixels that make gaze estimation work are the pixels that identify.** Every
protocol we tested that kept the gaze signal did so by keeping genuine ocular
imagery, and genuine ocular imagery verifies identity at 62–80% against a 1.6%
floor. Hybrid recovers 85–101% of the synthesis penalty, and 91–106% when the
models are retrained on the release, precisely because what it restores is the
participant's own eye pixels. Releasing no pixels at all does not escape this:
four iris coordinates per frame still link the release (ARI 0.27) while carrying
almost no gaze signal across participants. For gaze data there is no part of the
face that is useful for the task and safe to release unchanged.

**Re-identification survives on the face crop; linkability survives
everywhere.** Synthesising the eyes defeats an attacker holding the
participants' photographs on the eye crops — A1 is at the floor for every
full-synthesis method — but not on the face crop: SimSwap's and FAMS's synthetic
faces still verify against the participant's own photographs at 18–24% TAR under
ArcFace, against a 1.6% floor, and only DeepPrivacy2, which repaints the face,
reaches the floor. Linkability is harder still: the same synthetic eye crops
that score 0.2% against photographs verify at 43% within the release, and face
crops partition exactly by participant. Under the European test for
anonymisation, being able to single out each participant's records is enough
for a release to remain personal data, whether or not anyone is named
[wp29anon]. The channels that carry this — hair and skin at the crop border,
head pose, crop geometry, illumination — are left alone by inner-face editing,
and two of them need no face model at all: crop dimensions alone identify 66.7%
of participants, photometric statistics alone link at ARI 0.87–0.90. A privacy
number reported without its attacker therefore cannot tell a dataset author
whether a release is safe.

**Each half of the condition is attainable; their conjunction is not yet.**
Release-only linkage falls only when the synthetic identity is not held fixed
per participant *and* the release contains nothing the generator leaves
untouched; it matters for releases that strip participant identifiers, since
one that keeps them is linkable by construction. We observed each half
separately. The diffusion anonymiser breaks per-participant consistency by
construction and yields models that transfer to real faces — though its pupils
carry less gaze than SimSwap's — but leaves the face crop linkable (ARI
0.85–0.94) and, under ArcFace, verifiable against the participant's
photographs. Full-body
inpainting with a fresh identity per frame meets both halves and breaks
discovery of the cohort (face-crop ARI 0.297), but it repaints the eyes, and a
gaze model trained on it learns nothing. Even there, an attacker with a few
labelled frames per person still verifies 36% of them: the condition blocks
discovery, not verification. What a releasable gaze corpus would need is
therefore specific: a generator that keeps the pupil at least as faithfully as
face swapping does, produces eyes from which models transfer to real users as
the diffusion anonymiser's do, edits everything the released crop contains, and
resamples identity per recording session rather than per frame — per-frame
resampling breaks the temporal coherence gaze pipelines rely on, and
per-session resampling cannot be verified on a single-session corpus like ours.
We know of no method that meets all four.

**Utility has to be measured the way the release will be used.** Deploying a
model trained on real data on de-identified crops conflates two things that
retraining separates. For inpainting, most of the deployment penalty is domain
shift, yet what remains after retraining is the no-information level: the
repainted eyes carry no recoverable gaze. For face swapping the penalty survives
retraining, and for one architecture a model trained on the swapped release is
worse than predicting the mean on real faces. A utility figure for a released
gaze corpus should therefore come from models trained on the release and tested
on real faces, reported beside the error of predicting the mean gaze point —
and the pupil should be checked directly, because transfer and information can
diverge: the release that transfers best carries less pupil signal than the one
that transfers worst.

**What this means for releasing gaze data now.** None of the protocols we
measured makes a smartphone gaze corpus anonymous in the European sense: a
usable release keeps each participant's frames together, so it is linkable by
construction, and every gaze-preserving protocol also leaves residual
re-identification on the face or eye crops. CognitiveGaze is therefore shared
under controlled access, with derived gaze traces in an open tier
whose own linkage we measured rather than assumed (ARI 0.033, low but not zero).
We release the measurements as a tool, `privaudit`, and a fifteen-item checklist,
so that a gaze dataset can be audited against the attacks and floors used here
before it is shared; the tool reproduces every eye-ROI result in this paper.

In the terms of the cycle in §1, this paper does not break it, and we do not
claim to. Our own corpus ends where the cycle predicts — under controlled
access — because no de-identification we measured yields a release that is
both unlinkable and useful. What changes is the step at which the next group
decides: instead of assuming that a de-identified release is safe, or that it is
useless, a dataset author can measure both before publishing, and a generator
developer knows what a gaze-preserving method must achieve (§10) for the
cycle to open.

# 10 DESIGN GUIDELINES FOR RELEASING GAZE DATA

Table 15 distils the results into seven guidelines for authors of appearance-based gaze datasets, each with the evidence behind it and how to check it. The full fifteen-item checklist and the audit tool are in the supplementary material [privaudit-repo].

**Table 15.** Design guidelines for releasing gaze data.

| # | Guideline | Evidence in this paper | Check |
|---|---|---|---|
| G1 | **Report re-identification on every released stream, and test linkability.** Run A1 against photographs on each stream, face crop included, and report whether the content singles out participants (A2) as the anonymisation test. | Synthetic eye crops: 0.2% under A1, 43% under A2; synthetic face crops still 18–24% under A1. | privaudit R2–R4 |
| G2 | **Measure a floor on your own recordings.** Run every attack on a face-free patch of the same frames and quote results against it. | Single-session floor: TAR 1.57%, Rank-1 ~50%, ARI 0.074–0.089. | privaudit R0 |
| G3 | **Do not buy utility by releasing real eyes.** Retaining genuine ocular crops restores accuracy and releases the biometric with it. | Hybrid recovers 85–106% of utility; real eye crops verify at 62–80%. | privaudit R2–R3 on eye streams |
| G4 | **Close the channels that need no face model.** Release every crop at one fixed size and normalise per-crop photometry. | Crop size alone: 66.7% identification; photometry alone: ARI 0.87–0.90. | privaudit R5, R6 |
| G5 | **Meet both halves of the condition.** Do not hold the synthetic identity fixed per participant, and release nothing the generator does not edit. | Only the cell meeting both breaks face-crop linkage (1.000 → 0.297); verification still 36%. | privaudit R2, R3, R7 |
| G6 | **Measure utility by training on the release, and check the pupil directly.** Test on real faces, report the predict-the-mean error beside it, and measure whether the released pupil still follows true gaze. | Deployment-only penalties of +10–15 cm shrink to the no-information level after retraining; FAMS transfers best yet its pupils track gaze less closely than SimSwap's (17/17 participants). | — (retraining protocol, §8.3) |
| G7 | **If any check fails, use controlled access.** Share under an agreement that forbids re-identification and linkage, and state which channels leak. | No protocol tested passes G1–G5 with useful gaze accuracy. | privaudit overall verdict |

# 11 LIMITATIONS

Five constraints bound these conclusions. Participants were recorded in a
single session. We attacked the resulting confound directly — photometric
normalisation and a large within-session illumination change both leave the
linkage intact — so the recogniser is not exploiting imaging conditions. But
the dissociation is partial: changing the screen does not change the ambient
lamp, the camera position or the seating, and "this participant's skin tone
under this lamp" cannot be separated from "this recording's exposure" without a
second session. That also bounds the photometric attack we report, which is
high precisely because the two are confounded. The longitudinal tracking that
makes linkage consequential is itself the cross-session case, so a second
recording remains the decisive experiment and we did not run one.
The cohort is 18 healthy adults, moderately homogeneous in ethnicity and age
and recruited from a single institution, so we make no claim about clinical
populations and describe the contribution as an evaluation of dataset release
rather than of home screening. Privacy is measured with two recogniser
families, both trained on web-scale face corpora, and they can disagree:
FaceNet places the FAMS face crop at its floor where ArcFace does not.
Recognition by people who know the participants — who use hairstyle, glasses
and face shape, which FAMS preserves — was not measured. The structural claim is
tested on one replacement method (two templates), one removal method in two
modes and one diffusion method; methods editing in diffusion latent space are
untested. Full-body DeepPrivacy2
ran without its generator for persons that DensePose fails to map (the weights
are no longer distributed). The participant was mapped and repainted in every
frame; in 1.4% of frames a second person region took the unmapped path and was
left unedited, outside the released crops. On footage with more people, or
people DensePose cannot map, that path would release them unedited. Retraining on the release was
run with five initialisations for three architectures and one for iTracker
(about 9 h per fold); FAMS was retrained on the three faster architectures only.

# 12 CONCLUSION

De-identification for gaze data has been judged by whether the
face still looks like the participant. We measured instead whether the release
can still be linked and whether a model trained on it still works, across three
generator families, two release forms and three training conditions. No
configuration was both: the ones that keep the gaze signal keep the identity,
and the one that breaks linkage leaves nothing to learn from. The two-part
condition says what a generator would have to do to change that, and the audit
tool lets a dataset author check, before release, whether theirs does.

# DATA AVAILABILITY

CognitiveGaze contains facial video and is shared under controlled access: researchers apply to the data holder and sign a data-use agreement that prohibits re-identification and linkage. Derived per-frame gaze estimates and oculomotor measures form an open tier, whose residual linkage is reported in §7. The audit tool `privaudit`, the fifteen-item release checklist and the analysis code are available at [privaudit-repo].

# SUPPLEMENTARY FIGURES

![Figure S1](figs_core/fig_core2_utility.pdf)

**Figure S1.** **Utility under three training conditions.** **(a)** Full Synthetic and **(b)** Hybrid penalty against each model's real-data baseline, for models trained on real data and tested on de-identified crops (grey), and models retrained on the release and tested on real (green) or de-identified (purple) faces. Bars: mean of three architectures; dots: each architecture. Red band: no-information reference; grey band: initialisation noise floor. Retraining removes most of the DeepPrivacy2 penalty but stops at the no-information level; it does not remove SimSwap's; for FAMS it removes most of it on real faces (+0.94 cm) and less on the release (+2.15 cm). Hybrid recovers most of the penalty in every condition and, once retrained, lies within the noise floor.

![Figure S2](figures/fig3_threat_models.pdf)

**Figure S2.** **The same de-identified eye crops under two attackers.** True-accept rate at a false-accept rate of 1e-3 for the left and right eye ROI, unprocessed and SimSwap with two target identities (t1 = identity A, t2 = identity B). Blue: an attacker who holds enrolment photographs of the participants (A1) and matches them against the release. Orange: an attacker who holds only the release (A2), matching first-half templates against second-half probes. Vertical line: the eye-native floor (1.57%), measured on a face-free patch of the same frames. Unprocessed eyes verify at 62–67% under both attackers. SimSwap drives A1 to the floor (0.2–1.8%) while A2 still verifies 43–62%: a privacy figure reported without its attacker can differ by 60 points on identical data.
