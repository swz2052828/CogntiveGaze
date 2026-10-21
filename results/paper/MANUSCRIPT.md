# Keep Your Eyes On It: Measuring Gaze Utility and Re-identification Risk in De-identified Smartphone Gaze Data

## ABSTRACT

Smartphone eye tracking moves oculomotor assessment out of the clinic, but it
records video of the participant's face, which blocks the dataset sharing the
field depends on. Face de-identification has been proposed as the remedy and
judged by whether the de-identified face still matches the real one. We measure
what such a release protects and what a gaze model trained on it is worth.

We introduce CognitiveGaze: 18 participants performing seven oculomotor tasks,
recorded simultaneously by a smartphone and an EyeLink 1000, so every frame
carries physiological ground-truth gaze. We evaluate face swapping, inpainting
and diffusion de-identification with four gaze architectures, deployed on and
retrained on each release, and with two face recognisers.

Retaining the genuine eye region recovers 85–106% of the accuracy lost to
synthesis, yet those eye crops verify identity at 62–80% against a 1.6% floor:
the eye pixels that make gaze estimation work are the pixels that identify.
Synthesising the eyes defeats matching on the eye crops, but synthetic face
crops still verify at 18–24%, and every release that retains gaze information
still singles out each participant, which under European anonymisation guidance
leaves it personal data. Inpainted eyes carry no recoverable gaze. A diffusion
anonymiser yields the models that transfer best to real faces (0.4–1.3 cm
against 2.5–6.2 cm), but its pupils follow true gaze less closely than
face-swapped ones, and stronger settings trade pupil signal for identity without
reducing linkability. No configuration we tested is both private and useful. We
state what a releasable gaze corpus requires and release an audit tool and
checklist.

**CCS Concepts:** • **Security and privacy** → **Pseudonymity, anonymity and untraceability**; • **Human-centered computing** → **Empirical studies in ubiquitous and mobile computing**; • **Computing methodologies** → *Biometrics*.

**Additional Key Words and Phrases:** smartphone gaze estimation, face de-identification, re-identification, linkage attack, privacy–utility trade-off, ocular biometrics, dataset release, privacy auditing

# 1 INTRODUCTION

Oculomotor function is a sensitive marker of neurological and cognitive change, but the instruments that measure it well, such as the EyeLink 1000, are confined to the clinic. Smartphone gaze estimation moves the measurement to where patients are [krafka16, valliappan20]; it does not move the data. A smartphone eye tracker records the participant's face, and face video is not a shareable research artefact.

The result is a cycle that reproduces its own scarcity (Figure 1). Because few public corpora record the face under usable conditions, studies converge on the same ones — GazeCapture above all [krafka16] — and inherit their assumptions: a hand-held phone, a small screen, a short viewing distance, and the position of an on-screen target in place of measured gaze. Newer corpora adopt the same protocol for comparability [zjugaze, dynamicgaze]. Clinical protocols violate most of these assumptions: stimuli span a large display viewed from about a metre, and the quantity of interest is the gaze trajectory. Groups that record such data hold it in separate silos, which leaves analysis choices without a common reference [simmons11, carp12]. Releasing the data meets the privacy problem: face video of a study participant is personal data and, in a clinical study, health data under the GDPR [gdpr], and HIPAA lists full-face photographs among the identifiers to be removed [hipaa]. The standard exits do not yield a usable public corpus. Blurring or masking removes the geometry a gaze model reads [blurmask, disguises]; homomorphic encryption and federated learning leave the data with its holder and are costly for video [acar18he, kairouz21fl]; controlled access restricts who can use it. The next group again finds the same few corpora.

![Figure 1](figs_core/fig_cycle.pdf)

**Figure 1.** The data-scarcity cycle in smartphone gaze research. A de-identification that removes identity and keeps gaze would break it at step 5; this paper measures whether any current method does.

The cycle could break at one step: a transformation that removes identity and keeps the gaze signal would allow open release. Generative de-identification is the candidate. Face swapping is meant to preserve the geometry a gaze model reads — head pose, eyelid aperture, iris position — and has been applied to clinical video of movement disorders, autism assessments and seizures [zhu20deepfakes, disguises, li24seizure]. Gaze degrades under it unless the generator is constrained to keep it [wilson23gaze], and a recent evaluation concludes that face swapping hides identity effectively [mustu25]. These evaluations ask whether the swapped face still matches the real one.

For gaze data this question is insufficient, for two structural reasons. First, the eye region is itself a biometric: a 120 px eye crop verifies identity at 62–80% at a false-accept rate of 1e-3, against a measured floor of 1.6%, so a protocol that restores accuracy by retaining genuine eye pixels releases the biometric with them. Second, synthesising the whole face removes the correspondence between the released eye crops and the real face, but not between the synthetic face crop and the real face, and it leaves the corpus partitionable by individual. Under European guidance on anonymisation, that suffices for the data to remain personal data [wp29anon]. The channels that carry this lie outside the region generators edit.

To measure this we recorded CognitiveGaze: 18 participants performing seven oculomotor tasks, recorded simultaneously by a smartphone and an EyeLink 1000, so that every frame carries physiological ground-truth gaze rather than a target label. Its protocol deliberately departs from GazeCapture's — stimuli on a 53 × 30 cm display at 80–120 cm — and is the kind of clinical corpus the cycle keeps unreleased. On the privacy axis we follow security practice: a stated attacker, recognisers never trained on our data, a second recogniser as a cross-check, and a session-nuisance control that measures what "no identity signal" scores on single-session recordings.

Our contributions are:

- **A privacy evaluation protocol for gaze corpora.** We separate re-identification from photographs (A1) from a test of whether the content singles out individuals (A2, the linkability criterion of European anonymisation guidance), show that the two differ by up to 60 points on identical crops, and release the protocol as an audit tool and fifteen-item checklist [privaudit-repo] that reproduces every eye-crop result in this paper.
- **Evidence that gaze utility and identity share the eye region.** Retaining genuine eye crops recovers 85–106% of the synthesis penalty for SimSwap and DeepPrivacy2 in every architecture and condition, and those crops identify at 62–80%. Releasing eye landmarks instead of pixels still links the corpus while carrying almost no gaze signal.
- **A tested two-part condition for content-based linkage.** Linkage from released content falls only when the synthetic identity is not held fixed per participant *and* the release contains nothing outside the generator's edit domain. A 2×2 experiment with DeepPrivacy2 confirms it, and a diffusion method not used to derive it behaves as predicted.
- **A utility benchmark under de-identification.** Four architectures, three generator families and four train–test conditions, with an initialisation noise floor, a no-information reference, stream-isolated models and a model-free pupil test. It shows that the release whose models transfer best to real users is not the one whose pupils carry the most gaze.

# 2 RELATED WORK

**Smartphone gaze estimation.** iTracker and GazeCapture established appearance-based gaze estimation on phones [krafka16], and Valliappan et al. reached research-grade accuracy with per-user calibration [valliappan20]. AFFNet [affnet] and MGazeNet [zhu24] condition eye features on facial context through adaptive normalisation. All consume the same artefacts — a face crop, two eye crops and a face grid — and inherit the same data-sharing problem.

**Face de-identification.** Removal methods include k-Same and its neural successors [newton05, ksamenet]; DeepPrivacy and DeepPrivacy2, which inpaint the face or the whole body [deepprivacy, deepprivacy2]; CIAGAN [ciagan]; and diffusion methods that condition on the source face [fams] or edit diffusion latents [fluid, apl]. Swapping methods such as SimSwap and SimSwap++ replace identity instead [simswap, simswappp]. Inpainting the eyes is expected to destroy gaze, whereas swapping is meant to preserve it. We test both expectations and find that, on the privacy axis, what the edit domain leaves out matters more than the objective.

**Face swapping for clinical video.** Swapping has been applied to movement-disorder videos, preserving the body keypoints needed for diagnosis [zhu20deepfakes], and to seizure videos, rated subjectively for retained semiology [li24seizure]. Wilson et al. gave the first privacy–utility methodology for patient-privacy face swaps, measuring gaze and expression against blurring [disguises], and later added a gaze loss because full-face objectives do not preserve the eye region [wilson23gaze]. A digital mask for clinical video [digitalmask] was promptly contested [maskconcerns]. Muştu and Ekenel conclude that face swapping hides identity effectively, measured by matching the swapped face to the real one [mustu25]. Our results agree for synthesised eye crops but not for synthetic face crops, and that test does not ask whether the release singles out individuals — the linkability criterion of European guidance [wp29anon], which survives swapping almost intact.

**Eye movement as a biometric.** Identity is recoverable from eye movement itself [rigas, holland, bioeye]. Privacy research on eye tracking has developed differential privacy for gaze streams [bozkir, steil], learned obfuscation [fuhl] and private scanpath comparison [scanpath], reviewed in [etprivreview] and, for mobile devices, in [alsakar25]. We therefore measure, rather than assume, the risk of a pixel-free release tier.

# 3 THE COGNITIVEGAZE DATASET

**Participants.** Twenty-three volunteers (13 male, 10 female) were recruited; five were excluded before analysis for calibration failure or incomplete acquisition, leaving 18, aged 20–54 (median 29): White/Caucasian (12), East Asian (2), Black/African, Middle Eastern, Asian-Indian and mixed background (1 each). Exclusion criteria were photosensitive migraine, epilepsy, motion sickness and substance use affecting oculomotor function. Three participants wore glasses and were retained, because vision correction is often needed to complete a cognitive protocol. One participant has no usable ground-truth recording, so utility analyses use 17 participants and privacy analyses 18. The study was approved by the institutional ethics committee (reference withheld for review), and all participants gave written informed consent.

**Apparatus and tasks.** Participants sat 80–120 cm from a 53.3 × 29.6 cm display (1920 × 1080 px) without head restraint (Figure 2). An EyeLink 1000 recorded ground truth at 1000 Hz after nine-point calibration; a smartphone (Xiaomi Mi 8) mounted below the display recorded the face with its rear camera at 30 fps. The battery, designed with rehabilitation specialists and informed by [valliappan20], comprised fixation stability, prosaccades, anti-saccades, smooth pursuit, optokinetic nystagmus, free viewing of 30 OSIE scenes [osie] and visual search in 12 of them, presented with PsychoPy. Blink frames were retained and annotated.

**Synchronisation.** A 0.6 s tone preceding each block was recorded by the phone and timestamped by the experiment software. Its detected onset anchors video frames to EyeLink time, and the ground truth is resampled to 30 Hz, so each frame carries a gaze coordinate.

![Figure 2](FIGURE: original setup figure)

**Figure 2.** Data acquisition. (A) Participant station: display at 80–120 cm, smartphone below it, EyeLink 1000 recording ground truth. (B) Control workstation.

# 4 GAZE ESTIMATION MODELS

All models take the left-eye, right-eye and face crops (224 × 224) and a 25 × 25 face grid, and regress the gaze point on the display. We evaluate four published architectures (Table 1, Figure 3): two fuse streams late, and two modulate eye features with a guidance factor regressed from the face. Training minimises mean squared error; we report mean Euclidean error in centimetres on the display. We use participant-wise five-fold cross-validation [kfold], re-initialise weights for every fold, keep the best validation epoch, and train every configuration from five independent initialisations, whose spread is the noise floor of all utility comparisons. Arms are compared on the uncalibrated model; for the clinical analysis (§8.4) the regression head is additionally fine-tuned per participant on calibration frames recorded before the tasks. A face-only model, which receives no eye crops, serves as a null control.

**Table 1.** The four gaze architectures.

| Model | Streams | Fusion |
|---|---|---|
| iTracker [krafka16] | shared eye CNN, face CNN, grid MLP | late concatenation |
| MobileNet-V3 Large [mobilenetv3] | MobileNet-V3 backbones, grid encoding | late concatenation |
| AFFNet [affnet] | CNNs with SE blocks; face-regressed guidance factor | adaptive group normalisation of eye features |
| MGazeNet [zhu24] | CNNs with SE blocks; face-regressed guidance factor | linear adaptive batch normalisation (adapted to 224 px) |

![Figure 3](FIGURE: original architecture comparison figure)

**Figure 3.** (A) Layer-level comparison of the four architectures. (B) Workflow from ROI extraction to supervised training against EyeLink ground truth.

# 5 DE-IDENTIFICATION METHODS

We test three families that differ in what they edit and how they treat identity. Every operator runs on the original frame, and released crops are re-extracted with the stored per-frame geometry, so they are pixel-comparable across arms.

**Replacement: SimSwap++.** An adapted SimSwap++ [simswappp] transfers a source identity onto the participant's face while preserving pose, expression and gaze direction; each participant maps to the same synthetic appearance in every frame. Two source templates of different appearance (Identity A and B; Figure 4) separate the method from the template. We call this family SimSwap below.

**Removal: DeepPrivacy2.** DeepPrivacy2 [deepprivacy2] inpaints the detected face box (face mode) or the whole person, guided by a dense surface map (full-body mode). The latent code is fixed per participant or drawn per frame (truncation 1.0).

**Diffusion: FAMS.** Face Anonymization Made Simple [fams] conditions a diffusion model on the source face and pastes back an aligned square covering face, forehead and some hair. We use the authors' setting (anonymisation degree 1.25, 25 steps, guidance 4.0) with the noise seed fixed per participant or drawn per frame, on all 210,924 task frames, and sweep the degree from 1.35 to 3.0 on a privacy sample (§7).

**Release protocols.** *Full Synthetic* takes every crop from the de-identified frame. *Hybrid* combines the de-identified face crop with the original eye crops (Figure 5), testing whether the face can be de-identified while the eyes stay genuine. As §7 shows, Hybrid preserves utility, not privacy.

![Figure 4](FIGURE: original face-swap examples figure)

**Figure 4.** SimSwap++ with two source templates (Identity A and B): identity is replaced; pose, expression and surroundings are preserved.

![Figure 5](FIGURE: original Hybrid pipeline figure)

**Figure 5.** Release protocols. (A) Full Synthetic: all crops from the de-identified frame. (B) Hybrid: de-identified face crop with the original eye crops.

# 6 THREAT MODEL AND PRIVACY EVALUATION

**Released artefact.** A release contains the cropped streams the models consume — face crop, two eye crops and face grid — not the video. All attacks operate on these crops.

**Attacks.** *A1, re-identification from photographs,* is the primary threat: the attacker holds photographs of candidate individuals, for instance members of the recruitment pool, and asks whom a released crop shows. Gallery images are original crops from the first half of each recording; probes are released crops from the second half. *A2, linkability,* tests a property rather than an adversary: whether the content alone lets frames be grouped by individual. It operationalises two of the three criteria by which the Article 29 Working Party assesses anonymisation, singling out and linkability [wp29anon]; records that can be isolated per person remain personal data even without a name [gdpr]. A usable gaze corpus keeps each participant's frames together for participant-wise splits and calibration, and is therefore linkable by construction; A2 asks whether the content would restore that grouping if identifiers were removed. The two tests can differ by 60 points on identical crops (Figure S2).

**Recognisers.** ArcFace (InsightFace buffalo_l, w600k_r50) is used without adaptation, with cosine similarity to per-participant templates. Key measurements are repeated with FaceNet (Inception-ResNet-v1 trained on VGGFace2), which differs in backbone, training data and objective. Each recogniser is quoted against its own floor (ArcFace 1.57%, FaceNet 3.52%). Detection and alignment are taken from the original frame, so a method gains nothing by defeating the detector.

**Metrics.** For A1, TAR at FAR = 1e-3 is primary; closed-set Rank-1 is not evidence on these data (below). For A2 we report the adjusted Rand index (ARI) of a k-means partition at the true k = 18, the k selected unaided by silhouette, and within-release verification (first-half templates, second-half probes).

**Session-nuisance floor.** Each participant was recorded in one session, so lighting, seating and background are constant within a participant and differ between participants. Running every attack on a 120 px face-free patch of the same frames gives Rank-1 50.5% (chance 5.6%), TAR 1.6% and ARI 0.089 (0.074 for the eye pipeline), with k = 2 selected. Rank-1 is therefore uninformative here; TAR and ARI are read against these floors.

**Uncertainty.** Privacy intervals are 95% participant-level cluster bootstraps. Verification intervals keep the full-data operating threshold and resample participants' genuine scores (2,000 replicates), because re-estimating a 1-in-1,000 impostor quantile in each resample biases the interval upward. Linkage intervals resample participants, give duplicates a shared label, and refit (1,000 replicates).

**Tooling.** All attacks, floors and intervals are packaged as privaudit, which maps each result onto the release checklist. On this study's saved embeddings it reproduces all 32 eye-crop verification and linkage values exactly; re-embedding the crops from scratch, all 46 audited values agree within GPU inference noise.

# 7 IDENTITY LEAKAGE UNDER DE-IDENTIFICATION

Table 2 summarises SimSwap under both tests, Table 3 gives face-crop A1 for every family, and Figure 6 gives linkage.

**The eye region identifies.** An unprocessed 120 px eye crop verifies at 66.6% (left) and 62.2% (right) under ArcFace, and at 69.1% and 81.2% under FaceNet, against floors of 1.6% and 3.5%. Hybrid retains these crops and therefore their leakage: what restores utility in Hybrid also releases the eye biometric.

**Synthesis protects eye crops, not face crops.** With the eyes synthesised, eye-crop A1 falls to 0.2% and 1.8% (FaceNet 0.0% and 1.5%), at the floor for both templates and both recognisers. The synthetic face crop still verifies at 18.4% [5.1, 35.4] (Identity A) and 22.0% [6.8, 40.7] (Identity B) under ArcFace. FaceNet agrees for Identity A (14.4% against its 3.5% floor) but not for Identity B (0.2%). With 18 participants, these intervals support "above floor, an order of magnitude below unprocessed imagery", not a finer grading.

**The release remains linkable.** An unsupervised partition recovers the participants exactly from face crops (ARI 1.000 at k = 18; floor 0.089) for both templates, because replacement maps each participant to the same synthetic appearance every time. The synthetic eye crops that score 0.2% under A1 verify at 42.8% and 62.5% within the release. Linkage persists in the 120 px eye crop, which excludes hair, clothing and face outline: ARI 0.68–0.79 against 0.84–0.85 for unprocessed eyes, with overlapping intervals (e.g. [0.59, 0.86] against [0.70, 0.96]).

**Table 2.** SimSwap under A1 (TAR at FAR 1e-3, ArcFace) and A2 (ARI at k = 18), with 95% participant-level intervals. Hybrid face crops are synthetic; its eye crops are unaltered.

| Released stream | Unprocessed | Full Synthetic (A) | Full Synthetic (B) | Hybrid (Real Eyes) |
|---|---|---|---|---|
| Face ROI, TAR (A1) | 100% | 18.4% [5.1, 35.4] | 22.0% [6.8, 40.7] | 18.4% (face is synthetic) |
| Left eye ROI, TAR (A1) | 66.6% [53.1, 78.7] | 0.2% [0.0, 0.5] | 0.6% [0.0, 1.6] | **66.6%** (unaltered) |
| Right eye ROI, TAR (A1) | 62.2% [49.7, 73.3] | 1.8% [0.1, 4.4] | 1.1% [0.0, 2.8] | **62.2%** (unaltered) |
| *Session-nuisance floor* | *1.6%* | *1.6%* | *1.6%* | *1.6%* |
| Face ROI, ARI @ k=18 (A2) | 1.000 | 1.000 | 1.000 | 1.000 |
| Left eye ROI, ARI @ k=18 (A2) | 0.844 [0.70, 0.96] | 0.684 [0.59, 0.86] | 0.737 [0.60, 0.88] | 0.844 (unaltered) |
| Right eye ROI, ARI @ k=18 (A2) | 0.853 [0.73, 0.96] | 0.789 [0.64, 0.92] | 0.761 [0.62, 0.95] | 0.853 (unaltered) |
| *Linkage floor* | *0.089* | *0.089* | *0.089* | *0.089* |

**Linkage is not an illumination artefact, but photometry also links.** Removing per-crop brightness and contrast leaves eye-crop linkage intact (ARI 0.684 → 0.742), local histogram equalisation reduces it by only 0.05–0.14, and a large within-session illumination change (dark-screen versus natural-image blocks) does not reduce it. The recogniser therefore does not depend on photometry. Conversely, 21 photometric statistics per crop link participants without any face model (ARI 0.87–0.90), and identity replacement leaves them unchanged. The two channels are independent, so normalising photometry closes only the second; on single-session data, the photometric channel also mixes person and recording (§11). Crop geometry is a third channel: face and eye crop sizes alone identify 66.7% of participants, and 94.4% with inter-ocular distance. Our release normalises crop geometry.

**A two-part condition, tested on a second family.** DeepPrivacy2 removes rather than replaces identity. Its face mode edits 40% more of the crop than SimSwap; its full-body mode repaints the whole person, motivated by the concern that face anonymisation leaves identifiers outside the face [deepprivacy2]. Crossing edit domain with identity regime (Figure 6a), face-crop linkage is 1.000 with face editing and a fixed identity, 0.982 when either the domain is widened or the identity is resampled per frame, and 0.297 only when both are; an attacker who must estimate the cohort size then finds no structure (k = 2, ARI 0.000). Verification falls from 89.7–100% to 35.7% [27, 45], still well above the 1.6% floor: the condition prevents discovery of the partition, not verification by someone holding labelled frames. The fixed-identity full-body arm has the same low resolution and still links at 0.982, so the collapse reflects the identity regime rather than image quality. On the eye crop, which contains only edited pixels, per-frame identity alone suffices (ARI 0.805 → 0.148; floor 0.074). Against A1, every DeepPrivacy2 arm is at the floor on eye crops (≤ 1.2%) and face crops (≤ 1.6%; Table 3), consistent with DeepPrivacy2's own evaluation, which matches enrolment images to the release and does not test linkage.

**A diffusion method behaves as the condition predicts.** In FAMS, fixing the noise seed per participant does not fix the synthetic identity, which follows the source frame; the two seed regimes are indistinguishable on every measure. FAMS therefore breaks per-participant consistency by construction. Eye-crop linkage roughly halves (within-release verification 5–8% against 43–63% for SimSwap; ARI 0.29–0.33 against 0.68–0.80), though every interval stays above the eye floor. The face crop, whose border, pose and geometry the aligned square leaves intact, stays linkable (ARI 0.85–0.94). It also verifies against the participant's photographs at 24.1% [15.3, 33.6] under ArcFace, no lower than SimSwap, whereas FaceNet places it near its floor (5.9% [2.2, 10.5]; Table 3).

**Table 3.** Face-crop A1 for every family: released face crops matched against the participant's original face crops. TAR at FAR 1e-3 with 95% participant-level intervals; floors 1.6% (ArcFace) and 3.52% (FaceNet).

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

**Strength trades pupil signal for identity, not for linkability.** Raising the FAMS degree from 1.25 to 1.5 — the strongest setting that still produces a face in more than 90% of frames; beyond it the generator degrades rather than anonymises — lowers ArcFace face verification from 24.1% to 10.1% [3.0, 19.3]; FaceNet reaches its floor from 1.35. Face-crop linkage is unchanged (ARI 0.81–0.90), and the correlation between the synthetic pupil and true horizontal gaze falls from 0.68 to 0.53 (Table 4; §8.1).

**Table 4.** FAMS strength sweep on the privacy sample. Settings with more than 10% non-face outputs are not used for privacy conclusions. Pupil r: correlation of the synthetic pupil with horizontal EyeLink gaze (unprocessed 0.87; SimSwap A 0.79).

| Degree | Non-face outputs | Face TAR, ArcFace | Face TAR, FaceNet | Face linkage ARI | Eye linkage ARI | Pupil r |
|---|---|---|---|---|---|---|
| 1.25 (authors) | 0.0% | 24.1% [15.3, 33.6] | 5.9% | 0.853 | 0.292 | 0.68 |
| 1.35 | 0.1% | 14.1% [5.9, 23.5] | 2.6% | 0.810 | 0.305 | 0.62 |
| 1.5 | 6.5% | 10.1% [3.0, 19.3] | 2.2% | 0.899 | 0.283 | 0.53 |
| 1.6 | 18.7% | 5.9% [0.9, 12.0] | 1.7% | 0.792 | 0.302 | 0.41 |
| 1.75 | 36.0% | 3.7% [0.3, 9.4] | 1.9% | 0.784 | 0.259 | 0.23 |
| 2.0 | 93.9% | — | — | — | — | — |

**Other residual risks.** An adaptive attacker who fits a de-anonymisation map on other participants' paired imagery gains at most 3.3 points on the face crop and nothing on eye crops; fitted including the target, the same maps reach 100% Rank-1, so they memorise identities rather than invert the operator. Our open tier, which releases per-frame gaze estimates and oculomotor measures without imagery, links at ARI 0.033 at the correct k (purity 0.157 against 0.059 chance; permutation-null maximum 0.0015): low but not zero.

![Figure 6](figs_core/fig_core1_privacy.pdf)

**Figure 6.** Content-based linkage. **(a)** DeepPrivacy2 face crops, edit domain (face box or whole person) crossed with identity regime (fixed or per frame): ARI at k = 18 and within-release verification (TAR at FAR 1e-3, 95% interval). Only the cell meeting both halves of the condition falls (ARI 0.297), and verification there remains 36%. **(b)** Eye-crop linkage for every arm with 95% participant-level intervals; dashed line, eye floor (0.074). Fixed-identity SimSwap and DeepPrivacy2 overlap unprocessed eyes; per-frame DeepPrivacy2 approaches the floor (0.12–0.17); FAMS lies between (0.29–0.33) under either seed regime.

# 8 GAZE UTILITY UNDER DE-IDENTIFICATION

## 8.1 Protocol

**Four train–test conditions,** named train→test, with R for real and D for de-identified data:

- **R→R** — trained and tested on real data; the reference for every penalty.
- **R→D** — trained on real data, tested on de-identified crops: the cost to a pipeline that receives de-identified input.
- **D→R** — retrained on the release, tested on real faces: a third party deploying the model on its own users.
- **D→D** — retrained and tested on the release.

Retraining uses the R→R recipe, partition and hyperparameters. Read together, R→D and D→R separate domain shift, which retraining removes, from information loss, which it does not.

**References.** The initialisation noise floor is the spread of R→R over five initialisations (0.04–0.20 cm). The no-information reference is the error of predicting the training participants' mean gaze point, 9.37 cm, a penalty of +3.70 to +4.41 cm depending on the architecture.

**Pupil test.** End-to-end error mixes the gaze information in a release with how well models trained on it transfer. We therefore locate both iris centres with FaceMesh [facemesh] on the released and original face crops of the same frames, express each in its eye's corner-to-corner frame, which removes head pose and crop placement, and report the within-participant correlation of the released pupil position with the original position and with EyeLink gaze, on 4,680 frames common to all releases.

## 8.2 Models trained on real data (R→D)

**Table 5.** Gaze error (cm) under R→D for SimSwap: mean over five folds, then mean ± SD over five initialisations; in parentheses, the change against R→R.

| Model | R→R (real data) | Full Synthetic (A) | Hybrid (A) | Full Synthetic (B) | Hybrid (B) |
|---|---|---|---|---|---|
| iTracker | 5.85 ± 0.14 | 8.85 ± 0.17 (+3.00) | **6.16 ± 0.20 (+0.31)** | 9.48 ± 0.21 (+3.62) | **5.82 ± 0.20 (−0.03)** |
| MobileNet-V3 Large | 4.98 ± 0.13 | 8.65 ± 0.18 (+3.66) | **5.17 ± 0.16 (+0.18)** | 12.03 ± 0.53 (+7.04) | **6.03 ± 0.25 (+1.04)** |
| AFFNet | 5.67 ± 0.20 | 8.07 ± 0.25 (+2.41) | **5.85 ± 0.24 (+0.19)** | 9.64 ± 0.53 (+3.98) | **5.74 ± 0.25 (+0.08)** |
| MGazeNet | 4.96 ± 0.04 | 7.74 ± 0.20 (+2.78) | **5.21 ± 0.21 (+0.24)** | 9.72 ± 0.40 (+4.75) | **5.45 ± 0.25 (+0.48)** |

Restoring the genuine eye crops recovers most of the synthesis penalty. Full Synthetic costs +2.41 to +7.04 cm depending on model and template; Hybrid leaves +0.08 to +1.04 cm, recovering 85–101% (Table 6), and the paired difference is negative in every architecture, template and draw. The residual is small but mostly real: six of eight Hybrid arms are detectably worse than R→R, so Hybrid recovers rather than preserves accuracy. The face-only control, which cannot see the eye crops, gives identical results for both arms (within 0.001 cm), confirming that the comparison measures the eye stream, and the guidance-factor architectures recover as much as the late-fusion ones (AFFNet 92% and 98%). The synthesis penalty depends strongly on the template (Identity B minus A: +0.63 to +3.38 cm for Full Synthetic, −0.34 to +0.86 cm for Hybrid).

DeepPrivacy2 and FAMS under R→D are reported in Table S1. Real-data models deployed on DeepPrivacy2 crops are worse than predicting the mean (14.6–20.5 cm against 9.37 cm), and FAMS costs +4.1 to +5.1 cm, more than SimSwap A (+2.4 to +3.7 cm). Hybrid recovers 91–101% (DeepPrivacy2 face), 93–99% (full body) and 93–100% (FAMS). Every one of the 421,848 full-body frames had the participant repainted; in 1.4% of them a second person region without a surface map was left unedited, outside the released crops. Before any DeepPrivacy2 or FAMS number was computed, the real-data arm re-run through the same pipeline reproduced Table 5 to within 0.0012 cm on all 100 folds.

**Table 6.** Hybrid minus Full Synthetic under R→D (cm), paired within initialisation, mean ± SD over five draws; negative values mean the genuine eye crops recover accuracy.

| Model | Identity A | Identity B | Penalty recovered |
|---|---|---|---|
| iTracker | −2.693 ± 0.097 | −3.656 ± 0.320 | 90% / 101% |
| MobileNet-V3 Large | −3.481 ± 0.274 | −5.999 ± 0.655 | 95% / 85% |
| AFFNet | −2.221 ± 0.236 | −3.897 ± 0.428 | 92% / 98% |
| MGazeNet | −2.535 ± 0.119 | −4.273 ± 0.389 | 91% / 90% |
| *Face-only control* | *+0.000 ± 0.000* | *+0.000 ± 0.001* | *0%* |

## 8.3 Models retrained on the release (D→R and D→D)

**Table 7.** Penalty against R→R (cm): mean over MobileNet-V3 Large, AFFNet and MGazeNet, with [min, max] across them; D→R and D→D over five initialisations each. R→R errors are 4.98, 5.67 and 4.96 cm. FS: Full Synthetic; DP2: DeepPrivacy2. Negative Hybrid values denote no detectable difference, not gains.

| Arm | R→D | D→R | D→D |
|---|---|---|---|
| SimSwap A, FS | +2.95 [+2.41, +3.66] | +3.83 [+2.47, +4.56] | +2.76 [+2.50, +3.06] |
| SimSwap A, Hybrid | **+0.20** [+0.18, +0.24] | **+0.09** [+0.01, +0.23] | **−0.04** [−0.14, +0.04] |
| SimSwap B, FS | +5.26 [+3.98, +7.04] | +4.81 [+3.16, +6.21] | +2.54 [+2.17, +3.02] |
| SimSwap B, Hybrid | **+0.53** [+0.08, +1.04] | **+0.04** [−0.10, +0.14] | **−0.08** [−0.19, +0.04] |
| DP2 fixed, FS | +12.22 [+10.90, +13.66] | +4.51 [+3.94, +4.81] | +4.50 [+3.93, +4.81] |
| DP2 fixed, Hybrid | **+0.52** [+0.07, +1.03] | **−0.08** [−0.14, −0.02] | **−0.12** [−0.17, −0.09] |
| DP2 per frame, FS | +12.73 [+11.25, +14.85] | +4.40 [+3.79, +4.77] | +4.34 [+3.79, +4.68] |
| DP2 per frame, Hybrid | **+0.46** [+0.01, +0.97] | **+0.02** [−0.09, +0.10] | **+0.02** [−0.10, +0.08] |
| DP2 full body fixed, FS | +10.67 [+9.65, +12.05] | +4.72 [+4.03, +5.21] | +4.70 [+4.25, +5.02] |
| DP2 full body fixed, Hybrid | **+0.33** [+0.11, +0.61] | **+0.01** [−0.08, +0.15] | **−0.02** [−0.14, +0.10] |
| DP2 full body per frame, FS | +10.82 [+9.67, +12.16] | +4.34 [+3.91, +4.58] | +4.34 [+3.91, +4.59] |
| DP2 full body per frame, Hybrid | **+0.38** [+0.13, +0.63] | **−0.05** [−0.15, +0.05] | **−0.05** [−0.14, +0.05] |
| FAMS, FS | +4.64 [+4.06, +4.94] | +0.94 [+0.39, +1.29] | +2.15 [+1.56, +2.57] |
| FAMS, Hybrid | **+0.27** [+0.19, +0.35] | **−0.01** [−0.13, +0.07] | **−0.02** [−0.14, +0.06] |
| *No-information reference* | *+4.17 [+3.70, +4.41]* | | |

**Retraining does not rescue face swapping.** SimSwap's penalty survives retraining: +2.2 to +3.1 cm on the release and +2.5 to +6.2 cm on real faces, where two of three architectures reach the no-information level. Tested on the release, the penalty is positive in 148 of 150 fold-level cells. iTracker trained on SimSwap is worse than predicting the mean on real faces (+7.5 and +12.1 cm; Table S2) while fitting its synthetic training set normally.

**Inpainting leaves no gaze.** For DeepPrivacy2, 56–65% of the R→D penalty is domain shift, but the remainder is the no-information level in both modes, both identity regimes and both test sets: 580 of 600 fold-level errors are at or above predicting the mean, and none is more than 0.17 cm below it. Resampling the identity per frame changes utility by only −0.63 to +0.19 cm; repainting the eyes, not the identity regime, removes the gaze.

**Hybrid recovers the penalty in every condition:** 91–103% in D→R and 98–106% in D→D for SimSwap and DeepPrivacy2, with every retrained Hybrid cell within −0.19 to +0.23 cm of R→R, inside or at the edge of the noise floor (FAMS: −0.14 to +0.07 cm).

**The diffusion release transfers better; it does not carry more gaze.** Retrained on FAMS and deployed on real faces, models lose +0.94 cm on average (+0.39 to +1.29), against +3.83 for SimSwap A and +4.51 for DeepPrivacy2, and FAMS is better than both in all 15 paired draws. Two measurements show that this is not because its eyes carry more gaze. In the pupil test (Table 8), SimSwap pupils follow true horizontal gaze more closely than FAMS pupils in all 17 participants with ground truth (r = 0.79–0.83 against 0.69; unprocessed 0.87). Eyes-only models (Table 9) reproduce the FAMS advantage on real faces (+0.96 to +1.16 cm against +2.71 to +6.77) but not within the release, where the two methods are comparable and each is better for one architecture in every draw. The difference is therefore transfer: models trained on SimSwap eyes fail on real eyes, whereas models trained on FAMS eyes do not. A plausible but untested explanation is diversity, since SimSwap maps every participant onto one template's eyes and FAMS draws a new identity per frame. Tested on its own release, a FAMS-trained model is less accurate than on real faces (+1.56 to +2.57 cm), the reverse of SimSwap.

**Table 8.** Pupil-position preservation on 4,680 identical frames: within-participant correlation of the released pupil position with the original (horizontal, vertical) and with EyeLink gaze, mean over participants.

| Release | r with original, horizontal | r with original, vertical | r with gaze x | r with gaze y |
|---|---|---|---|---|
| Unprocessed | 1 | 1 | 0.87 | 0.56 |
| SimSwap, Identity A | 0.70 | 0.44 | 0.79 | 0.29 |
| SimSwap, Identity B | 0.70 | 0.44 | 0.83 | 0.28 |
| FAMS | 0.61 | 0.35 | 0.69 | 0.21 |
| DeepPrivacy2 (face and full body) | ≈ 0 | ≈ 0 | 0.06 | 0.05–0.07 |

**Table 9.** Stream isolation: eyes-only and face-only models retrained on each release (five initialisations); penalty against the same model under R→R (cm, mean ± SD).

| Model (real-data error) | Tested on real faces: SimSwap A / B / FAMS | Tested on the release: SimSwap A / B / FAMS |
|---|---|---|
| Eyes-only MobileNet-V3 (4.95) | +2.71 ± 0.93 / +3.56 ± 0.39 / **+1.16 ± 0.10** | **+2.44 ± 0.36 / +1.98 ± 0.22** / +2.72 ± 0.21 |
| Eyes-only MGazeNet (5.14) | +4.79 ± 0.94 / +6.77 ± 0.56 / **+0.96 ± 0.27** | +2.83 ± 0.38 / +2.95 ± 0.12 / **+2.04 ± 0.17** |
| Face-only MobileViT (5.29) | +4.51 ± 0.40 / +3.87 ± 0.59 / **+1.44 ± 0.26** | +3.67 ± 0.16 / +2.98 ± 0.33 / **+2.24 ± 0.29** |

**Releasing eye geometry instead of pixels does not escape the trade-off.** We regressed gaze from FaceMesh landmarks in three nested tiers, with the same participant split and attacks (Table 10). Four iris coordinates per frame predict gaze barely better than the mean (9.02 against 9.53 cm on these frames) yet link the release above the floor (ARI 0.267; 0.441 with 10-frame windows). Adding head position and eyelid shape improves gaze modestly and makes the features strongly linkable. Per-participant calibration lifts an iris-only mapping to about 5.2 cm, but requires every end user to enrol.

**Table 10.** Landmark release: gaze error of gradient-boosted regressors (uncalibrated) and privacy of the released features, with 95% participant-level intervals.

| Released per frame | Dims | Gaze error, uncalibrated | Verification | Linkage, frame | Linkage, 10-frame window |
|---|---|---|---|---|---|
| Iris position in each eye's corner frame | 4 | 9.02 cm | 10.2% [5.7, 15.2] | 0.267 [0.22, 0.48] | 0.441 [0.35, 0.68] |
| + face-box position and size | 7 | 9.27 cm | 62.2% [50.9, 72.2] | 0.665 [0.54, 0.84] | 0.843 [0.71, 0.98] |
| + both eyelid contours | 71 | 8.31 cm | 74.3% [64.0, 82.4] | 0.761 [0.63, 0.89] | 0.914 [0.79, 0.99] |

![Figure 7](figs_core/fig_core3_tradeoff.pdf)

**Figure 7.** Privacy against utility for every protocol. Horizontal: D→R penalty, mean of three architectures. Vertical: ARI at k = 18 of the most linkable released crop (face or eye). Circles: Full Synthetic; squares: Hybrid; star: unprocessed; diamonds: landmark releases. Dashed red: no-information reference; dashed grey: linkage floor. The shaded private-and-useful corner is empty. The only point that leaves the top row (DeepPrivacy2 full body, per-frame identity, ARI 0.297) is at the no-information level (+4.34 cm), and restoring its original eyes (square 13) restores linkage (0.849). FAMS (circle 14) is the full-synthesis release closest to real-data utility (+0.94 cm) because its models transfer to real eyes, not because its pupils carry more gaze (§8.3), and its face crop remains linkable (0.853).

## 8.4 Clinical meaning

Arms are compared uncalibrated, but a deployed screening tool calibrates each user. With per-participant calibration, the four architectures reach 3.70–5.11 cm on real data, or 1.8–3.7° at 80–120 cm. Table 11 sets this against what each biomarker must resolve. Anti-saccade direction, saccade latency (bias −16.8 ms, under one 33 ms frame) and search dispersion are usable; fixation stability and peak saccade velocity are beyond the instrument before de-identification is considered. Calibrated, Hybrid stays in the same regime (2.1–4.0°), whereas Full Synthetic moves to 3.5–8.0°, at which hemifield discrimination is no longer reliably resolved. Webcam eye tracking validated on clinical paradigms reports about 1.42° after per-user calibration [webcamval], the same regime as our calibrated 1.8–3.7°.

**Table 11.** What each biomarker must resolve, against the calibrated error of 1.8–3.7°; requirements from published normative ranges.

| Biomarker | What it must resolve | Requirement | Verdict at 1.8–3.7° |
|---|---|---|---|
| Anti-saccade directional error | which hemifield the gaze went to | coarse: roughly half the stimulus eccentricity | **usable** |
| Saccade latency | movement onset | one frame; 100°/s detection threshold [saccadethresh] | **usable** — measured bias −16.8 ms, under one 33 ms frame |
| Search-task dispersion | spread over tens of cm | coarse | **usable** |
| Fixation stability (BCEA) | healthy median 0.75–1.07 deg², a sub-degree ellipse [bcea] | < 0.5° | **below our noise floor** |
| Peak saccade velocity | a 30–80 ms movement | ≥ 250 Hz sampling | **undersampled at 30 Hz** |

# 9 DISCUSSION

Our claims concern appearance-based gaze data — face and eye crops released for training gaze models — and do not extend to other sensing modalities.

**Gaze utility and identity share the eye region.** Every protocol that preserved utility did so by keeping genuine eye pixels, which identify at 62–80%. Releasing landmarks instead of pixels still links the corpus while carrying almost no gaze signal across participants.

**Re-identification survives on the face crop; linkability survives everywhere.** Synthesised eyes defeat A1 on eye crops, but synthetic face crops from SimSwap and FAMS still verify at 18–24% under ArcFace; only DeepPrivacy2, which repaints the face, reaches the floor. Linkability is harder still: synthetic eye crops verify at up to 63% within the release, face crops partition exactly, and crop geometry and photometry link without any face model. Under European guidance, records that can be singled out remain personal data whether or not anyone is named [wp29anon]. A privacy figure reported without its test therefore cannot tell a dataset author whether a release is safe.

**What a releasable corpus would require.** Content-based linkage falls only when identity is not fixed per participant and the release contains nothing the generator leaves untouched. This matters only for releases that strip participant identifiers, since a usable gaze corpus keeps them. We observed each half separately: FAMS breaks per-participant consistency but leaves the face crop linkable, and full-body inpainting with per-frame identity meets both halves but repaints the eyes. A releasable corpus would need a generator that keeps the pupil at least as faithfully as face swapping, produces eyes from which models transfer to real users as FAMS's do, edits everything the released crop contains, and resamples identity per session rather than per frame, which a single-session corpus cannot verify. We know of no method that meets all four.

**Measure utility the way the release will be used.** R→D conflates domain shift with information loss; retraining separates them. Utility figures for a release should come from D→R, reported beside the no-information reference, together with a direct check of the pupil, because transfer and information can diverge: the release that transfers best carries less pupil signal than the one that transfers worst.

**Releasing gaze data now.** No protocol we measured makes a smartphone gaze corpus anonymous in the European sense: a usable release is linkable by construction, and every gaze-preserving protocol leaves residual re-identification. CognitiveGaze is therefore shared under controlled access, with derived gaze traces in an open tier whose linkage we measured (ARI 0.033). This leaves the cycle in Figure 1 unbroken, but changes how the next dataset author decides: both axes can be measured before release with privaudit, and generator developers know what a gaze-preserving method must achieve (§10).

# 10 DESIGN GUIDELINES FOR RELEASING GAZE DATA

Table 12 lists seven guidelines for authors of appearance-based gaze datasets; the full checklist and audit tool are in the supplementary material [privaudit-repo].

**Table 12.** Design guidelines for releasing gaze data. privaudit implements the checks for G1, G2, G4 and G5.

| # | Guideline | Evidence |
|---|---|---|
| G1 | Run A1 on every released stream, face crop included, and test linkability (A2) as the anonymisation criterion. | Synthetic eye crops: 0.2% under A1, 43% within the release; synthetic face crops: 18–24% under A1. |
| G2 | Measure the floor on your own recordings, using a face-free patch of the same frames. | TAR 1.57%, Rank-1 50.5%, ARI 0.074–0.089. |
| G3 | Do not buy utility by releasing genuine eyes. | Hybrid recovers 85–106% of utility; genuine eye crops verify at 62–80%. |
| G4 | Normalise crop size and per-crop photometry. | Crop size alone identifies 66.7%; photometry alone links at ARI 0.87–0.90. |
| G5 | When stripping identifiers, resample identity and edit everything released. | Only that configuration breaks face-crop linkage (0.297); verification remains 36%. |
| G6 | Report utility under D→R beside the no-information reference, and test the pupil directly. | FAMS transfers best, yet its pupils track gaze less closely than SimSwap's (17/17 participants). |
| G7 | If any check fails, release under controlled access and state which channels leak. | No tested protocol is both private and useful. |

# 11 LIMITATIONS

Participants were recorded in a single session. Photometric normalisation and a large within-session illumination change leave linkage intact, but ambient lighting, camera position and seating cannot be separated from the person without a second session, which also bounds the photometric attack; the longitudinal case that makes linkage consequential remains untested. The cohort of 18 healthy adults is moderately homogeneous and drawn from one institution, so absolute identity rates are cohort-dependent, whereas between-arm comparisons, which hold the cohort fixed, are not; we make no clinical claims. The two recognisers, both trained on web-scale face corpora, can disagree — FaceNet places FAMS face crops at its floor where ArcFace does not — and recognition by acquaintances, who use hairstyle, glasses and face shape that FAMS preserves, was not measured. We test one replacement method with two templates, one removal method in two modes and one diffusion method; latent-space diffusion editing is untested. Full-body DeepPrivacy2 left a second, unmapped person unedited in 1.4% of frames, outside the released crops. iTracker was retrained once per arm, and FAMS on the three faster architectures only.

# 12 CONCLUSION

De-identification of gaze data has been judged by whether the face still looks like the participant. We measured instead whether a release can be re-identified or linked and whether a model trained on it works, across three generator families, two release forms and four train–test conditions. No configuration was both private and useful: protocols that keep gaze keep identity, and the one that breaks linkage leaves nothing to learn from. The two-part condition states what a generator would have to achieve, and privaudit lets dataset authors check a release before sharing it.

# DATA AVAILABILITY

CognitiveGaze contains facial video and is shared under controlled access: researchers apply to the data holder and sign a data-use agreement that prohibits re-identification and linkage. Derived per-frame gaze estimates and oculomotor measures form an open tier, whose residual linkage is reported in §7. The audit tool privaudit, the release checklist and the analysis code are available at [privaudit-repo].

# SUPPLEMENTARY MATERIAL

**Table S1.** R→D for DeepPrivacy2 and FAMS: mean ± SD over five initialisations; in parentheses (FAMS: as listed), the change against R→R (Table 5).

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

| Model | R→R (real data) | Full Synthetic | Hybrid |
|---|---|---|---|
| iTracker | 5.85 ± 0.14 | +5.06 ± 0.17 | **+0.02 ± 0.07** |
| MobileNet-V3 Large | 4.98 ± 0.13 | +4.06 ± 0.24 | **+0.27 ± 0.20** |
| AFFNet | 5.67 ± 0.20 | +4.91 ± 0.20 | **+0.19 ± 0.10** |
| MGazeNet | 4.96 ± 0.04 | +4.94 ± 0.17 | **+0.35 ± 0.21** |

**Table S2.** iTracker: penalty against R→R (5.85 cm); D→R and D→D from one initialisation; predicting the mean corresponds to +3.52 cm.

| Arm | R→D | D→R | D→D |
|---|---|---|---|
| SimSwap A, Full Synthetic / Hybrid | +3.00 / **+0.31** | +7.50 / **−0.10** | +3.01 / **+0.00** |
| SimSwap B, Full Synthetic / Hybrid | +3.62 / **−0.03** | +12.05 / **+0.62** | +4.38 / **+0.43** |
| DeepPrivacy2 fixed, Full Synthetic / Hybrid | +10.58 / **−0.10** | +3.77 / **−0.14** | +3.78 / **−0.07** |
| DeepPrivacy2 per frame, Full Synthetic / Hybrid | +11.65 / **−0.02** | +3.61 / **−0.29** | +3.62 / **−0.28** |
| DeepPrivacy2 full body fixed, Full Synthetic / Hybrid | +10.06 / **+0.16** | +3.77 / **−0.22** | +4.07 / **−0.25** |
| DeepPrivacy2 full body per frame, Full Synthetic / Hybrid | +11.27 / **+0.11** | +3.65 / **−0.16** | +3.71 / **−0.17** |

![Figure S1](figs_core/fig_core2_utility.pdf)

**Figure S1.** Penalty against R→R under R→D (grey), D→R (green) and D→D (purple) for **(a)** Full Synthetic and **(b)** Hybrid. Bars: mean of three architectures; dots: individual architectures; red band: no-information reference; grey band: initialisation noise floor.

![Figure S2](figures/fig3_threat_models.pdf)

**Figure S2.** The same synthetic eye crops under the two tests: TAR at FAR 1e-3 for left and right eye crops, unprocessed and SimSwap (t1 = Identity A, t2 = Identity B). Blue: A1, matched against photographs; orange: within-release verification (A2). Vertical line: eye floor (1.57%). SimSwap drives A1 to the floor (0.2–1.8%) while within-release verification stays at 43–62%.
