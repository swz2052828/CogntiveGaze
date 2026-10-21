# Keep Your Eyes On It: Measuring Gaze Utility and Re-identification Risk in De-identified Smartphone Gaze Data

## ABSTRACT

Smartphone eye tracking can bring oculomotor assessment out of the clinic, but it records video of the participant's face, and this is what prevents the field from sharing data. Face de-identification has been proposed as a remedy and is usually judged by whether the processed face still matches the real one. We ask instead what such a release protects and how useful it remains for training gaze models.

We introduce CognitiveGaze, in which 18 participants performed seven oculomotor tasks while a smartphone and an EyeLink 1000 recorded them simultaneously, so every frame carries physiological ground-truth gaze. We evaluate face swapping, inpainting and diffusion-based de-identification with four gaze architectures, each deployed on and retrained on every release, and with two face recognisers.

Keeping the genuine eye region recovers 82–111% of the accuracy lost to synthesis, yet these eye crops verify identity at 62–80% against a 1.6% floor. The pixels that make gaze estimation work are also the pixels that identify the participant. Synthesising the eyes prevents matching on the eye crops, but synthetic face crops still verify at 18–24%, and every release that retains gaze information still singles out each participant, which under European anonymisation guidance leaves it personal data. Inpainted eyes carry no recoverable gaze. A diffusion anonymiser produces the models that transfer best to real faces (0.4–1.3 cm against 2.5–6.2 cm), although its pupils follow true gaze less closely than face-swapped ones, and stronger settings trade pupil signal for identity protection without reducing linkability. None of the configurations we tested is both private and useful. We describe what a releasable gaze corpus would require and release an audit tool and checklist.

**CCS Concepts:** • **Security and privacy** → **Pseudonymity, anonymity and untraceability**; • **Human-centered computing** → **Empirical studies in ubiquitous and mobile computing**; • **Computing methodologies** → *Biometrics*.

**Additional Key Words and Phrases:** smartphone gaze estimation, face de-identification, re-identification, linkage attack, privacy–utility trade-off, ocular biometrics, dataset release, privacy auditing

# 1 INTRODUCTION

Oculomotor function is a sensitive marker of neurological and cognitive change, but the instruments that measure it well, such as the EyeLink 1000, are largely confined to the clinic. Smartphone gaze estimation can take the measurement to where patients are [krafka16, valliappan20]. The data, however, stay where they were recorded. A smartphone eye tracker films the participant's face, and face video cannot simply be shared as a research artefact.

This produces a cycle in which data remain scarce (Figure 1). Few public corpora record the face under usable conditions, so most studies build on the same ones, GazeCapture above all [krafka16], and inherit their assumptions: a hand-held phone, a small screen, a short viewing distance and the position of an on-screen target used in place of measured gaze. Newer corpora follow the same protocol for comparability [zjugaze, dynamicgaze]. Clinical protocols differ on most of these points. Stimuli span a large display viewed from about a metre, and the quantity of interest is the gaze trajectory. Clinical groups that collect such data usually keep it in separate silos, so analysis choices are rarely checked against a common reference [simmons11, carp12]. Releasing the data then runs into privacy constraints. Face video of a study participant is personal data, and in a clinical study it is also health data under the GDPR [gdpr]. HIPAA lists full-face photographs among the identifiers that must be removed [hipaa]. The usual workarounds do not produce a usable public corpus either. Blurring or masking removes the geometry a gaze model relies on [blurmask, disguises], homomorphic encryption and federated learning keep the data with its holder and are expensive for video [acar18he, kairouz21fl], and controlled access limits who can use the data. The next group then faces the same shortage.

![Figure 1](figs_core/fig_cycle.pdf)

**Figure 1.** The data-scarcity cycle in smartphone gaze research. A de-identification method that removed identity while keeping gaze would break the cycle at step 5. This paper tests whether current methods do.

A transformation that removed identity while keeping the gaze signal would allow open release, and generative de-identification is the obvious candidate. Face swapping is designed to preserve head pose, eyelid aperture and iris position, which are what a gaze model uses, and it has been applied to clinical video of movement disorders, autism assessments and seizures [zhu20deepfakes, disguises, li24seizure]. Gaze is not well preserved unless the generator is explicitly constrained to keep it [wilson23gaze], and a recent evaluation concluded that face swapping hides identity effectively [mustu25]. These evaluations ask whether the swapped face can still be matched to the real one.

For gaze data we argue that this question is not sufficient, for two reasons. First, the eye region is itself a biometric. A 120 px eye crop verifies identity at 62–80% at a false-accept rate of 1e-3, against a measured floor of 1.6%, so any protocol that recovers accuracy by keeping the genuine eye pixels also releases this biometric. Second, synthesising the whole face breaks the link between the released eye crops and the real face but not between the synthetic face crop and the real face, and the corpus can still be partitioned by individual. Under European anonymisation guidance this is enough for the data to remain personal data [wp29anon]. The cues that allow this lie outside the region the generators edit.

To measure these effects we recorded CognitiveGaze. Eighteen participants performed seven oculomotor tasks while a smartphone and an EyeLink 1000 recorded them simultaneously, so every frame carries physiological ground-truth gaze instead of a target label. The protocol deliberately departs from that of GazeCapture, with stimuli on a 53 × 30 cm display at 80–120 cm, and it is the kind of clinical corpus that usually stays unreleased. For privacy we follow common practice in security research. We state the attacker, use recognisers that were never trained on our data, cross-check with a second recogniser, and measure what "no identity signal" scores on single-session recordings.

This paper makes four contributions.

- **A privacy evaluation protocol for gaze corpora.** We distinguish re-identification from photographs (A1) from a test of whether the released content singles out individuals (A2), which corresponds to the linkability criterion in European anonymisation guidance. The two can differ by up to 60 points on the same crops. We release the protocol as an audit tool with a fifteen-item checklist [privaudit-repo], which reproduces every eye-crop result in this paper.
- **Evidence that gaze utility and identity share the eye region.** Keeping the genuine eye crops recovers 82–111% of the synthesis penalty across architectures and conditions, and the same crops identify participants at 62–80%. Releasing eye landmarks instead of pixels still allows linkage while carrying almost no gaze signal.
- **A tested two-part condition for content-based linkage.** Linkage from the released content disappears only when the synthetic identity is not held fixed per participant and the release contains nothing outside the generator's edit region. We confirm this in a 2×2 experiment with DeepPrivacy2 and show that a diffusion method, which played no part in deriving the condition, behaves as it predicts.
- **A utility benchmark under de-identification.** We cover four architectures, three generator families and four train–test conditions, with an initialisation noise floor, a no-information reference, stream-isolated models and a model-free pupil test. The release whose models transfer best to real users turns out not to be the one whose pupils carry the most gaze information.

# 2 RELATED WORK

**Smartphone gaze estimation.** iTracker and the GazeCapture corpus established appearance-based gaze estimation on phones [krafka16], and Valliappan et al. reached research-grade accuracy with per-user calibration [valliappan20]. AFFNet [affnet] and MGazeNet [zhu24] condition the eye features on facial context through adaptive normalisation. All of these models take the same inputs (a face crop, two eye crops and a face grid) and therefore face the same data-sharing problem.

**Face de-identification.** Methods that remove identity include k-Same and its neural successors [newton05, ksamenet], DeepPrivacy and DeepPrivacy2, which inpaint the face or the whole body [deepprivacy, deepprivacy2], CIAGAN [ciagan], and diffusion methods that either condition on the source face [fams] or edit in latent space [fluid, apl]. Face-swapping methods such as SimSwap and SimSwap++ replace identity instead [simswap, simswappp]. One would expect inpainting the eyes to destroy gaze and swapping to preserve it. We test both expectations and find that, for privacy, what the edit region leaves untouched matters more than the training objective.

**Face swapping for clinical video.** Face swapping has been applied to videos of movement disorders, where the body keypoints needed for diagnosis were preserved [zhu20deepfakes], and to seizure videos, which were rated subjectively for retained semiology [li24seizure]. Wilson et al. proposed the first privacy–utility methodology for face swaps in patient video, comparing gaze and expression preservation with blurring [disguises], and later added a gaze loss because full-face objectives did not preserve the eye region [wilson23gaze]. A digital mask for clinical video [digitalmask] was soon questioned [maskconcerns]. Muştu and Ekenel concluded that face swapping hides identity effectively, based on matching swapped faces to real ones [mustu25]. Our results agree for synthesised eye crops but not for synthetic face crops. Their test also does not ask whether the release singles out individuals, which is the linkability criterion in European guidance [wp29anon] and which face swapping leaves almost intact.

**Eye movement as a biometric.** Identity can be recovered from eye movements alone [rigas, holland, bioeye]. Privacy research on eye tracking has proposed differential privacy for gaze streams [bozkir, steil], learned obfuscation [fuhl] and private scanpath comparison [scanpath], reviewed in [etprivreview] and, for mobile devices, in [alsakar25]. For this reason we measure the risk of a release tier without images rather than assume it is safe.

# 3 THE COGNITIVEGAZE DATASET

**Participants.** We recruited 23 volunteers (13 male, 10 female). Five were excluded before analysis because of calibration failure or incomplete recordings, which left 18 participants aged 20 to 54 (median 29). Twelve described themselves as White/Caucasian, two as East Asian, and one each as Black/African, Middle Eastern, Asian-Indian and mixed background. People with photosensitive migraine, epilepsy, motion sickness or substance use affecting oculomotor function were not included. Three participants wore glasses, and we kept them because many people need vision correction to complete a cognitive protocol. One participant has no usable ground-truth recording, so the utility analyses use 17 participants and the privacy analyses use 18. The study was approved by the institutional ethics committee (reference withheld for review), and all participants gave written informed consent.

**Apparatus and tasks.** Participants sat 80–120 cm from a 53.3 × 29.6 cm display (1920 × 1080 px) without head restraint (Figure 2). An EyeLink 1000 recorded gaze at 1000 Hz after a nine-point calibration, and a smartphone (Xiaomi Mi 8) mounted below the display filmed the face with its rear camera at 30 fps. The task battery was designed with rehabilitation specialists, following [valliappan20], and consisted of fixation stability, prosaccades, anti-saccades, smooth pursuit, optokinetic nystagmus, free viewing of 30 OSIE scenes [osie] and visual search in 12 of these scenes. All tasks were presented with PsychoPy. Blink frames were kept and annotated.

**Synchronisation.** Each block began with a 0.6 s tone, which the phone recorded and the experiment software timestamped. We use the detected onset to align video frames with EyeLink time and resample the ground truth to 30 Hz, so that each frame carries a gaze coordinate.

![Figure 2](FIGURE: original setup figure)

**Figure 2.** Data acquisition. (A) Participant station with the display at 80–120 cm, the smartphone below it and the EyeLink 1000 recording ground truth. (B) Control workstation.

# 4 GAZE ESTIMATION MODELS

All models take the left-eye, right-eye and face crops (224 × 224) together with a 25 × 25 face grid and regress the gaze point on the display. We evaluate four published architectures (Table 1, Figure 3). Two fuse the streams late, and two use a guidance factor regressed from the face to modulate the eye features. Models are trained with a mean-squared-error loss, and we report the mean Euclidean error in centimetres on the display. We use participant-wise five-fold cross-validation [kfold], re-initialise the weights for every fold and keep the checkpoint with the lowest validation error. Every configuration is trained from five independent initialisations, and their spread defines the noise floor for all utility comparisons. We compare arms on the uncalibrated model. For the clinical analysis (§8.4) we also fine-tune the regression head for each participant on calibration frames recorded before the tasks. A face-only model, which never sees the eye crops, serves as a null control.

**Table 1.** The four gaze architectures.

| Model | Streams | Fusion |
|---|---|---|
| iTracker [krafka16] | shared eye CNN, face CNN, grid MLP | late concatenation |
| MobileNet-V3 Large [mobilenetv3] | MobileNet-V3 backbones, grid encoding | late concatenation |
| AFFNet [affnet] | CNNs with SE blocks, face-regressed guidance factor | adaptive group normalisation of eye features |
| MGazeNet [zhu24] | CNNs with SE blocks, face-regressed guidance factor | linear adaptive batch normalisation (adapted to 224 px) |

![Figure 3](FIGURE: original architecture comparison figure)

**Figure 3.** (A) Layer-level comparison of the four architectures. (B) Workflow from ROI extraction to supervised training against EyeLink ground truth.

# 5 DE-IDENTIFICATION METHODS

We test three families of methods that differ in what they edit and how they handle identity. Each method runs on the original frame, and we cut the released crops from the processed frame using the stored per-frame geometry, so crops are pixel-aligned across arms.

**Replacement (SimSwap++).** An adapted SimSwap++ [simswappp] transfers a source identity onto the participant's face and preserves pose, expression and gaze direction. Each participant is mapped to the same synthetic appearance in every frame. We use two source templates with different appearance (Identity A and B, Figure 4) to separate the method from the choice of template, and refer to this family as SimSwap below.

**Removal (DeepPrivacy2).** DeepPrivacy2 [deepprivacy2] inpaints either the detected face box (face mode) or the whole person, guided by a dense surface map (full-body mode). The latent code is either fixed per participant or drawn anew for each frame, with truncation set to 1.0.

**Diffusion (FAMS).** Face Anonymization Made Simple [fams] conditions a diffusion model on the source face and pastes an aligned square covering the face, forehead and part of the hair back into the frame. We use the authors' setting (anonymisation degree 1.25, 25 steps, guidance 4.0) with the noise seed either fixed per participant or drawn per frame, on all 210,924 task frames. We also sweep the degree from 1.35 to 3.0 on a privacy sample (§7).

**Release protocols.** In *Full Synthetic*, every crop is taken from the de-identified frame. In *Hybrid*, the de-identified face crop is combined with the original eye crops (Figure 5), which tests whether the face can be de-identified while the eyes remain genuine. Section 7 shows that Hybrid preserves utility but not privacy.

![Figure 4](FIGURE: original face-swap examples figure)

**Figure 4.** SimSwap++ with two source templates (Identity A and B). Identity is replaced, while pose, expression and surroundings are preserved.

![Figure 5](FIGURE: original Hybrid pipeline figure)

**Figure 5.** Release protocols. (A) Full Synthetic, with all crops taken from the de-identified frame. (B) Hybrid, combining the de-identified face crop with the original eye crops.

# 6 THREAT MODEL AND PRIVACY EVALUATION

**Released data.** A release contains the cropped streams the models use (the face crop, the two eye crops and the face grid) and not the video. All attacks operate on these crops.

**Attacks.** We treat re-identification from photographs (A1) as the primary threat. The attacker holds photographs of candidate individuals, for example members of the recruitment pool, and asks which of them a released crop shows. The gallery consists of original crops from the first half of each recording and the probes are released crops from the second half. The second test (A2) concerns a property of the data, namely whether the content alone allows frames to be grouped by individual. It corresponds to two of the three criteria the Article 29 Working Party uses to assess anonymisation, singling out and linkability [wp29anon], and records that can be isolated per person remain personal data even when no name is attached [gdpr]. A usable gaze corpus has to keep each participant's frames together for participant-wise splits and calibration, so it is linkable by construction. A2 asks whether the content would restore this grouping if the identifiers were removed. On the same crops the two tests can differ by 60 points (Figure S2).

**Recognisers.** We use ArcFace (InsightFace buffalo_l, w600k_r50) without adaptation and compare embeddings to per-participant templates by cosine similarity. The key measurements are repeated with FaceNet (Inception-ResNet-v1 trained on VGGFace2), which differs from ArcFace in backbone, training data and objective. Each recogniser is compared against its own floor (1.57% for ArcFace and 3.52% for FaceNet). Faces are detected and aligned on the original frame, so a method cannot appear private simply by defeating the detector.

**Metrics.** For A1 we report the true-accept rate (TAR) at a false-accept rate (FAR) of 1e-3. Closed-set Rank-1 is not informative on these data, as explained below. For A2 we report the adjusted Rand index (ARI) of a k-means partition at the true k = 18, the k chosen without supervision by the silhouette score, and verification within the release, with templates from the first half and probes from the second half of each recording.

**Session-nuisance floor.** Each participant was recorded in a single session, so lighting, seating and background are constant for each person but differ between people. Running every attack on a 120 px face-free patch of the same frames gives a Rank-1 of 50.5% (chance is 5.6%), a TAR of 1.6% and an ARI of 0.089 (0.074 for the eye pipeline), with k = 2 selected. Rank-1 is therefore uninformative here, and we read TAR and ARI against these floors.

**Uncertainty.** All privacy intervals are 95% participant-level cluster bootstraps. For verification we keep the operating threshold estimated on the full data and resample participants' genuine scores (2,000 replicates), because re-estimating a 1-in-1,000 impostor quantile in each resample biases the interval upward. For linkage we resample participants, give duplicated participants a shared label and refit (1,000 replicates).

**Tooling.** All attacks, floors and intervals are implemented in privaudit, which maps each result onto the release checklist. Run on the embeddings saved in this study, it reproduces all 32 eye-crop verification and linkage values exactly. When the crops are re-embedded from scratch, all 46 audited values agree within GPU inference noise.

# 7 IDENTITY LEAKAGE UNDER DE-IDENTIFICATION

Table 2 summarises the results for SimSwap under both tests, Table 3 reports face-crop A1 for every family and Figure 6 shows the linkage results.

**Eye crops.** An unprocessed 120 px eye crop verifies at 66.6% (left) and 62.2% (right) with ArcFace, and at 69.1% and 81.2% with FaceNet, against floors of 1.6% and 3.5%. Hybrid releases these crops unchanged and therefore leaks the same information. The component that restores utility in Hybrid is also the one that exposes the eye biometric.

**Synthesised eyes and faces.** When the eyes are synthesised as well, eye-crop A1 drops to 0.2% and 1.8% (FaceNet 0.0% and 1.5%), which is at the floor for both templates and both recognisers. The synthetic face crop is a different matter. With ArcFace it still verifies at 18.4% [5.1, 35.4] for Identity A and 22.0% [6.8, 40.7] for Identity B. FaceNet agrees for Identity A (14.4% against its 3.5% floor) but not for Identity B (0.2%). With 18 participants the intervals support a statement that the face crop is above the floor and an order of magnitude below unprocessed imagery, and nothing finer.

**Linkage within the release.** An unsupervised partition of the face crops recovers the participants exactly (ARI 1.000 at k = 18, floor 0.089) for both templates, because each participant is always mapped to the same synthetic appearance. Synthetic eye crops that reach only 0.2% under A1 verify at 42.8% and 62.5% within the release. Linkage also survives in the 120 px eye crop, which excludes hair, clothing and the face outline, with ARI 0.68–0.79 compared with 0.84–0.85 for unprocessed eyes and overlapping intervals (for example [0.59, 0.86] and [0.70, 0.96]).

@@TABLE2@@

**Illumination and photometry.** Removing per-crop brightness and contrast does not reduce eye-crop linkage (ARI 0.684 to 0.742), local histogram equalisation lowers it by only 0.05–0.14, and a large change in illumination within the session (dark-screen against natural-image blocks) does not lower it at all. The recogniser therefore does not depend on photometry. Photometry on its own also links, however. Twenty-one photometric statistics per crop group the participants without any face model (ARI 0.87–0.90), and identity replacement leaves them unchanged. The two channels are independent, so normalising photometry removes only the second one. On single-session data the photometric channel also mixes properties of the person with properties of the recording (§11). Crop geometry provides a third channel, since face and eye crop sizes alone identify 66.7% of participants and 94.4% once inter-ocular distance is added. Our release uses fixed crop sizes.

**A two-part condition, tested on DeepPrivacy2.** DeepPrivacy2 removes identity instead of replacing it. In face mode it edits 40% more of the crop than SimSwap, and in full-body mode it repaints the whole person, a mode motivated by the concern that face anonymisation leaves identifiers outside the face [deepprivacy2]. We cross the edit region with the identity regime (Figure 6a). Face-crop linkage is 1.000 with face editing and a fixed identity, 0.982 when either the region is widened or the identity is redrawn per frame, and 0.297 only when both are changed. An attacker who has to estimate the cohort size then finds no structure (k = 2, ARI 0.000). Verification falls from 89.7–100% to 35.7% [27, 45], which is still far above the 1.6% floor. The condition thus prevents an attacker from discovering the partition but not from verifying identities with a few labelled frames. The fixed-identity full-body arm has the same low resolution and still links at 0.982, so the drop comes from the identity regime and not from image quality. On the eye crop, which contains only edited pixels, drawing a new identity per frame is enough on its own (ARI 0.805 to 0.148, floor 0.074). Under A1 every DeepPrivacy2 arm is at the floor for eye crops (at most 1.2%) and face crops (at most 1.6%, Table 3). This agrees with DeepPrivacy2's own evaluation, which matches enrolment images to the release and does not test linkage.

**Diffusion-based de-identification.** In FAMS, fixing the noise seed per participant does not fix the synthetic identity, which follows the source frame, and the two seed regimes are indistinguishable on every measure. FAMS therefore breaks per-participant consistency by design. Eye-crop linkage roughly halves (within-release verification of 5–8% against 43–63% for SimSwap, and ARI 0.29–0.33 against 0.68–0.80), but all intervals remain above the eye floor. The face crop, whose border, pose and geometry lie outside the aligned square that FAMS edits, remains linkable (ARI 0.85–0.94). It also still verifies against the participant's photographs at 24.1% [15.3, 33.6] with ArcFace, no lower than SimSwap, whereas FaceNet places it close to its floor (5.9% [2.2, 10.5], Table 3).

@@TABLE3@@

**Anonymisation strength.** Raising the FAMS degree from 1.25 to 1.5 reduces ArcFace face verification from 24.1% to 10.1% [3.0, 19.3], and FaceNet reaches its floor from 1.35 onwards. Degree 1.5 is the strongest setting at which the generator still produces a face in more than 90% of frames, and beyond it the output mostly degrades. Face-crop linkage does not change (ARI 0.81–0.90), and the correlation between the synthetic pupil and true horizontal gaze drops from 0.68 to 0.53 (Table 4, §8.1). Stronger settings therefore buy identity protection with pupil signal and leave linkability where it was.

@@TABLE4@@

**Other residual risks.** An adaptive attacker who learns a de-anonymisation mapping from other participants' paired images gains at most 3.3 points on the face crop and nothing on the eye crops. When the target is included in the fit, the same mappings reach 100% Rank-1, which indicates that they memorise identities and do not invert the method. Our open tier, which releases per-frame gaze estimates and oculomotor measures without any images, links at an ARI of 0.033 at the correct k (purity 0.157 against 0.059 by chance, with a permutation-null maximum of 0.0015). This risk is low but not zero.

![Figure 6](figs_core/fig_core1_privacy.pdf)

**Figure 6.** Content-based linkage. **(a)** DeepPrivacy2 face crops with the edit region (face box or whole person) crossed with the identity regime (fixed or per frame). Cells give the ARI at k = 18 and within-release verification (TAR at FAR 1e-3 with 95% interval). Linkage falls only in the cell that meets both parts of the condition (ARI 0.297), and verification there is still 36%. **(b)** Eye-crop linkage for every arm with 95% participant-level intervals, with the eye floor (0.074) as a dashed line. SimSwap and fixed-identity DeepPrivacy2 overlap with unprocessed eyes, per-frame DeepPrivacy2 approaches the floor (0.12–0.17), and FAMS lies in between (0.29–0.33) under both seed regimes.

# 8 GAZE UTILITY UNDER DE-IDENTIFICATION

## 8.1 Protocol

**Train–test conditions.** We name each condition by what the model was trained and tested on, with R for real and D for de-identified data.

- **R→R.** Trained and tested on real data. Every penalty below is measured against this reference.
- **R→D.** Trained on real data and tested on de-identified crops, which is the cost for a pipeline that receives de-identified input.
- **D→R.** Retrained on the release and tested on real faces, as a third party would do when deploying a model on its own users.
- **D→D.** Retrained and tested on the release.

Retraining uses the R→R recipe, participant partition and hyperparameters. Comparing R→D with D→R separates domain shift, which retraining removes, from information loss, which it cannot recover.

**References.** The initialisation noise floor is the spread of R→R over five initialisations (0.04–0.20 cm). The no-information reference is the error obtained by always predicting the training participants' mean gaze point. It is 9.37 cm, a penalty of +3.70 to +4.41 cm depending on the architecture.

**Pupil test.** End-to-end error confounds the gaze information contained in a release with how well models trained on it transfer. We therefore also measure the pupil directly. FaceMesh [facemesh] locates both iris centres on the released and the original face crop of each frame, and we express each centre in the coordinate frame defined by the corners of its eye, which removes head pose and crop placement. We then compute, within each participant, the correlation of the released pupil position with the original position and with EyeLink gaze, on 4,680 frames shared by all releases.

## 8.2 Models trained on real data (R→D)

@@TABLE5@@

Restoring the genuine eye crops recovers most of the synthesis penalty. Full Synthetic costs +2.41 to +7.04 cm depending on the model and template, whereas Hybrid leaves +0.08 to +1.04 cm, which corresponds to a recovery of 85–101% (Table 6). The paired difference is negative for every architecture, template and draw. The remaining gap is small but mostly real, since six of the eight Hybrid arms are measurably worse than R→R, so we describe Hybrid as recovering accuracy, not preserving it. The face-only control, which cannot see the eye crops, gives the same result for both arms (within 0.001 cm), which confirms that the comparison isolates the eye stream. The guidance-factor architectures recover as much as the late-fusion ones (92% and 98% for AFFNet). The synthesis penalty depends strongly on the template (Identity B minus Identity A is +0.63 to +3.38 cm for Full Synthetic and −0.34 to +0.86 cm for Hybrid).

Results for DeepPrivacy2 and FAMS under R→D are given in Table S1. Real-data models deployed on DeepPrivacy2 crops do worse than predicting the mean (14.6–20.5 cm against 9.37 cm), and FAMS costs +4.1 to +5.1 cm, more than SimSwap A (+2.4 to +3.7 cm). Hybrid recovers 91–101% for DeepPrivacy2 in face mode, 93–99% in full-body mode and 93–100% for FAMS. In all 421,848 full-body frames the participant was repainted. In 1.4% of these frames a second person region without a surface map was left unedited, but it lay outside the released crops. Before computing any DeepPrivacy2 or FAMS result, we re-ran the real-data arm through the same pipeline, and it reproduced Table 5 to within 0.0012 cm on all 100 folds.

@@TABLE6@@

## 8.3 Models retrained on the release (D→R and D→D)

@@TABLE7@@

**Face swapping.** Retraining does not remove SimSwap's penalty, which stays at +2.2 to +3.1 cm on the release and +2.5 to +6.2 cm on real faces, where two of the three architectures reach the no-information level. On the release the penalty is positive in 148 of 150 fold-level cells. iTracker trained on SimSwap data does worse than predicting the mean on real faces (+7.5 and +12.1 cm, Table S2), even though it fits its synthetic training set normally.

**Inpainting.** For DeepPrivacy2, 56–65% of the R→D penalty is domain shift, but what remains after retraining is the no-information level in both modes, both identity regimes and both test sets. Of 600 fold-level errors, 580 are at or above the mean-prediction error and none is more than 0.17 cm below it. Drawing a new identity per frame changes utility by only −0.63 to +0.19 cm, which shows that the gaze is lost because the eyes are repainted and not because of the identity regime.

**Hybrid.** Hybrid recovers the penalty in every condition, with 91–103% in D→R and 98–106% in D→D for SimSwap and DeepPrivacy2. Every retrained Hybrid cell lies within −0.19 to +0.23 cm of R→R, which is inside the noise floor or at its edge (−0.14 to +0.07 cm for FAMS).

**Diffusion-based de-identification.** Models retrained on FAMS and deployed on real faces lose +0.94 cm on average (+0.39 to +1.29), compared with +3.83 cm for SimSwap A and +4.51 cm for DeepPrivacy2, and FAMS is better than both in all 15 paired draws. Two measurements suggest that this advantage does not come from more gaze information in the eyes. In the pupil test (Table 8), SimSwap pupils follow true horizontal gaze more closely than FAMS pupils in all 17 participants with ground truth (r = 0.79–0.83 against 0.69, with 0.87 for unprocessed data). Eyes-only models (Table 9) reproduce the FAMS advantage on real faces (+0.96 to +1.16 cm against +2.71 to +6.77 cm) but not within the release, where the two methods perform similarly and each is better for one of the two architectures in every draw. The difference therefore lies in transfer. Models trained on SimSwap eyes fail on real eyes and models trained on FAMS eyes do not. One plausible explanation, which we did not test, is diversity, because SimSwap maps every participant onto the eyes of a single template while FAMS draws a new identity for each frame. When tested on its own release, a FAMS-trained model is less accurate than on real faces (+1.56 to +2.57 cm), which is the opposite of what we observe for SimSwap.

@@TABLE8@@

@@TABLE9@@

**Landmark release.** We also regressed gaze from FaceMesh landmarks in three nested tiers, using the same participant split and attacks (Table 10). Four iris coordinates per frame predict gaze only slightly better than the mean (9.02 against 9.53 cm on these frames), yet they still link the release above the floor (ARI 0.267, or 0.441 with 10-frame windows). Adding head position and eyelid shape improves gaze prediction modestly and makes the features strongly linkable. Per-participant calibration brings an iris-only mapping to about 5.2 cm, but this requires every end user to enrol.

@@TABLE10@@

![Figure 7](figs_core/fig_core3_tradeoff.pdf)

**Figure 7.** Privacy against utility for every protocol. The horizontal axis is the D→R penalty averaged over three architectures, and the vertical axis is the ARI at k = 18 of the most linkable released crop (face or eye). Circles show Full Synthetic, squares Hybrid, the star unprocessed data and diamonds the landmark releases. The red dashed line marks the no-information reference and the grey dashed line the linkage floor. No protocol falls in the shaded private-and-useful corner. The only point below the top row (DeepPrivacy2 full body with per-frame identity, ARI 0.297) is at the no-information level (+4.34 cm), and restoring its original eyes (square 13) also restores linkage (0.849). FAMS (circle 14) is the full-synthesis release closest to real-data utility (+0.94 cm), because its models transfer to real eyes rather than because its pupils carry more gaze (§8.3), and its face crop remains linkable (0.853).

## 8.4 Clinical meaning

We compare arms without calibration, but a deployed screening tool would calibrate for each user. With per-participant calibration the four architectures reach 3.70–5.11 cm on real data, or 1.8–3.7° at 80–120 cm. Table 11 compares this with what each biomarker needs to resolve. Anti-saccade direction, saccade latency (bias of −16.8 ms, less than one 33 ms frame) and search dispersion can be measured, but fixation stability and peak saccade velocity are beyond the instrument even before de-identification. After calibration, Hybrid stays in the same range (2.1–4.0°), whereas Full Synthetic moves to 3.5–8.0°, at which the hemifield of an anti-saccade may no longer be resolved reliably. Webcam eye tracking validated on clinical paradigms reports about 1.42° after per-user calibration [webcamval], which is in the same range as our calibrated 1.8–3.7°.

@@TABLE11@@

# 9 DISCUSSION

Our conclusions concern appearance-based gaze data, that is, face and eye crops released for training gaze models, and they do not extend to other sensing modalities.

**The eye region.** Every protocol that preserved utility did so by keeping genuine eye pixels, and these identify participants at 62–80%. Releasing landmarks instead of pixels still allows linkage while carrying almost no gaze signal across participants.

**Re-identification and linkability.** Synthesised eyes prevent A1 on the eye crops, but the synthetic face crops of SimSwap and FAMS still verify at 18–24% with ArcFace, and only DeepPrivacy2, which repaints the face, reaches the floor. Linkability is harder to remove. Synthetic eye crops verify at up to 63% within the release, face crops partition exactly, and crop geometry and photometry link participants without any face model. Under European guidance, records that can be singled out remain personal data whether or not anyone is named [wp29anon]. A privacy figure reported without the test behind it therefore says little about whether a release is safe.

**Requirements for a releasable corpus.** Linkage from the released content disappears only when identity is not fixed per participant and the release contains nothing the generator leaves untouched. This matters only for releases that remove participant identifiers, since a usable gaze corpus has to keep them. We observed each part of the condition separately. FAMS breaks per-participant consistency but leaves the face crop linkable, and full-body inpainting with per-frame identity meets both parts but repaints the eyes. A releasable corpus would need a generator that keeps the pupil at least as faithfully as face swapping, produces eyes from which models transfer to real users as well as FAMS does, edits everything in the released crop, and draws a new identity per session rather than per frame. The last requirement cannot be verified on a single-session corpus. We are not aware of any method that meets all four.

**Measuring utility.** R→D mixes domain shift with information loss, and retraining separates the two. Utility for a release should therefore be reported under D→R together with the no-information reference. The pupil should also be checked directly, because transfer and information can diverge, as the release that transfers best carries less pupil signal than the one that transfers worst.

**Releasing gaze data today.** None of the protocols we measured makes a smartphone gaze corpus anonymous in the European sense. A usable release is linkable by construction, and every protocol that preserves gaze leaves some re-identification risk. We therefore share CognitiveGaze under controlled access, with derived gaze traces in an open tier whose linkage we measured (ARI 0.033). This does not break the cycle in Figure 1, but it changes the information available to the next dataset author, who can measure both privacy and utility with privaudit before release. It also gives developers of de-identification methods a concrete target (§10).

# 10 DESIGN GUIDELINES FOR RELEASING GAZE DATA

Table 12 lists seven guidelines for authors of appearance-based gaze datasets. The full checklist and the audit tool are provided as supplementary material [privaudit-repo].

@@TABLE12@@

# 11 LIMITATIONS

Each participant was recorded in a single session. Photometric normalisation and a large illumination change within the session leave linkage intact, but ambient lighting, camera position and seating cannot be separated from the person without a second session. This also limits how far the photometric attack can be interpreted, and the longitudinal setting in which linkage matters most remains untested. The cohort of 18 healthy adults is fairly homogeneous and comes from a single institution, so absolute identification rates depend on the cohort, whereas comparisons between arms, which keep the cohort fixed, do not. We make no clinical claims. The two recognisers are both trained on web-scale face data and can disagree, as FaceNet places FAMS face crops at its floor while ArcFace does not. We did not measure recognition by acquaintances, who rely on hairstyle, glasses and face shape, all of which FAMS preserves. We tested one replacement method with two templates, one removal method in two modes and one diffusion method, and did not test latent-space diffusion editing. Full-body DeepPrivacy2 left a second, unmapped person unedited in 1.4% of frames, outside the released crops. Finally, iTracker was retrained only once per arm, and FAMS only on the three faster architectures.

# 12 CONCLUSION

De-identification of gaze data has so far been judged by whether the face still looks like the participant. We measured instead whether a release can be re-identified or linked and whether a model trained on it still works, across three generator families, two release forms and four train–test conditions. None of the configurations was both private and useful. The protocols that keep gaze information also keep identity, and the one configuration that breaks linkage leaves nothing to learn from. The two-part condition specifies what a generator would need to achieve, and privaudit allows dataset authors to check a release before they share it.

# DATA AVAILABILITY

CognitiveGaze contains facial video and is available under controlled access. Researchers apply to the data holder and sign a data-use agreement that prohibits re-identification and linkage. Derived per-frame gaze estimates and oculomotor measures form an open tier, whose residual linkage is reported in §7. The audit tool privaudit, the release checklist and the analysis code are available at [privaudit-repo].

# SUPPLEMENTARY MATERIAL

@@TABLES1@@

@@TABLES2@@

![Figure S1](figs_core/fig_core2_utility.pdf)

**Figure S1.** Penalty against R→R under R→D (grey), D→R (green) and D→D (purple) for **(a)** Full Synthetic and **(b)** Hybrid. Bars show the mean over three architectures and dots the individual architectures. The red band marks the no-information reference and the grey band the initialisation noise floor.

![Figure S2](figures/fig3_threat_models.pdf)

**Figure S2.** The same synthetic eye crops under the two tests. TAR at FAR 1e-3 for left and right eye crops, unprocessed and with SimSwap (t1 is Identity A and t2 is Identity B). Blue points show A1, matched against photographs, and orange points show verification within the release (A2). The vertical line marks the eye floor (1.57%). SimSwap brings A1 down to the floor (0.2–1.8%), while verification within the release stays at 43–62%.
