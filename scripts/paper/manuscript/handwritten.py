DATASET = r"""
# 3 THE COGNITIVEGAZE DATASET

CognitiveGaze was recorded to benchmark smartphone gaze estimation against a physiological ground truth rather than against the position of an on-screen target.

**Participants.** Twenty-three volunteers (13 male, 10 female) were recruited. Five were excluded before analysis for calibration failure or incomplete data acquisition, leaving a final cohort of N = 18. Inclusion required no history of photosensitive migraine, epilepsy, motion sickness, or substance use affecting oculomotor function. The final cohort spanned 20 to 54 years (median 29) and comprised White/Caucasian (n = 12), East Asian (n = 2), Black/African (n = 1), Middle Eastern (n = 1), Asian-Indian (n = 1) and mixed-background (n = 1) participants. Three wore glasses. Unlike many open gaze corpora, we did not exclude spectacle wearers, because vision correction is frequently a prerequisite for completing a cognitive protocol and excluding it would compromise clinical generalisability. One participant has no usable ground-truth gaze recording; utility analyses therefore use 17 participants and privacy analyses all 18. The protocol was approved by the institutional research ethics committee (reference withheld for review), and written informed consent was obtained after a briefing on the protocol and its risks.

**Apparatus.** Participants sat 80–120 cm from a 53.3 × 29.6 cm display (1920 × 1080 px) without head restraint (Figure 2). Ground-truth gaze was recorded at 1000 Hz by an EyeLink 1000, calibrated with a standard nine-point sequence; a target sticker on the forehead supported head tracking. A smartphone (Xiaomi Mi 8) was mounted directly below the display, and its rear camera (12 MP, 30 fps) recorded the participant's face. Stimuli were programmed in PsychoPy. Blinks and tracking losses appear as missing samples in the ground truth; unlike the GazeCapture protocol, blink frames were retained and annotated by a blink detector, because blink behaviour is itself a cognitive marker.

**Tasks.** The battery was designed with rehabilitation specialists and informed by established smartphone eye-tracking protocols [CITE: valliappan20]. After calibration, participants completed seven oculomotor tasks: *fixation stability* on a central cross; *prosaccades* to targets at the four screen corners and then five random locations; *anti-saccades*, in which participants suppressed the reflexive saccade to a red target at one of eight peripheral positions and looked instead to its mirror location; *smooth pursuit* of a target moving horizontally at constant speed along a triangular waveform; *optokinetic nystagmus*, tracking one stripe of a drifting vertical grating until it left the screen; *free viewing* of 30 natural scenes from the OSIE dataset [CITE: osie]; and *visual search* for a named target in 12 of those scenes. Each block was preceded by a 10–20 s instruction screen, rest breaks were interspersed between blocks, and participants were familiarised with the instructions beforehand.

**Synchronisation.** Participant and investigator kept silent during trials, because sound is the synchronisation channel. A 0.6 s tone preceded each task block; it was recorded by the smartphone microphone and timestamped by the experiment software. Detecting its onset in the smartphone audio gives a shared anchor t₀, against which video frame indices are mapped to EyeLink time. The 1000 Hz ground truth is then resampled to the 30 Hz video, so that every smartphone frame carries a synchronised ground-truth gaze coordinate (x, y).

![Figure 2](FIGURE: original setup figure)

**Figure 2.** Data acquisition setup. (A) Participant station: the participant sits 80–120 cm from the stimulus display without head restraint; the smartphone is mounted below the display and the EyeLink 1000 records ground-truth gaze. (B) Control workstation: the investigator monitors the live gaze trace and calibration accuracy.
"""

MODELS = r"""
# 4 GAZE ESTIMATION MODELS

**Inputs.** Every model consumes the same four streams extracted from each smartphone frame: the left-eye, right-eye and face ROIs, resized to 224 × 224 × 3, and a 25 × 25 binary face grid encoding the face's position in the frame. These cropped streams, not the video, are what a release would contain (§6).

**Architectures.** We evaluate four published architectures (Table 1, Figure 3). iTracker and MobileNet-V3 fuse the streams late, concatenating the per-stream embeddings before a fully connected regression head; AFFNet and MGazeNet instead regress a guidance factor from the face and use it to modulate the normalisation of the eye features. We adapted MGazeNet to 224 px input, with an additional squeeze-and-excitation and LABN block in its fusion stage.

**Table 1.** The four gaze architectures. All regress the gaze point (x, y) on the display.

| Model | Eye streams | Face and grid streams | Fusion and normalisation |
|---|---|---|---|
| iTracker [CITE: krafka16] | weight-shared CNN, 128-d | face CNN, 64-d; grid MLP, 128-d | late concatenation |
| MobileNet-V3 Large [CITE: mobilenetv3] | MobileNet-V3 backbones | MobileNet-V3 face backbone; grid encoding as iTracker | late concatenation |
| AFFNet [CITE: affnet] | CNN with SE blocks | face features regress a guidance factor | adaptive group normalisation of eye features, conditioned on the factor |
| MGazeNet [CITE: zhu24] | CNN with SE blocks | face features regress a guidance factor | linear adaptive batch normalisation (LABN) of eye features; SE + LABN fusion block |

![Figure 3](FIGURE: original architecture comparison figure)

**Figure 3.** System architecture and processing workflow. (A) Layer-level comparison of the four gaze architectures. (B) Gaze estimation workflow, from ROI extraction from the smartphone video through multi-stream feature fusion to supervised training against the EyeLink 1000 ground truth.

**Training and evaluation.** Models are trained with a mean-squared-error loss on the gaze point in normalised screen coordinates and evaluated by the mean Euclidean distance between predicted and ground-truth gaze, converted to centimetres on the display. To measure generalisation to unseen people rather than memorisation of their appearance, we use participant-wise five-fold cross-validation [CITE: kfold]: the data are partitioned by recording rather than by frame, weights are re-initialised for every fold, and the checkpoint with the lowest validation error is kept. Every configuration is trained from five independent initialisations, whose spread gives the noise floor of all utility comparisons. Arms are compared on the uncalibrated model, which isolates the effect of the imagery; for the clinical-usability analysis (§8.4), each model's regression head is additionally fine-tuned per participant on frames from the calibration sequence recorded before the tasks. A face-only model, which receives the face ROI and grid but no eye crops, serves as a null control: it cannot distinguish two releases that differ only in their eye crops.
"""

DEID = r"""
# 5 DE-IDENTIFICATION METHODS

We test three families of generative de-identification, chosen because they differ in what they edit and how they treat identity. Each operator is run on the original full frame, and the released crops are re-extracted from the processed frame with the stored per-frame geometry, so that crops are pixel-comparable across every arm.

**Replacement: SimSwap++.** An adapted SimSwap++ [CITE: simswappp] transfers the identity of a source template onto the participant's face while preserving the target's geometric attributes — gaze direction, head pose and expression. Its conditional dynamic convolution and morphable knowledge distillation give high-resolution synthesis at low cost (about 2.5 GFLOPs). Each participant is mapped to the same synthetic appearance in every frame. To separate the method from the choice of face, every experiment is run with two source templates of different appearance, Identity A and Identity B (Figure 4). We refer to this family as SimSwap below.

**Removal: DeepPrivacy2.** DeepPrivacy2 [CITE: deepprivacy2] removes identity by inpainting rather than replacing it. Its *face* mode repaints the detected face box; its *full-body* mode repaints the whole person, guided by a dense surface map. We run both, each under two identity regimes: a latent code held fixed per participant (*fixed*), or drawn afresh for every frame (*per frame*), with truncation 1.0 so that the regimes differ.

**Diffusion: FAMS.** Face Anonymization Made Simple [CITE: fams] conditions a diffusion model on the source face and pastes an aligned square covering face, forehead and some hair back into the frame. We use the authors' anonymisation setting (anonymisation degree 1.25, 25 denoising steps, guidance 4.0), with the noise seed either fixed per participant or drawn afresh per frame. Diffusion runs at about 6 s per frame; the utility experiments use all 210,924 task frames.

**Release protocols.** Each family yields two candidate releases (Figure 5). **Full Synthetic** takes every crop from the de-identified frame. **Hybrid** combines the de-identified face crop with the participant's original, unaltered eye crops, fused at the data-loading level. Hybrid tests whether the context (the face) can be de-identified while the signal (the eyes) remains genuine. §7 measures what each protocol protects: as shown there, retaining genuine ocular ROIs retains the identifying information they carry, so Hybrid is a utility-preserving protocol rather than a privacy-preserving one, and is not by itself sufficient for public release.

![Figure 4](FIGURE: original face-swap examples figure)

**Figure 4.** Face replacement with SimSwap++. The face in each target frame is replaced by the source template's facial identity, while the participant's pose, expression and surroundings are preserved. Rows show the effect of two source templates (Identity A and Identity B) on different participants.

![Figure 5](FIGURE: original Hybrid pipeline figure)

**Figure 5.** The two release protocols. (A) Full Synthetic: every crop is taken from the de-identified frame, replacing the facial identity, ocular region included. (B) Hybrid: the de-identified face crop is combined with the original, unaltered eye crops, which carry the gaze geometry. The streams are fused at the data-loading level, so the model learns to predict gaze from real ocular features within a synthetic facial context.
"""
