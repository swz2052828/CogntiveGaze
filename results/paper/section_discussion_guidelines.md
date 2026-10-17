## 9.5 Discussion — INSERT (new section, after the Results)

Scope: every claim below is about appearance-based gaze data — face and eye crops
released for training gaze models. We make no claim about other sensing
modalities; the evidence does not reach them.

> **The pixels that make gaze estimation work are the pixels that identify.** Every
> protocol we tested that kept the gaze signal did so by keeping genuine ocular
> imagery, and genuine ocular imagery verifies identity at 62–80% against a 1.6%
> floor. Hybrid recovers 85–101% of the synthesis penalty, and 91–106% when the
> models are retrained on the release, precisely because what it restores is the
> participant's own eye pixels. Releasing no pixels at all does not escape this:
> four iris coordinates per frame still link the release (ARI 0.27) while carrying
> almost no gaze signal across participants. For gaze data there is no part of the
> face that is useful for the task and safe to release unchanged.
>
> **Re-identification survives on the face crop; linkability survives
> everywhere.** Synthesising the eyes defeats an attacker holding the
> participants' photographs on the eye crops — A1 is at the floor for every
> full-synthesis method — but not on the face crop: SimSwap's and FAMS's synthetic
> faces still verify against the participant's own photographs at 18–24% TAR under
> ArcFace, against a 1.6% floor, and only DeepPrivacy2, which repaints the face,
> reaches the floor. Linkability is harder still: the same synthetic eye crops
> that score 0.2% against photographs verify at 43% within the release, and face
> crops partition exactly by participant. Under the European test for
> anonymisation, being able to single out each participant's records is enough
> for a release to remain personal data, whether or not anyone is named
> [CITE: wp29anon]. The channels that carry this — hair and skin at the crop border,
> head pose, crop geometry, illumination — are left alone by inner-face editing,
> and two of them need no face model at all: crop dimensions alone identify 66.7%
> of participants, photometric statistics alone link at ARI 0.87–0.90. A privacy
> number reported without its attacker therefore cannot tell a dataset author
> whether a release is safe.
>
> **Each half of the condition is attainable; their conjunction is not yet.**
> Release-only linkage falls only when the synthetic identity is not held fixed
> per participant *and* the release contains nothing the generator leaves
> untouched; it matters for releases that strip participant identifiers, since
> one that keeps them is linkable by construction. We observed each half
> separately. The diffusion anonymiser breaks per-participant consistency by
> construction and yields models that transfer to real faces — though its pupils
> carry less gaze than SimSwap's — but leaves the face crop linkable (ARI
> 0.85–0.94) and, under ArcFace, verifiable against the participant's
> photographs. Full-body
> inpainting with a fresh identity per frame meets both halves and breaks
> discovery of the cohort (face-crop ARI 0.297), but it repaints the eyes, and a
> gaze model trained on it learns nothing. Even there, an attacker with a few
> labelled frames per person still verifies 36% of them: the condition blocks
> discovery, not verification. What a releasable gaze corpus would need is
> therefore specific: a generator that keeps the pupil at least as faithfully as
> face swapping does, produces eyes from which models transfer to real users as
> the diffusion anonymiser's do, edits everything the released crop contains, and
> resamples identity per recording session rather than per frame — per-frame
> resampling breaks the temporal coherence gaze pipelines rely on, and
> per-session resampling cannot be verified on a single-session corpus like ours.
> We know of no method that meets all four.
>
> **Utility has to be measured the way the release will be used.** Deploying a
> model trained on real data on de-identified crops conflates two things that
> retraining separates. For inpainting, most of the deployment penalty is domain
> shift, yet what remains after retraining is the no-information level: the
> repainted eyes carry no recoverable gaze. For face swapping the penalty survives
> retraining, and for one architecture a model trained on the swapped release is
> worse than predicting the mean on real faces. A utility figure for a released
> gaze corpus should therefore come from models trained on the release and tested
> on real faces, reported beside the error of predicting the mean gaze point —
> and the pupil should be checked directly, because transfer and information can
> diverge: the release that transfers best carries less pupil signal than the one
> that transfers worst.
>
> **What this means for releasing gaze data now.** None of the protocols we
> measured makes a smartphone gaze corpus anonymous in the European sense: a
> usable release keeps each participant's frames together, so it is linkable by
> construction, and every gaze-preserving protocol also leaves residual
> re-identification on the face or eye crops. CognitiveGaze is therefore shared
> under controlled access, with derived gaze traces in an open tier
> whose own linkage we measured rather than assumed (ARI 0.033, low but not zero).
> We release the measurements as a tool, `privaudit`, and a fifteen-item checklist,
> so that a gaze dataset can be audited against the attacks and floors used here
> before it is shared; the tool reproduces every eye-ROI result in this paper.
>
> In the terms of the cycle in §1, this paper does not break it, and we do not
> claim to. Our own corpus ends where the cycle predicts — under controlled
> access — because no de-identification we measured yields a release that is
> both unlinkable and useful. What changes is the step at which the next group
> decides: instead of assuming that a de-identified release is safe, or that it is
> useless, a dataset author can measure both before publishing, and a generator
> developer knows what a gaze-preserving method must achieve (§9.6) for the
> cycle to open.

## 9.6 Design Guidelines for Releasing Gaze Data — INSERT (new section)

Seven guidelines for authors of appearance-based gaze datasets, each with the
evidence behind it and how to check it. The full fifteen-item checklist and the
audit tool are in the supplementary material [CITE: privaudit-repo].

| # | Guideline | Evidence in this paper | Check |
|---|---|---|---|
| G1 | **Report re-identification on every released stream, and test linkability.** Run A1 against photographs on each stream, face crop included, and report whether the content singles out participants (A2) as the anonymisation test. | Synthetic eye crops: 0.2% under A1, 43% under A2; synthetic face crops still 18–24% under A1. | privaudit R2–R4 |
| G2 | **Measure a floor on your own recordings.** Run every attack on a face-free patch of the same frames and quote results against it. | Single-session floor: TAR 1.57%, Rank-1 ~50%, ARI 0.074–0.089. | privaudit R0 |
| G3 | **Do not buy utility by releasing real eyes.** Retaining genuine ocular crops restores accuracy and releases the biometric with it. | Hybrid recovers 85–106% of utility; real eye crops verify at 62–80%. | privaudit R2–R3 on eye streams |
| G4 | **Close the channels that need no face model.** Release every crop at one fixed size and normalise per-crop photometry. | Crop size alone: 66.7% identification; photometry alone: ARI 0.87–0.90. | privaudit R5, R6 |
| G5 | **Meet both halves of the condition.** Do not hold the synthetic identity fixed per participant, and release nothing the generator does not edit. | Only the cell meeting both breaks face-crop linkage (1.000 → 0.297); verification still 36%. | privaudit R2, R3, R7 |
| G6 | **Measure utility by training on the release, and check the pupil directly.** Test on real faces, report the predict-the-mean error beside it, and measure whether the released pupil still follows true gaze. | Deployment-only penalties of +10–15 cm shrink to the no-information level after retraining; FAMS transfers best yet its pupils track gaze less closely than SimSwap's (17/17 participants). | — (retraining protocol, §9) |
| G7 | **If any check fails, use controlled access.** Share under an agreement that forbids re-identification and linkage, and state which channels leak. | No protocol tested passes G1–G5 with useful gaze accuracy. | privaudit overall verdict |
