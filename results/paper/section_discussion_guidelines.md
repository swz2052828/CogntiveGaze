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
> **The operative threat is release-only linkage, and it lives outside the edit
> domain.** An attacker holding photographs of participants is defeated by every
> full-synthesis method we tested (A1 at or near the floor). An attacker holding
> only the release is not: the same synthetic eye crops that score 0.2% against
> photographs verify at 43% within the release, and face crops partition exactly
> by participant. The channels that carry this — hair and skin at the crop border,
> head pose, crop geometry, illumination — are left alone by inner-face editing,
> and two of them need no face model at all: crop dimensions alone identify 66.7%
> of participants, photometric statistics alone link at ARI 0.87–0.90. A privacy
> number reported without its attacker therefore cannot tell a dataset author
> whether a release is safe.
>
> **Each half of the condition is attainable; their conjunction is not yet.**
> Release-only linkage falls only when the synthetic identity is not held fixed
> per participant *and* the release contains nothing the generator leaves
> untouched. We observed each half separately. The diffusion anonymiser breaks
> per-participant consistency by construction and keeps most of the gaze signal —
> models retrained on it lose 0.4–1.3 cm on real faces where SimSwap loses
> 2.5–4.6 — but leaves the face crop linkable (ARI 0.85–0.94). Full-body
> inpainting with a fresh identity per frame meets both halves and breaks
> discovery of the cohort (face-crop ARI 0.297), but it repaints the eyes, and a
> gaze model trained on it learns nothing. Even there, an attacker with a few
> labelled frames per person still verifies 36% of them: the condition blocks
> discovery, not verification. What a releasable gaze corpus would need is
> therefore specific: a generator that preserves the ocular geometry the way the
> diffusion anonymiser does, edits everything the released crop contains, and
> resamples identity per recording session rather than per frame — per-frame
> resampling breaks the temporal coherence gaze pipelines rely on, and
> per-session resampling cannot be verified on a single-session corpus like ours.
> We know of no method that meets all three.
>
> **Utility has to be measured the way the release will be used.** Deploying a
> model trained on real data on de-identified crops conflates two things that
> retraining separates. For inpainting, most of the deployment penalty is domain
> shift, yet what remains after retraining is the no-information level: the
> repainted eyes carry no recoverable gaze. For face swapping the penalty survives
> retraining, and for one architecture a model trained on the swapped release is
> worse than predicting the mean on real faces. A utility figure for a released
> gaze corpus should therefore come from models trained on the release and tested
> on real faces, reported beside the error of predicting the mean gaze point.
>
> **What this means for releasing gaze data now.** None of the protocols we
> measured makes a smartphone gaze corpus safe for open deposit, so CognitiveGaze
> is shared under controlled access, with derived gaze traces in an open tier
> whose own linkage we measured rather than assumed (ARI 0.033, low but not zero).
> We release the measurements as a tool, `privaudit`, and a fifteen-item checklist,
> so that a gaze dataset can be audited against the attacks and floors used here
> before it is shared; the tool reproduces every eye-ROI result in this paper.

## 9.6 Design Guidelines for Releasing Gaze Data — INSERT (new section)

Seven guidelines for authors of appearance-based gaze datasets, each with the
evidence behind it and how to check it. The full fifteen-item checklist and the
audit tool are in the supplementary material [CITE: privaudit-repo].

| # | Guideline | Evidence in this paper | Check |
|---|---|---|---|
| G1 | **Name both attackers and run both.** Report re-identification from photographs (A1) *and* linkage within the release (A2). | Identical synthetic eye crops: 0.2% under A1, 43% under A2. | privaudit R2–R4 |
| G2 | **Measure a floor on your own recordings.** Run every attack on a face-free patch of the same frames and quote results against it. | Single-session floor: TAR 1.57%, Rank-1 ~50%, ARI 0.074–0.089. | privaudit R0 |
| G3 | **Do not buy utility by releasing real eyes.** Retaining genuine ocular crops restores accuracy and releases the biometric with it. | Hybrid recovers 85–106% of utility; real eye crops verify at 62–80%. | privaudit R2–R3 on eye streams |
| G4 | **Close the channels that need no face model.** Release every crop at one fixed size and normalise per-crop photometry. | Crop size alone: 66.7% identification; photometry alone: ARI 0.87–0.90. | privaudit R5, R6 |
| G5 | **Meet both halves of the condition.** Do not hold the synthetic identity fixed per participant, and release nothing the generator does not edit. | Only the cell meeting both breaks face-crop linkage (1.000 → 0.297); verification still 36%. | privaudit R2, R3, R7 |
| G6 | **Measure utility by training on the release.** Test on real faces, and report the predict-the-mean error beside it. | Deployment-only penalties of +10–15 cm shrink to the no-information level after retraining; SimSwap can be worse than the mean. | — (retraining protocol, §9) |
| G7 | **If any check fails, use controlled access.** Share under an agreement that forbids re-identification and linkage, and state which channels leak. | No protocol tested passes G1–G5 with useful gaze accuracy. | privaudit overall verdict |

