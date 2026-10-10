# Release checklist for appearance-based gaze datasets

For authors deciding whether, and how, to share smartphone or webcam gaze data.
Each item states the requirement, the evidence behind it (from the CognitiveGaze
study unless cited otherwise), and how to check it. Items marked **[R*]** are
measured by `privaudit`; the others need a human decision.

Nothing here certifies anonymity. Passing every measured item means no channel
we know how to attack leaks above the floor *on the data you have*; it says
nothing about attacks not yet written, or about sessions you did not record.

---

## A. Before de-identifying

**C1. State the attacker.** Two attackers give different answers on identical
pixels. *A1* holds photographs of participants; *A2* holds only the release.
For a public release, A2 is the operative attacker: every recipient is one.
Synthetic eye crops that score 0.2% against A1 score 43% against A2. A privacy
number without its attacker is uninterpretable.

**C2. Measure a floor. [R0]** On single-session data, lighting, seating and
camera are constant within a participant, so every attack has a non-zero
baseline that owes nothing to the face. Run the same attack on a face-free patch
of the same recordings and quote every result against it (ours: TAR at FAR 1e-3
1.57%, Rank-1 ~50%). Closed-set Rank-1 is not evidence on such data. Quote each
recogniser against its own floor; never mix them.

## B. What the release contains

**C3. Treat the eye crops as a biometric.** An unmodified 120 px eye ROI
verifies identity at 62–80% TAR at FAR 1e-3 under two unrelated recognisers,
about forty times the floor. A protocol that keeps genuine eye pixels to
preserve gaze accuracy (e.g. "Hybrid": replaced face, original eyes) releases
that biometric in full. Utility and identity are the same pixels.

**C4. Normalise crop geometry. [R6]** Crop sizes are constant within a
participant. Face and eye crop size alone identify 66.7% of our participants
(chance 5.6%); with inter-ocular distance, 94.4% — without reading a pixel.
Resize every released crop to one fixed size and treat every released number
(grids, boxes, IOD) as attack surface.

**C5. Normalise photometry. [R5]** Twenty-one summary statistics per crop
(channel means, percentiles, white-balance ratios), with no face model at all,
partition our participants at ARI 0.87–0.90, and identity replacement does not
touch them. Normalise exposure and white balance per crop, or accept that the
release is linkable without a recogniser.

**C6. Release nothing the transformation does not edit.** Half of the two-part
condition. Linkage survives on whatever the generator leaves alone: hair, ears,
skin tone at the crop border, head pose, crop geometry. Either crop the release
to the edit domain or widen the edit domain to everything the crop contains.

## C. The transformation

**C7. Do not hold the synthetic identity fixed per participant.** The other half
of the condition. Per-participant-consistent replacement removes the link to the
real face and leaves the release internally self-consistent, i.e. exactly
partitionable (face-crop ARI 1.000). Only resampling the identity *and* meeting
C6 broke linkage in our 2×2 (ARI 0.297) — and even then an attacker holding a few
labelled frames per person still verified 36% of them: the condition stops
discovery, not verification. Run both attacks (R2 and R3). Per-frame resampling breaks temporal
coherence; per-session resampling is the deployable form, and a single-session
corpus cannot verify it.

**C8. Verify the transformation was applied. [R8]** Compare the release with the
original before attacking it. Generator settings can silently undo the method:
DeepPrivacy2 at its default truncation (0) maps every face to the mean face, which
makes the identity latent — and any experiment on it — a no-op.

**C9. Account for what the pipeline skipped.** Count frames where the detector
found no face (they ship unedited) and regions routed to a missing or fallback
generator. Report the counts and check whether they fall inside released crops.

## D. Measure

**C10. Attack the release itself. [R2, R3, R7]** Verification and unsupervised
linkage under A2, with participant-level bootstrap intervals (frames of one
person are not independent), at the true cohort size *and* with the cohort size
unknown.

**C11. Attack with enrolment imagery if it could exist. [R4]** Clinical and
social-media photographs make A1 realistic for many cohorts.

**C12. Use at least two recogniser families.** Agreement is evidence about the
imagery rather than one model's inductive bias (`--recogniser facenet`).

**C13. Measure the non-pixel tier too.** Eye movement is a biometric. A release
of gaze traces or oculomotor features is not safe by assertion; ours leaks at
ARI 0.033 — small, two orders of magnitude below the pixel tiers, not zero.

## E. Utility, measured the way users will use it

**C14. Report the conditions a user will meet.** Deploying a real-data model on
de-identified input mixes lost information with domain shift. Retrain on the
release, and test on real faces (the third party who trains on your corpus and
ships a model) and on the release (benchmarking). Report the error of predicting
the mean gaze point beside every number: a model trained on inpainted eyes in our
study was never better than it.

## F. Decide

**C15. If any measured item fails, do not deposit openly.** Share under
controlled access with an enforceable agreement (no re-identification, no
linkage, no redistribution), and say in the documentation which channels leak.

---

| Item | privaudit | Manual |
|---|---|---|
| C1 attacker stated | runs A1 and A2 | document it |
| C2 floor | R0 | supply full frames |
| C3 eye crops | R2, R3 on eye streams | protocol choice |
| C4 crop geometry | R6 | resize crops |
| C5 photometry | R5 | normalise |
| C6 edit domain | R2 on each stream | crop / widen edit |
| C7 identity regime | R2, R7 | per-session resampling |
| C8 transform applied | R8 | check generator settings |
| C9 skipped frames | — | count and report |
| C10 release attacks | R2, R3, R7 | — |
| C11 enrolment attack | R4 | — |
| C12 two recognisers | `--recogniser` | run twice |
| C13 non-pixel tier | — | see study's `gaze_signal_linkage.py` |
| C14 utility conditions | — | retrain on release |
| C15 decision | overall verdict | access agreement |
