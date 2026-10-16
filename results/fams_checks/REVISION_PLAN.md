# Revision plan, applied ONCE after the FAMS checks (C and D) are complete

Agreed with the author on 2026-10-16. Do not edit these paragraphs before then.

## 1. Threat model: A1 primary, A2 reframed
- §7 (paste file "7. Threat Model", final §6) "Attackers": A1 (enrolment photographs,
  closed-set cohort, e.g. candidates known from the recruitment pool) is the primary
  re-identification threat. A2 is NOT "the operative attacker"; it is the test of
  singling out and linkability, two of the three criteria in the Article 29 Working
  Party's Opinion 05/2014 on anonymisation techniques [CITE: wp29anon] (GDPR Recital 26
  [CITE: gdpr]): if records of one person can be isolated, the data remain personal data
  even without a name. State plainly that a usable gaze release keeps per-participant
  grouping (participant-wise splits, calibration), so it is linkable by construction;
  the A2 measurements matter for releases that strip participant identifiers, and
  establish that de-identified content cannot restore unlinkability.
- Remove "a single labelled example then names an entire cluster" and
  "the recipient of a public dataset is by definition A2".
- Related Work (Muştu paragraph): replace "the question a public release actually
  poses" with the singling-out / linkability framing.
- Contributions bullet 1 and the two-part-condition bullet: scope the condition to
  releases without participant identifiers.
- Discussion "The operative threat is release-only linkage" -> rewrite: lead with
  face-crop A1 (below); then linkability as the GDPR criterion; then the structural
  conclusion: a useful gaze release is linkable by design, so under the WP29 test it
  cannot be anonymous data; de-identification can only lower A1 risk, and release
  must be controlled access.
- Guideline G1: "Report A1 on every released stream, face crop included; report
  linkability as the anonymisation test."
- Abstract: "an attacker holding only the release still recovers exactly the right 18
  people" -> frame as singling out; add face-crop A1.
- Add [CITE: wp29anon] = Article 29 Data Protection Working Party, Opinion 05/2014 on
  Anonymisation Techniques, WP216, 10 April 2014.

## 2. Face-crop A1 for every family (new table; results/face_a1_all/face_a1_ci.json)
ArcFace closed-set Rank-1 [95% participant CI] / TAR@1e-3 [CI] (FaceNet in brackets):
FAMS 60.8 [50.9, 69.9] / 24.1 [15.3, 33.6] (FaceNet R1 42.8, TAR 5.9);
SimSwap A 57.4 [39.0, 76.6] / 18.4 [5.1, 35.4]; SimSwap B 34.9 / 22.0;
DP2 face 15.7–21.5 / 1.4–1.6; DP2 full body 3.8–7.8 / 0.0–0.3. Chance Rank-1 5.6%.
Discussion sentence "every full-synthesis method defeats A1 (at or near floor)" is
FALSE for the face crop -> correct it (true only for eye crops).
Table 2's earlier SimSwap face TAR (22.0 / 23.3) came from a separate sampling run:
replace with this run so all families share one run.

## 3. FAMS utility claims (pending C and D)
Withdraw "FAMS keeps most of the gaze signal / only full-synthesis release a third
party could train a useful model on". Pupil test: SimSwap preserves pupil position
better than FAMS (gaze-x correlation better in 17/17 participants). Explain FAMS's
small retraining penalty with D (stream isolation) and the transfer gap (test-anon
penalties similar: FAMS +2.15 vs SimSwap +2.54/+2.76). Add the pupil test as a
model-free utility measure (methods + results + guideline). Add the strength sweep
(usable range 1.25–1.5; >= 2.0 collapses).
