# FAMS vs SimSwap like-for-like checks — results

## C. FAMS anonymisation-strength sweep (complete 2026-10-16)

260 frames/participant, per-frame noise, all 18 participants. "Collapsed" = face crops
in which the detector finds no face (output is no longer a face). Degrees with >10%
collapse are excluded from privacy conclusions.

| degree | collapsed | face A1 Rank-1 ArcFace / FaceNet | face A1 TAR@1e-3 ArcFace | face ARI | eye-L A1 TAR | eye-L A2 TAR / ARI | pupil r (orig) | pupil r (gaze x) |
|---|---|---|---|---|---|---|---|---|
| 1.25 (authors) | 0.0% | 60.8% / 42.8% | 24.1% | 0.853 | 0.6% | 5.1% / 0.292 | 0.610 | 0.675 |
| 1.35 | 0.1% | 45.5% / 31.4% | 14.1% | 0.810 | 0.8% | 4.7% / 0.305 | 0.546 | 0.621 |
| 1.5 | 6.5% | 35.9% / 22.6% | 10.1% | 0.899 | 0.2% | 10.5% / 0.283 | 0.463 | 0.533 |
| 1.6 (excluded) | 18.7% | 29.1% / 15.3% | 5.9% | 0.792 | 0.6% | 5.8% / 0.302 | 0.350 | 0.413 |
| 1.75 (excluded) | 36.0% | 22.0% / 20.5% | 3.7% | 0.784 | 0.5% | 4.9% / 0.259 | 0.186 | 0.234 |
| 2.0 / 3.0 (excluded) | 94% / 100% | — | — | — | — | — | — | — |
| SimSwap A (reference) | — | 57.4% / 29.0% | 18.4% | 1.000 | 0.2% | 42.8% / 0.684 | 0.700 | 0.795 |
| Original | — | — | — | 1.000 | 66.6% | — | 1 | 0.869 |

Chance Rank-1 5.6%; TAR floor 1.57%; ARI floor 0.089 (face) / 0.074 (eye).

**Findings.**
1. Raising the strength trades pupil information for face-identity protection,
   monotonically: from 1.25 to 1.5, face Rank-1 falls 60.8% -> 35.9% and TAR 24.1% -> 10.1%,
   while the pupil's correlation with true horizontal gaze falls 0.675 -> 0.533
   (SimSwap 0.795, original 0.869).
2. No usable strength reaches the floor on the face crop. At 1.5, the strongest
   setting that still produces faces in >90% of frames, face A1 is still 6x chance
   (Rank-1) and 6x floor (TAR).
3. Linkability does not move with strength: face ARI 0.81-0.90 at every usable degree,
   eye ARI ~0.28-0.31 throughout. The strength parameter edits the inner face; the
   channels that link (border, pose, crop geometry) are outside it — the two-part
   condition again.
4. Eye-crop A1 stays at floor at every degree, as at 1.25.
5. Beyond 1.5 the generator degrades rather than anonymises (19-100% non-faces).

**For the paper.** FAMS has no setting that is private on the face crop and keeps the
pupil signal: each step that lowers face A1 removes gaze information, and none removes
linkability. Report as a sweep figure/table in §7–8; collapse rate stated explicitly.
