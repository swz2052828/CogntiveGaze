# FAMS (diffusion) as a third de-identification family — privacy

**Date:** 2026-10-10. Jobs 1315052 (smoke), 1315209 (generation), 1315210 (attacks).
Method: *Face Anonymization Made Simple* (Kung et al., WACV 2025), official code and
weights (`hkung/face-anon-simple`), anonymization_degree 1.25, 25 steps, guidance
4.0. SD 2.1 VAE and scheduler from the community mirror
`sd2-community/stable-diffusion-2-1` (the stabilityai repository is no longer
publicly downloadable). Driver: `scripts/anon/run_fams.py`; same 260 frames per
participant as DeepPrivacy2, re-cropped with the stored geometry.

## Integrity

- 9,360 frames, **0 undetected** (SFD); 61 frames with a second detected face, also
  anonymised.
- Null control PASS: per_subject vs per_frame 10.4, per_subject vs original 12.6
  (mean |delta|, face crops).
- Mean full-frame |delta| 3.2 (DeepPrivacy2 face mode ~4.4): FAMS edits the
  FFHQ-aligned square — face, forehead, some hair — and pastes it back.
- ~5.8 s per frame on an L40 (25 steps).

## Results (floors: TAR 1.57%; ARI 0.089 face pipeline, 0.074 eye-native)

| | fixed seed | per frame |
|---|---|---|
| A2 eye TAR@1e-3, L / R | 6.1% / 8.2% | 5.1% / 5.2% |
| A2 eye ARI, L | 0.329 [0.26, 0.52] | 0.292 [0.24, 0.52] |
| A2 eye ARI, R | 0.288 [0.19, 0.51] | 0.290 [0.20, 0.51] |
| A1 eye TAR@1e-3, L / R | 0.7% / 1.1% | 0.7% / 1.3% |
| **T1 face-crop ARI @ k=18** | **0.939** | **0.853** |
| T1 face, k selected / ARI | 20 / 0.710 | 28 / 0.772 |

## What it shows

1. **A fixed seed does not fix the identity.** The two regimes are statistically
   indistinguishable on every measure. FAMS conditions on the source face, so the
   synthetic identity follows the frame; holding the noise fixed does not hold the
   person fixed (visible in the smoke montage: one participant's frames, one seed,
   different-looking faces). FAMS therefore breaks per-participant consistency by
   construction — condition (a) of the two-part condition — without being asked to.
2. **Eye-ROI linkage falls by about half, not to the floor.** Verification 5–8%
   against 43–63% for SimSwap and fixed-identity DeepPrivacy2; ARI ~0.29–0.33
   against 0.68–0.80. Every interval's lower bound is above the eye-native floor.
3. **The face crop stays linkable (0.85–0.94),** because the aligned square leaves
   the crop border, head pose and crop geometry alone — condition (b) is not met.
   This is the two-part condition predicted out of sample, on a method family the
   condition was not derived from.
4. Against A1, FAMS is at the floor, like every other method.

Utility (does FAMS keep the gaze signal?) is running on an every-10th-frame
subsample: jobs 1315392-1315404.

## Utility (added 2026-10-10; every 10th frame, per_frame identity)

Subset manifest `meanno7_clean_sub10`: 18,402 frames (~1,082/participant), 0 missing
crops. Real-data error on the subset matches the full data within 0.01-0.02 cm.
Script: `scripts/anon/fams_util_table.py` -> `util_table.json`.

Condition (ii), 5 draws x 4 backbones, vs real data on the same subset:
Full Synthetic +4.06 to +5.10 cm; Hybrid +0.02 to +0.36 cm.

Condition (i), 3 fast backbones, ONE draw (draws 2-5 running), vs real-data models
retrained on the same subset:
| | test real | test FAMS |
|---|---|---|
| Full Synthetic | +0.57 [-0.25, +1.19] | +1.45 [+0.81, +2.13] |
| Hybrid | +0.08 [-0.14, +0.42] | +0.06 [-0.12, +0.34] |
Every fold of the FAMS-trained models is at 6.1-7.2 cm against a 9.4 cm mean
predictor: the FAMS eyes carry most of the gaze signal, and the condition (ii)
penalty is mostly domain shift. The only full-synthesis release we tested that a
third party could train a useful model on — and its face crops still link (0.85).

## Control for training-set size (added 2026-10-12, jobs 1318120-1318154)

Question (author): is FAMS's small penalty an artefact of the 10% subset? Audit of
every step found no pipeline error (hybrid identical in construction to SimSwap/DP2:
FAMS face + original eye crops; 0 undetected frames; FAMS eye-crop change ~17, same as
SimSwap; training read the FAMS root, 13,894/4,508 frames, same split as the matched
baseline; real test = full 184,103 frames). But the comparison with full-data SimSwap/
DP2 was not like-for-like, so SimSwap A and DP2 face fixed were retrained on the SAME
10% subset (3 fast backbones x 5 draws), penalties paired within draw vs the same
real-data subset model. `subset_control.json`.

| test on real faces | MobileNet-V3 | AFFNet | MGazeNet | mean |
|---|---|---|---|---|
| FAMS | +1.29 | -0.79 | +0.45 | +0.32 |
| SimSwap A, same subset | +3.15 | +2.16 | +4.46 | +3.26 |
| DP2 face fixed, same subset | +4.23 | +2.60 | +3.57 | +3.46 |

FAMS better in 15/15 paired draws vs SimSwap (mean -2.94 cm) and 15/15 vs DP2 (-3.15).
Tested on the de-identified subset: 14/15 vs SimSwap (-0.60), 15/15 vs DP2.
Subsampling compresses penalties (SimSwap A +3.83 full -> +3.26 subset), so FAMS's
0.32 cm is "on a 10% subset", not a full-scale claim; the advantage is method-specific.

## Full-data condition (i) — PRELIMINARY, draw 1 only (2026-10-13)

Models retrained on FAMS applied to every frame (`famsF`, per-subject identity),
baseline = the 5-draw clean real-data model (same as every Table 3 arm).
Penalty in cm; AFFNet test-real has 2 draws. Draws 2–5 still running.

| backbone | FS test real | FS test anon | Hybrid test real | Hybrid test anon |
|---|---|---|---|---|
| MobileNet-V3 | +1.14 | +2.70 | −0.19 | −0.21 |
| AFFNet | +0.36 (2 draws) | +1.48 | +0.43 | +0.42 |
| MGazeNet | +1.25 | +2.32 | −0.14 | −0.14 |

For comparison, cond (i) test real, 5 draws: SimSwap +2.47 / +4.47 / +4.56;
DP2 face +4.79 / +3.94 / +4.81 (no-information level). The 10%-subset matched
control gave FAMS +0.32; on all frames the FS penalty is +0.36 to +1.25,
still 2–4x smaller than SimSwap's. Test-anon > test-real for FAMS only (SimSwap
and DP2 are equal across test sets): a model trained on FAMS is less accurate on
FAMS crops than on real faces. AFFNet Hybrid +0.43 (single draw) exceeds its FS
penalty; do not interpret before 5 draws.
