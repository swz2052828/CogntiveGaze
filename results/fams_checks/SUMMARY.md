# FAMS vs SimSwap: like-for-like checks

## A. Face-crop A1 (gallery = original photos)

| arm | ArcFace Rank-1 | ArcFace TAR@1e-3 | AUC | FaceNet Rank-1 | FaceNet TAR@1e-3 |
|---|---|---|---|---|---|
| ProcessedSwap | 57.4% | 18.4% | 0.927 | 29.0% | 14.4% |
| ProcessedSwap2 | 34.9% | 22.0% | 0.813 | 10.9% | 0.2% |
| ProcessedDP2t1_per_subject | 15.7% | 1.6% | 0.689 | 20.5% | 0.7% |
| ProcessedDP2t1_per_frame | 21.5% | 1.4% | 0.698 | 30.3% | 2.8% |
| ProcessedDP2fb_per_subject | 3.8% | 0.0% | 0.509 | 11.8% | 0.0% |
| ProcessedDP2fb_per_frame | 7.8% | 0.3% | 0.569 | 10.1% | 0.6% |
| ProcessedFAMS_per_subject | 61.5% | 22.4% | 0.914 | 41.0% | 3.2% |
| ProcessedFAMS_per_frame | 60.8% | 24.1% | 0.907 | 42.8% | 5.9% |

## B. Pupil preservation (identical 4,680 frames)

| arm | r_u | r_v | gt r x | gt r y | mean abs du |
|---|---|---|---|---|---|
| ProcessedData | 1.000 | 1.000 | 0.869 | 0.557 | 0.000 |
| ProcessedSwap | 0.700 | 0.444 | 0.795 | 0.292 | 0.030 |
| ProcessedSwap2 | 0.703 | 0.443 | 0.832 | 0.283 | 0.034 |
| ProcessedFAMS_per_frame | 0.610 | 0.333 | 0.675 | 0.175 | 0.038 |
| ProcessedFAMSfull_per_frame | 0.611 | 0.346 | 0.687 | 0.212 | 0.038 |
| ProcessedDP2full_per_subject | 0.006 | -0.012 | 0.060 | 0.071 | 0.063 |
| ProcessedDP2full_per_frame | -0.024 | 0.014 | 0.065 | 0.056 | 0.070 |
| ProcessedDP2fbfull_per_frame | -0.014 | -0.003 | 0.056 | 0.051 | 0.068 |

## C. FAMS strength sweep (260 frames/participant, per-frame noise)

| degree | collapsed | face R1 Arc | face TAR Arc | face R1 FaceNet | face ARI | eyeL A1 TAR | eyeL A2 TAR | eyeL ARI | pupil r_u | pupil gt r x | complete |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.25 | 0.0% | 60.8% | 24.1% | 42.8% | 0.853 | 0.6% | 5.1% | 0.292 | 0.610 | 0.675 | True |
| 1.35 | 0.1% | 45.5% | 14.1% | 31.4% | 0.810 | 0.8% | 4.7% | 0.305 | 0.546 | 0.621 | True |
| 1.5 | 6.5% | 35.9% | 10.1% | 22.6% | 0.899 | 0.2% | 10.5% | 0.283 | 0.463 | 0.533 | True |
| 1.6 | 18.7% | 29.1% | 5.9% | 15.3% | 0.792 | 0.6% | 5.8% | 0.302 | 0.350 | 0.413 | True |
| 1.75 | 36.0% | 22.0% | 3.7% | 20.5% | 0.784 | 0.5% | 4.9% | 0.259 | 0.186 | 0.234 | True |
| 2.0 | 93.9% | 11.2% | 0.9% | 9.5% | 0.507 | 0.2% | 13.5% | 0.353 | 0.012 | 0.143 | True |
| 3.0 | 100.0% | 6.9% | 0.2% | 3.8% | — | 0.0% | 0.0% | 0.251 | — | — | False |
| SimSwap A (reference) | — | — | — | — | — | — | — | — | 0.700 | 0.795 | True |

## D. Stream isolation (retrained; penalty vs same-backbone real-data model, paired within draw)

| backbone | real-data error | SimSwap A test real | SimSwap B test real | FAMS test real | SimSwap A test anon | SimSwap B test anon | FAMS test anon | draws |
|---|---|---|---|---|---|---|---|---|
| eyes_only_mobilenet_v3 | 4.95 | +2.71 ± 0.93 | +3.56 ± 0.39 | +1.16 ± 0.10 | +2.44 ± 0.36 | +1.98 ± 0.22 | +2.72 ± 0.21 | 5–5 |
| eyes_only_mgazenet | 5.14 | +4.79 ± 0.94 | +6.77 ± 0.56 | +0.96 ± 0.27 | +2.83 ± 0.38 | +2.95 ± 0.12 | +2.04 ± 0.17 | 5–5 |
| face_only_mobile_vit | 5.29 | +4.51 ± 0.40 | +3.87 ± 0.59 | +1.44 ± 0.26 | +3.67 ± 0.16 | +2.98 ± 0.33 | +2.24 ± 0.29 | 5–5 |
