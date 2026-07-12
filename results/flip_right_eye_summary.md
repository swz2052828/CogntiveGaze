# Flip-right-eye retrain vs clean baseline (base val_error, cm; delta>0 = flip better)

| backbone | baseline | flipped | delta |
|---|---|---|---|
| convnextv2_film | 5.541 | 4.139 | +1.402 |
| eyes_only_mobilevitv2 | 5.471 | 4.775 | +0.696 |
| mgazenet | 4.963 | 4.330 | +0.633 |
| itracker | 5.922 | 5.346 | +0.575 |
| eyes_only_convnextv2 | 4.481 | 3.965 | +0.516 |
| eyes_only_eva02_tiny | 4.920 | 4.468 | +0.452 |
| eyes_only_mobile_vit | 4.205 | 3.783 | +0.422 |
| affnet | 5.792 | 5.397 | +0.395 |
| mobilenet_v4 | 4.503 | 4.118 | +0.386 |
| eyes_only_convnextv2_atto | 4.601 | 4.252 | +0.349 |
| repvit | 4.654 | 4.308 | +0.346 |
| eyes_only_mgazenet | 5.107 | 4.768 | +0.339 |
| face_only_mobile_vit | 5.422 | 5.103 | +0.318 |
| convnextv2_dualenc | 4.403 | 4.085 | +0.318 |
| dinov2 | 5.358 | 5.097 | +0.260 |
| convnext | 4.386 | 4.130 | +0.256 |
| convnextv2 | 4.321 | 4.140 | +0.181 |
| normface_convnext | 4.588 | 4.421 | +0.168 |
| cnn_transformer | 4.704 | 4.576 | +0.128 |
| mobilevitv2 | 4.791 | 4.667 | +0.123 |
| eyes_only_convnextv2_atto_binocular | 4.348 | 4.268 | +0.080 |
| eyes_only_vit | 4.385 | 4.355 | +0.030 |
| eva02 | 5.085 | 5.119 | -0.033 |
| eyes_only_fastvit | 4.472 | 4.519 | -0.047 |
| vit | 4.369 | 4.417 | -0.048 |
| convnextv2_atto | 4.320 | 4.370 | -0.050 |
| eyes_only_mobilenet_v4 | 4.326 | 4.379 | -0.053 |
| foveal_vit | 4.424 | 4.586 | -0.162 |
| cnn_transformer_raw | 5.136 | 5.499 | -0.363 |
| mobile_vit | 5.116 | 5.654 | -0.538 |
| convnextv2_nano | 7.361 | 8.475 | -1.113 |
| eyes_only_mobilenet_v3 | 4.992 | 6.648 | -1.657 |
| mobilenet_v3 | 5.043 | 6.701 | -1.658 |
| eyes_only_convnextv2_binocular | 4.587 | 6.256 | -1.669 |

Cohort: mean +0.029, median +0.175, flip better 22/34

## Conclusions
- Control (face_only_mobile_vit, no eye streams) shows +0.32 -> run-to-run init
  noise floor is ~±0.3-0.4 cm; only larger deltas are meaningful.
- Real winners = SHARED-TOWER architectures (itracker +0.58, mgazenet +0.63,
  eyes_only_mobilevitv2 +0.70, convnextv2_film +1.40): canonical chirality frees
  the shared eye tower from learning both mirror classes.
- Real losers: the BINOCULAR family (femto binocular -1.67) — flipping one eye
  destroys the |L-R|, L*R pairwise-difference correspondence. Rule: canonicalize
  chirality BEFORE pairwise difference features, or not at all. mobilenet_v3
  (-1.66 both variants) also systematically harmed.
- Cohort neutral (mean +0.03, median +0.18, 22/34) -> NOT adopted globally;
  selective adoption for shared-tower models only. Production leader
  (atto_binocular) unaffected (+0.08 = noise).

## Calibrated comparison (fc_ft/svr_embed, deploy-faithful, all 33 evaluable backbones)
Cohort fcft@K72: mean -0.327, median -0.123, flip better 12/33 -> **calibration
REVERSES the verdict; flip is net-negative calibrated.**
- Reversals: itracker +0.58 uncalib -> -1.92 calib (flip wrecks calibratability);
  atto_binocular (leader) +0.08 -> -1.16 (binocular damage exposed);
  femto binocular -1.67 -> -2.83 (feature damage amplifies).
- Surviving winners (all eyes-only shared-tower + film): convnextv2_film +1.20,
  eyes_only_mobilevitv2 +0.66, eyes_only_fastvit +0.66, eyes_only_mobilenet_v4
  +0.64, eyes_only_convnextv2_atto +0.50, eyes_only_mobile_vit +0.47, mgazenet +0.46.
- Mechanism: per-subject calibration exploits EYE-ASYMMETRY signal; the flip
  destroys it (explicitly in binocular diff features, implicitly wherever the
  refitted readout used cross-eye structure). Only pure eyes-only shared towers
  keep the canonical-chirality gain.
- FINAL: not adopted (harms leader); selective use only for eyes-only mobile family.
- cnn_transformer_raw excluded (no forward_features contract).
