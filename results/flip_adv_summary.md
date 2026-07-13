# Flip x adversarial interaction (cm, deploy-faithful; base_adv under flip)

cl=clean baseline, fl=flip, fl_adv=flip+adversarial (base_adv under flip). fcft@K72 = best deployable calibrated.

| backbone | base cl | base fl | base fl_adv | fcft cl | fcft fl | fcft fl_adv |
|---|---|---|---|---|---|---|
| mobilenet_v4 | 4.51 | 4.12 | 3.89 | 4.11 | 3.76 | 3.63 |
| face_only_mobile_vit | 5.42 | 5.14 | 5.09 | 3.54 | 3.69 | 3.67 |
| eyes_only_mobilenet_v4 | 4.32 | 4.38 | 4.09 | 4.45 | 3.81 | 3.70 |
| eyes_only_convnextv2_atto | 4.60 | 4.26 | 4.40 | 4.29 | 3.78 | 3.71 |
| convnextv2_dualenc | 4.41 | 4.08 | 4.50 | 3.97 | 4.03 | 3.82 |
| eyes_only_mobile_vit | 4.20 | 3.79 | 4.14 | 4.00 | 3.53 | 3.88 |
| eyes_only_convnextv2 | 4.47 | 3.96 | 4.16 | 4.41 | 4.39 | 3.93 |
| foveal_vit | 4.42 | 4.59 | 4.41 | 3.35 | 4.01 | 3.96 |
| eyes_only_eva02_tiny | 4.92 | 4.46 | 4.74 | 4.01 | 3.88 | 4.22 |
| eyes_only_fastvit | 4.46 | 4.53 | 4.38 | 4.66 | 3.99 | 4.22 |
| cnn_transformer | 4.72 | 4.59 | 5.40 | 3.68 | 4.16 | 4.35 |
| mgazenet | 4.96 | 4.35 | 5.08 | 4.66 | 4.20 | 4.39 |
| convnextv2 | 4.32 | 4.14 | 4.39 | 3.89 | 3.89 | 4.50 |
| convnextv2_atto | 4.33 | 4.37 | 4.42 | 3.50 | 3.71 | 4.69 |
| eyes_only_mgazenet | 5.11 | 4.78 | 4.61 | 4.43 | 4.49 | 4.93 |
| mobilenet_v3 | 5.03 | 6.69 | 6.65 | 4.03 | 5.27 | 5.00 |
| repvit | 4.65 | 4.31 | 4.63 | 4.31 | 5.12 | 5.11 |
| convnext | 4.39 | 4.13 | 4.51 | 3.80 | 3.79 | 5.15 |
| normface_convnext | 4.59 | 4.42 | 5.36 | 4.14 | 3.88 | 5.16 |
| eyes_only_convnextv2_atto_binocular | 4.36 | 4.27 | 4.18 | 3.26 | 4.40 | 5.25 |
| eyes_only_mobilevitv2 | 5.45 | 4.77 | 4.91 | 4.99 | 4.33 | 5.39 |
| mobile_vit | 5.11 | 5.65 | 6.85 | 3.33 | 4.06 | 5.45 |
| eyes_only_convnextv2_binocular | 4.59 | 6.26 | 4.16 | 3.26 | 6.09 | 5.63 |
| eva02 | 5.09 | 5.12 | 5.57 | 4.79 | 5.50 | 5.67 |
| eyes_only_mobilenet_v3 | 5.01 | 6.65 | 7.04 | 4.12 | 5.40 | 5.71 |
| eyes_only_vit | 4.38 | 4.35 | 4.66 | 3.80 | 3.91 | 5.79 |
| mobilevitv2 | 4.78 | 4.68 | 5.08 | 4.05 | 4.14 | 5.96 |
| affnet | 5.79 | 5.40 | 4.69 | 5.31 | 6.16 | 5.99 |
| convnextv2_film | 5.54 | 4.14 | 5.62 | 5.04 | 3.86 | 6.15 |
| vit | 4.37 | 4.42 | 4.48 | 3.68 | 3.96 | 6.93 |
| dinov2 | 5.36 | 5.10 | 5.62 | 4.79 | 5.98 | 7.81 |
| convnextv2_nano | 7.36 | 8.47 | 9.35 | 7.11 | 8.45 | 9.40 |
| **COHORT MEAN** | 4.84 | 4.82 | 5.03 | 4.21 | 4.49 | 5.10 |

## Verdict
Calibrated cohort worsens monotonically clean 4.21 -> flip 4.49 -> flip+adv 5.10. base_adv (uncalibrated) is a wash vs flip; adv's damage appears AFTER calibration (DANN subject-invariant features strip the per-subject signal calibration needs), worst for ViT-family (vit 3.96->6.93, dinov2 5.98->7.81). Best flip+adv = mobilenet_v4 3.63, still far behind the clean-baseline champion (atto_binocular/binocular 3.26). No arm beats the clean baseline. Flip+adv NOT adopted -- stacking two independently-harmful interventions compounds the harm.
