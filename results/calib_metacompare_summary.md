# Deploy-faithful calibration — cross-backbone summary (at BEST K, not mean over K)

## Accuracy at each backbone's best K (per method best-K error; best = winning method @ its K)

| backbone | base | svr | svr_embed | fc_ft | meta | meta_adv | **best @K** |
|---|---|---|---|---|---|---|---|
| face_only_mobile_vit | 5.04 | 3.83(K36) | 3.10(K18) | 3.58(K72) | 3.90(K36) | 4.28(K18) | **svr_embed 3.10@K18** |
| mobile_vit | 4.98 | 4.15(K72) | 4.02(K72) | 3.33(K72) | 3.65(K18) | 3.77(K18) | **fc_ft 3.33@K72** |
| foveal_vit | 4.53 | 3.99(K36) | 3.52(K18) | 3.63(K36) | 4.21(K36) | 4.37(K36) | **svr_embed 3.52@K18** |
| mobilevitv2 | 5.03 | 5.83(K18) | 4.94(K36) | 4.41(K72) | 3.77(K18) | 3.96(K72) | **meta 3.77@K18** |
| eyes_only_mobilenet_v3 | 4.84 | 4.51(K36) | 3.97(K36) | 4.23(K72) | 3.81(K36) | 4.12(K18) | **meta 3.81@K36** |
| eyes_only_vit | 4.62 | 4.65(K72) | 4.09(K72) | 3.85(K72) | 4.36(K18) | 4.44(K18) | **fc_ft 3.85@K72** |
| eva02 | 4.91 | 4.85(K18) | 4.44(K72) | 3.90(K72) | 4.98(K36) | 5.59(K72) | **fc_ft 3.90@K72** |
| convnextv2 | 4.33 | 4.74(K72) | 4.60(K18) | 4.05(K72) | 4.24(K36) | 4.11(K36) | **fc_ft 4.05@K72** |
| eyes_only_mobilevitv2 | 5.71 | 5.58(K4) | 5.86(K36) | 5.00(K36) | 4.08(K72) | 4.57(K18) | **meta 4.08@K72** |
| repvit | 4.92 | 4.43(K72) | 4.17(K72) | 4.55(K36) | 4.13(K36) | 4.22(K72) | **meta 4.13@K36** |
| eyes_only_eva02_tiny | 4.92 | 5.40(K36) | 5.00(K36) | 4.16(K72) | 5.01(K9) | 5.02(K9) | **fc_ft 4.16@K72** |
| eyes_only_convnextv2 | 4.59 | 5.68(K36) | 5.18(K4) | 4.45(K36) | 4.55(K36) | 4.29(K36) | **meta_adv 4.29@K36** |
| eyes_only_fastvit | 4.72 | 5.60(K36) | 6.25(K9) | 4.79(K9) | 4.43(K9) | 4.33(K36) | **meta_adv 4.33@K36** |
| eyes_only_mobilenet_v4 | 4.49 | 6.87(K9) | 6.52(K9) | 4.73(K9) | 4.35(K72) | 4.42(K36) | **meta 4.35@K72** |
| convnext | 4.82 | 4.87(K72) | 4.36(K18) | 4.55(K36) | 4.77(K72) | 4.58(K9) | **svr_embed 4.36@K18** |
| eyes_only_mgazenet | 5.61 | 6.20(K9) | 5.19(K36) | 4.91(K36) | 4.77(K36) | 4.67(K18) | **meta_adv 4.67@K18** |
| dinov2 | 5.23 | 5.31(K18) | 5.24(K18) | 4.89(K9) | 5.38(K9) | 5.92(K18) | **fc_ft 4.89@K9** |
| normface_convnext | 5.86 | 6.07(K72) | 5.52(K72) | 5.64(K72) | 5.89(K18) | 5.00(K72) | **meta_adv 5.00@K72** |
| itracker | 5.95 | 6.83(K9) | 5.92(K4) | 5.53(K72) | 5.18(K36) | 6.07(K18) | **meta 5.18@K36** |

## Impact of K (cross-backbone mean error per method at each K)

| method | K4 | K9 | K18 | K36 | K72 | K4->K72 |
|---|---|---|---|---|---|---|
| svr | 5.98 | 5.62 | 5.71 | 5.39 | 5.42 | +0.57 |
| svr_embed | 5.89 | 5.27 | 5.16 | 5.18 | 5.24 | +0.65 |
| fc_ft | 4.83 | 4.50 | 4.47 | 4.44 | 4.44 | +0.40 |
| meta | 4.77 | 4.55 | 4.52 | 4.52 | 4.51 | +0.26 |
| meta_adv | 4.85 | 4.69 | 4.62 | 4.66 | 4.63 | +0.22 |

Error drops fastest by K9-K18 then plateaus; K36/72 add little. Best overall: face_only_mobile_vit + svr_embed @K18 = 3.10 cm.
