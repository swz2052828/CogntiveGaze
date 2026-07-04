# Clean metacompare — leaderboard (best method @ best K), cm

| backbone | base | svr_embed | fc_ft | meta | **best @K** |
|---|---|---|---|---|---|
| face_only_mobile_vit | 5.43 | 2.94(K18) | 3.54(K72) | 3.79(K72) | **svr_embed 2.94@K18** |
| eyes_only_convnextv2_atto_binocular | 4.37 | 2.97(K72) | 3.26(K72) | 4.22(K72) | **svr_embed 2.97@K72** |
| eyes_only_convnextv2_binocular | 4.59 | 3.24(K72) | 3.25(K72) | 4.33(K72) | **svr_embed 3.24@K72** |
| foveal_vit | 4.44 | 3.24(K18) | 3.34(K36) | 4.01(K36) | **svr_embed 3.24@K18** |
| mobile_vit | 5.12 | 3.66(K72) | 3.33(K72) | 3.64(K18) | **fc_ft 3.33@K72** |
| eyes_only_mobilenet_v3 | 5.00 | 3.83(K72) | 4.12(K72) | 3.67(K18) | **meta_adv 3.47@K18** |
| cnn_transformer | 4.71 | 3.49(K9) | 3.68(K72) | 4.13(K72) | **svr_embed 3.49@K9** |
| convnextv2_atto | 4.34 | 4.12(K72) | 3.51(K36) | 4.25(K18) | **fc_ft 3.51@K36** |
| mobilenet_v3 | 5.04 | 3.53(K72) | 4.03(K18) | 3.69(K18) | **svr_embed 3.53@K72** |
| vit | 4.38 | 3.94(K72) | 3.67(K36) | 4.06(K72) | **fc_ft 3.67@K36** |
| eyes_only_vit | 4.40 | 4.61(K18) | 3.80(K72) | 4.06(K36) | **fc_ft 3.80@K72** |
| convnext | 4.40 | 3.85(K18) | 3.80(K36) | 4.34(K36) | **fc_ft 3.80@K36** |
| mobilevitv2 | 4.78 | 5.80(K72) | 4.04(K36) | 3.80(K9) | **meta 3.80@K9** |
| eyes_only_mobile_vit | 4.21 | 5.36(K9) | 4.00(K72) | 3.86(K36) | **meta 3.86@K36** |
| convnextv2 | 4.33 | 4.49(K36) | 3.88(K72) | 4.03(K72) | **fc_ft 3.88@K72** |
| eyes_only_eva02_tiny | 4.93 | 3.90(K72) | 4.01(K72) | 4.94(K36) | **svr_embed 3.90@K72** |
| repvit | 4.67 | 4.02(K72) | 4.30(K36) | 3.91(K72) | **meta 3.91@K72** |
| convnextv2_dualenc | 4.42 | 4.33(K18) | 3.95(K18) | 4.15(K72) | **fc_ft 3.95@K18** |
| eyes_only_mgazenet | 5.11 | 4.47(K36) | 4.42(K36) | 4.02(K72) | **meta 4.02@K72** |
| eyes_only_mobilenet_v4 | 4.33 | 6.57(K9) | 4.43(K36) | 4.22(K36) | **meta_adv 4.07@K72** |
| mobilenet_v4 | 4.51 | 5.54(K9) | 4.11(K72) | 4.57(K72) | **fc_ft 4.11@K72** |
| normface_convnext | 4.60 | 4.59(K72) | 4.12(K36) | 4.59(K9) | **fc_ft 4.12@K36** |
| eyes_only_fastvit | 4.46 | 5.07(K9) | 4.66(K72) | 4.31(K72) | **meta_adv 4.23@K9** |
| eyes_only_convnextv2_atto | 4.61 | 4.77(K18) | 4.25(K36) | 4.50(K36) | **fc_ft 4.25@K36** |
| eyes_only_convnextv2 | 4.47 | 4.80(K18) | 4.41(K36) | 4.28(K72) | **meta 4.28@K72** |
| eyes_only_mobilevitv2 | 5.46 | 5.08(K18) | 4.99(K72) | 4.53(K36) | **meta 4.53@K36** |
| mgazenet | 4.95 | 4.98(K9) | 4.66(K72) | 4.72(K72) | **fc_ft 4.66@K72** |
| dinov2 | 5.36 | 5.74(K4) | 4.76(K36) | 5.38(K36) | **fc_ft 4.76@K36** |
| eva02 | 5.09 | 5.15(K9) | 4.77(K36) | 5.14(K18) | **fc_ft 4.77@K36** |
| itracker | 5.93 | 6.22(K4) | 4.82(K72) | 5.12(K18) | **fc_ft 4.82@K72** |
| convnextv2_film | 5.56 | 4.93(K18) | 5.05(K72) | 5.48(K36) | **svr_embed 4.93@K18** |
| affnet | 5.79 | 5.47(K9) | 5.26(K36) | 5.11(K72) | **meta 5.11@K72** |
| convnextv2_nano | 7.36 | 7.47(K72) | 7.11(K72) | 7.35(K36) | **fc_ft 7.11@K72** |

| vivit (temporal, window 8) | 4.72 | 5.20(K72) | 4.50(K72) | 4.50(K9) | **fc_ft 4.50@K72** |
