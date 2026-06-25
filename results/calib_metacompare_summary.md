# Deploy-faithful calibration — cross-backbone summary (mean over folds x K)

| backbone | base | base_adv | svr | svr_embed | fc_ft | meta | meta_adv | best |
|---|---|---|---|---|---|---|---|---|
| mobile_vit | 4.98 | 4.73 | 4.43 | 4.44 | 3.50 | 3.80 | 3.84 | fc_ft 3.50 |
| face_only_mobile_vit | 5.04 | 5.36 | 4.20 | 3.59 | 3.81 | 4.14 | 4.47 | svr_embed 3.59 |
| foveal_vit | 4.53 | 4.57 | 4.31 | 3.85 | 3.80 | 4.24 | 4.39 | fc_ft 3.80 |
| eyes_only_mobilenet_v3 | 4.84 | 4.99 | 4.81 | 4.33 | 4.31 | 3.94 | 4.28 | meta 3.94 |
| eyes_only_vit | 4.62 | 4.53 | 4.84 | 4.62 | 3.98 | 4.38 | 4.46 | fc_ft 3.98 |
| mobilevitv2 | 5.03 | 4.84 | 6.10 | 5.25 | 4.53 | 4.02 | 4.06 | meta 4.02 |
| eva02 | 4.91 | 5.73 | 5.11 | 4.73 | 4.06 | 5.01 | 5.60 | fc_ft 4.06 |
| convnextv2 | 4.33 | 4.55 | 5.08 | 4.83 | 4.12 | 4.27 | 4.16 | fc_ft 4.12 |
| eyes_only_mobilevitv2 | 5.71 | 5.63 | 6.19 | 6.10 | 5.04 | 4.19 | 4.66 | meta 4.19 |
| repvit | 4.92 | 5.13 | 4.65 | 4.63 | 4.76 | 4.20 | 4.31 | meta 4.20 |
| eyes_only_eva02_tiny | 4.92 | 4.86 | 5.66 | 5.24 | 4.25 | 5.02 | 5.03 | fc_ft 4.25 |
| eyes_only_convnextv2 | 4.59 | 4.62 | 6.17 | 5.40 | 4.52 | 4.57 | 4.32 | meta_adv 4.32 |
| eyes_only_mobilenet_v4 | 4.49 | 4.33 | 7.60 | 7.59 | 4.77 | 4.39 | 4.45 | meta 4.39 |
| eyes_only_fastvit | 4.72 | 4.82 | 6.51 | 6.83 | 4.84 | 4.48 | 4.43 | meta_adv 4.43 |
| convnext | 4.82 | 4.34 | 5.59 | 5.13 | 4.70 | 4.79 | 4.58 | meta_adv 4.58 |
| eyes_only_mgazenet | 5.61 | 5.06 | 6.57 | 5.55 | 4.97 | 4.83 | 4.75 | meta_adv 4.75 |
| dinov2 | 5.23 | 6.00 | 5.48 | 5.78 | 4.94 | 5.40 | 5.95 | fc_ft 4.94 |
| normface_convnext | 5.86 | 4.91 | 6.47 | 6.36 | 5.76 | 5.94 | 5.01 | meta_adv 5.01 |
| itracker | 5.95 | 6.53 | 7.10 | 7.35 | 5.56 | 5.25 | 6.35 | meta 5.25 |
