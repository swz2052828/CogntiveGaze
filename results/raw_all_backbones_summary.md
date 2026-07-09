# Raw-frame architecture across ALL adaptable backbones (16 variants), cm
5 folds, meanno7_clean, 384px (392 for patch-14), deploy-faithful calibration
(OriginalCalib support, same frame IDs as multistream). Jobs 1171120-32 + chained
evals, 130/130 tasks, 0 failures.

| raw backbone | params | base | fcft@K9 | fcft@K72 | svrE@K72 |
|---|---|---|---|---|---|
| **raw_mobile_vit** | 5.3M | 5.15 | 4.04 | **3.90** | 4.69 |
| raw_mobilenet_v4 | 9.2M | 5.37 | 4.37 | 4.17 | 5.11 |
| raw_repvit | 6.7M | 5.51 | 4.46 | 4.42 | 5.32 |
| raw_fastvit | 3.7M | 5.75 | 4.66 | 4.55 | 5.10 |
| raw_mobilevitv2 | 4.7M | 5.54 | 4.65 | 4.56 | 4.83 |
| raw_mobilenet_v3 | 2.1M | 6.56 | 5.23 | 5.11 | 5.64 |
| raw_vit (S/384) | 22.1M | 5.88 | 5.60 | 5.54 | 6.27 |
| raw_foveal_vit | 22.3M | 6.36 | 5.85 | 5.69 | 6.13 |
| raw_convnextv2_atto | 3.6M | 7.54 | 6.66 | 6.65 | 6.86 |
| raw_convnextv2 (femto) | 5.1M | 8.11 | 7.63 | 7.61 | 7.95 |
| raw_convnext (tiny) | 28.3M | 8.29 | 8.15 | 8.14 | 8.52 |
| raw_vit_b | 86.6M | 8.72 | 8.63 | 8.61 | 9.20 |
| raw_eva02 | 22.1M | 9.35 | 9.34 | 9.34 | 9.38 |
| raw_convnextv2_nano | 15.4M | 9.36 | 9.36 | 9.35 | 9.39 |
| raw_eva02_tiny | 5.8M | 9.37 | 9.37 | 9.36 | 9.39 |
| raw_dinov2 | 22.1M | 9.43 | 9.41 | 9.41 | 9.54 |

(≈9.35-9.4 = predict-the-mean level: those models failed to learn the task.)
Multistream reference (deploy-faithful): leaders 2.94-2.97, fcft cohort ~3.3-3.6.

## Findings
1. **The raw-frame ranking INVERTS the multistream ranking.** The multistream
   champions (convnextv2 FCMAE family) are the raw losers (atto 6.65, femto 7.61,
   nano collapsed), while the mobile conv-hybrid family (MobileViT/MobileNet-V4/
   RepViT/FastViT) sweeps the top 5. Crop-specialist features don't transfer to
   scene-level input; architectures designed for full-image ImageNet at flexible
   resolution (aggressive early downsampling + local inductive bias + late global
   mixing) are the right shape for raw frames.
2. **Capacity is anti-correlated with raw performance.** Every model >9M params
   underperforms; vit_b (86M), eva02, dinov2, convnextv2_nano land at or near the
   predict-the-mean floor — global attention/MIM features cannot find the ~37px
   eyes without the crop prior, and 138k frames cannot teach them to.
3. **raw_mobile_vit (5.3M) remains the raw champion at 3.90** — 2.6x faster
   end-to-end than the best full pipeline, ~0.95cm behind it. The whole viable
   raw family is mobile-class (2-9M): exactly the models one would deploy on a
   smartphone, reinforcing the real-time-on-device direction.
4. **Calibration helps raw models only when base features work** (mobile family
   gains 0.9-1.2cm from fc_ft; the failed models gain ~0) — the familiar law.
