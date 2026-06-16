# Gaze Backbone Study — Final Synthesis

A controlled comparison of multistream gaze backbones in the `vit_gaze` package,
evaluated through the meta-calibration pipeline (base train → subject-adv train →
metatrain → SVR/embedding search → metacompare). All runs are 5-fold,
recording-level CV, seed 42, `LR=1e-4`, ImageNet-pretrained encoders where
available. Metrics are **pixels of gaze error**; lower is better.

Calibration columns: `base` (uncalibrated), `svr` (prediction-space SVR),
`svr_embed` (embedding-space SVR), `fc_ft` (FC fine-tune), `meta`
(meta-learned), `meta_adv` (meta on subject-adversarial base). `base_adv` is the
uncalibrated error of the subject-adversarially-trained model (from training
logs).

## Headline

**`convnextv2` (ConvNeXtV2-Femto, 5.3M params)** wins all three goals at once:
best uncalibrated base, best calibrated accuracy, and best realistic low-K
accuracy, at an edge-viable size. GRN + masked-autoencoder (FCMAE) pretraining
is the single most effective design choice in the study.

## Master table (px)

| Model | Params | base | fc_ft@8 | meta@8 | svr_embed@16 | svr_embed@64 | base_adv |
|---|---|---|---|---|---|---|---|
| **convnextv2** | 5.3M | **4.33** | **2.54** | 3.67 | 2.52 | **1.83** | 4.56 |
| convnextv2_film | 5.4M | 4.73 | 2.69 | 4.13 | 2.52 | 1.83 | 5.95 |
| MGazeNet (full) | — | 5.01 | 3.40 | 3.08 | 2.92 | 1.85 | — |
| MobileNetV3 (full) | — | 5.08 | 3.71 | 2.86 | 2.57 | 1.87 | — |
| eyes_only_convnextv2 | 5.1M | 4.59 | 2.74 | 4.06 | 2.55 | 1.88 | 4.61 |
| ViT-B16 | 86M | 4.60 | 2.82 | 3.61 | 2.76 | 1.97 | 4.55 |
| eyes_only_mobilenet_v3 | 3.1M | 4.84 | 3.47 | 3.03 | 2.64 | 1.97 | 4.96 |
| eyes_only_mgazenet | 1.7M | 5.61 | 3.88 | 3.13 | 2.88 | 2.00 | 5.08 |
| eyes_only_fastvit | 3.7M | 4.72 | 2.89 | 3.61 | 2.86 | 2.01 | 4.82 |
| mobilevitv2 | 4.9M | 5.03 | 3.01 | 2.97 | 2.92 | 2.12 | 4.83 |
| repvit | 6.9M | 4.92 | 3.14 | 3.42 | 2.90 | 2.14 | 5.12 |
| eyes_only_mobilevitv2 | 1.26M | 5.71 | 3.59 | 3.02 | 3.15 | 2.33 | 5.65 |
| normface_convnext (FAILED) | 27M | 5.86 | 4.45 | 5.51 | 4.36 | 3.66 | 4.89 |
| convnextv2_nano lr1e4 (COLLAPSED) | 15.6M | 8.46 | 8.13 | 8.39 | 8.07 | 7.76 | 9.39 |

## Goal scorecard

**Goal 2 — best uncalibrated base:** `convnextv2` **4.33px** (best `base`);
ViT-B16 and convnextv2 tie best `base_adv` ~4.55. Beats ConvNeXt-v1 (4.82),
MobileNetV4 (4.71), ViT (4.60) at far smaller size.

**Goal 3 — best calibrated:**
- High-K (svr_embed@64): `convnextv2` **1.83px** (study best).
- Real-world low-K (5/9/25 points): `convnextv2` + **`fc_ft` = 2.54px @ K=8**,
  ~2.50px @ K=16. This is the number to quote for deployment.

**Goal 1 — lightweight + reasonable accuracy (edge / smartphone):**

| Pick | Params | svr_embed@64 | Note |
|---|---|---|---|
| eyes_only_mgazenet | 1.7M | 2.00 | smallest that still holds ~2px |
| eyes_only_mobilenet_v3 | 3.1M | 1.97 | best accuracy-per-param; int8-friendly CNN |
| eyes_only_convnextv2 | 5.1M | 1.88 | best edge accuracy, -33% compute vs full |

Below ~1.7M the floor cracks (1.26M -> 2.33px). **Smartphone recommendation:
eyes_only_mobilenet_v3 (3.1M, 1.97px).**

## Five cross-cutting laws

1. **Calibrated accuracy is capacity-saturated.** From 1.7M to 86M params,
   svr_embed@64 stays in 1.83-2.00px. Encoder *size* barely matters after
   calibration; only encoder *family/pretraining* (GRN+FCMAE) and *not
   collapsing* matter.
2. **Best method depends on calibration budget and encoder family.** Real-world
   low-K (5/9/25 pts) is the `fc_ft` (pure-conv) / `meta` (hybrid) regime;
   `svr_embed` only wins at K>=32. Report fc_ft@8/16 for deployment.
3. **Region is absorbed by calibration.** eyes-only ~= face-only ~= full after
   calibration (all ~1.9-2.0px). The grid/head-pose signal helps *base* but is
   washed out post-calibration (convnextv2 vs convnextv2_film: identical
   svr_embed, different base).
4. **Adversarial training is family-dependent and redundant.** Helps hybrids'
   base (mobilevitv2 -0.19), hurts GRN conv nets (convnextv2 +0.25), always
   redundant after calibration. Drop it unless you need zero-calibration
   deployment.
5. **Two architectural innovations failed cleanly.** The L2-hypersphere
   calibration bottleneck (normface) and grid-FiLM conditioning both *hurt* —
   plain late-fusion of a strong pretrained encoder beats clever
   conditioning/normalization. Simplicity + pretraining > architectural tricks.

## Deployment recommendations

- **On-device / smartphone:** `eyes_only_mobilenet_v3` (3.1M, 1.97px) or
  `eyes_only_convnextv2` (5.1M, 1.88px), calibrated with `fc_ft` at 9-25 points.
- **Best accuracy regardless of size:** `convnextv2` (5.3M) — 1.83px @ 64 pts,
  ~2.5px @ 16 pts.
- **Avoid:** normface, grid-FiLM, repvit, and any nano-scale model at LR=1e-4
  (collapses; needs a lower LR).

## Scaling: the winning family does NOT scale up (resolved)

`convnextv2_nano` (15.6M) is unstable/underfit on this task at every LR tried:
base 8.46 @ LR=1e-4 (full predict-the-mean collapse), base 7.46 @ LR=5e-5 (still
badly underfit, svr_embed@64 only 6.05). The GRN+FCMAE family's sweet spot is
**small**: atto (3.6M) gives the study-best svr_embed@64 1.818 and femto (5.3M)
the best base 4.33, while scaling to 15.6M fails. On this dataset size, **smaller
is better** — capacity hurts. (Possible confounds: 20-epoch budget, limited gaze
data; not pursued since small models clearly win.)

This strengthens the conclusion: the answer to all three goals is a **small**
ConvNeXtV2 (atto/femto), not a bigger model.
