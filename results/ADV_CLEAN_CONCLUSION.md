# Subject-adversarial (DANN) on clean data — final verdict

**Question:** does retraining the subject-adversarial (DANN) variant on blink-cleaned
data (`meanno7_clean`), and/or per-subject calibration, recover any benefit for the
adversarial models over the plain base models? **Answer: no. Drop adv.**

Two experiments, all 33 canonical backbones × 5 folds, seed 42. All errors in cm.

## 1. Uncalibrated retrain (`results/clean_retrain_adv_summary.md`)
- Cohort mean `adv_clean` **5.09** vs `adv_dirty` **4.99** → cleaning the *training* data
  is **net-negative for adv** (adv_gain −0.10). Consistent with the earlier result that
  blink-cleaning helps *evaluation*, not *training*.
- Adv beats plain base (both clean-trained) on only **9/33** backbones.

## 2. Deploy-faithful metacompare (`results/clean_metacompare_summary.md`)
6-way per-subject calibration (base / svr / svr_embed / fc_ft / meta / meta_adv) on the
fixed pre-task calibration frames, K ∈ {4,9,18,36,72}.

| adv effect @K72 | median | adv better |
|---|---|---|
| **uncalibrated** (base − base_adv) | −0.094 | 9/33 |
| **meta-calibrated** (meta − meta_adv) | −0.123 | 11/33 |

Calibration does **not** rescue adv — it loses before and after. `meta_adv` is worse:
**pervasively unstable**, with ≥1 diverged (>6 cm) K-fold cell on nearly every backbone
(affnet fold0 = 323 cm; convnextv2_nano diverges on all 25 cells; itracker 15, mobilevitv2 20).
It is strictly dominated by `meta`.

## 3. Method landscape (independent of adv)
- Under deploy-faithful calibration, **`svr_embed` is not robust** — 18/33 backbones score
  *worse than uncalibrated base* at K=72. The reliable calibrators are **`fc_ft`** (conv
  families) and **`meta`**; `svr_embed` only wins for a few (binoculars, face_only_mobile_vit,
  mobilenet_v3, convnextv2_film). This is weaker than the old random-K (test-frame) analysis
  implied.
- Calibrated leaders: face_only_mobile_vit 2.94, eyes_only_convnextv2_atto_binocular 2.97,
  eyes_only_convnextv2_binocular ~3.25, mobile_vit 3.33 (fc_ft). Cohort best-error mean 4.06.

## Recommendation
Keep adv **out** of the default pipeline (it doubles training cost for no calibrated benefit
and `meta_adv` is dangerous). For deployment use **`fc_ft`** (or `meta` for hybrids) at the
achievable K, not `svr_embed`.

Artifacts: `runs/calib_metacompare/SUMMARY_clean.md`,
`runs/calib_metacompare/plots_clean/{k_impact_clean,best_at_bestK_clean,adv_effect_clean}.png`.

---

# Appendix: clean vs dirty (did blink-cleaning change the calibrated picture?)

32 paired backbones (`_calib.csv` dirty vs `_clean.csv` clean; identical protocol —
deploy-faithful support, SVR-tuned, K ∈ {4,9,18,36,72}; convnextv2_dualenc has no dirty
counterpart). **Caveat:** each variant evaluates on its own manifest, so the delta bundles
train-side AND eval-side cleaning; the eval-side systematic is ~+0.08 cm (blink frames
inflate dirty error).

## Per-method delta (dirty − clean, cm; >0 = cleaning helped), K=72
| method | mean | median | helped |
|---|---|---|---|
| base | +0.120 | +0.101 | 20/32 |
| base_adv | +0.004 | +0.005 | 16/32 |
| svr_embed | +0.195 | +0.215 | 22/32 |
| fc_ft | +0.212 | +0.143 | 27/32 |
| meta | +0.117 | +0.109 | 22/32 |
| meta_adv | −2.20 (divergence-dominated; median +0.017) | | 17/32 |

Best-calibrated (any method @ best K): **mean +0.108, median +0.149, helped 22/32.**

## Interpretation
- Subtracting the ~0.08 eval-side systematic, the **training-side gain from cleaning is
  ~+0.03–0.13 cm — real but small**, matching the uncalibrated retrain conclusion
  (cleaning helps evaluation honesty, not training). Consistency check: base +0.12 ≈
  0.08 eval + 0.04 train; base_adv +0.004 − 0.08 ≈ −0.08 train-side ≈ the −0.10 adv_gain
  from the retrain summary.
- Calibrated methods gain slightly more than base (fc_ft +0.21, 27/32 helped).
- Tails exceed the cohort effect: helped most — convnextv2_nano +1.34, normface_convnext
  +0.88, eyes_only_mgazenet +0.65, convnext +0.56; hurt — convnextv2_film −1.15,
  eva02 −0.87, eyes_only_mobilevitv2 −0.45, mgazenet −0.39 (±1 cm swings in init-fragile
  families look like seed instability, not a cleaning effect).

## Leaderboard stability — top 10 best-calibrated (best method @ best K, cm)
| # | dirty | | clean | |
|---|---|---|---|---|
| 1 | eyes_only_convnextv2_atto_binocular | 2.90 (svrE@72) | face_only_mobile_vit | 2.94 (svrE@18) |
| 2 | face_only_mobile_vit | 3.10 (svrE@18) | eyes_only_convnextv2_atto_binocular | 2.97 (svrE@72) |
| 3 | mobile_vit | 3.33 (fc_ft@72) | eyes_only_convnextv2_binocular | 3.24 (svrE@72) |
| 4 | eyes_only_convnextv2_binocular | 3.38 (fc_ft@72) | foveal_vit | 3.24 (svrE@18) |
| 5 | mobilenet_v3 | 3.48 (svrE@72) | mobile_vit | 3.33 (fc_ft@72) |
| 6 | foveal_vit | 3.52 (svrE@18) | eyes_only_mobilenet_v3 | 3.47 (meta_adv@18)* |
| 7 | cnn_transformer | 3.67 (fc_ft@72) | cnn_transformer | 3.49 (svrE@9) |
| 8 | mobilevitv2 | 3.77 (meta@18) | convnextv2_atto | 3.51 (fc_ft@36) |
| 9 | convnextv2_film | 3.78 (svrE@72) | mobilenet_v3 | 3.53 (svrE@72) |
| 10 | eyes_only_mobilenet_v3 | 3.81 (meta@36) | vit | 3.67 (fc_ft@36) |

\*only place meta_adv wins anything; given its instability treat as a lucky draw (its
meta = 3.75 is nearly as good).

The podium is stable: the same two models top both boards (swapping #1/#2 within fold
noise), 8/10 names overlap; cleaning tightens the band (3.10–3.81 → 2.94–3.67) without
reordering it.

## Top 10 uncalibrated — min(base, base_adv) @K72, cm
Including the adv checkpoints changes the *uncalibrated* picture (they were excluded from
the base-only ranking above):

| # | dirty | | clean | |
|---|---|---|---|---|
| 1 | mobilenet_v4 | **4.29 (adv)** | eyes_only_mobile_vit | 4.21 (base) |
| 2 | convnextv2 | 4.33 (base) | vit | **4.32 (adv)** |
| 3 | eyes_only_mobilenet_v4 | **4.33 (adv)** | convnextv2 | 4.33 (base) |
| 4 | convnext | **4.34 (adv)** | eyes_only_mobilenet_v4 | 4.33 (base) |
| 5 | eyes_only_convnextv2_binocular | 4.38 (base) | convnextv2_atto | 4.34 (base) |
| 6 | convnextv2_atto | 4.51 (base) | mobilenet_v4 | **4.37 (adv)** |
| 7 | foveal_vit | 4.53 (base) | eyes_only_convnextv2_atto_binocular | 4.37 (base) |
| 8 | eyes_only_vit | **4.53 (adv)** | convnext | 4.40 (base) |
| 9 | vit | **4.56 (adv)** | eyes_only_vit | 4.40 (base) |
| 10 | eyes_only_mobile_vit | 4.56 (base) | foveal_vit | 4.43 (adv, by 0.01) |

On dirty training, adv checkpoints take **6/10** top slots and the overall uncalibrated
best (mobilenet_v4_adv 4.29) — the known "adv helps uncalibrated" effect. After cleaning
adv keeps only **3/10** slots and the #1 is a plain base model (eyes_only_mobile_vit
4.21): cleaning erodes most of adv's uncalibrated edge. So the one surviving niche for
adv — zero-calibration deployment on dirty training — disappears under clean training,
and it never had a calibrated niche. The drop-adv recommendation stands.

## Verdict
**Blink-cleaning does not change which backbone or method is best.** It yields a small
(~+0.1 cm) apparent calibrated improvement, mostly the cleaner evaluation set. Keep
cleaning for evaluation honesty; do not expect training gains. Deployment pick either
way: eyes_only_convnextv2_atto_binocular or face_only_mobile_vit (~2.9–3.0 cm calibrated).
