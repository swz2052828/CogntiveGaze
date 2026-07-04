# Eval-side optimizations #1 (fc_ft tuning) and #2 (LOO auto-selection) — results

8 backbones × 5 folds, clean checkpoints + clean calib adapters, deploy-faithful
calib support, K ∈ {9, 72}. Errors = clean-frame mask, cm, fold-mean.
Source: `runs/calib_extras/extras_<bb>.csv` (jobs 1170062-69, 40/40, 0 failures).

## Cohort means
| K | base | fcft_def | fcft_tuned | svr_embed | meta | **auto** | oracle |
|---|---|---|---|---|---|---|---|
| 9  | 4.707 | **3.709** | 3.915 | 3.836 | 4.057 | **3.671** | 3.208 |
| 72 | 4.707 | **3.597** | 3.848 | 3.615 | 4.023 | 3.606 | 3.053 |

## #1 fc_ft hyperparameter tuning: NEGATIVE — keep Zhu defaults
Leak-free grid (45 combos, fit on train subjects' calib→task transfer) picked
lr=2e-5 / steps=20 / wd=5e-3 in 65/80 cells — a conservative setting that wins
on TRAINING subjects but under-adapts held-out subjects: fcft_tuned is
**+0.25 cm worse** than the untouched defaults (3.85 vs 3.60 @K72). fc_ft HPs
do not transfer across subjects the way SVR HPs did; the Zhu defaults
(lr 5e-5, 20 steps, wd 5e-4) are already at a generalization sweet spot.

## #2 LOO auto-selection: small win at K=9, protective everywhere
- K=9: auto (3.671) beats every fixed method including fcft_def (3.709).
- K=72: neutral vs fcft_def (3.606 vs 3.597).
- Real value is **protection on method-mismatched backbones**: mobilenet_v3
  auto 3.75 vs fcft_def 4.03 (its best is svr_embed/meta); face_only_mobile_vit
  3.28 vs 3.54. Auto never blows up (unlike committing to one method).
- Oracle (3.05-3.21) shows ~0.5 cm of per-subject selection headroom; LOO
  captures only ~10% of it — 9-point LOO is a noisy selector.
- Caveat: auto's menu contained fcft_TUNED (handicapped by #1). With fcft_def
  in the menu the auto numbers would improve slightly.

## Recommendation
Deploy fc_ft with Zhu defaults; add LOO auto-selection when K is small or the
backbone's best method is unknown. Do not tune fc_ft HPs on training subjects.

## #2 CORRECTED (re-run with fcft_def in the LOO menu, runs/calib_extras2)
The original auto column selected among {fcft_TUNED, svr_embed, meta} — handicapped
by #1's negative result. With the proper menu {fcft_def, svr_embed, meta}:

| K | base | fcft_def | svr_embed | meta | **auto** | oracle |
|---|---|---|---|---|---|---|
| 9  | 4.706 | 3.707 | 3.836 | 4.057 | **3.601** | 3.199 |
| 72 | 4.706 | 3.597 | 3.615 | 4.023 | **3.511** | 3.033 |

**Upgraded verdict: LOO auto-selection beats every fixed method at BOTH K**
(-0.11 @K9, -0.09 @K72 vs the best fixed method), capturing ~20%% of the oracle
headroom. It is the only eval-side optimization with a positive cohort effect.
Recommend as the default deployment calibrator-picker.

## Better-selector attempt: cycle-CV — NEGATIVE (K=72, runs/calib_extras3)
Hypothesis: validating across the two enrollment sweep cycles (fit cycle 1 → score
cycle 2, both directions) better matches the calib→task temporal shift than 9-point
LOO. Result: **auto_cv 3.614 vs auto(LOO) 3.515** — cycle-CV loses the entire LOO
edge (worst on convnextv2: 4.57 vs 4.02, where it over-picked svr_embed).

Diagnosis: a fit-set composition bias outweighs the validation-shift benefit. CV
fits each candidate on 36 single-cycle frames, but the deployed method is fitted
on all 72 two-cycle frames; methods that need both cycles' pose diversity (fc_ft,
meta) are systematically under-scored, tilting selection toward svr_embed (picks:
33 vs LOO's 24). LOO's 64-frame fits are near-identical to the deployed 72-frame
fit, so despite noisier validation its selection is less biased. **Selector rule of
thumb: keep the CV fit-set as close as possible to the deployed fit-set; accept
validation noise over fit-set mismatch.** LOO auto stays the recommendation; the
~0.48cm oracle gap remains open (untested: per-point outlier screening).
