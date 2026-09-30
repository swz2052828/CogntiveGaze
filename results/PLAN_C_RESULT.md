# Plan C: strict non-inferiority conclusions on the utility axis

Replaces "the error recovers" with a declared test. **Unit of analysis is the
subject, not the fold.** Recording-level 5-fold CV gives each of the 17
recordings exactly one held-out measurement per (backbone, operator, condition,
K), so every operator is paired within subject against `none` and the
subject random intercept cancels out of the difference instead of inflating its
interval.

Metric `fc_ft`. Margin **Δ = 0.40 cm**, pre-declared from the measured
unseeded-init noise floor — **0.19–0.29° of visual angle** at the measured
80–120 cm viewing distance. One-sided non-inferiority test at α = .025, Holm
corrected within backbone, subject-resampled bootstrap CIs (20k), Cohen's dz, and
the count of subjects hurt. Source `results/anon_stage4_subject_all.csv`
(1,836 rows), regenerated from the run logs and verified to reproduce the
fold-level CSVs to **0.000045 cm**.

## Why the pairing is the whole game

| backbone | between-subject SD | residual SD | ICC |
|---|---|---|---|
| convnextv2 | 1.006 cm | 0.110 cm | **0.988** |
| mobile_vit | 1.166 cm | 0.216 cm | **0.967** |
| itracker | 1.806 cm | 0.469 cm | **0.937** |
| face_only_mobile_vit | 1.182 cm | 0.711 cm | 0.734 |

73–99% of the variance is *which subject this is*. An unpaired comparison would
be testing a 0.09 cm effect against a 1.2 cm between-subject spread and would
conclude nothing. This is also why n = 17 subjects beats n = 5 folds: the folds
average 3–4 recordings each and discard the pairing the design actually has.

## Verdicts on the same 16 cells (4 backbones × 4 operators)

| | non-inferior | inferior | inconclusive |
|---|---|---|---|
| K=72, condition (ii) — clean enrolment | 7/16 | 4 | 5 |
| **K=72, condition (iii) — matched enrolment** | **11/16** | **1** | 4 |
| K=9, condition (ii) | 7/16 | 4 | 5 |
| **K=9, condition (iii)** | **11/16** | **1** | 4 |

Matched enrolment moves four cells from inferior/inconclusive to non-inferior and
leaves exactly one inferior verdict, **identically at both enrolment sizes**.

## Condition (iii), per cell

| backbone | operator | K=72 mean d | 95% CI | K=9 mean d | 95% CI | verdict | p(Holm) |
|---|---|---|---|---|---|---|---|
| **mobile_vit** | blur0.10 | +0.161 | [+0.04,+0.27] | +0.140 | [+0.01,+0.25] | **non-inferior** | 0.001 |
| | blur0.20 | +0.126 | [−0.00,+0.25] | +0.120 | [−0.01,+0.24] | **non-inferior** | 0.001 |
| | blur0.50 | +0.069 | [−0.08,+0.20] | +0.066 | [−0.10,+0.21] | **non-inferior** | 0.001 |
| | **blackbox** | **+0.087** | [−0.07,+0.23] | **+0.119** | [−0.06,+0.28] | **non-inferior** | 0.003 |
| **convnextv2** | blur0.10 | +0.014 | [−0.06,+0.08] | +0.016 | [−0.06,+0.10] | **non-inferior** | <0.001 |
| | blur0.20 | +0.033 | [−0.05,+0.11] | +0.033 | [−0.07,+0.14] | **non-inferior** | <0.001 |
| | blur0.50 | +0.010 | [−0.07,+0.09] | −0.003 | [−0.09,+0.09] | **non-inferior** | <0.001 |
| | blackbox | −0.062 | [−0.17,+0.03] | −0.076 | [−0.20,+0.04] | **non-inferior** | <0.001 |
| **itracker** | blur0.10 | −0.069 | [−0.19,+0.06] | −0.085 | [−0.20,+0.04] | **non-inferior** | <0.001 |
| | blur0.20 | −0.136 | [−0.29,+0.02] | −0.155 | [−0.30,−0.01] | **non-inferior** | <0.001 |
| | blur0.50 | −0.124 | [−0.30,+0.04] | −0.170 | [−0.33,−0.01] | **non-inferior** | <0.001 |
| | blackbox | +0.544 | [+0.12,+1.03] | +0.516 | [+0.03,+1.09] | inconclusive | 0.72 |
| **face_only_mobile_vit** | blur0.10 | +0.383 | [+0.12,+0.69] | +0.341 | [+0.14,+0.57] | inconclusive | 1.00 |
| | blur0.20 | +0.627 | [+0.24,+1.04] | +0.566 | [+0.23,+0.92] | inconclusive | 1.00 |
| | blur0.50 | +0.723 | [+0.22,+1.24] | +0.758 | [+0.28,+1.23] | inconclusive | 1.00 |
| | blackbox | +1.253 | [+0.57,+1.93] | +1.152 | [+0.48,+1.83] | **INFERIOR** | 1.00 |

**The claim the paper can make:** with matched anonymised enrolment, a
multistream gaze model tolerates **complete removal of the face region** —
`mobile_vit` + `blackbox`, +0.087 cm at K=72 and +0.119 cm at K=9, non-inferior
at Δ = 0.40 with Holm-corrected p ≤ 0.003, from a nine-point calibration.

**The claim it cannot make:** a face-only model does not survive this.
`face_only_mobile_vit` + `blackbox` is formally **inferior** (+1.25 cm), which is
the expected result and worth reporting — it shows the margin is capable of
rejecting, so the non-inferior verdicts elsewhere are not an artefact of a margin
set too loose.

## Condition (iii) against condition (ii), paired within subject

The direct test of the enrolment-mismatch claim. Negative = matched enrolment is
better.

| backbone | operator | (iii) − (ii) | 95% CI |
|---|---|---|---|
| mobile_vit | blur0.20 | **−0.418** | [−0.57, −0.28] |
| | blur0.50 | **−0.466** | [−0.65, −0.30] |
| | blackbox | **−0.459** | [−0.68, −0.24] |
| face_only_mobile_vit | blur0.50 | **−1.405** | [−1.81, −1.02] |
| | blackbox | **−1.293** | [−1.80, −0.78] |
| convnextv2 | blackbox | +0.155 | [−0.02, +0.34] |
| itracker | blackbox | +0.383 | [−0.07, +0.85] |
| *any backbone* | *none* | ≤ ±0.008 | control — as it must be |

Two directions, both expected. On the backbones with a real cost, matched
enrolment is better by 0.42–1.41 cm with intervals well clear of zero. On
`convnextv2` and `itracker`, matched enrolment is *worse* — because condition
(ii) was crediting them with a spurious improvement as the face was destroyed,
and matching the enrolment removes it. Neither of those two intervals excludes
zero, so the artefact removal is a direction, not a significant effect.

## Bland–Altman, `none` vs `blackbox`, condition (iii), K=72

| backbone | bias | limits of agreement | SD of differences | proportional bias |
|---|---|---|---|---|
| mobile_vit | +0.087 | [−0.544, +0.719] | 0.322 | r = −0.32, no trend |
| convnextv2 | −0.062 | [−0.485, +0.361] | 0.216 | r = −0.15, no trend |
| itracker | +0.544 | [−1.403, +2.492] | 0.994 | r = −0.03, no trend |
| face_only_mobile_vit | +1.253 | [−1.656, +4.162] | 1.484 | r = −0.02, no trend |

**This is the caveat that belongs next to the headline.** `mobile_vit`'s *mean*
effect is +0.087 cm, comfortably non-inferior — but its limits of agreement run
to **+0.72 cm**, past the margin. Non-inferiority is a statement about the
cohort, not a guarantee for an individual: an unlucky subject can lose more than
Δ even where the mean is clean. No backbone shows proportional bias, so the
spread is not a function of the subject's baseline accuracy.

## Effect sizes and who is hurt

Reported per cell rather than p-values alone. `mobile_vit`/`blur0.10` is
instructive: **dz = 0.67 with 14 of 17 subjects hurt**, and still non-inferior.
Most subjects are made slightly worse, consistently — the effect is real and
detectable, and it is *small enough not to matter* against a pre-declared margin.
That is precisely the distinction non-inferiority testing exists to draw, and it
would have been reported as a significant degradation under a naive test.

## Files

`results/plan_c/planc_cond{ii,iii}_K{9,72}.txt` and
`planc_compare_K{9,72}.txt`; subject-level source
`results/anon_stage4_subject_all.csv`.

---

## Addendum (2026-09-15): `fc_ft` is stochastic and unseeded

Found while anchoring the retrain-evaluation harness: re-running an identical
cell (mobile_vit, blur0.20, K=72, fold 0, same checkpoint, same data, same L40 GPU
model) reproduced `base` to 5×10⁻⁶ but not `fc_ft` (Δ 0.016 cm).

**Cause.** Every readout head contains `nn.Dropout` (0.1, 0.2 on vit_shared),
`_fc_ft_predict` puts the readout in `train()` for its 20 fine-tune steps, and no
code on the evaluation path calls `torch.manual_seed`. Dropout is therefore live
and unseeded during calibration. A smaller contribution comes from GPU float
nondeterminism in the cached features (`svr_embed` varies at ~4×10⁻⁴, `base` at
~1×10⁻⁴ on one of the five runs).

**Size, from five repeats of the identical cell:**

| | SD |
|---|---|
| fold mean (4 subjects) | 0.012 cm |
| per subject, pooled | **0.021 cm** |
| paired difference per subject (two independent `fc_ft` calls) | ≈0.029 cm |
| between-subject SD of the paired difference (mobile_vit, blur0.20 − none, cond iii, K=72) | 0.267 cm |

Run-to-run noise accounts for **~1.2%** of the variance of the paired differences
the non-inferiority tests are built on. It adds variance and no bias, so it
widens the intervals slightly and can only make a non-inferiority verdict harder
to reach. **No verdict in this document changes.**

Not fixed in code, deliberately: seeding `_fc_ft_predict` now would make new runs
non-comparable in noise terms with every existing `fc_ft` number. If the whole
utility axis is ever re-run, seed it then.
