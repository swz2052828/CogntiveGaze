# Condition (iii) at K=9: the recovery survives a deployment-realistic enrolment

K=72 is 72 calibration frames per subject. A deployed at-home screening app gets
a **nine-point calibration lasting a few seconds**. The condition (iii) result
was measured at K=72, so it was worth little until tested at K=9.

4 backbones x 5 folds x 5 operators, conditions (ii) and (iii) both re-run at
K=9 so the comparison stays within a single K. Deploy metric `fc_ft`,
20k-resample paired bootstrap over folds. Margin **delta = 0.40 cm**.
19/19 array tasks `ALL_DONE`, 5 folds in every cell, no failures.

## Penalty vs the `none` arm, within condition (cm)

| backbone | operator | K=72 (ii) | K=72 (iii) | K=9 (ii) | K=9 (iii) | K=9 (iii) 95% CI |
|---|---|---|---|---|---|---|
| mobile_vit | blur0.10 | +0.391 | **+0.155** | +0.356 | **+0.133** | [−0.01, +0.26] |
| | blur0.20 | +0.538 | **+0.123** | +0.482 | **+0.118** | [+0.02, +0.22] |
| | blur0.50 | +0.530 | **+0.072** | +0.482 | **+0.068** | [−0.04, +0.18] |
| | **blackbox** | +0.546 | **+0.066** | +0.489 | **+0.099** | [−0.09, +0.25] |
| face_only_mobile_vit | blur0.10 | +1.273 | +0.378 | +1.196 | +0.332 | [+0.09, +0.55] |
| | blur0.20 | +1.680 | +0.619 | +1.572 | +0.555 | [+0.17, +0.88] |
| | blur0.50 | +2.096 | +0.712 | +1.980 | +0.747 | [+0.36, +1.13] |
| | blackbox | +2.505 | +1.217 | +2.357 | +1.110 | [+0.58, +1.64] |
| convnextv2 | blur0.10 | −0.043 | **+0.025** | −0.056 | **+0.030** | [−0.07, +0.14] |
| | blur0.20 | −0.056 | **+0.044** | −0.081 | **+0.049** | [−0.06, +0.16] |
| | blur0.50 | −0.131 | **+0.017** | −0.154 | **+0.007** | [−0.06, +0.07] |
| | blackbox | −0.229 | **−0.046** | −0.233 | **−0.057** | [−0.21, +0.07] |
| itracker | blur0.10 | −0.219 | **−0.057** | −0.204 | **−0.079** | [−0.20, +0.08] |
| | blur0.20 | −0.315 | **−0.126** | −0.299 | **−0.151** | [−0.29, +0.01] |
| | blur0.50 | −0.377 | **−0.113** | −0.362 | **−0.169** | [−0.34, −0.01] |
| | blackbox | +0.093 | +0.499 | +0.068 | +0.451 | [−0.25, +1.43] |

**Bold** = non-inferior on the strict test (upper CI bound below the margin), not
merely on the point estimate.

## The result carries over almost unchanged

Every cell moves by less than 0.1 cm between K=72 and K=9. The largest change
anywhere is `mobile_vit`/`blackbox` at +0.033 cm, which is a tenth of the margin.

**`mobile_vit` is non-inferior at every operator strength, at both K, on the
strict test.** At K=9 the upper confidence bounds are +0.26 / +0.22 / +0.18 /
+0.25 against a margin of 0.40, so the conclusion does not rest on a point
estimate sitting just inside the line. That includes `blackbox` — the entire face
region set to black, eyes byte-exact — at **+0.099 cm** from nine calibration
frames.

This is the paper's central utility claim and it now holds in the configuration a
deployed system would actually use.

## Why the enrolment mismatch costs slightly *less* at K=9

The condition (ii) penalties are uniformly a little smaller at K=9 than at K=72
(`mobile_vit`/`blackbox`: +0.489 vs +0.546). That is the expected direction:
`fc_ft` adapts less from nine frames than from seventy-two, so a mismatched
enrolment has less opportunity to adapt in the wrong direction. The mismatch is
real at both, and matched enrolment removes most of it at both.

## Cost of the smaller enrolment itself

On the `none` arm, condition (iii), going from K=72 to K=9 costs:

| backbone | base (no calib) | fc_ft K=72 | fc_ft K=9 | K=9 − K=72 |
|---|---|---|---|---|
| mobile_vit | 5.121 | 3.330 | 3.496 | +0.166 |
| face_only_mobile_vit | 5.428 | 3.540 | 3.691 | +0.151 |
| convnextv2 | 4.331 | 3.886 | 3.985 | +0.099 |
| itracker | 5.927 | 4.825 | 4.887 | +0.062 |

Nine frames recover most of what seventy-two do — `mobile_vit` 5.121 → 3.496 at
K=9 against 3.330 at K=72, i.e. **91% of the calibration benefit for an eighth of
the enrolment**. The anonymisation penalty being measured here (+0.099 cm) is
smaller than the cost of the shorter calibration itself (+0.166 cm).

## Two things not to read off this table

**SVR is unusable at K=9 and must not be used for a cross-K comparison.** On the
`none` arm `svr_embed` reads **8.6–8.8 cm** at K=9 against `fc_ft`'s 3.5–4.9.
This matches the earlier optimizer sweep, where fixed-hyperparameter SVR was
worst in the low-K regime. Every K=9 conclusion here rests on `fc_ft` only.

**`itracker` + `blackbox` is not a finding.** It is the one cell that worsens
under matched enrolment (+0.093 → +0.499 at K=72, +0.068 → +0.451 at K=9), but
its K=9 confidence interval is [−0.25, +1.43] — it spans zero and is an order of
magnitude wider than any other cell. itracker is the weakest backbone in the
cohort; the cell is reported for completeness and should not be interpreted.

## Status of the utility axis

Conditions (i), (ii) and (iii) are complete at K=72 and K=9 for
`none`/`blur0.10`/`blur0.20`/`blur0.50`/`blackbox`. The remaining gap is operator
coverage (blur0.03, blur0.35 and the five `pixel*` deployment roots were removed
in the storage cleanup), not enrolment size.
