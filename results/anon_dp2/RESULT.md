# A second de-identification family, and the two-part condition for breaking linkage

**Date:** 2026-09-21. Jobs 1286170 / 1286179 (truncation 0) and 1286271
(truncation 1.0). Review points M5 (is the structural claim method-specific?) and
M7 (is there a constructive contribution?).

DeepPrivacy2 is the contrast SimSwap++ needed: a different family (identity
*removal* by inpainting a detected face box with a StyleGAN generator, versus
identity *replacement* by template transfer), a different objective, and a wider
edit domain — mean absolute change on the `appleFace` crop is **19.92** against
SimSwap's **14.23**, about 40% more of the crop touched. It also visibly repaints
the ocular region and alters gaze direction, which is the concrete demonstration
of the claim in Related Work that removal methods destroy the signal the task
depends on.

Weights came from the HuggingFace space mirrors (`wzkang/FaceDetection-DSFD` for
the detector, `haakohu/deep_privacy2_face` for the generator); the upstream host
`api.loke.aws.unit.no` does not resolve from this cluster. The generator's md5
matches the value pinned in `configs/fdf/stylegan.py`. Three local patches to the
vendored checkout make the DensePose/CSE imports optional; none touches the face
path's detection or generation. See `third_party/DP2_STATUS.md`.

## A mistake worth recording

The first run used `anonymize.py`'s default `truncation_value=0`. The generator
computes `w = w_avg.lerp(w(z), truncation)`, so at 0 the identity latent is
**discarded entirely** and every face becomes the mean face. The `per_subject`
and `per_frame` arms therefore produced **bit-identical images** (mean |delta| =
0.0000 over 60 pairs) and identical attack numbers to two decimal places — the
same failure signature we flagged in the draft's iTracker row.

Those numbers stand as "DP2 at its default setting" and are reported below. The
consistency contrast was re-run at truncation 1.0, behind a **null control that
aborts the job if the two arms are not actually different images**. It passed:
mean |delta| 19.457 between arms, 30.88 between consecutive frames within
`per_frame`.

## Results

Floors: TAR 1.57%, ARI 0.089.

### DP2 at truncation 0 (tool default: every participant gets the mean face)

| | value |
|---|---|
| A1 eye ROI, TAR | **0.00% / 0.00%** |
| A2 eye ROI, TAR | 27.7% / 32.9% |
| T1 face crop, ARI @ k=18 | **1.000** |

A1 at exactly zero is a clean control: when the synthetic identity carries no
source information at all, the cross-domain attack goes to nothing — which makes
SimSwap's residual 0.2–1.8% a real signal rather than noise.

### DP2 at truncation 1.0, by identity regime

| Measure | `per_subject` | `per_frame` | floor |
|---|---|---|---|
| A2 eye ROI, TAR | 63.2% / 40.3% | **2.04% / 4.17%** | 1.57% |
| A2 eye ROI, ARI @ k=18 | 0.795 / 0.749 | **0.148 / 0.169** | 0.089 |
| A1 eye ROI, TAR | 0.00% / 1.20% | 0.09% / 0.56% | 1.57% |
| **T1 face crop, ARI @ k=18** | **1.000** | **0.982** | 0.089 |

## What this establishes

**M5 — the structural claim is not method-specific.** A different family, a
different objective and a 40% wider edit domain still leave the released face
crop **exactly partitionable: ARI 1.000**. Widening the edit domain within the
inner-face region does not help.

**M7 — and here is the constructive half.** Breaking per-participant consistency
is dramatic where the crop contains nothing but the edited region: on the eye
ROI, verification falls 63.2% → 2.04% and linkage 0.795 → 0.148, both essentially
to the floor. On the face crop the very same change does almost nothing:
1.000 → 0.982, because hair at the border, skin tone, head pose and crop geometry
are untouched by the generator and carry the partition on their own.

So the falsifiable claim we drafted — *protection against a release-only attacker
is bounded by how far the transformation breaks per-participant consistency* — is
**half right, and the correction makes it more useful**:

> Release-only linkage is broken only when **both** hold:
> **(a)** the transformation breaks per-participant consistency, **and**
> **(b)** the released crop contains nothing outside the transformation's edit
> domain.
>
> Neither is sufficient alone. Per-frame identities on a face crop still link at
> 0.982; per-participant identities on a fully synthesised eye crop still link at
> 0.75–0.80.

That is actionable rather than merely critical: it tells a dataset author to
resample the identity per recording session **and** to crop the release down to
the region the transformation actually edits. It is also falsifiable, and we have
tested it on two generator families.

**Caveat carried forward.** Per-*frame* randomisation is not a deployable fix for
gaze data — it destroys the temporal coherence a gaze pipeline needs, and DP2
destroys the ocular signal outright. The deployable version is per-*session*
resampling, which our single-session design cannot evaluate. What the experiment
establishes is the mechanism, not a finished method.

## Files

`results/anon_dp2/` — `peri_*`, `a1_*`, `t1_*`, `edit_domain_*`, `emb_*`
`datasets/ProcessedDP2{,t1}_{per_subject,per_frame}`
`scripts/anon/run_deepprivacy2.py`, `run_dp2_generate.sbatch`,
`run_dp2_full.sbatch`, `run_dp2_trunc1.sbatch` (the last carries the null control)
