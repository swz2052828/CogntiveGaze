# Comprehensive optimization analysis — seven experiments (2026-07-02 → 2026-07-04)

Seven optimization directions were tested independently (each isolated, per-experiment
fleets on clean checkpoints, 5 folds, seed 42; all cm, deploy-faithful calibration
unless noted). **Result: one positive, six decisive negatives — and the negatives are
the finding.** The facemesh+multistream+calibrate architecture is at its
input-limited optimum for this camera.

| # | Experiment | Verdict | Key numbers | Evidence |
|---|---|---|---|---|
| 1 | fc_ft HP tuning (leak-free grid) | **NEGATIVE** | tuned 3.85 vs default 3.60 @K72 | calib_extras_summary.md |
| 2 | LOO auto method-selection | **POSITIVE (the only one)** | 3.601 vs 3.707 @K9; 3.511 vs 3.597 @K72 | calib_extras_summary.md |
| 5 | Temporal smoothing at inference | **NEGATIVE** | all filters ≤ +0.002 | temporal_smoothing_summary.md |
| 4a | Native eye-resolution recrop | **NEGATIVE** | native span 105px ≈ 120px crops | native_eye_span.md |
| 4b | --eye-size sweep (112/168/224/336) | **224 OPTIMAL** | 7.74 / 5.39 / 4.35 / 4.77 | eyesize_sweep_summary.md |
| 6 | Raw-frame single-input models | **NEGATIVE** | best raw 3.90 vs multistream ~3.0 | raw_vs_multistream_summary.md |
| 7 | ViViT temporal backbone | **NEGATIVE** | fc_ft@72 4.50 vs leaders ~3.0-3.5 | calib_metacompare_vivit_clean.csv |

## 1. The one positive: per-subject LOO calibrator selection
Hold out one of the 9 calibration points, fit fc_ft/svr_embed/meta on the other 8,
pick the per-subject winner. Beats every fixed method at both K (−0.11 @K9, −0.09
@K72), captures ~20% of the oracle headroom (oracle 3.03–3.20), and protects against
method-mismatch (mobilenet_v3: 3.75 vs 4.03 if fc_ft were forced). **Adopt as the
default deployment calibrator-picker.** Cost: 3×9 extra tiny fits at enrollment.

## 2. Why everything else failed — three converging laws
**(a) The input is information-limited.** The eyes span ~105px at the source; the
120px crops are already lossless (4a), upscaling adds nothing and even hurts via
pretrain-resolution mismatch (4b: 336→+0.42), and any architecture that sees the
eyes at lower resolution pays immediately (raw ViT-S/384: eyes ~37px → 5.54
calibrated; the 5.3M raw MobileViT recovers to 3.90 by conv inductive bias but still
trails by ~1cm). The crop pipeline is a hard-coded attention prior that preserves
native iris resolution — facemesh costs 7ms/frame and is worth every ms.

**(b) The residual error is per-subject systematic bias, not noise.** Temporal
smoothing does nothing (≤0.002) because predictions are already temporally stable;
blink-frame error is genuine degradation, not impulses (5). Only per-subject
calibration addresses bias — and its HPs must NOT be tuned across subjects (1:
fc_ft HPs don't transfer; Zhu defaults are the generalization sweet spot).

**(c) Added model capacity/context cannot substitute for input signal.** ViViT's
8-frame temporal attention (7): base 4.72, calibrated 4.50, flat in K — at ~7×
the training cost and 8× the inference cost of the leaders. Motion context does
not recover iris detail. Same lesson as dinov2/eva02 (semantic pretraining) and
the raw-frame ViT (global attention): for this task, signal > capacity > context.

## 3. Efficiency picture (10,000-frame protocol, L40 b1 + facemesh CPU incl. decode)
facemesh 70.0s; pipelines: atto_binocular 97.0s ≈ binocular 97.0s < face_only_mobile_vit
107.4s < foveal_vit 115.9s < mobile_vit 181.9s. Raw: raw_vit 26.4s, raw_mobile_vit
37.2s, raw_foveal_vit 52.7s. Every pipeline ≥55fps end-to-end; dropping facemesh
saves 7–16ms/frame but costs 0.9–2.6cm. **Latency is not a binding constraint anywhere.**

## 4. Final deployment recommendation
- **Model**: eyes_only_convnextv2_atto_binocular (3.7M) — best calibrated (2.97
  deploy-faithful), cheapest pipeline (97s/10k incl. facemesh).
- **Calibration**: 9-point pre-task enrollment; **LOO auto-selection** over
  {fc_ft (Zhu defaults), svr_embed (swarm-tuned), meta (calib-trained FiLM)}.
- **Keep**: facemesh crops (attention prior), eye-size 224, blink-cleaned EVAL.
- **Don't**: adv training, meta_adv, fc_ft tuning, output smoothing, bigger eye
  crops, raw-frame end-to-end, temporal backbones, capacity beyond ~5M.
- **Only remaining lever**: a higher-resolution camera (raises the ~105px iris
  ceiling — everything else is downstream of it). Fallback niche: raw_mobile_vit
  (3.90 @5.3M, no face detection) where facemesh is impossible.

## Caveats
- ViViT: 4/5 folds early-stopped at ~epoch 17/20 (16h wall timeout; best ckpts
  near-converged); its calib support used static (repeated-frame) windows.
- Raw-vs-ms random-K table (first section of raw_vs_multistream_summary.md) is
  leaky-K; the deploy-faithful update section supersedes it for absolute numbers.
- LOO auto's oracle gap (~0.5cm) suggests a better selector (e.g. 2-cycle CV over
  the calib recording) could add another ~0.2-0.3 — the one cheap open idea left.
