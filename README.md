# CogntiveGaze

# ViT Gaze Segmentation

This version avoids the previous raw+synthetic fusion by default.

Recommended design:

```text
train gaze predictor on raw images
run the same predictor on raw and synthetic images independently
explain each prediction with occlusion or SmoothGrad
compare the attribution maps
```

This is more stable because the model does not have to learn from two visually inconsistent inputs at once.

## 1. Train Raw-Image Gaze Predictor

Training now uses 5-fold cross validation split by recording id. This keeps every video/subject recording fully in either train or validation for a fold.

For uncropped before-cropping folders:

```bash
python vit_gaze_segmenter.py train ^
  --data-path ./datasets/ProcessedData ^
  --raw-root ./datasets/OriginalData ^
  --synthetic-root ./datasets/SwapData2 ^
  --mean-path mean7 ^
  --input-mode raw ^
  --out-path ./vit_gaze_segmenter_output ^
  --weights imagenet ^
  --epochs 10 ^
  --batch-size 8 ^
  --folds 5
```

For a Slurm array job, run one fold per task:

```bash
python vit_gaze_segmenter.py train ^
  --data-path ./datasets/ProcessedData ^
  --raw-root ./datasets/OriginalData ^
  --synthetic-root ./datasets/SwapData2 ^
  --mean-path mean7 ^
  --input-mode raw ^
  --out-path ./vit_gaze_segmenter_output ^
  --weights imagenet ^
  --epochs 10 ^
  --batch-size 8 ^
  --folds 5 ^
  --fold-index $SLURM_ARRAY_TASK_ID
```

Each epoch prints Slurm-friendly summary lines:

```text
Fold 0 epoch 1/10 train_loss=... val_loss=... val_coord_error=... epoch_time_sec=...
```

Checkpoints are saved as:

```text
fold0_best_vit_gaze_segmenter.pth
fold0_last_vit_gaze_segmenter.pth
```

If ImageNet weights are not available in your environment, use:

```bash
--weights none
```

But expect worse gaze accuracy. Attribution is only meaningful after gaze prediction error is acceptably low.

## Performance and tuning

The same commands run on either machine and auto-detect the GPU at runtime, so
you do not change code per card.

### On automatically (no flags, no change to results)

- **TF32 matmul/conv + cuDNN autotuner** — a free speedup on Ampere+/Blackwell
  (the 5090), silently ignored on Turing (the 2070 Super).
- **Warm, prefetching dataloaders** — workers and prefetch buffers are kept
  alive between epochs so a fast GPU is not starved by image decode/resize.
- **Single-pass shared encoder (multistream)** — the shared ViT-B/16 runs once
  over face + left eye + right eye stacked on the batch dim instead of three
  sequential calls. This is bit-identical (ViT has no cross-sample ops and no
  active dropout in the encoder) and is the main fix for low GPU utilization at
  small batch sizes.

### Recommended settings per GPU

These are the fast, accuracy-safe combinations:

```bash
# RTX 5090 (Blackwell, 32 GB): bf16 autocast + compile.
python vit_gaze_segmenter.py train ... --amp --compile

# RTX 2070 Super (Turing, 8 GB): fp16 autocast (also halves activation memory,
# which is what lets the 8 GB card fit a usable batch). Keep the batch small.
python vit_gaze_segmenter.py train ... --amp
```

### Tuning flags (what each does, and the trade-off)

- **`--amp`** — mixed precision; **off by default** so numerics are bit-for-bit
  unchanged unless you ask. On the 5090 it picks **bf16** (no loss scaling
  needed, accuracy-safe). On the 2070 Super it picks **fp16 + gradient loss
  scaling**. This is the single biggest reliable win — rarely a reason to skip.
- **`--compile`** — wraps the model with `torch.compile` (`dynamic=True`, so the
  ragged final batch does not trigger recompiles). Real extra throughput on
  transformers, at the cost of a one-time graph compile (a few minutes) at
  startup; falls back to eager mode if the backend/GPU does not support it.
  Worth it for multi-epoch runs; gains are smaller and less predictable on
  Turing. On HPC the Triton/Inductor compile cache is redirected to `$TMPDIR`
  (node-local scratch) to avoid blowing the `$HOME` quota; override with
  `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` if you want a persistent cache.
- **`--no-tf32`** — disables TF32 if you want bit-exact fp32 matmuls (slower).
  No effect on the 2070 Super, which has no TF32 path.
- **`--num-workers N`** — dataloader processes. Match it to the CPUs you
  allocate (e.g. `--num-workers 8` with `--cpus-per-task=12`).
- **`--batch-size N`** — the one lever that is **not** accuracy-neutral. A larger
  batch improves 5090 utilization but changes the optimization dynamics
  (effective learning rate / gradient-noise scale), so validation numbers can
  shift. Treat a batch-size change as a training change, not a free speedup, and
  do not compare it head-to-head with a different-batch baseline.
- **`explain ... --occlusion-batch N`** — number of occluded patches evaluated
  per forward pass (raise on the 5090, lower on the 2070 Super). The heatmap is
  identical regardless of `N`; this only removes per-patch launch overhead.

The diffusion generator picks attention slicing automatically
(`--attention-slicing auto`): on for low-VRAM GPUs like the 2070 Super, off on
the 5090 where it would only slow generation down.

### Regularization and generalization (small-cohort overfit)

With ~17 subjects the per-fold train/val loss gap reaches 30–200× on every
backbone — classic small-cohort overfit. These flags address it. All are
**off by default** so prior runs are unchanged; opt in to compose them.

- **`--patience N`** — early-stop the fold if the monitored metric does not
  improve for `N` epochs. Off by default. The best checkpoint is always saved.
- **`--min-delta D`** — minimum improvement that counts as progress (gates
  both the patience counter and best-checkpoint saving). Use `>0` with
  `--patience` to ignore noisy single-epoch dips. Default `0.0` = strict.
- **`--early-stop-metric {val_loss,val_error}`** — which metric drives
  patience and best-checkpoint selection. Defaults to `val_error`, the
  metric actually reported in cm.
- **`--lr-scheduler {none,cosine,step}`** — anneal LR within each fold to
  combat the constant-LR overfit plateau. `cosine` =
  `CosineAnnealingLR(T_max=epochs, eta_min=0)`. `step` =
  `StepLR(step_size=--step-size, gamma=--step-gamma)` (defaults 3 / 0.5).
  Current LR is now logged each epoch so schedules are easy to verify.
- **`--augment {none,light,medium}`** — per-image augmentation on multistream
  crops, **training only** (validation is always clean). No horizontal flip
  (gaze labels are screen-relative).
  - `light`: `ColorJitter(0.2)` + `RandomResizedCrop(scale=0.90–1.0)`.
  - `medium`: `ColorJitter(0.4)` + `RandomResizedCrop(scale=0.85–1.0)` +
    `RandomGrayscale(p=0.05)`.

### Subject-invariant features (`--subject-adv`)

Domain-adversarial training (DANN, Ganin & Lempitsky 2016) attaches a
subject-ID classifier to the fused per-stream feature through a
**gradient-reversal layer**. The discriminator learns to identify the subject
from the features; the reversed gradient pushes the encoder toward features
from which subject identity (head shape, skin tone, camera distance
appearance) cannot be recovered while gaze cues are retained. This is a
**regularizer against subject-specific overfit**, not a replacement for
per-subject calibration — calibration still removes the residual geometric
offset.

- **`--subject-adv`** — enable it (multistream only; off by default).
- **`--adv-weight W`** — ceiling for the gradient-reversal strength λ
  (default `0.1`). λ ramps `0 → W` using the Ganin
  `2/(1+exp(−10p))−1` schedule so the regressor stabilizes before
  invariance pressure ramps in. Too high → invariance erases the signal
  calibration would have used, and `val_coord_error` worsens. Sweep
  `{0.05, 0.1, 0.3}`.
- **`--adv-warmup-frac F`** — fraction of total training over which λ
  reaches `--adv-weight` (default `1.0`, the whole run; smaller values
  reach full strength sooner).

What to watch in the log: `adv_loss` should rise toward
`ln(num_subjects)` as λ ramps (encoder winning ⇒ discriminator
confused). Subject classes are the *current fold's training subjects only*
— held-out subjects are never seen by the adversary, so the CV protocol is
preserved. The discriminator is training-only and is **not** saved in the
inference checkpoint, so `explain` and the `gaze_dynamics` export bridge
are unchanged. Compose with calibration:

```bash
# training
python vit_gaze_segmenter.py train ... \
  --input-mode multistream --backbone vit \
  --subject-adv --adv-weight 0.1 \
  --augment light --lr-scheduler cosine \
  --patience 8 --min-delta 0.005
```

The honest expectation is a modest cross-subject error reduction with the
bigger structural win being reduced overfit. The result that justifies
the approach in a writeup is **adversary + calibration beating
calibration-alone** — especially on the hardest fold.

## Meta-learned per-subject calibration (`metatrain`)

A separate subcommand that learns a **feature-space, nonlinear** replacement
for per-subject SVR and meta-trains it so adapting on only K calibration
frames generalizes to the rest of a subject's session. This attacks the ~5 cm
floor directly: SVR linearly corrects the 2D output, whereas this adapts the
full fused feature the head reads.

- **Each recording is a task.** An episode splits a recording into a *support*
  set (K calibration frames) and a *query* set (the rest) — the enrollment
  scenario, optimized end to end (ANIL + FOMAML; Raghu 2020 / Finn 2017).
- **Encoder frozen** (ANIL), so the fused feature is constant per frame and is
  **cached once per fold** — meta-training runs on cached `[N, dim]` tensors
  with no ViT in the loop (fast even on the 2070 Super).
- Meta-learned parameters: the shared gaze **head** + the **adapter init**.
  The inner loop adapts only the adapter; the outer loop (first-order)
  minimizes post-adaptation query loss.

Adapters (`--adapter`):
- **`film`** (default) — per-subject `γ,β` scale+shift on the fused feature
  (`2·dim` params; robust at small K).
- **`lora`** — low-rank residual at the head input (`--lora-rank`,
  `--lora-alpha`; more expressive, higher overfit risk at small K).

Key flags: `--init-checkpoint` (load a trained `train` checkpoint's
encoder+head so meta-learning starts from gaze-tuned features — strongly
recommended), `--meta-support K`, `--meta-query`, `--inner-steps`,
`--inner-lr`, `--outer-lr`, `--meta-iters`, `--tasks-per-batch`,
`--adapt-steps` (inner steps at enrollment/eval).

```bash
# 1) train the base model normally (produces fold checkpoints)
python vit_gaze_segmenter.py train ... --input-mode multistream --backbone vit \
  --weights imagenet --out-path ./base_out

# 2) meta-learn the calibration adapter on top of the frozen, trained encoder
python vit_gaze_segmenter.py metatrain ... --backbone vit \
  --init-checkpoint ./base_out/fold0_best_vit_gaze_segmenter.pth \
  --adapter film --meta-support 16 --inner-steps 5 --meta-iters 2000 \
  --fold-index 0
```

Each fold logs `meta_pre_adapt_error` vs `meta_post_adapt_error` (cm) on the
held-out recordings — the apples-to-apples number against the SVR-calibrated
floor. The headline comparison for a writeup is **meta-adaptation vs SVR**
at the same K, and ideally **meta-adaptation stacked on `--subject-adv`
features**.

### Enrollment-aware export (`gaze_dynamics/export.py --meta`)

To get adapter-calibrated predictions into the `gaze_dynamics` analyzers, run
the bridge with `--meta`. The exporter enrolls per recording on the first
`--enroll-k` time-ordered frames (a realistic "calibration phase at session
start"), then writes one per-recording file containing predictions on the
remaining frames:

```bash
python -m gaze_dynamics.export ... \
  --checkpoint ./meta_out/fold0_meta_film_vit_gaze.pth \
  --meta --enroll-k 16 --out-dir ./meta_calibrated_gaze
```

`--inner-steps` / `--inner-lr` override the values baked into the checkpoint
if you want to tune enrollment without retraining.

### Comparing base / SVR / meta at matched K (`metacompare`)

The result that justifies the meta approach in a writeup is **meta beats SVR
at the same K on the same support/query draws.** The `metacompare` subcommand
does exactly that — for each held-out recording it draws `--trials` random
K-subsets and scores all three:

* **base**: the model's prediction with no per-subject calibration.
* **svr**: two RBF-SVRs (`--svr-C`, `--svr-eps`, `--svr-gamma`) fit on K
  `(predicted_xy, true_xy)` pairs, applied to the base predictions.
* **meta**: `--inner-steps` of SGD on the meta-learned adapter init using K
  support features.

```bash
python vit_gaze_segmenter.py metacompare ... \
  --base-checkpoint ./base_out/fold0_best_vit_gaze_segmenter.pth \
  --meta-checkpoint ./meta_out/fold0_meta_film_vit_gaze.pth \
  --k 16 --trials 5 --fold-index 0 \
  --csv-out metacompare.csv
```

Per fold the log prints per-recording `base / svr / meta` cm errors plus the
three deltas (`svr_gain`, `meta_gain`, `meta_vs_svr`); a CV summary line at
the end aggregates across folds. The `--csv-out` flag appends a row per fold,
which is useful for sweeping `--k` over `{4, 8, 16, 32, 64}` to plot a
calibration-points-vs-error curve for both methods on the same axes.

**Four-way (stacking on subject-adv features):** pass
`--meta-adv-checkpoint` — a second `metatrain` checkpoint whose
`--init-checkpoint` was a `--subject-adv` run — to add a `meta_adv` method
scored on the *same* support/query draws. This answers "does meta-adaptation
on subject-invariant features beat meta-adaptation on plain features, and
both beat SVR?":

```bash
python vit_gaze_segmenter.py metacompare ... \
  --base-checkpoint  ./base_out/fold0_best_vit_gaze_segmenter.pth \
  --meta-checkpoint  ./meta_out/fold0_meta_film_vit_gaze.pth \
  --meta-adv-checkpoint ./meta_adv_out/fold0_meta_film_vit_gaze.pth \
  --k 16 --trials 5 --fold-index 0 --csv-out metacompare.csv
```

The summary then also reports `meta_adv_gain`, `meta_adv_vs_svr`, and
`meta_adv_vs_meta`.

### Plotting the K-sweep curve

After running `metacompare` at several `--k` values (all appending to the
same `--csv-out`), render the error-vs-K figure:

```bash
python -m vit_gaze.plot_metacompare --csv metacompare.csv --out kcurve.png
```

It groups by K and plots mean ± std-across-folds for every method present
(`base / svr / meta / meta_adv`), x-axis log-scaled in K. The publishable
claim is meta reaching SVR's asymptotic accuracy at a **smaller K** (fewer
calibration points), and `meta_adv` undercutting both.

### Writing the log to a file

By default the per-batch / per-epoch / accel log lines go to stdout. Pass
`--log-file PATH` to **also** append every line to a timestamped, line-buffered
file you can follow live (`tail -f PATH`) and read after the run. Stdout is left
unchanged, so this is useful on Slurm where stdout is often buffered or split
across array-task `.out` files:

```bash
# one self-contained, tailable log per fold of an array job
python vit_gaze_segmenter.py train ... \
  --fold-index $SLURM_ARRAY_TASK_ID \
  --log-file train_fold${SLURM_ARRAY_TASK_ID}.log
```

## 2. Explain Raw Image

Occlusion is the recommended segmentation method because it directly measures how much gaze error increases when each image patch is hidden.

```bash
python vit_gaze_segmenter.py explain ^
  --data-path ./datasets/ProcessedData ^
  --raw-root ./datasets/OriginalData ^
  --synthetic-root ./datasets/SwapData2 ^
  --mean-path mean7 ^
  --checkpoint ./vit_gaze_segmenter_output/fold0_best_vit_gaze_segmenter.pth ^
  --rec 6 ^
  --frame 5462 ^
  --explain-source raw ^
  --attribution occlusion ^
  --out-dir ./vit_gaze_segments
```

## 3. Compare Raw vs Synthetic

After generating eye-preserving synthetic images:

```bash
python vit_gaze_segmenter.py explain ^
  --data-path ./datasets/ProcessedData ^
  --raw-root ./datasets/OriginalData ^
  --synthetic-root ./datasets/SwapData2 ^
  --mean-path mean7 ^
  --checkpoint ./vit_gaze_segmenter_output/fold0_best_vit_gaze_segmenter.pth ^
  --rec 6 ^
  --frame 5462 ^
  --explain-source both ^
  --attribution occlusion ^
  --out-dir ./vit_gaze_segments
```

Outputs include:

- `*_raw_occlusion_heatmap.png`
- `*_raw_occlusion_mask.png`
- `*_raw_occlusion_segment.png`
- `*_synthetic_occlusion_heatmap.png`
- `*_synthetic_occlusion_mask.png`
- `*_synthetic_occlusion_segment.png`
- `*_consensus_mask.png`
- `manifest.json`

## Optional: SmoothGrad

SmoothGrad is faster, but it is often noisier than occlusion:

```bash
--attribution smoothgrad
```

You can save both:

```bash
--attribution both
```

## Tuning Segmentation

Smaller, stricter mask:

```bash
--threshold-percentile 90
```

Broader mask:

```bash
--threshold-percentile 75
```

Finer occlusion map:

```bash
--occlusion-patch 16 --occlusion-stride 8
```

Faster but coarser occlusion:

```bash
--occlusion-patch 32 --occlusion-stride 16
```

## Interpretation

Good sign:

```text
mask highlights eyes, eyelids, nose bridge, face orientation, or head-pose boundary
```

Bad sign:

```text
mask highlights background, crop border, compression artifacts, or synthetic texture artifacts
```

# Diffusion Fake Face Integration

The generator now uses **inpainting**, not whole-image image-to-image generation.

White mask pixels are synthesized. Black mask pixels are copied back exactly from the original image. By default, the script protects:

- eye and iris region
- nose bridge between the eyes
- background
- face boundary / head-pose contour

This is designed for gaze work, where moving the iris invalidates the label.

## Install

```bash
pip install diffusers transformers accelerate safetensors pillow tqdm opencv-python
```

## Recommended: Uncropped OriginalData to SwapData2

Your before-cropping layout is expected to be:

```text
./datasets/OriginalData/<recording>/<frame>.jpg
./datasets/SwapData2/<recording>/<frame>.jpg
```

Dry run:

```bash
python generate_fake_faces_diffusion.py ^
  --input-root ./datasets/OriginalData ^
  --output-root ./datasets/SwapData2 ^
  --limit 5 ^
  --dry-run
```

Generate a small batch and save masks for checking:

```bash
python generate_fake_faces_diffusion.py ^
  --input-root ./datasets/OriginalData ^
  --output-root ./datasets/SwapData2 ^
  --limit 20 ^
  --strength 0.25 ^
  --mask-mode face_non_eye ^
  --debug-mask-dir ./debug_inpaint_masks
```

Inspect `./debug_inpaint_masks` before running the full dataset. The eye/iris region should be black in the mask.

## Safer Settings

More conservative identity change:

```bash
--strength 0.15 --mask-mode lower_face
```

More anonymization, still preserving eyes:

```bash
--strength 0.30 --mask-mode face_non_eye
```

More temporal consistency:

```bash
--seed-mode constant
```

More visual variation:

```bash
--seed-mode per-frame
```

## Old Cropped Layout

The old cropped layout still works:

```bash
python generate_fake_faces_diffusion.py ^
  --dataset-path ./datasets/ProcessedData ^
  --input-folder appleFace ^
  --output-folder appleFaceFake ^
  --limit 20 ^
  --strength 0.25
```

## Quality Control

Reject or regenerate frames when:

- iris center moves
- eyelids become closed or distorted
- eye corners change noticeably
- face pose changes
- output becomes cartoon-like or animated

The generator copies protected pixels back exactly, but the mask detection should still be checked on a sample of every subject.

If gaze prediction error is high, do not trust the segmentation yet.
