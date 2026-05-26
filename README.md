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
  Turing.
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
