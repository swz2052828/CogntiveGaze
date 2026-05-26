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

## Performance on different GPUs

The same commands run on either machine and auto-detect the GPU. By default
TF32 and the cuDNN autotuner are enabled (a free speedup on the 5090, ignored
on the 2070 Super) and dataloader workers are kept warm with extra prefetch, so
a fast GPU is not starved by image decoding. None of this changes results.

For extra throughput, opt into mixed precision (off by default so numerics are
unchanged unless requested):

```bash
# On the 5090: bf16 autocast is selected automatically.
# On the 2070 Super: fp16 autocast with loss scaling, which also lets the
# 8 GB card fit larger batches.
python vit_gaze_segmenter.py train ... --amp
```

Other optional flags:

- `--compile` wraps the model with `torch.compile` (falls back to eager mode if
  unsupported).
- `--no-tf32` disables TF32 if you want bit-exact fp32 matmuls.
- `explain ... --occlusion-batch N` evaluates `N` occluded patches per forward
  pass (raise it on the 5090, lower it on the 2070 Super). The heatmap is
  identical regardless of `N`; this only removes per-patch launch overhead.

The diffusion generator picks attention slicing automatically (`--attention-slicing auto`):
on for low-VRAM GPUs like the 2070 Super, off on the 5090 where it would only
slow generation down.

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
