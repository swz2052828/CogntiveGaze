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

If gaze prediction error is high, do not trust the segmentation yet.
