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

Each training batch prints Slurm-friendly progress lines:

```text
Fold 0 epoch 1/10 batch 1/123 batch_loss=... running_train_loss=... batch_time_sec=...
```

Validation is still summarized after each epoch so checkpoints can be selected by validation loss.

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

The training command auto-detects the GPU; the same flags work on both the
RTX 5090 and the 8 GB RTX 2070 Super. TF32, the cuDNN autotuner, warm
dataloaders, and the single-pass shared encoder (multistream) are on
automatically and do not change results.

Fast, accuracy-safe presets:

```bash
# RTX 5090: bf16 autocast + compile
... train ... --amp --compile

# RTX 2070 Super (8 GB): fp16 autocast (also helps fit the 8 GB card)
... train ... --amp
```

Key flags: `--amp` (mixed precision, off by default; biggest reliable win),
`--compile` (`torch.compile`, extra throughput after a one-time warmup),
`--no-tf32` (force bit-exact fp32), `--num-workers` (match your allocated CPUs).
`--batch-size` is the one lever that is **not** accuracy-neutral — a larger batch
helps utilization but shifts the optimization dynamics, so validation numbers
can move. See the README "Performance and tuning" section for the full table.

To capture the optimization log in a readable file (in addition to stdout), pass
`--log-file PATH`. It is timestamped and line-buffered, so you can `tail -f` it
live — handy on Slurm where stdout is often buffered or split across array-task
`.out` files. Combine with `--fold-index` for a per-fold log:

```bash
... train ... --fold-index $SLURM_ARRAY_TASK_ID \
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

If gaze prediction error is high, do not trust the segmentation yet.

## 4. Switching Input Modes

`vit_gaze_segmenter.py train` accepts four `--input-mode` choices. Pick based on what your dataset looks like.

| `--input-mode` | What the model sees per frame | Dataset layout |
|---|---|---|
| `raw` (default) | Single face image | `<data_path>/<rec>/<raw-folder>/<frame>.jpg` |
| `synthetic` | Single synthetic image (gaze-preserving swap output) | `<data_path>/<rec>/<synthetic-folder>/<frame>.jpg` |
| `paired` | Raw + synthetic fused four ways (legacy) | both folders, aligned by frame |
| `multistream` | Face + left eye + right eye crops (+ optional face-grid) | iTracker layout — see below |

The first three are single-stream (one image -> 2D gaze). The fourth one mirrors the input format of the project's CNN baselines (MGazeNet / AFFNet / MobileNetV3) and is required if you want to swap to those backbones.

### `multistream` data layout

```text
<data_path>/<rec:05d>/appleFace/<frame:05d>.jpg
<eye_path >/<rec:05d>/appleLeftEye/<frame:05d>.jpg
<eye_path >/<rec:05d>/appleRightEye/<frame:05d>.jpg
<data_path>/<mean_path>/metadata.mat   # labelRecNum, frameIndex, labelDotXCam,
                                       # labelDotYCam, labelFaceGrid
```

`--eye-path` defaults to `--data-path` when the eye crops live in the same root as the face crops. The face-grid is read from `labelFaceGrid` in `metadata.mat` and only loaded when `--use-grid` is passed.

If you start from raw smartphone video, the `video_preprocess` package writes exactly this layout — see [`video_preprocess/README.md`](video_preprocess/README.md).

### Minimal multistream training command

```bash
python vit_gaze_segmenter.py train \
  --data-path ./datasets/ProcessedData \
  --eye-path  ./datasets/ProcessedData \
  --mean-path mean7 \
  --input-mode multistream \
  --weights imagenet \
  --epochs 30 \
  --batch-size 16 \
  --folds 5 \
  --out-path ./vit_gaze_multistream_output
```

This trains the default ViT backbone with no face-grid. The grid is **off by default** because for seated, static-head desktop setups the face-grid is near-constant per subject and provides no within-subject signal (and across subjects acts as an identity leak under recording-level K-fold).

Add `--use-grid` to enable it. This is required when switching to any of the CNN backbones — see below.

## 5. Switching Multistream Backbones

In `--input-mode multistream` you can choose the model architecture with `--backbone`. Five options, all wired through the same training / K-fold / normalisation pipeline so comparisons are apples-to-apples.

| `--backbone` | Source | Params | `--use-grid` |
|---|---|---|---|
| `vit` (default) | Shared ViT-B/16 encoder across face + both eyes; optional 625 → 256 → 128 grid MLP | 87M | optional |
| `itracker` | Original GazeCapture iTracker CNN (AlexNet-ish), eye weights shared, grid concat at head | 6.3M | **required** |
| `mobilenet_v3` | MobileNetV3-Large feature extractors with iTracker-style fusion head | 6.4M | **required** |
| `affnet` | Adaptive Group Norm; eye stream conditioned on `(face, grid)` factor at every block | 3.0M | **required** (used in AGN) |
| `mgazenet` | LABN + SE blocks; same factor-conditioned eye streams as AFFNet | 3.0M | **required** (used in LABN) |

The four CNN backbones use the face-grid as a conditioning factor (AFFNet / MGazeNet via AGN / LABN) or concatenate it into the head (iTracker / MobileNetV3). They will refuse to construct without `--use-grid` and tell you why.

### Compare all five backbones on the same dataset

```bash
for bb in vit itracker mobilenet_v3 affnet mgazenet; do
  python vit_gaze_segmenter.py train \
    --data-path ./datasets/ProcessedData \
    --eye-path  ./datasets/ProcessedData \
    --mean-path mean7 \
    --input-mode multistream \
    --backbone $bb \
    --use-grid \
    --weights imagenet \
    --epochs 30 --folds 5 \
    --out-path ./runs/$bb
done
```

(`--use-grid` is harmless for `vit` and required for the other four, so leaving it on for the loop is fine.)

After the runs finish, each `./runs/<backbone>/` holds `fold{0..4}_best_vit_gaze_segmenter.pth` plus a per-fold log. Compare `val_coord_error` across backbones — both per-fold and the `CV summary` line at the end of each log.

### Slurm array per backbone

```bash
sbatch --array=0-4 --job-name=$bb scripts/train_vit_multistream.sbatch \
  --input-mode multistream --backbone $bb --use-grid \
  --data-path ./datasets/ProcessedData \
  --eye-path  ./datasets/ProcessedData \
  --out-path ./runs/$bb
```

One job per fold per backbone. Five jobs per backbone, fifteen total for the five-backbone sweep with all folds.

### Quick reference: which backbone to pick

- **`vit`** — default. Use when you want attention-based attribution maps (works with `vit_gaze.explain`) and the dataset is large enough that 87M params won't overfit.
- **`mobilenet_v3`** — fast, modern, ImageNet-pretrained CNN. Good default when ViT is too heavy.
- **`itracker`** — comparison baseline matching the original GazeCapture paper.
- **`affnet` / `mgazenet`** — small parameter budget, eye streams adaptively conditioned on (face, grid). Strong on small datasets where overfitting is the main concern.

### Loading a checkpoint with a specific backbone

`load_checkpoint` reads the backbone from the saved `args` dict, so once you've trained a model you can re-instantiate it without passing `--backbone` again:

```python
from vit_gaze.training import load_checkpoint
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, gaze_mean, gaze_std, ckpt, input_mode = load_checkpoint(
    "./runs/affnet/fold0_best_vit_gaze_segmenter.pth", device
)
```

The model class is reconstructed via `create_model(input_mode, backbone=..., use_grid=..., grid_size=...)` from the values stored in the checkpoint's `args`.
