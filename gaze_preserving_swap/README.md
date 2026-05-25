# Gaze-Preserving Face Swap

This module is for swapping face identity while keeping gaze direction stable.

The recommended solution is not a plain GAN and not a plain Stable Diffusion prompt. The generator must be trained with a frozen gaze estimator:

```text
total_loss =
  adversarial_loss
  + lambda_gaze * gaze_error(fake_face, true_gaze)
  + lambda_eye * L1(fake_eye_region, source_eye_region)
  + lambda_identity/domain * target_face_loss
  + lambda_tv * smoothness
```

The eye, iris, eyelid and nose-bridge region can also be copied exactly from the source image. This is the strongest safeguard for gaze direction. The GAN changes the editable face region; the protected gaze-critical pixels remain untouched.

## Train The Gaze-Preserving GAN

Train a ViT gaze model first, then use its best checkpoint as the frozen gaze loss:

```bash
python -m gaze_preserving_swap.cli train-gan \
  --data-path ./datasets/ProcessedData \
  --source-root ./datasets/OriginalData \
  --target-root ./datasets/TargetFaces \
  --gaze-checkpoint ./vit_gaze_segmenter_output/fold0_best_vit_gaze_segmenter.pth \
  --out-dir ./gaze_preserving_swap_runs \
  --epochs 10 \
  --batch-size 8
```

If you do not have a separate target face bank yet, omit `--target-root` for a smoke test. Identity change will be weak in that mode.

For a quick debug run:

```bash
python -m gaze_preserving_swap.cli train-gan \
  --data-path ./datasets/ProcessedData \
  --source-root ./datasets/OriginalData \
  --target-root ./datasets/TargetFaces \
  --gaze-checkpoint ./vit_gaze_segmenter_output/fold0_best_vit_gaze_segmenter.pth \
  --out-dir ./gaze_preserving_swap_runs \
  --limit 32 \
  --max-batches 2 \
  --epochs 1 \
  --batch-size 2
```

The training output prints every batch:

```text
GAN epoch 1/10 batch 1/120 g_loss=... d_loss=... gaze_loss=... gaze_error=... eye_loss=... batch_time_sec=...
```

## Run GAN Inference

```bash
python -m gaze_preserving_swap.cli infer-gan \
  --data-path ./datasets/ProcessedData \
  --source-root ./datasets/OriginalData \
  --target-root ./datasets/TargetFaces \
  --checkpoint ./gaze_preserving_swap_runs/last_gaze_preserving_swap_gan.pth \
  --output-root ./datasets/GazeSwapGAN
```

Output layout:

```text
datasets/GazeSwapGAN/<recording>/<frame>.jpg
```

## Stronger Diffusion Alternative

This route changes more of the face than the earlier conservative diffusion script. It inpaints the face outside gaze-critical pixels and then copies the protected eye/iris/nose-bridge region back from the source.

```bash
python -m gaze_preserving_swap.cli diffusion-inpaint \
  --source-root ./datasets/OriginalData \
  --output-root ./datasets/GazeSwapDiffusion \
  --prompt "photorealistic different person, realistic adult face, same head pose, same lighting" \
  --strength 0.62 \
  --guidance-scale 7.0 \
  --steps 35 \
  --device cuda \
  --overwrite
```

Use the GAN route when you need a trainable penalty on gaze error. Use the diffusion route when you need quick identity variation and can verify gaze afterward.
