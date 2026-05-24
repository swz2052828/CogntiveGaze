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
