# Gaze-Locked Face Swap

A reference-guided face-swap pipeline that **locks gaze direction by construction** while allowing strong identity change.

This package was added because the two existing swap routes had complementary failure modes:

| Route | Gaze preservation | Identity change |
|---|---|---|
| `gaze_preserving_swap` (GAN) | Weak — fixed rectangular eye mask drifts when the head tilts, and the gaze loss has to compete with adversarial + L1 target losses | Modest — small generator trained from scratch |
| `generate_fake_faces_diffusion.py` (SD inpaint) | OK on the eye region, but only because the protected box is large | Weak — `strength=0.62` + generic "different person" prompt produces faces that still look like the original |

The package here does not try to retrain anything. It combines a few pretrained models more aggressively and adds a hard guarantee on gaze.

## Design

1. **Landmark-based protection mask.** [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh) with `refine_landmarks=True` gives 478 landmarks including the iris. The mask is a per-frame union of:
   - the actual eye contour polygons, dilated by a few pixels so eyelid skin is also protected,
   - an ellipse around each iris expanded by `iris_padding_factor` (default 1.8) for a safety ring,
   - the nose-bridge polyline, dilated lightly.
   The mask is feathered with a Gaussian blur so the eventual eye-paste seam is invisible. If MediaPipe is not installed, the pipeline silently falls back to the same fixed-rectangle mask the existing GAN uses.

2. **Reference-guided identity via IP-Adapter Face.** Instead of telling Stable Diffusion "make a different person" via a prompt and hoping for the best, we condition the inpainting pipeline on an **image** of the target identity using [IP-Adapter Plus Face](https://huggingface.co/h94/IP-Adapter). The diffusion model is pushed toward that face, the editable region of the mask is most of the face (only the eyes are protected), and `strength` is high (default `0.92`). The combination gives strong identity change without touching gaze pixels.

3. **Hard pixel copy-back via the feathered mask.** After inpainting, the original source pixels in the eye/iris region are pasted back through the soft alpha mask. The iris and eyelid pixels are byte-exact from the source where the mask is 1.0, and blend smoothly into the swap over the feather band.

4. **Gaze QC gate.** The user's frozen ViT gaze checkpoint (`vit_gaze.training.load_checkpoint`) runs on both `(source, swap)` and reports L2 drift in predicted-gaze coordinates per frame. A `drift_threshold` flags frames whose predicted gaze moved more than expected — those are written to `manifest.json` for review.

## Install

The `vit_gaze` package needs to be importable (run from the repo root or `pip install -e .`).

```bash
pip install -r gaze_locked_swap/requirements.txt
```

The IP-Adapter weights are downloaded on demand by `diffusers` from `h94/IP-Adapter`. The Stable Diffusion inpainting checkpoint defaults to `runwayml/stable-diffusion-inpainting`.

## Usage

Per-folder swap with IP-Adapter and gaze QC:

```bash
python -m gaze_locked_swap.cli swap \
  --source-root ./datasets/OriginalData \
  --output-root ./datasets/GazeLockedSwap \
  --reference-face ./references/target_person.jpg \
  --gaze-checkpoint ./vit_gaze_segmenter_output/fold0_best_vit_gaze_segmenter.pth \
  --ip-adapter-on \
  --strength 0.92 \
  --steps 35 \
  --limit 20
```

The output folder gets a `manifest.json` with per-frame `gaze_drift`, `flagged`, and which mask mode was used.

Prompt-only mode (no IP-Adapter, no reference image required):

```bash
python -m gaze_locked_swap.cli swap \
  --source-root ./datasets/OriginalData \
  --output-root ./datasets/GazeLockedSwap_prompt_only \
  --gaze-checkpoint ./vit_gaze_segmenter_output/fold0_best_vit_gaze_segmenter.pth \
  --strength 0.85
```

Re-run gaze QC on an already-generated swap folder:

```bash
python -m gaze_locked_swap.cli check \
  --source-root ./datasets/OriginalData \
  --swap-root ./datasets/GazeLockedSwap \
  --gaze-checkpoint ./vit_gaze_segmenter_output/fold0_best_vit_gaze_segmenter.pth \
  --drift-threshold 1.5
```

## Tuning

| Knob | What it does | Try |
|---|---|---|
| `--strength` | How far SD drifts from the original. Higher = more identity change. | Start at 0.92 with IP-Adapter, 0.85 without. |
| `--ip-adapter-scale` | How strongly IP-Adapter pulls toward the reference face. | 0.7–1.0. Lower = more diversity, higher = more faithfulness to the reference. |
| `--iris-padding-factor` | Safety ring around the iris. | 1.5 for cropped face data, 2.0 for noisier landmarks. |
| `--eye-dilate-px` | How far past the eye contour to protect. | 3 for tight crops, 6 for low-resolution frames. |
| `--feather-radius` | Width of the alpha-blend band at the eye/swap seam. | 3–8. Larger removes seams but lets a bit of swap leak into eyelid pixels. |
| `--drift-threshold` | Maximum allowed L2 drift in predicted gaze before a frame is flagged. | Calibrate from `check` output on a few hundred frames. |

## What this does *not* do

- It does not train anything. If you need a custom face swap model, the existing `gaze_preserving_swap` GAN is the place.
- It does not guarantee a particular identity. IP-Adapter biases toward the reference, but the result is still a generative draw.
- It does not work on multi-face images — MediaPipe is run with `max_num_faces=1`.
