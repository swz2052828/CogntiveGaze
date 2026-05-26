# Video Preprocessing

Turn smartphone video into the iTracker-style dataset layout the rest of the project consumes (face + left eye + right eye crops at 224×224, plus a metadata.mat with face-grid params, EAR, and blink flags).

## What it does, per frame

1. **Detect face** with MediaPipe FaceMesh in video mode (`static_image_mode=False`, tracking on, `refine_landmarks=True` for iris).
2. **Compute three bounding boxes** from landmark groups: face oval, left eye contour, right eye contour. Each box is padded outward (face by 10%, eyes by 50% horizontally / 80% vertically) and then made square so the 224×224 resize doesn't distort iris geometry.
3. **Crop and save** each region as a JPG at the iTracker path:
   ```
   <output_root>/<rec:05d>/appleFace/<frame:05d>.jpg
   <output_root>/<rec:05d>/appleLeftEye/<frame:05d>.jpg
   <output_root>/<rec:05d>/appleRightEye/<frame:05d>.jpg
   ```
4. **Compute EAR per eye** from six landmarks per eye (corners + upper/lower lid pairs). EAR drops sharply when the eye closes; default `--blink-threshold 0.2` flags closed-eye frames.
5. **Compute face-grid params** `[x0, y0, w, h]` in 25×25 grid coordinates (iTracker's `labelFaceGrid` format).
6. **Append a row** to a `MetadataAccumulator` with: `labelRecNum, frameIndex, labelDotXCam, labelDotYCam, labelFaceGrid, earLeft, earRight, blink, faceBbox, leftEyeBbox, rightEyeBbox`.
7. **Write metadata.mat** at `<output_root>/<mean_path>/metadata.mat` once all videos are processed.

The output drops straight into `vit_gaze.MultiStreamGazeDataset` and the CNN baselines.

## Install

```bash
pip install -r video_preprocess/requirements.txt
```

(MediaPipe needs glibc ≥ 2.27 on Linux. Check with `python -c "import mediapipe; print(mediapipe.__version__)"` on the target machine.)

## CLI

Process one video:

```bash
python -m video_preprocess.cli \
  --video ./videos/sub01.mp4 \
  --rec 1 \
  --gaze-csv ./labels/sub01.csv \
  --output-root ./datasets/ProcessedFromVideo
```

Process multiple videos in one run (each video gets its own recording id under `--output-root`, and the final `metadata.mat` covers all rows):

```bash
python -m video_preprocess.cli \
  --video ./videos/sub01.mp4 --rec 1 --gaze-csv ./labels/sub01.csv \
  --video ./videos/sub02.mp4 --rec 2 --gaze-csv ./labels/sub02.csv \
  --video ./videos/sub03.mp4 --rec 3 \
  --output-root ./datasets/ProcessedFromVideo \
  --mean-path mean7 \
  --skip-blinks
```

After processing, train multistream ViT directly on the new dataset:

```bash
python vit_gaze_segmenter.py train \
  --data-path ./datasets/ProcessedFromVideo \
  --eye-path  ./datasets/ProcessedFromVideo \
  --mean-path mean7 \
  --input-mode multistream \
  --weights imagenet
```

## Gaze labels

The video itself has no gaze coordinates — those come from your stimulus protocol (which dot the subject was told to look at, at which timestamp). Provide them per video as a CSV:

```csv
frame_index,gaze_x,gaze_y
0,-3.21,8.40
1,-3.21,8.40
2,-3.18,8.35
...
```

Pass with `--gaze-csv <path>` once per `--video`. Frames without a matching row get `NaN` for `labelDotXCam / labelDotYCam`; the multistream training will reject NaN-gaze rows naturally (loss becomes NaN), so filter them out at dataset-load time if you have any.

Without `--gaze-csv`, all rows get `NaN` gaze — useful for inference-only datasets where you only need the crops.

## Tuning

| Knob | What it does | When to change |
|---|---|---|
| `--face-pad 0.1` | How far to expand the face bbox beyond the face-oval landmarks. | Bump to 0.15-0.2 if hair/forehead is being cut off. |
| `--eye-pad-w 0.5` / `--eye-pad-h 0.8` | Padding around each eye. | Reduce if you want tighter eye crops; increase if eyelids/brows get cut. |
| `--blink-threshold 0.2` | EAR cutoff for blink. | Print EAR distributions on a sample (`metadata.mat earLeft/earRight`) and pick a value below open-eye mode. |
| `--skip-blinks` | Drop blink frames from output. | Off by default (you keep them, marked in metadata). Turn on if you want gaze-only training. |
| `--frame-stride 1` | Take every Nth frame. | Bump to 2-5 if 30/60fps video is overkill and disk space matters. |
| `--max-frames N` | Hard cap per video. | Use for smoke tests before processing the full dataset. |

## What this does NOT do

- Detect gaze direction. That's the trained gaze model's job. This just prepares inputs.
- Handle multi-face frames. `max_num_faces=1`; the most prominent face wins.
- Synchronise audio. Frame index is sequential; if your gaze labels are time-aligned, you map time → frame_index yourself (use the video's FPS).
- Detect head pose explicitly. Pose is implicit in the face crop; the multistream ViT picks it up from there.

## Files

```
video_preprocess/
  __init__.py            Public API
  detector.py            Streaming FaceMesh wrapper + LandmarkSet + index constants
  bbox.py                Tight bbox + padding + square + integer rounding helpers
  blink.py               EAR computation and threshold-based blink detection
  face_grid.py           25x25 face-grid params from face bbox
  metadata_writer.py     Row accumulator + scipy.io.savemat dump
  pipeline.py            process_video orchestrator (the per-video loop)
  cli.py                 python -m video_preprocess.cli
  requirements.txt
  README.md
```
