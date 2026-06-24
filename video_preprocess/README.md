# Video Preprocessing

Turn smartphone video into the iTracker-style dataset layout the rest of the project consumes (face + left eye + right eye crops at 224×224, plus a `metadata.mat` with face-grid params, per-eye scores, and blink flags).

Face / eye / blink detection are all **swappable** so you can run two strategies on the same video, side-by-side, and compare both visually and against downstream gaze accuracy.

## What it does, per frame

1. **Detect face** with the chosen face method (`--face-method`).
2. **Detect both eyes** with the chosen eye method (`--eye-method`). Some methods reuse the face-detection landmarks; the pipeline shares a single FaceMesh session across stages when needed.
3. **Detect blink** with the chosen blink method (`--blink-method`) and optional `--blink-threshold`.
4. **Pad and square** the face / left / right bboxes outward (face by 10%, eyes by 50% horizontally / 80% vertically), then crop and resize to 224×224. The square keeps iris geometry undistorted; a cap-to-frame-size safeguard handles narrow frames.
5. **Save** each region as a JPG at the iTracker path:
   ```
   <output_root>/<rec:05d>/appleFace/<frame:05d>.jpg
   <output_root>/<rec:05d>/appleLeftEye/<frame:05d>.jpg
   <output_root>/<rec:05d>/appleRightEye/<frame:05d>.jpg
   ```
6. **Compute face-grid params** `[x0, y0, w, h]` in 25×25 grid coordinates (iTracker's `labelFaceGrid` format).
7. **Append a row** to a `MetadataAccumulator` with: `labelRecNum, frameIndex, labelDotXCam, labelDotYCam, labelFaceGrid, earLeft, earRight, blink, faceBbox, leftEyeBbox, rightEyeBbox`. (The two `ear*` fields hold whatever score the chosen blink detector emits — EAR for `ear`, contour height/width for `contour_ratio`, normalised iris radius for `iris_visibility`. Polarity is consistent within a method.)
8. **Write `metadata.mat`** at `<output_root>/<mean_path>/metadata.mat` once all videos are processed.

The output drops straight into `vit_gaze.MultiStreamGazeDataset` (and the project's CNN baselines).

## Switching detection methods

Three options per stage, picked at the CLI. Defaults in **bold**.

### `--face-method`

| Choice | What | When to pick |
|---|---|---|
| **`mediapipe_facemesh`** | 478-landmark FaceMesh (`refine_landmarks=True` for iris). Bbox derived from the face-oval contour. Also gives landmarks the eye/blink detectors can reuse → fewest passes per frame. | Default. Best overall and most consistent with the rest of the toolbox. |
| `mediapipe_facedetect` | BlazeFace from `mediapipe.solutions.face_detection`. Single bbox per face, no landmarks. ~10× faster than FaceMesh. | When you don't need landmarks for the eye/blink stages (i.e. paired with `opencv_haar` eye method) and want raw throughput. |
| `opencv_haar` | Classical Viola-Jones cascade, bundled with `opencv-python`. No extra deps. Picks the largest face if there are multiple detections. | Comparison baseline; useful where MediaPipe is unavailable. |

### `--eye-method`

| Choice | What | When to pick |
|---|---|---|
| **`facemesh_contour`** | Bbox of the 16-point FaceMesh eye contour per side. Generous, includes eyelids. | Default. Stable, matches what the existing pipeline used to do. |
| `facemesh_iris` | Square box centred on each iris's landmark centroid, side scaled by the iris radius. **Tightest** crop of the three. | When you want the eye region focused on the iris itself — useful for very small face/eyes in frame. |
| `opencv_haar` | Haar eye cascade run inside the upper 60% of the face bbox. Picks the two largest detections and orders them by x-centre to assign left/right. | Comparison baseline that doesn't depend on FaceMesh. Slightly less reliable on subjects with glasses. |

### `--blink-method` (+ `--blink-threshold`)

| Choice | Score | Default threshold | What it measures |
|---|---|---|---|
| **`ear`** | 6-point Eye Aspect Ratio | `0.20` | Standard EAR. ~0.0 closed, ~0.4 wide open. |
| `contour_ratio` | bbox height / width of the 16-point eye contour | `0.18` | Uses more landmarks than EAR, slightly more stable. |
| `iris_visibility` | iris radius / inter-canthi distance | `0.04` | Closed eye occludes the iris, so the iris-landmark spread shrinks. Independent of the eye contour — good cross-check. |

A frame is flagged as a blink when **either** eye's score falls below the threshold. Pass `--blink-threshold <val>` to override the method's default.

### Mix-and-match

The three slots are independent — pick any of the 3×3×3 combinations. The pipeline runs a single FaceMesh session per frame if any of the chosen methods needs it (`needs_mesh=True`), so combinations that mix `mediapipe_facemesh` with `opencv_haar` aren't paying for FaceMesh twice.

## Visualisation for comparison

`--vis-dir <path>` writes one annotated JPG per processed frame to `<vis-dir>/<rec>/<frame>.jpg`, showing the detected face / left / right boxes coloured per region, the chosen method names, and a blink badge with the per-eye scores. `--vis-stride N` writes only every N-th frame to keep disk usage sane.

Use it to A/B two strategies on the same video:

```bash
# Strategy A: FaceMesh everywhere + EAR
python -m video_preprocess.cli \
  --video ./videos/sub01.mp4 --rec 1 \
  --output-root ./data/strategy_a --vis-dir ./vis/strategy_a \
  --face-method mediapipe_facemesh \
  --eye-method  facemesh_iris \
  --blink-method ear

# Strategy B: Haar everywhere + contour ratio
python -m video_preprocess.cli \
  --video ./videos/sub01.mp4 --rec 1 \
  --output-root ./data/strategy_b --vis-dir ./vis/strategy_b \
  --face-method opencv_haar \
  --eye-method  opencv_haar \
  --blink-method contour_ratio
```

Eyeball `./vis/strategy_a` and `./vis/strategy_b` side by side, and / or train multistream ViT on each dataset and compare `val_coord_error`.

## Install

```bash
pip install -r video_preprocess/requirements.txt
```

(MediaPipe needs glibc ≥ 2.27 on Linux. Check on the target node: `python -c "import mediapipe; print(mediapipe.__version__)"`. The Haar-based detectors fall back gracefully if MediaPipe is unavailable.)

## Minimal CLI examples

Process one video with all defaults (FaceMesh + FaceMesh contour + EAR):

```bash
python -m video_preprocess.cli \
  --video ./videos/sub01.mp4 \
  --rec 1 \
  --gaze-csv ./labels/sub01.csv \
  --output-root ./datasets/ProcessedFromVideo
```

Process multiple videos in one run (each gets its own recording id; the final `metadata.mat` covers all rows):

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

The video itself has no gaze coordinates — those come from your stimulus protocol (which dot the subject was told to look at, at which timestamp). Provide them per video as an aligned CSV:

```csv
frame_index,gaze_x,gaze_y
0,-3.21,8.40
1,-3.21,8.40
2,-3.18,8.35
...
```

Pass with `--gaze-csv <path>` once per `--video`. Frames without a matching row get `NaN` for `labelDotXCam / labelDotYCam`; the multistream training will reject NaN-gaze rows naturally (loss becomes NaN), so filter them out at dataset-load time if you have any.

Without `--gaze-csv`, all rows get `NaN` gaze — useful for inference-only datasets where you only need the crops.

## Other tuning knobs

| Knob | What it does | When to change |
|---|---|---|
| `--face-pad 0.1` | Expand face bbox outward as a fraction of its size. | Bump to 0.15-0.2 if hair/forehead is being cut off. |
| `--eye-pad-w 0.5` / `--eye-pad-h 0.8` | Padding around each eye bbox. | Reduce for tighter crops; increase if eyelids/brows get cut. |
| `--face-size 224` / `--eye-size 224` | Output crop side length. | Bump if you want higher-res inputs; the multistream ViT expects 224. |
| `--grid-size 25` | Face-grid resolution. | Stays at 25 to match the iTracker convention the dataset loader expects. |
| `--skip-blinks` | Drop blink frames from output. | Off by default (you keep them, marked in metadata). Turn on for gaze-only training. |
| `--frame-stride 1` | Take every Nth frame. | Bump to 2-5 if 30/60fps video is overkill and disk space matters. |
| `--max-frames N` | Hard cap per video. | Smoke tests before processing the full dataset. |
| `--vis-dir` / `--vis-stride` | Annotated JPGs for visual comparison. | Use when picking detector methods. |

## What this does NOT do

- Detect gaze direction. That's the trained gaze model's job; this just prepares its inputs.
- Handle multi-face frames. All face detectors use `max_num_faces=1` (or take the largest detection); the most prominent face wins.
- Synchronise audio. Frame index is sequential; if your gaze labels are time-aligned, you map time → frame_index yourself (use the video's FPS).
- Detect head pose explicitly. Pose is implicit in the face crop; the multistream model picks it up from there.

## Files

```
video_preprocess/
  __init__.py            Public API + detector registries
  detector.py            Streaming FaceMesh wrapper + LandmarkSet + landmark indices
  detection_base.py      FaceDetector / EyeDetector / BlinkDetector ABCs
  face_detectors.py      3 face detectors + build_face_detector + FACE_DETECTORS dict
  eye_detectors.py       3 eye detectors + build_eye_detector + EYE_DETECTORS dict
  blink_detectors.py     3 blink detectors + build_blink_detector + BLINK_DETECTORS dict
  bbox.py                Tight bbox + padding + square + cap-to-frame helpers
  blink.py               6-point EAR (used by the ear detector)
  face_grid.py           25x25 face-grid params from face bbox
  metadata_writer.py     Row accumulator + scipy.io.savemat dump
  pipeline.py            process_video orchestrator (the per-video loop)
  visualize.py           Annotated-frame writer for --vis-dir
  cli.py                 python -m video_preprocess.cli
  requirements.txt       opencv-python + mediapipe + numpy + scipy + Pillow
  README.md
```

## Programmatic API

If you'd rather drive it from Python than the CLI, the same detectors are usable directly:

```python
from video_preprocess import (
    build_face_detector, build_eye_detector, build_blink_detector,
    process_video,
)

face_det  = build_face_detector("mediapipe_facemesh")
eye_det   = build_eye_detector("facemesh_iris")
blink_det = build_blink_detector("ear", threshold=0.20)

accumulator = process_video(
    video_path="./videos/sub01.mp4",
    output_root="./datasets/ProcessedFromVideo",
    rec_num=1,
    face_detector=face_det,
    eye_detector=eye_det,
    blink_detector=blink_det,
    gaze_lookup=None,         # or a callable: frame_idx -> (x, y) or None
    vis_dir="./vis/sub01",    # optional
)
accumulator.write("./datasets/ProcessedFromVideo/mean7/metadata.mat")
```
