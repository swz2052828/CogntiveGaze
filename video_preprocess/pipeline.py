"""Per-video orchestrator: read frames, detect, crop, save, accumulate metadata.

Output layout (iTracker convention, matches vit_gaze.MultiStreamGazeDataset):
  <output_root>/<rec:05d>/appleFace/<frame:05d>.jpg
  <output_root>/<rec:05d>/appleLeftEye/<frame:05d>.jpg
  <output_root>/<rec:05d>/appleRightEye/<frame:05d>.jpg

The returned MetadataAccumulator carries the per-frame rows; the CLI combines
accumulators across multiple videos before writing a single metadata.mat.
"""

import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from .bbox import bbox_from_points, pad_bbox, square_bbox, to_int_box
from .blink import compute_ear_per_eye, is_blink
from .detector import VideoFaceDetector
from .face_grid import face_grid_params
from .metadata_writer import MetadataAccumulator


def _crop_and_resize(frame_rgb, bbox_int, size):
    x0, y0, x1, y1 = bbox_int
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return Image.fromarray(crop).resize((size, size), Image.BICUBIC)


def process_video(
    video_path,
    output_root,
    rec_num: int,
    face_size: int = 224,
    eye_size: int = 224,
    grid_size: int = 25,
    face_pad: float = 0.1,
    eye_pad_w: float = 0.5,
    eye_pad_h: float = 0.8,
    blink_threshold: float = 0.2,
    skip_blinks: bool = False,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    gaze_lookup: Optional[Callable[[int], Optional[tuple]]] = None,
    face_folder: str = "appleFace",
    left_eye_folder: str = "appleLeftEye",
    right_eye_folder: str = "appleRightEye",
    verbose: bool = True,
) -> MetadataAccumulator:
    """Process one video into iTracker-format crops + metadata rows.

    gaze_lookup, if given, maps frame_index -> (gaze_x, gaze_y) or None for
    unknown. Frames without labels get NaN in metadata. skip_blinks=True drops
    blink frames from the output entirely; default keeps them and marks them.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required. Install with: pip install opencv-python"
        ) from exc

    video_path = Path(video_path)
    output_root = Path(output_root)
    rec_dir = output_root / f"{int(rec_num):05d}"
    face_dir = rec_dir / face_folder
    left_dir = rec_dir / left_eye_folder
    right_dir = rec_dir / right_eye_folder
    for d in (face_dir, left_dir, right_dir):
        d.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    accumulator = MetadataAccumulator()
    frame_idx = 0
    written = 0
    no_face = 0
    blinks = 0
    skipped_blinks = 0
    start = time.perf_counter()

    with VideoFaceDetector() as detector:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            current = frame_idx
            frame_idx += 1

            if frame_stride > 1 and (current % frame_stride) != 0:
                continue
            if max_frames is not None and written >= max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            landmarks = detector.detect(frame_rgb)
            if landmarks is None:
                no_face += 1
                continue

            height, width = frame_rgb.shape[:2]
            frame_size = (width, height)

            face_tight = bbox_from_points(landmarks.face_oval)
            face_padded = pad_bbox(face_tight, face_pad, face_pad, frame_size)
            face_square = square_bbox(face_padded, frame_size)

            left_tight = bbox_from_points(landmarks.left_eye)
            left_padded = pad_bbox(left_tight, eye_pad_w, eye_pad_h, frame_size)
            left_square = square_bbox(left_padded, frame_size)

            right_tight = bbox_from_points(landmarks.right_eye)
            right_padded = pad_bbox(right_tight, eye_pad_w, eye_pad_h, frame_size)
            right_square = square_bbox(right_padded, frame_size)

            face_image = _crop_and_resize(frame_rgb, to_int_box(face_square), face_size)
            left_image = _crop_and_resize(frame_rgb, to_int_box(left_square), eye_size)
            right_image = _crop_and_resize(
                frame_rgb, to_int_box(right_square), eye_size
            )
            if face_image is None or left_image is None or right_image is None:
                continue

            ear_l, ear_r = compute_ear_per_eye(landmarks)
            blink = is_blink(ear_l, ear_r, blink_threshold)
            if blink:
                blinks += 1
            if blink and skip_blinks:
                skipped_blinks += 1
                continue

            grid_params = face_grid_params(
                face_square, frame_size, grid_size=grid_size
            )

            gaze_x = float("nan")
            gaze_y = float("nan")
            if gaze_lookup is not None:
                xy = gaze_lookup(current)
                if xy is not None:
                    gaze_x, gaze_y = float(xy[0]), float(xy[1])

            filename = f"{current:05d}.jpg"
            face_image.save(face_dir / filename, quality=95)
            left_image.save(left_dir / filename, quality=95)
            right_image.save(right_dir / filename, quality=95)

            accumulator.add(
                rec_num=rec_num,
                frame_index=current,
                face_grid_params=grid_params,
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                ear_left=ear_l,
                ear_right=ear_r,
                blink=blink,
                face_bbox=face_square,
                left_eye_bbox=left_square,
                right_eye_bbox=right_square,
            )
            written += 1

            if verbose and written % 50 == 0:
                elapsed = time.perf_counter() - start
                fps = written / max(1e-6, elapsed)
                print(
                    f"[{video_path.name}] written={written} blinks={blinks} "
                    f"skipped_blinks={skipped_blinks} no_face={no_face} "
                    f"elapsed={elapsed:.1f}s fps={fps:.1f}"
                )

    cap.release()
    elapsed = time.perf_counter() - start
    if verbose:
        print(
            f"[{video_path.name}] done: written={written} blinks={blinks} "
            f"skipped_blinks={skipped_blinks} no_face={no_face} "
            f"elapsed={elapsed:.1f}s"
        )
    return accumulator
