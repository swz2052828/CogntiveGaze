"""Per-video orchestrator: read frames, detect, crop, save, accumulate metadata.

The pipeline plugs three detectors (face / eye / blink) selected at the CLI
level via build_face_detector / build_eye_detector / build_blink_detector.
If any chosen detector reports needs_mesh=True, a single VideoFaceDetector
session is opened and its LandmarkSet handed to whichever detectors want it,
so FaceMesh runs at most once per frame.

Output layout (iTracker convention, matches vit_gaze.MultiStreamGazeDataset):
  <output_root>/<rec:05d>/appleFace/<frame:05d>.jpg
  <output_root>/<rec:05d>/appleLeftEye/<frame:05d>.jpg
  <output_root>/<rec:05d>/appleRightEye/<frame:05d>.jpg
"""

import time
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from .bbox import bbox_from_points, pad_bbox, square_bbox, to_int_box
from .detection_base import BlinkDetector, EyeDetector, FaceDetector
from .detector import VideoFaceDetector
from .face_grid import face_grid_params
from .metadata_writer import MetadataAccumulator
from .visualize import write_annotated_frame


def _crop_and_resize(frame_rgb, bbox_int, size):
    x0, y0, x1, y1 = bbox_int
    if x1 <= x0 or y1 <= y0:
        return None
    crop = frame_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return Image.fromarray(crop).resize((size, size), Image.BICUBIC)


def _pad_and_square(bbox, frame_size, frac_w, frac_h):
    padded = pad_bbox(bbox, frac_w, frac_h, frame_size)
    return square_bbox(padded, frame_size)


def process_video(
    video_path,
    output_root,
    rec_num: int,
    face_detector: FaceDetector,
    eye_detector: EyeDetector,
    blink_detector: BlinkDetector,
    face_size: int = 224,
    eye_size: int = 224,
    grid_size: int = 25,
    face_pad: float = 0.1,
    eye_pad_w: float = 0.5,
    eye_pad_h: float = 0.8,
    skip_blinks: bool = False,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    gaze_lookup: Optional[Callable[[int], Optional[tuple]]] = None,
    face_folder: str = "appleFace",
    left_eye_folder: str = "appleLeftEye",
    right_eye_folder: str = "appleRightEye",
    vis_dir: Optional[Path] = None,
    vis_stride: int = 1,
    verbose: bool = True,
) -> MetadataAccumulator:
    """Process one video into iTracker-format crops + metadata rows.

    face_detector / eye_detector / blink_detector are instances built from
    video_preprocess.face_detectors / eye_detectors / blink_detectors. The
    pipeline owns their lifecycle and calls close() at the end.

    vis_dir, if given, receives one annotated JPG per processed frame (every
    vis_stride-th frame) showing the detected boxes + blink badge — useful for
    visually comparing detector strategies.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required. Install: pip install opencv-python"
        ) from exc

    video_path = Path(video_path)
    output_root = Path(output_root)
    rec_dir = output_root / f"{int(rec_num):05d}"
    face_dir = rec_dir / face_folder
    left_dir = rec_dir / left_eye_folder
    right_dir = rec_dir / right_eye_folder
    for d in (face_dir, left_dir, right_dir):
        d.mkdir(parents=True, exist_ok=True)

    if vis_dir is not None:
        vis_dir = Path(vis_dir) / f"{int(rec_num):05d}"
        vis_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    needs_mesh = (
        face_detector.needs_mesh
        or eye_detector.needs_mesh
        or blink_detector.needs_mesh
    )
    mesh_session = VideoFaceDetector() if needs_mesh else None

    accumulator = MetadataAccumulator()
    frame_idx = 0
    written = 0
    no_face = 0
    no_eyes = 0
    blinks = 0
    skipped_blinks = 0
    start = time.perf_counter()

    try:
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
            mesh_landmarks = (
                mesh_session.detect(frame_rgb) if mesh_session is not None else None
            )

            face_bbox_raw = face_detector.detect(frame_rgb, mesh_landmarks)
            if face_bbox_raw is None:
                no_face += 1
                continue

            eye_boxes_raw = eye_detector.detect(frame_rgb, face_bbox_raw, mesh_landmarks)
            if eye_boxes_raw is None:
                no_eyes += 1
                continue
            left_eye_raw, right_eye_raw = eye_boxes_raw

            blink, score_l, score_r = blink_detector.detect(
                frame_rgb, left_eye_raw, right_eye_raw, mesh_landmarks
            )
            if blink:
                blinks += 1
            if blink and skip_blinks:
                skipped_blinks += 1
                continue

            height, width = frame_rgb.shape[:2]
            frame_size = (width, height)

            face_square = _pad_and_square(face_bbox_raw, frame_size, face_pad, face_pad)
            left_square = _pad_and_square(left_eye_raw, frame_size, eye_pad_w, eye_pad_h)
            right_square = _pad_and_square(right_eye_raw, frame_size, eye_pad_w, eye_pad_h)

            face_image = _crop_and_resize(frame_rgb, to_int_box(face_square), face_size)
            left_image = _crop_and_resize(frame_rgb, to_int_box(left_square), eye_size)
            right_image = _crop_and_resize(
                frame_rgb, to_int_box(right_square), eye_size
            )
            if face_image is None or left_image is None or right_image is None:
                continue

            grid_params = face_grid_params(face_square, frame_size, grid_size=grid_size)

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

            if vis_dir is not None and (written % vis_stride) == 0:
                write_annotated_frame(
                    frame_rgb=frame_rgb,
                    out_path=vis_dir / f"{current:05d}.jpg",
                    face_bbox=face_square,
                    left_eye_bbox=left_square,
                    right_eye_bbox=right_square,
                    blink=blink,
                    score_left=score_l,
                    score_right=score_r,
                    face_method=face_detector.name,
                    eye_method=eye_detector.name,
                    blink_method=blink_detector.name,
                )

            accumulator.add(
                rec_num=rec_num,
                frame_index=current,
                face_grid_params=grid_params,
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                ear_left=score_l,
                ear_right=score_r,
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
                    f"no_eyes={no_eyes} elapsed={elapsed:.1f}s fps={fps:.1f}"
                )

    finally:
        cap.release()
        face_detector.close()
        eye_detector.close()
        blink_detector.close()
        if mesh_session is not None:
            mesh_session.close()

    elapsed = time.perf_counter() - start
    if verbose:
        print(
            f"[{video_path.name}] done: written={written} blinks={blinks} "
            f"skipped_blinks={skipped_blinks} no_face={no_face} "
            f"no_eyes={no_eyes} elapsed={elapsed:.1f}s methods="
            f"{face_detector.name}/{eye_detector.name}/{blink_detector.name}"
        )
    return accumulator
