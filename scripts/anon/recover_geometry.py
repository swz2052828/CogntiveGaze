"""Recover per-frame crop geometry (face + eye squares) for the TASK frames.

The shipped task manifests (ProcessedData/meanno7/metadata.mat) kept only
labelFaceGrid, not the pixel bboxes, so full-frame anonymisation cannot re-crop
to the stored appleFace/appleLeftEye/appleRightEye images without re-running
FaceMesh. This script re-runs the video_preprocess detector stack over the
already-extracted OriginalData frames and writes the bbox table.

It also reports the *fidelity* of the regenerated crops against the stored ones
(mean abs pixel error / PSNR). Byte-exactness is NOT required by the study
design -- the anonymisation experiments regenerate the control arm with the same
recovered geometry -- but the drift tells us how comparable the new control is
to the legacy leaderboard.

Smoke:
  python scripts/anon/recover_geometry.py --rec 00006 --limit 200 --report-only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from video_preprocess.bbox import bbox_from_points, to_int_box            # noqa: E402
from video_preprocess.detector import VideoFaceDetector                   # noqa: E402
from video_preprocess.face_grid import face_grid_params                   # noqa: E402
from video_preprocess.pipeline import _crop_and_resize, _pad_and_square   # noqa: E402

ROOT = Path("/springbrook/share/eng/esrpxk/datasets")


def eye_boxes_from_mesh(lms):
    """Mirror video_preprocess' mesh eye detector: tight box per eye contour."""
    return bbox_from_points(lms.left_eye), bbox_from_points(lms.right_eye)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--face-size", type=int, default=320)
    ap.add_argument("--eye-size", type=int, default=120)
    ap.add_argument("--face-pad", type=float, default=0.1)
    ap.add_argument("--eye-pad-w", type=float, default=0.5)
    ap.add_argument("--eye-pad-h", type=float, default=0.8)
    ap.add_argument("--static", action="store_true",
                    help="static_image_mode=True (no tracking history)")
    ap.add_argument("--out", default=None, help="npz bbox table out")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    orig_dir = ROOT / "OriginalData" / args.rec
    proc_dir = ROOT / "ProcessedData" / args.rec
    frames = sorted(p for p in orig_dir.glob("*.jpg"))
    if args.limit:
        frames = frames[: args.limit]
    print(f"[{args.rec}] {len(frames)} frames  static={args.static}", flush=True)

    det = VideoFaceDetector(refine_landmarks=True)
    if args.static:
        det.close()
        import mediapipe as _mp
        det._mesh = _mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                             refine_landmarks=True, min_detection_confidence=0.5)

    rows, mae = [], {"face": [], "left": [], "right": []}
    misses = 0
    for fp in frames:
        frame = np.asarray(Image.open(fp).convert("RGB"))
        h, w = frame.shape[:2]
        lms = det.detect(frame)
        if lms is None:
            misses += 1
            continue
        face_raw = bbox_from_points(lms.face_oval)
        left_raw, right_raw = eye_boxes_from_mesh(lms)
        fsz = (w, h)
        face_sq = _pad_and_square(face_raw, fsz, args.face_pad, args.face_pad)
        left_sq = _pad_and_square(left_raw, fsz, args.eye_pad_w, args.eye_pad_h)
        right_sq = _pad_and_square(right_raw, fsz, args.eye_pad_w, args.eye_pad_h)

        crops = {
            "face": (_crop_and_resize(frame, to_int_box(face_sq), args.face_size), "appleFace"),
            "left": (_crop_and_resize(frame, to_int_box(left_sq), args.eye_size), "appleLeftEye"),
            "right": (_crop_and_resize(frame, to_int_box(right_sq), args.eye_size), "appleRightEye"),
        }
        for key, (img, folder) in crops.items():
            ref_p = proc_dir / folder / fp.name
            if img is None or not ref_p.is_file():
                continue
            ref = np.asarray(Image.open(ref_p).convert("RGB"), dtype=np.float32)
            new = np.asarray(img, dtype=np.float32)
            if ref.shape != new.shape:
                mae[key].append(float("nan"))
                continue
            mae[key].append(float(np.abs(ref - new).mean()))

        rows.append(dict(frame=fp.stem,
                         face=[float(v) for v in face_sq],
                         left=[float(v) for v in left_sq],
                         right=[float(v) for v in right_sq],
                         grid=[int(v) for v in face_grid_params(face_sq, fsz, grid_size=25)]))
    det.close()

    print(f"detected {len(rows)}  misses {misses}")
    for key, vals in mae.items():
        v = np.asarray([x for x in vals if np.isfinite(x)])
        if not len(v):
            print(f"  {key}: no comparable refs")
            continue
        psnr = 20 * np.log10(255.0 / np.maximum(v, 1e-6))
        print(f"  {key}: MAE mean {v.mean():6.2f}  median {np.median(v):6.2f}  "
              f"<1.0 {(v < 1).mean():5.1%}  <5.0 {(v < 5).mean():5.1%}  "
              f"medianPSNR {np.median(psnr):5.1f} dB")

    if args.out and not args.report_only:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(out, "w"))
        print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
