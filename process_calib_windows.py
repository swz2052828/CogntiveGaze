"""Final preprocessing of each subject's 1200-frame calibration window.

Per subject: seek to calib_begins, process 1200 frames with
  face  = mediapipe_facemesh  (default; supplies the shared mesh)
  eye   = facemesh_contour    (mesh-based eye boxes)
  blink = ear                 (eye-aspect-ratio blink flag)
all on UPRIGHT frames (rotate 90 deg CW — the source video is stored 90 CCW).

Output layout (iTracker convention) under --out-root:
  <out>/<rec:05d>/appleFace/<frame>.jpg
  <out>/<rec:05d>/appleLeftEye/<frame>.jpg
  <out>/<rec:05d>/appleRightEye/<frame>.jpg
  <out>/<rec:05d>/metadata.mat      (per-subject)
  <out>/vis/<rec:05d>/<frame>.jpg   (annotated review frames, 1-in-30)
  <out>/metadata_all.mat            (all subjects combined)
  <out>/SUMMARY.csv                 (per-subject written/blinks/no_face/no_eyes)

calib_begins from Experiment 2/code/GazeDataLoader_swz.py.
Runs on a SLURM compute node (never the login node).
"""

import argparse
import csv
import sys
from pathlib import Path

import mediapipe  # noqa
import mediapipe.python.solutions as _sol  # noqa
sys.modules.setdefault("mediapipe.solutions", _sol)
sys.modules.setdefault("mediapipe.solutions.face_mesh", _sol.face_mesh)
sys.modules.setdefault("mediapipe.solutions.face_detection", _sol.face_detection)

from video_preprocess.face_detectors import build_face_detector
from video_preprocess.eye_detectors import build_eye_detector
from video_preprocess.blink_detectors import build_blink_detector
from video_preprocess.metadata_writer import MetadataAccumulator
from video_preprocess.pipeline import process_video

SUB_IDS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
CALIB = dict(zip(SUB_IDS,
    [3750, 13170, 9900, 3200, 4950, 6300, 1500, 2400, 3600, 2850,
     1800, 2250, 11250, 2550, 4500, 600, 2400, 2250]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default="/springbrook/share/eng/esrpxk/datasets/videos")
    ap.add_argument("--out-root", default="/springbrook/share/eng/esrpxk/datasets/calib_processed")
    ap.add_argument("--window", type=int, default=1200)
    ap.add_argument("--rotate", default="cw", choices=["none", "cw", "ccw", "180"])
    ap.add_argument("--face-method", default="mediapipe_facemesh")
    ap.add_argument("--eye-method", default="facemesh_contour")
    ap.add_argument("--blink-method", default="ear")
    ap.add_argument("--vis-stride", type=int, default=30)
    ap.add_argument("--subjects", default="all")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    subjects = SUB_IDS if args.subjects == "all" else [int(s) for s in args.subjects.split(",")]

    combined = MetadataAccumulator()
    summary = []
    for sub in subjects:
        video = Path(args.videos_dir) / f"subid_{sub}.mp4"
        if not video.exists():
            print(f"[sub {sub}] MISSING {video}", flush=True)
            continue
        face = build_face_detector(args.face_method)
        eye = build_eye_detector(args.eye_method)
        blink = build_blink_detector(args.blink_method, threshold=None)

        acc = process_video(
            video_path=video,
            output_root=out_root,
            rec_num=sub,
            face_detector=face,
            eye_detector=eye,
            blink_detector=blink,
            max_frames=args.window,
            start_frame=CALIB[sub],
            rotate=args.rotate,
            vis_dir=out_root / "vis",
            vis_stride=args.vis_stride,
            verbose=True,
        )
        n = len(acc.rows)
        n_blink = sum(int(r.get("blink", 0)) for r in acc.rows)
        # per-subject metadata (skip write if nothing was saved)
        if n > 0:
            per = MetadataAccumulator()
            per.extend(acc.rows)
            per.write(out_root / f"{sub:05d}" / "metadata.mat")
            combined.extend(acc.rows)
        summary.append(dict(sub_id=sub, calib_begin=CALIB[sub], window=args.window,
                            rows_written=n, blinks=n_blink))
        print(f"[sub {sub}] window [{CALIB[sub]}, {CALIB[sub]+args.window}) -> "
              f"{n} frames saved, {n_blink} blinks", flush=True)

    combined.write(out_root / "metadata_all.mat")
    with open(out_root / "SUMMARY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sub_id", "calib_begin", "window",
                                          "rows_written", "blinks"])
        w.writeheader()
        w.writerows(summary)
    print(f"\nWrote {out_root}/metadata_all.mat and SUMMARY.csv  "
          f"({len(combined.rows)} total rows)", flush=True)


if __name__ == "__main__":
    main()
