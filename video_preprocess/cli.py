"""Batch CLI: turn one or more videos into iTracker-style crops + metadata.mat.

Swappable detectors via --face-method / --eye-method / --blink-method. Drop
--vis-dir <path> to also save annotated frames for visual comparison.

Example - compare two strategies on the same video by running twice:

  python -m video_preprocess.cli \\
    --video ./videos/sub01.mp4 --rec 1 \\
    --output-root ./datasets/StrategyA \\
    --face-method mediapipe_facemesh \\
    --eye-method  facemesh_iris \\
    --blink-method ear \\
    --vis-dir ./vis/StrategyA

  python -m video_preprocess.cli \\
    --video ./videos/sub01.mp4 --rec 1 \\
    --output-root ./datasets/StrategyB \\
    --face-method opencv_haar \\
    --eye-method  opencv_haar \\
    --blink-method contour_ratio \\
    --vis-dir ./vis/StrategyB

Then inspect ./vis/StrategyA and ./vis/StrategyB side by side, and train
multistream ViT on each dataset to compare downstream gaze accuracy.
"""

import argparse
import csv
from pathlib import Path

from .blink_detectors import BLINK_DETECTORS, build_blink_detector
from .eye_detectors import EYE_DETECTORS, build_eye_detector
from .face_detectors import FACE_DETECTORS, build_face_detector
from .metadata_writer import MetadataAccumulator
from .pipeline import process_video


def _load_gaze_csv(path: Path):
    table = {}
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                idx = int(row["frame_index"])
                x = float(row["gaze_x"])
                y = float(row["gaze_y"])
            except (KeyError, ValueError):
                continue
            table[idx] = (x, y)
    return lambda frame_index: table.get(int(frame_index))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess smartphone video(s) into iTracker-format face + eye "
            "crops + metadata.mat. Detectors are swappable per stage for "
            "side-by-side evaluation."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        required=True,
        help="Path to a video. Repeat to process multiple videos in one run.",
    )
    parser.add_argument(
        "--rec",
        type=int,
        action="append",
        required=True,
        help="Recording id for the corresponding --video. Order must match.",
    )
    parser.add_argument(
        "--gaze-csv",
        type=Path,
        action="append",
        default=None,
        help=(
            "Optional aligned CSV with columns frame_index, gaze_x, gaze_y. "
            "Repeat once per --video. Missing rows or unset flag => NaN labels."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mean-path",
        default="mean7",
        help="Sub-directory under --output-root where metadata.mat is written.",
    )

    parser.add_argument(
        "--face-method",
        choices=sorted(FACE_DETECTORS.keys()),
        default="mediapipe_facemesh",
    )
    parser.add_argument(
        "--eye-method",
        choices=sorted(EYE_DETECTORS.keys()),
        default="facemesh_contour",
    )
    parser.add_argument(
        "--blink-method",
        choices=sorted(BLINK_DETECTORS.keys()),
        default="ear",
    )
    parser.add_argument(
        "--blink-threshold",
        type=float,
        default=None,
        help=(
            "Override the blink detector's default threshold. ear default 0.2; "
            "contour_ratio default 0.18; iris_visibility default 0.04."
        ),
    )

    parser.add_argument("--face-size", type=int, default=224)
    parser.add_argument("--eye-size", type=int, default=224)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--face-pad", type=float, default=0.1)
    parser.add_argument("--eye-pad-w", type=float, default=0.5)
    parser.add_argument("--eye-pad-h", type=float, default=0.8)

    parser.add_argument(
        "--skip-blinks",
        action="store_true",
        help="Drop blink frames from output. Default keeps them, marks in metadata.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--face-folder", default="appleFace")
    parser.add_argument("--left-eye-folder", default="appleLeftEye")
    parser.add_argument("--right-eye-folder", default="appleRightEye")

    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=None,
        help=(
            "If set, write per-frame annotated JPGs here for visual comparison "
            "of detector strategies. Layout: <vis-dir>/<rec>/<frame>.jpg."
        ),
    )
    parser.add_argument(
        "--vis-stride",
        type=int,
        default=1,
        help="Write 1-in-N annotated frames. Use to control vis disk usage.",
    )

    return parser


def main():
    args = build_parser().parse_args()
    if len(args.video) != len(args.rec):
        raise SystemExit("--video and --rec counts must match.")
    if args.gaze_csv is not None and len(args.gaze_csv) != len(args.video):
        raise SystemExit("--gaze-csv (when given) must be passed once per --video.")

    combined = MetadataAccumulator()
    for i, (video, rec) in enumerate(zip(args.video, args.rec)):
        gaze_lookup = None
        if args.gaze_csv is not None:
            gaze_lookup = _load_gaze_csv(args.gaze_csv[i])

        face_det = build_face_detector(args.face_method)
        eye_det = build_eye_detector(args.eye_method)
        blink_det = build_blink_detector(args.blink_method, threshold=args.blink_threshold)

        per_video = process_video(
            video_path=video,
            output_root=args.output_root,
            rec_num=rec,
            face_detector=face_det,
            eye_detector=eye_det,
            blink_detector=blink_det,
            face_size=args.face_size,
            eye_size=args.eye_size,
            grid_size=args.grid_size,
            face_pad=args.face_pad,
            eye_pad_w=args.eye_pad_w,
            eye_pad_h=args.eye_pad_h,
            skip_blinks=args.skip_blinks,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            gaze_lookup=gaze_lookup,
            face_folder=args.face_folder,
            left_eye_folder=args.left_eye_folder,
            right_eye_folder=args.right_eye_folder,
            vis_dir=args.vis_dir,
            vis_stride=args.vis_stride,
        )
        combined.extend(per_video.rows)

    metadata_path = args.output_root / args.mean_path / "metadata.mat"
    combined.write(metadata_path)
    print(
        f"Wrote {len(combined.rows)} rows to {metadata_path} "
        f"(methods: face={args.face_method} eye={args.eye_method} "
        f"blink={args.blink_method})"
    )


if __name__ == "__main__":
    main()
