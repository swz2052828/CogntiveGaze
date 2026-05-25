"""Batch CLI: turn one or more videos into iTracker-style crops + metadata.mat.

Example:
  python -m video_preprocess.cli \\
    --video ./videos/sub01.mp4 --rec 1 --gaze-csv ./labels/sub01.csv \\
    --video ./videos/sub02.mp4 --rec 2 --gaze-csv ./labels/sub02.csv \\
    --output-root ./datasets/ProcessedFromVideo \\
    --mean-path mean7 \\
    --skip-blinks

This drops in for vit_gaze.MultiStreamGazeDataset:
  python vit_gaze_segmenter.py train \\
    --data-path ./datasets/ProcessedFromVideo \\
    --eye-path  ./datasets/ProcessedFromVideo \\
    --mean-path mean7 \\
    --input-mode multistream \\
    --weights imagenet
"""

import argparse
import csv
from pathlib import Path

from .metadata_writer import MetadataAccumulator
from .pipeline import process_video


def _load_gaze_csv(path: Path):
    """Load a CSV with columns frame_index, gaze_x, gaze_y; return a lookup fn."""
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
            "crops + metadata.mat compatible with vit_gaze and the project's "
            "CNN baselines."
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

    parser.add_argument("--face-size", type=int, default=224)
    parser.add_argument("--eye-size", type=int, default=224)
    parser.add_argument("--grid-size", type=int, default=25)

    parser.add_argument("--face-pad", type=float, default=0.1)
    parser.add_argument("--eye-pad-w", type=float, default=0.5)
    parser.add_argument("--eye-pad-h", type=float, default=0.8)

    parser.add_argument("--blink-threshold", type=float, default=0.2)
    parser.add_argument(
        "--skip-blinks",
        action="store_true",
        help="Drop blink frames from output. Default keeps them, marks them in metadata.",
    )

    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Take every Nth frame from each video. 1 = every frame.",
    )
    parser.add_argument("--max-frames", type=int, default=None)

    parser.add_argument("--face-folder", default="appleFace")
    parser.add_argument("--left-eye-folder", default="appleLeftEye")
    parser.add_argument("--right-eye-folder", default="appleRightEye")
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
        per_video = process_video(
            video_path=video,
            output_root=args.output_root,
            rec_num=rec,
            face_size=args.face_size,
            eye_size=args.eye_size,
            grid_size=args.grid_size,
            face_pad=args.face_pad,
            eye_pad_w=args.eye_pad_w,
            eye_pad_h=args.eye_pad_h,
            blink_threshold=args.blink_threshold,
            skip_blinks=args.skip_blinks,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            gaze_lookup=gaze_lookup,
            face_folder=args.face_folder,
            left_eye_folder=args.left_eye_folder,
            right_eye_folder=args.right_eye_folder,
        )
        combined.extend(per_video.rows)

    metadata_path = args.output_root / args.mean_path / "metadata.mat"
    combined.write(metadata_path)
    print(f"Wrote {len(combined.rows)} rows to {metadata_path}")


if __name__ == "__main__":
    main()
