import argparse

from .explain import explain
from .training import train


def add_common_args(parser):
    parser.add_argument(
        "--data-path",
        default=None,
        help="Dataset root that contains <mean-path>/metadata.mat. Used for the old cropped layout too.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Explicit path to metadata.mat. Use this if raw/synthetic images live outside the processed dataset.",
    )
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Root for uncropped original images laid out as <raw-root>/<recording>/<frame>.jpg.",
    )
    parser.add_argument(
        "--synthetic-root",
        default=None,
        help="Root for uncropped synthetic images laid out as <synthetic-root>/<recording>/<frame>.jpg.",
    )
    parser.add_argument("--mean-path", default="mean7")
    parser.add_argument("--raw-folder", default="appleFace")
    parser.add_argument("--synthetic-folder", default="appleFaceFake")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--allow-missing-synthetic", action="store_true")
    parser.add_argument("--cpu", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train a ViT gaze regressor and output segmented regions that contribute "
            "to a true gaze coordinate. Training uses recording-level K-fold cross validation."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_common_args(train_parser)
    train_parser.add_argument("--out-path", default="./vit_gaze_segmenter_output")
    train_parser.add_argument(
        "--input-mode",
        choices=("raw", "synthetic", "paired"),
        default="raw",
        help="raw is recommended. paired keeps the older raw+synthetic fusion model.",
    )
    train_parser.add_argument("--weights", choices=("none", "imagenet"), default="none")
    train_parser.add_argument("--freeze-encoder", action="store_true")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--folds", type=int, default=5)
    train_parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Run only one 0-based fold. Useful with Slurm array jobs.",
    )
    train_parser.add_argument("--seed", type=int, default=42)

    explain_parser = subparsers.add_parser("explain")
    add_common_args(explain_parser)
    explain_parser.add_argument("--checkpoint", required=True)
    explain_parser.add_argument("--out-dir", default="./vit_gaze_segments")
    explain_parser.add_argument("--index", type=int, default=None)
    explain_parser.add_argument("--rec", type=int, default=None)
    explain_parser.add_argument("--frame", type=int, default=None)
    explain_parser.add_argument("--num-examples", type=int, default=5)
    explain_parser.add_argument(
        "--explain-source",
        choices=("raw", "synthetic", "both"),
        default="raw",
        help="For single-image checkpoints, run attribution on this image source.",
    )
    explain_parser.add_argument(
        "--attribution",
        choices=("smoothgrad", "occlusion", "both"),
        default="occlusion",
        help="Occlusion is slower but usually more trustworthy for segmentation.",
    )
    explain_parser.add_argument("--smoothgrad-samples", type=int, default=12)
    explain_parser.add_argument("--noise-std", type=float, default=0.03)
    explain_parser.add_argument("--occlusion-patch", type=int, default=24)
    explain_parser.add_argument("--occlusion-stride", type=int, default=12)
    explain_parser.add_argument("--threshold-percentile", type=float, default=85.0)

    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "explain":
        explain(args)
    else:
        raise ValueError(args.command)
