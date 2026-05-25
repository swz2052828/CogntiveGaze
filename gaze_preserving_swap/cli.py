import argparse

from . import diffusion_inpaint, infer_gan, train_gan


def add_data_args(parser):
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--mean-path", default="mean7")
    parser.add_argument("--source-root", required=True, help="Root laid out as <root>/<recording>/<frame>.jpg.")
    parser.add_argument("--target-root", default=None, help="Identity/reference face bank. Can be flat or nested.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--random-targets", action="store_true")
    parser.add_argument("--cpu", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(description="Gaze-preserving face swap training and inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-gan")
    add_data_args(train_parser)
    train_parser.add_argument("--gaze-checkpoint", required=True)
    train_parser.add_argument("--out-dir", default="./gaze_preserving_swap_runs")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--max-batches", type=int, default=None)
    train_parser.add_argument("--lr-g", type=float, default=2e-4)
    train_parser.add_argument("--lr-d", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--base-channels", type=int, default=64)
    train_parser.add_argument("--residual-blocks", type=int, default=4)
    train_parser.add_argument("--adv-weight", type=float, default=1.0)
    train_parser.add_argument("--gaze-weight", type=float, default=20.0)
    train_parser.add_argument("--eye-weight", type=float, default=30.0)
    train_parser.add_argument("--target-weight", type=float, default=0.0)
    train_parser.add_argument("--source-difference-weight", type=float, default=0.05)
    train_parser.add_argument("--source-difference-temperature", type=float, default=0.25)
    train_parser.add_argument("--tv-weight", type=float, default=0.001)
    train_parser.add_argument("--copy-protected", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--save-every", type=int, default=1)

    infer_parser = subparsers.add_parser("infer-gan")
    add_data_args(infer_parser)
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--output-root", required=True)
    infer_parser.add_argument("--batch-size", type=int, default=8)
    infer_parser.add_argument("--num-workers", type=int, default=2)
    infer_parser.add_argument("--base-channels", type=int, default=64)
    infer_parser.add_argument("--residual-blocks", type=int, default=4)
    infer_parser.add_argument("--copy-protected", action=argparse.BooleanOptionalAction, default=True)

    diffusion_parser = subparsers.add_parser("diffusion-inpaint")
    diffusion_parser.add_argument("--source-root", required=True)
    diffusion_parser.add_argument("--output-root", required=True)
    diffusion_parser.add_argument("--model-id", default="stable-diffusion-v1-5/stable-diffusion-inpainting")
    diffusion_parser.add_argument("--prompt", default=diffusion_inpaint.DEFAULT_PROMPT)
    diffusion_parser.add_argument("--negative-prompt", default=diffusion_inpaint.DEFAULT_NEGATIVE_PROMPT)
    diffusion_parser.add_argument("--strength", type=float, default=0.62)
    diffusion_parser.add_argument("--guidance-scale", type=float, default=7.0)
    diffusion_parser.add_argument("--steps", type=int, default=35)
    diffusion_parser.add_argument("--work-size", type=int, default=512)
    diffusion_parser.add_argument("--seed", type=int, default=1234)
    diffusion_parser.add_argument("--limit", type=int, default=None)
    diffusion_parser.add_argument("--overwrite", action="store_true")
    diffusion_parser.add_argument("--dry-run", action="store_true")
    diffusion_parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    diffusion_parser.add_argument("--cpu", action="store_true")
    diffusion_parser.add_argument("--cpu-offload", action="store_true")
    diffusion_parser.add_argument("--weight-format", choices=("bin", "safetensors"), default="bin")
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "train-gan":
        train_gan.train(args)
    elif args.command == "infer-gan":
        infer_gan.infer(args)
    elif args.command == "diffusion-inpaint":
        diffusion_inpaint.run(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
