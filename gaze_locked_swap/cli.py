"""Command-line entry point for batch gaze-locked face swapping.

Layout matches the rest of the repo: source images at
  <source-root>/<recording>/<frame>.jpg
go to
  <output-root>/<recording>/<frame>.jpg

A reference face image (--reference-face) is the identity the IP-Adapter pushes
toward. Without IP-Adapter, the reference image is ignored and the swap is
prompt-only.

Examples:
  python -m gaze_locked_swap.cli swap \\
    --source-root ./datasets/OriginalData \\
    --output-root ./datasets/GazeLockedSwap \\
    --reference-face ./references/target_person.jpg \\
    --gaze-checkpoint ./vit_gaze_segmenter_output/fold0_best_vit_gaze_segmenter.pth \\
    --ip-adapter-on \\
    --limit 20
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from .gaze_qc import GazeChecker
from .identity_pipeline import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    IPAdapterSpec,
    IdentitySwapPipeline,
)
from .swap import swap_one


def iter_source_images(source_root: Path):
    for recording_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        for image_path in sorted(recording_dir.glob("*.jpg")):
            yield image_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gaze-locked face swap: landmark-tight eye preservation, "
            "reference-guided identity change, and a gaze-QC gate."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    swap_parser = sub.add_parser("swap")
    swap_parser.add_argument("--source-root", required=True, type=Path)
    swap_parser.add_argument("--output-root", required=True, type=Path)
    swap_parser.add_argument("--reference-face", type=Path, default=None)
    swap_parser.add_argument("--limit", type=int, default=None)
    swap_parser.add_argument("--overwrite", action="store_true")
    swap_parser.add_argument(
        "--model-id", default="runwayml/stable-diffusion-inpainting"
    )
    swap_parser.add_argument("--work-size", type=int, default=512)
    swap_parser.add_argument("--strength", type=float, default=0.92)
    swap_parser.add_argument("--guidance-scale", type=float, default=7.5)
    swap_parser.add_argument("--steps", type=int, default=35)
    swap_parser.add_argument("--seed", type=int, default=1234)
    swap_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    swap_parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    swap_parser.add_argument("--feather-radius", type=int, default=4)
    swap_parser.add_argument("--eye-dilate-px", type=int, default=4)
    swap_parser.add_argument("--iris-padding-factor", type=float, default=1.8)
    swap_parser.add_argument(
        "--no-nose-bridge", dest="include_nose_bridge", action="store_false"
    )
    swap_parser.add_argument("--no-landmarks", dest="use_landmarks", action="store_false")
    swap_parser.add_argument("--ip-adapter-on", action="store_true")
    swap_parser.add_argument("--ip-adapter-repo", default="h94/IP-Adapter")
    swap_parser.add_argument("--ip-adapter-subfolder", default="models")
    swap_parser.add_argument(
        "--ip-adapter-weight", default="ip-adapter-plus-face_sd15.bin"
    )
    swap_parser.add_argument("--ip-adapter-scale", type=float, default=0.85)
    swap_parser.add_argument("--cpu-offload", action="store_true")
    swap_parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    swap_parser.add_argument(
        "--gaze-checkpoint",
        default=None,
        help="If set, runs gaze QC and writes per-frame drift to manifest.json.",
    )
    swap_parser.add_argument("--drift-threshold", type=float, default=1.5)

    check_parser = sub.add_parser("check")
    check_parser.add_argument("--source-root", required=True, type=Path)
    check_parser.add_argument("--swap-root", required=True, type=Path)
    check_parser.add_argument("--gaze-checkpoint", required=True)
    check_parser.add_argument("--output", type=Path, default=None)
    check_parser.add_argument("--limit", type=int, default=None)
    check_parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    check_parser.add_argument("--drift-threshold", type=float, default=1.5)

    return parser


def _build_pipeline(args) -> IdentitySwapPipeline:
    ip_spec = None
    if args.ip_adapter_on:
        ip_spec = IPAdapterSpec(
            repo=args.ip_adapter_repo,
            subfolder=args.ip_adapter_subfolder,
            weight_name=args.ip_adapter_weight,
            scale=args.ip_adapter_scale,
        )
    return IdentitySwapPipeline(
        model_id=args.model_id,
        device=args.device,
        ip_adapter=ip_spec,
        use_cpu_offload=args.cpu_offload,
    )


def _maybe_load_reference(args) -> "Image.Image | None":
    if args.reference_face is None:
        return None
    return Image.open(args.reference_face).convert("RGB")


def run_swap(args):
    source_root = args.source_root
    output_root = args.output_root
    if not source_root.is_dir():
        raise SystemExit(f"--source-root not found: {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    if args.ip_adapter_on and args.reference_face is None:
        raise SystemExit("--reference-face is required when --ip-adapter-on is set.")

    pipeline = _build_pipeline(args)
    reference_face = _maybe_load_reference(args)
    gaze_checker = None
    if args.gaze_checkpoint:
        gaze_checker = GazeChecker(args.gaze_checkpoint, device=args.device)

    jobs = list(iter_source_images(source_root))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(f"Found {len(jobs)} source images. Output to {output_root}.")

    manifest = []
    flagged_count = 0
    start = time.perf_counter()
    for index, source_path in enumerate(jobs, start=1):
        rel = source_path.relative_to(source_root)
        dst = output_root / rel
        if dst.exists() and not args.overwrite:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)

        source_image = Image.open(source_path).convert("RGB")
        per_seed = args.seed + index
        result = swap_one(
            source_image=source_image,
            pipeline=pipeline,
            reference_face=reference_face,
            gaze_checker=gaze_checker,
            drift_threshold=args.drift_threshold,
            work_size=args.work_size,
            feather_radius=args.feather_radius,
            eye_dilate_px=args.eye_dilate_px,
            iris_padding_factor=args.iris_padding_factor,
            include_nose_bridge=args.include_nose_bridge,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            seed=per_seed,
            use_landmarks=args.use_landmarks,
        )
        result.swapped.save(dst, quality=95)

        record = {
            "source": str(source_path),
            "output": str(dst),
            "used_landmarks": result.used_landmarks,
            "seed": per_seed,
            "gaze_drift": result.gaze_drift,
            "flagged": result.flagged,
        }
        if result.gaze_original is not None:
            record["gaze_original"] = result.gaze_original.tolist()
            record["gaze_swapped"] = result.gaze_swapped.tolist()
        manifest.append(record)
        if result.flagged:
            flagged_count += 1

        print(
            f"[{index}/{len(jobs)}] {source_path.name} "
            f"landmarks={result.used_landmarks} "
            f"drift={result.gaze_drift} flagged={result.flagged}"
        )

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(
        f"Done. Wrote {len(manifest)} entries to {manifest_path}. "
        f"Flagged {flagged_count}. "
        f"Total time {time.perf_counter() - start:.1f}s."
    )


def run_check(args):
    if not args.source_root.is_dir():
        raise SystemExit(f"--source-root not found: {args.source_root}")
    if not args.swap_root.is_dir():
        raise SystemExit(f"--swap-root not found: {args.swap_root}")

    gaze_checker = GazeChecker(args.gaze_checkpoint, device=args.device)
    output_path = args.output or args.swap_root / "gaze_qc.json"

    pairs = []
    for source_path in iter_source_images(args.source_root):
        rel = source_path.relative_to(args.source_root)
        swap_path = args.swap_root / rel
        if swap_path.exists():
            pairs.append((source_path, swap_path))
        if args.limit is not None and len(pairs) >= args.limit:
            break

    print(f"Checking {len(pairs)} pairs against {args.gaze_checkpoint}.")
    results = []
    flagged_count = 0
    for index, (source_path, swap_path) in enumerate(pairs, start=1):
        original = Image.open(source_path).convert("RGB")
        swapped = Image.open(swap_path).convert("RGB")
        drift = gaze_checker.drift(original, swapped)
        flagged = drift > args.drift_threshold
        if flagged:
            flagged_count += 1
        results.append(
            {
                "source": str(source_path),
                "swap": str(swap_path),
                "gaze_drift": drift,
                "flagged": flagged,
            }
        )
        print(
            f"[{index}/{len(pairs)}] {source_path.name} drift={drift:.4f} "
            f"flagged={flagged}"
        )

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(
        f"Done. Wrote {output_path}. Flagged {flagged_count} of {len(results)} "
        f"(threshold={args.drift_threshold})."
    )


def main():
    args = build_parser().parse_args()
    if args.command == "swap":
        run_swap(args)
    elif args.command == "check":
        run_check(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
