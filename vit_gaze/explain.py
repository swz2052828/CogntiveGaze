import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import accel
from .attribution import (
    make_mask,
    occlusion_single_saliency,
    overlay_heatmap,
    overlay_mask,
    smoothgrad_saliency,
    smoothgrad_single_saliency,
)
from .dataset import build_dataset, tensor_to_image
from .training import denormalize_gaze, load_checkpoint, normalize_gaze


def choose_sources(explain_source):
    if explain_source == "both":
        return ["raw", "synthetic"]
    return [explain_source]


def source_tensor(sample, source, device):
    return sample[source].unsqueeze(0).to(device)


def save_image(path, image):
    image_u8 = np.uint8(np.clip(image, 0, 1) * 255)
    Image.fromarray(image_u8).save(path)


def explain(args):
    start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    accel.configure_backends(enable_tf32=not getattr(args, "no_tf32", False))
    amp_enabled, amp_dtype = accel.resolve_amp(device, getattr(args, "amp", False))
    print(f"Explain accel {accel.describe(device, amp_enabled, amp_dtype)}")
    model, gaze_mean, gaze_std, checkpoint, input_mode = load_checkpoint(args.checkpoint, device)
    use_synthetic = input_mode == "paired" or args.explain_source in ("synthetic", "both")
    require_synthetic = use_synthetic and not args.allow_missing_synthetic
    dataset = build_dataset(args, use_synthetic=use_synthetic, require_synthetic=require_synthetic)

    sample_indices = resolve_explain_indices(dataset, args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for count, idx in enumerate(sample_indices, start=1):
        sample_start = time.perf_counter()
        sample = dataset[idx]
        gaze = sample["gaze"].unsqueeze(0).to(device)
        target_norm = normalize_gaze(gaze, gaze_mean, gaze_std)
        true_gaze = gaze.detach().cpu().numpy()[0]
        base = f"idx{idx:06d}_rec{int(sample['rec']):05d}_frame{int(sample['frame']):05d}"
        sample_outputs = {}
        sample_predictions = {}
        source_heatmaps = {}

        if input_mode == "paired":
            raw = sample["raw"].unsqueeze(0).to(device)
            synthetic = sample["synthetic"].unsqueeze(0).to(device)
            pred_norm = model(raw, synthetic)
            pred = denormalize_gaze(pred_norm, gaze_mean, gaze_std).detach().cpu().numpy()[0]
            raw_sal, synthetic_sal = smoothgrad_saliency(
                model=model,
                raw=raw,
                synthetic=synthetic,
                target_gaze_norm=target_norm,
                samples=args.smoothgrad_samples,
                noise_std=args.noise_std,
            )
            source_heatmaps["raw"] = raw_sal.cpu().numpy()
            source_heatmaps["synthetic"] = synthetic_sal.cpu().numpy()
            sample_predictions["paired"] = pred.tolist()
        else:
            explain_single_image_sources(args, model, sample, target_norm, gaze_mean, gaze_std, device, source_heatmaps, sample_outputs, sample_predictions, base, out_dir, amp_enabled, amp_dtype)

        if input_mode == "paired":
            save_paired_outputs(args, sample, source_heatmaps, sample_outputs, base, out_dir)

        if "raw" in source_heatmaps and "synthetic" in source_heatmaps:
            consensus_heatmap = np.minimum(source_heatmaps["raw"], source_heatmaps["synthetic"])
            consensus_mask = make_mask(consensus_heatmap, args.threshold_percentile)
            consensus_path = out_dir / f"{base}_consensus_mask.png"
            Image.fromarray(np.uint8(consensus_mask * 255)).save(consensus_path)
            np.save(out_dir / f"{base}_consensus_mask.npy", consensus_mask.astype(np.uint8))
            sample_outputs["consensus_mask"] = str(consensus_path)

        sample_time = time.perf_counter() - sample_start
        print(
            f"Explain sample {count}/{len(sample_indices)} "
            f"index={idx} rec={int(sample['rec'])} frame={int(sample['frame'])} "
            f"time_sec={sample_time:.2f}"
        )

        manifest.append(
            {
                "index": idx,
                "recording": int(sample["rec"]),
                "frame": int(sample["frame"]),
                "true_gaze": true_gaze.tolist(),
                "predicted_gaze": sample_predictions,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_fold": checkpoint.get("fold"),
                "input_mode": input_mode,
                "outputs": sample_outputs,
            }
        )

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Explain done samples={len(sample_indices)} total_time_sec={time.perf_counter() - start_time:.2f}")


def explain_single_image_sources(args, model, sample, target_norm, gaze_mean, gaze_std, device, source_heatmaps, sample_outputs, sample_predictions, base, out_dir, amp_enabled=False, amp_dtype=None):
    def amp_autocast():
        return accel.autocast(device, amp_enabled, amp_dtype)

    for source in choose_sources(args.explain_source):
        image = source_tensor(sample, source, device)
        with torch.no_grad(), amp_autocast():
            pred_norm = model(image)
        pred = denormalize_gaze(pred_norm.float(), gaze_mean, gaze_std).detach().cpu().numpy()[0]
        sample_predictions[source] = pred.tolist()

        heatmaps = {}
        if args.attribution in ("smoothgrad", "both"):
            heatmaps["smoothgrad"] = smoothgrad_single_saliency(
                model=model,
                image=image,
                target_gaze_norm=target_norm,
                samples=args.smoothgrad_samples,
                noise_std=args.noise_std,
            ).cpu().numpy()
        if args.attribution in ("occlusion", "both"):
            heatmaps["occlusion"] = occlusion_single_saliency(
                model=model,
                image=image,
                target_gaze_norm=target_norm,
                patch_size=args.occlusion_patch,
                stride=args.occlusion_stride,
                batch_size=getattr(args, "occlusion_batch", 16),
                amp_autocast=amp_autocast,
            ).cpu().numpy()

        preferred = "occlusion" if "occlusion" in heatmaps else "smoothgrad"
        source_heatmaps[source] = heatmaps[preferred]
        for method, heatmap in heatmaps.items():
            save_attribution_outputs(args, sample, source, method, heatmap, sample_outputs, base, out_dir)


def save_attribution_outputs(args, sample, source, method, heatmap, sample_outputs, base, out_dir):
    image_np = tensor_to_image(sample[source])
    mask = make_mask(heatmap, args.threshold_percentile)
    heatmap_path = out_dir / f"{base}_{source}_{method}_heatmap.png"
    mask_path = out_dir / f"{base}_{source}_{method}_mask.png"
    segment_path = out_dir / f"{base}_{source}_{method}_segment.png"
    npy_path = out_dir / f"{base}_{source}_{method}_heatmap.npy"
    save_image(heatmap_path, overlay_heatmap(image_np, heatmap))
    save_image(mask_path, overlay_mask(image_np, mask))
    save_image(segment_path, image_np * mask[..., None])
    np.save(npy_path, heatmap.astype(np.float32))
    sample_outputs[f"{source}_{method}_heatmap"] = str(heatmap_path)
    sample_outputs[f"{source}_{method}_mask"] = str(mask_path)
    sample_outputs[f"{source}_{method}_segment"] = str(segment_path)
    sample_outputs[f"{source}_{method}_heatmap_npy"] = str(npy_path)


def save_paired_outputs(args, sample, source_heatmaps, sample_outputs, base, out_dir):
    for source, heatmap in source_heatmaps.items():
        save_attribution_outputs(args, sample, source, "smoothgrad", heatmap, sample_outputs, base, out_dir)


def resolve_explain_indices(dataset, args):
    if args.index is not None:
        return [args.index]

    if args.rec is not None and args.frame is not None:
        matches = [
            i
            for i, (_, _, _, rec, frame) in enumerate(dataset.samples)
            if rec == args.rec and frame == args.frame
        ]
        if not matches:
            raise RuntimeError(f"No sample found for rec={args.rec} frame={args.frame}")
        return matches[:1]

    if args.num_examples <= 0:
        raise ValueError("--num-examples must be positive")
    return list(range(min(args.num_examples, len(dataset))))
