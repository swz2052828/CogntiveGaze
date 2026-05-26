import contextlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def normalize_map(tensor):
    tensor = tensor.detach()
    tensor = tensor - tensor.min()
    denom = tensor.max().clamp_min(1e-8)
    return tensor / denom


def smoothgrad_saliency(model, raw, synthetic, target_gaze_norm, samples, noise_std):
    raw_saliency = torch.zeros(raw.shape[-2:], device=raw.device)
    synthetic_saliency = torch.zeros(synthetic.shape[-2:], device=synthetic.device)

    for _ in range(samples):
        raw_in = raw.clone().detach()
        synthetic_in = synthetic.clone().detach()
        if noise_std > 0:
            raw_in = raw_in + torch.randn_like(raw_in) * noise_std
            synthetic_in = synthetic_in + torch.randn_like(synthetic_in) * noise_std
        raw_in.requires_grad_(True)
        synthetic_in.requires_grad_(True)

        pred = model(raw_in, synthetic_in)
        loss = F.mse_loss(pred, target_gaze_norm, reduction="sum")
        model.zero_grad(set_to_none=True)
        loss.backward()

        raw_saliency += raw_in.grad.detach().abs().mean(dim=1).squeeze(0)
        synthetic_saliency += synthetic_in.grad.detach().abs().mean(dim=1).squeeze(0)

    raw_saliency /= samples
    synthetic_saliency /= samples
    return normalize_map(raw_saliency), normalize_map(synthetic_saliency)


def smoothgrad_single_saliency(model, image, target_gaze_norm, samples, noise_std):
    saliency = torch.zeros(image.shape[-2:], device=image.device)

    for _ in range(samples):
        image_in = image.clone().detach()
        if noise_std > 0:
            image_in = image_in + torch.randn_like(image_in) * noise_std
        image_in.requires_grad_(True)

        pred = model(image_in)
        loss = F.mse_loss(pred, target_gaze_norm, reduction="sum")
        model.zero_grad(set_to_none=True)
        loss.backward()

        saliency += image_in.grad.detach().abs().mean(dim=1).squeeze(0)

    saliency /= samples
    return normalize_map(saliency)


@torch.no_grad()
def occlusion_single_saliency(
    model,
    image,
    target_gaze_norm,
    patch_size,
    stride,
    batch_size=16,
    amp_autocast=None,
):
    """Occlusion attribution with batched forward passes.

    Mathematically identical to occluding one patch at a time, but evaluates up
    to ``batch_size`` occluded variants in a single forward pass, which removes
    the per-patch Python/launch overhead that dominated the original loop. Raise
    ``batch_size`` on a high-VRAM GPU and lower it on an 8 GB card; the result
    does not depend on it. ``amp_autocast`` is an optional zero-arg context
    manager factory (used only when the caller opted into mixed precision).
    """
    autocast_factory = amp_autocast if amp_autocast is not None else contextlib.nullcontext
    _, _, height, width = image.shape
    heatmap = torch.zeros((height, width), device=image.device)
    counts = torch.zeros((height, width), device=image.device)

    with autocast_factory():
        base_pred = model(image)
    base_loss = ((base_pred.float() - target_gaze_norm.float()) ** 2).sum()

    y_positions = list(range(0, max(1, height - patch_size + 1), stride))
    x_positions = list(range(0, max(1, width - patch_size + 1), stride))
    if y_positions[-1] != height - patch_size:
        y_positions.append(max(0, height - patch_size))
    if x_positions[-1] != width - patch_size:
        x_positions.append(max(0, width - patch_size))

    positions = [(y, x) for y in y_positions for x in x_positions]
    for start in range(0, len(positions), max(1, batch_size)):
        chunk = positions[start : start + max(1, batch_size)]
        occluded = image.expand(len(chunk), -1, -1, -1).clone()
        for i, (y, x) in enumerate(chunk):
            occluded[i, :, y : y + patch_size, x : x + patch_size] = 0.0
        with autocast_factory():
            preds = model(occluded)
        losses = ((preds.float() - target_gaze_norm.float()) ** 2).sum(dim=1)
        scores = torch.clamp(losses - base_loss, min=0.0)
        for i, (y, x) in enumerate(chunk):
            heatmap[y : y + patch_size, x : x + patch_size] += scores[i]
            counts[y : y + patch_size, x : x + patch_size] += 1.0

    heatmap = heatmap / counts.clamp_min(1.0)
    return normalize_map(heatmap)


def make_mask(heatmap, percentile):
    heatmap_u8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    heatmap_u8 = cv2.GaussianBlur(heatmap_u8, (0, 0), 1.0)
    threshold = np.percentile(heatmap_u8, percentile)
    mask = (heatmap_u8 >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask


def overlay_heatmap(image, heatmap, alpha=0.45):
    heatmap_u8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip((1 - alpha) * image + alpha * colored, 0, 1)


def overlay_mask(image, mask, alpha=0.45):
    color = np.zeros_like(image)
    color[..., 0] = 1.0
    mask_f = mask[..., None].astype(np.float32)
    return np.clip(image * (1 - alpha * mask_f) + color * (alpha * mask_f), 0, 1)
