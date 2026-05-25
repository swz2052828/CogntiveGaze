from PIL import Image, ImageDraw
import torch


def make_gaze_protection_mask(batch_size, height, width, device, dtype=torch.float32):
    """Return a mask where 1 means copy pixels from the original source image."""
    mask = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)

    eye_y0 = int(height * 0.24)
    eye_y1 = int(height * 0.48)
    left_x0 = int(width * 0.14)
    left_x1 = int(width * 0.47)
    right_x0 = int(width * 0.53)
    right_x1 = int(width * 0.86)
    bridge_x0 = int(width * 0.40)
    bridge_x1 = int(width * 0.60)
    bridge_y1 = int(height * 0.62)

    mask[:, :, eye_y0:eye_y1, left_x0:left_x1] = 1.0
    mask[:, :, eye_y0:eye_y1, right_x0:right_x1] = 1.0
    mask[:, :, eye_y0:bridge_y1, bridge_x0:bridge_x1] = 1.0
    return mask


def blend_protected_regions(generated, source, protection_mask):
    return generated * (1.0 - protection_mask) + source * protection_mask


def masked_l1(first, second, mask):
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(first - second) * mask).sum() / denom


def make_pil_gaze_protection_mask(width, height):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    eye_y0 = int(height * 0.24)
    eye_y1 = int(height * 0.48)
    left_x0 = int(width * 0.14)
    left_x1 = int(width * 0.47)
    right_x0 = int(width * 0.53)
    right_x1 = int(width * 0.86)
    bridge_x0 = int(width * 0.40)
    bridge_x1 = int(width * 0.60)
    bridge_y1 = int(height * 0.62)

    draw.rectangle((left_x0, eye_y0, left_x1, eye_y1), fill=255)
    draw.rectangle((right_x0, eye_y0, right_x1, eye_y1), fill=255)
    draw.rectangle((bridge_x0, eye_y0, bridge_x1, bridge_y1), fill=255)
    return mask


def make_pil_inpaint_mask(width, height):
    """Stable Diffusion inpainting mask: white means change, black means preserve."""
    protection = make_pil_gaze_protection_mask(width, height)
    mask = Image.new("L", (width, height), 255)
    mask.paste(0, mask=protection)
    return mask
