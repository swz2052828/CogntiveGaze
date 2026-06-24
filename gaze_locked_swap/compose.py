"""Alpha-blend the inpainted swap with the original, locking gaze-critical pixels.

The protection mask is in source pixel space; SD inpainting may have run at a
different resolution. resize_mask_to_image handles the up/downsample.
"""

import numpy as np
from PIL import Image


def resize_mask_to_image(mask: np.ndarray, image: Image.Image) -> np.ndarray:
    """Resample a [0,1] float mask to match image.size using bilinear filtering."""
    target_width, target_height = image.size
    if mask.shape == (target_height, target_width):
        return mask
    mask_pil = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8))
    resized = mask_pil.resize((target_width, target_height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def feather_composite(
    base: Image.Image,
    foreground: Image.Image,
    base_alpha: np.ndarray,
) -> Image.Image:
    """Per-pixel alpha blend.

    base_alpha is the weight of `base` at each pixel (in [0,1]). The protected
    region (eyes/iris/nose-bridge) carries base_alpha=1.0 so the source pixels
    pass through unchanged; the seam fades to 0 where the foreground takes over.
    """
    if foreground.size != base.size:
        foreground = foreground.resize(base.size, Image.BICUBIC)
    base_arr = np.asarray(base.convert("RGB"), dtype=np.float32)
    fg_arr = np.asarray(foreground.convert("RGB"), dtype=np.float32)
    alpha = resize_mask_to_image(base_alpha, base)[..., None]
    blended = base_arr * alpha + fg_arr * (1.0 - alpha)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def hard_paste_protected(
    swap_image: Image.Image,
    source_image: Image.Image,
    protection_soft_mask: np.ndarray,
) -> Image.Image:
    """Paste source's protected pixels onto the swap with a feathered seam.

    This is the operation that guarantees the gaze label is unchanged: the iris
    and eyelid pixels come byte-exact from the source where mask=1.0, and blend
    smoothly into the inpainted result over the feather band.
    """
    return feather_composite(
        base=source_image,
        foreground=swap_image,
        base_alpha=protection_soft_mask,
    )
