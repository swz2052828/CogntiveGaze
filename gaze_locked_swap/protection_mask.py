"""Build the per-frame mask that protects gaze-critical pixels from the diffusion model.

Two modes:
  - landmark mode (preferred): polygon fill around the actual eye contours +
    expanded ellipse around each iris, then Gaussian-feathered for smooth blend.
  - geometric fallback: matches the existing gaze_preserving_swap.masks rectangle
    layout so the new pipeline still runs without MediaPipe.

Convention: mask value 1.0 means "preserve original pixel" (eye region);
mask value 0.0 means "let the inpainter generate". IdentitySwapPipeline inverts
this at the SD boundary, where diffusers expects 1.0 = inpaint.
"""

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .landmarks import LandmarkSet


def _polygon_to_array(width: int, height: int, points: np.ndarray) -> np.ndarray:
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([(float(x), float(y)) for x, y in points], fill=255)
    return np.asarray(img, dtype=np.uint8) > 0


def _ellipses_to_array(
    width: int, height: int, centers, radii
) -> np.ndarray:
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    for (cx, cy), r in zip(centers, radii):
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
    return np.asarray(img, dtype=np.uint8) > 0


def _dilate(mask_bool: np.ndarray, radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return mask_bool
    img = Image.fromarray(mask_bool.astype(np.uint8) * 255)
    kernel = max(3, 2 * radius_px + 1)
    if kernel % 2 == 0:
        kernel += 1
    dilated = img.filter(ImageFilter.MaxFilter(kernel))
    return np.asarray(dilated, dtype=np.uint8) > 0


def build_landmark_mask(
    landmarks: LandmarkSet,
    eye_dilate_px: int = 4,
    iris_padding_factor: float = 1.8,
    include_nose_bridge: bool = True,
    nose_bridge_dilate_px: int = 3,
) -> np.ndarray:
    """Build a hard 0/1 protection mask from landmarks.

    eye_dilate_px expands the eye polygon outward so eyelid pixels are protected
    too (a closed eyelid would shift the gaze label).
    iris_padding_factor multiplies the iris radius for an extra safety ring.
    """
    width, height = landmarks.image_size
    union = np.zeros((height, width), dtype=bool)

    for contour in (landmarks.left_eye, landmarks.right_eye):
        eye_poly = _polygon_to_array(width, height, contour)
        eye_poly = _dilate(eye_poly, eye_dilate_px)
        union |= eye_poly

    iris_centers = []
    iris_radii = []
    for iris in (landmarks.left_iris, landmarks.right_iris):
        cx, cy = iris.mean(axis=0)
        radius = np.linalg.norm(iris - np.array([cx, cy]), axis=1).max()
        iris_centers.append((cx, cy))
        iris_radii.append(max(2.0, radius * iris_padding_factor))
    union |= _ellipses_to_array(width, height, iris_centers, iris_radii)

    if include_nose_bridge:
        bridge = _polygon_to_array(width, height, landmarks.nose_bridge)
        bridge = _dilate(bridge, nose_bridge_dilate_px)
        union |= bridge

    return union.astype(np.float32)


def fallback_geometric_mask(width: int, height: int) -> np.ndarray:
    """Rectangle-based mask matching gaze_preserving_swap.masks geometry."""
    mask = np.zeros((height, width), dtype=np.float32)
    eye_y0 = int(height * 0.24)
    eye_y1 = int(height * 0.48)
    left_x0 = int(width * 0.14)
    left_x1 = int(width * 0.47)
    right_x0 = int(width * 0.53)
    right_x1 = int(width * 0.86)
    bridge_x0 = int(width * 0.40)
    bridge_x1 = int(width * 0.60)
    bridge_y1 = int(height * 0.62)

    mask[eye_y0:eye_y1, left_x0:left_x1] = 1.0
    mask[eye_y0:eye_y1, right_x0:right_x1] = 1.0
    mask[eye_y0:bridge_y1, bridge_x0:bridge_x1] = 1.0
    return mask


def feather_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Gaussian-blur a 0/1 mask into a [0,1] soft mask for seamless blending."""
    if radius <= 0:
        return mask.astype(np.float32)
    image = Image.fromarray((mask * 255).astype(np.uint8))
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def mask_for_image(
    image: Image.Image,
    landmarks: Optional[LandmarkSet],
    feather_radius: int = 3,
    eye_dilate_px: int = 4,
    iris_padding_factor: float = 1.8,
    include_nose_bridge: bool = True,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Build hard + feathered protection masks for one image.

    Returns (hard_mask, feathered_mask, used_landmarks). Falls back to the
    geometric mask if landmarks is None.
    """
    width, height = image.size
    if landmarks is None:
        hard = fallback_geometric_mask(width, height)
        used = False
    else:
        hard = build_landmark_mask(
            landmarks,
            eye_dilate_px=eye_dilate_px,
            iris_padding_factor=iris_padding_factor,
            include_nose_bridge=include_nose_bridge,
        )
        used = True
    soft = feather_mask(hard, radius=feather_radius)
    return hard, soft, used
