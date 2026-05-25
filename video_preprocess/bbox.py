"""Bounding-box helpers: tight bbox, padding, squaring, clamping.

All bboxes are (x0, y0, x1, y1) in pixel coordinates with the origin at the
top-left of the frame. Floats throughout until we crop, then convert to ints.
"""

from typing import Tuple

import numpy as np


def bbox_from_points(points: np.ndarray) -> Tuple[float, float, float, float]:
    """Tight (x0, y0, x1, y1) around a set of (x, y) landmark points."""
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def pad_bbox(
    bbox: Tuple[float, float, float, float],
    frac_w: float,
    frac_h: float,
    image_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Expand bbox by frac_w of its width and frac_h of its height, clamped."""
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    px = w * frac_w
    py = h * frac_h
    img_w, img_h = image_size
    return (
        max(0.0, x0 - px),
        max(0.0, y0 - py),
        min(float(img_w), x1 + px),
        min(float(img_h), y1 + py),
    )


def square_bbox(
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """Expand the smaller side so the bbox is square; shift inside the frame."""
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    img_w, img_h = image_size
    # Cap side so the square can actually fit inside the frame.
    side = min(max(w, h), float(min(img_w, img_h)))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    half = side / 2.0
    nx0 = cx - half
    ny0 = cy - half
    nx1 = cx + half
    ny1 = cy + half

    if nx0 < 0:
        nx1 -= nx0
        nx0 = 0.0
    if ny0 < 0:
        ny1 -= ny0
        ny0 = 0.0
    if nx1 > img_w:
        shift = nx1 - img_w
        nx0 -= shift
        nx1 = float(img_w)
    if ny1 > img_h:
        shift = ny1 - img_h
        ny0 -= shift
        ny1 = float(img_h)

    return (
        max(0.0, nx0),
        max(0.0, ny0),
        min(float(img_w), nx1),
        min(float(img_h), ny1),
    )


def to_int_box(
    bbox: Tuple[float, float, float, float]
) -> Tuple[int, int, int, int]:
    """Round bbox to ints for array slicing."""
    x0, y0, x1, y1 = bbox
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
