"""Compute iTracker-style face-grid params from a face bounding box.

The face-grid is a 25x25 binary mask showing where the face sits in the
original frame. Stored compactly as (x0, y0, w, h) in grid coordinates; the
dataset loader renders it into a 625-d float vector via vit_gaze's makeGrid.
"""

from typing import Tuple


def face_grid_params(
    face_bbox: Tuple[float, float, float, float],
    frame_size: Tuple[int, int],
    grid_size: int = 25,
):
    """Return [x0, y0, w, h] in 0..grid_size-1 integer coords."""
    x0, y0, x1, y1 = face_bbox
    frame_w, frame_h = frame_size
    gx0 = int(round(x0 / frame_w * grid_size))
    gy0 = int(round(y0 / frame_h * grid_size))
    gw = max(1, int(round((x1 - x0) / frame_w * grid_size)))
    gh = max(1, int(round((y1 - y0) / frame_h * grid_size)))
    gx0 = max(0, min(grid_size - 1, gx0))
    gy0 = max(0, min(grid_size - 1, gy0))
    gw = min(grid_size - gx0, gw)
    gh = min(grid_size - gy0, gh)
    return [gx0, gy0, gw, gh]
