"""Small coordinate helpers shared by every analyzer.

These collapse patterns that were copy-pasted across the original scripts:
the ``np.concatenate(xy, axis=1).astype(int)`` segment join, the ``H - y``
Y-flip for image/origin='lower' plotting, and the pixel rescaling between the
model's output resolution and the analysis screen resolution.
"""

import numpy as np


def concat_xy(xy_list):
    """Join a list of ``[2, n_i]`` gaze segments into one ``[2, N]`` int array."""
    return np.concatenate(xy_list, axis=1).astype(int)


def flip_y(y, height):
    """Flip Y so screen-top maps to plot-top under ``origin='lower'``."""
    return height - y


def scale_to_screen(xy, src_res, dst_res):
    """Rescale ``[2, N]`` coordinates from ``src_res`` (W, H) to ``dst_res`` (W, H).

    Returns a new float array; the input is not modified.
    """
    out = np.asarray(xy, dtype=float).copy()
    out[0] = out[0] / src_res[0] * dst_res[0]
    out[1] = out[1] / src_res[1] * dst_res[1]
    return out


def anti_coordinates(screen_w, screen_h, x, y):
    """Mirror a point across screen center (anti-saccade target location)."""
    cx, cy = screen_w / 2, screen_h / 2
    if x == cx and y == cy:
        return x, y
    return cx - (x - cx), cy - (y - cy)
