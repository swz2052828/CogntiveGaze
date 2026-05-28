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


def cm_to_pixels(xy, cm_size=(54.4, 30.4), screen_px=(1920, 1080)):
    """Convert gaze coordinates from screen-cm to screen pixels.

    The metadata produced by ``MakeMeta.py`` stores ``labelDotXCam/Y`` in cm
    (``[0, 54.4]`` x ``[0, 30.4]`` for a 1920x1080 / 54.4x30.4 cm screen), and
    ``vit_gaze`` trains directly on those targets. The gaze_dynamics analyzers,
    in contrast, default to ``screen_res=(1920, 1080)`` pixels. Use this as the
    ``transform=`` hook to ``ViTGazeExporter`` so the exported files are already
    in pixels and need no further rescaling.

    Accepts either ``[N, 2]`` or ``[2, N]`` and returns the same shape.
    """
    arr = np.asarray(xy, dtype=float)
    out = arr.copy()
    if arr.ndim == 2 and arr.shape[0] == 2:        # [2, N]
        out[0] = arr[0] / cm_size[0] * screen_px[0]
        out[1] = arr[1] / cm_size[1] * screen_px[1]
    elif arr.ndim == 2 and arr.shape[1] == 2:      # [N, 2]
        out[:, 0] = arr[:, 0] / cm_size[0] * screen_px[0]
        out[:, 1] = arr[:, 1] / cm_size[1] * screen_px[1]
    else:
        raise ValueError(f"Expected [N, 2] or [2, N], got {arr.shape}")
    return out
