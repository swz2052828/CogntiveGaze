"""Annotated-frame writer for visual comparison of detector strategies.

Draws the face bbox, left/right eye bboxes, blink badge, and detector names
onto a copy of the frame and saves it as a JPG. Disabled unless the pipeline
is run with --vis-dir.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


BBox = Tuple[float, float, float, float]


def _draw_box(draw: ImageDraw.ImageDraw, bbox: Optional[BBox], color, label):
    if bbox is None:
        return
    x0, y0, x1, y1 = bbox
    draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
    draw.text((x0 + 4, max(0, y0 - 14)), label, fill=color)


def write_annotated_frame(
    frame_rgb: np.ndarray,
    out_path: Path,
    face_bbox: Optional[BBox],
    left_eye_bbox: Optional[BBox],
    right_eye_bbox: Optional[BBox],
    blink: bool,
    score_left: float,
    score_right: float,
    face_method: str,
    eye_method: str,
    blink_method: str,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(frame_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)

    _draw_box(draw, face_bbox, (0, 255, 0), f"face/{face_method}")
    _draw_box(draw, left_eye_bbox, (0, 200, 255), f"eyeL/{eye_method}")
    _draw_box(draw, right_eye_bbox, (255, 200, 0), f"eyeR/{eye_method}")

    badge = (
        f"blink={blink_method}: {'YES' if blink else 'no'}  "
        f"L={score_left:.3f}  R={score_right:.3f}"
    )
    color = (255, 64, 64) if blink else (200, 200, 200)
    draw.text((10, 10), badge, fill=color)

    image.save(out_path, quality=90)
