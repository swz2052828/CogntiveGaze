"""Eye Aspect Ratio (EAR)-based blink detection.

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
where (p1, p4) are the eye corners and (p2, p3, p5, p6) are upper/lower lid
landmarks. EAR drops sharply when the eye closes; a threshold around 0.20 is
typical for "eye closed". Open-eye EAR is usually 0.25-0.35 depending on the
subject and camera angle.

Calibrate per dataset if needed: print EAR distributions on a sample and pick
a threshold below the open-eye mode but above the closed-eye mode.
"""

from typing import Tuple

import numpy as np

from .detector import LandmarkSet


def _ear(points6: np.ndarray) -> float:
    """6 landmark points around one eye -> scalar EAR."""
    p1, p2, p3, p4, p5, p6 = points6
    v1 = float(np.linalg.norm(p2 - p6))
    v2 = float(np.linalg.norm(p3 - p5))
    h = float(np.linalg.norm(p1 - p4))
    if h < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h)


def compute_ear_per_eye(landmarks: LandmarkSet) -> Tuple[float, float]:
    """Return (ear_left, ear_right) using the 6-point landmark sets."""
    return _ear(landmarks.left_eye_ear), _ear(landmarks.right_eye_ear)


def is_blink(ear_left: float, ear_right: float, threshold: float = 0.2) -> bool:
    """Either eye below threshold counts as a blink for that frame."""
    return ear_left < threshold or ear_right < threshold
