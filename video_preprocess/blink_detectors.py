"""Three swappable blink detectors.

  ear              6-point Eye Aspect Ratio (current default). Score range
                   roughly 0.0 (closed) to 0.4 (wide open). Default
                   threshold 0.2.
  contour_ratio    Uses all 16 eye contour points: height / width of the
                   axis-aligned eye bbox. Similar polarity to EAR but uses
                   more landmarks so it is a touch more stable.
  iris_visibility  When the eye closes the iris is occluded, so the iris
                   landmark spread (radius proxy) shrinks. Score is the iris
                   radius normalised by inter-canthi distance; threshold
                   default 0.04. Independent of the eye-contour landmarks,
                   so good as a sanity check against the two contour-based
                   methods.

A blink is "either eye is closed", consistent across detectors.
"""

from typing import Optional, Tuple

import numpy as np

from .bbox import bbox_from_points
from .blink import _ear
from .detection_base import BBox, BlinkDetector
from .detector import LandmarkSet


class EARBlinkDetector(BlinkDetector):
    name = "ear"
    needs_mesh = True

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def detect(
        self,
        frame_rgb: np.ndarray,
        left_eye_bbox: Optional[BBox] = None,
        right_eye_bbox: Optional[BBox] = None,
        mesh_landmarks: Optional[LandmarkSet] = None,
    ) -> Tuple[bool, float, float]:
        if mesh_landmarks is None:
            return (False, float("nan"), float("nan"))
        ear_l = _ear(mesh_landmarks.left_eye_ear)
        ear_r = _ear(mesh_landmarks.right_eye_ear)
        return (ear_l < self.threshold or ear_r < self.threshold, ear_l, ear_r)


class ContourRatioBlinkDetector(BlinkDetector):
    """Height / width of the eye contour bbox. Closed eye -> small ratio."""

    name = "contour_ratio"
    needs_mesh = True

    def __init__(self, threshold: float = 0.18):
        self.threshold = threshold

    @staticmethod
    def _ratio(contour: np.ndarray) -> float:
        x0, y0, x1, y1 = bbox_from_points(contour)
        w = x1 - x0
        h = y1 - y0
        if w < 1e-6:
            return 0.0
        return h / w

    def detect(
        self,
        frame_rgb: np.ndarray,
        left_eye_bbox=None,
        right_eye_bbox=None,
        mesh_landmarks: Optional[LandmarkSet] = None,
    ) -> Tuple[bool, float, float]:
        if mesh_landmarks is None:
            return (False, float("nan"), float("nan"))
        r_l = self._ratio(mesh_landmarks.left_eye)
        r_r = self._ratio(mesh_landmarks.right_eye)
        return (r_l < self.threshold or r_r < self.threshold, r_l, r_r)


class IrisVisibilityBlinkDetector(BlinkDetector):
    """Iris radius (max landmark distance from iris center) / inter-canthi span.

    When the eye is closed the iris is occluded and the iris landmark spread
    shrinks. Normalising by the face's inter-canthi (eye-corner to eye-corner)
    distance makes the score scale-invariant across subjects.
    """

    name = "iris_visibility"
    needs_mesh = True

    def __init__(self, threshold: float = 0.04):
        self.threshold = threshold

    @staticmethod
    def _norm_radius(iris_pts: np.ndarray, scale: float) -> float:
        center = iris_pts.mean(axis=0)
        radius = float(np.linalg.norm(iris_pts - center, axis=1).max())
        if scale < 1e-6:
            return 0.0
        return radius / scale

    def detect(
        self,
        frame_rgb: np.ndarray,
        left_eye_bbox=None,
        right_eye_bbox=None,
        mesh_landmarks: Optional[LandmarkSet] = None,
    ) -> Tuple[bool, float, float]:
        if mesh_landmarks is None:
            return (False, float("nan"), float("nan"))
        # Inter-canthi distance from the two outer corners of left and right eye
        # (landmark indices 33 and 263 in FaceMesh, which sit at positions 0 and
        # 8 of our LEFT_EYE_INDICES and RIGHT_EYE_INDICES contours).
        left_outer = mesh_landmarks.left_eye[0]
        right_outer = mesh_landmarks.right_eye[8]
        inter_canthi = float(np.linalg.norm(right_outer - left_outer))
        score_l = self._norm_radius(mesh_landmarks.left_iris, inter_canthi)
        score_r = self._norm_radius(mesh_landmarks.right_iris, inter_canthi)
        return (
            score_l < self.threshold or score_r < self.threshold,
            score_l,
            score_r,
        )


BLINK_DETECTORS = {
    EARBlinkDetector.name: EARBlinkDetector,
    ContourRatioBlinkDetector.name: ContourRatioBlinkDetector,
    IrisVisibilityBlinkDetector.name: IrisVisibilityBlinkDetector,
}


def build_blink_detector(name: str, threshold: Optional[float] = None) -> BlinkDetector:
    if name not in BLINK_DETECTORS:
        raise ValueError(
            f"Unknown blink detector '{name}'. Available: {list(BLINK_DETECTORS)}"
        )
    cls = BLINK_DETECTORS[name]
    return cls(threshold=threshold) if threshold is not None else cls()
