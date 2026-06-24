"""Common interfaces for swappable face / eye / blink detectors.

Each detector type follows a small ABC. The pipeline picks one of each via the
CLI; if any of the three chose a FaceMesh-based variant, the FaceMesh session
is run once per frame and the resulting LandmarkSet is passed in to whichever
detector wants it.

All bounding boxes are (x0, y0, x1, y1) in pixel coordinates, top-left origin.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from .detector import LandmarkSet


BBox = Tuple[float, float, float, float]


class FaceDetector(ABC):
    """Detect the single most prominent face. Return (x0, y0, x1, y1) or None."""

    name: str = "abstract"
    needs_mesh: bool = False

    @abstractmethod
    def detect(
        self, frame_rgb: np.ndarray, mesh_landmarks: Optional[LandmarkSet] = None
    ) -> Optional[BBox]:
        ...

    def close(self) -> None:
        pass


class EyeDetector(ABC):
    """Detect left and right eye bounding boxes given a frame.

    Some detectors use the face bbox as a region-of-interest. Others use the
    FaceMesh landmarks directly. Pass both; the detector uses what it needs.
    """

    name: str = "abstract"
    needs_mesh: bool = False

    @abstractmethod
    def detect(
        self,
        frame_rgb: np.ndarray,
        face_bbox: Optional[BBox] = None,
        mesh_landmarks: Optional[LandmarkSet] = None,
    ) -> Optional[Tuple[BBox, BBox]]:
        ...

    def close(self) -> None:
        pass


class BlinkDetector(ABC):
    """Classify whether the subject is blinking on this frame.

    Returns (is_blink, score_left, score_right). The two scores are method-
    specific (EAR for the EAR detector, height/width for the contour-ratio
    detector, iris-area z-score for the iris detector) but their semantics are
    consistent within a detector so you can plot them across frames.
    """

    name: str = "abstract"
    needs_mesh: bool = False

    @abstractmethod
    def detect(
        self,
        frame_rgb: np.ndarray,
        left_eye_bbox: Optional[BBox] = None,
        right_eye_bbox: Optional[BBox] = None,
        mesh_landmarks: Optional[LandmarkSet] = None,
    ) -> Tuple[bool, float, float]:
        ...

    def close(self) -> None:
        pass
