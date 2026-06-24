"""Three swappable eye detectors.

  facemesh_contour Uses FaceMesh's 16-point eye contour per side (current default)
  facemesh_iris    Centres a fixed-size box on the iris landmarks (tightest)
  opencv_haar      Classical Haar eye cascade, runs inside the face bbox ROI

Each returns (left_eye_bbox, right_eye_bbox). "Left" is the subject's left eye
(viewer's right side), consistent with iTracker / GazeCapture convention.
"""

from typing import Optional, Tuple

import numpy as np

from .bbox import bbox_from_points
from .detection_base import BBox, EyeDetector
from .detector import LandmarkSet


class FaceMeshContourEyeDetector(EyeDetector):
    name = "facemesh_contour"
    needs_mesh = True

    def detect(
        self,
        frame_rgb: np.ndarray,
        face_bbox: Optional[BBox] = None,
        mesh_landmarks: Optional[LandmarkSet] = None,
    ) -> Optional[Tuple[BBox, BBox]]:
        if mesh_landmarks is None:
            return None
        return (
            bbox_from_points(mesh_landmarks.left_eye),
            bbox_from_points(mesh_landmarks.right_eye),
        )


class FaceMeshIrisEyeDetector(EyeDetector):
    """Centre a box of width = iris_radius * box_factor on each iris centre."""

    name = "facemesh_iris"
    needs_mesh = True

    def __init__(self, box_factor: float = 6.0):
        self.box_factor = box_factor

    @staticmethod
    def _box_around(center_xy, side):
        cx, cy = center_xy
        half = side / 2.0
        return (cx - half, cy - half, cx + half, cy + half)

    def detect(
        self,
        frame_rgb: np.ndarray,
        face_bbox: Optional[BBox] = None,
        mesh_landmarks: Optional[LandmarkSet] = None,
    ) -> Optional[Tuple[BBox, BBox]]:
        if mesh_landmarks is None:
            return None
        left_center = mesh_landmarks.left_iris.mean(axis=0)
        right_center = mesh_landmarks.right_iris.mean(axis=0)
        left_radius = float(
            np.linalg.norm(mesh_landmarks.left_iris - left_center, axis=1).max()
        )
        right_radius = float(
            np.linalg.norm(mesh_landmarks.right_iris - right_center, axis=1).max()
        )
        return (
            self._box_around(left_center, left_radius * self.box_factor),
            self._box_around(right_center, right_radius * self.box_factor),
        )


class OpenCVHaarEyeDetector(EyeDetector):
    """Classical Haar eye cascade, run inside the face bbox ROI."""

    name = "opencv_haar"
    needs_mesh = False

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required.") from exc
        self._cv2 = cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load eye cascade from {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(
        self,
        frame_rgb: np.ndarray,
        face_bbox: Optional[BBox] = None,
        mesh_landmarks=None,
    ) -> Optional[Tuple[BBox, BBox]]:
        if face_bbox is None:
            return None
        x0, y0, x1, y1 = (int(round(v)) for v in face_bbox)
        roi = frame_rgb[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        gray = self._cv2.cvtColor(roi, self._cv2.COLOR_RGB2GRAY)
        # Restrict to upper half of face (eyes are not in the chin/mouth area)
        h, _ = gray.shape[:2]
        gray_upper = gray[: int(h * 0.6), :]
        rects = self._cascade.detectMultiScale(
            gray_upper,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
        )
        if len(rects) < 2:
            return None
        # Keep the two largest detections, then sort by x-center (subject's left
        # eye is on viewer's right = larger x)
        top_two = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)[:2]
        top_two = sorted(top_two, key=lambda r: r[0] + r[2] / 2.0)
        # Viewer's left rect (smaller x) is the subject's right eye
        viewer_left, viewer_right = top_two
        right_eye_rect = viewer_left
        left_eye_rect = viewer_right

        def to_full(rect):
            rx, ry, rw, rh = rect
            return (
                float(x0 + rx),
                float(y0 + ry),
                float(x0 + rx + rw),
                float(y0 + ry + rh),
            )

        return (to_full(left_eye_rect), to_full(right_eye_rect))


EYE_DETECTORS = {
    FaceMeshContourEyeDetector.name: FaceMeshContourEyeDetector,
    FaceMeshIrisEyeDetector.name: FaceMeshIrisEyeDetector,
    OpenCVHaarEyeDetector.name: OpenCVHaarEyeDetector,
}


def build_eye_detector(name: str) -> EyeDetector:
    if name not in EYE_DETECTORS:
        raise ValueError(
            f"Unknown eye detector '{name}'. Available: {list(EYE_DETECTORS)}"
        )
    return EYE_DETECTORS[name]()
