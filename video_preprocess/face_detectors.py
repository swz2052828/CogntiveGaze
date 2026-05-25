"""Three swappable face detectors.

  mediapipe_facemesh   478 landmarks, bbox from face_oval (current default)
  mediapipe_facedetect BlazeFace, bbox directly, ~10x faster than FaceMesh
  opencv_haar          Classical Haar cascade, no extra deps, fastest CPU

The MediaPipe FaceMesh detector returns a bbox but ALSO carries landmarks the
downstream eye/blink detectors can reuse; the pipeline sets needs_mesh=True
on this option so the FaceMesh session is shared across the three stages.
"""

from typing import Optional

import numpy as np

from .bbox import bbox_from_points
from .detection_base import BBox, FaceDetector
from .detector import LandmarkSet


class MediaPipeFaceMeshFaceDetector(FaceDetector):
    name = "mediapipe_facemesh"
    needs_mesh = True

    def detect(self, frame_rgb, mesh_landmarks: Optional[LandmarkSet] = None):
        if mesh_landmarks is None:
            return None
        return bbox_from_points(mesh_landmarks.face_oval)


class MediaPipeFaceDetectionFaceDetector(FaceDetector):
    """BlazeFace via mediapipe.solutions.face_detection. Bbox only, no landmarks."""

    name = "mediapipe_facedetect"
    needs_mesh = False

    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 1):
        try:
            from mediapipe.solutions.face_detection import FaceDetection
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required. Install: pip install mediapipe"
            ) from exc
        # model_selection=0 short-range (within 2m), 1 full-range (up to 5m).
        # Desktop setups at 80-120cm work better with 0; default kept at 1 for
        # general use, override if needed.
        self._det = FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame_rgb: np.ndarray, mesh_landmarks=None) -> Optional[BBox]:
        result = self._det.process(frame_rgb)
        if not result.detections:
            return None
        height, width = frame_rgb.shape[:2]
        # Pick the highest-score detection
        best = max(result.detections, key=lambda d: d.score[0] if d.score else 0.0)
        box = best.location_data.relative_bounding_box
        x0 = max(0.0, box.xmin) * width
        y0 = max(0.0, box.ymin) * height
        x1 = min(1.0, box.xmin + box.width) * width
        y1 = min(1.0, box.ymin + box.height) * height
        return (float(x0), float(y0), float(x1), float(y1))

    def close(self):
        self._det.close()


class OpenCVHaarFaceDetector(FaceDetector):
    """Classical Viola-Jones face detector. Uses the cascade bundled with opencv."""

    name = "opencv_haar"
    needs_mesh = False

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 60,
    ):
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required.") from exc
        self._cv2 = cv2
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame_rgb: np.ndarray, mesh_landmarks=None) -> Optional[BBox]:
        gray = self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2GRAY)
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        if len(rects) == 0:
            return None
        # Pick the largest face (most likely the subject)
        x, y, w, h = max(rects, key=lambda r: r[2] * r[3])
        return (float(x), float(y), float(x + w), float(y + h))


FACE_DETECTORS = {
    MediaPipeFaceMeshFaceDetector.name: MediaPipeFaceMeshFaceDetector,
    MediaPipeFaceDetectionFaceDetector.name: MediaPipeFaceDetectionFaceDetector,
    OpenCVHaarFaceDetector.name: OpenCVHaarFaceDetector,
}


def build_face_detector(name: str) -> FaceDetector:
    if name not in FACE_DETECTORS:
        raise ValueError(
            f"Unknown face detector '{name}'. Available: {list(FACE_DETECTORS)}"
        )
    return FACE_DETECTORS[name]()
