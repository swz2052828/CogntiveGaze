"""Streaming MediaPipe FaceMesh wrapper for video frames.

Video mode (static_image_mode=False) enables tracking, which is much faster
and more temporally stable than per-frame re-detection. Use as a context
manager so the FaceMesh session is closed cleanly when the video ends.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


LEFT_EYE_INDICES = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246,
]

RIGHT_EYE_INDICES = [
    362, 382, 381, 380, 374, 373, 390, 249, 263,
    466, 388, 387, 386, 385, 384, 398,
]

LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
    132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

NOSE_BRIDGE_INDICES = [168, 6, 197, 195, 5, 4, 1]

LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]


@dataclass
class LandmarkSet:
    """Pixel-space coordinates for the landmark groups we consume."""

    face_oval: np.ndarray
    left_eye: np.ndarray
    right_eye: np.ndarray
    left_iris: np.ndarray
    right_iris: np.ndarray
    nose_bridge: np.ndarray
    left_eye_ear: np.ndarray
    right_eye_ear: np.ndarray
    image_size: tuple


def _extract(proto, indices, width, height):
    coords = np.empty((len(indices), 2), dtype=np.float32)
    for i, idx in enumerate(indices):
        lm = proto.landmark[idx]
        coords[i, 0] = lm.x * width
        coords[i, 1] = lm.y * height
    return coords


def _landmark_set_from_proto(proto, width, height) -> LandmarkSet:
    return LandmarkSet(
        face_oval=_extract(proto, FACE_OVAL_INDICES, width, height),
        left_eye=_extract(proto, LEFT_EYE_INDICES, width, height),
        right_eye=_extract(proto, RIGHT_EYE_INDICES, width, height),
        left_iris=_extract(proto, LEFT_IRIS_INDICES, width, height),
        right_iris=_extract(proto, RIGHT_IRIS_INDICES, width, height),
        nose_bridge=_extract(proto, NOSE_BRIDGE_INDICES, width, height),
        left_eye_ear=_extract(proto, LEFT_EYE_EAR_INDICES, width, height),
        right_eye_ear=_extract(proto, RIGHT_EYE_EAR_INDICES, width, height),
        image_size=(width, height),
    )


class VideoFaceDetector:
    """FaceMesh in tracking mode. Use as a context manager per video."""

    def __init__(
        self,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        try:
            from mediapipe.solutions.face_mesh import FaceMesh
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required. Install with: pip install mediapipe"
            ) from exc
        self._mesh = FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_rgb: np.ndarray) -> Optional[LandmarkSet]:
        """Process one RGB HxWx3 frame; return None if no face is found."""
        result = self._mesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return None
        height, width = frame_rgb.shape[:2]
        return _landmark_set_from_proto(result.multi_face_landmarks[0], width, height)

    def close(self):
        self._mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
