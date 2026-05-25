"""MediaPipe FaceMesh wrapper for per-frame landmark detection.

We need landmarks for two reasons: (1) build a mask that follows the actual eyes
when the head tilts or translates, and (2) locate the iris precisely so we never
let the diffusion model touch it.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


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


@dataclass
class LandmarkSet:
    """Pixel-space coordinates for the regions we care about."""

    face_oval: np.ndarray
    left_eye: np.ndarray
    right_eye: np.ndarray
    left_iris: np.ndarray
    right_iris: np.ndarray
    nose_bridge: np.ndarray
    image_size: tuple


def _import_face_mesh():
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for landmark-based masks. "
            "Install with: pip install mediapipe"
        ) from exc
    return mp.solutions.face_mesh


def _extract(landmarks_proto, indices, width, height):
    coords = np.empty((len(indices), 2), dtype=np.float32)
    for i, idx in enumerate(indices):
        lm = landmarks_proto.landmark[idx]
        coords[i, 0] = lm.x * width
        coords[i, 1] = lm.y * height
    return coords


def detect_landmarks(image: Image.Image) -> Optional[LandmarkSet]:
    """Return landmarks for the most prominent face, or None if none found."""
    face_mesh_module = _import_face_mesh()
    width, height = image.size
    rgb = np.asarray(image.convert("RGB"))

    with face_mesh_module.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as mesh:
        result = mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    proto = result.multi_face_landmarks[0]
    return LandmarkSet(
        face_oval=_extract(proto, FACE_OVAL_INDICES, width, height),
        left_eye=_extract(proto, LEFT_EYE_INDICES, width, height),
        right_eye=_extract(proto, RIGHT_EYE_INDICES, width, height),
        left_iris=_extract(proto, LEFT_IRIS_INDICES, width, height),
        right_iris=_extract(proto, RIGHT_IRIS_INDICES, width, height),
        nose_bridge=_extract(proto, NOSE_BRIDGE_INDICES, width, height),
        image_size=(width, height),
    )
