"""Accumulate per-frame metadata and write iTracker-format metadata.mat.

Required fields for vit_gaze.MultiStreamGazeDataset:
  labelRecNum   int32   recording id per row
  frameIndex    int32   frame index per row
  labelDotXCam  float32 gaze X in cm (NaN if unknown)
  labelDotYCam  float32 gaze Y in cm (NaN if unknown)
  labelFaceGrid (N, 4)  face-grid params [x0, y0, w, h] in 25x25 grid coords

Optional extras (handy downstream, ignored by the dataset loader):
  earLeft       float32
  earRight      float32
  blink         int32 (0/1)
  faceBbox      (N, 4) face square bbox in pixel coords
  leftEyeBbox   (N, 4)
  rightEyeBbox  (N, 4)
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import scipy.io as sio


REQUIRED_KEYS = (
    "labelRecNum",
    "frameIndex",
    "labelDotXCam",
    "labelDotYCam",
    "labelFaceGrid",
)

EXTRA_KEYS = (
    "earLeft",
    "earRight",
    "blink",
    "faceBbox",
    "leftEyeBbox",
    "rightEyeBbox",
)


class MetadataAccumulator:
    """Append rows during processing, then write metadata.mat in one shot."""

    def __init__(self):
        self.rows = []

    def add(
        self,
        rec_num: int,
        frame_index: int,
        face_grid_params: Sequence[int],
        gaze_x: float = float("nan"),
        gaze_y: float = float("nan"),
        ear_left: float = float("nan"),
        ear_right: float = float("nan"),
        blink: bool = False,
        face_bbox: Sequence[float] = (),
        left_eye_bbox: Sequence[float] = (),
        right_eye_bbox: Sequence[float] = (),
    ):
        self.rows.append(
            {
                "labelRecNum": int(rec_num),
                "frameIndex": int(frame_index),
                "labelDotXCam": float(gaze_x),
                "labelDotYCam": float(gaze_y),
                "labelFaceGrid": list(face_grid_params),
                "earLeft": float(ear_left),
                "earRight": float(ear_right),
                "blink": int(bool(blink)),
                "faceBbox": list(face_bbox),
                "leftEyeBbox": list(left_eye_bbox),
                "rightEyeBbox": list(right_eye_bbox),
            }
        )

    def extend(self, rows):
        self.rows.extend(rows)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            raise RuntimeError("No metadata rows to write.")

        data = {}
        for key in REQUIRED_KEYS + EXTRA_KEYS:
            values = [row.get(key) for row in self.rows]
            if key in ("labelRecNum", "frameIndex", "blink"):
                data[key] = np.asarray(values, dtype=np.int32)
            elif key in ("labelDotXCam", "labelDotYCam", "earLeft", "earRight"):
                data[key] = np.asarray(values, dtype=np.float32)
            else:
                data[key] = np.asarray(values, dtype=np.float32)
        sio.savemat(path, data)
        return path
