"""Preprocess smartphone video into iTracker-format face/eye crops + metadata.

Pipeline (per video):
  OpenCV VideoCapture
    -> MediaPipe FaceMesh (video mode, refine_landmarks=True for iris)
    -> face / left eye / right eye bounding boxes (squared + padded)
    -> 224x224 crops saved as <out>/<rec>/appleFace|appleLeftEye|appleRightEye/<frame>.jpg
    -> 25x25 face-grid params + per-eye EAR + blink flag
    -> appended to a metadata.mat row
  -> final metadata.mat dropped at <out>/<mean_path>/metadata.mat
     compatible with vit_gaze.MultiStreamGazeDataset and the project's CNN baselines.

The gaze label (labelDotXCam / labelDotYCam) comes from your stimulus
protocol, not the video itself. Pass an aligned --gaze-csv per video to fill
those in, otherwise they are NaN (useful for inference-only datasets).
"""

from .bbox import bbox_from_points, pad_bbox, square_bbox, to_int_box
from .blink import compute_ear_per_eye, is_blink
from .detector import LandmarkSet, VideoFaceDetector
from .face_grid import face_grid_params
from .metadata_writer import MetadataAccumulator
from .pipeline import process_video

__all__ = [
    "LandmarkSet",
    "VideoFaceDetector",
    "bbox_from_points",
    "pad_bbox",
    "square_bbox",
    "to_int_box",
    "compute_ear_per_eye",
    "is_blink",
    "face_grid_params",
    "MetadataAccumulator",
    "process_video",
]
