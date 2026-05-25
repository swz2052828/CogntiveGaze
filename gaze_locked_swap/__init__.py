"""Reference-guided face swap with landmark-tight eye preservation and gaze QC.

Pipeline (per frame):
  source image -> face landmarks (MediaPipe FaceMesh, with iris refinement)
              -> tight feathered eye/iris protection mask
              -> SD inpainting (optionally IP-Adapter-conditioned on a reference face)
              -> hard copy-back of source's eye/iris pixels via the feathered mask
              -> gaze QC using the frozen ViT gaze checkpoint
"""

from .compose import feather_composite, hard_paste_protected
from .gaze_qc import GazeChecker
from .identity_pipeline import IdentitySwapPipeline
from .landmarks import LandmarkSet, detect_landmarks
from .protection_mask import (
    build_landmark_mask,
    fallback_geometric_mask,
    feather_mask,
)
from .swap import SwapResult, swap_one

__all__ = [
    "LandmarkSet",
    "detect_landmarks",
    "build_landmark_mask",
    "fallback_geometric_mask",
    "feather_mask",
    "feather_composite",
    "hard_paste_protected",
    "IdentitySwapPipeline",
    "GazeChecker",
    "SwapResult",
    "swap_one",
]
