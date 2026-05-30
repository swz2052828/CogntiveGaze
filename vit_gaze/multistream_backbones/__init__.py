"""Swappable backbones for vit_gaze multistream mode.

Each backbone is a single nn.Module that exposes the same forward signature:

    model(face, eye_left, eye_right, grid) -> (B, 2) gaze prediction

A small ABC `MultistreamBackboneBase` documents the convention. Concrete
backbones:

  vit          MultiStreamViTGaze (shared ViT-B/16 + optional grid MLP, ours)
  foveal_vit   FovealViTMultistream - single ViT-B/16 over a concatenated
               token sequence (face low-res + eyes high-res + grid token),
               with cross-region attention and learnable region-type embeddings.
  itracker     ITrackerCNN, the original GazeCapture iTracker (AlexNet-ish)
  mobilenet_v3 MobileNetV3-Large feature extractor with the iTracker fusion head
  affnet       GazeAGNModel - Adaptive Group Normalisation, eyes conditioned on
               (face, grid). Grid is required (used as a conditioning factor).
  mgazenet     MGazeNet - same idea as AFFNet but with LABN + SE blocks. Grid
               is required.

The build_multistream_backbone() factory dispatches by name and validates the
grid/--use-grid combination before instantiating.
"""

from .adapter import (
    REQUIRES_GRID,
    SUPPORTS_NO_GRID,
    MultistreamBackboneBase,
    build_multistream_backbone,
)
from .affnet import AFFNetMultistream
from .foveal_vit import FovealViTMultistream
from .itracker import ITrackerMultistream
from .mgazenet import MGazeNetMultistream
from .mobilenet_v3 import MobileNetV3Multistream
from .vit_shared import MultiStreamViTGaze

__all__ = [
    "MultistreamBackboneBase",
    "MultiStreamViTGaze",
    "FovealViTMultistream",
    "ITrackerMultistream",
    "MobileNetV3Multistream",
    "AFFNetMultistream",
    "MGazeNetMultistream",
    "REQUIRES_GRID",
    "SUPPORTS_NO_GRID",
    "build_multistream_backbone",
]
