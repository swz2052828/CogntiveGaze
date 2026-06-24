"""Swappable backbones for vit_gaze multistream mode.

Each backbone is a single nn.Module that exposes the same forward signature:

    model(face, eye_left, eye_right, grid) -> (B, 2) gaze prediction

A small ABC `MultistreamBackboneBase` documents the convention. Concrete
backbones:

  vit          MultiStreamViTGaze (shared ViT-B/16 + optional grid MLP, ours)
  foveal_vit   FovealViTMultistream - single ViT-B/16 over a concatenated
               token sequence (face low-res + eyes high-res + grid token),
               with cross-region attention and learnable region-type embeddings.
  eyes_only_vit  EyesOnlyViTGaze - shared ViT-B/16 over the two eye crops
               only; face and grid are ignored. Ablation: does per-subject
               calibration absorb the head-pose / distance info that face+grid
               otherwise provide?
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
    OUTPUT_ACTIVATIONS,
    REQUIRES_GRID,
    SUPPORTS_NO_GRID,
    MultistreamBackboneBase,
    attach_output_activation,
    build_multistream_backbone,
)
from .affnet import AFFNetMultistream
from .cnn_transformer import CNNTransformerGaze
from .eyes_only import EyesOnlyViTGaze
from .eyes_only_mgazenet import EyesOnlyMGazeNetGaze
from .eyes_only_mobile_vit import EyesOnlyMobileViTGaze
from .eyes_only_mobilenet_v3 import EyesOnlyMobileNetV3Gaze
from .eyes_only_mobilenet_v4 import EyesOnlyMobileNetV4Gaze
from .eyes_only_fastvit import EyesOnlyFastViTGaze
from .eyes_only_convnextv2 import EyesOnlyConvNeXtV2Gaze
from .eyes_only_mobilevitv2 import EyesOnlyMobileViTv2Gaze
from .face_only_mobile_vit import FaceOnlyMobileViTGaze
from .mobilevitv2_multistream import MobileViTv2Multistream
from .repvit_multistream import RepViTMultistream
from .convnextv2_multistream import ConvNeXtV2Multistream, ConvNeXtV2NanoMultistream
from .convnextv2_dualenc import ConvNeXtV2DualEncMultistream
from .convnextv2_film import ConvNeXtV2FiLMMultistream
from .convnextv2_atto_multistream import ConvNeXtV2AttoMultistream
from .dinov2_multistream import DINOv2Multistream
from .eva02_multistream import EVA02Multistream
from .eyes_only_eva02_tiny import EyesOnlyEVA02TinyGaze
from .eyes_only_convnextv2_atto import EyesOnlyConvNeXtV2AttoGaze
from .eyes_only_convnextv2_binocular import EyesOnlyConvNeXtV2BinocularGaze, EyesOnlyConvNeXtV2AttoBinocularGaze
from .normface_convnext import NormFaceConvNeXtMultistream
from .foveal_vit import FovealViTMultistream
from .itracker import ITrackerMultistream
from .mgazenet import MGazeNetMultistream
from .convnext import ConvNeXtMultistream
from .mobile_vit import MobileViTMultistream
from .mobilenet_v3 import MobileNetV3Multistream
from .mobilenet_v4 import MobileNetV4Multistream
from .vit_shared import MultiStreamViTGaze

__all__ = [
    "MultistreamBackboneBase",
    "MultiStreamViTGaze",
    "EyesOnlyViTGaze",
    "EyesOnlyMobileViTGaze",
    "EyesOnlyMGazeNetGaze",
    "EyesOnlyMobileNetV3Gaze",
    "EyesOnlyMobileNetV4Gaze",
    "EyesOnlyFastViTGaze",
    "EyesOnlyConvNeXtV2Gaze",
    "EyesOnlyMobileViTv2Gaze",
    "FaceOnlyMobileViTGaze",
    "MobileViTv2Multistream",
    "RepViTMultistream",
    "ConvNeXtV2Multistream",
    "ConvNeXtV2NanoMultistream",
    "ConvNeXtV2DualEncMultistream",
    "ConvNeXtV2FiLMMultistream",
    "ConvNeXtV2AttoMultistream",
    "DINOv2Multistream",
    "EVA02Multistream",
    "EyesOnlyEVA02TinyGaze",
    "EyesOnlyConvNeXtV2AttoGaze",
    "EyesOnlyConvNeXtV2BinocularGaze",
    "EyesOnlyConvNeXtV2AttoBinocularGaze",
    "NormFaceConvNeXtMultistream",
    "CNNTransformerGaze",
    "FovealViTMultistream",
    "ITrackerMultistream",
    "MobileViTMultistream",
    "MobileNetV3Multistream",
    "ConvNeXtMultistream",
    "MobileNetV4Multistream",
    "AFFNetMultistream",
    "MGazeNetMultistream",
    "REQUIRES_GRID",
    "SUPPORTS_NO_GRID",
    "OUTPUT_ACTIVATIONS",
    "attach_output_activation",
    "build_multistream_backbone",
]
