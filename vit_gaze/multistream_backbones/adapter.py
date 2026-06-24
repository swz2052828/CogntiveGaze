"""Common ABC and factory for multistream backbones.

All backbones accept (face, eye_left, eye_right, grid) and return (B, 2) gaze
predictions. Some backbones internally require the grid as a conditioning
factor (AFFNet, MGazeNet); others accept None.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class MultistreamBackboneBase(nn.Module, ABC):
    """Each backbone is one nn.Module with this fixed forward signature."""

    requires_grid: bool = False

    @abstractmethod
    def forward(
        self,
        face: torch.Tensor,
        eye_left: torch.Tensor,
        eye_right: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...

    def forward_features(
        self,
        face: torch.Tensor,
        eye_left: torch.Tensor,
        eye_right: torch.Tensor,
        grid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the fused per-stream vector that the final readout consumes.

        Backbones that implement this (and end their ``forward`` in a single
        readout module) opt into the meta-learned calibration path
        (``metatrain`` / ``metacompare`` / meta export). The default raises so
        callers can detect unsupported backbones.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose forward_features; it cannot "
            f"be used with the meta-learned calibration path.")

    @property
    def readout(self) -> nn.Module:
        """The final regression module (``.head`` for ViT, ``.fc`` for CNN baselines).

        The adapter modulates ``forward_features`` output; this is the module
        that maps the (modulated) fused vector to the 2D gaze prediction.
        """
        for name in ("head", "fc"):
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                return module
        raise AttributeError(
            f"{type(self).__name__} has neither a .head nor a .fc readout module.")

    def calibration_feature(self, fused: torch.Tensor) -> torch.Tensor:
        """Penultimate readout activation -- the compact pre-output embedding.

        This is the faithful analogue of Zhu et al.'s ``gaze_feature`` bottleneck
        (SwarmIntelligentCalibration): the activation that *feeds* the final
        ``Linear(.,2)``, NOT the wide ``forward_features`` vector. For the ViT
        head (LayerNorm -> Linear -> GELU -> Dropout -> Linear(512,128) -> GELU
        -> Linear(128,2)) this is 128-d, vs 2432-d for ``forward_features``.
        Used as the per-subject SVR calibration feature so the SVR replaces only
        the final linear readout, matching their recipe (they fit SVR on a 256-d
        ``gaze_feature``, not the raw backbone output).

        ``fused`` is the ``forward_features`` output, so callers that already
        have it avoid a second encoder pass.
        """
        readout = self.readout
        if not isinstance(readout, nn.Sequential) or len(readout) < 2:
            raise NotImplementedError(
                f"{type(self).__name__}.readout is not a trimmable Sequential "
                f"head; embedding-space SVR calibration needs a head ending in "
                f"a final Linear(.,2).")
        return readout[:-1](fused)


REQUIRES_GRID = ("itracker", "mobilenet_v3", "affnet", "mgazenet")
SUPPORTS_NO_GRID = ("vit", "foveal_vit", "vivit", "eyes_only_vit", "eyes_only_mobile_vit", "eyes_only_mgazenet", "eyes_only_mobilenet_v3", "eyes_only_mobilenet_v4", "eyes_only_fastvit", "face_only_mobile_vit", "mobile_vit", "mobilevitv2", "repvit", "convnextv2", "normface_convnext", "cnn_transformer", "cnn_transformer_raw", "convnext", "mobilenet_v4")


# ---------------------------------------------------------------------------
# Optional output activation on the final gaze prediction.
#
# Gaze targets are z-scored (training.normalize_gaze), so the model predicts in
# standardized units that span roughly [-4, +4] (screen corners are several std
# from the per-fold mean). A bounded activation must therefore be *scaled* by a
# gaze_range that covers that span, or the periphery becomes unreachable.
#
# Applied as a forward hook on the backbone so it transforms the final (B, 2)
# prediction in forward() WITHOUT touching forward_features / readout. NB: the
# meta-learned and SVR calibration paths bypass forward() (they fit a separate
# readout on forward_features), so the activation is a base-model transform --
# evaluate it base-only, not through the meta/SVR pipeline.
# ---------------------------------------------------------------------------
OUTPUT_ACTIVATIONS = ("none", "scaled_tanh", "scaled_sin")


def _output_activation_fn(kind: str, gaze_range: float):
    if kind == "scaled_tanh":
        return lambda out: gaze_range * torch.tanh(out)
    if kind == "scaled_sin":
        return lambda out: gaze_range * torch.sin(out)
    raise ValueError(
        f"Unknown output_activation '{kind}'. Choices: {OUTPUT_ACTIVATIONS}.")


def attach_output_activation(model, kind: str = "none", gaze_range: float = 4.0):
    """Register a forward hook that maps the (B, 2) gaze output through ``kind``.

    No-op when ``kind == 'none'``. Returns ``model`` for chaining.
    """
    if kind == "none":
        return model
    fn = _output_activation_fn(kind, float(gaze_range))

    def _hook(_module, _inputs, output):
        return fn(output)

    model.register_forward_hook(_hook)
    # Record for introspection / checkpoint round-tripping.
    model._output_activation = kind
    model._gaze_range = float(gaze_range)
    return model


def build_multistream_backbone(
    backbone: str,
    weights: str = "none",
    freeze_encoder: bool = False,
    use_grid: bool = False,
    grid_size: int = 25,
    # vivit-specific (ignored for other backbones)
    vivit_spatial: str = "vit",
    vivit_temporal_window: int = 8,
    vivit_temporal_layers: int = 4,
    vivit_temporal_heads: int = 8,
) -> MultistreamBackboneBase:
    """Factory. Validates the grid requirement before instantiating."""

    if backbone in REQUIRES_GRID and not use_grid:
        raise ValueError(
            f"--backbone {backbone} requires --use-grid (the architecture "
            f"conditions on or concatenates the face-grid). Pass --use-grid "
            f"or pick --backbone vit (the only one with optional grid)."
        )

    if backbone == "vit":
        from .vit_shared import MultiStreamViTGaze

        return MultiStreamViTGaze(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "itracker":
        from .itracker import ITrackerMultistream

        return ITrackerMultistream(grid_size=grid_size)
    if backbone == "mobilenet_v3":
        from .mobilenet_v3 import MobileNetV3Multistream

        return MobileNetV3Multistream(
            weights=weights,
            mobilenet_type="large",
            grid_size=grid_size,
        )
    if backbone == "affnet":
        from .affnet import AFFNetMultistream

        return AFFNetMultistream(grid_size=grid_size)
    if backbone == "mgazenet":
        from .mgazenet import MGazeNetMultistream

        return MGazeNetMultistream(grid_size=grid_size)
    if backbone == "foveal_vit":
        from .foveal_vit import FovealViTMultistream

        return FovealViTMultistream(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "vivit":
        from .vivit import build_vivit

        return build_vivit(
            spatial_backbone_name=vivit_spatial,
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
            temporal_window=vivit_temporal_window,
            num_temporal_layers=vivit_temporal_layers,
            num_temporal_heads=vivit_temporal_heads,
        )
    if backbone == "eyes_only_vit":
        from .eyes_only import EyesOnlyViTGaze

        return EyesOnlyViTGaze(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "eyes_only_mobile_vit":
        from .eyes_only_mobile_vit import EyesOnlyMobileViTGaze

        return EyesOnlyMobileViTGaze(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "eyes_only_mgazenet":
        from .eyes_only_mgazenet import EyesOnlyMGazeNetGaze

        return EyesOnlyMGazeNetGaze(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "eyes_only_mobilenet_v3":
        from .eyes_only_mobilenet_v3 import EyesOnlyMobileNetV3Gaze

        return EyesOnlyMobileNetV3Gaze(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "eyes_only_fastvit":
        from .eyes_only_fastvit import EyesOnlyFastViTGaze

        return EyesOnlyFastViTGaze(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "mobilevitv2":
        from .mobilevitv2_multistream import MobileViTv2Multistream

        return MobileViTv2Multistream(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "repvit":
        from .repvit_multistream import RepViTMultistream

        return RepViTMultistream(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "convnextv2":
        from .convnextv2_multistream import ConvNeXtV2Multistream

        return ConvNeXtV2Multistream(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "normface_convnext":
        from .normface_convnext import NormFaceConvNeXtMultistream

        return NormFaceConvNeXtMultistream(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "eyes_only_mobilenet_v4":
        from .eyes_only_mobilenet_v4 import EyesOnlyMobileNetV4Gaze

        return EyesOnlyMobileNetV4Gaze(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "face_only_mobile_vit":
        from .face_only_mobile_vit import FaceOnlyMobileViTGaze

        return FaceOnlyMobileViTGaze(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "mobile_vit":
        from .mobile_vit import MobileViTMultistream

        return MobileViTMultistream(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone in ("cnn_transformer", "cnn_transformer_raw"):
        from .cnn_transformer import CNNTransformerGaze

        return CNNTransformerGaze(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
            preserve_meta_contract=(backbone == "cnn_transformer"),
        )
    if backbone == "convnext":
        from .convnext import ConvNeXtMultistream

        return ConvNeXtMultistream(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    if backbone == "mobilenet_v4":
        from .mobilenet_v4 import MobileNetV4Multistream

        return MobileNetV4Multistream(
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    raise ValueError(
        f"Unknown backbone '{backbone}'. Choices: vit, foveal_vit, vivit, "
        f"eyes_only_vit, eyes_only_mobile_vit, eyes_only_mgazenet, "
        f"eyes_only_mobilenet_v3, eyes_only_mobilenet_v4, eyes_only_fastvit, "
        f"face_only_mobile_vit, mobile_vit, mobilevitv2, repvit, convnextv2, "
        f"normface_convnext, cnn_transformer, "
        f"cnn_transformer_raw, convnext, mobilenet_v4, itracker, mobilenet_v3, "
        f"affnet, mgazenet."
    )
