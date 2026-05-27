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


REQUIRES_GRID = ("itracker", "mobilenet_v3", "affnet", "mgazenet")
SUPPORTS_NO_GRID = ("vit",)


def build_multistream_backbone(
    backbone: str,
    weights: str = "none",
    freeze_encoder: bool = False,
    use_grid: bool = False,
    grid_size: int = 25,
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
    raise ValueError(
        f"Unknown backbone '{backbone}'. Choices: vit, itracker, "
        f"mobilenet_v3, affnet, mgazenet."
    )
