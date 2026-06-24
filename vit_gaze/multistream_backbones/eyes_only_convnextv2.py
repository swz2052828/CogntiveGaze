"""Eyes-only ConvNeXtV2 multistream backbone (no face crop, no face grid).

Shared ConvNeXtV2-Femto encoder over LEFT eye + RIGHT eye only; the two eye
feature vectors are concatenated and fed to the regression head. The face crop
and face grid are accepted (interface compatibility) but ignored.

Motivation: the full convnextv2 backbone was the strongest result so far -- best
uncalibrated base (4.33px) AND best svr_embed@64 (1.83px) at only 5.3M params,
validating ConvNeXtV2's GRN + FCMAE pretraining. Our most robust finding is that
*eyes-only ~= full* after calibration (eyes_only_mobilenet_v4 even beat full).
This backbone tests whether the best encoder keeps its accuracy when stripped to
eyes-only (~3.5M params) -- the headline edge candidate (Goal 1) that may also
hold the best-calibrated crown (Goal 3).

forward_features returns 2*384 = 768-d. The dataloader contract is unchanged.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .timm_encoders import build_timm_encoder, timm_feature_dim

_CONVNEXTV2_MODEL = "convnextv2_femto.fcmae_ft_in1k"


class EyesOnlyConvNeXtV2Gaze(MultistreamBackboneBase):
    """Shared ConvNeXtV2-Femto over the two eye crops only. Face / grid ignored."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "imagenet",
        freeze_encoder: bool = False,
        use_grid: bool = False,    # accepted but ignored
        grid_size: int = 25,        # accepted but ignored
    ):
        super().__init__()

        self.encoder = build_timm_encoder(_CONVNEXTV2_MODEL, weights)
        hidden_dim = timm_feature_dim(self.encoder)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        fused_dim = hidden_dim * 2                            # left + right eye
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward_features(self, face, eye_left, eye_right, grid=None):
        del face, grid
        if eye_left.shape == eye_right.shape:
            out = self.encoder(torch.cat([eye_left, eye_right], dim=0))
            eye_l_feat, eye_r_feat = out.chunk(2, dim=0)
        else:
            eye_l_feat = self.encoder(eye_left)
            eye_r_feat = self.encoder(eye_right)
        return torch.cat([eye_l_feat, eye_r_feat], dim=1)

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.head(self.forward_features(face, eye_left, eye_right, grid))
