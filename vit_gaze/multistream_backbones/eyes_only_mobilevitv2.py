"""Eyes-only MobileViTv2-0.50 multistream backbone (extreme-edge frontier).

Shared MobileViTv2-0.50 encoder (timm, ~1.1M params, 256-d feature) over LEFT
eye + RIGHT eye only; face crop and grid accepted (interface compatibility) but
ignored.

Motivation (Goal 1, extreme edge): every eyes-only backbone so far holds ~2px
after calibration regardless of encoder (fastvit 3.7M -> 2.01, mobilenet_v3
3.1M -> 1.97, mgazenet 1.7M -> 2.00). The unexplored question is how far down we
can shrink before accuracy breaks. MobileViTv2-0.50 (Apple'22, separable
linear-attention hybrid at width multiplier 0.5) is ~1.1M params; eyes-only it
is the smallest credible model we can build (~1.2M total). A sub-2M model
holding ~2.1px would be the headline extreme-edge result for on-phone gaze.

forward_features returns 2*256 = 512-d. The dataloader contract is unchanged.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .timm_encoders import build_timm_encoder, timm_feature_dim

_MOBILEVITV2_MODEL = "mobilevitv2_050.cvnets_in1k"


class EyesOnlyMobileViTv2Gaze(MultistreamBackboneBase):
    """Shared MobileViTv2-0.50 over the two eye crops only. Face / grid ignored."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "imagenet",
        freeze_encoder: bool = False,
        use_grid: bool = False,    # accepted but ignored
        grid_size: int = 25,        # accepted but ignored
    ):
        super().__init__()

        self.encoder = build_timm_encoder(_MOBILEVITV2_MODEL, weights)
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
