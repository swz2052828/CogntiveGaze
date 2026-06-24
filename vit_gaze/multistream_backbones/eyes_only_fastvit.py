"""Eyes-only FastViT multistream backbone (no face crop, no face grid).

Architecture: shared FastViT-T8 encoder over LEFT eye + RIGHT eye only; the two
eye feature vectors are concatenated and fed to the regression head. The face
crop and face grid are accepted (interface compatibility) but ignored.

Motivation (Goal 1, edge): FastViT (Vasu et al., Apple, ICCV 2023, "FastViT: A
Fast Hybrid Vision Transformer using Structural Reparameterization") uses the
RepMixer token mixer, which structurally reparameterises away its skip
connections at inference, giving very low latency on mobile (iPhone) hardware.
T8 is the smallest variant (~3.3M params), so eyes-only FastViT is our lightest
real-mobile-architecture candidate. forward_features returns 2*768 = 1536-d.

The dataloader contract is unchanged -- the multistream dataset still produces
(face, eye_left, eye_right, grid); this backbone just doesn't read face or grid.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .timm_encoders import build_timm_encoder, timm_feature_dim

_FASTVIT_MODEL = "fastvit_t8.apple_in1k"


class EyesOnlyFastViTGaze(MultistreamBackboneBase):
    """Shared FastViT-T8 over the two eye crops only. Face / grid ignored."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "imagenet",
        freeze_encoder: bool = False,
        use_grid: bool = False,    # accepted but ignored
        grid_size: int = 25,        # accepted but ignored
    ):
        super().__init__()

        self.encoder = build_timm_encoder(_FASTVIT_MODEL, weights)
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
