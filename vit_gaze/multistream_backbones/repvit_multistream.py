"""RepViT multistream backbone for gaze estimation.

Shared RepViT-M1.0 encoder (timm, ~6.4M params, 448-d feature) over face + left
eye + right eye + optional face-grid, late-fused to an MLP head.

Motivation (Goal 1, edge + accuracy): RepViT (Wang et al., CVPR 2024,
"RepViT: Revisiting Mobile CNN From ViT Perspective") redesigns a pure
MobileNet-style CNN using ViT macro/micro design choices (separate token/channel
mixers, structural reparameterisation, SE placement). It is reparameterisable to
a plain conv net at inference and reports SOTA accuracy/latency on mobile
hardware, beating MobileViT/EfficientFormer on-device -- a strong edge candidate
that is also a pure CNN (cheap, quantisation-friendly).

forward_features returns the concatenated per-stream vector; ``.head`` is a
trimmable Sequential ending in Linear(.,2), so meta / SVR calibration work.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .timm_encoders import build_timm_encoder, timm_feature_dim

_REPVIT_MODEL = "repvit_m1_0.dist_300e_in1k"


class RepViTMultistream(MultistreamBackboneBase):
    """Shared RepViT-M1.0 over face + both eyes (+ optional grid) -> MLP head."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "imagenet",
        freeze_encoder: bool = False,
        use_grid: bool = False,
        grid_size: int = 25,
    ):
        super().__init__()

        self.encoder = build_timm_encoder(_REPVIT_MODEL, weights)
        hidden_dim = timm_feature_dim(self.encoder)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.use_grid = use_grid
        grid_feat_dim = 0
        if use_grid:
            self.grid_mlp = nn.Sequential(
                nn.Linear(grid_size * grid_size, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
            )
            grid_feat_dim = 64
            self.grid_gate = nn.Parameter(torch.zeros(grid_feat_dim))

        fused_dim = hidden_dim * 3 + grid_feat_dim
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
        face_feat = self.encoder(face)
        if eye_left.shape == eye_right.shape:
            out = self.encoder(torch.cat([eye_left, eye_right], dim=0))
            eye_l_feat, eye_r_feat = out.chunk(2, dim=0)
        else:
            eye_l_feat = self.encoder(eye_left)
            eye_r_feat = self.encoder(eye_right)
        feats = [face_feat, eye_l_feat, eye_r_feat]
        if self.use_grid:
            if grid is None:
                raise ValueError("Grid input expected but not provided.")
            feats.append(self.grid_gate * self.grid_mlp(grid))
        return torch.cat(feats, dim=1)

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.head(self.forward_features(face, eye_left, eye_right, grid))
