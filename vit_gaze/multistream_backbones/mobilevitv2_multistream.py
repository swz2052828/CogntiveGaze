"""MobileViTv2 multistream backbone for gaze estimation.

Shared MobileViTv2-1.0 encoder (timm, ~4.4M params, 512-d feature) over face +
left eye + right eye + optional face-grid, late-fused to an MLP head -- same
recipe as the ViT/ConvNeXt multistream backbones, with an efficient mobile
encoder.

Motivation (Goal 1, edge): MobileViTv2 (Mehta & Rastegari, Apple, 2022,
"Separable Self-attention for Mobile Vision Transformers") replaces the O(k^2)
multi-head attention of MobileViTv1 with a *separable* (linear-complexity)
self-attention, removing v1's main on-device latency bottleneck while keeping
the conv+transformer hybrid's accuracy. It is the natural successor to our
mobile_vit backbone (and avoids v1's instability).

forward_features returns the concatenated per-stream vector; ``.head`` is a
trimmable Sequential ending in Linear(.,2), so meta / SVR calibration work.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .timm_encoders import build_timm_encoder, timm_feature_dim

_MOBILEVITV2_MODEL = "mobilevitv2_100.cvnets_in1k"


class MobileViTv2Multistream(MultistreamBackboneBase):
    """Shared MobileViTv2-1.0 over face + both eyes (+ optional grid) -> MLP head."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "imagenet",
        freeze_encoder: bool = False,
        use_grid: bool = False,
        grid_size: int = 25,
    ):
        super().__init__()

        self.encoder = build_timm_encoder(_MOBILEVITV2_MODEL, weights)
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
