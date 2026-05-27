"""Shared ViT-B/16 backbone over face + both eyes + optional face-grid.

Moved here from vit_gaze/models.py so all multistream backbones live in one
package. The original class is re-exported from vit_gaze.models for backward
compatibility.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase


class MultiStreamViTGaze(MultistreamBackboneBase):
    """Shared ViT-B/16 encoder over face + left eye + right eye, optional grid.

    Mirrors the iTracker-family CNNs (face + eyes + grid -> fused head) but the
    convolutional backbones are replaced by a single shared ViT-B/16. Eye crops
    are 224x224 (same as the face crop), so the same encoder applies to all
    three streams without resizing. Weight sharing is the main lever against
    overfitting at small subject counts.
    """

    requires_grid = False

    def __init__(
        self,
        weights: str = "none",
        freeze_encoder: bool = False,
        use_grid: bool = False,
        grid_size: int = 25,
    ):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        if weights == "imagenet":
            vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
        elif weights == "none":
            vit_weights = None
        else:
            raise ValueError("--weights must be 'none' or 'imagenet'")

        self.encoder = vit_b_16(weights=vit_weights)
        hidden_dim = self.encoder.heads.head.in_features
        self.encoder.heads = nn.Identity()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.use_grid = use_grid
        grid_feat_dim = 0
        if use_grid:
            self.grid_mlp = nn.Sequential(
                nn.Linear(grid_size * grid_size, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
            )
            grid_feat_dim = 128

        fused_dim = hidden_dim * 3 + grid_feat_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, face, eye_left, eye_right, grid=None):
        face_feat = self.encoder(face)
        eye_l_feat = self.encoder(eye_left)
        eye_r_feat = self.encoder(eye_right)
        feats = [face_feat, eye_l_feat, eye_r_feat]
        if self.use_grid:
            if grid is None:
                raise ValueError("Grid input expected but not provided.")
            feats.append(self.grid_mlp(grid))
        return self.head(torch.cat(feats, dim=1))
