"""MobileViT multistream backbone for gaze estimation.

Lightweight transformer-based architecture optimized for mobile/edge devices.
Uses the real MobileViT-S from ``timm`` (torchvision does not ship MobileViT),
which combines MobileNetV2-style conv blocks with local-windowed transformer
blocks. The encoder is built with ``num_classes=0`` so it returns a 640-d
globally-pooled feature vector per stream.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase

_MOBILEVIT_FEAT_DIM = 640  # timm mobilevit_s num_features


def _build_mobilevit_encoder(weights: str) -> nn.Module:
    """Build a timm MobileViT-S feature extractor (num_classes=0 -> pooled vec)."""
    import timm

    if weights == "imagenet":
        pretrained = True
    elif weights == "none":
        pretrained = False
    else:
        raise ValueError("--weights must be 'none' or 'imagenet'")
    return timm.create_model("mobilevit_s", pretrained=pretrained, num_classes=0)


class MobileViTMultistream(MultistreamBackboneBase):
    """Shared MobileViT-S backbone over face + left eye + right eye + optional grid.

    Same multistream architecture as MultiStreamViTGaze but with MobileViT-S
    (~5M params) instead of ViT-B/16 (86M params). Attention layers use
    conv-local windowing and separable convolutions for efficiency. Eye crops
    are 224x224 (same as face crop), so the same encoder applies to all three
    streams without resizing. Weight sharing reduces overfitting at small
    subject counts.
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

        self.encoder = _build_mobilevit_encoder(weights)
        hidden_dim = _MOBILEVIT_FEAT_DIM

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
            # Zero-init LayerScale gate on the grid branch (same as ViT backbone)
            self.grid_gate = nn.Parameter(torch.zeros(grid_feat_dim))

        fused_dim = hidden_dim * 3 + grid_feat_dim
        # Lightweight head for mobile (fewer parameters than ViT head)
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
        """Return the fused per-stream vector the gaze head consumes."""
        face_feat = self.encoder(face)
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
