"""ConvNeXt-Tiny multistream backbone for gaze estimation.

Shared ConvNeXt-Tiny encoder (from timm, ~28M params, 768-d feature) over
face + left eye + right eye + optional face-grid, features concatenated and fed
to an MLP head -- the same late-fusion recipe as MultiStreamViTGaze, with a
modern pure-conv encoder. ConvNeXt modernises a ResNet with depthwise 7x7
convs, LayerNorm, GELU and inverted bottlenecks, rivaling ViTs on ImageNet.

forward_features returns the concatenated per-stream vector and ``.head`` is a
trimmable Sequential ending in Linear(.,2), so the FiLM/LoRA meta-adapters and
embedding-space SVR calibration all work (same contract as the ViT backbones).
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase

_CONVNEXT_MODEL = "convnext_tiny"


def _build_convnext_encoder(weights: str) -> nn.Module:
    """Build a timm ConvNeXt-Tiny feature extractor (num_classes=0 -> pooled)."""
    import timm

    if weights == "imagenet":
        pretrained = True
    elif weights == "none":
        pretrained = False
    else:
        raise ValueError("--weights must be 'none' or 'imagenet'")
    return timm.create_model(_CONVNEXT_MODEL, pretrained=pretrained, num_classes=0)


@torch.no_grad()
def _feature_dim(encoder: nn.Module) -> int:
    """Detect the pooled feature width via a dummy forward (224x224)."""
    was_training = encoder.training
    encoder.eval()
    out = encoder(torch.zeros(1, 3, 224, 224))
    if was_training:
        encoder.train()
    return out.shape[1]


class ConvNeXtMultistream(MultistreamBackboneBase):
    """Shared ConvNeXt-Tiny over face + both eyes (+ optional grid) -> MLP head."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "none",
        freeze_encoder: bool = False,
        use_grid: bool = False,
        grid_size: int = 25,
    ):
        super().__init__()

        self.encoder = _build_convnext_encoder(weights)
        hidden_dim = _feature_dim(self.encoder)

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
            # Zero-init LayerScale gate on the grid branch (same as ViT backbone).
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
        """Return the fused per-stream vector the gaze head consumes."""
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
