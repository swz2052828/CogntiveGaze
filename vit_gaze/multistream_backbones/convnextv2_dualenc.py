"""ConvNeXtV2 dual-encoder multistream backbone (iTracker-style stream split).

Architecture: a dedicated ConvNeXtV2-Femto encoder for the FACE crop + a second,
weight-shared ConvNeXtV2-Femto encoder for the two EYE crops (+ optional
face-grid MLP), late-fused to an MLP head.

Contrast with ``convnextv2`` (ConvNeXtV2Multistream), which shares ONE encoder
across face AND both eyes. Here the face gets its own tower while the two eyes
still share a tower (the classic iTracker split: shared eye tower + separate
face tower). This is the ablation point: does a dedicated face representation
buy anything beyond the per-subject head-pose prior that per-subject calibration
already absorbs? It costs ~2x the encoder params (two femto encoders, ~9.7M)
versus the fully-shared variant (~4.9M).

forward_features returns the concatenated per-stream vector; ``.head`` is a
trimmable Sequential ending in Linear(.,2), so meta / SVR calibration work.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .timm_encoders import build_timm_encoder, timm_feature_dim

_CONVNEXTV2_MODEL = "convnextv2_femto.fcmae_ft_in1k"


class ConvNeXtV2DualEncMultistream(MultistreamBackboneBase):
    """Separate face encoder + weight-shared eye encoder (+ optional grid) -> MLP head."""

    requires_grid = False
    model_name = _CONVNEXTV2_MODEL  # subclasses override to scale the encoders

    def __init__(
        self,
        weights: str = "imagenet",
        freeze_encoder: bool = False,
        use_grid: bool = False,
        grid_size: int = 25,
    ):
        super().__init__()

        # Two independent encoders: one dedicated to the face, one shared by the
        # two eye crops (eyes are the same image domain, so they weight-tie).
        self.face_encoder = build_timm_encoder(self.model_name, weights)
        self.eye_encoder = build_timm_encoder(self.model_name, weights)
        hidden_dim = timm_feature_dim(self.face_encoder)

        if freeze_encoder:
            for param in self.face_encoder.parameters():
                param.requires_grad = False
            for param in self.eye_encoder.parameters():
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
        face_feat = self.face_encoder(face)
        if eye_left.shape == eye_right.shape:
            out = self.eye_encoder(torch.cat([eye_left, eye_right], dim=0))
            eye_l_feat, eye_r_feat = out.chunk(2, dim=0)
        else:
            eye_l_feat = self.eye_encoder(eye_left)
            eye_r_feat = self.eye_encoder(eye_right)
        feats = [face_feat, eye_l_feat, eye_r_feat]
        if self.use_grid:
            if grid is None:
                raise ValueError("Grid input expected but not provided.")
            feats.append(self.grid_gate * self.grid_mlp(grid))
        return torch.cat(feats, dim=1)

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.head(self.forward_features(face, eye_left, eye_right, grid))
