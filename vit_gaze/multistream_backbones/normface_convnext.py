"""NormFace-ConvNeXt multistream backbone for gaze estimation.

Shared ConvNeXt-Tiny encoder over face + both eyes (+ optional grid), late-fused
to a head whose *penultimate* activation is an explicit L2-normalised
(hyperspherical) bottleneck. That bottleneck is what ``calibration_feature``
(``readout[:-1]``) returns, so the per-subject embedding-space SVR is fit on a
unit-norm, isotropic feature.

Motivation (Goal 3, best calibrated svr_embed): the embedding-space SVR uses an
RBF kernel, whose single global gamma assumes roughly isotropic, comparably
scaled features. cnn_transformer's svr_embed FAILED (4.88px, ~= base) precisely
because its raw fusion embedding is anisotropic/unnormalised. L2-normalising the
calibration feature onto a unit hypersphere (NormFace, Wang et al. ACM-MM 2017;
ArcFace, Deng et al. CVPR 2019) removes magnitude variation and bounds pairwise
distances to [0, 2], which is exactly the regime RBF-SVR handles best. The
encoder/head are otherwise identical to the ConvNeXt backbone, so any change in
svr_embed is attributable to the normalised bottleneck.

forward_features returns the concatenated per-stream vector; ``.head`` is a
trimmable Sequential ending in Linear(.,2) with the L2-norm as the second-to-last
module, so ``calibration_feature`` is the unit-norm embedding and meta / SVR /
fc_ft calibration all work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import MultistreamBackboneBase
from .timm_encoders import build_timm_encoder, timm_feature_dim

_CONVNEXT_MODEL = "convnext_tiny"


class _L2Normalize(nn.Module):
    """Project the feature onto the unit hypersphere (NormFace bottleneck)."""

    def __init__(self, dim: int = 1, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)


class NormFaceConvNeXtMultistream(MultistreamBackboneBase):
    """ConvNeXt-Tiny multistream with an L2-normalised calibration bottleneck."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "none",
        freeze_encoder: bool = False,
        use_grid: bool = False,
        grid_size: int = 25,
    ):
        super().__init__()

        self.encoder = build_timm_encoder(_CONVNEXT_MODEL, weights)
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
        # Head ends: ... -> Linear(.,128) -> L2Normalize -> Linear(128,2).
        # calibration_feature = readout[:-1](fused) -> the unit-norm 128-d vector.
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            _L2Normalize(dim=1),
            nn.Linear(128, 2),
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
