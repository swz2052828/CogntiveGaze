"""Generic timm-encoder multistream base classes (DRY).

Wave-2 backbones differ only by which pretrained encoder they use, so they share
two parametrised bases here instead of duplicating the fusion/head boilerplate:

  TimmMultistreamBase : shared encoder over face + both eyes (+ optional grid),
                        late-fusion -> trimmable head. (full multistream)
  TimmEyesOnlyBase    : shared encoder over the two eye crops only; face/grid
                        accepted but ignored. (edge / ablation)

A concrete backbone is a 2-line subclass setting ``model_name`` (and optionally
``encoder_kwargs`` for things like DINOv2's ``img_size=224``). Existing
per-backbone files are untouched; these bases are additive.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .timm_encoders import build_timm_encoder, timm_feature_dim


def _make_head(fused_dim: int) -> nn.Sequential:
    """The standard trimmable late-fusion head used across the backbone zoo."""
    return nn.Sequential(
        nn.LayerNorm(fused_dim),
        nn.Linear(fused_dim, 256),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(256, 64),
        nn.GELU(),
        nn.Linear(64, 2),
    )


class TimmMultistreamBase(MultistreamBackboneBase):
    """Shared timm encoder over face + both eyes (+ optional grid) -> head."""

    requires_grid = False
    model_name: str = None
    encoder_kwargs: dict = {}

    def __init__(self, weights="imagenet", freeze_encoder=False,
                 use_grid=False, grid_size=25):
        super().__init__()
        self.encoder = build_timm_encoder(self.model_name, weights, **self.encoder_kwargs)
        hidden_dim = timm_feature_dim(self.encoder)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.use_grid = use_grid
        grid_feat_dim = 0
        if use_grid:
            self.grid_mlp = nn.Sequential(
                nn.Linear(grid_size * grid_size, 128), nn.GELU(),
                nn.Linear(128, 64), nn.GELU(),
            )
            grid_feat_dim = 64
            self.grid_gate = nn.Parameter(torch.zeros(grid_feat_dim))
        self.head = _make_head(hidden_dim * 3 + grid_feat_dim)

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


class TimmEyesOnlyBase(MultistreamBackboneBase):
    """Shared timm encoder over the two eye crops only. Face / grid ignored."""

    requires_grid = False
    model_name: str = None
    encoder_kwargs: dict = {}
    n_streams: int = 2  # left + right (subclasses may add interaction streams)

    def __init__(self, weights="imagenet", freeze_encoder=False,
                 use_grid=False, grid_size=25):
        super().__init__()
        self.encoder = build_timm_encoder(self.model_name, weights, **self.encoder_kwargs)
        hidden_dim = timm_feature_dim(self.encoder)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = _make_head(hidden_dim * self.n_streams)

    def _eye_feats(self, eye_left, eye_right):
        if eye_left.shape == eye_right.shape:
            out = self.encoder(torch.cat([eye_left, eye_right], dim=0))
            return out.chunk(2, dim=0)
        return self.encoder(eye_left), self.encoder(eye_right)

    def forward_features(self, face, eye_left, eye_right, grid=None):
        del face, grid
        l, r = self._eye_feats(eye_left, eye_right)
        return torch.cat([l, r], dim=1)

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.head(self.forward_features(face, eye_left, eye_right, grid))
