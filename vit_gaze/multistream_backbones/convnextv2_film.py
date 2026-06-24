"""Grid-FiLM ConvNeXtV2 multistream backbone (Goal 2: best uncalibrated base).

Shared ConvNeXtV2-Femto encoder over face + both eyes, but instead of simply
concatenating a grid MLP feature (iTracker-style late fusion), the face-grid is
used to **FiLM-modulate the eye features** (AFFNet/MGazeNet-style conditioning):
gamma/beta are predicted from the grid and applied to each eye's feature vector.

Motivation: we found that the face-grid (head-pose / camera-distance) chiefly
helps the *uncalibrated* base -- per-subject calibration later absorbs that
static offset, so at high K the grid barely matters, but at base/low-K it does.
FiLM injects that head-pose signal multiplicatively into the eye representation
(a stronger interaction than concatenation), which AFFNet/MGazeNet showed helps
gaze. Built on the strongest encoder (convnextv2) to push base below 4.33.

The FiLM heads are zero-initialised so the module starts as identity
(gamma=1, beta=0) -- training begins from the plain-concat behaviour and learns
the modulation, for stability. forward_features returns the fused 3*384 = 1152-d
vector; ``.head`` stays a trimmable Linear(.,2) so meta / SVR calibration work.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .timm_encoders import build_timm_encoder, timm_feature_dim

_CONVNEXTV2_MODEL = "convnextv2_femto.fcmae_ft_in1k"


class _GridFiLM(nn.Module):
    """Predict per-channel (gamma, beta) for the eye features from the grid."""

    def __init__(self, grid_size: int, feat_dim: int):
        super().__init__()
        self.grid_mlp = nn.Sequential(
            nn.Linear(grid_size * grid_size, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.to_gamma = nn.Linear(128, feat_dim)
        self.to_beta = nn.Linear(128, feat_dim)
        # Zero-init -> starts as identity (gamma=1, beta=0).
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, eye_feat, grid):
        h = self.grid_mlp(grid)
        gamma = 1.0 + self.to_gamma(h)
        beta = self.to_beta(h)
        return gamma * eye_feat + beta


class ConvNeXtV2FiLMMultistream(MultistreamBackboneBase):
    """ConvNeXtV2-Femto with grid-FiLM-conditioned eye features."""

    requires_grid = True

    def __init__(
        self,
        weights: str = "imagenet",
        freeze_encoder: bool = False,
        use_grid: bool = True,
        grid_size: int = 25,
    ):
        super().__init__()

        self.encoder = build_timm_encoder(_CONVNEXTV2_MODEL, weights)
        hidden_dim = timm_feature_dim(self.encoder)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.film = _GridFiLM(grid_size, hidden_dim)

        fused_dim = hidden_dim * 3                            # face + 2 FiLM'd eyes
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
        if grid is None:
            raise ValueError("ConvNeXtV2FiLMMultistream requires --use-grid.")
        face_feat = self.encoder(face)
        if eye_left.shape == eye_right.shape:
            out = self.encoder(torch.cat([eye_left, eye_right], dim=0))
            eye_l_feat, eye_r_feat = out.chunk(2, dim=0)
        else:
            eye_l_feat = self.encoder(eye_left)
            eye_r_feat = self.encoder(eye_right)
        eye_l_feat = self.film(eye_l_feat, grid)
        eye_r_feat = self.film(eye_r_feat, grid)
        return torch.cat([face_feat, eye_l_feat, eye_r_feat], dim=1)

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.head(self.forward_features(face, eye_left, eye_right, grid))
