"""Eyes-only MobileNetV4 multistream backbone (no face crop, no face grid).

Architecture: shared MobileNetV4-Conv-Medium encoder over LEFT eye + RIGHT eye
only; the two eye feature vectors are concatenated and fed to the regression
head. The face crop and the face grid arguments to forward / forward_features
are accepted (for interface compatibility with the rest of the multistream
pipeline) but ignored.

Eyes-only reduction of the ``mobilenet_v4`` baseline: that backbone runs the
same shared encoder over face + both eyes (+ optional grid); here we keep the
shared eye encoder verbatim and drop the face stream and the grid branch.

Same hypothesis as the other eyes_only_* backbones: does per-subject SVR / meta
calibration absorb the head-pose / distance information that the face crop and
face grid otherwise provide? forward_features returns a 2*1280 = 2560-d vector
(vs 3*1280 (+64) for the three-stream mobilenet_v4).

Note: timm's ``num_features`` (960) differs from the actual pooled output width
(1280, after conv_head expansion), so the feature dim is detected with a dummy
forward pass -- same as the parent mobilenet_v4 backbone.

The dataloader contract is unchanged -- the multistream dataset still produces
(face, eye_left, eye_right, grid); this backbone just doesn't read face or grid.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .mobilenet_v4 import _build_mobilenetv4_encoder, _feature_dim


class EyesOnlyMobileNetV4Gaze(MultistreamBackboneBase):
    """Shared MobileNetV4-Conv-Medium over the two eye crops only. Face / grid ignored."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "none",
        freeze_encoder: bool = False,
        use_grid: bool = False,    # accepted but ignored
        grid_size: int = 25,        # accepted but ignored
    ):
        super().__init__()

        self.encoder = _build_mobilenetv4_encoder(weights)
        hidden_dim = _feature_dim(self.encoder)

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
        # face and grid are intentionally unused; accepted for interface
        # compatibility so this backbone is a drop-in for --backbone <name>
        # without changing the dataset or the calling convention.
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
