"""Face-only MobileViT multistream backbone (no eye crops, no face grid).

Architecture: a single MobileViT-S encoder over the FACE crop only; its pooled
feature vector is fed to the regression head. The two eye crops and the face
grid arguments to forward / forward_features are accepted (for interface
compatibility with the rest of the multistream pipeline) but ignored.

This is the face-only counterpart to EyesOnlyMobileViTGaze: where the eyes-only
ablation asks "does calibration absorb the head-pose/distance info that
face+grid carry?", the face-only ablation asks the converse -- "how far can the
face crop alone go?". The face crop contains coarse head-pose and gaze-direction
cues but not the high-resolution iris detail the eye crops provide, so the
expectation is that face_only underperforms eyes_only / full multistream,
especially before calibration.

forward_features returns a single 640-d vector (vs 2*640 = 1280-d for
eyes_only_mobile_vit and 3*640 (+64) for the three-stream mobile_vit).

The dataloader contract is unchanged -- the multistream dataset still produces
(face, eye_left, eye_right, grid); this backbone just doesn't read the eyes or
the grid. That means switching backbones requires no dataset change.
"""

import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .mobile_vit import _MOBILEVIT_FEAT_DIM, _build_mobilevit_encoder


class FaceOnlyMobileViTGaze(MultistreamBackboneBase):
    """MobileViT-S over the face crop only. Eyes / grid are ignored."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "none",
        freeze_encoder: bool = False,
        use_grid: bool = False,    # accepted but ignored
        grid_size: int = 25,        # accepted but ignored
    ):
        super().__init__()

        self.encoder = _build_mobilevit_encoder(weights)
        hidden_dim = _MOBILEVIT_FEAT_DIM

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward_features(self, face, eye_left, eye_right, grid=None):
        # eyes and grid are intentionally unused; accepted for interface
        # compatibility so this backbone is a drop-in for --backbone <name>
        # without changing the dataset or the calling convention.
        del eye_left, eye_right, grid
        return self.encoder(face)

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.head(self.forward_features(face, eye_left, eye_right, grid))
