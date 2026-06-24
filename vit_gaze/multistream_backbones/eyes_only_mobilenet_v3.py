"""Eyes-only MobileNetV3 multistream backbone (no face crop, no face grid).

Architecture: shared MobileNetV3 feature extractor over LEFT eye + RIGHT eye
only; the two eye embeddings are concatenated and fed to the regression head.
The face crop and the face grid arguments to forward / forward_features are
accepted (for interface compatibility with the rest of the multistream
pipeline) but ignored.

This is the eyes-only ablation of the ``mobilenet_v3`` baseline: that backbone
already processes each eye with an *unconditioned* MobileNetV3 stream (unlike
AFFNet / MGazeNet, whose eye streams are FiLM-modulated by a face+grid factor),
so the eyes-only reduction is exact -- we keep the shared eye stream verbatim
and simply drop the separate face stream and the grid MLP.

Same hypothesis as EyesOnlyViTGaze / EyesOnlyMobileViTGaze: does per-subject
SVR / meta calibration absorb the head-pose / distance information that the
face crop and face grid otherwise provide? forward_features returns a
2*128 = 256-d vector (vs 128+64+128 = 320-d for the three-stream
mobilenet_v3).

The dataloader contract is unchanged -- the multistream dataset still produces
(face, eye_left, eye_right, grid); this backbone just doesn't read face or grid.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .mobilenet_v3 import _MobileNetFeatureExtractor


class EyesOnlyMobileNetV3Gaze(MultistreamBackboneBase):
    """Shared MobileNetV3 over the two eye crops only. Face / grid are ignored."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "imagenet",
        freeze_encoder: bool = False,
        mobilenet_type: str = "large",
        use_grid: bool = False,    # accepted but ignored
        grid_size: int = 25,        # accepted but ignored
    ):
        super().__init__()
        pretrained = weights == "imagenet"
        eye_dim = 128
        self.eye_model = _MobileNetFeatureExtractor(
            model_type=mobilenet_type, out_dim=eye_dim, pretrained=pretrained
        )

        if freeze_encoder:
            for param in self.eye_model.features.parameters():
                param.requires_grad = False

        fused_dim = eye_dim * 2                               # left + right eye
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward_features(self, face, eye_left, eye_right, grid=None):
        # face and grid are intentionally unused; accepted for interface
        # compatibility so this backbone is a drop-in for --backbone <name>
        # without changing the dataset or the calling convention.
        del face, grid
        x_eye_l = self.eye_model(eye_left)
        x_eye_r = self.eye_model(eye_right)
        return torch.cat([x_eye_l, x_eye_r], dim=1)

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.head(self.forward_features(face, eye_left, eye_right, grid))
