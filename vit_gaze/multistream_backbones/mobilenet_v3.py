"""MobileNetV3 multistream baseline.

Drop-in replacement for the AlexNet-style iTracker backbone with the same
fusion head. Eye stream weights are shared, face stream is separate, grid goes
through a small MLP.

Faithful port of the project's MobileNetV3Model.py reference using torchvision's
mobilenet_v3_large / mobilenet_v3_small. Updated to the modern `weights=` API
so it does not emit the `pretrained=True is deprecated` warning.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase


class _MobileNetFeatureExtractor(nn.Module):
    def __init__(self, model_type: str = "large", out_dim: int = 128, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        if model_type == "small":
            weights = (
                models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            )
            base = models.mobilenet_v3_small(weights=weights)
        else:
            weights = (
                models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            )
            base = models.mobilenet_v3_large(weights=weights)
        self.features = base.features
        last_channel = base.classifier[0].in_features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_channel, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.head(self.features(x))


class _FaceGridMLP(nn.Module):
    def __init__(self, grid_size: int = 25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(grid_size * grid_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class MobileNetV3Multistream(MultistreamBackboneBase):
    requires_grid = True

    def __init__(
        self,
        weights: str = "imagenet",
        mobilenet_type: str = "large",
        grid_size: int = 25,
    ):
        super().__init__()
        pretrained = weights == "imagenet"
        self.eye_model = _MobileNetFeatureExtractor(
            model_type=mobilenet_type, out_dim=128, pretrained=pretrained
        )
        self.face_model = _MobileNetFeatureExtractor(
            model_type=mobilenet_type, out_dim=64, pretrained=pretrained
        )
        self.grid_model = _FaceGridMLP(grid_size=grid_size)
        self.eyes_fc = nn.Sequential(
            nn.Linear(2 * 128, 128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, face, eye_left, eye_right, grid=None):
        if grid is None:
            raise ValueError("MobileNetV3Multistream requires --use-grid.")
        x_eye_l = self.eye_model(eye_left)
        x_eye_r = self.eye_model(eye_right)
        x_eyes = self.eyes_fc(torch.cat([x_eye_l, x_eye_r], dim=1))
        x_face = self.face_model(face)
        x_grid = self.grid_model(grid)
        return self.fc(torch.cat([x_eyes, x_face, x_grid], dim=1))
