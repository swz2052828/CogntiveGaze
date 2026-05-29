"""Original iTracker CNN ported into the multistream-backbone interface.

Source: Krafka et al., "Eye Tracking for Everyone" (CVPR 2016) /
gazecapture.csail.mit.edu. The architecture is faithfully reproduced from
the project's ITrackerModel.py reference; here it just wraps in the
(face, eye_left, eye_right, grid) interface used by all multistream backbones.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase


class _ItrackerImageModel(nn.Module):
    """Shared AlexNet-ish stack used for both eyes and the face."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.CrossMapLRN2d(size=5, alpha=0.0001, beta=0.75, k=1.0),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


class _FaceImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = _ItrackerImageModel()
        self.fc = nn.Sequential(
            nn.Linear(12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class _FaceGridModel(nn.Module):
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


class ITrackerMultistream(MultistreamBackboneBase):
    requires_grid = True

    def __init__(self, grid_size: int = 25):
        super().__init__()
        self.eye_model = _ItrackerImageModel()
        self.face_model = _FaceImageModel()
        self.grid_model = _FaceGridModel(grid_size=grid_size)
        self.eyes_fc = nn.Sequential(
            nn.Linear(2 * 12 * 12 * 64, 128),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward_features(self, face, eye_left, eye_right, grid=None):
        if grid is None:
            raise ValueError("ITrackerMultistream requires --use-grid.")
        x_eye_l = self.eye_model(eye_left)
        x_eye_r = self.eye_model(eye_right)
        x_eyes = self.eyes_fc(torch.cat([x_eye_l, x_eye_r], dim=1))
        x_face = self.face_model(face)
        x_grid = self.grid_model(grid)
        return torch.cat([x_eyes, x_face, x_grid], dim=1)

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.fc(self.forward_features(face, eye_left, eye_right, grid))
