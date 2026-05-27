"""AFFNet (Adaptive Group Normalisation) multistream baseline.

Eye stream is conditioned on (face features, grid features) via AGN at every
block, which lets the eye branch specialise per face geometry. Grid input is
required - it goes into the conditioning factor.

Faithful port of the project's AFFNetModel.py reference (GazeAGNModel). Eye
input is 224x224, face input is 224x224, grid is the flattened 625-d vector.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase


class _AGN(nn.Module):
    """Adaptive Group Normalisation: GroupNorm with gamma/beta predicted from a factor."""

    def __init__(self, factor_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(factor_dim, channels * 2),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x, groups, factor):
        style = self.fc(factor)
        style = style.view(-1, 2, x.size(1), 1, 1)
        n, c, h, w = x.shape
        x = x.view(n * groups, -1)
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True)
        x = (x - mean) / (var + 1e-8).sqrt()
        x = x.view(n, c, h, w)
        return x * (style[:, 0] + 1.0) + style[:, 1]


class _SELayer(nn.Module):
    def __init__(self, channels: int, compress_rate: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channels, channels // compress_rate, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // compress_rate, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feature):
        b, c, _, _ = feature.size()
        s = self.gap(feature).view(b, c)
        s = self.se(s).view(b, c, 1, 1)
        return feature * s


class _EyeImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features1_1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.GroupNorm(3, 24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0),
        )
        self.features1_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _SELayer(48, 16),
            nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1),
        )
        self.features1_3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.features2_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            _SELayer(128, 16),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        )
        self.features2_3 = nn.ReLU(inplace=True)

        self.agn1_1 = _AGN(128, 48)
        self.agn1_2 = _AGN(128, 64)
        self.agn2_1 = _AGN(128, 128)
        self.agn2_2 = _AGN(128, 64)

    def forward(self, x, factor):
        x = self.features1_1(x)
        x = self.agn1_1(x, 6, factor)
        x = self.features1_2(x)
        x = self.agn1_2(x, 8, factor)
        x = self.features1_3(x)

        x1 = x
        x = self.features2_1(x1)
        x = self.agn2_1(x, 16, factor)
        x = self.features2_2(x)
        x = self.agn2_2(x, 8, factor)
        x2 = self.features2_3(x)
        return torch.cat([x1, x2], dim=1)


class _FaceImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=0),
            nn.GroupNorm(6, 48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=0),
            nn.GroupNorm(12, 96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 192),
            nn.ReLU(inplace=True),
            _SELayer(192, 16),
            nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            _SELayer(128, 16),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            _SELayer(64, 16),
        )
        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 64, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class AFFNetMultistream(MultistreamBackboneBase):
    requires_grid = True

    def __init__(self, grid_size: int = 25):
        super().__init__()
        self.face_model = _FaceImageModel()
        self.rect_fc = nn.Sequential(
            nn.Linear(grid_size * grid_size, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 96),
            nn.LeakyReLU(inplace=True),
            nn.Linear(96, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )
        self.eye_model = _EyeImageModel()
        self.eyes_merge_1 = nn.Sequential(
            _SELayer(256, 16),
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
        )
        self.eyes_merge_agn = _AGN(128, 64)
        self.eyes_merge_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            _SELayer(64, 16),
        )
        self.eyes_fc = nn.Sequential(
            nn.Linear(12 * 12 * 64, 128),
            nn.LeakyReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 + 64 + 64, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, face, eye_left, eye_right, grid=None):
        if grid is None:
            raise ValueError("AFFNetMultistream requires --use-grid.")
        x_face = self.face_model(face)
        x_rect = self.rect_fc(grid)
        factor = torch.cat([x_face, x_rect], dim=1)

        x_eye_l = self.eye_model(eye_left, factor)
        x_eye_r = self.eye_model(eye_right, factor)
        x_eyes = torch.cat([x_eye_l, x_eye_r], dim=1)
        x_eyes = self.eyes_merge_1(x_eyes)
        x_eyes = self.eyes_merge_agn(x_eyes, 8, factor)
        x_eyes = self.eyes_merge_2(x_eyes)
        x_eyes = x_eyes.view(x_eyes.size(0), -1)
        x_eyes = self.eyes_fc(x_eyes)

        feature = torch.cat([x_eyes, x_face, x_rect], dim=1)
        return self.fc(feature)
