"""MGazeNet multistream baseline.

Uses LABN (Linear Adaptive Batch Normalisation) instead of AGN for the eye
conditioning. Same overall structure as AFFNet: face + grid -> factor; eye
streams conditioned on factor at every block. Grid is required.

Faithful port of the project's MGazeNetModel.py reference. Eye and face crops
are both 224x224; grid is the flattened 625-d vector.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase


class _SELayer(nn.Module):
    def __init__(self, channels: int, compress_rate: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channels, channels // compress_rate, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels // compress_rate, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)
        s = self.se(s).view(b, c, 1, 1)
        return x * s


class _LABN(nn.Module):
    """Linear Adaptive BatchNorm: BN with gamma/beta predicted from a factor."""

    def __init__(self, factor_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(factor_dim, channels * 2),
            nn.LeakyReLU(inplace=True),
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x, factor):
        style = self.fc(factor).view(-1, 2, x.size(1), 1, 1)
        x = self.bn(x)
        return x * (style[:, 0] + 1.0) + style[:, 1]


class _EyeBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0),
        )
        self.se_block_1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            _SELayer(48, 16),
            nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1),
        )
        self.down_sampling = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.se_block_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            _SELayer(128, 16),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

        self.labn_48 = _LABN(128, 48)
        self.labn_64_a = _LABN(128, 64)
        self.labn_128 = _LABN(128, 128)
        self.labn_64_b = _LABN(128, 64)

    def forward(self, x, factor):
        x = self.conv_block(x)
        x = self.labn_48(x, factor)
        x = self.se_block_1(x)
        x = self.labn_64_a(x, factor)
        x1 = self.down_sampling(x)

        x2 = self.conv(x1)
        x2 = self.labn_128(x2, factor)
        x2 = self.se_block_2(x2)
        x2 = self.labn_64_b(x2, factor)
        x2 = self.relu(x2)
        return torch.cat([x1, x2], dim=1)


class _FaceBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_branch = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            _SELayer(192, 16),
            nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            _SELayer(128, 16),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            _SELayer(64, 16),
        )
        self.two_fc = nn.Sequential(
            nn.Linear(5 * 5 * 64, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.main_branch(x)
        return self.two_fc(x.view(x.size(0), -1))


class MGazeNetMultistream(MultistreamBackboneBase):
    requires_grid = True

    def __init__(self, grid_size: int = 25):
        super().__init__()
        self.face_branch = _FaceBranch()
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
        self.eye_branch = _EyeBranch()
        self.eye_se_block_a = nn.Sequential(
            _SELayer(256, 16),
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
        )
        self.labn_layer = _LABN(128, 64)
        self.eye_se_block_b = nn.Sequential(
            nn.ReLU(inplace=True),
            _SELayer(64, 16),
        )
        self.eye_fc = nn.Sequential(
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
            raise ValueError("MGazeNetMultistream requires --use-grid.")
        out_face = self.face_branch(face)
        out_rect = self.rect_fc(grid)
        factor = torch.cat([out_face, out_rect], dim=1)

        out_left_eye = self.eye_branch(eye_left, factor)
        out_right_eye = self.eye_branch(eye_right, factor)
        out_eyes = torch.cat([out_left_eye, out_right_eye], dim=1)
        out_eyes = self.eye_se_block_a(out_eyes)
        out_eyes = self.labn_layer(out_eyes, factor)
        out_eyes = self.eye_se_block_b(out_eyes)
        out_eyes = out_eyes.view(out_eyes.size(0), -1)
        out_eyes = self.eye_fc(out_eyes)

        return self.fc(torch.cat([out_eyes, out_face, out_rect], dim=1))
