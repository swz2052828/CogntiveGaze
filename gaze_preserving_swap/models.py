import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class GazePreservingGenerator(nn.Module):
    """Generator conditioned on source gaze geometry and target face appearance."""

    def __init__(self, base_channels=64, residual_blocks=4):
        super().__init__()
        channels = base_channels
        self.encoder = nn.Sequential(
            ConvBlock(6, channels, stride=1, norm=False),
            ConvBlock(channels, channels * 2, stride=2),
            ConvBlock(channels * 2, channels * 4, stride=2),
            ConvBlock(channels * 4, channels * 8, stride=2),
        )
        self.residual = nn.Sequential(*[ResidualBlock(channels * 8) for _ in range(residual_blocks)])
        self.decoder = nn.Sequential(
            UpBlock(channels * 8, channels * 4),
            UpBlock(channels * 4, channels * 2),
            UpBlock(channels * 2, channels),
            nn.Conv2d(channels, 3, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, source, target):
        x = torch.cat([source, target], dim=1)
        return self.decoder(self.residual(self.encoder(x)))


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        channels = base_channels
        self.net = nn.Sequential(
            ConvBlock(3, channels, stride=2, norm=False),
            ConvBlock(channels, channels * 2, stride=2),
            ConvBlock(channels * 2, channels * 4, stride=2),
            ConvBlock(channels * 4, channels * 8, stride=1),
            nn.Conv2d(channels * 8, 1, kernel_size=4, padding=1),
        )

    def forward(self, image):
        return self.net(image)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
