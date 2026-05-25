import torch
import torch.nn as nn


class PairedFaceViTGaze(nn.Module):
    def __init__(self, weights="none", freeze_encoder=False):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        if weights == "imagenet":
            vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
        elif weights == "none":
            vit_weights = None
        else:
            raise ValueError("--weights must be 'none' or 'imagenet'")

        self.encoder = vit_b_16(weights=vit_weights)
        hidden_dim = self.encoder.heads.head.in_features
        self.encoder.heads = nn.Identity()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        fused_dim = hidden_dim * 4
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, raw, synthetic):
        raw_feat = self.encoder(raw)
        synthetic_feat = self.encoder(synthetic)
        fused = torch.cat(
            [
                raw_feat,
                synthetic_feat,
                torch.abs(raw_feat - synthetic_feat),
                raw_feat * synthetic_feat,
            ],
            dim=1,
        )
        return self.head(fused)


class SingleFaceViTGaze(nn.Module):
    def __init__(self, weights="none", freeze_encoder=False):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        if weights == "imagenet":
            vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
        elif weights == "none":
            vit_weights = None
        else:
            raise ValueError("--weights must be 'none' or 'imagenet'")

        self.encoder = vit_b_16(weights=vit_weights)
        hidden_dim = self.encoder.heads.head.in_features
        self.encoder.heads = nn.Identity()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, image):
        return self.head(self.encoder(image))


# MultiStreamViTGaze moved into multistream_backbones/vit_shared.py; re-exported
# here so existing imports (e.g. from vit_gaze.models import MultiStreamViTGaze)
# continue to work.
from .multistream_backbones import (  # noqa: E402,F401
    MultiStreamViTGaze,
    build_multistream_backbone,
)


def create_model(
    input_mode,
    weights="none",
    freeze_encoder=False,
    use_grid=False,
    grid_size=25,
    backbone="vit",
):
    if input_mode == "paired":
        return PairedFaceViTGaze(weights=weights, freeze_encoder=freeze_encoder)
    if input_mode == "multistream":
        return build_multistream_backbone(
            backbone=backbone,
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
        )
    return SingleFaceViTGaze(weights=weights, freeze_encoder=freeze_encoder)


def batch_images_for_mode(batch, input_mode, device):
    raw = batch["raw"].to(device, non_blocking=True)
    synthetic = batch["synthetic"].to(device, non_blocking=True)
    if input_mode == "raw":
        return raw, None
    if input_mode == "synthetic":
        return synthetic, None
    return raw, synthetic


def forward_for_mode(model, input_mode, first, second=None):
    if input_mode == "paired":
        return model(first, second)
    return model(first)


def batch_multistream_for_mode(batch, device):
    inputs = {
        "face": batch["face"].to(device, non_blocking=True),
        "eye_left": batch["eye_left"].to(device, non_blocking=True),
        "eye_right": batch["eye_right"].to(device, non_blocking=True),
    }
    if "grid" in batch:
        inputs["grid"] = batch["grid"].to(device, non_blocking=True)
    return inputs


def forward_multistream(model, inputs):
    return model(
        inputs["face"], inputs["eye_left"], inputs["eye_right"], inputs.get("grid")
    )
