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


class RawFrameViTGaze(nn.Module):
    """Single-input gaze model over the RAW (uncropped, above-shoulder) frame.

    No face detection, no eye crops, no grid -- the timm ViT's global attention
    must learn to ignore background and localize the eyes on its own. Built for
    the facemesh+multistream vs raw-frame comparison: exposes the same
    ``forward_features`` / ``calibration_feature`` contract as the multistream
    backbones so fc_ft / svr_embed calibration reuse the cached-feature path.
    Default encoder vit_small_patch16_384 (21.8M, weights pre-cached in HF_HOME).
    """

    def __init__(self, weights="none", freeze_encoder=False, image_size=384,
                 encoder_name="vit_small_patch16_384.augreg_in21k_ft_in1k"):
        super().__init__()
        import timm

        self.image_size = image_size
        self.encoder = timm.create_model(
            encoder_name, pretrained=(weights == "imagenet"),
            num_classes=0, img_size=image_size)
        hidden_dim = self.encoder.num_features
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    # -- calibration contract (mirrors MultistreamBackboneBase) --
    def forward_features(self, image):
        return self.encoder(image)

    @property
    def readout(self):
        return self.head

    def calibration_feature(self, feats):
        """Penultimate 128-d activation of the readout, for embedding-space SVR."""
        x = feats
        for layer in list(self.head)[:-1]:
            x = layer(x)
        return x

    def forward(self, image):
        return self.head(self.encoder(image))


class RawFrameTimmGaze(RawFrameViTGaze):
    """RawFrameViTGaze with a pluggable timm encoder (raw_mobile_vit etc.)."""

    def __init__(self, weights="none", freeze_encoder=False, image_size=384,
                 encoder_name="mobilevit_s.cvnets_in1k"):
        super().__init__(weights=weights, freeze_encoder=freeze_encoder,
                         image_size=image_size, encoder_name=encoder_name)


class RawFrameFovealGaze(nn.Module):
    """Foveal raw-frame model: full frame downsized + native-res center crop
    (the face region sits top-center in the 1080x750 above-shoulder crop),
    shared timm ViT-S encoder, concat features -> readout. Same calibration
    contract as the other raw models."""

    def __init__(self, weights="none", freeze_encoder=False, image_size=384,
                 encoder_name="vit_small_patch16_384.augreg_in21k_ft_in1k"):
        super().__init__()
        import timm
        self.image_size = image_size
        self.encoder = timm.create_model(encoder_name, pretrained=(weights == "imagenet"),
                                         num_classes=0, img_size=image_size)
        hidden = self.encoder.num_features * 2
        if freeze_encoder:
            for p_ in self.encoder.parameters():
                p_.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 2))

    def _two_views(self, image):
        B, C, H, W = image.shape
        s = self.image_size
        full = nn.functional.interpolate(image, size=(s, s), mode="bilinear",
                                         align_corners=False)
        # fovea: top-center square (face region), native-ish res
        side = min(H, W) * 2 // 3
        x0 = (W - side) // 2
        crop = image[:, :, 0:side, x0:x0 + side]
        crop = nn.functional.interpolate(crop, size=(s, s), mode="bilinear",
                                         align_corners=False)
        return full, crop

    def forward_features(self, image):
        full, crop = self._two_views(image)
        return torch.cat([self.encoder(full), self.encoder(crop)], dim=1)

    @property
    def readout(self):
        return self.head

    def calibration_feature(self, feats):
        x = feats
        for layer in list(self.head)[:-1]:
            x = layer(x)
        return x

    def forward(self, image):
        return self.head(self.forward_features(image))


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
    attach_output_activation,
    build_multistream_backbone,
)


def vivit_kwargs_from_args(args):
    """Extract vivit-specific kwargs from an argparse namespace (or a dict-like).

    All values fall back to safe defaults that are ignored for non-vivit
    backbones, so this can be passed unconditionally to ``create_model``.
    """
    if hasattr(args, "get"):
        get = args.get
    else:
        def get(name, default=None):
            return getattr(args, name, default)
    return dict(
        vivit_spatial=get("vivit_spatial", "vit"),
        vivit_temporal_window=int(get("temporal_window", 8)),
        vivit_temporal_layers=int(get("vivit_temporal_layers", 4)),
        vivit_temporal_heads=int(get("vivit_temporal_heads", 8)),
        vivit_temporal_dim=int(get("vivit_temporal_dim", 512)),
    )


def create_model(
    input_mode,
    weights="none",
    freeze_encoder=False,
    use_grid=False,
    grid_size=25,
    backbone="vit",
    # optional output activation on the final (B, 2) gaze prediction
    output_activation="none",
    gaze_range=4.0,
    # vivit-specific (ignored for other backbones)
    vivit_spatial="vit",
    vivit_temporal_window=8,
    vivit_temporal_layers=4,
    vivit_temporal_heads=8,
    vivit_temporal_dim=512,
    # raw-frame single-input model (input_mode="raw" + backbone="raw_vit")
    image_size=224,
):
    if input_mode == "paired":
        model = PairedFaceViTGaze(weights=weights, freeze_encoder=freeze_encoder)
    elif input_mode == "raw" and backbone == "raw_vit":
        model = RawFrameViTGaze(weights=weights, freeze_encoder=freeze_encoder,
                                image_size=image_size)
    elif input_mode == "raw" and backbone == "raw_mobile_vit":
        model = RawFrameTimmGaze(weights=weights, freeze_encoder=freeze_encoder,
                                 image_size=image_size)
    elif input_mode == "raw" and backbone == "raw_foveal_vit":
        model = RawFrameFovealGaze(weights=weights, freeze_encoder=freeze_encoder,
                                   image_size=image_size)
    elif input_mode == "multistream":
        model = build_multistream_backbone(
            backbone=backbone,
            weights=weights,
            freeze_encoder=freeze_encoder,
            use_grid=use_grid,
            grid_size=grid_size,
            vivit_spatial=vivit_spatial,
            vivit_temporal_window=vivit_temporal_window,
            vivit_temporal_layers=vivit_temporal_layers,
            vivit_temporal_heads=vivit_temporal_heads,
            vivit_temporal_dim=vivit_temporal_dim,
        )
    else:
        model = SingleFaceViTGaze(weights=weights, freeze_encoder=freeze_encoder)
    return attach_output_activation(model, output_activation, gaze_range)


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
