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


class MultiStreamViTGaze(nn.Module):
    """Shared ViT-B/16 encoder over face + left eye + right eye, optional grid.

    Architecturally mirrors the iTracker-family CNNs the project's CNN baselines
    use (FaceImageModel + EyeImageModel x2 + grid MLP, fused at the head), but
    the convolutional backbones are replaced by a single shared ViT-B/16. Eye
    crops are assumed to be the same 224x224 as the face crop, so the same
    encoder applies to all three streams without resizing. ImageNet weight
    sharing is the main lever against overfitting at small subject counts.
    """

    def __init__(
        self,
        weights="none",
        freeze_encoder=False,
        use_grid=False,
        grid_size=25,
    ):
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

        self.use_grid = use_grid
        grid_feat_dim = 0
        if use_grid:
            self.grid_mlp = nn.Sequential(
                nn.Linear(grid_size * grid_size, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
            )
            grid_feat_dim = 128

        fused_dim = hidden_dim * 3 + grid_feat_dim
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, face, eye_left, eye_right, grid=None):
        face_feat = self.encoder(face)
        eye_l_feat = self.encoder(eye_left)
        eye_r_feat = self.encoder(eye_right)
        feats = [face_feat, eye_l_feat, eye_r_feat]
        if self.use_grid:
            if grid is None:
                raise ValueError("Grid input expected but not provided.")
            feats.append(self.grid_mlp(grid))
        return self.head(torch.cat(feats, dim=1))


def create_model(
    input_mode,
    weights="none",
    freeze_encoder=False,
    use_grid=False,
    grid_size=25,
):
    if input_mode == "paired":
        return PairedFaceViTGaze(weights=weights, freeze_encoder=freeze_encoder)
    if input_mode == "multistream":
        return MultiStreamViTGaze(
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
