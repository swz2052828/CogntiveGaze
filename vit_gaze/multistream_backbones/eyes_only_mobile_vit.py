"""Eyes-only MobileViT multistream backbone (no face crop, no face grid).

Architecture: shared MobileViT-S encoder over LEFT eye + RIGHT eye only;
features concatenated and fed to the regression head. The face crop and the
face grid arguments to forward / forward_features are accepted (for interface
compatibility with the rest of the multistream pipeline) but ignored.

Same ablation as EyesOnlyViTGaze but with lightweight MobileViT-S instead of
ViT-B/16: does per-subject calibration absorb the head-pose / distance
information when using a lighter model that might be less redundant?

forward_features returns a 2*320 = 640-d vector (vs 1536-d for eyes_only_vit
and 2304-d for the three-stream vit), which makes FiLM/LoRA meta adapters
much smaller and faster to train.

The dataloader contract is unchanged -- the multistream dataset still
produces (face, eye_left, eye_right, grid); this backbone just doesn't
read face or grid. That means switching backbones requires no dataset change.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase


class EyesOnlyMobileViTGaze(MultistreamBackboneBase):
    """Shared MobileViT-S over the two eye crops only. Face / grid are ignored."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "none",
        freeze_encoder: bool = False,
        use_grid: bool = False,    # accepted but ignored
        grid_size: int = 25,        # accepted but ignored
    ):
        super().__init__()

        # Try to use torchvision MobileViT, fall back to ViT if unavailable
        try:
            from torchvision.models import MobileViT_S_Weights, mobilevit_s
            if weights == "imagenet":
                mobilevit_weights = MobileViT_S_Weights.IMAGENET1K_V1
            elif weights == "none":
                mobilevit_weights = None
            else:
                raise ValueError("--weights must be 'none' or 'imagenet'")

            self.encoder = mobilevit_s(weights=mobilevit_weights)
            hidden_dim = 320  # MobileViT-S final feature dimension
        except (ImportError, AttributeError):
            # Fallback: use ViT-B/16 if MobileViT is unavailable
            from torchvision.models import ViT_B_16_Weights, vit_b_16
            if weights == "imagenet":
                vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
            elif weights == "none":
                vit_weights = None
            else:
                raise ValueError("--weights must be 'none' or 'imagenet'")

            self.encoder = vit_b_16(weights=vit_weights)
            hidden_dim = 768

        # Remove the original classification head
        if hasattr(self.encoder, 'heads'):
            self.encoder.heads = nn.Identity()
        elif hasattr(self.encoder, 'classifier'):
            self.encoder.classifier = nn.Identity()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        fused_dim = hidden_dim * 2                            # left + right eye
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward_features(self, face, eye_left, eye_right, grid=None):
        # face and grid are intentionally unused; accepted for interface
        # compatibility so this backbone is a drop-in for --backbone <name>
        # without changing the dataset or the calling convention.
        del face, grid

        # Single shared-encoder pass over both eyes stacked on the batch dim
        # (same trick MultiStreamViTGaze uses for its 3 streams). Falls back to
        # per-eye calls if the two eye crops ever have different shapes.
        if eye_left.shape == eye_right.shape:
            out = self.encoder(torch.cat([eye_left, eye_right], dim=0))
            eye_l_feat, eye_r_feat = out.chunk(2, dim=0)
        else:
            eye_l_feat = self.encoder(eye_left)
            eye_r_feat = self.encoder(eye_right)
        return torch.cat([eye_l_feat, eye_r_feat], dim=1)

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.head(self.forward_features(face, eye_left, eye_right, grid))
