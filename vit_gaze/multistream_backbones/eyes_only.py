"""Eyes-only ViT multistream backbone (no face crop, no face grid).

Architecture: shared ViT-B/16 encoder over LEFT eye + RIGHT eye only;
features concatenated and fed to the regression head. The face crop and the
face grid arguments to forward / forward_features are accepted (for interface
compatibility with the rest of the multistream pipeline) but ignored.

Use case: ablation of "does per-subject calibration absorb the head-pose /
distance information that the face crop and face grid otherwise provide?" -- a
question that the original iTracker / AFFNet / MGazeNet / etc. papers cannot
answer because they do not apply per-subject calibration. With our SVR or
meta-learned calibration on top, the hypothesis is:

  At large K (e.g. 64), eyes_only_vit + calibration ~= vit + calibration
                      (calibration absorbs static head-pose/distance offsets)
  At small K (e.g. 4), vit + calibration still wins
                      (face/grid provide per-frame head-pose info that K=4
                      calibration cannot infer from so few enrollment frames)

forward_features returns a 2*768 = 1536-d vector (vs 2304-d for the three-
stream vit and 768-d for foveal_vit), which makes FiLM/LoRA meta adapters
2/3 the size of the three-stream variant.

The dataloader contract is unchanged -- the multistream dataset still
produces (face, eye_left, eye_right, grid); this backbone just doesn't
read face or grid. That means switching backbones requires no dataset change.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase


class EyesOnlyViTGaze(MultistreamBackboneBase):
    """Shared ViT-B/16 over the two eye crops only. Face / grid are ignored."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "none",
        freeze_encoder: bool = False,
        use_grid: bool = False,    # accepted but ignored
        grid_size: int = 25,        # accepted but ignored
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
        hidden_dim = self.encoder.heads.head.in_features      # 768 for ViT-B/16
        self.encoder.heads = nn.Identity()

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        fused_dim = hidden_dim * 2                            # left + right eye
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),
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
