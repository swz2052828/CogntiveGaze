"""Foveal ViT multistream backbone.

One ViT-B/16 encoder operating on a *concatenated* token sequence of
[face patches at low resolution | left-eye patches at high resolution |
right-eye patches at high resolution], with learnable region-type embeddings
tagging each region. A single self-attention pass over the whole sequence, so
information can flow across regions in every transformer block (unlike the
shared-encoder multistream variant which runs the same encoder three times
in parallel with no cross-region attention).

Motivation (biologically inspired "foveal" resolution allocation):
* The eyes carry most of the discriminative gaze signal -> high resolution
  (224x224 -> 14x14 = 196 patch tokens per eye, default).
* The face provides head-pose context but doesn't need fine detail ->
  downsample to face_size (default 112x112 -> 7x7 = 49 patch tokens).
* Token count totals ~441 patches + 1 CLS + (1 grid) -- roughly 2.2x the
  original ViT's 197 tokens, ~1.7x the per-layer attention FLOPs of the
  stacked-batch multistream variant. Acceptable for the explanatory and
  cross-region benefits.

The architecture preserves the multistream input contract (same dataloader
contract: face + eye_left + eye_right + optional grid), so it slots into
``--backbone foveal_vit`` exactly like the other backbones. ``forward_features``
returns the [CLS] token embedding (a single 768-d vector), which is what the
meta-learned-calibration path's adapters modulate -- a *smaller* fused
dimension than the 2304-d concatenation used by the shared-encoder ViT.

Pretrained weights: we keep the ViT-B/16 patch-embed conv, transformer blocks,
and final LayerNorm. The original CLS token and 14x14 position embedding are
replaced with: (a) a fresh learnable CLS token, (b) per-region position
embeddings initialized by bilinear-interpolating the pretrained 14x14
pos_embed to each region's patch grid, (c) learnable region-type embeddings
initialized to zero so the model starts as close to the pretrained
initialization as possible.

Single attention map for explanation: the CLS token's attention over the
patch tokens spans all three regions at once. Future work can plug this into
``vit_gaze.explain`` for a unified "which patches mattered" visualization,
which was the original intuition behind using ViT at all.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import MultistreamBackboneBase


def _interpolate_pos_embed(pos_embed_flat, src_grid, target_grid):
    """Bilinear-interpolate a ``(1, src_grid*src_grid, dim)`` pos-embed to a
    ``(1, target_grid*target_grid, dim)`` grid.

    Standard ViT recipe when fine-tuning to a different input resolution.
    """
    dim = pos_embed_flat.shape[-1]
    grid = pos_embed_flat.reshape(1, src_grid, src_grid, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(target_grid, target_grid),
                         mode="bilinear", align_corners=False)
    return grid.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, dim)


class FovealViTMultistream(MultistreamBackboneBase):
    """Single-ViT foveal architecture over face + eyes + optional grid.

    The dataloader still produces 224x224 face + 224x224 eyes; the face is
    downsampled internally to ``face_size`` (default 112) before patch-embed
    so the multistream input contract is unchanged. Pass ``--backbone
    foveal_vit`` (with ``--input-mode multistream``) to use it.
    """

    requires_grid = False  # grid is optional, conditioned via a learnable token

    def __init__(
        self,
        weights: str = "none",
        freeze_encoder: bool = False,
        use_grid: bool = False,
        grid_size: int = 25,
        face_size: int = 112,
        eye_size: int = 224,
        patch_size: int = 16,
    ):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        if weights == "imagenet":
            vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
        elif weights == "none":
            vit_weights = None
        else:
            raise ValueError("--weights must be 'none' or 'imagenet'")

        backbone = vit_b_16(weights=vit_weights)
        hidden_dim = backbone.heads.head.in_features  # 768 for ViT-B/16

        if face_size % patch_size != 0 or eye_size % patch_size != 0:
            raise ValueError("face_size and eye_size must be divisible by patch_size.")

        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.face_size = face_size
        self.eye_size = eye_size
        self.face_grid = face_size // patch_size
        self.eye_grid = eye_size // patch_size

        # Reuse pretrained patch-embed, transformer blocks, and final LayerNorm.
        # We bypass the original Encoder's pos_embedding + dropout because we
        # need a region-aware position scheme.
        self.conv_proj = backbone.conv_proj
        self.encoder_blocks = backbone.encoder.layers
        self.encoder_ln = backbone.encoder.ln

        # Fresh CLS token + its position embedding.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Region-type embeddings: face=0, left_eye=1, right_eye=2, grid=3.
        # Init to zero so the pretrained initialization is the starting point.
        self.region_emb = nn.Embedding(4, hidden_dim)
        nn.init.zeros_(self.region_emb.weight)

        # Per-region position embeddings, initialized from the pretrained 14x14
        # pos_embed via bilinear interpolation when ImageNet weights are used.
        if weights == "imagenet":
            # backbone.encoder.pos_embedding: (1, 197, dim). Index 0 = CLS slot,
            # indices 1..196 = the 14x14 patch grid.
            orig_pos = backbone.encoder.pos_embedding[:, 1:, :].detach()
            face_pos = _interpolate_pos_embed(orig_pos, 14, self.face_grid)
            eye_pos = _interpolate_pos_embed(orig_pos, 14, self.eye_grid)
            cls_pos = backbone.encoder.pos_embedding[:, 0:1, :].detach().clone()
            self.face_pos_embed = nn.Parameter(face_pos)
            self.left_eye_pos_embed = nn.Parameter(eye_pos.clone())
            self.right_eye_pos_embed = nn.Parameter(eye_pos.clone())
            self.cls_pos_embed = nn.Parameter(cls_pos)
        else:
            self.face_pos_embed = nn.Parameter(
                torch.zeros(1, self.face_grid * self.face_grid, hidden_dim))
            self.left_eye_pos_embed = nn.Parameter(
                torch.zeros(1, self.eye_grid * self.eye_grid, hidden_dim))
            self.right_eye_pos_embed = nn.Parameter(
                torch.zeros(1, self.eye_grid * self.eye_grid, hidden_dim))
            self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            for p in (self.face_pos_embed, self.left_eye_pos_embed,
                      self.right_eye_pos_embed, self.cls_pos_embed):
                nn.init.trunc_normal_(p, std=0.02)

        if freeze_encoder:
            for module in (self.conv_proj, self.encoder_blocks, self.encoder_ln):
                for param in module.parameters():
                    param.requires_grad = False

        self.use_grid = use_grid
        if use_grid:
            # Project the (grid_size*grid_size) face-grid binary mask to a
            # single token in the transformer's hidden space.
            self.grid_mlp = nn.Sequential(
                nn.Linear(grid_size * grid_size, 256),
                nn.GELU(),
                nn.Linear(256, hidden_dim),
            )

        # Standard regression head over the CLS-token embedding.
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def _patchify(self, x: torch.Tensor, target_size: int) -> torch.Tensor:
        """Resize x to (target_size, target_size) if needed, then patch-embed.

        Returns ``(B, N, hidden_dim)`` where ``N = (target_size / patch_size)**2``.
        The input ``x`` is already ImageNet-normalized; bilinear resize on a
        normalized tensor is mathematically fine (resizing is linear).
        """
        if x.shape[-1] != target_size or x.shape[-2] != target_size:
            x = F.interpolate(x, size=(target_size, target_size),
                              mode="bilinear", align_corners=False)
        # conv_proj: (B, 3, H, W) -> (B, hidden_dim, H/p, W/p)
        x = self.conv_proj(x)
        # flatten spatial dims and put hidden last: (B, N, hidden_dim)
        return x.flatten(2).transpose(1, 2)

    def forward_features(self, face, eye_left, eye_right, grid=None):
        B = face.shape[0]

        # Tokenize each region at its native resolution.
        face_tokens = self._patchify(face, self.face_size)
        left_tokens = self._patchify(eye_left, self.eye_size)
        right_tokens = self._patchify(eye_right, self.eye_size)

        # Per-region position embeddings + region-type embeddings.
        face_tokens = face_tokens + self.face_pos_embed + self.region_emb.weight[0]
        left_tokens = left_tokens + self.left_eye_pos_embed + self.region_emb.weight[1]
        right_tokens = right_tokens + self.right_eye_pos_embed + self.region_emb.weight[2]

        # Prepend CLS token.
        cls = self.cls_token.expand(B, -1, -1) + self.cls_pos_embed

        seq = [cls, face_tokens, left_tokens, right_tokens]

        # Optional grid input as one extra token at the end of the sequence.
        if self.use_grid:
            if grid is None:
                raise ValueError("Grid input expected but not provided.")
            grid_token = self.grid_mlp(grid).unsqueeze(1)        # (B, 1, hidden_dim)
            grid_token = grid_token + self.region_emb.weight[3]
            seq.append(grid_token)

        x = torch.cat(seq, dim=1)

        # Run the pretrained transformer stack and final LayerNorm. We bypass
        # the original Encoder's pos_embedding + dropout (we did pos ourselves;
        # default ViT-B/16 dropout is 0 anyway so this is behavior-preserving).
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_ln(x)

        # CLS-token feature is the standard ViT readout point. (B, hidden_dim).
        return x[:, 0, :]

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.head(self.forward_features(face, eye_left, eye_right, grid))
