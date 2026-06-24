"""CNN feature extractor + Transformer predictor multistream backbone.

This backbone inverts the usual recipe: instead of a transformer *encoder* and
an MLP *head*, it uses lightweight CNN feature extractors per stream and a
small Transformer *as the predictor* that fuses the per-stream feature tokens
via self-attention before a final Linear(.,2) readout.

Two variants are exposed (selected by ``preserve_meta_contract``):

* ``cnn_transformer``  (preserve_meta_contract=True)
  ``forward_features`` still returns the concatenated per-stream feature vector
  (N * D), exactly like every other backbone, so the FiLM/LoRA meta-adapters
  and the embedding-space SVR calibration keep working. The Transformer lives
  *inside* ``self.head`` as an ``nn.Sequential`` whose first module reshapes the
  fused vector back into N tokens, attends over them (mean-pool), and whose last
  module is the ``Linear(D, 2)`` readout. ``calibration_feature`` therefore
  trims to the post-attention D-d embedding.

* ``cnn_transformer_raw``  (preserve_meta_contract=False)
  A learnable CLS token is prepended to the stream tokens and the Transformer's
  CLS output is read out by a bare ``Linear(D, 2)``. ``forward_features`` raises
  NotImplementedError -- this variant deliberately does NOT preserve the fused-
  vector contract, so it opts out of the meta-learned / SVR-embed calibration
  path and is only usable for plain base training (and full-feature SVR).

The CNN extractor is MobileNetV3-Small (shared across the two eyes, separate for
the face); the face-grid, when enabled, is projected by a small MLP into one
extra token. All streams are projected to a common token dimension D so the
Transformer can attend over them.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase

_TOKEN_DIM = 256


class _CNNStreamExtractor(nn.Module):
    """MobileNetV3-Small conv features -> pooled D-dim token for one stream."""

    def __init__(self, token_dim: int = _TOKEN_DIM, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        weights = (
            models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        base = models.mobilenet_v3_small(weights=weights)
        self.features = base.features
        last_channel = base.classifier[0].in_features
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_channel, token_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(self.features(x))


class _TransformerFusion(nn.Module):
    """Reshape a concatenated (B, N*D) vector into N tokens, attend, mean-pool.

    Used as the penultimate module of the meta-contract-preserving head so that
    ``readout[:-1]`` yields the post-attention D-d calibration embedding.
    """

    def __init__(self, num_tokens: int, token_dim: int, num_layers: int = 2,
                 num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        # Learnable per-stream type embedding so attention can tell which token
        # is face vs. eye vs. grid (the reshape alone loses that identity).
        self.type_emb = nn.Parameter(torch.zeros(1, num_tokens, token_dim))
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads, dim_feedforward=token_dim * 2,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, fused):
        b = fused.shape[0]
        tokens = fused.view(b, self.num_tokens, self.token_dim) + self.type_emb
        tokens = self.transformer(tokens)
        return self.norm(tokens.mean(dim=1))


class CNNTransformerGaze(MultistreamBackboneBase):
    """CNN per-stream extractor + Transformer fusion predictor.

    ``preserve_meta_contract`` toggles between the two variants documented in
    the module docstring.
    """

    requires_grid = False

    def __init__(
        self,
        weights: str = "none",
        freeze_encoder: bool = False,
        use_grid: bool = False,
        grid_size: int = 25,
        preserve_meta_contract: bool = True,
        token_dim: int = _TOKEN_DIM,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.2,
    ):
        super().__init__()
        if weights not in ("none", "imagenet"):
            raise ValueError("--weights must be 'none' or 'imagenet'")
        pretrained = weights == "imagenet"

        self.preserve_meta_contract = preserve_meta_contract
        self.use_grid = use_grid
        self.token_dim = token_dim

        # Shared CNN over both eyes; separate CNN for the face.
        self.eye_extractor = _CNNStreamExtractor(token_dim, pretrained=pretrained)
        self.face_extractor = _CNNStreamExtractor(token_dim, pretrained=pretrained)
        if use_grid:
            self.grid_mlp = nn.Sequential(
                nn.Linear(grid_size * grid_size, 128),
                nn.GELU(),
                nn.Linear(128, token_dim),
                nn.GELU(),
            )

        if freeze_encoder:
            for m in (self.eye_extractor, self.face_extractor):
                for p in m.parameters():
                    p.requires_grad = False

        # face + left eye + right eye (+ grid)
        self.num_tokens = 3 + (1 if use_grid else 0)

        if preserve_meta_contract:
            # Transformer is the penultimate module of a trimmable Sequential
            # head; forward_features returns the concatenated (B, N*D) vector.
            self.head = nn.Sequential(
                _TransformerFusion(self.num_tokens, token_dim, num_layers,
                                   num_heads, dropout),
                nn.Linear(token_dim, 2),
            )
        else:
            # CLS-token Transformer fusion; forward_features opts out.
            self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.type_emb = nn.Parameter(torch.zeros(1, self.num_tokens, token_dim))
            nn.init.trunc_normal_(self.type_emb, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=token_dim, nhead=num_heads,
                dim_feedforward=token_dim * 2, dropout=dropout,
                activation="gelu", batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(token_dim)
            self.head = nn.Linear(token_dim, 2)

    def _stream_tokens(self, face, eye_left, eye_right, grid):
        """Return the per-stream feature tokens as a list of (B, D) tensors."""
        face_feat = self.face_extractor(face)
        # Shared eye extractor; stack on batch dim for a single pass when shapes
        # match (same trick the shared-ViT backbones use).
        if eye_left.shape == eye_right.shape:
            out = self.eye_extractor(torch.cat([eye_left, eye_right], dim=0))
            eye_l_feat, eye_r_feat = out.chunk(2, dim=0)
        else:
            eye_l_feat = self.eye_extractor(eye_left)
            eye_r_feat = self.eye_extractor(eye_right)
        feats = [face_feat, eye_l_feat, eye_r_feat]
        if self.use_grid:
            if grid is None:
                raise ValueError("Grid input expected but not provided.")
            feats.append(self.grid_mlp(grid))
        return feats

    def forward_features(self, face, eye_left, eye_right, grid=None):
        if not self.preserve_meta_contract:
            raise NotImplementedError(
                "cnn_transformer_raw does not expose forward_features; the "
                "CLS-token Transformer fusion does not preserve the fused-vector "
                "contract, so it cannot be used with the meta-learned / SVR-embed "
                "calibration path. Use --backbone cnn_transformer for that.")
        feats = self._stream_tokens(face, eye_left, eye_right, grid)
        # Concatenate into the (B, N*D) fused vector the head reshapes back.
        return torch.cat(feats, dim=1)

    def forward(self, face, eye_left, eye_right, grid=None):
        if self.preserve_meta_contract:
            return self.head(self.forward_features(face, eye_left, eye_right, grid))
        # Raw variant: CLS-token Transformer fusion, read CLS, Linear(D, 2).
        feats = self._stream_tokens(face, eye_left, eye_right, grid)
        b = feats[0].shape[0]
        tokens = torch.stack(feats, dim=1) + self.type_emb       # (B, N, D)
        cls = self.cls_token.expand(b, -1, -1)                   # (B, 1, D)
        seq = torch.cat([cls, tokens], dim=1)                    # (B, 1+N, D)
        seq = self.transformer(seq)
        return self.head(self.norm(seq[:, 0, :]))
