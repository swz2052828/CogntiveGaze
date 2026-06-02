"""ViViT (Video ViT) multistream backbone -- factorized encoder (ViViT Model 2).

Spatial-then-temporal video understanding:
* Per frame, a spatial backbone (any existing MultistreamBackboneBase with
  forward_features) produces a D-dim feature.
* Across the T frames of a temporal window, a small temporal transformer
  attends over the T features (with learnable temporal positional embeddings).
* The LAST frame's temporal-fused output is the read-out, because we predict
  gaze for the most recent frame given T frames of context (matches deployment).

Compared to the four ViViT factorizations in Arnab et al. (ICCV 2021), this is
"Model 2" (Factorized Encoder). It is by far the cheapest -- spatial attention
is unchanged from the underlying backbone, and the temporal transformer only
attends over T tokens. Model 1 (full space-time) would scale as (T*N_spatial)^2
which is ~30x more attention for T=8 with our 442-token foveal backbone.

The dataloader contract grows a time dimension: face / eye_left / eye_right /
grid all gain shape (B, T, ...) instead of (B, ...). The MultiStreamVideoDataset
wrapper in vit_gaze.dataset constructs those windows from the existing
per-frame MultiStreamGazeDataset (one window per (recording, end_frame) pair,
labels for the last frame). The wrapper is applied automatically by the
training entry points when --backbone vivit is selected.

forward_features returns a D-dim vector (D = spatial backbone's feature dim),
so FiLM/LoRA meta adapters and the SVR baselines all compose with vivit
unchanged -- they operate on the temporal-fused feature exactly like they
operate on any other backbone's fused feature.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase


class ViViTMultistream(MultistreamBackboneBase):
    """Factorized spatial+temporal video encoder over a multistream input.

    Inputs are 5D: (B, T, C, H, W) for face/eyes and (B, T, grid_size**2) for
    grid. The spatial backbone is run on (B*T, C, H, W) (frames flattened into
    the batch dim) so its existing 4D path works unchanged; outputs are then
    reshaped to (B, T, D) for the temporal transformer.

    requires_grid is delegated to the spatial backbone.
    """

    def __init__(
        self,
        spatial_backbone: MultistreamBackboneBase,
        feature_dim: int,
        temporal_window: int = 8,
        num_temporal_layers: int = 4,
        num_temporal_heads: int = 8,
        temporal_mlp_ratio: float = 4.0,
        temporal_dropout: float = 0.1,
    ):
        super().__init__()
        if temporal_window < 2:
            raise ValueError("temporal_window must be >= 2 for the temporal transformer.")

        self.spatial = spatial_backbone
        self.T = temporal_window
        self.feature_dim = feature_dim
        # Inherit grid requirement from the spatial backbone (no metadata change).
        self.requires_grid = bool(getattr(spatial_backbone, "requires_grid", False))

        # Learnable temporal positional embedding (T positions, D dim).
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, temporal_window, feature_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # Temporal transformer (encoder-only; pre-LN; no causal mask).
        layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_temporal_heads,
            dim_feedforward=int(feature_dim * temporal_mlp_ratio),
            dropout=temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_temporal_layers)
        self.temporal_ln = nn.LayerNorm(feature_dim)

        # LayerScale-style residual gate (CaiT, Touvron et al. 2021), init 0.
        # At init the temporal branch contributes nothing and the read-out is the
        # *clean* last-frame spatial feature -- i.e. the model starts exactly at
        # the (strong, pretrained) per-frame solution instead of feeding the head
        # the output of a randomly-initialised transformer. Without this, the
        # random temporal block scrambles the read-out token on step 0, the head
        # collapses to predicting the dataset-mean gaze point, and training never
        # escapes that floor (observed: train loss flat ~0.405, val error flat
        # ~9.4 cm, best epoch = 1). The gate learns to open as temporal context
        # earns its keep; if it never does, we recover the per-frame model.
        self.temporal_gate = nn.Parameter(torch.zeros(feature_dim))

        # Standard regression head over the last-frame temporal-fused token.
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    @staticmethod
    def _flatten_time(x):
        """Collapse the time dim into the batch dim. (B, T, ...) -> (B*T, ...)."""
        return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

    def forward_features(self, face, eye_left, eye_right, grid=None):
        if face.dim() != 5:
            raise ValueError(
                f"vivit expects 5D input (B, T, C, H, W); got face shape {tuple(face.shape)}. "
                f"Did the dataset wrap with MultiStreamVideoDataset?")
        B, T = face.shape[0], face.shape[1]
        if T != self.T:
            raise ValueError(
                f"Input temporal window T={T} != temporal_window={self.T} set at init. "
                f"Pass --temporal-window {T} or rebuild the model.")

        # 1) Spatial encoding: flatten time into batch, run spatial backbone
        # in one shot, reshape back to (B, T, D). Lets us reuse the existing
        # 4D path of every multistream backbone unchanged.
        face_f = self._flatten_time(face)
        eye_l_f = self._flatten_time(eye_left)
        eye_r_f = self._flatten_time(eye_right)
        grid_f = self._flatten_time(grid) if grid is not None else None

        feats_flat = self.spatial.forward_features(face_f, eye_l_f, eye_r_f, grid_f)  # (B*T, D)
        feats = feats_flat.reshape(B, T, self.feature_dim)

        # 2) Temporal encoding, added back to the clean spatial read-out through
        # a zero-initialised LayerScale gate (see __init__). The identity path is
        # the last frame's raw spatial feature; the temporal branch only
        # perturbs it once the gate has learned to open.
        spatial_readout = feats[:, -1, :]                                        # (B, D)
        temporal = self.temporal_encoder(feats + self.temporal_pos_embed)
        temporal = self.temporal_ln(temporal)[:, -1, :]                          # (B, D)

        # 3) Gated residual read-out.
        return spatial_readout + self.temporal_gate * temporal                   # (B, D)

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.head(self.forward_features(face, eye_left, eye_right, grid))


def build_vivit(
    spatial_backbone_name: str,
    weights: str,
    freeze_encoder: bool,
    use_grid: bool,
    grid_size: int,
    temporal_window: int,
    num_temporal_layers: int,
    num_temporal_heads: int,
):
    """Factory: build a ViViTMultistream around an existing spatial backbone.

    ``spatial_backbone_name`` must be one of the names the multistream factory
    accepts. We dispatch via the same factory and then probe the resulting
    backbone's feature dim by running a dummy forward.
    """
    from .adapter import build_multistream_backbone

    if spatial_backbone_name == "vivit":
        raise ValueError("vivit cannot use itself as a spatial backbone.")
    spatial = build_multistream_backbone(
        backbone=spatial_backbone_name,
        weights=weights,
        freeze_encoder=freeze_encoder,
        use_grid=use_grid,
        grid_size=grid_size,
    )

    # Probe feature dim by running a single dummy frame through forward_features.
    # Done on CPU to avoid hardware assumptions at construction time.
    feature_dim = _probe_feature_dim(spatial, use_grid=use_grid, grid_size=grid_size)

    return ViViTMultistream(
        spatial_backbone=spatial,
        feature_dim=feature_dim,
        temporal_window=temporal_window,
        num_temporal_layers=num_temporal_layers,
        num_temporal_heads=num_temporal_heads,
    )


@torch.no_grad()
def _probe_feature_dim(spatial, use_grid, grid_size, image_size=224):
    """Run a single dummy frame through ``spatial.forward_features`` and return
    the last dim of the output. Avoids hardcoding backbone-specific knowledge."""
    spatial.eval()
    face = torch.zeros(1, 3, image_size, image_size)
    eye_l = torch.zeros(1, 3, image_size, image_size)
    eye_r = torch.zeros(1, 3, image_size, image_size)
    grid = torch.zeros(1, grid_size * grid_size) if use_grid else None
    out = spatial.forward_features(face, eye_l, eye_r, grid)
    if out.dim() != 2:
        raise RuntimeError(
            f"Spatial backbone's forward_features should return (B, D); got {tuple(out.shape)}.")
    return out.shape[-1]
