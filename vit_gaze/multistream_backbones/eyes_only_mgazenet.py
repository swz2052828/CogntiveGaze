"""Eyes-only MGazeNet multistream backbone (no face crop, no face grid).

Architecture: the MGazeNet eye stream (SE blocks + LABN) applied to LEFT eye +
RIGHT eye only, then fused to the regression head. The face crop and the face
grid arguments to forward / forward_features are accepted (for interface
compatibility) but ignored.

The eyes-only reduction is NOT exact for MGazeNet the way it is for
mobilenet_v3: MGazeNet's eye stream is FiLM-modulated (LABN) by a 128-d
``factor`` that the full model derives from (face, grid). With no face/grid
here, we replace that factor with a single LEARNABLE, input-independent
parameter (``self.factor``). This keeps the eye stream's capacity and the LABN
affine modulation intact -- it degenerates to a learned constant style rather
than a per-sample conditioned one -- so the only thing removed is the head-pose
/ distance information that face+grid would have injected. That is exactly the
variable the eyes-only ablation is meant to isolate.

Same hypothesis as the other eyes_only_* backbones: does per-subject SVR / meta
calibration absorb the head-pose / distance information that face+grid provide?
forward_features returns the 128-d fused eye vector (vs 128+64+64 = 256-d for
the three-stream mgazenet).

The dataloader contract is unchanged -- the multistream dataset still produces
(face, eye_left, eye_right, grid); this backbone just doesn't read face or grid.
"""

import torch
import torch.nn as nn

from .adapter import MultistreamBackboneBase
from .mgazenet import _LABN, _SELayer, _EyeBranch

_FACTOR_DIM = 128  # face(64) + grid(64) in full MGazeNet; here a learned constant


class EyesOnlyMGazeNetGaze(MultistreamBackboneBase):
    """MGazeNet eye stream over the two eye crops only. Face / grid are ignored."""

    requires_grid = False

    def __init__(
        self,
        weights: str = "none",      # MGazeNet trains from scratch; accepted/ignored
        freeze_encoder: bool = False,
        use_grid: bool = False,    # accepted but ignored
        grid_size: int = 25,        # accepted but ignored
    ):
        super().__init__()

        # Learnable, input-independent FiLM factor replacing the face+grid
        # factor of the full MGazeNet. Non-zero init so the LABN affine is not
        # degenerate at step 0 (the LABN fc has a bias, so even zeros train).
        self.factor = nn.Parameter(torch.zeros(1, _FACTOR_DIM))

        self.eye_branch = _EyeBranch()
        self.eye_se_block_a = nn.Sequential(
            _SELayer(256, 16),
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
        )
        self.labn_layer = _LABN(_FACTOR_DIM, 64)
        self.eye_se_block_b = nn.Sequential(
            nn.ReLU(inplace=True),
            _SELayer(64, 16),
        )
        self.eye_fc = nn.Sequential(
            nn.Linear(12 * 12 * 64, 128),
            nn.LeakyReLU(inplace=True),
        )

        if freeze_encoder:
            for module in (self.eye_branch, self.eye_se_block_a,
                           self.labn_layer, self.eye_se_block_b):
                for param in module.parameters():
                    param.requires_grad = False

        self.fc = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward_features(self, face, eye_left, eye_right, grid=None):
        # face and grid are intentionally unused; accepted for interface
        # compatibility so this backbone is a drop-in for --backbone <name>
        # without changing the dataset or the calling convention.
        del face, grid
        factor = self.factor.expand(eye_left.size(0), -1)

        out_left_eye = self.eye_branch(eye_left, factor)
        out_right_eye = self.eye_branch(eye_right, factor)
        out_eyes = torch.cat([out_left_eye, out_right_eye], dim=1)
        out_eyes = self.eye_se_block_a(out_eyes)
        out_eyes = self.labn_layer(out_eyes, factor)
        out_eyes = self.eye_se_block_b(out_eyes)
        out_eyes = out_eyes.view(out_eyes.size(0), -1)
        return self.eye_fc(out_eyes)

    def forward(self, face, eye_left, eye_right, grid=None):
        return self.fc(self.forward_features(face, eye_left, eye_right, grid))
