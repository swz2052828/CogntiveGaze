"""Eyes-only ConvNeXtV2 with binocular interaction features (Goal 2/3 structural).

Shared ConvNeXtV2-Femto over the two eye crops, but the fusion adds explicit
*binocular interaction* terms: fused = [L, R, |L - R|, L * R]. Face/grid ignored.

Motivation: gaze direction is a binocular quantity -- the relationship between
the two eyes (vergence, relative iris offset) carries gaze geometry that simple
concatenation forces the head to recover. Explicit difference/product
interaction features are the second-order-pooling idea (Lin et al., Bilinear
CNN, ICCV 2015) and are already used in this codebase's PairedFaceViTGaze
([raw, syn, |raw-syn|, raw*syn]); here we apply it to the eye pair on the
winning encoder. Unlike the failed normface/grid-FiLM tricks, this adds
information (it strictly augments the concat features) rather than constraining
them, so the head can ignore it if unhelpful.

forward_features returns 4*384 = 1536-d; head stays a trimmable Linear(.,2).
"""

import torch

from .timm_bases import TimmEyesOnlyBase


class EyesOnlyConvNeXtV2BinocularGaze(TimmEyesOnlyBase):
    model_name = "convnextv2_femto.fcmae_ft_in1k"
    n_streams = 4  # L, R, |L-R|, L*R

    def forward_features(self, face, eye_left, eye_right, grid=None):
        del face, grid
        l, r = self._eye_feats(eye_left, eye_right)
        return torch.cat([l, r, torch.abs(l - r), l * r], dim=1)


class EyesOnlyConvNeXtV2AttoBinocularGaze(EyesOnlyConvNeXtV2BinocularGaze):
    """Binocular interaction on the ATTO encoder -- stacks the two confirmed-good
    findings (atto = study-best svr_embed@64 1.818 at 3.6M; binocular interaction
    = -0.2px on base and low-K fc_ft) into one ~3.6M model."""

    model_name = "convnextv2_atto.fcmae_ft_in1k"
