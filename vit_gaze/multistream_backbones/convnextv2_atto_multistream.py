"""Full ConvNeXtV2-Atto multistream backbone (Goal 1+2: best base-per-param).

Shared ConvNeXtV2-Atto (timm, ~3.4M, 320-d, GRN+FCMAE) over face + both eyes
(+ optional grid), late-fused to a head.

Motivation: eyes_only_convnextv2_atto became the best *calibrated* model in the
study (svr_embed@64 1.818) at only 3.6M. This adds the face + grid streams back
to test whether the same tiny winning-family encoder also delivers the best
*uncalibrated base per parameter* (Goal 2 at edge size) -- i.e. whether the
face/head-pose streams lower atto's base (4.67 eyes-only) toward femto-full's
study-best 4.33, at ~2/3 the params.
"""

from .timm_bases import TimmMultistreamBase


class ConvNeXtV2AttoMultistream(TimmMultistreamBase):
    model_name = "convnextv2_atto.fcmae_ft_in1k"
