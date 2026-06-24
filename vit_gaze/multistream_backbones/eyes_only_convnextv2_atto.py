"""Eyes-only ConvNeXtV2-Atto backbone (Goal 1 edge, winning family).

Shared ConvNeXtV2-Atto (timm, ~3.4M, 320-d, GRN+FCMAE) over the two eye crops
only; face/grid ignored.

Motivation: convnextv2-femto was the overall winner; this takes the *same*
GRN+FCMAE family at its smallest size (~3.4M) to the edge frontier. Where the
generic mobilevitv2-0.50 (1.26M) cracked to 2.33px, the question is whether the
winning family's pretraining holds accuracy at comparable size to the ~1.7-3.1M
incumbents (mgazenet 2.00, mobilenet_v3 1.97).
"""

from .timm_bases import TimmEyesOnlyBase


class EyesOnlyConvNeXtV2AttoGaze(TimmEyesOnlyBase):
    model_name = "convnextv2_atto.fcmae_ft_in1k"
