"""DINOv2 multistream backbone (Goal 3 real-world low-K + Goal 2 base).

Shared DINOv2 ViT-S/14 encoder (timm, ~22M, 384-d) over face + both eyes
(+ optional grid), late-fused to a head.

Motivation: the real-world calibration regime is low-K (5/9/25 points), where the
winning method is fc_ft (a linear probe on frozen features) or meta. DINOv2
(Oquab et al., Meta, 2024, "DINOv2: Learning Robust Visual Features without
Supervision") is the strongest self-supervised ViT for *frozen-feature* /
linear-probe / k-NN transfer -- exactly the property that should help low-K
fc_ft/meta calibration. Run at img_size=224 to match our eye crops.
"""

from .timm_bases import TimmMultistreamBase


class DINOv2Multistream(TimmMultistreamBase):
    model_name = "vit_small_patch14_dinov2.lvd142m"
    encoder_kwargs = {"img_size": 224}
