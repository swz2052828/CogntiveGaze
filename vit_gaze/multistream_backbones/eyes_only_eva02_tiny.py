"""Eyes-only EVA-02-Tiny backbone (Goal 1 edge, MIM pretraining).

Shared EVA-02-Tiny (patch14, 224, MIM IN-22k; timm, ~5.5M, 192-d) over the two
eye crops only; face/grid ignored.

Motivation: tests whether masked-image-modeling pretraining (the ingredient that
made convnextv2 win) gives a *small* ViT an edge over the existing ~3-5M
incumbents. EVA-02-Tiny is the smallest strong MIM-pretrained ViT; eyes-only it
sits in the 1.7-5M edge sweet spot identified in the study.
"""

from .timm_bases import TimmEyesOnlyBase


class EyesOnlyEVA02TinyGaze(TimmEyesOnlyBase):
    model_name = "eva02_tiny_patch14_224.mim_in22k"
