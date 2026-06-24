"""EVA-02 multistream backbone (Goal 2 best base).

Shared EVA-02-Small (patch14, 224, MIM-pretrained on IN-22k; timm, ~22M, 384-d)
over face + both eyes (+ optional grid), late-fused to a head.

Motivation: ConvNeXtV2 won largely via its masked-autoencoder (FCMAE)
pretraining, which yields stronger, less-redundant representations -> better
*uncalibrated* base. EVA-02 (Fang et al., 2023, "EVA-02: A Visual
Representation for Neon Genesis") is a ViT pretrained with masked image modeling
(MIM) distilling from a strong CLIP teacher, with SOTA transfer per parameter.
This tests whether a top MIM-pretrained ViT beats convnextv2 on base.
"""

from .timm_bases import TimmMultistreamBase


class EVA02Multistream(TimmMultistreamBase):
    model_name = "eva02_small_patch14_224.mim_in22k"
