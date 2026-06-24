"""Shared helpers for timm-encoder multistream backbones.

These utilities centralise the two things every timm-based backbone needs: a
weight-string -> pretrained-bool encoder builder, and a dummy-forward feature
width detector (timm's ``num_features`` is unreliable for some conv heads, e.g.
MobileNetV4). Each concrete backbone (fastvit / mobilevitv2 / repvit /
convnextv2 / ...) lives in its own file and just supplies a model name -- the
architectures stay separate and individually selectable.
"""

import torch
import torch.nn as nn


def build_timm_encoder(model_name: str, weights: str, **kwargs) -> nn.Module:
    """Build a timm feature extractor (num_classes=0 -> globally-pooled vector).

    Extra kwargs are forwarded to ``timm.create_model`` (e.g. ``img_size=224``
    for models like DINOv2 whose default input size differs from our 224 crops).
    """
    import timm

    if weights == "imagenet":
        pretrained = True
    elif weights == "none":
        pretrained = False
    else:
        raise ValueError("--weights must be 'none' or 'imagenet'")
    return timm.create_model(
        model_name, pretrained=pretrained, num_classes=0, **kwargs)


@torch.no_grad()
def timm_feature_dim(encoder: nn.Module) -> int:
    """Detect the pooled feature width via a dummy forward (224x224)."""
    was_training = encoder.training
    encoder.eval()
    out = encoder(torch.zeros(1, 3, 224, 224))
    if was_training:
        encoder.train()
    return out.shape[1]
