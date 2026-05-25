import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_gaze.dataset import IMAGENET_MEAN, IMAGENET_STD
from vit_gaze.training import denormalize_gaze, load_checkpoint, normalize_gaze


def lsgan_discriminator_loss(real_logits, fake_logits):
    real_loss = torch.mean((real_logits - 1.0) ** 2)
    fake_loss = torch.mean(fake_logits**2)
    return 0.5 * (real_loss + fake_loss)


def lsgan_generator_loss(fake_logits):
    return 0.5 * torch.mean((fake_logits - 1.0) ** 2)


def total_variation_loss(image):
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return vertical + horizontal


def source_difference_penalty(fake, source, editable_mask, temperature=0.25):
    diff = torch.abs(fake - source) * editable_mask
    per_sample = diff.flatten(1).mean(dim=1)
    return torch.exp(-per_sample / max(temperature, 1e-6)).mean()


def to_vit_input(image_minus_one_to_one, image_size=224):
    image = (image_minus_one_to_one + 1.0) * 0.5
    if image.shape[-1] != image_size or image.shape[-2] != image_size:
        image = F.interpolate(image, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = IMAGENET_MEAN.to(image.device, image.dtype)
    std = IMAGENET_STD.to(image.device, image.dtype)
    return (image - mean) / std


class FrozenGazeCriterion(nn.Module):
    def __init__(self, checkpoint_path, device, image_size=224):
        super().__init__()
        model, gaze_mean, gaze_std, _, input_mode = load_checkpoint(checkpoint_path, device)
        self.model = model
        self.gaze_mean = gaze_mean
        self.gaze_std = gaze_std
        self.input_mode = input_mode
        self.image_size = image_size
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    def forward(self, generated_image, true_gaze):
        vit_image = to_vit_input(generated_image, self.image_size)
        target = normalize_gaze(true_gaze, self.gaze_mean, self.gaze_std)
        if self.input_mode == "paired":
            pred_norm = self.model(vit_image, vit_image)
        else:
            pred_norm = self.model(vit_image)
        loss = F.smooth_l1_loss(pred_norm, target)
        pred_gaze = denormalize_gaze(pred_norm, self.gaze_mean, self.gaze_std)
        coord_error = torch.linalg.norm(pred_gaze - true_gaze, dim=1).mean()
        return loss, coord_error
