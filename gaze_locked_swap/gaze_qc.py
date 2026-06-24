"""Run the user's frozen ViT gaze checkpoint to measure swap-induced gaze drift.

This is the quality control gate: if a swap shifts the predicted gaze beyond
threshold, the swap has invalidated the label and should be retried or rejected.
"""

from typing import Optional

import numpy as np
from PIL import Image


class GazeChecker:
    """Wraps vit_gaze.training.load_checkpoint with a PIL-friendly API."""

    def __init__(self, checkpoint_path: str, device: str = "auto", image_size: int = 224):
        try:
            import torch
            from vit_gaze.dataset import IMAGENET_MEAN, IMAGENET_STD
            from vit_gaze.training import denormalize_gaze, load_checkpoint
        except ImportError as exc:
            raise ImportError(
                "vit_gaze must be importable (run from the repo root, or "
                "pip install -e .)"
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.image_size = image_size

        torch_device = torch.device(device)
        model, gaze_mean, gaze_std, _, input_mode = load_checkpoint(
            checkpoint_path, torch_device
        )
        self.model = model
        self.gaze_mean = gaze_mean
        self.gaze_std = gaze_std
        self.input_mode = input_mode
        self.torch = torch
        self.imagenet_mean = IMAGENET_MEAN.to(torch_device)
        self.imagenet_std = IMAGENET_STD.to(torch_device)
        self.denormalize_gaze = denormalize_gaze

    def _preprocess(self, image: Image.Image):
        torch = self.torch
        img = image.convert("RGB").resize(
            (self.image_size, self.image_size), Image.BICUBIC
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(self.device)
        normalized = (tensor - self.imagenet_mean) / self.imagenet_std
        return normalized.unsqueeze(0)

    @property
    def _forward(self):
        if self.input_mode == "paired":
            return lambda x: self.model(x, x)
        return self.model

    def predict_gaze(self, image: Image.Image) -> np.ndarray:
        """Return predicted (x, y) gaze in the original (de-normalized) units."""
        torch = self.torch
        with torch.no_grad():
            tensor = self._preprocess(image)
            pred_norm = self._forward(tensor)
            pred = self.denormalize_gaze(pred_norm, self.gaze_mean, self.gaze_std)
        return pred.squeeze(0).detach().cpu().numpy()

    def drift(self, original: Image.Image, swapped: Image.Image) -> float:
        """L2 distance between predicted gaze on (original, swapped).

        Low drift = the swap did not change what the gaze model sees. Use this as
        a per-frame acceptance gate.
        """
        gaze_a = self.predict_gaze(original)
        gaze_b = self.predict_gaze(swapped)
        return float(np.linalg.norm(gaze_a - gaze_b))

    def gaze_error_against(
        self, image: Image.Image, true_gaze: Optional[np.ndarray]
    ) -> Optional[float]:
        """If a ground-truth label exists, report the predicted-vs-true L2 error."""
        if true_gaze is None:
            return None
        pred = self.predict_gaze(image)
        return float(np.linalg.norm(pred - np.asarray(true_gaze, dtype=np.float32)))
