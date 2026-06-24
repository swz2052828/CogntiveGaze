"""Stable Diffusion inpainting with optional IP-Adapter identity conditioning.

The default Stable Diffusion inpainting model conditioned only on a generic
prompt tends to harmonize back toward the original face. IP-Adapter Face takes
an image of a target identity and biases the latent toward that face, which
produces a much stronger identity change while still respecting the surrounding
unmasked context (pose, lighting, hair, background).

If IP-Adapter weights are unavailable, the pipeline still runs in prompt-only
mode, just with higher strength and a stronger negative prompt.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image


DEFAULT_PROMPT = (
    "photorealistic photograph of a different adult person, natural skin texture, "
    "same head pose, same camera angle, same lighting, sharp focus"
)

DEFAULT_NEGATIVE_PROMPT = (
    "same person, identical face, original identity, cartoon, anime, illustration, "
    "deformed eyes, moved iris, crossed eyes, closed eyes, extra eyes, blurry, "
    "low quality, watermark, text"
)


@dataclass
class IPAdapterSpec:
    """Where to load IP-Adapter from. The defaults work with diffusers' loader."""

    repo: str = "h94/IP-Adapter"
    subfolder: str = "models"
    weight_name: str = "ip-adapter-plus-face_sd15.bin"
    scale: float = 0.85


class IdentitySwapPipeline:
    """Thin wrapper around StableDiffusionInpaintPipeline with IP-Adapter support."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-inpainting",
        device: str = "auto",
        torch_dtype: Optional[str] = None,
        ip_adapter: Optional[IPAdapterSpec] = None,
        use_cpu_offload: bool = False,
    ):
        try:
            import torch
            from diffusers import (
                DPMSolverMultistepScheduler,
                StableDiffusionInpaintPipeline,
            )
        except ImportError as exc:
            raise ImportError(
                "diffusers and torch are required. Install with: "
                "pip install torch diffusers transformers accelerate safetensors"
            ) from exc

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if torch_dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            dtype = getattr(torch, torch_dtype)

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        if use_cpu_offload and device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        pipe.enable_attention_slicing()

        self.pipe = pipe
        self.torch = torch
        self.ip_adapter = None
        if ip_adapter is not None:
            self._load_ip_adapter(ip_adapter)

    def _load_ip_adapter(self, spec: IPAdapterSpec):
        self.pipe.load_ip_adapter(
            spec.repo, subfolder=spec.subfolder, weight_name=spec.weight_name
        )
        self.pipe.set_ip_adapter_scale(spec.scale)
        self.ip_adapter = spec

    def inpaint(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        reference_face: Optional[Image.Image] = None,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        strength: float = 0.92,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 35,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Run inpainting; mask_image white = generate, black = keep original.

        mask_image is the SD-convention mask (the INVERSE of the protection mask).
        Callers should hand in image and mask at the same resolution.
        """
        generator = None
        if seed is not None:
            gen_device = "cuda" if self.device == "cuda" else "cpu"
            generator = self.torch.Generator(device=gen_device).manual_seed(int(seed))

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "mask_image": mask_image,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }
        if self.ip_adapter is not None:
            if reference_face is None:
                raise ValueError(
                    "ip_adapter is enabled but no reference_face was passed."
                )
            kwargs["ip_adapter_image"] = reference_face

        return self.pipe(**kwargs).images[0]


def invert_protection_mask(protection_soft_mask: np.ndarray) -> Image.Image:
    """Convert protection-soft-mask (1=keep) into SD inpaint mask (1=generate)."""
    inverted = np.clip(1.0 - protection_soft_mask, 0.0, 1.0)
    return Image.fromarray((inverted * 255).astype(np.uint8))


def resize_to_multiple_of_8(image: Image.Image, max_side: int) -> Image.Image:
    """Stable Diffusion needs sides divisible by 8; bicubic resample to fit."""
    width, height = image.size
    scale = float(max_side) / float(max(width, height)) if max_side else 1.0
    new_width = max(64, int(round(width * scale / 8.0)) * 8)
    new_height = max(64, int(round(height * scale / 8.0)) * 8)
    if (new_width, new_height) == (width, height):
        return image
    return image.resize((new_width, new_height), Image.BICUBIC)
