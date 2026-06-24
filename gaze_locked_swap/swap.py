"""High-level: turn one source image into a gaze-locked, identity-swapped image.

This orchestrator stitches together: landmark detection, mask building,
inpainting (optionally reference-guided), eye copy-back, and gaze QC.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from .compose import hard_paste_protected, resize_mask_to_image
from .gaze_qc import GazeChecker
from .identity_pipeline import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_PROMPT,
    IdentitySwapPipeline,
    invert_protection_mask,
    resize_to_multiple_of_8,
)
from .landmarks import detect_landmarks
from .protection_mask import mask_for_image


@dataclass
class SwapResult:
    swapped: Image.Image
    used_landmarks: bool
    gaze_original: Optional[np.ndarray]
    gaze_swapped: Optional[np.ndarray]
    gaze_drift: Optional[float]
    flagged: bool


def swap_one(
    source_image: Image.Image,
    pipeline: IdentitySwapPipeline,
    reference_face: Optional[Image.Image] = None,
    gaze_checker: Optional[GazeChecker] = None,
    drift_threshold: float = 1.5,
    work_size: int = 512,
    feather_radius: int = 4,
    eye_dilate_px: int = 4,
    iris_padding_factor: float = 1.8,
    include_nose_bridge: bool = True,
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    strength: float = 0.92,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 35,
    seed: Optional[int] = None,
    use_landmarks: bool = True,
) -> SwapResult:
    """Swap the identity in source_image while locking gaze pixels.

    Steps:
      1. Detect landmarks on the source (or fall back to geometric mask).
      2. Build a feathered protection mask in source resolution.
      3. Resize source + SD-convention inpaint mask to a working size that is
         a multiple of 8 and the model's training resolution.
      4. Inpaint with the pipeline (IP-Adapter if reference_face + adapter loaded).
      5. Resize the inpainted result back to source resolution and hard-paste
         the source's protected pixels via the feathered mask.
      6. If a GazeChecker is provided, measure predicted-gaze drift between the
         original source and the final swap; flag if it exceeds drift_threshold.
    """
    source = source_image.convert("RGB")

    landmarks = None
    if use_landmarks:
        try:
            landmarks = detect_landmarks(source)
        except ImportError:
            landmarks = None

    _, soft_protection, used_landmarks = mask_for_image(
        source,
        landmarks,
        feather_radius=feather_radius,
        eye_dilate_px=eye_dilate_px,
        iris_padding_factor=iris_padding_factor,
        include_nose_bridge=include_nose_bridge,
    )

    work_image = resize_to_multiple_of_8(source, work_size)
    work_protection = resize_mask_to_image(soft_protection, work_image)
    inpaint_mask = invert_protection_mask(work_protection)

    work_reference = (
        reference_face.convert("RGB") if reference_face is not None else None
    )

    raw_swap = pipeline.inpaint(
        image=work_image,
        mask_image=inpaint_mask,
        reference_face=work_reference,
        prompt=prompt,
        negative_prompt=negative_prompt,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    if raw_swap.size != source.size:
        raw_swap = raw_swap.resize(source.size, Image.BICUBIC)

    final = hard_paste_protected(raw_swap, source, soft_protection)

    gaze_original = None
    gaze_swapped = None
    drift = None
    flagged = False
    if gaze_checker is not None:
        gaze_original = gaze_checker.predict_gaze(source)
        gaze_swapped = gaze_checker.predict_gaze(final)
        drift = float(np.linalg.norm(gaze_original - gaze_swapped))
        flagged = drift > drift_threshold

    return SwapResult(
        swapped=final,
        used_landmarks=used_landmarks,
        gaze_original=gaze_original,
        gaze_swapped=gaze_swapped,
        gaze_drift=drift,
        flagged=flagged,
    )
