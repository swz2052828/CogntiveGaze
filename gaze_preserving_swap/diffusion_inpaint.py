from pathlib import Path

from PIL import Image, ImageOps

from .masks import make_pil_inpaint_mask


DEFAULT_PROMPT = (
    "photorealistic different person, realistic human face, natural skin texture, "
    "same head pose, same camera, same lighting, clinical video frame"
)

DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, distorted eyes, moved iris, crossed eyes, closed eyes, "
    "extra eyes, deformed face, blurry, low quality, watermark, text"
)


def iter_images(source_root, output_root):
    source_root = Path(source_root)
    output_root = Path(output_root)
    for recording_dir in sorted(source_root.iterdir()):
        if not recording_dir.is_dir():
            continue
        for source_path in sorted(recording_dir.glob("*.jpg")):
            yield source_path, output_root / recording_dir.name / source_path.name


def load_pipeline(args, device):
    try:
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with: "
            "pip install torch diffusers transformers accelerate pillow safetensors"
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        use_safetensors=args.weight_format == "safetensors",
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if args.cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe, torch, device


def resize_to_multiple_of_8(image, max_side):
    width, height = image.size
    scale = float(max_side) / float(max(width, height)) if max_side else 1.0
    new_width = max(64, int(round(width * scale / 8.0)) * 8)
    new_height = max(64, int(round(height * scale / 8.0)) * 8)
    return image.resize((new_width, new_height), Image.BICUBIC)


def run(args):
    jobs = list(iter_images(args.source_root, args.output_root))
    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(f"Found {len(jobs)} images.")
    if args.dry_run:
        for src, dst in jobs[:10]:
            print(f"{src} -> {dst}")
        return

    pipe, torch, device = load_pipeline(args, args.device)
    generated = 0
    skipped = 0
    for index, (source_path, dst_path) in enumerate(jobs):
        if dst_path.exists() and not args.overwrite:
            skipped += 1
            continue

        original = Image.open(source_path).convert("RGB")
        inpaint_mask = make_pil_inpaint_mask(*original.size)
        work_image = resize_to_multiple_of_8(original, args.work_size)
        work_mask = resize_to_multiple_of_8(inpaint_mask, args.work_size)
        generator = torch.Generator(device="cuda" if device == "cuda" else "cpu").manual_seed(args.seed + index)

        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=work_image,
            mask_image=work_mask,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]

        result = result.resize(original.size, Image.BICUBIC)
        protected = ImageOps.invert(inpaint_mask)
        result = Image.composite(original, result, protected)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(dst_path, quality=95)
        generated += 1
        print(f"{source_path} -> {dst_path}")

    print(f"Done. Generated {generated} images. Skipped {skipped} existing images.")
