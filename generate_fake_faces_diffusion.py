import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm


DEFAULT_PROMPT = (
    "photorealistic anonymous human face, natural skin texture, same lighting, "
    "same head pose, same expression, realistic clinical video frame"
)

DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, anime, painting, illustration, distorted eyes, moved iris, crossed eyes, "
    "closed eyes, extra eyes, deformed face, blurry, low quality, watermark, text"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate anonymized face images with diffusion inpainting while preserving "
            "gaze-critical pixels such as eyes, iris position, nose bridge, background, "
            "and face boundary."
        )
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Old cropped iTracker layout root, e.g. ./datasets/ProcessedData. "
            "Used with --input-folder and --output-folder."
        ),
    )
    parser.add_argument("--input-folder", default="appleFace")
    parser.add_argument("--output-folder", default="appleFaceFake")
    parser.add_argument(
        "--input-root",
        default=None,
        help="Uncropped source root laid out as <input-root>/<recording>/<frame>.jpg.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Uncropped output root laid out as <output-root>/<recording>/<frame>.jpg.",
    )
    parser.add_argument(
        "--model-id",
        default="stable-diffusion-v1-5/stable-diffusion-inpainting",
        help="Hugging Face inpainting model id or local model path.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--strength", type=float, default=0.25, help="Lower values preserve more structure.")
    parser.add_argument("--guidance-scale", type=float, default=5.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--work-size", type=int, default=512, help="Long side used during diffusion.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--seed-mode",
        choices=("constant", "per-frame"),
        default="constant",
        help="constant is more temporally stable; per-frame gives more visual variety.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument(
        "--attention-slicing",
        choices=("auto", "on", "off"),
        default="auto",
        help="Attention slicing trades speed for VRAM. 'auto' enables it only on "
             "low-VRAM GPUs (e.g. the 8 GB 2070 Super) and leaves it off on "
             "high-VRAM GPUs (e.g. the 5090) where it would only slow generation.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--weight-format",
        choices=("auto", "safetensors", "bin"),
        default="bin",
        help="Use bin if the cached model has no diffusion_pytorch_model.safetensors file.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--eye-protect-scale",
        type=float,
        default=2.2,
        help="Expand detected eye boxes by this factor before protecting them.",
    )
    parser.add_argument(
        "--face-edge-protect-px",
        type=int,
        default=10,
        help="Preserve this many pixels near the detected face boundary.",
    )
    parser.add_argument(
        "--mask-mode",
        choices=("face_non_eye", "lower_face"),
        default="face_non_eye",
        help="Which part of the detected face can be changed.",
    )
    parser.add_argument(
        "--debug-mask-dir",
        default=None,
        help="Optional folder where generated inpainting masks are saved for inspection.",
    )
    return parser.parse_args()


def iter_image_pairs(args):
    if args.input_root or args.output_root:
        if not args.input_root or not args.output_root:
            raise SystemExit("Use --input-root and --output-root together.")
        input_root = Path(args.input_root)
        output_root = Path(args.output_root)
        if not input_root.is_dir():
            raise SystemExit(f"Input root does not exist: {input_root}")
        for recording_dir in sorted(input_root.iterdir()):
            if not recording_dir.is_dir():
                continue
            dst_dir = output_root / recording_dir.name
            for src_path in sorted(recording_dir.glob("*.jpg")):
                yield src_path, dst_dir / src_path.name
        return

    if args.dataset_path is None:
        raise SystemExit("Provide either --dataset-path or --input-root/--output-root.")

    dataset_root = Path(args.dataset_path)
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset path does not exist or is not a directory: {dataset_root}")

    for recording_dir in sorted(dataset_root.iterdir()):
        if not recording_dir.is_dir():
            continue
        if recording_dir.name.lower().startswith("mean"):
            continue
        src_dir = recording_dir / args.input_folder
        if not src_dir.is_dir():
            continue
        dst_dir = recording_dir / args.output_folder
        for src_path in sorted(src_dir.glob("*.jpg")):
            yield src_path, dst_dir / src_path.name


def should_slice_attention(mode, torch, device):
    if mode == "on":
        return True
    if mode == "off":
        return False
    if device != "cuda" or not torch.cuda.is_available():
        return False
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        return True
    # Slice only when VRAM is tight (< 10 GiB), e.g. the 8 GB 2070 Super.
    return total_bytes < 10 * (1024 ** 3)


def load_pipeline(model_id, device, cpu_offload, weight_format, attention_slicing="auto"):
    try:
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install them with: "
            "pip install diffusers transformers accelerate safetensors pillow tqdm opencv-python"
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32
    load_kwargs = {"torch_dtype": dtype}
    if weight_format == "safetensors":
        load_kwargs["use_safetensors"] = True
    elif weight_format == "bin":
        load_kwargs["use_safetensors"] = False

    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, **load_kwargs)
    except OSError as exc:
        message = str(exc)
        should_retry_bin = (
            weight_format == "auto"
            and "diffusion_pytorch_model.safetensors" in message
            and "no file named" in message.lower()
        )
        if not should_retry_bin:
            raise

        print("Safetensors weights were not found. Retrying with PyTorch .bin weights...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=False,
        )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if cpu_offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    if should_slice_attention(attention_slicing, torch, device):
        pipe.enable_attention_slicing()
        print("Attention slicing: on (low VRAM).")
    else:
        if hasattr(pipe, "disable_attention_slicing"):
            pipe.disable_attention_slicing()
        print("Attention slicing: off (high VRAM, faster).")
    return pipe, torch, device


def load_cascade(name):
    path = Path(cv2.data.haarcascades) / name
    if not path.is_file():
        return None
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        return None
    return cascade


FACE_CASCADE = load_cascade("haarcascade_frontalface_default.xml")
EYE_CASCADE = load_cascade("haarcascade_eye.xml")


def largest_box(boxes):
    if len(boxes) == 0:
        return None
    return max(boxes, key=lambda box: int(box[2]) * int(box[3]))


def clamp_box(box, width, height):
    x, y, w, h = [int(v) for v in box]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(width, x + w)
    y1 = min(height, y + h)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def expand_box(box, scale, width, height):
    x, y, w, h = box
    cx = x + w / 2.0
    cy = y + h / 2.0
    nw = w * scale
    nh = h * scale
    return clamp_box((cx - nw / 2.0, cy - nh / 2.0, nw, nh), width, height)


def fallback_face_box(width, height):
    margin_x = int(width * 0.12)
    margin_y = int(height * 0.08)
    return margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y


def fallback_eye_boxes(face_box):
    x, y, w, h = face_box
    eye_w = int(w * 0.22)
    eye_h = int(h * 0.14)
    eye_y = y + int(h * 0.32)
    left_x = x + int(w * 0.23)
    right_x = x + int(w * 0.55)
    return [(left_x, eye_y, eye_w, eye_h), (right_x, eye_y, eye_w, eye_h)]


def detect_face_and_eyes(image):
    width, height = image.size
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    face_box = None
    if FACE_CASCADE is not None:
        min_size = (max(40, width // 8), max(40, height // 8))
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=min_size)
        face_box = largest_box(faces)
    if face_box is None:
        face_box = fallback_face_box(width, height)
    face_box = clamp_box(face_box, width, height)

    eye_boxes = []
    if EYE_CASCADE is not None:
        x, y, w, h = face_box
        upper_y1 = y + int(h * 0.68)
        roi = gray[y:upper_y1, x : x + w]
        eyes = EYE_CASCADE.detectMultiScale(
            roi,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(max(12, w // 12), max(8, h // 18)),
        )
        for ex, ey, ew, eh in eyes:
            abs_box = clamp_box((x + ex, y + ey, ew, eh), width, height)
            if abs_box[1] < y + int(h * 0.62):
                eye_boxes.append(abs_box)

    if len(eye_boxes) < 2:
        eye_boxes = fallback_eye_boxes(face_box)
    else:
        eye_boxes = sorted(eye_boxes, key=lambda box: box[2] * box[3], reverse=True)[:2]
        eye_boxes = sorted(eye_boxes, key=lambda box: box[0])

    return face_box, eye_boxes


def draw_box(draw, box, fill):
    x, y, w, h = box
    draw.rectangle((x, y, x + w, y + h), fill=fill)


def draw_ellipse(draw, box, fill):
    x, y, w, h = box
    draw.ellipse((x, y, x + w, y + h), fill=fill)


def build_inpaint_mask(image, args):
    width, height = image.size
    face_box, eye_boxes = detect_face_and_eyes(image)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    x, y, w, h = face_box
    edge = max(0, int(args.face_edge_protect_px))
    inpaint_face_box = clamp_box((x + edge, y + edge, w - 2 * edge, h - 2 * edge), width, height)

    if args.mask_mode == "lower_face":
        lx, ly, lw, lh = inpaint_face_box
        lower_box = (lx, ly + int(lh * 0.42), lw, int(lh * 0.58))
        draw_ellipse(draw, lower_box, 255)
    else:
        draw_ellipse(draw, inpaint_face_box, 255)

    protected_boxes = []
    for eye_box in eye_boxes:
        protected_boxes.append(expand_box(eye_box, args.eye_protect_scale, width, height))

    if len(protected_boxes) >= 2:
        left, right = sorted(protected_boxes[:2], key=lambda box: box[0])
        lx, ly, lw, lh = left
        rx, ry, rw, rh = right
        bridge_x0 = lx + lw
        bridge_x1 = rx
        bridge_y0 = min(ly, ry)
        bridge_y1 = max(ly + lh, ry + rh) + int(h * 0.16)
        if bridge_x1 > bridge_x0:
            protected_boxes.append(clamp_box((bridge_x0, bridge_y0, bridge_x1 - bridge_x0, bridge_y1 - bridge_y0), width, height))

    protect_draw = ImageDraw.Draw(mask)
    for box in protected_boxes:
        draw_box(protect_draw, box, 0)

    return mask, {"face_box": face_box, "eye_boxes": eye_boxes, "protected_boxes": protected_boxes}


def rounded_to_multiple_of_8(value):
    return max(64, int(round(value / 8.0)) * 8)


def resize_for_diffusion(image, mask, max_side):
    width, height = image.size
    if max_side and max(width, height) != max_side:
        scale = float(max_side) / float(max(width, height))
    else:
        scale = 1.0
    new_width = rounded_to_multiple_of_8(width * scale)
    new_height = rounded_to_multiple_of_8(height * scale)
    work_image = image.resize((new_width, new_height), Image.BICUBIC)
    work_mask = mask.resize((new_width, new_height), Image.NEAREST)
    return work_image, work_mask


def save_debug_mask(debug_dir, src_path, mask, info):
    debug_root = Path(debug_dir)
    debug_root.mkdir(parents=True, exist_ok=True)
    stem = f"{src_path.parent.name}_{src_path.stem}"
    mask.save(debug_root / f"{stem}_mask.png")
    with open(debug_root / f"{stem}_mask.txt", "w", encoding="utf-8") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")


def generate_one(pipe, torch, device, args, src_path, dst_path, index):
    original = Image.open(src_path).convert("RGB")
    mask, info = build_inpaint_mask(original, args)
    work_image, work_mask = resize_for_diffusion(original, mask, args.work_size)
    seed = args.seed if args.seed_mode == "constant" else args.seed + index
    generator_device = "cuda" if device == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)

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
    preserve_mask = ImageOps.invert(mask)
    result = Image.composite(original, result, preserve_mask)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(dst_path, quality=95)
    if args.debug_mask_dir:
        save_debug_mask(args.debug_mask_dir, src_path, mask, info)


def prepare_output_dirs(jobs, debug_mask_dir=None):
    output_dirs = {dst_path.parent for _, dst_path in jobs}
    for output_dir in sorted(output_dirs):
        output_dir.mkdir(parents=True, exist_ok=True)

    if debug_mask_dir:
        Path(debug_mask_dir).mkdir(parents=True, exist_ok=True)

    print(f"Prepared {len(output_dirs)} output directories.")


def main():
    args = parse_args()
    jobs = list(iter_image_pairs(args))
    if args.limit is not None:
        jobs = jobs[: args.limit]

    print(f"Found {len(jobs)} images.")
    if args.dry_run:
        print("DRY RUN: no directories or images will be created. Remove --dry-run to generate outputs.")
        for src, dst in jobs[:10]:
            print(f"{src} -> {dst}")
        return

    prepare_output_dirs(jobs, args.debug_mask_dir)
    pipe, torch, device = load_pipeline(
        args.model_id, args.device, args.cpu_offload, args.weight_format, args.attention_slicing
    )

    generated = 0
    skipped = 0
    for i, (src_path, dst_path) in enumerate(tqdm(jobs, desc="Inpainting fake faces", unit="image")):
        if dst_path.exists() and not args.overwrite:
            skipped += 1
            continue
        generate_one(pipe, torch, device, args, src_path, dst_path, i)
        generated += 1

    print(
        "Done. Only masked non-gaze-critical regions were synthesized; "
        "protected pixels were copied back exactly."
    )
    print(f"Generated {generated} images. Skipped {skipped} existing images.")


if __name__ == "__main__":
    main()
