"""Face Anonymization Made Simple (Kung et al., WACV 2025) as the diffusion family.

Why this method. Review risk: SimSwap (2021) and DeepPrivacy2 (2023) are not the
current state of the art, and diffusion-based de-identification is. This one is
published (WACV 2025, oral), released with weights, needs no landmarks or masks,
and handles unaligned full frames itself: SFD detection, FFHQ-style alignment to
512x512, diffusion, paste-back. Its edit domain is therefore the aligned square
(face, forehead, some hair) -- wider than SimSwap's inner face, narrower than
DeepPrivacy2's full body.

Identity regimes, as for DeepPrivacy2 (run_deepprivacy2.py):
  per_subject  the same diffusion noise for every frame of a participant
               (generator re-seeded from the participant id each frame) --
               the per-participant-consistent arm
  per_frame    fresh noise every frame -- the linkage-breaking arm
`--degree` is the method's anonymization_degree (0 = face swap, 1.25 = the
authors' anonymisation setting).

Weights: hkung/face-anon-simple (UNet + two ReferenceNets),
openai/clip-vit-large-patch14, and the SD 2.1 VAE + scheduler config from the
community mirror sd2-community/stable-diffusion-2-1 (stabilityai/stable-diffusion-2-1
is no longer publicly downloadable; only its VAE and scheduler are used).

Runs on the ORIGINAL frames and re-crops with the stored per-frame geometry, so
crops are pixel-comparable with every other arm. Frames where the detector finds
no face come out unchanged; they are counted and listed, never passed off.
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")
FAMS = Path("/springbrook/share/eng/esrpxk/third_party/face_anon_simple")


def build_pipe(device="cuda"):
    sys.path.insert(0, str(FAMS))
    import torch
    import face_alignment
    from transformers import CLIPImageProcessor, CLIPVisionModel
    from diffusers import AutoencoderKL, DDPMScheduler
    from src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
        ReferenceNetModel)
    from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
    from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
        StableDiffusionReferenceNetPipeline)

    face, clip, sd = ("hkung/face-anon-simple", "openai/clip-vit-large-patch14",
                      "sd2-community/stable-diffusion-2-1")
    pipe = StableDiffusionReferenceNetPipeline(
        unet=UNet2DConditionModel.from_pretrained(face, subfolder="unet", use_safetensors=True),
        referencenet=ReferenceNetModel.from_pretrained(face, subfolder="referencenet",
                                                       use_safetensors=True),
        conditioning_referencenet=ReferenceNetModel.from_pretrained(
            face, subfolder="conditioning_referencenet", use_safetensors=True),
        vae=AutoencoderKL.from_pretrained(sd, subfolder="vae", use_safetensors=True),
        feature_extractor=CLIPImageProcessor.from_pretrained(clip, use_safetensors=True),
        image_encoder=CLIPVisionModel.from_pretrained(clip, use_safetensors=True),
        scheduler=DDPMScheduler.from_pretrained(sd, subfolder="scheduler", use_safetensors=True),
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                      face_detector="sfd", device=device)
    from utils.extractor import extract_faces
    from utils.merger import paste_foreground_onto_background
    return pipe, fa, extract_faces, paste_foreground_onto_background, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--geometry", default=str(DATA / "anon_geometry"))
    ap.add_argument("--identity", default="per_subject", choices=["per_subject", "per_frame"])
    ap.add_argument("--degree", type=float, default=1.25)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--n-per-subject", type=int, default=260,
                    help="frames per participant (privacy sample; the attacks use 240)")
    ap.add_argument("--all-frames", action="store_true",
                    help="every frame in the geometry table (overrides --n-per-subject)")
    ap.add_argument("--every", type=int, default=1,
                    help="with --all-frames, keep every k-th frame (utility subsample)")
    ap.add_argument("--recs", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    from PIL import Image
    pipe, fa, extract_faces, paste, torch = build_pipe()
    out_root, geo_dir = Path(args.out_root), Path(args.geometry)
    rng = np.random.default_rng(0)
    recs = sorted(p.stem for p in geo_dir.glob("*.npz"))
    if args.recs:
        recs = [r for r in recs if r in set(args.recs)]
    edit = {}
    for rec in recs:
        z = np.load(geo_dir / f"{rec}.npz", allow_pickle=True)
        names = [str(v) for v in z["frames"]]
        boxes = {fld: {n: z[fld][i] for i, n in enumerate(names)}
                 for fld in ("appleFace", "appleLeftEye", "appleRightEye")}
        src_dir = DATA / "OriginalData" / rec
        avail = [n for n in names if (src_dir / n).is_file()]
        if args.all_frames:
            pick = names[::args.every]
        else:
            # same draw as run_deepprivacy2.py so the two families see the same frames
            pick = sorted(rng.choice(avail, size=args.n_per_subject, replace=False).tolist())
        for fld in boxes:
            (out_root / rec / fld).mkdir(parents=True, exist_ok=True)
        subj_seed = int(rec) * 100003 % (2**31)
        diffs, undetected, n_faces = [], [], []
        for n in pick:
            if args.resume and all((out_root / rec / f / n).is_file() for f in boxes):
                continue
            frame = cv2.imread(str(src_dir / n))
            if frame is None:
                continue
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                faces, mats = extract_faces(fa, img, 512)
            except TypeError:     # get_landmarks returns None when no face is found
                faces, mats = [], []
            n_faces.append(len(faces))
            if not faces:
                undetected.append(n)
            seed = subj_seed if args.identity == "per_subject" else \
                zlib.crc32(f"{rec}/{n}".encode()) % (2**31)
            gen = torch.Generator(device="cuda").manual_seed(seed)
            anon = img
            for fimg, mat in zip(faces, mats):
                with torch.no_grad():
                    out = pipe(source_image=fimg, conditioning_image=fimg,
                               num_inference_steps=args.steps, guidance_scale=args.guidance,
                               generator=gen, anonymization_degree=args.degree,
                               width=512, height=512).images[0]
                anon = paste(out, anon, mat)
            an = cv2.cvtColor(np.asarray(anon), cv2.COLOR_RGB2BGR)
            diffs.append(float(np.abs(an.astype(np.int16) - frame.astype(np.int16)).mean()))
            for fld, table in boxes.items():
                x, y, w, h = (int(round(v)) for v in table[n][:4])
                crop = an[max(0, y):y + h, max(0, x):x + w]
                if crop.size:
                    cv2.imwrite(str(out_root / rec / fld / n), crop)
        edit[rec] = dict(n=len(diffs), mean_abs_diff=float(np.mean(diffs)) if diffs else None,
                         n_undetected=len(undetected), undetected=undetected,
                         multi_face_frames=int(sum(k > 1 for k in n_faces)),
                         identity=args.identity, degree=args.degree, steps=args.steps)
        print(f"  {rec}: {len(diffs)} frames, |delta| {edit[rec]['mean_abs_diff']}, "
              f"undetected {len(undetected)}", flush=True)
    if args.report:
        p = Path(args.report); p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(edit, open(p, "w"), indent=2)


if __name__ == "__main__":
    main()
