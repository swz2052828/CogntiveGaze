"""DeepPrivacy2 (face) as a second de-identification family, for the M5 question.

Why a second method at all. Our structural claim is that de-identification whose
edit domain is inner-face texture cannot break release-only linkage, because the
channels that carry it lie outside that domain. So far it rests on one method
(SimSwap++) and two templates. A generator from a different family, with a
different objective, is the obvious cross-check: DeepPrivacy2 *removes* identity
by inpainting a detected face box with a StyleGAN generator, where SimSwap
*replaces* identity by transferring a template.

We drive the anonymiser directly rather than through `anonymize.py`, which
imports detectron2 only to read EXIF orientation. `dp2/utils/__init__.py` and
`vis_utils.py` carry a one-line local patch making the DensePose/CSE import
lazy -- the face anonymiser never reaches it. Nothing about generation changes.

We also measure the **edit domain** empirically (mean absolute difference per
pixel between input and output, summarised over the crop) so the comparison with
SimSwap is a comparison of how much of the image each method actually touches,
not of what its paper claims.

Runs on the ORIGINAL frames and re-crops with the stored per-frame geometry, so
the resulting crops are pixel-comparable with every other arm.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")
DP2 = Path("/springbrook/share/eng/esrpxk/third_party/deep_privacy2")


CONFIGS = {
    "face": "configs/anonymizers/face.py",
    # DP2's full pipeline: Mask R-CNN + CSE (DensePose) person detection, the
    # whole person repainted by the CSE-guided FDH generator. Faces lying inside
    # a detected person are DROPPED by CSeMaskFaceDetector, so on our
    # single-person frames the face is repainted by the BODY generator; the face
    # generator is only a fallback for frames with no person found.
    # person_G (styleganL_nocse, for persons CSE misses) is unobtainable -- its
    # host returns 410 Gone -- so it is set to None and such persons are left
    # UNTOUCHED. That case is counted per frame, never passed off as protected.
    "fullbody": "configs/anonymizers/FB_cse_mask_face.py",
}


def build_anonymiser(cfg_name="configs/anonymizers/face.py", drop_person_G=False):
    """The generator config path inside the anonymiser config is relative
    ("configs/fdf/stylegan.py"), so it only resolves with the repo as CWD."""
    import os

    sys.path.insert(0, str(DP2))
    import tops
    from tops.config import instantiate, LazyConfig

    cwd = os.getcwd()
    os.chdir(DP2)
    try:
        cfg = LazyConfig.load(str(DP2 / cfg_name))
        if drop_person_G:
            cfg.anonymizer.person_G_cfg = None
        anonymiser = instantiate(cfg.anonymizer)
    finally:
        os.chdir(cwd)
    anonymiser.initialize_tracker(fps=30)
    return anonymiser, tops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(DATA / "ProcessedDP2"))
    ap.add_argument("--geometry", default=str(DATA / "anon_geometry"))
    ap.add_argument("--n-per-subject", type=int, default=150,
                    help="frames per participant; the linkage attack uses 120")
    ap.add_argument("--limit-subjects", type=int, default=0)
    ap.add_argument("--all-frames", action="store_true",
                    help="every frame in the geometry table, in order: the full "
                         "root the gaze models are evaluated on (Table 3 arm). "
                         "Overrides --n-per-subject.")
    ap.add_argument("--recs", nargs="+", default=None,
                    help="only these recordings (one per array task)")
    ap.add_argument("--resume", action="store_true",
                    help="skip frames whose three crops already exist")
    ap.add_argument("--report", default=None, help="edit-domain report json")
    ap.add_argument("--truncation", type=float, default=1.0,
                    help="StyleGAN truncation. The generator computes "
                         "w = w_avg.lerp(w(z), truncation), so **truncation=0 "
                         "discards z entirely** and every face becomes the mean "
                         "face -- which is anonymize.py's default and makes "
                         "--identity a no-op. Use 1.0 whenever the identity "
                         "regime is the variable under test.")
    ap.add_argument("--identity", default="per_subject",
                    choices=["per_subject", "per_frame"],
                    help="per_subject: one fixed latent seed for all of a "
                         "participant's frames, matching SimSwap's per-participant "
                         "template -- the arm comparable on edit domain. per_frame: "
                         "a fresh identity every frame, which is the fix our "
                         "falsifiable claim predicts (break per-participant "
                         "consistency and release-only linkage should collapse).")
    ap.add_argument("--mode", default="face", choices=sorted(CONFIGS),
                    help="face: DP2's face anonymiser (the arm used so far). "
                         "fullbody: DP2's full-body pipeline (FB_cse_mask_face).")
    args = ap.parse_args()

    import torch

    anonymiser, tops = build_anonymiser(CONFIGS[args.mode],
                                        drop_person_G=args.mode == "fullbody")
    out_root = Path(args.out_root)
    geo_dir = Path(args.geometry)
    rng = np.random.default_rng(0)

    edit = {}
    recs = sorted(p.stem for p in geo_dir.glob("*.npz"))
    if args.recs:
        recs = [r for r in recs if r in set(args.recs)]
    if args.limit_subjects:
        recs = recs[: args.limit_subjects]

    for rec in recs:
        z = np.load(geo_dir / f"{rec}.npz", allow_pickle=True)
        names = [str(v) for v in z["frames"]]
        boxes = {n: z["appleFace"][i] for i, n in enumerate(names)}
        eyeL = {n: z["appleLeftEye"][i] for i, n in enumerate(names)}
        eyeR = {n: z["appleRightEye"][i] for i, n in enumerate(names)}

        src_dir = DATA / "OriginalData" / rec
        avail = [n for n in names if (src_dir / n).is_file()]
        if args.all_frames:
            if len(avail) < len(names):
                raise SystemExit(f"{rec}: {len(names) - len(avail)} geometry frames "
                                 f"have no original -- a full root would be short")
            pick = names
        else:
            if len(avail) < args.n_per_subject:
                print(f"  {rec}: only {len(avail)} original frames, skipping", flush=True)
                continue
            pick = sorted(rng.choice(avail, size=args.n_per_subject, replace=False).tolist())

        for fld in ("appleFace", "appleLeftEye", "appleRightEye"):
            (out_root / rec / fld).mkdir(parents=True, exist_ok=True)

        # Deterministic per-participant latent seed, stable across runs.
        subj_seed = int(rec) * 100003 % (2**31)
        diffs = []
        # Frames where the detector finds no face come out UNCHANGED. In a
        # utility arm those are original pixels and flatter the method, so they
        # are counted and reported rather than silently passed through.
        undetected = []
        paths = {}   # detection type -> frames in which it fired
        for n in pick:
            if args.resume and all((out_root / rec / f / n).is_file()
                                   for f in ("appleFace", "appleLeftEye", "appleRightEye")):
                continue
            frame = cv2.imread(str(src_dir / n))
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = tops.to_cuda(torch.from_numpy(rgb).permute(2, 0, 1))
            # Drive detection and synthesis directly rather than through
            # Anonymizer.forward, whose only route to a fixed latent is the motpy
            # tracker -- unusable here because the sampled frames are not
            # consecutive. z_idx is a seed (np.random.RandomState(z_idx)), so a
            # constant per participant reproduces SimSwap's per-participant
            # template exactly, and None draws a fresh identity per frame.
            with torch.no_grad():
                out = t
                n_det = 0
                for det in anonymiser.detector(out):
                    if len(det) == 0:
                        continue
                    kind = type(det).__name__
                    paths.setdefault(kind, []).append(n)
                    # Only detections with a generator are actually edited.
                    if anonymiser.generators.get(type(det)) is not None:
                        n_det += len(det)
                    z = (None if args.identity == "per_frame"
                         else np.full(len(det), subj_seed, dtype=np.int64))
                    out = anonymiser.anonymize_detections(
                        out, det, z_idx=z,
                        multi_modal_truncation=False, amp=True,
                        truncation_value=args.truncation)
            if n_det == 0:
                undetected.append(n)
            an = cv2.cvtColor(out.permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                              cv2.COLOR_RGB2BGR)

            diffs.append(float(np.abs(an.astype(np.int16) - frame.astype(np.int16)).mean()))

            for fld, table in (("appleFace", boxes), ("appleLeftEye", eyeL),
                               ("appleRightEye", eyeR)):
                x, y, w, h = (int(round(v)) for v in table[n][:4])
                crop = an[max(0, y):y + h, max(0, x):x + w]
                if crop.size == 0:
                    continue
                cv2.imwrite(str(out_root / rec / fld / n), crop)

        edit[rec] = dict(n=len(diffs), mean_abs_diff=float(np.mean(diffs)) if diffs else None,
                         n_undetected=len(undetected), undetected=undetected,
                         mode=args.mode,
                         detections={k: len(v) for k, v in paths.items()},
                         unedited_person_frames=sorted(set(paths.get("PersonDetection", []))),
                         identity=args.identity, truncation=args.truncation)
        md = edit[rec]['mean_abs_diff']
        print(f"  {rec}: {len(diffs)} frames, mean |delta| = "
              f"{md if md is None else round(md, 2)}, undetected = {len(undetected)}",
              flush=True)

    if args.report:
        p = Path(args.report); p.parent.mkdir(parents=True, exist_ok=True)
        json.dump(edit, open(p, "w"), indent=2)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
