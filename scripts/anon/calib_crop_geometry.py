"""Locate the eye regions INSIDE the stored 224x224 calibration face crop.

Needed because the calibration support sets must get the same eye-preservation
guarantee as the task data: the released face crop must not show a processed eye
region. On the task side the eye boxes came from frame coordinates, but K=9/18/36
have no original frame on disk, so the geometry is recovered entirely within the
stored crops -- the eye crop (224 px representing ~110 native px) is matched
inside the face crop (224 px representing ~310 native px), so it appears there at
roughly 224 * 110/310 ~ 80 px and the scale is searched.

Also emits the per-subject inter-ocular distance IN FACE-CROP PIXELS, which is
what operator strength must be scaled by: sigma_crop = frac * iod_crop.

The source root matters and is not obvious. `datasets/calib_processed` and
`datasets/calib_support_K<K>` hold DIFFERENT face crops of the same frames:
`calib_support` is a 1.40x tighter box (multi-scale match corr 0.992, MAE 39.4
levels between them). Only `calib_support` matches the convention of the task
data the models were trained on -- IOD/face-crop-width 0.40-0.43 against the task
data's 0.40-0.47, where `calib_processed` reads 0.29-0.31. Geometry recovered
against the wrong root silently produces a calibration support the model has
never seen the likes of; it cost the whole of the first condition (iii) run
(job 1272012, quarantined). Hence `--src-root` defaults to the support set, and
`--expect-ratio` fails loudly if the root's convention is off.

  python scripts/anon/calib_crop_geometry.py --rec 00006 --out geom/calib_00006.npz
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")


def match_in_crop(face, eye, scales, prev=None, pad=40):
    """Best (x, y, w, h, scale, corr) for `eye` inside `face`, in face coords.

    The face crop can cut the eye region: for 00012 the left eye box sits at
    x <= 0 in 99% of frames, because that subject's face box is among the
    tightest (IOD/width 0.442) and the eye crop covers a fixed multiple of eye
    width. `cv2.matchTemplate` only evaluates positions where the whole template
    fits, so a truly-negative offset is unrepresentable: the best it can do is
    pin the template at x=0, which misaligns it and drags the correlation down
    (0.795 on 00012's left eye, against 0.98-0.99 everywhere else). Read as a
    match failure that would have dropped 67 of that subject's 72 enrolment
    frames -- silently removing the subject with the tightest crop, which is
    exactly the subject the result is most sensitive to.

    So the search runs on a border-replicated pad and returns coordinates that
    may be negative or past the edge. Callers preserve the intersection with the
    crop, which is the correct answer: the rest of the eye is genuinely not in
    the released image.
    """
    fp = cv2.copyMakeBorder(face, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    order = scales if prev is None else sorted(scales, key=lambda s: abs(s - prev))
    best = (None, None, None, None, None, -1.0)
    for sc in order:
        w = int(round(eye.shape[1] * sc)); h = int(round(eye.shape[0] * sc))
        if w < 8 or h < 8 or w > fp.shape[1] or h > fp.shape[0]:
            continue
        t = cv2.resize(eye, (w, h), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(fp, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[5]:
            best = (loc[0] - pad, loc[1] - pad, w, h, sc, float(mx))
        if prev is not None and mx >= 0.97:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--frames-json", default=None,
                    help="JSON mapping rec -> list of filenames. The K levels are "
                         "NOT nested, so a uniform sample of the 1200 calibration "
                         "frames misses most of the union of the K selections; the "
                         "support builder needs geometry on exactly those frames.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--src-root", default="calib_support_K72",
                    help="Root holding the face/eye crops to recover geometry IN. "
                         "Must be the SAME root the anonymised support is built "
                         "from -- see the module docstring.")
    ap.add_argument("--expect-ratio", default="0.36,0.52",
                    help="Permitted range for median(iod_crop)/face_crop_width. "
                         "The task data sits at 0.40-0.47; calib_processed sits at "
                         "0.29-0.31 and is the wrong convention. Set to 0,1 to "
                         "disable the guard.")
    args = ap.parse_args()

    cp = DATA / args.src_root / args.rec
    if args.frames_json:
        import json as _json
        names = sorted(_json.load(open(args.frames_json)).get(args.rec, []))
        if not names:
            print(f"[{args.rec}] no frames requested"); return
    else:
        names = sorted(p.name for p in (cp / "appleFace").glob("*.jpg"))
    if args.limit and not args.frames_json:
        names = names[:: max(1, len(names) // args.limit)][: args.limit]

    coarse = np.arange(0.15, 0.85, 0.02)
    fine = np.arange(0.15, 0.851, 0.005)
    prev = {"appleLeftEye": None, "appleRightEye": None}
    rows, corrs = [], {k: [] for k in prev}
    t0 = time.time()
    for name in names:
        face = cv2.imread(str(cp / "appleFace" / name))
        if face is None:
            continue
        row = {"frame": name}
        for fold in prev:
            eye = cv2.imread(str(cp / fold / name))
            if eye is None:
                continue
            x, y, w, h, sc, c = match_in_crop(
                face, eye, fine if prev[fold] is not None else coarse, prev[fold])
            if x is None:
                continue
            if c >= 0.95:
                prev[fold] = sc
            corrs[fold].append(c)
            row[fold] = (x, y, w, h, round(sc, 4), round(c, 5))
        if len(row) == 3:
            rows.append(row)

    dt = time.time() - t0
    iod = np.array([abs((r["appleLeftEye"][0] + r["appleLeftEye"][2] / 2)
                        - (r["appleRightEye"][0] + r["appleRightEye"][2] / 2))
                    for r in rows])
    print(f"[{args.rec}] {len(rows)}/{len(names)} frames, {dt:.1f}s "
          f"({dt/max(1,len(rows))*1000:.0f} ms/frame)")
    for fold in prev:
        c = np.asarray(corrs[fold])
        if len(c):
            w = np.array([r[fold][2] for r in rows if fold in r])
            print(f"  {fold:14s} corr min {c.min():.4f} mean {c.mean():.4f} "
                  f"| <0.95: {(c < 0.95).sum()} | box {w.mean():.1f}+-{w.std():.2f} px in crop")
    if len(iod):
        print(f"  IOD in face-crop px: {iod.mean():.2f} +- {iod.std():.2f} "
              f"(min {iod.min():.0f} max {iod.max():.0f})")
        for frac in (0.10, 0.20, 0.50):
            print(f"    sigma {frac:.2f}*IOD -> {frac*iod.mean():5.1f} px in the 224 crop")
    # Convention guard. A face crop recovered from the wrong root looks perfectly
    # healthy -- high match correlation, tight box variance -- because the eye
    # crops genuinely do sit inside it. The only thing that gives it away is the
    # SCALE of the face box relative to the face, so that is what we check.
    lo, hi = (float(v) for v in args.expect_ratio.split(","))
    if len(iod):
        face_w = cv2.imread(str(cp / "appleFace" / rows[0]["frame"])).shape[1]
        ratio = float(np.median(iod)) / face_w
        print(f"  IOD / face-crop width = {ratio:.3f}  (permitted {lo}-{hi}; "
              f"task data 0.40-0.47)")
        if not (lo <= ratio <= hi):
            raise SystemExit(
                f"FATAL [{args.rec}] face-crop convention of --src-root "
                f"{args.src_root!r} is {ratio:.3f}, outside {lo}-{hi}. This root's "
                f"face box does not match the task data the models were trained "
                f"on; geometry from it would produce a calibration support that "
                f"silently degrades every result. Refusing to write.")

    if args.out and rows:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, frames=np.array([r["frame"] for r in rows]),
                            iod_crop=iod,
                            **{f: np.array([r[f] for r in rows], dtype=np.float64)
                               for f in prev})
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
