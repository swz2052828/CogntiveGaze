"""Stage 3: apply an anonymisation operator to the FULL frame, then re-cut the
released crops with the recovered geometry.

Protection mask
---------------
The released artefact is the three crops. To *guarantee* that the eye crops are
byte-exact -- the utility constraint every operator must satisfy (§4) -- the
protected region is exactly the union of the two recovered eye-crop boxes, not a
landmark-tight mask. A landmark mask smaller than the crop box would let the
operator modify part of the released eye crop, which is precisely what the
guarantee forbids.

Operator strength is expressed as a fraction of inter-ocular distance, measured
per frame from the recovered eye-box centres, so a given setting means the same
thing for every subject (face crops are 300-350 px and IOD is 130-180 px).

  python scripts/anon/make_anon_frames.py --rec 00006 --op blur0.10 \
      --dst-root /.../datasets/anon/blur0.10
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")
GEOM = DATA / "anon_geometry"


def apply_op(img, op, iod, exclude=None):
    """`img` is the face-box region. Strength is a fraction of IOD.

    `exclude` is an optional list of (x, y, w, h) rects in region-local
    coordinates naming pixels that will later be overwritten by the protection
    mask. Only `meanfill_skin` uses it, to keep those pixels out of the statistic
    it computes."""
    kind = op.rstrip("0123456789.")
    frac = float(op[len(kind):]) if op[len(kind):] else 0.0
    if kind == "none":
        return img
    if kind == "blur":
        s = max(0.5, frac * iod)
        # Cap the kernel at the region extent: a Gaussian wider than the image
        # contributes negligible weight beyond it (measured max difference 3
        # levels at sigma = 1.0*IOD) but costs ~40% more time.
        k = min(int(2 * round(3 * s) + 1), 2 * max(img.shape[:2]) + 1)
        k += (k + 1) % 2
        return cv2.GaussianBlur(img, (k, k), s)
    if kind == "pixel":
        b = max(2, int(round(frac * iod)))
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if kind == "blackbox":
        return np.zeros_like(img)
    if kind == "meanfill_skin":
        # As meanfill, but the mean is taken over the face box EXCLUDING the eye
        # boxes. Plain meanfill averages pixels that the protection mask then
        # overwrites, so its fill colour carries an eye-region contribution and is
        # biased toward looking subject-specific -- exactly the direction that
        # would fake a skin-tone channel. This variant removes that bias, so
        # meanfill vs meanfill_skin also measures how large the contamination is.
        m = np.ones(img.shape[:2], dtype=bool)
        for (ex, ey, ew, eh) in (exclude or []):
            x0, y0 = max(0, ex), max(0, ey)
            x1, y1 = min(img.shape[1], ex + ew), min(img.shape[0], ey + eh)
            if x1 > x0 and y1 > y0:
                m[y0:y1, x0:x1] = False
        if not m.any():
            m[:] = True
        return np.full_like(img, img[m].mean(axis=0))
    if kind == "meanfill":
        # The face box replaced by its OWN mean colour. Geometry, eye preservation
        # and uniformity are identical to blackbox; the single difference is that
        # the surround carries the subject's mean face colour. meanfill vs blackbox
        # therefore isolates the skin-tone channel exactly, which inferring it from
        # the sigma -> infinity blur limit does not.
        return np.full_like(img, img.mean(axis=(0, 1)))
    raise ValueError(f"unknown operator {op!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--op", required=True)
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--protect-eyes", type=int, default=1,
                    help="1 = masked (eye crops byte-exact, the fair comparator); "
                         "0 = unmasked (naive practice, reported once)")
    ap.add_argument("--eye-mode", choices=("exact", "flat"), default="exact",
                    help="exact = byte-exact original eye pixels (the deployed "
                         "guarantee). flat = eye boxes kept at the SAME position "
                         "and size but filled with a constant: the layout cue "
                         "survives, the texture does not. This is the control "
                         "that separates 'the eye pixels leak identity' from "
                         "'the subject-constant rectangle layout leaks identity' "
                         "(crop geometry is per-subject constant, see protocol "
                         "SS2.2, and resizing to 112x112 removes absolute size "
                         "but not relative layout).")
    ap.add_argument("--eye-fill", type=int, default=128,
                    help="grey level for --eye-mode flat")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    g = np.load(GEOM / f"{args.rec}.npz")
    frames = [f if isinstance(f, str) else f.decode() for f in g["frames"]]
    face, left, right = g["appleFace"], g["appleLeftEye"], g["appleRightEye"]
    if args.limit and args.limit < len(frames):
        # Subsample EVENLY across the recording, not the first N: the attack
        # protocol draws probes from the second temporal half, so a head-biased
        # subset would leave the probe set empty.
        step = len(frames) / float(args.limit)
        keep = sorted({int(i * step) for i in range(args.limit)})
        idx = np.asarray(keep)
        frames = [frames[i] for i in idx]
        face, left, right = face[idx], left[idx], right[idx]

    odir = DATA / "OriginalData" / args.rec
    dst = Path(args.dst_root) / args.rec
    for sub in ("appleFace", "appleLeftEye", "appleRightEye"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    n, skipped = 0, 0
    for i, name in enumerate(frames):
        frame = cv2.imread(str(odir / name))
        if frame is None:
            skipped += 1
            continue
        fx, fy, fw, fh = [int(round(v)) for v in face[i][:4]]
        lx, ly, lw, lh = [int(round(v)) for v in left[i][:4]]
        rx, ry, rw, rh = [int(round(v)) for v in right[i][:4]]
        iod = abs((lx + lw / 2.0) - (rx + rw / 2.0))

        out = frame.copy()
        region = out[fy:fy + fh, fx:fx + fw]
        if region.size:
            excl = [(lx - fx, ly - fy, lw, lh), (rx - fx, ry - fy, rw, rh)]
            out[fy:fy + fh, fx:fx + fw] = apply_op(region, args.op, iod, exclude=excl)

        if args.protect_eyes:
            if args.eye_mode == "exact":
                # restore the exact released eye boxes from the untouched original
                out[ly:ly + lh, lx:lx + lw] = frame[ly:ly + lh, lx:lx + lw]
                out[ry:ry + rh, rx:rx + rw] = frame[ry:ry + rh, rx:rx + rw]
            else:
                out[ly:ly + lh, lx:lx + lw] = args.eye_fill
                out[ry:ry + rh, rx:rx + rw] = args.eye_fill

        q = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]
        cv2.imwrite(str(dst / "appleFace" / name), out[fy:fy + fh, fx:fx + fw], q)
        cv2.imwrite(str(dst / "appleLeftEye" / name), out[ly:ly + lh, lx:lx + lw], q)
        cv2.imwrite(str(dst / "appleRightEye" / name), out[ry:ry + rh, rx:rx + rw], q)
        n += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(frames)}", flush=True)

    # The manifest subdirs are what --mean-path resolves to; without them the
    # evaluation CLI cannot find metadata.mat in an anonymised root.
    for m in ("mean7", "meanno7", "meanno7_clean", "meanno7nonorm"):
        src, link = DATA / "ProcessedData" / m, Path(args.dst_root) / m
        if src.is_dir() and not link.exists():
            try:
                link.symlink_to(src)
            except FileExistsError:
                pass

    print(f"[{args.rec}] op={args.op} protect_eyes={args.protect_eyes} "
          f"eye_mode={args.eye_mode} wrote {n} frames x3 crops, skipped {skipped}")


if __name__ == "__main__":
    main()
