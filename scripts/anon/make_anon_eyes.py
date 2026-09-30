"""Periocular preservation axis (protocol SS4.1): P1 and P2.

The face-region sweep established a floor it cannot cross -- with the eye crop
preserved byte-exact, leakage stays at d' ~5 / TAR@1e-3 ~99% / ARI 1.000 no
matter what is done to the face. The eye region is therefore the only remaining
lever, and this is the axis that moves it.

  P0  whole eye crop preserved            (already measured: the masked sweep)
  P1  eye OPENING preserved               (this file) -- contour polygon kept,
                                          periocular skin and brow removed
  P2  iris DISC preserved                 (this file) -- only the iris circle
                                          kept, eyelids and sclera removed
  P4  nothing preserved                   (already measured: the eye-flat arm)

P3 (re-render a synthetic iris at the measured centre/radius) is deliberately
NOT built: measured iris radius is 11.6-16.1 px, i.e. 23-32 px diameter, where
iris texture is unresolved -- iris recognition needs roughly an order of
magnitude more. That is a measurement, not a shortcut; it is reported rather
than assumed.

Non-preserved pixels are set to a constant grey, the same convention as the
eye-flat control, so the axis measures what the PRESERVED region leaks rather
than confounding it with whatever was used to fill the rest.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")
GEOM = DATA / "anon_geometry"
LM = DATA / "anon_eye_landmarks"
FOLDERS = ("appleFace", "appleLeftEye", "appleRightEye")


def eye_mask(shape, contour, centre, radius, level, feather=2, dilate=0):
    m = np.zeros(shape[:2], np.uint8)
    if level.startswith("P1"):
        cv2.fillPoly(m, [np.round(contour).astype(np.int32)], 255)
    elif level == "P2":
        cv2.circle(m, tuple(np.round(centre).astype(int)), int(round(radius)), 255, -1)
    else:
        raise ValueError(level)
    if dilate:
        # The released 120x120 eye crop is ~95% NOT eye: the measured palpebral
        # opening is 655-766 px^2, i.e. 4.5-5.3% of the crop. So P1 (opening only)
        # and P0 (whole crop) sit at opposite ends with nothing between them.
        # Dilating the mask parameterises the surround continuously, with P0 as
        # the limit, which is what makes this a graded axis rather than two points.
        k = 2 * int(dilate) + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if feather:
        m = cv2.GaussianBlur(m, (2 * feather + 1,) * 2, feather)
    return (m.astype(np.float32) / 255.0)[..., None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--level", required=True,
                    help="P1 (eye opening), P2 (iris disc), or P1d<N> for the "
                         "opening dilated by N px -- the graded surround axis")
    ap.add_argument("--dst-root", required=True)
    ap.add_argument("--face-op", default="blackbox",
                    help="what happens to the FACE region; the axis is about the "
                         "eyes, so the face is held at its most destructive setting")
    ap.add_argument("--fill", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    g = np.load(GEOM / f"{args.rec}.npz")
    lm = np.load(LM / f"{args.rec}.npz", allow_pickle=True)
    gframes = [f if isinstance(f, str) else f.decode() for f in g["frames"]]
    lframes = [f if isinstance(f, str) else f.decode() for f in lm["frames"]]
    gpos = {n: i for i, n in enumerate(gframes)}

    names = lframes if not args.limit else lframes[:: max(1, len(lframes) // args.limit)][: args.limit]
    odir = DATA / "OriginalData" / args.rec
    dst = Path(args.dst_root) / args.rec
    for sub in FOLDERS:
        (dst / sub).mkdir(parents=True, exist_ok=True)

    n = skipped = 0
    for li, name in enumerate(names):
        gi = gpos.get(name)
        if gi is None:
            skipped += 1
            continue
        frame = cv2.imread(str(odir / name))
        if frame is None:
            skipped += 1
            continue
        fx, fy, fw, fh = [int(round(v)) for v in g["appleFace"][gi][:4]]
        out = frame.copy()
        reg = out[fy:fy + fh, fx:fx + fw]
        if reg.size and args.face_op == "blackbox":
            out[fy:fy + fh, fx:fx + fw] = 0

        li_real = lframes.index(name) if args.limit else li
        for fold in ("appleLeftEye", "appleRightEye"):
            ex, ey, ew, eh = [int(round(v)) for v in g[fold][gi][:4]]
            crop = frame[ey:ey + eh, ex:ex + ew]
            if crop.size == 0:
                continue
            dil = int(args.level[3:]) if args.level.startswith("P1d") else 0
            m = eye_mask(crop.shape,
                         lm[f"{fold}_contour"][li_real],
                         lm[f"{fold}_centre"][li_real],
                         float(lm[f"{fold}_radius"][li_real]),
                         "P1" if dil else args.level, dilate=dil)
            kept = (crop.astype(np.float32) * m
                    + np.full_like(crop, args.fill, np.float32) * (1 - m))
            out[ey:ey + eh, ex:ex + ew] = kept.astype(np.uint8)

        q = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        cv2.imwrite(str(dst / "appleFace" / name), out[fy:fy + fh, fx:fx + fw], q)
        for fold in ("appleLeftEye", "appleRightEye"):
            ex, ey, ew, eh = [int(round(v)) for v in g[fold][gi][:4]]
            cv2.imwrite(str(dst / fold / name), out[ey:ey + eh, ex:ex + ew], q)
        n += 1

    # The evaluation CLI resolves --mean-path RELATIVE TO --data-path, so every
    # manifest that might be used against this root has to exist inside it. Link
    # every mean* manifest dir rather than a hard-coded list: a subset manifest
    # (make_subset_manifest.py) is created AFTER the root, and a missing link is
    # a FileNotFoundError at evaluation time -- it cost the first P-axis utility
    # run (job 1271776, three arms, all failed at dataset construction).
    for src in sorted(DATA.glob("ProcessedData/mean*")):
        if not src.is_dir():
            continue
        link = Path(args.dst_root) / src.name
        if not link.exists():
            try:
                link.symlink_to(src)
            except FileExistsError:
                pass

    print(f"[{args.rec}] level={args.level} wrote {n}, skipped {skipped}")


if __name__ == "__main__":
    main()
