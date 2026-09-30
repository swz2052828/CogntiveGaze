"""Iris and eyelid landmarks in EYE-CROP coordinates, for the periocular axis.

The P-axis (protocol SS4.1) needs to preserve anatomically-defined sub-regions of
the eye crop -- iris disc, eye opening, eyelid contour -- rather than the whole
box. FaceMesh with refine_landmarks gives 478 points including the iris, but in
FRAME coordinates, so they have to be mapped into each released eye crop using
the geometry recovered in Stage 2.

Emits, per frame and per eye: the 16-point eye contour and the 4-point iris ring,
expressed in the eye crop's own pixel coordinates, plus the fitted iris centre
and radius which is what P3 re-renders from.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")
GEOM = DATA / "anon_geometry"

# FaceMesh indices: refine_landmarks adds 468-477 for the two irises
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]


def to_crop(pts, box):
    """Frame coords -> crop coords. box = (x, y, w, h) of the released crop."""
    x, y, w, h = box[:4]
    return np.stack([pts[:, 0] - x, pts[:, 1] - y], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = np.load(GEOM / f"{args.rec}.npz")
    frames = [f if isinstance(f, str) else f.decode() for f in g["frames"]]
    boxes = {"appleLeftEye": g["appleLeftEye"], "appleRightEye": g["appleRightEye"]}
    idx = range(len(frames))
    if args.limit:
        step = max(1, len(frames) // args.limit)
        idx = list(range(0, len(frames), step))[: args.limit]

    import mediapipe as mp
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    odir = DATA / "OriginalData" / args.rec
    rows, miss, t0 = [], 0, time.time()
    for i in idx:
        name = frames[i]
        frame = cv2.imread(str(odir / name))
        if frame is None:
            miss += 1
            continue
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            miss += 1
            continue
        h, w = frame.shape[:2]
        lm = res.multi_face_landmarks[0].landmark
        P = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
        row = {"frame": name}
        # NOTE the deliberate cross-pairing. FaceMesh names eyes from the VIEWER's
        # side of the image, the iTracker crop folders from the SUBJECT's, so
        # FaceMesh LEFT_EYE lands in appleRightEye and vice versa. Measured rather
        # than assumed: pairing them by name put every iris centre outside its crop
        # (x = -81 and +203 in a 120 px box, offset by almost exactly one IOD).
        for fold, eye_i, iris_i in (("appleRightEye", LEFT_EYE, LEFT_IRIS),
                                    ("appleLeftEye", RIGHT_EYE, RIGHT_IRIS)):
            if max(iris_i) >= len(P):
                continue
            box = boxes[fold][i]
            contour = to_crop(P[eye_i], box)
            iris = to_crop(P[iris_i], box)
            centre = iris[0]
            radius = float(np.linalg.norm(iris[1:] - centre, axis=1).mean())
            row[fold] = dict(contour=contour, iris=iris, centre=centre, radius=radius)
        if len(row) == 3:
            rows.append(row)
    mesh.close()

    dt = time.time() - t0
    print(f"[{args.rec}] {len(rows)}/{len(list(idx))} frames, {miss} missed, "
          f"{dt:.1f}s ({dt/max(1,len(rows))*1000:.0f} ms/frame)")
    for fold in ("appleLeftEye", "appleRightEye"):
        r = np.array([x[fold]["radius"] for x in rows if fold in x])
        c = np.array([x[fold]["centre"] for x in rows if fold in x])
        if len(r):
            print(f"  {fold:14s} iris radius {r.mean():5.2f}+-{r.std():4.2f} px | "
                  f"centre ({c[:,0].mean():5.1f},{c[:,1].mean():5.1f}) "
                  f"sd ({c[:,0].std():4.2f},{c[:,1].std():4.2f}) | "
                  f"in-crop {(  (c[:,0]>0)&(c[:,0]<120)&(c[:,1]>0)&(c[:,1]<120)).mean()*100:.0f}%")
    if args.out and rows:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out, frames=np.array([r["frame"] for r in rows]),
            **{f"{f}_{k}": np.array([r[f][k] for r in rows])
               for f in ("appleLeftEye", "appleRightEye")
               for k in ("contour", "iris", "centre", "radius")})
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
