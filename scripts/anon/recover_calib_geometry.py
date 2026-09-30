"""Recover crop geometry for the CALIBRATION support sets.

Differs from the task-data recovery in one way that matters: calibration crops
are stored resized to 224x224 while occupying ~300-400 px (face) and ~110-130 px
(eyes) in the frame, so the scale must be searched rather than assumed. Reuses
match_scaled from recover_geometry_tm.py.

The metadata.mat faceBbox/eyeBbox fields are NOT usable for this -- cropping with
them and resizing reproduces the stored crops only to MAE 28-49 (face) and 5-21
(eyes), so they came from a different detector pass than the released images.

  python scripts/anon/recover_calib_geometry.py --rec 00006 --k 72
"""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recover_geometry_tm import match_scaled                      # noqa: E402

DATA = Path("/springbrook/share/eng/esrpxk/datasets")
FOLDERS = ("appleFace", "appleLeftEye", "appleRightEye")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--k", type=int, default=72)
    ap.add_argument("--frame-root", default=None,
                    help="default OriginalCalib (K=72 only); use OriginalCalibFull "
                         "once the other K levels have been extracted")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sup = DATA / f"calib_support_K{args.k}" / args.rec
    frames_dir = Path(args.frame_root) if args.frame_root else DATA / "OriginalCalib" / args.rec
    names = sorted(p.name for p in (sup / "appleFace").glob("*.jpg"))

    coarse = np.arange(0.25, 2.05, 0.05)
    fine = np.arange(0.25, 2.005, 0.01)
    prev = {f: None for f in FOLDERS}
    rows, corrs, missing = [], {f: [] for f in FOLDERS}, 0
    t0 = time.time()
    for name in names:
        fp = frames_dir / name
        frame = cv2.imread(str(fp))
        if frame is None:
            missing += 1
            continue
        row = {"frame": name}
        for f in FOLDERS:
            t = cv2.imread(str(sup / f / name))
            if t is None:
                continue
            scales = fine if prev[f] is not None else coarse
            x, y, w, h, sc, c = match_scaled(frame, t, scales, prev_scale=prev[f])
            if x is None:
                continue
            if c >= 0.97:
                prev[f] = sc
            corrs[f].append(c)
            row[f] = (x, y, w, h, round(sc, 4), round(c, 5))
        if len(row) == 4:
            rows.append(row)

    dt = time.time() - t0
    print(f"[{args.rec} K={args.k}] {len(rows)}/{len(names)} frames, "
          f"{missing} source frames missing, {dt:.1f}s ({dt/max(1,len(rows))*1000:.0f} ms/frame)")
    for f in FOLDERS:
        c = np.asarray(corrs[f])
        if len(c):
            sc = np.asarray([r[f][4] for r in rows if f in r])
            print(f"  {f:14s} corr min {c.min():.4f} mean {c.mean():.4f} | "
                  f"<0.97: {(c < 0.97).sum()} | native px {np.mean([r[f][2] for r in rows if f in r]):.0f} "
                  f"| scale sd {sc.std():.4f}")

    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, frames=np.array([r["frame"] for r in rows]),
                            **{f: np.array([r[f] for r in rows], dtype=np.float64)
                               for f in FOLDERS})
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
