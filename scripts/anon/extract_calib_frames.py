"""Extract the calibration support frames that OriginalCalib does not hold.

OriginalCalib contains exactly the K=72 selection, so K=9/18/36 -- including the
deployable 9-point enrolment -- have no raw frames on disk. They live in the
calibration segment of the source video.

Seeking is not an option: per the project's own finding, cv2 CAP_PROP_POS_FRAMES
lands on the wrong frame for these H.264 files by a per-video constant. So this
decodes sequentially from frame 0 and keeps only the wanted indices, stopping as
soon as the last one is passed.

--verify compares the frames that DO exist in OriginalCalib against the decoded
ones, which establishes that the stored filename equals the video frame number
before any new frames are trusted.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")


def wanted_indices(rec, ks=(9, 18, 36, 72)):
    idx = set()
    for k in ks:
        d = DATA / f"calib_support_K{k}" / rec / "appleFace"
        if d.is_dir():
            idx |= {int(os.path.splitext(p.name)[0]) for p in d.glob("*.jpg")}
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out-root", default=str(DATA / "OriginalCalibFull"))
    ap.add_argument("--verify", action="store_true",
                    help="only decode frames already present in OriginalCalib and "
                         "report agreement; writes nothing")
    args = ap.parse_args()

    sub = int(args.rec)
    vid = DATA / "videos" / f"subid_{sub}.mp4"
    want = wanted_indices(args.rec)
    have = {int(os.path.splitext(p.name)[0])
            for p in (DATA / "OriginalCalib" / args.rec).glob("*.jpg")}
    target = have if args.verify else want
    if not target:
        print(f"[{args.rec}] nothing to do"); return
    last = max(target)
    print(f"[{args.rec}] video={vid.name} want={len(want)} have={len(have)} "
          f"missing={len(want - have)} decode_to={last}", flush=True)

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {vid}")
    out = Path(args.out_root) / args.rec
    if not args.verify:
        out.mkdir(parents=True, exist_ok=True)

    maes, offsets, n, i = [], [], 0, 0
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if i in target:
            if args.verify:
                ref = cv2.imread(str(DATA / "OriginalCalib" / args.rec / f"{i:05d}.jpg"))
                if ref is not None:
                    # The stored frames are a CROP of the video frame (1080x750 out
                    # of 1920x1080), so locate the crop rather than comparing whole
                    # frames. The offset should be constant within a recording.
                    r = cv2.matchTemplate(frame, ref, cv2.TM_CCOEFF_NORMED)
                    _, mx, _, loc = cv2.minMaxLoc(r)
                    patch = frame[loc[1]:loc[1] + ref.shape[0],
                                  loc[0]:loc[0] + ref.shape[1]]
                    if patch.shape == ref.shape:
                        maes.append(float(np.abs(patch.astype(np.float32)
                                                 - ref.astype(np.float32)).mean()))
                        offsets.append((loc[0], loc[1], float(mx)))
            else:
                cv2.imwrite(str(out / f"{i:05d}.jpg"), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 98])
            n += 1
        i += 1
    cap.release()

    if args.verify:
        if not maes:
            print(f"[{args.rec}] NO COMPARABLE FRAMES -- convention unverified"); return
        m = np.asarray(maes); off = np.asarray(offsets)
        print(f"[{args.rec}] compared {len(m)} frames | MAE mean {m.mean():.2f} "
              f"median {np.median(m):.2f} max {m.max():.2f}")
        print(f"  crop offset x {off[:,0].min():.0f}..{off[:,0].max():.0f} "
              f"y {off[:,1].min():.0f}..{off[:,1].max():.0f} | corr min {off[:,2].min():.4f}")
        const = off[:, 0].std() < 0.5 and off[:, 1].std() < 0.5
        print("  VERDICT:", "filename == video frame number, crop offset CONSTANT"
              if m.mean() < 8 and const else
              ("filename == video frame number, but offset VARIES" if m.mean() < 8
               else "MISMATCH -- do not extract with this convention"))
    else:
        print(f"[{args.rec}] wrote {n} frames to {out}")


if __name__ == "__main__":
    main()
