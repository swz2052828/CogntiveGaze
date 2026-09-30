"""Find how a stored OriginalCalib/OriginalData frame maps to a video frame number.

The stored frames are 1080x750 crops of 1920x1080 video, and decoding the video at
the stored filename's index does NOT produce them (corr 0.14-0.17). So either the
video frame numbering is offset from the stored numbering, or the stored frames do
not come from these video files at all. This scans the video and reports which
decoded frame best matches a given stored frame.

Works at 1/4 resolution for speed; a hit is confirmed at full resolution.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/springbrook/share/eng/esrpxk/datasets")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", required=True)
    ap.add_argument("--src", default="OriginalCalib")
    ap.add_argument("--max-frames", type=int, default=9000)
    ap.add_argument("--n-probe", type=int, default=3)
    args = ap.parse_args()

    sub = int(args.rec)
    vid = DATA / "videos" / f"subid_{sub}.mp4"
    stored = sorted((DATA / args.src / args.rec).glob("*.jpg"))
    probes = [stored[i] for i in np.linspace(0, len(stored) - 1, args.n_probe).astype(int)]
    tpl = {}
    for p in probes:
        im = cv2.imread(str(p))
        tpl[int(os.path.splitext(p.name)[0])] = cv2.resize(
            im, (im.shape[1] // 4, im.shape[0] // 4), interpolation=cv2.INTER_AREA)
    print(f"[{args.rec}] {args.src}: {len(stored)} frames, probing {sorted(tpl)}")

    cap = cv2.VideoCapture(str(vid))
    best = {k: (-1.0, None, None) for k in tpl}
    i = 0
    while i < args.max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        small = cv2.resize(fr, (fr.shape[1] // 4, fr.shape[0] // 4), interpolation=cv2.INTER_AREA)
        for k, t in tpl.items():
            if t.shape[0] > small.shape[0] or t.shape[1] > small.shape[1]:
                continue
            r = cv2.matchTemplate(small, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[k][0]:
                best[k] = (float(mx), i, (loc[0] * 4, loc[1] * 4))
        i += 1
    cap.release()
    print(f"[{args.rec}] scanned {i} video frames")
    for k in sorted(best):
        c, vi, off = best[k]
        delta = None if vi is None else vi - k
        print(f"  stored {k:6d} -> best video frame {vi}  corr {c:.4f}  "
              f"crop offset {off}  delta {delta}")
    deltas = [best[k][1] - k for k in best if best[k][1] is not None]
    if deltas and len(set(deltas)) == 1 and min(best[k][0] for k in best) > 0.9:
        print(f"  VERDICT: constant offset {deltas[0]:+d}, crop offset "
              f"{best[sorted(best)[0]][2]}")
    else:
        print(f"  VERDICT: no constant offset (deltas {deltas}, "
              f"min corr {min(best[k][0] for k in best):.3f})")


if __name__ == "__main__":
    main()
